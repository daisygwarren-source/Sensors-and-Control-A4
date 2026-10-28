"""Tier 3 (Parts B, D, E on hardware): vision-guided pick on the real UR3e. main.py's run_trial with the sim parts
swapped out: UR3e kinematics, velocity-input joint KF + LQR, stop-and-look camera, RG2 gripper. The object is lifted
and put back where it was.

    python real_pick.py --sim                                                   # whole state machine, dry run
    python real_pick.py --sim --hover 0
    python real_pick.py --ip ROBOT_IP --config ...\\config.json                 # dry run, stops 5 cm short
    python real_pick.py --ip ROBOT_IP --config ...\\config.json --hover 0 --trials 5
    python real_pick.py --ip ROBOT_IP --config ...\\config.json --track         # dry run, tracking while moving

--track (fixed camera only): the camera also runs while the arm approaches and the target follows the object; the
stops stay as the checks before descending. Without it the camera only looks with the arm stopped.
"""
import argparse
import contextlib
import json
import queue
import threading
import time
import traceback
from collections import defaultdict
from datetime import datetime

import numpy as np
import cv2
from realsense_lab import RealSenseCamera

from real import (DT, MAX_SPEED, Abort, Gripper, JointLoop, UR3eKinematics, add_args, check_args, connect,
                  load_config, move_time, pose_error, quintic, sim_start, ur_pose_to_T)
from main import CAM_DT, NIS_GATE, wrist_view
from src.controller import PBVS, PHASES, JointVelLQR, grasp_R
from src.estimation import JointVelKF, KalmanFilter3D
from src.model import rot_error
from src.perception import locate_cube, project

NEEDED = ["tau", "sigma_q", "K", "mount", "T_tool0_tcp", "look_q", "table_z", "half", "object_width", "hsv",
          "workspace", "grip_time"]
W, H = 640, 480
# (UR3e, --sim) defaults. the UR3e ran the first on 28 Oct; at them the sim's ~0.2 s joint lag carries the TCP ~1 cm
# past the grasp height while the arm stops to grip, into the table guard, so --sim keeps the earlier slow ones
SPEEDS = dict(accel=(0.5, 0.1), v_max=(0.1, 0.02), w_max=(0.5, 0.1), acc=(0.2, 0.05), alpha=(1.0, 0.2))


class Camera:
    # stop-and-look: capture() blocks until the next frame (~33 ms), so this only runs with the arm stopped
    def __init__(self, rs, cfg, kin, n_drop=2):
        self.rs, self.cfg, self.kin, self.n_drop = rs, cfg, kin, n_drop
        self.K = np.array(cfg["K"], float)

    def T_bc(self, q):
        if self.cfg["mount"] == "wrist":
            return self.kin.T_base_cam(q, self.cfg["T_tool0_cam"])
        return np.array(self.cfg["T_base_cam"], float)

    def locate(self, f, T_bc):
        c = self.cfg
        r = locate_cube(f.rgb, f.depth_m, self.K, T_bc, c["table_z"], c["half"], with_yaw=True, hsv=c["hsv"],
                        top_pct=c.get("top_pct", 90))
        return None if r is None else (r[0] - np.array(c.get("loc_offset", [0, 0, 0])), r[1])

    def look(self, q, n):
        T_bc = self.T_bc(q)
        # frames queued while the arm was still moving
        for _ in range(self.n_drop):
            self.rs.capture()
        z = []
        for _ in range(n):
            f = self.rs.capture()
            z.append(self.locate(f, T_bc))
        return z, f.rgb, T_bc

    def close(self):
        pass


class CameraThread(Camera):
    # --track, fixed camera only, so T_base_cam is constant: a frame needs no joint angles and the arm moving doesn't
    # blur it. a daemon thread owns the RealSense (only it calls capture(): two callers on one pipeline clash), runs
    # locate_cube every `every` s while `active` is set and queues (t, p, yaw, rgb). the main loop drains the queue,
    # and the stops take their frames from it too
    AGE = 0.1  # capture() can return a frame up to ~3 frames old, so t is the return time less this

    def __init__(self, rs, cfg, kin):
        super().__init__(rs, cfg, kin)
        self.T, self.every, self.error, self.looking = self.T_bc(None), CAM_DT, None, False
        self.q, self.active, self.done = queue.Queue(maxsize=50), threading.Event(), threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        try:
            last = -np.inf
            while not self.done.is_set():
                if not self.active.wait(0.05):
                    continue
                # a stop takes its frames back to back, the arm isn't moving then
                if not self.looking:
                    time.sleep(max(0.0, last + self.every - time.perf_counter()))
                f = self.rs.capture()
                last = time.perf_counter()
                z = self.locate(f, self.T)
                item = (last - self.AGE, *(z or (None, None)), f.rgb)
                while True:
                    try:
                        self.q.put_nowait(item)
                        break
                    except queue.Full:
                        with contextlib.suppress(queue.Empty):
                            self.q.get_nowait()
        except BaseException as e:
            self.error = e

    def check(self):
        if self.error is not None:
            raise Abort(f"camera thread stopped: {self.error!r}")

    def frames(self):
        self.check()
        out = []
        while True:
            try:
                out.append(self.q.get_nowait())
            except queue.Empty:
                return out

    def look(self, q, n):
        # n frames exposed after this call, i.e. with the arm stopped. the camera idles afterwards until the main
        # loop wants it (a failed look ends the trial with no moving cycle to switch it off)
        t0, z, rgb = time.perf_counter(), [], None
        self.looking = True
        self.active.set()
        try:
            while len(z) < n:
                try:
                    t, p, a, rgb_i = self.q.get(timeout=1.0)
                except queue.Empty:
                    self.check()
                    if time.perf_counter() - t0 > 5 + n * self.every:
                        raise Abort(f"camera thread: no frames for {time.perf_counter() - t0:.0f} s")
                    continue
                if t > t0:
                    z.append(None if p is None else (p, a))
                    rgb = rgb_i
        finally:
            self.looking = False
            self.active.clear()
        return z, rgb, self.T

    def close(self):
        # longer than capture()'s 5 s frame timeout, so the RealSense isn't stopped under a waiting capture
        self.done.set()
        self.thread.join(6.0)


class SimCamera(Camera):
    # --sim stub: perception returns the --sim-object plus ~3 mm noise, no image. move: the object shifts by this
    # much 1 s after the first look (about 1 s into the approach) and stays there (the disturbed trial)
    def __init__(self, cfg, kin, obj, rng, move=(0, 0)):
        super().__init__(None, cfg, kin)
        self.obj, self.rng, self.move = np.asarray(obj, float), rng, np.r_[move, 0.0]
        self.t_move, self.last, self.active, self.every = None, -np.inf, threading.Event(), CAM_DT

    def measure(self):
        moved = self.t_move is not None and time.perf_counter() >= self.t_move
        return self.obj + (self.move if moved else 0) + self.rng.normal(0, 0.003, 3), 0.0

    def look(self, q, n):
        z = [self.measure() for _ in range(n)]
        if self.t_move is None:
            self.t_move = time.perf_counter() + 1.0
        return z, None, self.T_bc(q)

    def frames(self):
        # --track: one measurement every `every` s while active, as the camera thread gives
        now = time.perf_counter()
        if not self.active.is_set() or now - self.last < self.every:
            return []
        self.last = now
        return [(now, *self.measure(), None)]


class RealPickTask:
    # PickTask's phases, with the sim's timers replaced by events the main loop runs with the arm stopped:
    # look, refine (look again before descending), grasp (grip), release (open)
    STOPS = ("look", "refine", "grasp", "release")

    def __init__(self, p_home, R_home, table_z, half, pre, lift=0.10):
        self.p_home, self.R_home, self.table_z, self.half, self.pre, self.lift = p_home, R_home, table_z, half, pre, lift
        self.phase, self.t0, self.p_grasp, self.failed = "look", 0.0, None, None

    def go(self, phase, t):
        self.phase, self.t0 = phase, t

    def __call__(self, t, at, cube, R, slow):
        cube = np.r_[cube[:2], self.table_z + self.half - 0.006]
        # pbvs error is a step behind, give it time to see the new target
        at = at and t - self.t0 > 0.1
        ph = self.phase
        if ph == "approach":
            if at:
                self.go("refine", t)
            return cube + [0, 0, self.pre], R, None
        if ph == "descend":
            if at:
                self.p_grasp = cube.copy()
                self.go("grasp", t)
            return cube, R, slow
        # put it back where it was: place = the grasp point, just above the table
        place = np.r_[self.p_grasp[:2], self.table_z + self.half + 0.005] if self.p_grasp is not None else cube
        if ph == "lift":
            if at:
                self.go("transfer", t)
            return self.p_grasp + [0, 0, self.lift], R, None
        if ph == "transfer":
            if at:
                self.go("lower", t)
            return place + [0, 0, self.lift], R, None
        if ph == "lower":
            if at:
                self.go("release", t)
            return place, R, slow
        if ph == "retreat":
            if at:
                self.go("home", t)
            return place + [0, 0, 0.10], R, None
        if at and ph == "home":
            self.go("done", t)
        return self.p_home, self.R_home, None


def grasp_rotation(yaw, R_home):
    # yaw is only known mod 90 deg: take the face the fingers reach with the least wrist turn from the look pose
    return min((grasp_R(yaw + k * np.pi / 2) for k in range(4)), key=lambda R: np.linalg.norm(rot_error(R, R_home)))


def refine_height(cfg, R, pre, min_depth):
    # wrist camera at the refine pose (TCP `pre` above the object centre): raise pre until the object is at least
    # min_depth from the camera (the D435 can't measure closer) and fully in the image
    if cfg["mount"] != "wrist":
        return pre
    T_tcp_cam = np.linalg.inv(np.array(cfg["T_tool0_tcp"], float)) @ np.array(cfg["T_tool0_cam"], float)
    K, h = np.array(cfg["K"], float), cfg["half"]
    # margin: the refine stop fires anywhere within `at`'s 5 mm
    min_depth = min_depth + 0.02
    box = np.array([[x, y, z] for x in (-h, h) for y in (-h, h) for z in (-h, h)])
    for p in np.arange(pre, 0.40, 0.005):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, [0, 0, p]
        Tc = np.linalg.inv(T @ T_tcp_cam)
        P = box @ Tc[:3, :3].T + Tc[:3, 3]
        uv = project(K, P.T).T
        if Tc[2, 3] >= min_depth and (P[:, 2] > 0).all() and (uv > 20).all() and (uv[:, 0] < W - 20).all() \
                and (uv[:, 1] < H - 20).all():
            return p
    raise Abort(f"no refine height up to 0.40 m keeps the object {min_depth} m from the camera and in view")


def in_box(p, ws, margin=0.0):
    return ws[0] - margin <= p[0] <= ws[1] + margin and ws[2] - margin <= p[1] <= ws[3] + margin


def loop_rate(t, sel=None):
    # mean rate over consecutive moving cycles (the stops leave gaps), only where sel is True if given
    dt = np.diff(np.asarray(t, float))
    ok = dt < 0.1
    if sel is not None:
        ok &= sel[1:] & sel[:-1]
    return 1 / dt[ok].mean() if ok.any() else np.nan


def run_trial(robot, kin, cam, gripper, cfg, args, log, out, n):
    look_q, ws = np.array(cfg["look_q"]), cfg["workspace"]
    table_z, half = cfg["table_z"], cfg["half"]
    obj_mm = 1e3 * cfg["object_width"]
    close_mm = max(0.0, obj_mm - args.squeeze)
    tau = np.broadcast_to(np.asarray(cfg["tau"], float), 6)
    q0 = robot.get_joint_positions()
    kf = JointVelKF(q0, DT, tau, cfg["sigma_q"], cfg["sigma_qd"] if args.use_qd else None,
                    delay=np.round(np.asarray(cfg.get("td", 0.0)) / DT).astype(int), sigma_t=cfg.get("sigma_t", 0.0))
    lqr = JointVelLQR(DT, tau, args.vmax, args.accel)
    loop = JointLoop(robot, kf, lqr, args.use_qd, True, max_err=0.1)
    pbvs = PBVS(kin, DT, q0, v_max=args.v_max, w_max=args.w_max, acc=args.acc, alpha=args.alpha,
                qd_max=0.6 * args.vmax, qdd_max=0.6 * args.accel)
    p_home, R_home, _ = kin.tcp(look_q)
    R = grasp_rotation(0.0, R_home)
    pre = refine_height(cfg, R, args.pre, args.min_depth)
    task = RealPickTask(p_home, R_home, table_z, half, pre)
    cube_kf = KalmanFilter3D(dt=1.0, constant_velocity=False, q_pos=1e-6)
    cube_kf.R = np.diag(np.square(cfg.get("cube_sigma", [0.003] * 3)))
    P0 = cube_kf.P.copy()
    st = dict(yaws=[], missed=[], n_meas=0, n_rej=0, n_skip=0, looks=0, reapproach=0, resets=0)
    res = dict(trial=n, pre=pre)
    e, q_ref, cube, cube_prev, view, fresh = np.ones(6), q0.copy(), None, None, None, False
    has_window = False
    max_reach = cfg.get("max_reach", 0.43)
    print(f"\ntrial {n}: refine height {pre:.3f} m, grip to {close_mm:.0f} mm at {args.force:.0f} N"
          f"{f', DRY RUN {args.hover:.2f} m above every target' if args.hover else ''}"
          f"{', tracking the object while approaching' if args.track else ''}")

    def measure(t, p, a, moving):
        # one locate_cube result into the cube KF: NIS gate, three agreeing misses in a row restart it, yaw list.
        # while moving, an estimate off the table, out of the work area or out of reach is skipped, not a failure
        st["n_meas"] += 1
        ok, nis, event = True, np.nan, "used"
        if moving and (not in_box(p, ws) or abs(p[2] - (table_z + half)) > 0.015 or np.hypot(*p[:2]) > max_reach):
            st["n_skip"] += 1
            ok, event = False, "skipped"
        elif not cube_kf.initialized:
            cube_kf.initialize(p)
        else:
            nis = cube_kf.nis(p)
            ok = nis < NIS_GATE
            if ok:
                cube_kf.update(p)
                st["missed"] = []
            else:
                st["n_rej"] += 1
                st["missed"] = st["missed"][-2:] + [p]
                event = "rejected"
                m = np.array(st["missed"])
                if len(m) == 3 and np.linalg.norm(m - m.mean(0), axis=1).max() < 0.01:
                    # three misses in a row that agree: the object really moved, restart the filter on it. misses
                    # that scatter (a part-hidden object, the gripper in the way) don't
                    cube_kf.initialize(p)
                    cube_kf.P = P0.copy()
                    st["missed"], st["yaws"] = [], []
                    ok, event = True, "reinit"
        if ok:
            st["yaws"].append(a)
        log["meas_t"].append(t)
        log["meas"].append(p)
        log["meas_ok"].append(ok)
        log["meas_nis"].append(nis)
        return event

    def estimate():
        nonlocal cube, R
        if cube_kf.initialized:
            cube = cube_kf.position.copy()
            yaw = np.angle(np.mean(np.exp(4j * np.array(st["yaws"])))) / 4 if st["yaws"] else 0.0
            R = grasp_rotation(yaw, R_home)

    def look(t):
        nonlocal has_window
        if cube_kf.initialized:
            cube_kf.predict()
        z, rgb, T_bc = cam.look(kf.X[0], args.n_look)
        found = 0
        for zi in z:
            if zi is not None:
                found += 1
                measure(t, *zi, moving=False)
        st["looks"] += 1
        estimate()
        if rgb is not None:
            cv2.imwrite(str(out / f"trial{n}_look{st['looks']}_{task.phase}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            wrist_view(rgb, cam.K, T_bc, cube, f"trial {n} {task.phase}: {found}/{args.n_look} found", cfg["hsv"])
            has_window = True
        depth = (np.linalg.inv(T_bc) @ np.r_[cube, 1])[2] if cube is not None else np.nan
        print(f"  look {st['looks']} ({task.phase}): found {found}/{args.n_look}, estimate "
              f"{np.round(cube, 4) if cube is not None else None}, camera depth {depth:.3f} m")
        log["look_t"].append(t)
        log["look_found"].append(found)
        if depth < args.min_depth:
            print(f"  WARNING: object only {depth:.3f} m from the camera, closer than {args.min_depth} m")
        if found < args.n_look // 2 or cube is None:
            return f"object found in only {found} of {args.n_look} frames"
        if not in_box(cube, ws) or abs(cube[2] - (table_z + half)) > 0.015:
            return f"estimate {np.round(cube, 3)} outside the work area or not on the table"
        # past the reach PBVS only creeps until the phase time limit (49 s for the far corner on 28 Oct)
        if np.hypot(*cube[:2]) > max_reach:
            return (f"estimate {np.round(cube, 3)} is {np.hypot(*cube[:2]):.3f} m from the base, past the arm's "
                    f"reach ({max_reach} m)")
        return None

    t0 = time.perf_counter()
    k, timed, deadline, budget = 0, None, 0.0, 0.0
    while task.phase not in ("done", "fail"):
        t = time.perf_counter() - t0
        phase = task.phase
        if phase in RealPickTask.STOPS:
            fail = None
            if not loop.halt():
                raise Abort("the arm did not stop within 5 s")
            if phase == "look":
                fail = look(t)
                task.go("fail" if fail else "approach", t)
            elif phase == "refine":
                cube_prev = cube.copy()
                fail = look(t)
                moved = np.linalg.norm((cube - cube_prev)[:2]) if not fail else 0.0
                if fail:
                    task.go("retreat", t)
                elif moved > 0.01 and st["reapproach"] < 3:
                    print(f"  object moved {1e3 * moved:.0f} mm since the last look, approaching again")
                    st["reapproach"] += 1
                    task.go("approach", t)
                else:
                    task.go("descend", t)
            elif phase == "grasp":
                w = gripper.grip(close_mm, args.force)
                ok = w > close_mm + 5
                res["grip_width"] = w
                log["grip"].append((t, w))
                print(f"  gripper closed at {w:.1f} mm (object {obj_mm:.0f} mm, target {close_mm:.0f} mm): "
                      f"{'holding' if ok else 'missed'}")
                if ok or args.hover:
                    task.go("lift", t)
                else:
                    fail = f"grasp missed, width {w:.1f} mm"
                    gripper.open(80.0, args.force)
                    task.go("retreat", t)
            else:
                log["grip"].append((t, gripper.open(80.0, args.force)))
                task.go("retreat", t)
            if task.phase in ("fail", "retreat") and fail and not res.get("failed"):
                res["failed"] = fail
                print(f"  FAILED: {fail}")
            # time passed while stopped: restart the integrator and PBVS from the estimate, and the phase clock
            lqr.reset()
            pbvs.reset(kf.X[0])
            e = np.ones(6)
            task.t0 = time.perf_counter() - t0
            timed = None
            print(f"{task.t0:6.2f} s  {task.phase}")
            continue

        loop.read()
        # the object while the arm approaches. from descend on the fingers start to cover its top face and pull the
        # estimate to the visible side, so it is frozen there and the refine stop is the last check
        tracking = args.track and phase == "approach"
        if args.track:
            (cam.active.set if tracking else cam.active.clear)()
            batch = cam.frames()
            if batch and batch[-1][3] is not None:
                view, fresh = batch[-1][3], True
            for tm, p, a, _ in batch if tracking else []:
                if cube_kf.initialized:
                    cube_kf.predict()
                if p is not None and measure(tm - t0, p, a, moving=True) == "reinit" and st["resets"] < 3:
                    # a new target: its phase time limit starts again
                    st["resets"] += 1
                    timed = None
                    print(f"{t:6.2f} s  object moved, following it (filter restarted)")
            if tracking and batch:
                estimate()
        q, qd, d = kf.X
        p_now, R_now, _ = kin.tcp(q)
        at = np.linalg.norm(e[:3]) < 0.005 and np.linalg.norm(e[3:]) < 0.05 and np.abs(q - q_ref).max() < 0.01
        p_des, R_des, v_max = task(t, at, cube, R, 0.5 * args.v_max)
        if task.phase != phase:
            print(f"{t:6.2f} s  {task.phase}")
        p_t = p_des + [0, 0, args.hover if phase != "home" else 0.0]
        if phase != timed:
            # each moving phase gets time for its distance at well under the slowest tool speed seen (~5 mm/s
            # straight up at the joint cap). running out means the target is out of reach or near a singularity,
            # where PBVS only creeps and `at` never comes
            timed = phase
            budget = (15 + np.linalg.norm(p_t - p_now) / min(0.003, 0.5 * args.v_max)
                      + np.linalg.norm(rot_error(R_des, R_now)) / min(0.02, 0.5 * args.w_max))
            deadline = t + budget
        elif t > deadline:
            fail = f"{phase} not finished in {budget:.0f} s, target {np.round(p_t, 3)} out of reach or near a singularity"
            res.setdefault("failed", fail)
            print(f"  FAILED: {fail}. The arm stops where it is"
                  + (", still holding the object" if phase in ("lift", "transfer", "lower") else ""))
            task.go("fail", t)
            break
        # safety, every cycle, before anything is sent. targets stay above the object's centre height except the
        # descent onto it; the TCP itself never goes within 5 mm of the table
        z_min = table_z + 0.005 if phase == "descend" else table_z + half - 0.01
        if not in_box(p_t, ws) or not in_box(p_now, ws, 0.02):
            raise Abort(f"outside the work area: target {np.round(p_t, 3)}, tcp {np.round(p_now, 3)}")
        if p_t[2] < z_min or p_now[2] < table_z + 0.005:
            raise Abort(f"too low: target {np.round(p_t, 3)} (min {z_min:.3f}), tcp {np.round(p_now, 3)}")
        q_ref, qd_ref, qdd_ref, e = pbvs(p_t, R_des, v_max)
        u, clipped = loop.command(q_ref, qd_ref, qdd_ref)

        tq, tR, _ = kin.tcp(loop.q)
        tr = robot.get_tcp_pose() if hasattr(robot, "get_tcp_pose") else np.full(6, np.nan)
        for key, v in (("t", t), ("phase", PHASES.index(phase)), ("y", loop.q), ("qd_meas", loop.qd),
                       ("q_hat", q.copy()), ("qd_hat", qd.copy()), ("d_hat", d.copy()), ("sig", kf.sigma),
                       ("P", kf.P.copy()), ("q_ref", q_ref), ("u", u), ("clipped", clipped), ("p_target", p_t),
                       ("tcp_kin", np.r_[tq, rot_error(tR, np.eye(3))]), ("tcp_robot", tr),
                       ("cube", cube if cube is not None else np.full(3, np.nan)),
                       ("cube_sig", np.sqrt(np.diag(cube_kf.P)[:3]) if cube is not None else np.full(3, np.nan))):
            log[key].append(v)
        k += 1
        if fresh and k % 20 == 0:
            # OpenCV windows aren't thread-safe on Windows: only this thread draws, and only a new frame
            wrist_view(view, cam.K, cam.T_bc(q), cube, f"trial {n} {phase}{' (tracking)' if tracking else ''}",
                       cfg["hsv"], wait=False)
            has_window, fresh = True, False
        elif has_window and k % 10 == 0:
            cv2.pollKey()
        if isinstance(cam, CameraThread) and tracking and k % 100 == 0 and cam.every == CAM_DT:
            # the camera thread competes with this loop for the CPU: if the loop slows, process every other frame
            rate = loop_rate(log["t"][-100:], np.array(log["phase"][-100:]) == PHASES.index("approach"))
            if rate < args.min_rate:
                cam.every = 2 * CAM_DT
                print(f"  loop at {rate:.0f} Hz while tracking (< {args.min_rate:g}): camera every other frame, "
                      f"{1 / cam.every:.0f} Hz")

    # ur_rtde keeps running the last speedJ until told otherwise, and PBVS is still easing in when `at` fires
    if not loop.halt():
        raise Abort("the arm did not stop within 5 s")
    res.update(phase="done" if task.phase == "done" and not res.get("failed") else "fail", t=t,
               n_meas=st["n_meas"], n_rej=st["n_rej"], n_skip=st["n_skip"], looks=st["looks"],
               reapproach=st["reapproach"], resets=st["resets"],
               clipped=float(np.mean(np.any(log["clipped"], axis=1))) if log["clipped"] else 0.0,
               loop_hz=loop_rate(log["t"]))
    if args.track:
        res["track_hz"] = loop_rate(log["t"], np.array(log["phase"]) == PHASES.index("approach"))
    if log["t"]:
        print(f"  loop rate {res['loop_hz']:.1f} Hz over the moving phases"
              + (f", {res['track_hz']:.1f} Hz while tracking (camera at {1 / cam.every:.0f} Hz)" if args.track else ""))
    return res


def save(log, res, path):
    arrays = {k: np.array(v) for k, v in log.items() if k not in ("grip",)}
    with open(path, "xb") as f:
        np.savez(f, **arrays, grip=np.array(log.get("grip", []), float).reshape(-1, 2), result=json.dumps(res))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    ap.add_argument("--hover", type=float, default=0.05,
                    help="m added to every target: 0.05 dry run (0.10 is past the UR3e's reach), 0 real pick")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--vmax", type=float, default=MAX_SPEED, help=f"joint speed limit, rad/s, at most {MAX_SPEED}")
    ap.add_argument("--accel", type=float, help="speedJ acceleration and the LQR's rate limit, rad/s^2 (0.5, --sim 0.1)")
    ap.add_argument("--v-max", type=float, help="PBVS tool speed, m/s (0.1, --sim 0.02)")
    ap.add_argument("--w-max", type=float, help="PBVS tool rotation speed, rad/s (0.5, --sim 0.1)")
    ap.add_argument("--acc", type=float, help="PBVS tool acceleration, m/s^2 (0.2, --sim 0.05)")
    ap.add_argument("--alpha", type=float, help="PBVS tool angular acceleration, rad/s^2 (1.0, --sim 0.2)")
    ap.add_argument("--pre", type=float, default=0.10, help="refine height above the object, m (raised if needed)")
    ap.add_argument("--min-depth", type=float, default=0.30, help="closest the D435 should look, m")
    ap.add_argument("--n-look", type=int, default=8, help="frames per look")
    ap.add_argument("--squeeze", type=float, default=10.0, help="grip target this many mm under the object width")
    ap.add_argument("--force", type=float, default=20.0, help="grip force, N")
    ap.add_argument("--use-qd", action="store_true", help="also feed the measured joint velocity to the KF")
    ap.add_argument("--sim-object", type=float, nargs=3, metavar=("X", "Y", "Z"),
                    help="--sim: object position (default: work area centre, on the table)")
    ap.add_argument("--sim-move", type=float, nargs=2, default=(0.0, 0.0), metavar=("DX", "DY"),
                    help="--sim: the object moves this far (m) 1 s after the first look, the disturbed trial")
    ap.add_argument("--track", action="store_true",
                    help="fixed camera only: follow the object while approaching, not only at the stops")
    ap.add_argument("--min-rate", type=float, default=70.0,
                    help="--track: below this loop rate (Hz) the camera drops to every other frame. stop-and-look "
                         "alone ran at 72-90 Hz on the lab laptop (28 Oct logs)")
    ap.add_argument("--viewer", action="store_true", help="MuJoCo viewer with --sim")
    args = ap.parse_args()
    for key, (ur3e, sim) in SPEEDS.items():
        if getattr(args, key) is None:
            setattr(args, key, sim if args.sim else ur3e)
    check_args(ap, args)
    if not 0 < args.vmax <= MAX_SPEED:
        ap.error(f"--vmax must be > 0 and at most {MAX_SPEED}")
    if args.hover < 0 or args.trials < 1:
        ap.error("--hover must be >= 0 and --trials >= 1")

    cfg = load_config(args.config)
    missing = [k for k in NEEDED + ["T_tool0_cam" if cfg.get("mount") == "wrist" else "T_base_cam"] if k not in cfg]
    if missing:
        raise SystemExit(f"config {args.config} is missing {missing}")
    if args.track and cfg["mount"] != "fixed":
        raise SystemExit("--track needs the fixed camera (mount fixed in the config): a wrist camera's pose changes "
                         "and its images blur while the arm moves")
    kin = UR3eKinematics(np.array(cfg["T_tool0_tcp"], float))
    ws, look_q = cfg["workspace"], np.array(cfg["look_q"])
    if not in_box(kin.tcp(look_q)[0], ws):
        raise SystemExit(f"the look pose's TCP {np.round(kin.tcp(look_q)[0], 3)} is outside the work area {ws}")
    rng = np.random.default_rng()
    obj = np.array(args.sim_object) if args.sim_object else np.r_[np.mean(ws[:2]), np.mean(ws[2:]),
                                                                    cfg["table_z"] + cfg["half"]]

    robot = connect(args, control=True, max_speed=args.vmax, accel=args.accel)
    out = args.out / f"pick_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True)
    logs, results, reason = [], [], "completed"
    try:
        if args.sim:
            sim_start(robot, look_q)
            gripper = Gripper(None, cfg["grip_time"], sim_width=None if args.hover else 1e3 * cfg["object_width"])
        else:
            q = robot.get_joint_positions()
            p, Rk, _ = kin.tcp(q)
            Tk = np.eye(4)
            Tk[:3, :3], Tk[:3, 3] = Rk, p
            dp, dr = pose_error(Tk, ur_pose_to_T(robot.get_tcp_pose()))
            print(f"kinematics vs robot TCP: {1e3 * dp:.1f} mm, {np.rad2deg(dr):.2f} deg")
            if dp > 0.005 or dr > np.deg2rad(2):
                raise Abort("kinematics disagree with get_tcp_pose() (> 5 mm or 2 deg): check the base frame and "
                            "T_tool0_tcp (eval_objects.py's kinematics check)")
            gripper = Gripper(args.ip, cfg["grip_time"])
            print(f"gripper width {gripper.width():.1f} mm, opening to 80 mm")
            gripper.open(80.0, args.force)
        with contextlib.nullcontext() if args.sim else RealSenseCamera(alignment="colour") as rs:
            # stop the arm before the camera shuts down: the last speedJ keeps it moving until speedStop. the camera
            # thread stops before the RealSense does
            cam = None
            try:
                if args.sim:
                    cam = SimCamera(cfg, kin, obj, rng, args.sim_move)
                else:
                    cam = CameraThread(rs, cfg, kin) if args.track else Camera(rs, cfg, kin)
                tau = np.broadcast_to(np.asarray(cfg["tau"], float), 6)
                delay = np.round(np.asarray(cfg.get("td", 0.0)) / DT).astype(int)
                for n in range(1, args.trials + 1):
                    if n > 1 and not args.sim:
                        input("move the object, then Enter for the next trial (Ctrl+C to stop) ")
                    if np.abs(robot.get_joint_velocities()).max() > 0.01:
                        raise Abort("the arm is moving")
                    q = robot.get_joint_positions()
                    if np.abs(q - look_q).max() > 0.02:
                        T = move_time(look_q - q, args.vmax, args.accel)
                        path = quintic(q, look_q, T)
                        low = min(kin.tcp(p)[0][2] for p in path[0][::20])
                        if low < cfg["table_z"] + 0.02:
                            raise Abort(f"the joint move to look_q would take the TCP down to z = {low:.3f} m: "
                                        "jog the arm closer to the look pose first")
                        if not args.sim:
                            input(f"the arm is {np.abs(q - look_q).max():.3f} rad from look_q: Enter moves it "
                                  f"there in {T:.0f} s, lowest TCP z {low:.3f} m (Ctrl+C to stop) ")
                        move = JointLoop(robot, JointVelKF(q, DT, tau, cfg["sigma_q"], delay=delay,
                                                           sigma_t=cfg.get("sigma_t", 0.0)),
                                         JointVelLQR(DT, tau, args.vmax, args.accel))
                        for r in zip(*path):
                            move.step(*r)
                        if not move.halt():
                            raise Abort("the arm did not stop at look_q")
                    log = defaultdict(list)
                    logs.append(log)
                    results.append(run_trial(robot, kin, cam, gripper, cfg, args, log, out, n))
                    print(f"trial {n}: {results[-1]}")
            except BaseException:
                robot.stop()
                raise
            finally:
                if cam is not None:
                    cam.close()
    except Abort as e:
        reason = f"aborted: {e}"
    except KeyboardInterrupt:
        reason = "stopped with Ctrl+C"
    except Exception as e:
        traceback.print_exc()
        reason = f"error: {e!r}"
    finally:
        robot.stop()
        robot.close()
        cv2.destroyAllWindows()
    print(reason)
    for n, log in enumerate(logs, 1):
        res = results[n - 1] if n <= len(results) else {"trial": n, "phase": "interrupted"}
        save(log, dict(res, reason=reason, config=cfg, args={k: str(v) for k, v in vars(args).items()}),
             out / f"trial{n}.npz")
    print(f"logs in {out}")
    done = sum(r.get("phase") == "done" for r in results)
    print(f"{done} of {len(logs)} trials finished" + (" (dry run)" if args.hover else ""))
