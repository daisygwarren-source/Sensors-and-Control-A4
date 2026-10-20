"""Track B pipeline: wrist RGB-D + encoders -> estimators -> PBVS + LQR joint control -> pick and place.

    python main.py                 # viewer, back-to-back randomised picks until the window is closed
    python main.py --headless      # one trial, prints the result and saves plots to results/
"""
import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2
import mujoco

from simulator import build, TABLE_POS, TABLE_TOP_Z, OBJ_HALF, DROP_XY, ARM_JOINTS
from src.perception import T_base_cam, locate_cube, d435_noise, cube_mask, project
from src.model import RobotModel
from src.estimation import JointKF, KalmanFilter3D
from src.controller import JointLQR, ComputedTorque, PBVS, PickTask, PHASES

HOME = np.array([2.817, -1.788, 1.681, -1.463, -1.571, -0.325])  # look pose: wrist cam ~0.51 m up, centred over the spawn area, looking straight down
W, H = 640, 480
K_CALIB = Path(__file__).resolve().parent / "K_calib.npy"
CAM_DT = 0.1        # camera processed at 10 Hz
NIS_GATE = 16.27    # chi2, 3 dof, 99.9 %
LOG_EVERY = 5


def random_cube(rng, yaw_range=0.0):
    xy = TABLE_POS[:2] + rng.uniform([-0.15, -0.25], [0.15, 0.25])
    return xy, rng.uniform(-yaw_range, yaw_range) if yaw_range else 0.0


def make_trial(rng, motors=False, yaw_range=0.0):
    """New model with the target at a random tabletop position (spec requirement)."""
    xy, yaw = random_cube(rng, yaw_range)
    model = build(object_xy=xy, object_yaw=yaw, motors=motors).compile()
    data = mujoco.MjData(model)
    data.qpos[:6] = HOME
    if not motors:
        data.ctrl[:6] = HOME
    data.ctrl[6] = 0          # gripper open (0..255)
    mujoco.mj_forward(model, data)
    return model, data


def reset_trial(model, data, rng, yaw_range=0.0):
    xy, yaw = random_cube(rng, yaw_range)
    mujoco.mj_resetData(model, data)
    data.qpos[:6] = HOME
    a = model.joint("target_free").qposadr[0]
    data.qpos[a:a + 7] = [*xy, TABLE_TOP_Z + OBJ_HALF + 0.001, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    mujoco.mj_forward(model, data)


def grab_rgbd(renderer, data, cam="wrist_cam"):
    renderer.update_scene(data, camera=cam)
    rgb = renderer.render()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=cam)
    depth = renderer.render()          # metres along camera z
    renderer.disable_depth_rendering()
    return rgb, depth


def read_encoders(data, rng, bits=20):
    lsb = 2 * np.pi / 2 ** bits
    q = np.array([data.sensor(f"enc_{j}").data[0] for j in ARM_JOINTS])
    return np.round((q + lsb * rng.standard_normal(len(q))) / lsb) * lsb


def intrinsics(model, cam="wrist_cam"):
    """Pinhole K from MuJoCo fovy - use as ground-truth to compare against your calibration."""
    fovy = np.deg2rad(model.camera(cam).fovy[0])
    f = 0.5 * H / np.tan(fovy / 2)
    return np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])


def cube_truth_eval_only(data):
    b = data.body("base")
    return b.xmat.reshape(3, 3).T @ (data.body("target").xpos - b.xpos)


def truth_eval_only(data):
    # simulator ground truth, only ever logged and scored, never fed back
    b = data.body("base")
    pinch = b.xmat.reshape(3, 3).T @ (data.site("gripper/pinch").xpos - b.xpos)
    return data.qpos[:6].copy(), data.qvel[:6].copy(), cube_truth_eval_only(data), pinch


# cube_R scales the R measured from static trials at HOME; x4 brings NEES to about 3 (evaluate.py --study cube)
DEFAULTS = dict(cam_noise=1.0, enc_bits=20, mass_scale=1.0, use_dhat=True, lqr={}, kf={}, cube_R=4.0,
                cube_cv=False, cube_q=1e-7,
                push=None, nudge=None, t_max=30.0)


def run_trial(model, data, rng, viewer=None, verbose=False, **cfg):
    cfg = {**DEFAULTS, **cfg}
    dt = model.opt.timestep
    robot = RobotModel(build(motors=True).compile(), mass_scale=cfg["mass_scale"])
    renderer = mujoco.Renderer(model, height=H, width=W)
    K = np.load(K_CALIB)
    bits = cfg["enc_bits"]
    noise = dict(subpixel=0.08 * cfg["cam_noise"], wave=0.002 * cfg["cam_noise"])

    y = read_encoders(data, rng, bits)
    kf = JointKF(y, dt, 2 * np.pi / 2 ** bits * np.sqrt(13 / 12), **cfg["kf"])
    cube_kf = KalmanFilter3D(dt=CAM_DT, constant_velocity=cfg["cube_cv"], q_pos=cfg["cube_q"])
    cube_kf.R = cube_kf.R * cfg["cube_R"]
    P0 = cube_kf.P.copy()
    lqr = JointLQR(dt, **cfg["lqr"])
    ct = ComputedTorque(robot, model.actuator_ctrlrange[:6, 1])
    pbvs = PBVS(robot, dt, y)
    p_home, R_home, _ = robot.tcp(HOME)
    drop = (robot.R_bw @ (np.r_[DROP_XY, 0] - robot.p_wb))[:2]
    task = PickTask(p_home, R_home, drop, TABLE_TOP_Z, OBJ_HALF)

    cam_every = round(CAM_DT / dt)
    wrist = model.body("wrist_3_link").id
    cq = model.joint("target_free").qposadr[0]
    u, e, q_ref = np.zeros(6), np.zeros(6), y.copy()
    yaws, n_meas, n_rej, streak, nudged = [], 0, 0, 0, False
    log, res = defaultdict(list), dict(lifted=False, placed=False, events={})
    rgb, cube, t_end, frame = None, None, None, -1
    ui = dict(text=-1.0, phase=None, frame=-1, live=-1.0, t=0.0, wall=time.time())
    for k in range(int(cfg["t_max"] / dt)):
        t = k * dt

        if cfg["nudge"] and not nudged and task.phase == cfg["nudge"][0] and t - task.t0 > cfg["nudge"][1]:
            data.qpos[cq:cq + 2] += cfg["nudge"][2]
            nudged = True
        # step1 = positions, velocities and sensors for the current state, step2 integrates after ctrl is set
        mujoco.mj_step1(model, data)
        truth = truth_eval_only(data)  # logged only

        # encoders every step -> joint KF
        y = read_encoders(data, rng, bits)
        kf.predict(u)
        kf.update(y)
        q, qd, d = kf.X

        # camera at 10 Hz while the cube is the target -> cube KF
        if k % cam_every == 0 and task.camera_on:
            rgb, depth = d435_noise(*grab_rgbd(renderer, data), rng, **noise)
            frame = k
            z = locate_cube(rgb, depth, K, T_base_cam(q), with_yaw=True)
            if cube_kf.initialized:
                cube_kf.predict()
            if z is not None:
                p, a = z
                n_meas += 1
                yaws.append(a)
                if not cube_kf.initialized:
                    cube_kf.initialize(p)
                elif cube_kf.nis(p) < NIS_GATE:
                    cube_kf.update(p)
                    streak = 0
                else:
                    n_rej += 1
                    streak += 1
                    if streak >= 3:
                        # three misses in a row means the cube really moved, restart the filter on it
                        cube_kf.initialize(p)
                        cube_kf.P = P0.copy()
                        streak = 0
                log["meas_t"].append(t)
                log["meas"].append(p)
            cube = cube_kf.position.copy() if cube_kf.initialized else None
        yaw = np.angle(np.mean(np.exp(4j * np.array(yaws)))) / 4 if yaws else 0.0

        # task -> PBVS reference -> LQR -> computed torque, all on estimated state
        at = np.linalg.norm(e[:3]) < 0.002 and np.linalg.norm(e[3:]) < 0.02 and np.abs(q - q_ref).max() < 2e-3
        phase = task.phase
        p_des, R_des, v_max, grip = task(t, at, cube, yaw, n_meas)
        q_ref, qd_ref, qdd_ref, e = pbvs(p_des, R_des, v_max)
        v = lqr(q, qd, q_ref, qd_ref, qdd_ref)
        if cfg["use_dhat"]:
            v = v - d
        tau, u = ct(q, qd, v)
        data.ctrl[:6] = tau
        data.ctrl[6] = grip

        # scripted disturbance (environment side). left alone otherwise so viewer mouse pushes still work
        if cfg["push"]:
            ph, delay, F, dur = cfg["push"]
            on = task.phase == ph and delay <= t - task.t0 < delay + dur
            data.xfrc_applied[wrist, :3] = F if on else 0

        mujoco.mj_step2(model, data)

        # evaluation only from here down
        if task.phase != phase:
            res["events"][task.phase] = t
            tq, _, tc, tp = truth_eval_only(data)
            if task.phase == "grasp":
                res["grasp_err"] = np.linalg.norm((tp - tc)[:2])
                res["est_err"] = np.linalg.norm(cube - tc)
            if task.phase == "transfer":
                res["lifted"] = tc[2] > TABLE_TOP_Z + OBJ_HALF + 0.05
            if verbose:
                print(f"{t:6.2f} s  {task.phase}")
        if k % LOG_EVERY == 0:
            tq, tqd, tc, _ = truth
            log["t"].append(t)
            log["phase"].append(PHASES.index(task.phase))
            log["q"].append(q.copy())
            log["qd"].append(qd.copy())
            log["d"].append(d.copy())
            log["sig"].append(kf.sigma)
            log["P"].append(kf.P.copy())
            log["cube_P"].append(cube_kf.P[:3, :3].copy() if cube is not None else np.full((3, 3), np.nan))
            log["q_ref"].append(q_ref.copy())
            log["tau"].append(tau / ct.tau_max)
            log["cube"].append(cube if cube is not None else np.full(3, np.nan))
            log["cube_sig"].append(np.sqrt(np.diag(cube_kf.P)[:3]) if cube is not None else np.full(3, np.nan))
            log["true_q"].append(tq)
            log["true_qd"].append(tqd)
            log["true_d"].append(data.qacc[:6] - u)
            log["true_cube"].append(tc)

        if viewer is not None and k % 8 == 0:
            if not viewer.is_running():
                break
            draw(viewer, robot, cube if task.camera_on else None, p_des)
            if t - ui["text"] > 0.5 or task.phase != ui["phase"]:
                ui["text"], ui["phase"] = t, task.phase
                status(viewer, task.phase, cube, cube_kf, truth[2], d, None if task.camera_on else res.get("est_err"))
            if frame != ui["frame"]:
                ui["frame"] = frame
                wrist_view(rgb, K, T_base_cam(q), cube, task.phase)
            elif not task.camera_on and t - ui["live"] >= CAM_DT:
                # display only from grasp on, the pipeline has stopped using the camera and nothing here reaches the cube KF
                ui["live"] = t
                renderer.update_scene(data, camera="wrist_cam")
                wrist_view(renderer.render(), K, T_base_cam(q), None, f"{task.phase}, display only")
            # keep the wrist camera window pumping events after the camera turns off at grasp
            cv2.pollKey()
            viewer.sync()
            # real time when we can keep up, slow motion (no catching up) while the camera is busy
            time.sleep(max(0, (t - ui["t"]) - (time.time() - ui["wall"])))
            ui["t"], ui["wall"] = t, time.time()

        if task.phase in ("done", "fail"):
            t_end = t_end or t
            if t > t_end + 0.5:
                break

    renderer.close()
    tc = truth_eval_only(data)[2]
    w = robot.R_bw.T @ tc + robot.p_wb
    res["placed"] = task.phase == "done" and np.linalg.norm(w[:2] - DROP_XY) < 0.05 and tc[2] < TABLE_TOP_Z + OBJ_HALF + 0.01
    res.update(phase=task.phase, t=t, n_meas=n_meas, n_rej=n_rej)
    return res, {k: np.array(v) for k, v in log.items()}


def draw(viewer, robot, cube, p_des):
    scn = viewer.user_scn
    with viewer.lock():
        scn.ngeom = 0
        if cube is not None:
            mujoco.mjv_initGeom(scn.geoms[0], mujoco.mjtGeom.mjGEOM_BOX, np.full(3, OBJ_HALF * 1.15),
                                robot.to_world(cube), np.eye(3).ravel(), np.array([0.1, 0.9, 0.1, 0.35]))
            scn.ngeom = 1
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.006, 0, 0]),
                            robot.to_world(p_des), np.eye(3).ravel(), np.array([0.2, 0.4, 1.0, 0.8]))
        scn.ngeom += 1


def status(viewer, phase, cube, cube_kf, cube_true, d, grasp_err=None):
    # set_texts waits for a viewer frame, so this only runs a couple of times a second
    # grasp_err given = camera off since grasp: the cube KF is frozen and the true cube now moves with the gripper
    left, right = ["phase", "joint disturbance est, max |d| (rad/s^2)"], [phase, f"{np.abs(d).max():.2f}"]
    if cube is not None:
        s = 1000 * np.sqrt(np.diag(cube_kf.P)[:3])
        est, sig = " ".join(f"{1000 * c:7.1f}" for c in cube), " ".join(f"{c:7.2f}" for c in s)
        if grasp_err is None:
            left += ["cube est x y z (mm)", "1 sigma (mm)", "truth, eval only (mm)", "error (mm)"]
            right += [est, sig, " ".join(f"{1000 * c:7.1f}" for c in cube_true),
                      f"{1000 * np.linalg.norm(cube - cube_true):.2f}"]
        else:
            left += ["cube KF stopped at grasp, last est (mm)", "1 sigma (mm)", "error at grasp, eval only (mm)"]
            right += [est, sig, f"{1000 * grasp_err:.2f}"]
    viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT, "\n".join(left), "\n".join(right)))


def wrist_view(rgb, K, T_bc, cube, phase):
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    m = cube_mask(rgb)
    if m is not None:
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs, -1, (0, 255, 255), 1)
    if cube is not None:
        P = np.linalg.inv(T_bc) @ np.r_[cube, 1]
        if P[2] > 0:
            u, v = project(K, P[:3])
            cv2.drawMarker(img, (int(u), int(v)), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
    legend = "yellow: red mask, green: cube KF estimate" if cube is not None else "yellow: red mask"
    cv2.putText(img, f"{phase}  ({legend})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2)
    cv2.imshow("wrist camera", img)
    cv2.waitKey(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--yaw", type=float, default=0.0, help="cube yaw range +/- deg")
    ap.add_argument("--cam-noise", type=float, default=1.0)
    ap.add_argument("--push", action="store_true", help="40 N shove on the wrist while carrying the cube")
    ap.add_argument("--nudge", action="store_true", help="move the cube 3 cm while the arm approaches")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    cfg = dict(cam_noise=args.cam_noise)
    if args.push:
        cfg["push"] = ("transfer", 0.3, np.array([0, 40.0, 0]), 0.3)
    if args.nudge:
        cfg["nudge"] = ("approach", 0.4, rng.choice([-1, 1], 2) * 0.03)
    model, data = make_trial(rng, motors=True, yaw_range=np.deg2rad(args.yaw))

    if args.headless:
        from src.plots import plot_trial
        res, log = run_trial(model, data, rng, verbose=True, **cfg)
        print({k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in res.items() if k != "events"})
        out = Path(__file__).resolve().parent / "results"
        out.mkdir(exist_ok=True)
        plot_trial(log, out, f"seed{args.seed}")
        print("plots saved to", out)
    else:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
            n = 0
            while viewer.is_running():
                res, _ = run_trial(model, data, rng, viewer=viewer, verbose=True, **cfg)
                n += 1
                print(f"trial {n}: {res['phase']}, lifted={res['lifted']}, placed={res['placed']}, "
                      f"grasp offset {1000 * res.get('grasp_err', np.nan):.1f} mm (eval only)\n")
                reset_trial(model, data, rng, np.deg2rad(args.yaw))
