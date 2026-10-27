"""Shared helpers for the hardware scripts: connect, UR3e kinematics, RG2 gripper, config file, joint control loop.

    python real.py                       # kinematics self-test
    python real.py --write-sim-config    # placeholder config for --sim runs (hw-data/sim/config.json)
"""
import argparse
import contextlib
import io
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
STARTER = ROOT / "_reference" / "UR3_RG2_rosbridge"
TRACK_B = Path(__file__).resolve().parents[1]
for p in (STARTER, TRACK_B):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.perception import RED

DT = 0.01
MAX_SPEED = 0.05
SIM_MODEL = STARTER / "models" / "ur3e_official" / "ur3e_velocity.xml"
SIM_CONFIG = ROOT / "hw-data" / "sim" / "config.json"


def add_args(ap):
    ap.add_argument("--ip")
    ap.add_argument("--sim", action="store_true", help="MuJoCo arm, only checks the script runs")
    ap.add_argument("--config", type=Path, help=f"config.json (default with --sim: {SIM_CONFIG})")
    ap.add_argument("--out", type=Path, default=ROOT / "hw-data" / date.today().isoformat())


def check_args(ap, args):
    if not args.sim and not args.ip:
        ap.error("--ip is required unless --sim is set")
    if args.config is None:
        if not args.sim:
            ap.error("--config is required unless --sim is set")
        args.config = SIM_CONFIG
    if not args.config.exists():
        ap.error(f"config not found: {args.config}" + (" (make it with: python real.py --write-sim-config)"
                                                         if args.config == SIM_CONFIG else ""))


def connect(args, control=False, max_speed=MAX_SPEED, accel=0.10):
    if args.sim:
        from robot.mujoco_backend import MuJoCoUR3e
        return MuJoCoUR3e(model_path=SIM_MODEL, show_viewer=getattr(args, "viewer", False))
    from robot.real_ur_backend import RealUR3e
    return RealUR3e(args.ip, enable_control=control, max_joint_speed=max_speed, acceleration=accel)


def sim_start(robot, q, settle=1.0):
    # MuJoCoUR3e only: start at q and let it sag under gravity onto its position servo before anything reads q0.
    # contacts off: in the starter model they lock the base joint (it doesn't move under any command)
    import mujoco
    robot.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    robot.reset(np.asarray(q, float))
    for _ in range(int(settle / DT)):
        robot.set_joint_velocity(np.zeros(6))


class UR3eKinematics:
    # RTB's UR3e from the URDF. start="base" is the UR controller's base frame (the panel, get_tcp_pose()); RTB's
    # default base_link is turned 180 deg about z from it, which flips x and y. tcp() returns what RobotModel.tcp does
    def __init__(self, T_tool0_tcp=np.eye(4)):
        if not sys.flags.utf8_mode:
            raise SystemExit("URDFRobot('ur3e') needs Python UTF-8 mode: set PYTHONUTF8=1 or run python -X utf8")
        from roboticstoolbox.models.URDF.URDFRobot import URDFRobot
        self.robot = URDFRobot("ur3e")
        self.tool = np.asarray(T_tool0_tcp, float)

    def T_base_tool0(self, q):
        return self.robot.fkine(q, start="base", end="tool0").A

    def tcp(self, q):
        T = self.robot.fkine(q, start="base", end="tool0", tool=self.tool).A
        return T[:3, 3], T[:3, :3], self.robot.jacob0(q, start="base", end="tool0", tool=self.tool)

    def T_base_cam(self, q, T_tool0_cam):
        return self.T_base_tool0(q) @ np.asarray(T_tool0_cam, float)


def ur_pose_to_T(pose):
    # UR pose [x, y, z, rx, ry, rz], rotation vector
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(pose[3:6]).as_matrix()
    T[:3, 3] = pose[:3]
    return T


def pose_error(T_a, T_b):
    # position (m) and rotation (rad) between two poses
    return np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]), np.linalg.norm(Rotation.from_matrix(T_a[:3, :3] @ T_b[:3, :3].T).as_rotvec())


class Gripper:
    # OnRobot RG2 through the robot's port 41414, widths in mm. rg_grip doesn't report when the fingers stop, so poll
    # the width every 0.1 s until it changes by < 0.5 mm. call only with the arm stopped. ip=None is the --sim stub:
    # it sleeps grip_time and grip() returns sim_width (the object's width, or the target if None)
    def __init__(self, ip=None, grip_time=1.5, sim_width=None):
        self.rg, self.grip_time, self.sim_width, self.w = None, grip_time, sim_width, 80.0
        if ip:
            import onRobot.gripper as gripper
            self.rg = gripper.RG(ip, 0)

    def width(self):
        if self.rg is None:
            return self.w
        # the onRobot library prints every raw XML reply
        with contextlib.redirect_stdout(io.StringIO()):
            return float(self.rg.get_rg_width())

    def _move(self, width, force, stub_width):
        if self.rg is None:
            time.sleep(self.grip_time)
            self.w = stub_width
            return self.w
        with contextlib.redirect_stdout(io.StringIO()):
            self.rg.rg_grip(width, force)
        t0, last = time.perf_counter(), self.width()
        while time.perf_counter() - t0 < 10.0:
            time.sleep(0.1)
            w = self.width()
            # the fingers may not have started yet on the first polls
            if abs(w - last) < 0.5 and time.perf_counter() - t0 > 0.5:
                return w
            last = w
        return last

    def open(self, width=80.0, force=20.0):
        return self._move(width, force, width)

    def grip(self, width, force=20.0):
        return self._move(width, force, width if self.sim_width is None else self.sim_width)


def load_config(path):
    return json.loads(Path(path).read_text())


def update_config(path, **keys):
    # merge the keys in, never overwrite the whole file
    path = Path(path)
    cfg = load_config(path) if path.exists() else {}
    cfg.update(keys)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2, default=lambda o: o.tolist()) + "\n")
    print(f"config {path}: wrote {', '.join(keys)}")


def quintic(q0, q1, T, dt=DT):
    # q0 -> q1 in T s with zero velocity and acceleration at both ends. peaks: qd 1.875 d / T, qdd 5.77 d / T^2
    s = np.arange(int(round(T / dt)) + 1) * dt / T
    d = np.asarray(q1, float) - q0
    p = 10 * s**3 - 15 * s**4 + 6 * s**5
    v = (30 * s**2 - 60 * s**3 + 30 * s**4) / T
    a = (60 * s - 180 * s**2 + 120 * s**3) / T**2
    return q0 + np.outer(p, d), np.outer(v, d), np.outer(a, d)


def move_time(dq, vmax, amax, share=0.6):
    # shortest quintic whose peak speed and acceleration use at most `share` of the limits
    dq = np.abs(dq).max()
    return max(1.0, 1.875 * dq / (share * vmax), np.sqrt(5.77 * dq / (share * amax)))


class Abort(Exception):
    pass


class JointLoop:
    # one 100 Hz cycle on the arm: read encoders, KF predict with the last command then update, LQR on the estimate,
    # send. set_joint_velocity holds the pace. the tracking and joint-bounds checks run before anything is sent
    def __init__(self, robot, kf, lqr, use_qd=False, use_dhat=True, max_err=0.1, bounds=None):
        self.robot, self.kf, self.lqr = robot, kf, lqr
        self.use_qd, self.use_dhat, self.max_err, self.bounds = use_qd, use_dhat, max_err, bounds
        self.q = self.qd = None

    def read(self):
        self.q, self.qd = self.robot.get_joint_positions(), self.robot.get_joint_velocities()
        self.kf.predict(self.lqr.u)
        self.kf.update(np.r_[self.q, self.qd] if self.use_qd else self.q)
        return self.kf.X

    def step(self, q_ref, qd_ref, qdd_ref):
        self.read()
        return self.command(q_ref, qd_ref, qdd_ref)

    def command(self, q_ref, qd_ref, qdd_ref):
        # on the estimate from the last read()
        q, qd, d = self.kf.X
        err = np.abs(q - q_ref).max()
        if err > self.max_err:
            raise Abort(f"|q_hat - q_ref| = {err:.3f} rad > {self.max_err} rad")
        if self.bounds is not None and ((self.q < self.bounds[0]) | (self.q > self.bounds[1])).any():
            raise Abort(f"joint outside its allowed range, q = {np.round(self.q, 3)}")
        u, clipped = self.lqr(q, qd, q_ref, qd_ref, qdd_ref, d if self.use_dhat else 0.0)
        self.robot.set_joint_velocity(u)
        return u, clipped

    def halt(self, timeout=5.0):
        # command ramps to zero at the LQR's rate limit, then wait for |qd| < 0.002 rad/s. KF keeps running
        a = (self.lqr.a_max or np.inf) * self.lqr.dt
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            self.read()
            self.lqr.u = np.clip(0.0, self.lqr.u - a, self.lqr.u + a)
            self.robot.set_joint_velocity(self.lqr.u)
            if not self.lqr.u.any() and np.abs(self.qd).max() < 0.002:
                return True
        return False


SIM_PLACEHOLDER = dict(
    # tau: the starter sim's own lag (fit_steps.py on a step_test.py --sim file); real values come from fit_steps.py
    tau=[0.2] * 6, td=[0.0] * 6, sigma_q=[1e-5] * 6, sigma_qd=[1e-3] * 6,
    # D435 colour 640 x 480, typical factory intrinsics
    K=[[615.0, 0, 320.0], [0, 615.0, 240.0], [0, 0, 1]],
    mount="wrist",
    # guess: camera 7 cm off the tool axis and 3 cm below the flange, looking along tool0 z
    T_tool0_cam=[[1, 0, 0, 0], [0, 1, 0, 0.07], [0, 0, 1, 0.03], [0, 0, 0, 1]],
    T_tool0_tcp=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.19], [0, 0, 0, 1]],
    # tool straight down, camera 0.32 m above the object
    look_q=np.deg2rad([0, -85, 65, -70, -90, 0]).round(4).tolist(),
    table_z=0.0, half=0.025, object_width=0.05,
    hsv=RED,
    workspace=[-0.45, -0.15, -0.30, 0.05],
    grip_time=1.0,
)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write-sim-config", action="store_true")
    args = ap.parse_args()
    if args.write_sim_config:
        update_config(SIM_CONFIG, **SIM_PLACEHOLDER)
        raise SystemExit
    tool = np.eye(4)
    tool[2, 3] = 0.19
    kin = UR3eKinematics(tool)
    p0 = kin.T_base_tool0(np.zeros(6))[:3, 3]
    print("T_base_tool0(0) =", p0.round(4), "(UR3e: -0.457 -0.223 0.067)")
    assert np.allclose(p0, [-0.457, -0.223, 0.067], atol=1e-3), "base frame wrong, use start='base'"
    q = np.array([0.3, -1.2, 1.1, -1.4, -1.5, 0.4])
    p, R, J = kin.tcp(q)
    Jn = np.zeros((6, 6))
    for i in range(6):
        dq = np.zeros(6)
        dq[i] = 1e-6
        p1, R1, _ = kin.tcp(q + dq)
        Jn[:, i] = np.r_[p1 - p, Rotation.from_matrix(R1 @ R.T).as_rotvec()] / 1e-6
    print(f"tcp Jacobian vs finite differences: max error {np.abs(J - Jn).max():.1e}")
    assert np.abs(J - Jn).max() < 1e-5
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    pose = np.r_[p, Rotation.from_matrix(R).as_rotvec()]
    print("ur_pose_to_T round trip error (m, rad):", np.round(pose_error(ur_pose_to_T(pose), T), 12))
    t0 = time.perf_counter()
    for _ in range(1000):
        kin.tcp(q)
    print(f"tcp(q): {(time.perf_counter() - t0):.3f} ms per call")
    print("kinematics OK")
