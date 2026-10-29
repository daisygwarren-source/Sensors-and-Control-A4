import argparse
import sys
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
STARTER = ROOT / "_reference" / "UR3_RG2_rosbridge"
sys.path.insert(0, str(STARTER))
MAX_SPEED = 0.05


def connect(args):
    if args.sim:
        from robot.mujoco_backend import MuJoCoUR3e
        return MuJoCoUR3e(model_path=STARTER / "models" / "ur3e_official" / "ur3e_velocity.xml", show_viewer=False)
    from robot.real_ur_backend import RealUR3e
    return RealUR3e(args.ip, enable_control=True, acceleration=args.accel)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Velocity step response of one joint: 0, +speed, 0, -speed, 0.")
    ap.add_argument("--ip")
    ap.add_argument("--out", type=Path, default=ROOT / "hw-data" / date.today().isoformat())
    ap.add_argument("--sim", action="store_true", help="MuJoCo backend, only checks the script runs")
    ap.add_argument("--joint", type=int, default=5, choices=range(6), help="5 = wrist 3")
    ap.add_argument("--speed", type=float, default=0.03, help=f"rad/s, at most {MAX_SPEED}")
    ap.add_argument("--accel", type=float, default=0.10, help="speedJ acceleration, rad/s^2")
    ap.add_argument("--hold", type=float, default=3.0, help="seconds at each speed")
    args = ap.parse_args()
    if not args.sim and not args.ip:
        ap.error("--ip is required unless --sim is set")
    if not 0 < args.speed <= MAX_SPEED:
        ap.error(f"--speed must be > 0 and at most {MAX_SPEED}")
    if args.accel <= 0 or args.hold <= 0:
        ap.error("--accel and --hold must be > 0")

    robot = connect(args)
    rate = robot.control_frequency
    plan = [(1.0, 0.0), (args.hold, args.speed), (1.0, 0.0), (args.hold, -args.speed), (1.0, 0.0)]
    v = np.concatenate([np.full(int(round(d * rate)), s) for d, s in plan])
    n = len(v)
    cmd = np.zeros((n, 6))
    cmd[:, args.joint] = v
    t, q, qd = np.empty(n), np.empty((n, 6)), np.empty((n, 6))
    try:
        if np.max(np.abs(robot.get_joint_velocities())) > 0.01:
            raise RuntimeError("Robot is already moving. Test cancelled.")
        t0 = time.perf_counter()
        for i in range(n):
            robot.set_joint_velocity(cmd[i])
            q[i] = robot.get_joint_positions()
            qd[i] = robot.get_joint_velocities()
            t[i] = time.perf_counter() - t0
    finally:
        robot.stop()
        robot.close()

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"step_j{args.joint}_v{args.speed:g}_a{args.accel:g}_{datetime.now():%Y%m%d_%H%M%S}.npz"
    with open(path, "xb") as f:
        np.savez(f, t=t, cmd=cmd, q=q, qd=qd, joint=args.joint, speed=args.speed,
                 accel=args.accel, hold=args.hold, rate=rate, sim=args.sim)

    j = args.joint
    dq = np.rad2deg(q[:, j] - q[0, j])
    print(f"\nsaved {path}")
    print(f"{n} cycles over {t[-1]:.2f} s, mean rate {(n - 1) / (t[-1] - t[0]):.1f} Hz")
    print(f"joint {j}: max {dq.max():+.2f} deg, min {dq.min():+.2f} deg, net {dq[-1]:+.3f} deg, "
          f"peak |qd| {np.abs(qd[:, j]).max():.4f} rad/s")
