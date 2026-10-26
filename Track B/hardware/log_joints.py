import argparse
import sys
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
STARTER = ROOT / "_reference" / "UR3_RG2_rosbridge"
sys.path.insert(0, str(STARTER))


def connect(args):
    if args.sim:
        from robot.mujoco_backend import MuJoCoUR3e
        return MuJoCoUR3e(model_path=STARTER / "models" / "ur3e_official" / "ur3e_velocity.xml", show_viewer=False)
    from robot.real_ur_backend import RealUR3e
    return RealUR3e(args.ip, enable_control=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Log joint encoders with the arm stationary (joint KF R).")
    ap.add_argument("--ip")
    ap.add_argument("--out", type=Path, default=ROOT / "hw-data" / date.today().isoformat())
    ap.add_argument("--sim", action="store_true", help="MuJoCo backend, only checks the script runs")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--rate", type=float, default=100.0, help="Hz")
    args = ap.parse_args()
    if not args.sim and not args.ip:
        ap.error("--ip is required unless --sim is set")
    if args.seconds <= 0 or args.rate <= 0:
        ap.error("--seconds and --rate must be > 0")

    n = int(round(args.seconds * args.rate))
    dt = 1.0 / args.rate
    t, q, qd = np.empty(n), np.empty((n, 6)), np.empty((n, 6))
    robot = connect(args)
    try:
        t0 = time.perf_counter()
        for i in range(n):
            t[i] = time.perf_counter() - t0
            q[i] = robot.get_joint_positions()
            qd[i] = robot.get_joint_velocities()
            pause = t0 + (i + 1) * dt - time.perf_counter()
            if pause > 0:
                time.sleep(pause)
    finally:
        robot.close()

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"joints_{datetime.now():%Y%m%d_%H%M%S}.npz"
    with open(path, "xb") as f:
        np.savez(f, t=t, q=q, qd=qd, rate=args.rate)

    print(f"\nsaved {path}")
    print(f"{n} samples over {t[-1]:.2f} s, mean rate {(n - 1) / (t[-1] - t[0]):.1f} Hz")
    print("joint  std q [rad]  std qd [rad/s]")
    for j, (sq, sqd) in enumerate(zip(q.std(axis=0), qd.std(axis=0))):
        print(f"{j:5d}  {sq:11.2e}  {sqd:14.2e}")
