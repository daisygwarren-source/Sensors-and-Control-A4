import argparse
import sys
import time
from datetime import date, datetime
from pathlib import Path

import cv2
from realsense_lab import DemoCamera, RealSenseCamera, save_frame, save_json

ROOT = Path(__file__).resolve().parents[3]
STARTER = ROOT / "_reference" / "UR3_RG2_rosbridge"
sys.path.insert(0, str(STARTER))
WINDOW = "capture (space = save, esc = quit)"


def connect(args):
    if args.sim:
        from robot.mujoco_backend import MuJoCoUR3e
        return MuJoCoUR3e(model_path=STARTER / "models" / "ur3e_official" / "ur3e_velocity.xml", show_viewer=False)
    from robot.real_ur_backend import RealUR3e
    return RealUR3e(args.ip, enable_control=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Capture RGB-D frames with the arm's joint angles and TCP pose.")
    ap.add_argument("--label", required=True, choices=["checkerboard", "handeye", "object"])
    ap.add_argument("--ip")
    ap.add_argument("--out", type=Path, default=ROOT / "hw-data" / date.today().isoformat())
    ap.add_argument("--sim", action="store_true", help="MuJoCo backend, only checks the script runs")
    ap.add_argument("--demo", action="store_true", help="fake DemoCamera instead of the RealSense")
    ap.add_argument("--no-robot", action="store_true", help="images only, no pose files")
    args = ap.parse_args()
    if not (args.sim or args.no_robot or args.ip):
        ap.error("--ip is required unless --sim or --no-robot is set")

    folder = args.out / f"{args.label}_{datetime.now():%Y%m%d_%H%M%S}"
    robot = None if args.no_robot else connect(args)
    n = 0
    try:
        with (DemoCamera() if args.demo else RealSenseCamera(alignment="colour")) as camera:
            folder.mkdir(parents=True, exist_ok=False)
            if not args.demo:
                save_json(folder / "intrinsics.json", camera.calibration)
            print(f"\nsaving to {folder}\nspace = capture, esc = quit")
            while True:
                frame = camera.capture()
                bgr = cv2.cvtColor(frame.native_rgb, cv2.COLOR_RGB2BGR)
                cv2.imshow(WINDOW, bgr)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key != 32:
                    continue
                stem = f"{n:03d}"
                if not cv2.imwrite(str(folder / f"{stem}_rgb.png"), bgr):
                    raise RuntimeError(f"Could not write {stem}_rgb.png")
                save_frame(folder / f"{stem}_frame.npz", frame)
                if robot is not None:
                    tcp = robot.get_tcp_pose().tolist() if hasattr(robot, "get_tcp_pose") else None
                    save_json(folder / f"{stem}_pose.json",
                              {"q": robot.get_joint_positions().tolist(), "tcp": tcp, "time": time.time()})
                print(f"capture {stem}")
                n += 1
    finally:
        cv2.destroyAllWindows()
        if robot is not None:
            robot.close()
    print(f"{n} captures in {folder}")
