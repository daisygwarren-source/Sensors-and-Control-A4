"""Capture RGB-D frames with the arm's joint angles and TCP pose.

    python capture.py --label checkerboard --no-robot
    python capture.py --label handeye --ip ROBOT_IP
    python capture.py --label object --ip ROBOT_IP

checkerboard / handeye: space saves the frame, only if the board / marker is found in it (drawn on the preview).
object: space = object image (arm out of view), t = truth touch (TCP on the object's centre, after its image),
b = table touch, c = work-area corner. Each pose file records the role, which eval_objects.py reads.
"""
import argparse
import sys
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import cv2
from realsense_lab import DemoCamera, RealSenseCamera, save_frame, save_json

from real import find_marker, marker_detector, rtde_link

ROOT = Path(__file__).resolve().parents[3]
STARTER = ROOT / "_reference" / "UR3_RG2_rosbridge"
sys.path.insert(0, str(STARTER))
WINDOW = "capture (esc = quit)"
CHECKERBOARD = (8, 6)  # inner corners, perception.calibrate's default
KEYS = {"checkerboard": {32: "image"}, "handeye": {32: "image"},
        "object": {32: "image", ord("t"): "truth", ord("b"): "table", ord("c"): "corner"}}
HELP = {"checkerboard": "space: save (board needed)", "handeye": "space: save (marker 0 needed)",
        "object": "space: image  t: truth touch  b: table  c: corner"}


def connect(args):
    if args.sim:
        from robot.mujoco_backend import MuJoCoUR3e
        return MuJoCoUR3e(model_path=STARTER / "models" / "ur3e_official" / "ur3e_velocity.xml", show_viewer=False)
    from robot.real_ur_backend import RealUR3e
    return RealUR3e(args.ip, enable_control=False)


def detect(label, bgr, detector, fast=True):
    # what the calibration will look for: (found, preview with it drawn). object captures need nothing.
    # checkerboard: the preview uses the fast check, which missed 10 of the 42 boards calibrate() found in the 28 Oct
    # images, so a save is checked again (fast=False) the way calibrate() looks: blurred, no fast check
    view = bgr.copy()
    if label == "object":
        return True, view
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if label == "handeye":
        uv = find_marker(detector, gray)
        if uv is not None:
            cv2.aruco.drawDetectedMarkers(view, [uv.reshape(1, 4, 2).astype(np.float32)], np.array([[0]]))
        return uv is not None, view
    if fast:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        ok, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags=flags)
    else:
        ok, corners = cv2.findChessboardCorners(cv2.GaussianBlur(gray, (5, 5), 0), CHECKERBOARD)
    if ok:
        cv2.drawChessboardCorners(view, CHECKERBOARD, corners, ok)
    return ok, view


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    detector = marker_detector()
    n, status = 0, ""
    try:
        with (DemoCamera() if args.demo else RealSenseCamera(alignment="colour")) as camera:
            folder.mkdir(parents=True, exist_ok=False)
            if not args.demo:
                save_json(folder / "intrinsics.json", camera.calibration)
            print(f"\nsaving to {folder}\n{HELP[args.label]}, esc = quit")
            while True:
                frame = camera.capture()
                bgr = cv2.cvtColor(frame.native_rgb, cv2.COLOR_RGB2BGR)
                found, view = detect(args.label, bgr, detector)
                cv2.putText(view, f"{HELP[args.label]}   {status}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 2)
                cv2.imshow(WINDOW, view)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                role = KEYS[args.label].get(key)
                if role is None:
                    continue
                stem = f"{n:03d}"
                if not found and args.label == "checkerboard":
                    found = detect(args.label, bgr, detector, fast=False)[0]
                if not found:
                    status = f"{'board' if args.label == 'checkerboard' else 'marker'} not found, not saved"
                    print(status)
                    continue
                pose = None
                if robot is not None:
                    # getActualQ() keeps returning the last reading after an RTDE drop: only save a pose read while
                    # the robot's timestamp is still moving
                    link = rtde_link(robot)
                    tcp = robot.get_tcp_pose().tolist() if hasattr(robot, "get_tcp_pose") else None
                    pose = {"q": robot.get_joint_positions().tolist(), "tcp": tcp, "time": time.time(), "role": role}
                    if link is not None:
                        time.sleep(0.02)
                        up, stamp = rtde_link(robot)
                        if not (link[0] and up) or stamp <= link[1]:
                            status = "robot link lost, not saved"
                            print(f"{status} (restart capture.py to reconnect)")
                            continue
                if not cv2.imwrite(str(folder / f"{stem}_rgb.png"), bgr):
                    raise RuntimeError(f"Could not write {stem}_rgb.png")
                save_frame(folder / f"{stem}_frame.npz", frame)
                if pose is not None:
                    save_json(folder / f"{stem}_pose.json", pose)
                status = f"saved {stem} ({role})"
                print(f"capture {stem} ({role})")
                n += 1
    finally:
        cv2.destroyAllWindows()
        if robot is not None:
            robot.close()
    print(f"{n} captures in {folder}")
