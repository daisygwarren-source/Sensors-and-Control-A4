"""Lunch: on the capture.py --label object run, (1) check the arm kinematics against the robot's own TCP, (2) tune the
colour mask on the real object, (3) measure the localisation error against the true positions, (4) table height and
work area. Writes hsv, table_z, workspace and look_q to the config.

    python eval_objects.py FOLDER --config ...\\config.json --pairs 0 1 2 3 4 5 --table 12 --corners 13 14

--pairs: image capture, then its true-position capture, for each object position (default: consecutive pairs of
the captures not named in --table / --corners). Mask tuner: n / p next / previous image, s save, Esc quit unsaved.
"""
import argparse
from pathlib import Path

import numpy as np
import cv2
from realsense_lab import load_frame

from real import UR3eKinematics, load_config, pose_error, update_config, ur_pose_to_T
from src.perception import RED, cube_mask, locate_cube, red_mask

WINDOW = "mask (n/p image, s save, esc quit)"
BARS = ["H1 lo", "H1 hi", "H2 lo", "H2 hi", "S lo", "V lo"]


def kinematics_check(caps, kin):
    rows = [(n, *kin.tcp(c["q"])[:2], ur_pose_to_T(c["tcp"])) for n, c in caps.items() if c.get("tcp") is not None]
    if not rows:
        print("no captures with a robot tcp (sim?), kinematics check skipped")
        return
    print("capture  position error (mm)  rotation error (deg)")
    off, errs = [], []
    for n, p, R, T in rows:
        Tk = np.eye(4)
        Tk[:3, :3], Tk[:3, 3] = R, p
        dp, dr = pose_error(Tk, T)
        errs.append((1e3 * dp, np.rad2deg(dr)))
        print(f"{n:>7s}  {1e3 * dp:19.2f}  {np.rad2deg(dr):20.3f}")
        off.append(R.T @ (T[:3, 3] - p))
    off = np.array(off)
    worst, worst_r = np.max(errs, axis=0)
    if worst < 1.5 and worst_r < 0.5:
        print(f"kinematics OK: worst {worst:.2f} mm, {worst_r:.2f} deg (expect ~1 mm and 0.5 deg)")
        return
    flipped = np.mean([np.linalg.norm(T[:3, 3] - p * [-1, -1, 1]) for _, p, _, T in rows])
    if flipped < 0.5 * np.mean([np.linalg.norm(T[:3, 3] - p) for _, p, _, T in rows]):
        print("x and y have the wrong sign: the base frame is wrong (use start='base')")
    elif np.linalg.norm(off.std(0)) < 0.3 * np.linalg.norm(off.mean(0)):
        print(f"a constant offset in the tool frame: T_tool0_tcp is off by {np.round(1e3 * off.mean(0), 1)} mm "
              "(add it to T_tool0_tcp's translation)")
    else:
        print(f"worst error {worst:.1f} mm and not a simple flip or tool offset: check q and the TCP setting")


def tune(images, hsv):
    ranges = [list(map(list, r)) for r in hsv] + [[[180, 0, 0], [0, 255, 255]]]
    start = [ranges[0][0][0], ranges[0][1][0], ranges[1][0][0], ranges[1][1][0], ranges[0][0][1], ranges[0][0][2]]
    cv2.namedWindow(WINDOW)
    for name, v in zip(BARS, start):
        cv2.createTrackbar(name, WINDOW, int(v), 180 if name[0] == "H" else 255, lambda _: None)
    print("hue is 0-180; set H2 lo above H2 hi to use one range only (non-red objects)")
    i = 0
    while True:
        h1l, h1h, h2l, h2h, s, v = (cv2.getTrackbarPos(b, WINDOW) for b in BARS)
        cur = [[[h1l, s, v], [h1h, 255, 255]]] + ([[[h2l, s, v], [h2h, 255, 255]]] if h2l <= h2h else [])
        rgb = images[i % len(images)]
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        img[red_mask(rgb, cur) > 0] = (0, 255, 255)
        blob = cube_mask(rgb, cur)
        if blob is not None:
            cs, _ = cv2.findContours(blob.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cs, -1, (0, 255, 0), 2)
        cv2.putText(img, f"image {i % len(images) + 1}/{len(images)}  yellow: mask, green: largest blob", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.imshow(WINDOW, img)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("n"):
            i += 1
        elif key == ord("p"):
            i -= 1
        elif key == ord("s"):
            cv2.destroyWindow(WINDOW)
            return cur
        elif key == 27 or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()
            return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--pairs", type=int, nargs="+", help="image, true position, image, true position, ...")
    ap.add_argument("--table", type=int, help="capture with the gripper tip touching the table")
    ap.add_argument("--corners", type=int, nargs=2, help="captures at two opposite corners of the safe work area")
    ap.add_argument("--tip-offset", type=float, default=0.0,
                    help="m from the TCP point down to the part that touched the table (0 if the TCP is the tip)")
    ap.add_argument("--no-tune", action="store_true", help="skip the mask tuner, use the config's hsv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    caps = {p.name[:3]: load_config(p) for p in sorted(args.folder.glob("*_pose.json"))}
    if not caps:
        raise SystemExit(f"no *_pose.json in {args.folder}")
    name = lambda n: f"{n:03d}"
    kin = UR3eKinematics(np.array(cfg.get("T_tool0_tcp", np.eye(4)), float))

    print("1. kinematics: kin.tcp(q) against the robot's get_tcp_pose()")
    kinematics_check(caps, kin)

    special = [name(n) for n in ([args.table] if args.table is not None else []) + (args.corners or [])]
    order = [n for n in caps if n not in special]
    pairs = [name(n) for n in args.pairs] if args.pairs else order[:len(order) // 2 * 2]
    if len(pairs) % 2 or not pairs or any(n not in caps for n in pairs):
        raise SystemExit(f"--pairs needs an even number of existing captures, have {sorted(caps)}")
    pairs = list(zip(pairs[::2], pairs[1::2]))
    frames = {i: load_frame(args.folder / f"{i}_frame.npz") for i, _ in pairs}

    keys = {"look_q": caps[pairs[0][0]]["q"]}
    hsv = cfg.get("hsv", RED)
    if not args.no_tune:
        print("\n2. mask tuning")
        tuned = tune([frames[i].rgb for i, _ in pairs], hsv)
        if tuned is None:
            print("not saved, keeping the config's hsv")
        else:
            hsv = keys["hsv"] = tuned
            print("hsv =", tuned)

    def tcp_p(n):
        c = caps[n]
        return np.array(c["tcp"][:3]) if c.get("tcp") is not None else kin.tcp(c["q"])[0]

    table_z = cfg["table_z"]
    if args.table is not None:
        table_z = keys["table_z"] = float(tcp_p(name(args.table))[2] - args.tip_offset)
        print(f"\ntable_z = {table_z:.4f} m (capture {name(args.table)})")
    if args.corners:
        a, b = tcp_p(name(args.corners[0])), tcp_p(name(args.corners[1]))
        keys["workspace"] = [min(a[0], b[0]), max(a[0], b[0]), min(a[1], b[1]), max(a[1], b[1])]
        print("workspace xmin xmax ymin ymax =", np.round(keys["workspace"], 4))

    print("\n3. localisation error against the true-position captures (mm)")
    K, half = np.array(cfg["K"], float), cfg["half"]
    errs = []
    print("image  truth      err x    err y    err z  |xy|")
    for i, t in pairs:
        f = frames[i]
        q = np.array(caps[i]["q"])
        T_bc = kin.T_base_cam(q, cfg["T_tool0_cam"]) if cfg["mount"] == "wrist" else np.array(cfg["T_base_cam"])
        p = locate_cube(f.rgb, f.depth_m, K, T_bc, table_z, half, hsv=hsv)
        if p is None:
            print(f"{i:>5s}  {t:>5s}  not found (mask empty or touching the image edge, or too few table points)")
            continue
        e = 1e3 * (p - np.r_[tcp_p(t)[:2], table_z + half])
        errs.append(e)
        print(f"{i:>5s}  {t:>5s}  {e[0]:7.1f}  {e[1]:7.1f}  {e[2]:7.1f}  {np.linalg.norm(e[:2]):5.1f}")
    if errs:
        e = np.array(errs)
        xy = np.linalg.norm(e[:, :2], axis=1)
        print(f"found {len(e)} of {len(pairs)}. |xy| mean {xy.mean():.1f} max {xy.max():.1f} mm; "
              f"|z| mean {np.abs(e[:, 2]).mean():.1f} max {np.abs(e[:, 2]).max():.1f} mm; "
              f"mean (bias) x {e[:, 0].mean():.1f} y {e[:, 1].mean():.1f} z {e[:, 2].mean():.1f} mm")
    update_config(args.config, **keys)
