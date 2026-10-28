"""Is loc_offset hand-eye error or locate_cube error? Marker 0 lying flat on the table: where the camera puts its centre
(solvePnP on the corners, then T_base_cam) against the TCP touching that centre. No colour mask or depth is involved,
so this error is the camera pose's (hand-eye and K) alone.

capture.py --label object: space with the marker on the table and the arm out of view, then t with the TCP on the
marker's centre, at several places on the table.

    python marker_check.py FOLDER --config ...\\config.json --marker 0.1
"""
import argparse
from pathlib import Path

import numpy as np
import cv2

from real import UR3eKinematics, find_marker, load_config, marker_detector
from calib_handeye import corners_3d, pnp
from eval_objects import by_role, summary

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--marker", type=float, default=0.100, help="marker size (outer black edge), m")
    ap.add_argument("--pairs", type=int, nargs="+", help="image, touch, image, touch, ... (default: the roles)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    caps = {p.name[:3]: load_config(p) for p in sorted(args.folder.glob("*_pose.json"))}
    if args.pairs:
        named = [f"{n:03d}" for n in args.pairs]
        if len(named) % 2 or any(n not in caps for n in named):
            raise SystemExit(f"--pairs needs an even number of existing captures, have {sorted(caps)}")
        pairs = list(zip(named[::2], named[1::2]))
    else:
        pairs = by_role(caps)[0]
    if not pairs:
        raise SystemExit(f"no image / touch pairs in {args.folder} (capture.py roles, or --pairs)")
    kin = UR3eKinematics(np.array(cfg["T_tool0_tcp"], float))
    K, P, detector = np.array(cfg["K"], float), corners_3d(args.marker), marker_detector()

    errs = []
    print("image  touch    err x    err y    err z   |xy|  (mm, camera's marker centre - TCP touch)")
    for i, t in pairs:
        uv = find_marker(detector, cv2.imread(str(args.folder / f"{i}_rgb.png"), cv2.IMREAD_GRAYSCALE))
        if uv is None:
            print(f"{i:>5s}  {t:>5s}  marker 0 not found")
            continue
        q = np.array(caps[i]["q"])
        T_bc = kin.T_base_cam(q, cfg["T_tool0_cam"]) if cfg["mount"] == "wrist" else np.array(cfg["T_base_cam"])
        p = (T_bc @ pnp(uv, K, P))[:3, 3]
        c = caps[t]
        touch = np.array(c["tcp"][:3]) if c.get("tcp") is not None else kin.tcp(c["q"])[0]
        e = 1e3 * (p - touch)
        errs.append(e)
        print(f"{i:>5s}  {t:>5s}  {e[0]:7.1f}  {e[1]:7.1f}  {e[2]:7.1f}  {np.linalg.norm(e[:2]):5.1f}")
    if not errs:
        raise SystemExit("marker found in none of the images")
    e = np.array(errs)
    bias, loc = e.mean(0), 1e3 * np.array(cfg.get("loc_offset", [np.nan] * 3))
    print(f"found {len(e)} of {len(pairs)}: {summary(e)}")
    print(f"marker bias (camera pose alone):  x {bias[0]:6.1f}  y {bias[1]:6.1f}  z {bias[2]:6.1f} mm")
    print(f"loc_offset (locate_cube's bias):  x {loc[0]:6.1f}  y {loc[1]:6.1f}  z {loc[2]:6.1f} mm")
    print("the part of loc_offset the marker shares is camera pose (hand-eye or K); what's left is locate_cube's")
