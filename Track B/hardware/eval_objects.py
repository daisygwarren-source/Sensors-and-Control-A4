"""Lunch: on the capture.py --label object run, (1) check the arm kinematics against the robot's own TCP, (2) tune the
colour mask on the real object, (3) measure the localisation error against the true positions, with each image's yaw
and an offset fitted on half the pairs and scored on the other half, (4) table height and work area. Writes hsv,
table_z, workspace and look_q to the config, and loc_offset with --set-offset.

    python eval_objects.py FOLDER --config ...\\config.json --set-offset --plot
    python eval_objects.py FOLDER --config ...\\config.json --pairs 0 1 2 3 4 5 --table 12 --corners 13 14

Captures with roles (capture.py: space image, t truth, b table, c corner) pair each truth touch with the image just
before it. --pairs (image, truth, image, truth, ...), --table and --corners override the roles; captures without
roles need --table / --corners, and pair the rest in order. Mask tuner: n / p next / previous image, s save, Esc quit
unsaved.
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


def by_role(caps):
    # capture.py --label object roles: each truth touch goes with the object image captured just before it
    pairs, table, corners, image = [], [], [], None
    for n, c in caps.items():
        role = c.get("role")
        if role == "image":
            if image is not None:
                print(f"image {image} has no truth touch after it, not used")
            image = n
        elif role == "truth":
            if image is None:
                print(f"truth touch {n} has no image before it, not used")
            else:
                pairs.append((image, n))
                image = None
        elif role == "table":
            table.append(n)
        elif role == "corner":
            corners.append(n)
    if image is not None:
        print(f"image {image} has no truth touch after it, not used")
    return pairs, table, corners


def summary(e):
    xy = np.linalg.norm(e[:, :2], axis=1)
    return f"|xy| mean {xy.mean():.1f} max {xy.max():.1f} mm, |z| mean {np.abs(e[:, 2]).mean():.1f} mm"


def plot(path, truth, est, offset, held):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True)
    for ax, p, title in ((axes[0], est, "raw estimate"),
                         (axes[1], est - offset, f"minus the mean bias {np.round(1e3 * offset, 1)} mm (in-sample)\n"
                                                 f"{held} (offset from the even ones)")):
        e = 1e3 * (p - truth)
        ax.plot(*truth[:, :2].T, "ko", ms=4, label="truth (TCP touch)")
        ax.plot(*p[:, :2].T, "x", color="C3", ms=5, label="estimate")
        ax.quiver(*truth[:, :2].T, *(p - truth)[:, :2].T, angles="xy", scale_units="xy", scale=1, color="C3",
                  width=0.004, label="truth -> estimate")
        for (x, y), d in zip(truth[:, :2], np.linalg.norm(e[:, :2], axis=1)):
            ax.annotate(f"{d:.0f}", (x, y), textcoords="offset points", xytext=(4, -10), fontsize=7)
        ax.set(xlabel="base x (m)", aspect="equal")
        ax.set_title(f"{title}\n{summary(e)}", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("base y (m)")
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("localisation error in the table plane (arrows to scale, labels in mm)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


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
    ap.add_argument("--set-offset", action="store_true", help="write the mean bias over all pairs as loc_offset")
    ap.add_argument("--plot", action="store_true", help="arrows truth -> estimate in FOLDER/eval_objects.png")
    args = ap.parse_args()

    cfg = load_config(args.config)
    caps = {p.name[:3]: load_config(p) for p in sorted(args.folder.glob("*_pose.json"))}
    if not caps:
        raise SystemExit(f"no *_pose.json in {args.folder}")
    name = lambda n: f"{n:03d}"
    kin = UR3eKinematics(np.array(cfg.get("T_tool0_tcp", np.eye(4)), float))

    print("1. kinematics: kin.tcp(q) against the robot's get_tcp_pose()")
    kinematics_check(caps, kin)

    roles = any("role" in c for c in caps.values())
    if roles:
        pairs, table, corners = by_role(caps)
        print(f"\nroles: {len(pairs)} image / truth pairs, table {table}, corners {corners}")
    else:
        # captures from before roles: pairs in capture order, minus the ones named in --table / --corners
        special = [name(n) for n in ([args.table] if args.table is not None else []) + (args.corners or [])]
        order = [n for n in caps if n not in special]
        pairs = list(zip(order[0:len(order) // 2 * 2:2], order[1:len(order) // 2 * 2:2]))
        table, corners = [], []
    if args.pairs:
        named = [name(n) for n in args.pairs]
        if len(named) % 2 or any(n not in caps for n in named):
            raise SystemExit(f"--pairs needs an even number of existing captures, have {sorted(caps)}")
        pairs = list(zip(named[::2], named[1::2]))
    if args.table is not None:
        table = [name(args.table)]
    if args.corners:
        corners = [name(n) for n in args.corners]
    if not pairs:
        raise SystemExit(f"no image / truth pairs in {args.folder}")
    frames = {i: load_frame(args.folder / f"{i}_frame.npz") for i, _ in pairs}

    def tcp_p(n):
        c = caps[n]
        return np.array(c["tcp"][:3]) if c.get("tcp") is not None else kin.tcp(c["q"])[0]

    keys = {"look_q": caps[pairs[0][0]]["q"]}
    if table:
        table_z = keys["table_z"] = float(np.mean([tcp_p(n)[2] for n in table]) - args.tip_offset)
        print(f"\ntable_z = {table_z:.4f} m (captures {table})")
    elif cfg.get("table_z") is not None:
        table_z = cfg["table_z"]
        print(f"\ntable_z = {table_z:.4f} m (config)")
    else:
        raise SystemExit("no table height: the config has no table_z and there is no table capture "
                         "(b in capture.py, or --table N)")
    if len(corners) == 1:
        print(f"only one corner capture ({corners[0]}), workspace not written")
    elif corners:
        c = np.array([tcp_p(n) for n in corners])
        keys["workspace"] = [c[:, 0].min(), c[:, 0].max(), c[:, 1].min(), c[:, 1].max()]
        print("workspace xmin xmax ymin ymax =", np.round(keys["workspace"], 4), f"(captures {corners})")

    hsv = cfg.get("hsv", RED)
    if not args.no_tune:
        print("\n2. mask tuning")
        tuned = tune([frames[i].rgb for i, _ in pairs], hsv)
        if tuned is None:
            print("not saved, keeping the config's hsv")
        else:
            hsv = keys["hsv"] = tuned
            print("hsv =", tuned)

    print("\n3. localisation error against the true-position captures (mm), raw: no loc_offset")
    K, half, top_pct = np.array(cfg["K"], float), cfg["half"], cfg.get("top_pct", 90)
    found, est, truth = [], [], []
    print("pair  image  truth    err x    err y    err z   |xy|  yaw (deg)")
    for k, (i, t) in enumerate(pairs):
        f = frames[i]
        q = np.array(caps[i]["q"])
        T_bc = kin.T_base_cam(q, cfg["T_tool0_cam"]) if cfg["mount"] == "wrist" else np.array(cfg["T_base_cam"])
        r = locate_cube(f.rgb, f.depth_m, K, T_bc, table_z, half, with_yaw=True, hsv=hsv, top_pct=top_pct)
        if r is None:
            print(f"{k:4d}  {i:>5s}  {t:>5s}  not found (mask empty or touching the image edge, or too few table points)")
            continue
        p, yaw = r
        found.append(k)
        est.append(p)
        truth.append(np.r_[tcp_p(t)[:2], table_z + half])
        e = 1e3 * (est[-1] - truth[-1])
        print(f"{k:4d}  {i:>5s}  {t:>5s}  {e[0]:7.1f}  {e[1]:7.1f}  {e[2]:7.1f}  {np.linalg.norm(e[:2]):5.1f}  "
              f"{np.rad2deg(yaw):9.1f}")
    if found:
        est, truth, found = np.array(est), np.array(truth), np.array(found)
        e = 1e3 * (est - truth)
        bias = e.mean(0)
        print(f"found {len(e)} of {len(pairs)}: {summary(e)}; "
              f"mean (bias) x {bias[0]:.1f} y {bias[1]:.1f} z {bias[2]:.1f} mm")
        print("yaw is mod 90 deg, in (-45, 45]: a turned object reading near 0 means the top face was missed")

        # an offset scored on the pairs it came from looks better than it is: fit on the even pairs, score the odd
        fit_on, test_on = found % 2 == 0, found % 2 == 1
        held = "held out: needs an even and an odd pair"
        if fit_on.any() and test_on.any():
            off = e[fit_on].mean(0)
            held = f"held out: |xy| mean {np.linalg.norm((e[test_on] - off)[:, :2], axis=1).mean():.1f} mm on the odd pairs"
            print(f"\noffset from the {fit_on.sum()} even pairs: {np.round(off, 1)} mm")
            print(f"  residual on those even pairs (in-sample):  {summary(e[fit_on] - off)}")
            print(f"  residual on the {test_on.sum()} odd pairs (held out):  {summary(e[test_on] - off)}")
        else:
            print("\nvalidation needs an even and an odd pair found, skipped")
        if args.set_offset:
            keys["loc_offset"] = np.round(bias / 1e3, 5)
            print(f"loc_offset = {keys['loc_offset']} m (mean bias over all {len(e)} pairs), subtracted by real_pick.py")
        if args.plot:
            plot(args.folder / "eval_objects.png", truth, est, bias / 1e3, held)
            print(f"plot {args.folder / 'eval_objects.png'}")
    else:
        print("found in none of the pairs")
    update_config(args.config, **keys)
