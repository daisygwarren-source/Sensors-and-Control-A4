"""Lunch: where the camera sits relative to the robot, from capture.py --label handeye (ArUco DICT_4X4_50 id 0).
This OpenCV has no calibrateHandEye, so: closed-form start (Park-Martin on the per-capture solvePnP poses), then
least squares on the marker corners' reprojection error over all captures. Writes T_tool0_cam (wrist camera) or
T_base_cam (fixed camera) and mount to the config.

    python calib_handeye.py FOLDER --config ...\\config.json --mount wrist
    python calib_handeye.py --selftest
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from real import SIM_PLACEHOLDER, UR3eKinematics, find_marker, load_config, marker_detector, update_config
from src.perception import project

inv = np.linalg.inv


def corners_3d(s):
    # marker frame, ArUco's corner order: top-left, top-right, bottom-right, bottom-left
    return np.array([[-s / 2, s / 2, 0], [s / 2, s / 2, 0], [s / 2, -s / 2, 0], [-s / 2, -s / 2, 0]])


def to_T(p):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = Rotation.from_rotvec(p[:3]).as_matrix(), p[3:]
    return T


def to_p(T):
    return np.r_[Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]]


def cam_marker(A, X, Y, mount):
    # wrist: X = T_tool0_cam, Y = T_base_marker. fixed: X = T_base_cam, Y = T_tool0_marker
    return inv(A @ X) @ Y if mount == "wrist" else inv(X) @ A @ Y


def residuals(p, A, uv, K, P, mount):
    X, Y = to_T(p[:6]), to_T(p[6:])
    r = []
    for Ai, uvi in zip(A, uv):
        T = cam_marker(Ai, X, Y, mount)
        r.append(project(K, T[:3, :3] @ P.T + T[:3, 3:]).T - uvi)
    return np.concatenate(r).ravel()


def pnp(uv, K, P):
    ok, rvec, tvec = cv2.solvePnP(P, uv, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return to_T(np.r_[rvec.ravel(), tvec.ravel()])


def solve_ax_xb(pairs):
    # Park & Martin: A Z = Z B. rotation: log maps satisfy a = R b, Kabsch fit. translation: (R_A - I) t = R t_B - t_A
    a = np.array([Rotation.from_matrix(A[:3, :3]).as_rotvec() for A, _ in pairs])
    b = np.array([Rotation.from_matrix(B[:3, :3]).as_rotvec() for _, B in pairs])
    U, _, Vt = np.linalg.svd(b.T @ a)
    R = Vt.T @ np.diag([1, 1, np.linalg.det(Vt.T @ U.T)]) @ U.T
    M = np.vstack([A[:3, :3] - np.eye(3) for A, _ in pairs])
    v = np.concatenate([R @ B[:3, 3] - A[:3, 3] for A, B in pairs])
    Z = np.eye(4)
    Z[:3, :3], Z[:3, 3] = R, np.linalg.lstsq(M, v, rcond=None)[0]
    return Z


def initial_X(A, B, mount):
    idx = list(combinations(range(len(A)), 2))
    if mount == "wrist":
        # A_i X B_i = Y for every i -> (A_j^-1 A_i) X = X (B_j B_i^-1)
        return solve_ax_xb([(inv(A[j]) @ A[i], B[j] @ inv(B[i])) for i, j in idx])
    # X B_i = A_i Y -> (A_j^-1 A_i) Y = Y (B_j^-1 B_i), then X = A_0 Y B_0^-1
    Y = solve_ax_xb([(inv(A[j]) @ A[i], inv(B[j]) @ B[i]) for i, j in idx])
    return A[0] @ Y @ inv(B[0])


def fit(A, uv, K, s, mount, X0=None):
    P = corners_3d(s)
    B = [pnp(u, K, P) for u in uv]
    if X0 is None:
        X0 = initial_X(A, B, mount)
    Y0 = A[0] @ X0 @ B[0] if mount == "wrist" else inv(A[0]) @ X0 @ B[0]
    sol = least_squares(residuals, np.r_[to_p(X0), to_p(Y0)], args=(A, uv, K, P, mount), x_scale="jac")
    r = sol.fun.reshape(len(A), 4, 2)
    per = np.sqrt((r**2).sum(-1).mean(-1))
    return to_T(sol.x[:6]), to_T(sol.x[6:]), np.sqrt((r**2).sum(-1).mean()), per, B


def spread(A, X, B, mount):
    # the marker pose each capture implies on its own given X: it should not move
    Ys = [Ai @ X @ Bi if mount == "wrist" else inv(Ai) @ X @ Bi for Ai, Bi in zip(A, B)]
    p = np.array([Y[:3, 3] for Y in Ys])
    R0 = Rotation.from_matrix(np.array([Y[:3, :3] for Y in Ys])).mean()
    ang = [np.linalg.norm((Rotation.from_matrix(Y[:3, :3]) * R0.inv()).as_rotvec()) for Y in Ys]
    d = np.linalg.norm(p - p.mean(0), axis=1)
    return 1e3 * np.sqrt((d**2).mean()), 1e3 * d.max(), np.rad2deg(np.sqrt(np.mean(np.square(ang)))), np.rad2deg(max(ang))


def pose_diff(T1, T2):
    return 1e3 * np.linalg.norm(T1[:3, 3] - T2[:3, 3]), np.rad2deg(np.linalg.norm(to_p(T1 @ inv(T2))[:3]))


def selftest():
    rng = np.random.default_rng(0)
    kin = UR3eKinematics()
    K, s, look = np.array(SIM_PLACEHOLDER["K"]), 0.1, np.array(SIM_PLACEHOLDER["look_q"])
    flip = np.diag([1.0, -1, -1, 1])  # marker z towards the camera
    A0 = kin.T_base_tool0(look)
    cases = {
        # wrist camera, marker on the table 0.35 m in front of it
        "wrist": (to_T(np.r_[0.05, -0.03, 1.2, 0.01, 0.07, 0.03]),
                  lambda X: A0 @ X @ to_T(np.r_[0.1, -0.05, 0.2, 0.02, -0.01, 0.35]) @ flip),
        # fixed camera 0.45 m above the look pose looking down, marker on the flange facing up. accuracy falls with
        # distance: at 0.6 m the same 15 poses give ~1.1 mm
        "fixed": (A0 @ to_T(np.r_[0.08, 0.05, -0.1, 0.03, -0.02, -0.45]),
                  lambda X: to_T(np.r_[0.0, 0.0, 0.0, 0.0, 0.03, -0.01]) @ flip),
    }
    ok = True
    for mount, (X, make_Y) in cases.items():
        Y, P = make_Y(X), corners_3d(s)
        A, uv = [], []
        for _ in range(5000):
            if len(A) == 15:
                break
            Ai = kin.T_base_tool0(look + rng.uniform(-1, 1, 6) * [0.15, 0.1, 0.1, 0.3, 0.3, 0.6])
            T = cam_marker(Ai, X, Y, mount)
            Pc = T[:3, :3] @ P.T + T[:3, 3:]
            u = project(K, Pc).T
            if (Pc[2] > 0.2).all() and (u > 10).all() and (u[:, 0] < 630).all() and (u[:, 1] < 470).all():
                A.append(Ai)
                uv.append(u + rng.normal(0, 0.3, u.shape))
        if len(A) < 15:
            raise SystemExit(f"selftest: only {len(A)} poses keep the marker in view ({mount})")
        X_bad = X @ to_T(np.r_[Rotation.from_euler("xyz", [6, -5, 8], degrees=True).as_rotvec(), 0.02, -0.015, 0.01])
        for name, X0 in (("closed-form start", None), ("start 2 cm / 10 deg off", X_bad)):
            Xf, _, rms, _, _ = fit(A, uv, K, s, mount, X0)
            dp, dr = pose_diff(Xf, X)
            good = dp < 1.0 and dr < 0.1
            ok &= good
            print(f"{mount:5s} {name:24s}: X error {dp:.3f} mm, {dr:.4f} deg, RMS {rms:.3f} px "
                  f"-> {'pass' if good else 'FAIL'} (need < 1 mm, < 0.1 deg)")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, nargs="?")
    ap.add_argument("--config", type=Path)
    ap.add_argument("--mount", choices=["wrist", "fixed"])
    ap.add_argument("--marker", type=float, default=0.100, help="marker size (outer black edge), m")
    ap.add_argument("--x0", type=float, nargs=6, metavar=("RX", "RY", "RZ", "X", "Y", "Z"),
                    help="start for X (rotation vector, translation) instead of the closed-form one")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    if not (args.folder and args.config and args.mount):
        ap.error("FOLDER, --config and --mount are required (or --selftest)")

    cfg = load_config(args.config)
    K = np.array(cfg["K"], float)
    kin = UR3eKinematics()
    detector = marker_detector()
    names, A, uv = [], [], []
    for png in sorted(args.folder.glob("*_rgb.png")):
        pose = png.with_name(png.name.replace("_rgb.png", "_pose.json"))
        if not pose.exists():
            print(f"{png.name}: no pose file, skipped")
            continue
        hit = find_marker(detector, cv2.imread(str(png), cv2.IMREAD_GRAYSCALE))
        if hit is None:
            print(f"{png.name}: marker 0 not found, skipped")
            continue
        names.append(png.name[:3])
        A.append(kin.T_base_tool0(np.array(load_config(pose)["q"])))
        uv.append(hit)
    if len(A) < 3:
        raise SystemExit(f"marker found in {len(A)} captures, need at least 3")
    turn = max(np.rad2deg(np.linalg.norm(to_p(inv(Ai) @ Aj)[:3])) for Ai in A for Aj in A)
    if turn < 15:
        print(f"WARNING: the tool turned at most {turn:.0f} deg between captures, too little to pin the rotation down")

    X0 = to_T(np.array(args.x0)) if args.x0 else None
    keep = np.ones(len(A), bool)
    for attempt in ("all captures", "refit"):
        X, Y, rms, per, B = fit([a for a, k in zip(A, keep) if k], [u for u, k in zip(uv, keep) if k], K,
                                args.marker, args.mount, X0)
        used = [n for n, k in zip(names, keep) if k]
        print(f"\n{attempt}: {len(used)} captures, reprojection RMS {rms:.3f} px")
        print("  per capture (px): " + ", ".join(f"{n} {e:.2f}" for n, e in zip(used, per)))
        bad = per > 3 * np.median(per)
        if attempt == "refit" or not bad.any():
            break
        print(f"  dropping {[n for n, b in zip(used, bad) if b]} (above 3x the median) and refitting")
        keep[np.flatnonzero(keep)[bad]] = False
    dp, dpm, dr, drm = spread([a for a, k in zip(A, keep) if k], X, B, args.mount)
    print(f"marker pose implied by each capture: spread RMS {dp:.2f} mm (max {dpm:.2f}), {dr:.3f} deg (max {drm:.3f})")
    name = "T_tool0_cam" if args.mount == "wrist" else "T_base_cam"
    print(f"{name} =\n{np.round(X, 4)}")
    update_config(args.config, **{name: X, "mount": args.mount, "handeye_rms_px": rms})
