import argparse
import numpy as np
import mujoco

from simulator import build, ARM_JOINTS
from main import W, H, K_CALIB, make_trial, grab_rgbd, read_encoders, cube_truth_eval_only
from src.perception import fk, T_base_cam, locate_cube, d435_noise

ap = argparse.ArgumentParser()
ap.add_argument("--trials", type=int, default=50)
ap.add_argument("--study", action="store_true")
args = ap.parse_args()
rng = np.random.default_rng(0)


def in_base_eval_only(data, R, p):
    b = data.body("base")
    Rb = b.xmat.reshape(3, 3)
    T = np.eye(4)
    T[:3, :3] = Rb.T @ R
    T[:3, 3] = Rb.T @ (p - b.xpos)
    return T


model = build().compile()
data = mujoco.MjData(model)
cam = model.camera("wrist_cam").id
fk_err, cam_err = 0.0, 0.0
for _ in range(200):
    q = rng.uniform(-np.pi, np.pi, 6)
    data.qpos[:6] = q
    mujoco.mj_forward(model, data)
    wrist = data.body("wrist_3_link")
    fk_err = max(fk_err, np.abs(fk(q) - in_base_eval_only(data, wrist.xmat.reshape(3, 3), wrist.xpos)).max())
    T_cam = in_base_eval_only(data, data.cam_xmat[cam].reshape(3, 3) @ np.diag([1.0, -1, -1]), data.cam_xpos[cam])
    cam_err = max(cam_err, np.abs(T_base_cam(q) - T_cam).max())
print(f"FK vs MuJoCo, 200 random configurations: wrist pose max abs diff {fk_err:.1e}, camera pose {cam_err:.1e}")

K = np.load(K_CALIB)
scenes, enc_err = [], {14: 0.0, 16: 0.0, 20: 0.0}
for _ in range(args.trials):
    model, data = make_trial(rng)
    for _ in range(500):
        mujoco.mj_step(model, data)
    renderer = mujoco.Renderer(model, height=H, width=W)
    scenes.append((*grab_rgbd(renderer, data), read_encoders(data, rng), cube_truth_eval_only(data)))
    renderer.close()
    q_exact = np.array([data.sensor(f"enc_{j}").data[0] for j in ARM_JOINTS])
    for bits in enc_err:
        dp = T_base_cam(read_encoders(data, rng, bits))[:3, 3] - T_base_cam(q_exact)[:3, 3]
        enc_err[bits] = max(enc_err[bits], np.abs(dp).max())
print("encoder resolution vs worst camera position error:",
      ", ".join(f"{bits} bit {1000 * e:.3f} mm" for bits, e in enc_err.items()))

noisy = {s: [d435_noise(rgb, depth, rng, subpixel=0.08 * s, wave=0.002 * s) if s else (rgb, depth)
             for rgb, depth, _, _ in scenes] for s in [0, 1, 3, 10]}


def evaluate(s, **kw):
    errs, radial = [], []
    for (rgb, depth), (_, _, q_enc, truth) in zip(noisy[s], scenes):
        T = T_base_cam(q_enc)
        est = locate_cube(rgb, depth, K, T, **kw)
        if est is not None:
            e = est - truth
            off = truth[:2] - T[:2, 3]
            errs.append(e)
            radial.append(e[:2] @ off / np.linalg.norm(off))
    return 1000 * np.array(errs), 1000 * np.array(radial)


print("\nnoise | detected | mean mm | max mm | std x/y/z mm")
for s in noisy:
    e, _ = evaluate(s)
    n = np.linalg.norm(e, axis=1)
    print(f"{s:4}x | {len(e):3}/{len(scenes)}  | {n.mean():7.2f} | {n.max():6.2f} | {e.std(axis=0).round(2)}")

if args.study:
    print("\nerode | band  | mean/max mm at 0x | 1x | 3x | 10x noise | bias toward camera at 0x")
    for erode in [1, 3, 5]:
        for band in [0.002, 0.003, 0.005, 0.007, 0.010, 0.015, 0.020]:
            cells = []
            for s in noisy:
                e, radial = evaluate(s, band=band, erode=erode)
                n = np.linalg.norm(e, axis=1)
                cells.append(f"{n.mean():.2f}/{n.max():.2f}")
                if s == 0:
                    toward = -radial.mean()
            print(f"{erode}x{erode}   | {1000 * band:2.0f} mm | {' | '.join(cells)} | {toward:.2f} mm")
