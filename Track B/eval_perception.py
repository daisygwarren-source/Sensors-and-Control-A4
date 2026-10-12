import argparse
import numpy as np
import mujoco

from main import W, H, K_CALIB, make_trial, grab_rgbd, read_encoders, cube_truth_eval_only
from src.perception import T_base_cam, locate_cube, d435_noise

ap = argparse.ArgumentParser()
ap.add_argument("--trials", type=int, default=50)
args = ap.parse_args()

K = np.load(K_CALIB)
rng = np.random.default_rng(0)
scenes = []
for _ in range(args.trials):
    model, data = make_trial(rng)
    for _ in range(500):
        mujoco.mj_step(model, data)
    renderer = mujoco.Renderer(model, height=H, width=W)
    scenes.append((*grab_rgbd(renderer, data), read_encoders(data, rng), cube_truth_eval_only(data)))
    renderer.close()

print("noise | detected | mean mm | max mm | std x/y/z mm")
for scale in [0, 1, 3, 10]:
    errs = []
    for rgb, depth, q_enc, truth in scenes:
        if scale:
            rgb, depth = d435_noise(rgb, depth, rng, subpixel=0.08 * scale, wave=0.002 * scale)
        est = locate_cube(rgb, depth, K, T_base_cam(q_enc))
        if est is not None:
            errs.append(est - truth)
    e = 1000 * np.array(errs)
    n = np.linalg.norm(e, axis=1)
    print(f"{scale:4}x | {len(e):3}/{len(scenes)}  | {n.mean():7.2f} | {n.max():6.2f} | {e.std(axis=0).round(2)}")
