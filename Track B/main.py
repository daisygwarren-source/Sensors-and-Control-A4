"""Run the scene: interactive viewer + wrist RGB-D capture.

    python run_sim.py            # opens viewer, robot holds home pose
    python run_sim.py --headless # no window; saves wrist RGB/depth images
"""
import argparse
import time
import numpy as np
import mujoco

import sys
from pathlib import Path


from simulator import build, TABLE_POS, ARM_JOINTS
from src.perception import T_base_cam, locate_cube, d435_noise

HOME = np.array([2.817, -1.788, 1.681, -1.463, -1.571, -0.325])  # look pose: wrist cam ~0.51 m up, centred over the spawn area, looking straight down
W, H = 640, 480
K_CALIB = Path(__file__).resolve().parent / "K_calib.npy"


def make_trial(rng):
    """New model with the target at a random tabletop position (spec requirement)."""
    xy = TABLE_POS[:2] + rng.uniform([-0.15, -0.25], [0.15, 0.25])
    model = build(object_xy=xy).compile()
    data = mujoco.MjData(model)
    data.qpos[:6] = HOME
    data.ctrl[:6] = HOME
    data.ctrl[6] = 0          # gripper open (0..255)
    mujoco.mj_forward(model, data)
    return model, data


def grab_rgbd(renderer, data, cam="wrist_cam"):
    renderer.update_scene(data, camera=cam)
    rgb = renderer.render()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=cam)
    depth = renderer.render()          # metres along camera z
    renderer.disable_depth_rendering()
    return rgb, depth


def read_encoders(data, rng, bits=20):
    lsb = 2 * np.pi / 2 ** bits
    q = np.array([data.sensor(f"enc_{j}").data[0] for j in ARM_JOINTS])
    return np.round((q + lsb * rng.standard_normal(len(q))) / lsb) * lsb


def intrinsics(model, cam="wrist_cam"):
    """Pinhole K from MuJoCo fovy - use as ground-truth to compare against your calibration."""
    fovy = np.deg2rad(model.camera(cam).fovy[0])
    f = 0.5 * H / np.tan(fovy / 2)
    return np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])


def cube_truth_eval_only(data):
    b = data.body("base")
    return b.xmat.reshape(3, 3).T @ (data.body("target").xpos - b.xpos)


def controller(model, data):
    """PLACEHOLDER: replace with your state-space joint controller / IBVS / PBVS.
    Must act on *estimated* state, not data.qpos ground truth."""
    data.ctrl[:6] = HOME


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    model, data = make_trial(rng)
    renderer = mujoco.Renderer(model, height=H, width=W)
    print("K =\n", intrinsics(model).round(1))

    if args.headless:
        for _ in range(500):
            controller(model, data)
            mujoco.mj_step(model, data)
        rgb, depth = d435_noise(*grab_rgbd(renderer, data), rng)
        q_enc = read_encoders(data, rng)
        cube = locate_cube(rgb, depth, np.load(K_CALIB), T_base_cam(q_enc))
        print("cube estimate (base frame):", None if cube is None else cube.round(4))
        print("cube ground truth (evaluation only):", cube_truth_eval_only(data).round(4))
        renderer.update_scene(data, camera="overview")
        overview = renderer.render()
        import imageio.v3 as iio
        iio.imwrite("wrist_rgb.png", rgb)
        iio.imwrite("overview.png", overview)
        iio.imwrite("wrist_depth.png", (255 * np.clip(depth / 1.5, 0, 1)).astype(np.uint8))
    else:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                t0 = time.time()
                controller(model, data)
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(max(0, model.opt.timestep - (time.time() - t0)))
