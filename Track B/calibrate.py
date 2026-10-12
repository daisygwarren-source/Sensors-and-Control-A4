import argparse
import numpy as np
import cv2
import mujoco

from simulator import build
from main import HOME, W, H, K_CALIB, intrinsics, grab_rgbd
from src.perception import calibrate, d435_noise

SQ, COLS, ROWS = 0.02, 9, 7
BOX = mujoco.mjtGeom.mjGEOM_BOX


def calib_model():
    spec = build()
    board = spec.worldbody.add_body(name="board", mocap=True)
    board.add_geom(type=BOX, size=[(COLS + 2) * SQ / 2, (ROWS + 2) * SQ / 2, 0.001],
                   rgba=[1, 1, 1, 1], contype=0, conaffinity=0)
    for i in range(ROWS):
        for j in range(COLS):
            if (i + j) % 2 == 0:
                board.add_geom(type=BOX, size=[SQ / 2, SQ / 2, 0.001],
                               pos=[(j - (COLS - 1) / 2) * SQ, (i - (ROWS - 1) / 2) * SQ, 0.001],
                               rgba=[0, 0, 0, 1], contype=0, conaffinity=0)
    return spec.compile()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--free-distortion", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    model = calib_model()
    data = mujoco.MjData(model)
    data.qpos[:6] = HOME
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=H, width=W)

    # Scene setup only: the camera's true pose is used to place the checkerboard in front of it,
    # like a person holding the board up. calibrate() below sees only the rendered images.
    cam = model.camera("wrist_cam").id
    cam_pos, cam_R = data.cam_xpos[cam].copy(), data.cam_xmat[cam].reshape(3, 3).copy()
    images = []
    for _ in range(40):
        offset = [*rng.uniform(-0.03, 0.03, 2), -rng.uniform(0.22, 0.32)]
        R = cam_R @ cv2.Rodrigues(rng.uniform([-0.5, -0.5, -0.3], [0.5, 0.5, 0.3]))[0]
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R.flatten())
        data.mocap_pos[0] = cam_pos + cam_R @ offset
        data.mocap_quat[0] = q
        mujoco.mj_forward(model, data)
        rgb, _ = d435_noise(*grab_rgbd(renderer, data), rng)
        images.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))

    K, dist, rms, n = calibrate(images, (COLS - 1, ROWS - 1), SQ, fix_distortion=not args.free_distortion)
    print(f"used {n}/{len(images)} images, reprojection RMS {rms:.3f} px")
    print("K calibrated =\n", K.round(2))
    print("K true (evaluation only) =\n", intrinsics(model).round(2))
    print("distortion =", dist.ravel().round(4))
    if not args.free_distortion:
        np.save(K_CALIB, K)
