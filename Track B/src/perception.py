import numpy as np
import cv2
import mujoco

CV_TO_MJ = np.diag([1.0, -1.0, -1.0])


def project(K, P_c):
    p = K @ P_c
    return p[:2] / p[2]


def backproject(K, u, v, Z):
    x = (u - K[0, 2]) * Z / K[0, 0]
    y = (v - K[1, 2]) * Z / K[1, 1]
    return np.stack([x, y, Z], axis=-1)


def make_T(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def transform(T, P):
    return P @ T[:3, :3].T + T[:3, 3]


def T_wrist_cam(model, cam="wrist_cam"):
    c = model.camera(cam)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, c.quat)
    return make_T(R.reshape(3, 3) @ CV_TO_MJ, c.pos)

def T_base_wrist(data):
    base, wrist = data.body("base"), data.body("wrist_3_link")
    T_world_base = make_T(base.xmat.reshape(3, 3), base.xpos)
    T_world_wrist = make_T(wrist.xmat.reshape(3, 3), wrist.xpos)
    return np.linalg.inv(T_world_base) @ T_world_wrist


def red_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lo = cv2.inRange(hsv, (0, 150, 50), (10, 255, 255))
    hi = cv2.inRange(hsv, (170, 150, 50), (180, 255, 255))
    return (lo | hi) > 0


def locate_cube(rgb, depth, K, T_base_cam, half=0.02):
    mask = red_mask(rgb)
    v, u = np.nonzero(mask)
    if u.size < 20 or mask[[0, -1]].any() or mask[:, [0, -1]].any():
        return None
    P = transform(T_base_cam, backproject(K, u, v, depth[v, u]))
    top = P[P[:, 2] > P[:, 2].max() - 0.005]
    return top.mean(axis=0) - [0, 0, half]


def calibrate(images, pattern=(8, 6), square=0.02):
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square
    obj, img = [], []
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    for gray in images:
        ok, corners = cv2.findChessboardCorners(gray, pattern)
        if not ok:
            continue
        obj.append(objp)
        img.append(cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), crit))
    flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
    rms, K, _, _, _ = cv2.calibrateCamera(obj, img, images[0].shape[::-1], None, None, flags=flags)
    return K, rms, len(obj)