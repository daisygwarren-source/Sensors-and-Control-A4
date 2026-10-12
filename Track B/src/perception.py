import numpy as np
import cv2

LINKS = [((0, 0, 0.163), "z", 0.0), ((0, 0.138, 0), "y", np.pi / 2), ((0, -0.131, 0.425), "y", 0.0),
         ((0, 0, 0.392), "y", np.pi / 2), ((0, 0.127, 0), "z", 0.0), ((0, 0, 0.1), "y", 0.0)]
T_WRIST_CAM = np.array([[1, 0, 0, 0], [0, 0, 1, 0.10], [0, -1, 0, 0.08], [0, 0, 0, 1.0]])


def trans(x, y, z):
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


def rot(axis, a):
    c, s = np.cos(a), np.sin(a)
    T = np.eye(4)
    if axis == "y":
        T[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    else:
        T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return T


def fk(q):
    T = np.eye(4)
    for (p, axis, offset), qi in zip(LINKS, q):
        T = T @ trans(*p) @ rot(axis, qi + offset)
    return T


def T_base_cam(q):
    return fk(q) @ T_WRIST_CAM


def project(K, P_c):
    p = K @ P_c
    return p[:2] / p[2]


def backproject(K, u, v, Z):
    x = (u - K[0, 2]) * Z / K[0, 0]
    y = (v - K[1, 2]) * Z / K[1, 1]
    return np.stack([x, y, Z], axis=-1)


def transform(T, P):
    return P @ T[:3, :3].T + T[:3, 3]


def d435_noise(rgb, depth, rng, subpixel=0.08, wave=0.002, f_ir=424.0, baseline=0.05, edge_drop=0.5, rgb_sigma=3.0):
    h, w = depth.shape
    d = depth + depth ** 2 * subpixel / (f_ir * baseline) * rng.standard_normal((h, w))
    v, u = np.mgrid[0:h, 0:w]
    d += wave * np.sin(2 * np.pi * u / w) * np.sin(2 * np.pi * v / h)
    edges = cv2.dilate((np.abs(cv2.Laplacian(depth, cv2.CV_32F)) > 0.01).astype(np.uint8), np.ones((3, 3), np.uint8))
    d[(edges > 0) & (rng.random((h, w)) < edge_drop)] = 0
    rgb = np.clip(rgb + rng.normal(0, rgb_sigma, rgb.shape), 0, 255).astype(np.uint8)
    return rgb, np.round(d, 3)


def red_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lo = cv2.inRange(hsv, (0, 150, 50), (10, 255, 255))
    hi = cv2.inRange(hsv, (170, 150, 50), (180, 255, 255))
    return lo | hi


def cube_mask(rgb):
    m = cv2.morphologyEx(red_mask(rgb), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    return labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])


def points(sel, depth, K, T_bc):
    v, u = np.nonzero(sel)
    return transform(T_bc, backproject(K, u, v, depth[v, u]))


def locate_cube(rgb, depth, K, T_bc, table_z=0.04, half=0.02, band=0.01):
    mask = cube_mask(rgb)
    if mask is None or mask[[0, -1]].any() or mask[:, [0, -1]].any():
        return None
    m = mask.astype(np.uint8)
    inner = cv2.erode(m, np.ones((5, 5), np.uint8)) > 0
    ring = (cv2.dilate(m, np.ones((31, 31), np.uint8)) > 0) & ~(cv2.dilate(m, np.ones((11, 11), np.uint8)) > 0)
    table = points(ring & (depth > 0), depth, K, T_bc)
    if len(table) < 20:
        return None
    bias = np.median(table[:, 2]) - table_z
    cube = points(inner & (depth > 0), depth - bias / T_bc[2, 2], K, T_bc)
    if len(cube) < 20:
        return None
    z = np.median(cube[:, 2])
    for _ in range(3):
        top = cube[np.abs(cube[:, 2] - z) < band]
        z = np.median(top[:, 2])
    x, y = np.median(top[:, :2], axis=0)
    return np.array([x, y, z - half])


def calibrate(images, pattern=(8, 6), square=0.02, fix_distortion=True):
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square
    obj, img = [], []
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    for gray in images:
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        ok, corners = cv2.findChessboardCorners(gray, pattern)
        if ok:
            obj.append(objp)
            img.append(cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), crit))
    if len(obj) < 3:
        raise ValueError(f"checkerboard found in {len(obj)} of {len(images)} images, need at least 3")
    flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 if fix_distortion else 0
    size = images[0].shape[::-1]
    err = cv2.calibrateCameraExtended(obj, img, size, None, None, flags=flags)[7].ravel()
    keep = err < 2 * np.median(err)
    obj = [o for o, k in zip(obj, keep) if k]
    img = [i for i, k in zip(img, keep) if k]
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, img, size, None, None, flags=flags)
    return K, dist, rms, len(obj)
