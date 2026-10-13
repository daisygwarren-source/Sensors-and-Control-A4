import numpy as np
import mujoco
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

N = 6


def joint_model():
    # one joint after computed torque: q'' = v, encoder gives q
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    C = np.array([[1.0, 0.0]])
    return A, B, C


def arm_model(n=N):
    # x = [q, qd] for all joints
    A1, B1, C1 = joint_model()
    return np.kron(A1, np.eye(n)), np.kron(B1, np.eye(n)), np.kron(C1, np.eye(n))


def c2d(A, B, dt):
    n, m = B.shape
    E = expm(np.block([[A, B], [np.zeros((m, n + m))]]) * dt)
    return E[:n, :n], E[:n, n:]


def ctrb(A, B):
    return np.hstack([np.linalg.matrix_power(A, k) @ B for k in range(A.shape[0])])


def obsv(A, C):
    return np.vstack([C @ np.linalg.matrix_power(A, k) for k in range(A.shape[0])])


def gramians(Ad, Bd, Cd, steps):
    Wc = sum(np.linalg.matrix_power(Ad, k) @ Bd @ Bd.T @ np.linalg.matrix_power(Ad, k).T for k in range(steps))
    Wo = sum(np.linalg.matrix_power(Ad, k).T @ Cd.T @ Cd @ np.linalg.matrix_power(Ad, k) for k in range(steps))
    return Wc, Wo


def pbh(A, B, C):
    n = A.shape[0]
    out = []
    for lam in np.unique(np.round(np.linalg.eigvals(A), 9)):
        rc = np.linalg.matrix_rank(np.hstack([lam * np.eye(n) - A, B]))
        ro = np.linalg.matrix_rank(np.vstack([lam * np.eye(n) - A, C]))
        out.append((lam, rc, ro))
    return out


def rot_error(R_des, R):
    return Rotation.from_matrix(R_des @ R.T).as_rotvec()


class RobotModel:
    # controller's own copy of the arm, compiled separately from the sim, so it can't see the true state.
    # only ever evaluated at the estimated joint angles. outputs are in the base frame
    def __init__(self, model, site="gripper/pinch", mass_scale=1.0):
        self.m = model
        self.m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        if mass_scale != 1.0:
            arm = [i for i in range(model.nbody) if model.body_rootid[i] == model.body("base").rootid[0]]
            self.m.body_mass[arm] *= mass_scale
            self.m.body_inertia[arm] *= mass_scale
        self.d = mujoco.MjData(self.m)
        self.site = self.m.site(site).id
        mujoco.mj_forward(self.m, self.d)
        self.R_bw = self.d.body("base").xmat.reshape(3, 3).T.copy()
        self.p_wb = self.d.body("base").xpos.copy()
        self.M = np.zeros((self.m.nv, self.m.nv))
        self.Jp = np.zeros((3, self.m.nv))
        self.Jr = np.zeros((3, self.m.nv))

    def _set(self, q, qd=None):
        self.d.qpos[:N] = q
        self.d.qvel[:N] = 0.0 if qd is None else qd
        mujoco.mj_forward(self.m, self.d)

    def dynamics(self, q, qd):
        self._set(q, qd)
        mujoco.mj_fullM(self.m, self.d, self.M)
        return self.M[:N, :N].copy(), self.d.qfrc_bias[:N].copy()

    def tcp(self, q):
        self._set(q)
        p = self.R_bw @ (self.d.site_xpos[self.site] - self.p_wb)
        R = self.R_bw @ self.d.site_xmat[self.site].reshape(3, 3)
        mujoco.mj_jacSite(self.m, self.d, self.Jp, self.Jr, self.site)
        J = np.vstack([self.R_bw @ self.Jp[:, :N], self.R_bw @ self.Jr[:, :N]])
        return p, R, J

    def to_world(self, p):
        return self.R_bw.T @ p + self.p_wb
