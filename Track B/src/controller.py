import numpy as np
from scipy.linalg import solve_discrete_are

from src.model import N, TAU_MIN, c2d, rot_error


def dlqr(A, B, Q, R):
    P = solve_discrete_are(A, B, Q, R)
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)


# Bryson: a 0.15 rad error, a 0.05 rad/s error rate or 1 rad s of integrated error each cost as much as the full
# 0.05 rad/s command, so a 0.1 rad step stays under u_max and the loop is far slower than 100 Hz and the lag
VEL_Q, VEL_R = (1 / 0.15**2, 1 / 0.05**2, 1.0), 1 / 0.05**2


def vel_lqr_design(dt, tau, Q=VEL_Q, R=VEL_R):
    # error state per joint [e, ed, z] on the speedJ lag model qd' = (u - qd) / tau. with feedforward
    # u_ff = qd_ref + tau qdd_ref the reference itself satisfies the model, so ed' = (du - ed) / tau with du = u - u_ff,
    # and perfect tracking needs du = 0
    A = np.array([[0, 1, 0], [0, -1 / tau, 0], [1, 0, 0]])
    Ad, Bd = c2d(A, np.array([[0], [1 / tau], [0]]), dt)
    return Ad, Bd, dlqr(Ad, Bd, np.diag(Q), np.array([[R]]))


class JointVelLQR:
    # u = u_ff - K [e, ed, z] - d, clipped to u_max and to a_max dt per step. the second is speedJ's own ramp, so the
    # joints never ramp on their own and the KF input is what they actually got. z freezes on clipped cycles
    # (anti-windup). tau per joint -> K per joint, 6 x 3
    def __init__(self, dt, tau, u_max, a_max=None, Q=VEL_Q, R=VEL_R, z_max=0.02):
        self.tau = np.maximum(np.broadcast_to(np.asarray(tau, float), N), TAU_MIN)
        self.K = np.array([vel_lqr_design(dt, t, Q, R)[2][0] for t in self.tau])
        self.dt, self.u_max, self.a_max, self.z_max = dt, u_max, a_max, z_max
        self.reset()

    def reset(self):
        self.z = np.zeros(N)
        self.u = np.zeros(N)

    def __call__(self, q, qd, q_ref, qd_ref, qdd_ref, d=0.0):
        e, ed = q - q_ref, qd - qd_ref
        u = qd_ref + self.tau * qdd_ref - self.K[:, 0] * e - self.K[:, 1] * ed - self.K[:, 2] * self.z - d
        out = np.clip(u, -self.u_max, self.u_max)
        if self.a_max:
            out = np.clip(out, self.u - self.a_max * self.dt, self.u + self.a_max * self.dt)
        clipped = out != u
        self.z = np.where(clipped, self.z, np.clip(self.z + self.dt * e, -self.z_max, self.z_max))
        self.u = out
        return out, clipped


def lqr_design(dt, Q=(1e6, 1111.0, 4e8), R=1.0):
    # error state per joint [e, ed, z], z = integral of e
    A = np.array([[1, dt, 0], [0, 1, 0], [dt, 0, 1]])
    B = np.array([[dt**2 / 2], [dt], [0]])
    K = dlqr(A, B, np.diag(Q), np.array([[R]]))
    return A, B, K


class JointLQR:
    def __init__(self, dt, Q=(1e6, 1111.0, 4e8), R=1.0, z_max=0.02):
        self.A, self.B, K = lqr_design(dt, Q, R)
        self.K = K[0]
        self.dt, self.z_max = dt, z_max
        self.z = np.zeros(N)

    def __call__(self, q, qd, q_ref, qd_ref, qdd_ref):
        e, ed = q - q_ref, qd - qd_ref
        v = qdd_ref - self.K[0] * e - self.K[1] * ed - self.K[2] * self.z
        self.z = np.clip(self.z + self.dt * e, -self.z_max, self.z_max)
        return v


class ComputedTorque:
    def __init__(self, robot, tau_max):
        self.robot, self.tau_max = robot, tau_max

    def __call__(self, q, qd, v):
        M, h = self.robot.dynamics(q, qd)
        tau = np.clip(M @ v + h, -self.tau_max, self.tau_max)
        # accel actually commanded once the torque is clipped, this is the KF input
        u = np.linalg.solve(M, tau - h)
        return tau, u


class PBVS:
    # v = lam * e on the pinch pose, capped and rate limited, then damped least squares to joint rates.
    # integrating qd gives the joint reference for the LQR
    def __init__(self, robot, dt, q0, lam=3.0, v_max=0.25, w_max=1.0, acc=1.0, alpha=3.0, damp=0.02, qd_max=None,
                 qdd_max=None):
        self.robot, self.dt = robot, dt
        self.lam, self.v_max, self.w_max, self.acc, self.alpha, self.damp = lam, v_max, w_max, acc, alpha, damp
        self.qd_max, self.qdd_max = qd_max, qdd_max
        self.reset(q0)

    def reset(self, q):
        self.q = np.array(q, float)
        self.qd = np.zeros(N)
        self.v = np.zeros(6)

    def __call__(self, p_des, R_des, v_max=None):
        p, R, J = self.robot.tcp(self.q)
        e = np.r_[p_des - p, rot_error(R_des, R)]
        v = self.lam * e
        v[:3] *= min(1, (v_max or self.v_max) / (np.linalg.norm(v[:3]) + 1e-9))
        v[3:] *= min(1, self.w_max / (np.linalg.norm(v[3:]) + 1e-9))
        for s, a in ((slice(0, 3), self.acc), (slice(3, 6), self.alpha)):
            dv = v[s] - self.v[s]
            self.v[s] += dv * min(1, a * self.dt / (np.linalg.norm(dv) + 1e-9))
        qd = J.T @ np.linalg.solve(J @ J.T + self.damp**2 * np.eye(6), self.v)
        if self.qd_max is not None:
            # joint speed cap (real arm): slow the whole twist so the path keeps its direction
            s = min(1, self.qd_max / (np.abs(qd).max() + 1e-12))
            qd, self.v = qd * s, self.v * s
        if self.qdd_max is not None:
            # and joint acceleration cap, so the joint loop's rate limit can follow the reference
            dq = qd - self.qd
            qd = self.qd + dq * min(1, self.qdd_max * self.dt / (np.abs(dq).max() + 1e-12))
            self.v = J @ qd
        qdd = (qd - self.qd) / self.dt
        q = self.q
        self.q = q + qd * self.dt
        self.qd = qd
        return q, qd, qdd, e


def grasp_R(yaw):
    # gripper z down, fingers close along a cube face normal (the one nearest the home orientation)
    y = np.array([-np.sin(yaw), np.cos(yaw), 0])
    z = np.array([0, 0, -1.0])
    return np.column_stack([np.cross(y, z), y, z])


PHASES = ["look", "approach", "refine", "descend", "grasp", "lift", "transfer", "lower", "release", "retreat",
          "home", "done", "fail"]


class PickTask:
    def __init__(self, p_home, R_home, drop, table_z, half, n_look=8, pre=0.10, lift=0.15):
        self.p_home, self.R_home, self.drop = p_home, R_home, drop
        self.table_z, self.half, self.n_look, self.pre, self.lift = table_z, half, n_look, pre, lift
        self.phase, self.t0 = "look", 0.0
        self.p_grasp = None

    def go(self, phase, t):
        self.phase, self.t0 = phase, t

    @property
    def camera_on(self):
        return self.phase in ("look", "approach", "refine", "descend")

    def __call__(self, t, at, cube, yaw, n_meas):
        el = t - self.t0
        # pbvs error is a step behind, give it time to see the new target
        at = at and el > 0.1
        ph = self.phase
        R = grasp_R(yaw)
        if ph == "look":
            if n_meas >= self.n_look:
                self.go("approach", t)
            elif el > 3.0:
                self.go("fail", t)
            return self.p_home, self.R_home, None, 0
        if ph in ("approach", "refine"):
            if ph == "approach" and at and el > 0.3:
                self.go("refine", t)
            if ph == "refine" and el > 0.5:
                self.go("descend", t)
            return cube + [0, 0, self.pre], R, None, 0
        if ph == "descend":
            if at:
                self.p_grasp = cube.copy()
                self.go("grasp", t)
            return cube, R, 0.08, 0
        if ph == "grasp":
            if el > 0.8:
                self.go("lift", t)
            return self.p_grasp, R, None, 255
        if ph == "lift":
            if at:
                self.go("transfer", t)
            return self.p_grasp + [0, 0, self.lift], R, 0.15, 255
        place = np.r_[self.drop, self.table_z + self.half + 0.005]
        if ph == "transfer":
            if at:
                self.go("lower", t)
            return place + [0, 0, self.lift], R, None, 255
        if ph == "lower":
            if at:
                self.go("release", t)
            return place, R, 0.08, 255
        if ph == "release":
            if el > 0.6:
                self.go("retreat", t)
            return place, R, None, 0
        if ph == "retreat":
            if at:
                self.go("home", t)
            return place + [0, 0, 0.10], R, None, 0
        if ph == "home":
            if at:
                self.go("done", t)
            return self.p_home, self.R_home, None, 0
        return self.p_home, self.R_home, None, 0
