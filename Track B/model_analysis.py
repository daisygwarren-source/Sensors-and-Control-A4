"""State-space model of the arm, controllability / observability, and a check of the model against MuJoCo.

    python model_analysis.py
"""
import numpy as np
import mujoco
from scipy.linalg import solve_discrete_are

from simulator import build
from main import HOME, CAM_DT
from src.model import N, joint_model, arm_model, c2d, ctrb, obsv, gramians, pbh, RobotModel
from src.controller import lqr_design
from src.estimation import JointKF, KalmanFilter3D

np.set_printoptions(precision=4, suppress=True, linewidth=140)
dt = 0.002

print("A1. one joint under computed torque, x = [q, qd], u = v (commanded accel), y = q")
A1, B1, C1 = joint_model()
Ad1, Bd1 = c2d(A1, B1, dt)
print("A =\n", A1, "\nB =", B1.ravel(), " C =", C1.ravel())
print("ZOH, dt = 2 ms:\nAd =\n", Ad1, "\nBd =", Bd1.ravel())

print("\nA2. all six joints (12 states, 6 inputs, 6 encoder outputs)")
A, B, C = arm_model()
Ad, Bd = c2d(A, B, dt)
print("rank ctrb =", np.linalg.matrix_rank(ctrb(A, B)), "of", A.shape[0],
      "  rank obsv (encoders, C = [I 0]) =", np.linalg.matrix_rank(obsv(A, C)), "of", A.shape[0])
for lam, rc, ro in pbh(A, B, C):
    print(f"PBH at eigenvalue {lam.real:g}: rank[lI-A, B] = {rc}, rank[lI-A; C] = {ro}")
Cv = np.hstack([np.zeros((N, N)), np.eye(N)])
print("if we only had joint velocity (C = [0 I]): rank obsv =", np.linalg.matrix_rank(obsv(A, Cv)),
      "-> absolute angles unobservable, an offset in q never shows in qd")
Wc, Wo = gramians(Ad, Bd, np.hstack([np.eye(N), np.zeros((N, N))]), int(1.0 / dt))
for name, Wm in [("controllability", Wc), ("observability", Wo)]:
    ev = np.linalg.eigvalsh(Wm)
    print(f"{name} Gramian over 1 s: eig min {ev.min():.3g}, max {ev.max():.3g}, condition {ev.max() / ev.min():.3g}")

print("\nA3. augmented models")
kf = JointKF(np.zeros(N), dt, 6.25e-6)
F, G, H = kf.F, kf.G[:, None], kf.H[None, :]
print("joint KF model x = [q, qd, d], q'' = u + d: rank obsv from q =", np.linalg.matrix_rank(obsv(F, H)),
      "of 3; rank ctrb from u =", np.linalg.matrix_rank(ctrb(F, G)), "of 3 (d is a disturbance, cancelled by feedforward)")
Al, Bl, K = lqr_design(dt)
print("LQR model x = [e, ed, int e]: rank ctrb =", np.linalg.matrix_rank(ctrb(Al, Bl)), "of 3")
cube = KalmanFilter3D(dt=CAM_DT, constant_velocity=False)
print("cube KF (constant position) x = p: A = I, C = I -> rank obsv =",
      np.linalg.matrix_rank(obsv(np.eye(3), np.eye(3))), "of 3")

print("\nA4. closed-loop poles (s = ln(z) / dt)")
z = np.linalg.eigvals(Al - Bl @ K)
print("LQR K =", K.ravel(), "\n  poles", np.round(np.log(z.astype(complex)) / dt, 1))
Pp = solve_discrete_are(F.T, H.T, kf.Q, np.array([[kf.R]]))
L = Pp @ H.T / (H @ Pp @ H.T + kf.R)
z = np.linalg.eigvals((np.eye(3) - L @ H) @ F)
print("joint KF steady-state gain L =", L.ravel(), "\n  estimator poles", np.round(np.log(z.astype(complex)) / dt, 1),
      "-> faster than the controller, as the separation principle design wants")

print("\nA5. check against MuJoCo's own dynamics at HOME (torque motors)")
m = build(motors=True).compile()
d = mujoco.MjData(m)
d.qpos[:N] = HOME
mujoco.mj_forward(m, d)
nx = 2 * m.nv + m.na
Ah, Bh = np.zeros((nx, nx)), np.zeros((nx, m.nu))
mujoco.mjd_transitionFD(m, d, 1e-6, True, Ah, Bh, None, None)
arm = np.r_[np.arange(N), m.nv + np.arange(N)]
Aa, Ba = Ah[np.ix_(arm, arm)], Bh[np.ix_(arm, range(N))]
Mf = np.zeros((m.nv, m.nv))
mujoco.mj_fullM(m, d, Mf)
M = Mf[:N, :N]
print("mass matrix M(HOME) (kg m^2), off-diagonals are the coupling computed torque removes:\n", M)
print("raw torque model: max |A - A_double_integrator| =", np.abs(Aa - Ad).max().round(5),
      "(gravity stiffness, removed by the h(q) term)")
print("raw torque model times M: max |B M - Bd| =", np.abs(Ba @ M - Bd).max().round(7))

robot = RobotModel(build(motors=True).compile())
m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT  # free-space dynamics, contact is a separate disturbance
rng = np.random.default_rng(0)
worst = 0.0
for _ in range(200):
    q = HOME + rng.uniform(-0.5, 0.5, N)
    qd = rng.uniform(-1, 1, N)
    v = rng.uniform(-5, 5, N)
    Mq, h = robot.dynamics(q, qd)
    mujoco.mj_resetData(m, d)
    d.qpos[:N], d.qvel[:N] = q, qd
    d.ctrl[:N] = np.clip(Mq @ v + h, -m.actuator_ctrlrange[:N, 1], m.actuator_ctrlrange[:N, 1])
    mujoco.mj_forward(m, d)
    worst = max(worst, np.abs(d.qacc[:N] - np.linalg.solve(Mq, d.ctrl[:N] - h)).max())
print(f"computed torque, 200 random states, no contact: max |q''_mujoco - v| = {worst:.3g} rad/s^2 "
      "(left over: the finger joints' inertia coupling, which the KF's d state absorbs)")
