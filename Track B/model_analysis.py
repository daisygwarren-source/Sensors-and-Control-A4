"""State-space model of the arm, controllability / observability, and a check of the model against MuJoCo.
A6 is the real UR3e's velocity-input model, with the lag measured by hardware/fit_steps.py.

    python model_analysis.py
    python model_analysis.py --config C:\\Dev\\University\\41014\\hw-data\\2026-10-28\\config.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import mujoco
from scipy.linalg import solve_discrete_are

from simulator import build
from main import HOME, CAM_DT
from src.model import (N, joint_model, arm_model, velocity_joint_model, velocity_arm_model, c2d, ctrb, obsv,
                       gramians, pbh, RobotModel)
from src.controller import lqr_design, vel_lqr_design, JointVelLQR
from src.estimation import JointKF, JointVelKF, KalmanFilter3D

ap = argparse.ArgumentParser()
ap.add_argument("--config", help="hardware config.json with tau (and td, sigma_q) for A6")
args = ap.parse_args()

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

print("\nA6. real UR3e under speedJ, x = [q, qd], u = commanded joint velocity, qd' = (u - qd) / tau, y = q")
cfg = json.loads(Path(args.config).read_text()) if args.config else {}
dtr, u_max, a_max = 0.01, 0.05, 0.1
tau = np.broadcast_to(np.asarray(cfg.get("tau", 0.2), float), N)
td = np.round(np.broadcast_to(np.asarray(cfg.get("td", 0.0), float), N) / dtr).astype(int)
sig_q = np.broadcast_to(np.asarray(cfg.get("sigma_q", 1e-5), float), N)
print("tau (s) =", tau, "from", args.config or "the starter sim's own lag (no --config given)")
A1, B1, C1 = velocity_joint_model(tau[5])
Ad1, Bd1 = c2d(A1, B1, dtr)
print("wrist 3: A =\n", A1, "\nB =", B1.ravel(), " C =", C1.ravel())
print("ZOH, dt = 10 ms:\nAd =\n", Ad1, "\nBd =", Bd1.ravel())
Av, Bv, Cv = velocity_arm_model(tau)
print("all six joints: rank ctrb =", np.linalg.matrix_rank(ctrb(Av, Bv)), "of 12;  rank obsv with C = q:",
      np.linalg.matrix_rank(obsv(Av, Cv)), ", with C = [q, qd]:", np.linalg.matrix_rank(obsv(Av, np.eye(2 * N))))
for lam, rc, ro in pbh(Av, Bv, Cv):
    print(f"PBH at eigenvalue {lam.real:g}: rank[lI-A, B] = {rc}, rank[lI-A; C] = {ro}")
lag = tau < 2 * dtr
print(f"joints with tau < 2 control periods (lag negligible, model reduces to qd = u): {np.flatnonzero(lag).tolist()}")
kf = JointVelKF(np.zeros(N), dtr, tau, sig_q, delay=td)
F, G = kf.F[5], kf.G[5][:, None]
print("joint KF x = [q, qd, d], qd' = (u + d - qd) / tau: rank obsv from q =",
      np.linalg.matrix_rank(obsv(F, np.eye(3)[:1])), ", from [q, qd] =", np.linalg.matrix_rank(obsv(F, np.eye(3)[:2])),
      "of 3; rank ctrb from u =", np.linalg.matrix_rank(ctrb(F, G)), "of 3 (d is a disturbance, cancelled by u - d)")
lqr = JointVelLQR(dtr, tau, u_max, a_max)
print("closed-loop poles s = ln(z) / dt, per joint (controller | estimator, steady-state KF from q):")
for j in range(N):
    Al, Bl, K = vel_lqr_design(dtr, lqr.tau[j])
    pc = np.log(np.linalg.eigvals(Al - Bl @ K).astype(complex)) / dtr
    H = np.eye(3)[:1]
    Pp = solve_discrete_are(kf.F[j].T, H.T, kf.Q[j], kf.R[j])
    L = Pp @ H.T / (H @ Pp @ H.T + kf.R[j])
    pe = np.log(np.linalg.eigvals((np.eye(3) - L @ H) @ kf.F[j]).astype(complex)) / dtr
    print(f"  joint {j}: K = {K.ravel()}  controller {np.round(pc, 2)}  estimator {np.round(pe, 1)}")

# closed loop on the model alone: exact discrete lag plant (plus the measured dead time), encoder noise, KF + LQR
rng = np.random.default_rng(0)
plant = [c2d(*velocity_joint_model(t)[:2], dtr) for t in lqr.tau]
Ap, Bp = np.array([p[0] for p in plant]), np.array([p[1][:, 0] for p in plant])
x, sent, step = np.zeros((N, 2)), np.zeros((td.max() + 1, N)), np.full(N, 0.1)
kf, lqr = JointVelKF(np.zeros(N), dtr, tau, sig_q, delay=td), JointVelLQR(dtr, tau, u_max, a_max)
u, qs, us, clip = np.zeros(N), [], [], []
for k in range(2000):
    kf.predict(u)
    kf.update(x[:, 0] + sig_q * rng.standard_normal(N))
    qh, qdh, dh = kf.X
    u, c = lqr(qh, qdh, step, 0, 0, dh)
    sent = np.roll(sent, 1, axis=0)
    sent[0] = u
    x = np.einsum("jab,jb->ja", Ap, x) + Bp * sent[td, np.arange(N)][:, None]
    qs.append(x[:, 0])
    us.append(u)
    clip.append(c)
qs, us = np.array(qs), np.array(us)
settle = [(np.flatnonzero(np.abs(qs[:, j] - 0.1) > 0.005)[-1] + 1) * dtr for j in range(N)]
print(f"0.1 rad step in q_ref on every joint, 20 s, u_max {u_max} rad/s, a_max {a_max} rad/s^2:")
print("  peak |u| (rad/s)  ", np.abs(us).max(0).round(4), "-> under u_max:", bool((np.abs(us).max(0) < u_max).all()))
print("  overshoot (mrad)  ", ((qs.max(0) - 0.1) * 1e3).round(2))
print("  settle to 5 % (s) ", np.round(settle, 2))
print("  error at 20 s (mrad)", ((qs[-1] - 0.1) * 1e3).round(3))
print("  clipped cycles (%)", (100 * np.mean(clip, axis=0)).round(1), "(the a_max ramp at the start)")
