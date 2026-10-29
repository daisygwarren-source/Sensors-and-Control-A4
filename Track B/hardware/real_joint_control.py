"""Tier 2 (Part D on hardware): the joint KF + LQR move the real arm out to q0 + delta and back, logging the estimate
and its uncertainty.

    python real_joint_control.py --sim
    python real_joint_control.py --ip ROBOT_IP --config ...\\hw-data\\2026-10-28\\config.json --delta 0 0 0.1 0 0 0.1
"""
import argparse
import json
import time
import traceback
from datetime import datetime

import numpy as np

from real import (DT, MAX_SPEED, Abort, JointLoop, add_args, check_args, connect, load_config, move_time, quintic,
                  sim_start)
from src.controller import JointVelLQR, VEL_Q, VEL_R
from src.estimation import JointVelKF


def reference(q0, delta, duration, hold, repeats):
    # 1 s still at q0 while the KF settles, then out, hold, back, hold
    def still(q, T):
        n = int(round(T / DT))
        return np.tile(q, (n, 1)), np.zeros((n, 6)), np.zeros((n, 6))
    q1 = q0 + delta
    segs = [still(q0, 1.0)]
    for _ in range(repeats):
        segs += [quintic(q0, q1, duration), still(q1, hold), quintic(q1, q0, duration), still(q0, hold)]
    return [np.vstack(s) for s in zip(*segs)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    ap.add_argument("--delta", type=float, nargs=6, default=[0, 0, 0.1, 0, 0, 0.1], help="rad per joint")
    ap.add_argument("--duration", type=float, default=8.0, help="s per move")
    ap.add_argument("--hold", type=float, default=2.0, help="s at each end")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--vmax", type=float, default=MAX_SPEED, help=f"rad/s, at most {MAX_SPEED}")
    ap.add_argument("--accel", type=float, default=0.10, help="speedJ acceleration and the LQR's rate limit, rad/s^2")
    ap.add_argument("--use-qd", action="store_true", help="also feed the measured joint velocity to the KF")
    ap.add_argument("--no-dhat", action="store_true", help="don't cancel the KF's disturbance estimate")
    ap.add_argument("--viewer", action="store_true", help="MuJoCo viewer with --sim")
    args = ap.parse_args()
    check_args(ap, args)
    if not 0 < args.vmax <= MAX_SPEED:
        ap.error(f"--vmax must be > 0 and at most {MAX_SPEED}")
    if args.accel <= 0 or args.duration <= 0 or args.hold < 0 or args.repeats < 1:
        ap.error("--accel, --duration must be > 0, --hold >= 0, --repeats >= 1")

    cfg = load_config(args.config)
    tau = np.broadcast_to(np.asarray(cfg["tau"], float), 6)
    delta = np.array(args.delta)
    # the reference has to leave room for feedback: peak speed and the feedforward's rate within 60 % of the limits
    _, qd_ref, qdd_ref = reference(np.zeros(6), delta, args.duration, args.hold, args.repeats)
    u_ff = qd_ref + tau * qdd_ref
    peak, ramp = np.abs(qd_ref).max(), np.abs(np.diff(u_ff, axis=0)).max() / DT
    if peak > 0.6 * args.vmax or ramp > 0.6 * args.accel:
        ap.error(f"peak |qd_ref| {peak:.4f} rad/s (limit {0.6 * args.vmax:.4f}), feedforward rate {ramp:.4f} rad/s^2 "
                 f"(limit {0.6 * args.accel:.4f}): use --duration {np.ceil(move_time(delta, args.vmax, args.accel)):.0f} "
                 f"or a smaller --delta")

    robot = connect(args, control=True, max_speed=args.vmax, accel=args.accel)
    log, reason = {k: [] for k in ("t", "q", "qd", "x", "sig", "q_ref", "qd_ref", "u", "clipped")}, "completed"
    try:
        if args.sim and "look_q" in cfg:
            sim_start(robot, cfg["look_q"])
        q0 = robot.get_joint_positions()
        if np.abs(robot.get_joint_velocities()).max() > 0.01:
            raise Abort("the arm is already moving")
        q_ref, qd_ref, qdd_ref = reference(q0, delta, args.duration, args.hold, args.repeats)
        sig_qd = cfg["sigma_qd"] if args.use_qd else None
        delay = np.round(np.asarray(cfg.get("td", 0.0)) / DT).astype(int)
        kf = JointVelKF(q0, DT, tau, cfg["sigma_q"], sig_qd, delay=delay, sigma_t=cfg.get("sigma_t", 0.0))
        lqr = JointVelLQR(DT, tau, args.vmax, args.accel)
        reach = np.abs(delta) + 0.05
        loop = JointLoop(robot, kf, lqr, args.use_qd, not args.no_dhat, max_err=0.1, bounds=(q0 - reach, q0 + reach))
        print(f"\nq0 = {np.round(q0, 4)}\nmoving to q0 + {delta}, {len(q_ref) * DT:.1f} s. Ctrl+C stops the arm")
        t0 = time.perf_counter()
        for i in range(len(q_ref)):
            u, clipped = loop.step(q_ref[i], qd_ref[i], qdd_ref[i])
            for k, v in zip(log, (time.perf_counter() - t0, loop.q, loop.qd, kf.X.copy(), kf.sigma, q_ref[i],
                                  qd_ref[i], u, clipped)):
                log[k].append(v)
    except Abort as e:
        reason = f"aborted: {e}"
    except KeyboardInterrupt:
        reason = "stopped with Ctrl+C"
    except Exception as e:
        traceback.print_exc()
        reason = f"error: {e!r}"
    finally:
        robot.stop()
        robot.close()
    print(reason)
    if not log["t"]:
        raise SystemExit("nothing logged")

    log = {k: np.array(v) for k, v in log.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"joint_control_{datetime.now():%Y%m%d_%H%M%S}.npz"
    with open(path, "xb") as f:
        np.savez(f, **{k: v for k, v in log.items() if k != "x"}, q_hat=log["x"][:, 0], qd_hat=log["x"][:, 1],
                 d_hat=log["x"][:, 2], K=lqr.K, lqr_Q=VEL_Q, lqr_R=VEL_R, kf_Q=kf.Q, kf_R=kf.R, tau=lqr.tau,
                 delay=kf.delay, delta=delta, vmax=args.vmax, accel=args.accel, use_qd=args.use_qd,
                 use_dhat=not args.no_dhat, config=json.dumps(cfg), reason=reason)

    err = log["q"] - log["q_ref"]
    t = log["t"]
    print(f"\nsaved {path}")
    print("joint                      " + "".join(f"{j:>9d}" for j in range(6)))
    print("RMS tracking error (mrad)  " + "".join(f"{v:9.2f}" for v in 1e3 * np.sqrt((err**2).mean(0))))
    print("max tracking error (mrad)  " + "".join(f"{v:9.2f}" for v in 1e3 * np.abs(err).max(0)))
    print("KF sigma q at end (urad)   " + "".join(f"{v:9.2f}" for v in 1e6 * log["sig"][-1, 0]))
    print(f"clipped cycles {100 * log['clipped'].any(1).mean():.1f} %, "
          f"{len(t)} cycles, mean loop rate {(len(t) - 1) / (t[-1] - t[0]):.1f} Hz")
