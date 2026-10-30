"""Plots of the hardware logs. There is no ground truth on hardware, so each plot compares against the best
independent reference it has and says which.

    python plot_logs.py ..\\..\\..\\hw-data\\2026-10-28                        # every Tier 2 and Tier 3 log in it
    python plot_logs.py ...\\joint_control_20261028_152739.npz --sim ...\\sim\\joint_control_20261028_134100.npz
    python plot_logs.py ...\\pick_20261028_152630

Tier 2 (joint_control_*.npz): tracking, joint KF, sim against hardware (the sim run with the same reference, found
in sim\\ next to the log unless --sim is given). Tier 3 (pick_*\\trial*.npz): object estimate, control, kinematics
check, and a results table. PNGs and tier3_results.md go to --out (default: the date folder's plots\\).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from real import DT
from src.plots import JOINTS, shade


def loop_rate(t):
    # moving cycles only: the stops (looks, grip) leave gaps in a Tier 3 log
    dt = np.diff(t)
    dt = dt[dt < 0.1]
    return 1 / dt.mean(), 1e3 * dt.max(), 100 * np.mean(dt > 1.2 * DT)


def joints_table(rows):
    print("joint                     " + "".join(f"{JOINTS[j]:>9s}" for j in range(6)))
    for name, v, fmt in rows:
        print(f"{name:26s}" + "".join(f"{x:9{fmt}}" for x in v))


def same_move(d, s):
    # the same reference relative to its start, the same length, speed limits and plant model (tau from the config)
    a, b = d["q_ref"] - d["q_ref"][0], s["q_ref"] - s["q_ref"][0]
    return (a.shape == b.shape and np.allclose(a, b, atol=1e-9) and float(d["vmax"]) == float(s["vmax"])
            and float(d["accel"]) == float(s["accel"]) and np.allclose(d["tau"], s["tau"]))


def find_sim(path, d):
    for s in sorted((path.parent / "sim").glob("joint_control_*.npz"), reverse=True):
        if same_move(d, np.load(s)):
            return s
    return None


def tier2(path, out, sim):
    d = np.load(path)
    t, err = d["t"], 1e3 * (d["q"] - d["q_ref"])
    moved = np.flatnonzero(d["delta"])
    clipped, vmax, sig = d["clipped"], float(d["vmax"]), d["sig"]
    e_qd, e_q = 1e3 * (d["qd_hat"] - d["qd"]), 1e6 * (d["q_hat"] - d["q"])
    rate, worst, late = loop_rate(t)
    print(f"\n{path.name}: {len(t)} cycles, loop rate {rate:.1f} Hz (slowest cycle {worst:.1f} ms, {late:.1f} % over "
          f"{1.2e3 * DT:.0f} ms), clipped cycles {100 * clipped.any(1).mean():.1f} %")
    joints_table([("RMS tracking error (mrad)", np.sqrt((err ** 2).mean(0)), ".2f"),
                  ("max tracking error (mrad)", np.abs(err).max(0), ".2f"),
                  ("clipped (%)", 100 * clipped.mean(0), ".1f"),
                  ("qd_hat - qd in 2 sigma (%)", 100 * (np.abs(e_qd) < 2e3 * sig[:, 1]).mean(0), ".1f"),
                  ("q_hat - q in 2 sigma (%)", 100 * (np.abs(e_q) < 2e6 * sig[:, 0]).mean(0), ".1f")])

    # tracking: the measured q against the reference, and the command with its limits
    fig, ax = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    for j in range(6):
        ax[0].plot(t, err[:, j], lw=1.0 if j in moved else 0.6, label=JOINTS[j])
        ax[1].plot(t, d["u"][:, j], lw=0.8, color=f"C{j}")
        ax[1].plot(t[clipped[:, j]], d["u"][clipped[:, j], j], "x", color=f"C{j}", ms=4)
    for s in (-1, 1):
        ax[1].axhline(s * vmax, color="k", ls="--", lw=0.8)
    ax[0].set_ylabel("q - q_ref (mrad)\nmeasured joint angle - reference")
    ax[0].legend(ncol=6, fontsize=7)
    ax[1].set(xlabel="t (s)", ylabel="command u (rad/s)")
    ax[1].text(0.01, 0.95, f"dashed: vmax {vmax:g} rad/s, x: clipped cycle (speed or rate limit)",
               transform=ax[1].transAxes, fontsize=7, va="top")
    ax[0].set_title(f"{path.name}: joint tracking on the UR3e")
    fig.tight_layout()
    fig.savefig(out / f"{path.stem}_tracking.png", dpi=130)
    plt.close(fig)

    # joint KF: qd is an independent check unless the KF was fed the robot's qd (--use-qd)
    fig, ax = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    b_qd, b_q = 2e3 * sig[:, 1].max(1), 2e6 * sig[:, 0].max(1)
    ax[0].fill_between(t, -b_qd, b_qd, color="0.8", label="KF 2 sigma (largest joint)")
    ax[1].fill_between(t, -b_q, b_q, color="0.8", label="KF 2 sigma (largest joint)")
    for j in range(6):
        ax[0].plot(t, e_qd[:, j], lw=0.6, label=JOINTS[j])
        ax[1].plot(t, e_q[:, j], lw=0.6, label=JOINTS[j])
    for a, b in zip(ax, (b_qd, b_q)):
        a.plot(t, b, "k--", lw=0.6)
        a.plot(t, -b, "k--", lw=0.6)
    ax[0].set_ylabel("qd_hat - qd (mrad/s)")
    ax[0].set_title("velocity estimate against the robot's measured qd: " +
                    ("NOT independent, the KF was fed qd (--use-qd)" if bool(d["use_qd"]) else
                     "an independent check, the KF never sees qd"), fontsize=9)
    ax[1].set(xlabel="t (s)", ylabel="q_hat - q (urad)")
    ax[1].set_title("position: residual against the measurement the KF is fed (not an independent check)", fontsize=9)
    ax[0].legend(ncol=7, fontsize=7)
    fig.suptitle(f"{path.name}: joint KF")
    fig.tight_layout()
    fig.savefig(out / f"{path.stem}_kf.png", dpi=130)
    plt.close(fig)

    if sim is not None and not same_move(d, np.load(sim)):
        print(f"{Path(sim).name} is not the same move, limits and tau as {path.name}, sim vs hardware skipped")
        return
    sim = sim or find_sim(path, d)
    if sim is None:
        print("no sim run of the same move in sim\\ (or --sim), sim vs hardware skipped")
        return
    s = np.load(sim)
    s_err = 1e3 * (s["q"] - s["q_ref"])
    # the reference moves on one sample per cycle, and the sim runs slower than real time: line them up by cycle
    fig, ax = plt.subplots(len(moved), 1, figsize=(9, 2.6 * len(moved) + 0.8), sharex=True, squeeze=False)
    for a, j in zip(ax[:, 0], moved):
        a.plot(DT * np.arange(len(t)), err[:, j], lw=0.8,
               label=f"UR3e, RMS {np.sqrt((err[:, j] ** 2).mean()):.2f} mrad")
        a.plot(DT * np.arange(len(s_err)), s_err[:, j], lw=0.8,
               label=f"sim (MuJoCo UR3e), RMS {np.sqrt((s_err[:, j] ** 2).mean()):.2f} mrad")
        a.set_ylabel(f"{JOINTS[j]}: q - q_ref (mrad)")
        a.legend(fontsize=7)
    ax[-1, 0].set_xlabel(f"reference time (cycle x {1e3 * DT:.0f} ms)")
    ax[0, 0].set_title(f"same reference, controller and config: {path.name} against {Path(sim).name}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / f"{path.stem}_sim_vs_hw.png", dpi=130)
    plt.close(fig)
    print(f"sim vs hardware: {Path(sim).name}, RMS (mrad) "
          + ", ".join(f"{JOINTS[j]} UR3e {np.sqrt((err[:, j] ** 2).mean()):.2f} sim {np.sqrt((s_err[:, j] ** 2).mean()):.2f}"
                      for j in moved))


def tier3(path, out):
    d = np.load(path)
    res = json.loads(str(d["result"]))
    tag = f"{path.parent.name}_{path.stem}"
    if "t" not in d.files or len(d["t"]) < 2:
        print(f"{tag}: no moving cycles logged, no plots")
        return res
    log = {k: d[k] for k in ("t", "phase")}
    t = d["t"]
    res["loop_hz"] = loop_rate(t)[0]

    # object estimate: the cube KF with its 2 sigma, and the measurements it was given
    fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    mt, m, ok = d["meas_t"], d["meas"], d["meas_ok"].astype(bool)
    for i, a in enumerate(ax):
        shade(a, log)
        c, s = 1e3 * d["cube"][:, i], 2e3 * d["cube_sig"][:, i]
        a.fill_between(t, c - s, c + s, color="C0", alpha=0.25, lw=0, label="cube KF 2 sigma")
        a.plot(t, c, color="C0", lw=1, label="cube KF estimate")
        if len(mt):
            a.plot(mt[ok], 1e3 * m[ok, i], ".", color="C2", ms=4, label="measurement, used")
            a.plot(mt[~ok], 1e3 * m[~ok, i], "x", color="C3", ms=5, label="measurement, rejected or skipped")
        # a filter restart sets P back to 1 m^2: limits from the data, not the band
        v = np.r_[c[np.isfinite(c)], 1e3 * m[:, i] if len(mt) else []]
        if len(v):
            pad = max(5.0, 0.15 * np.ptp(v))
            a.set_ylim(v.min() - pad, v.max() + pad)
        a.set_ylabel(f"{'xyz'[i]} (mm)")
    ax[0].legend(ncol=2, fontsize=7, loc="best", framealpha=0.9)
    ax[-1].set_xlabel("t (s), stops (looks, grip) are the gaps between phases")
    fig.suptitle(f"{tag}: object estimate in the base frame (loc_offset applied), {res.get('n_rej', 0)} rejected", y=0.995)
    fig.tight_layout()
    fig.savefig(out / f"{tag}_object.png", dpi=130)
    plt.close(fig)

    # control: the PBVS target against where the robot says its TCP is, joint tracking, clipping
    fig, ax = plt.subplots(5, 1, figsize=(9, 10), sharex=True, gridspec_kw=dict(height_ratios=[1, 1, 1, 1.2, 0.6]))
    for i in range(3):
        shade(ax[i], log)
        ax[i].plot(t, d["p_target"][:, i], "k--", lw=0.8, label="PBVS target (incl. --hover)")
        ax[i].plot(t, d["tcp_robot"][:, i], color="C0", lw=1, label="robot TCP (get_tcp_pose)")
        ax[i].set_ylabel(f"{'xyz'[i]} (m)")
    ax[0].legend(fontsize=7, loc="lower left")
    shade(ax[3], log)
    e = 1e3 * (d["y"] - d["q_ref"])
    for j in range(6):
        ax[3].plot(t, e[:, j], lw=0.6, label=JOINTS[j])
    ax[3].set_ylabel("q - q_ref (mrad)")
    ax[3].legend(ncol=6, fontsize=6, loc="lower left")
    shade(ax[4], log)
    cl = d["clipped"].astype(bool)
    ax[4].eventplot([t[cl[:, j]] for j in range(6)], lineoffsets=range(6), linelengths=0.8,
                    colors=[f"C{j}" for j in range(6)])
    ax[4].set(yticks=range(6), yticklabels=JOINTS, xlabel="t (s)")
    ax[4].set_ylabel(f"clipped\n{100 * cl.any(1).mean():.0f} % of cycles")
    ax[4].tick_params(axis="y", labelsize=6)
    fig.suptitle(f"{tag}: control", y=0.995)
    fig.tight_layout()
    fig.savefig(out / f"{tag}_control.png", dpi=130)
    plt.close(fig)

    # kinematics: the RTB model at the measured q against the controller's own TCP, every cycle
    kin, rob = d["tcp_kin"], d["tcp_robot"]
    if np.isnan(rob).all():
        return res
    dp = 1e3 * (kin[:, :3] - rob[:, :3])
    ang = np.rad2deg(np.linalg.norm((Rotation.from_rotvec(kin[:, 3:]) * Rotation.from_rotvec(rob[:, 3:]).inv())
                                    .as_rotvec(), axis=1))
    res["kin_mm"], res["kin_deg"] = float(np.linalg.norm(dp, axis=1).max()), float(ang.max())
    fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    for a in ax:
        shade(a, log)
    for i in range(3):
        ax[0].plot(t, dp[:, i], lw=0.8, label="xyz"[i])
    ax[0].plot(t, np.linalg.norm(dp, axis=1), "k", lw=0.8, label="|error|")
    ax[0].set_ylabel("position (mm)")
    ax[0].legend(ncol=4, fontsize=7)
    ax[1].plot(t, ang, "k", lw=0.8)
    ax[1].set(xlabel="t (s)", ylabel="rotation (deg)")
    fig.suptitle(f"{tag}: kinematic model - robot's own TCP\n(RTB UR3e + T_tool0_tcp at the measured q, against "
                 "get_tcp_pose)", y=0.995, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"{tag}_kinematics.png", dpi=130)
    plt.close(fig)
    return res


def results_table(rows, out):
    head = ["run", "trial", "mode", "success", "time (s)", "looks", "n_meas", "n_rej", "n_skip", "reapproach",
            "resets", "clipped %", "grip (mm)", "loop (Hz)", "FK - robot max", "note"]
    fmt = lambda v, f: "-" if v is None else format(v, f)
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for run, r in rows:
        a = r.get("args", {})
        hover = float(a.get("hover", 0) or 0)
        mode = (("sim " if a.get("sim") == "True" else "") + (f"dry run {hover:g} m" if hover else "pick")
                + (", track" if a.get("track") == "True" else ""))
        note = r.get("failed") or (r.get("reason") if r.get("phase") == "interrupted" else "")
        kin = f"{r['kin_mm']:.1f} mm, {r['kin_deg']:.2f} deg" if "kin_mm" in r else "-"
        clip = 100 * r["clipped"] if r.get("clipped") is not None else None
        loop = fmt(r.get("loop_hz"), ".1f") + (f" (tracking {r['track_hz']:.1f})" if r.get("track_hz") else "")
        lines.append("| " + " | ".join([run, str(r.get("trial")), mode, "yes" if r.get("phase") == "done" else "no",
                                         fmt(r.get("t"), ".1f"), fmt(r.get("looks"), "d"), fmt(r.get("n_meas"), "d"),
                                         fmt(r.get("n_rej"), "d"), fmt(r.get("n_skip"), "d"),
                                         fmt(r.get("reapproach"), "d"), fmt(r.get("resets"), "d"), fmt(clip, ".0f"),
                                         fmt(r.get("grip_width"), ".1f"), loop, kin, note]) + " |")
    text = "\n".join(lines)
    print("\n" + text)
    (out / "tier3_results.md").write_text(text + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="a joint_control_*.npz or trial*.npz, a pick_* folder or a date folder")
    ap.add_argument("--out", type=Path, help="default: the date folder's plots\\")
    ap.add_argument("--sim", type=Path, help="Tier 2: the sim run to compare against")
    args = ap.parse_args()

    p = args.path
    base = p if p.is_dir() else p.parent
    while base.name.startswith("pick_") or base.name == "sim":
        base = base.parent
    out = args.out or base / "plots"
    out.mkdir(parents=True, exist_ok=True)
    if p.is_dir():
        tier2_files = sorted(p.glob("joint_control_*.npz"))
        trials = sorted(p.glob("trial*.npz")) if p.name.startswith("pick_") else sorted(p.glob("pick_*/trial*.npz"))
    else:
        tier2_files = [p] if p.name.startswith("joint_control_") else []
        trials = [p] if p.name.startswith("trial") else []
    if not tier2_files and not trials:
        raise SystemExit(f"no joint_control_*.npz or pick_*\\trial*.npz in {p}")
    for f in tier2_files:
        tier2(f, out, args.sim)
    rows = [(f.parent.name, tier3(f, out)) for f in trials]
    if rows:
        results_table(rows, out)
    print(f"\nplots in {out}")
