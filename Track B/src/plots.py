import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.controller import PHASES

JOINTS = ["pan", "lift", "elbow", "wrist1", "wrist2", "wrist3"]


def shade(ax, log):
    t, ph = log["t"], log["phase"]
    edges = np.r_[0, np.flatnonzero(np.diff(ph)) + 1, len(ph) - 1]
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        if i % 2:
            ax.axvspan(t[a], t[b], color="0.92", lw=0)
        if ax.get_subplotspec().is_first_row():
            ax.text((t[a] + t[b]) / 2, 1.01, PHASES[ph[a]], transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=6, rotation=45)


def plot_trial(log, out, tag):
    t = log["t"]

    # joint KF: errors against truth with the filter's own 2 sigma
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    err = [(log["q"] - log["true_q"]) * 1e6, (log["qd"] - log["true_qd"]) * 1e3]
    sig = [log["sig"][:, 0] * 1e6, log["sig"][:, 1] * 1e3]
    for a, e, s, lab in zip(ax, err, sig, ["q error (urad)", "qd error (mrad/s)"]):
        shade(a, log)
        a.fill_between(t, -2 * s, 2 * s, color="C0", alpha=0.2, label="KF 2 sigma")
        for j in range(6):
            a.plot(t, e[:, j], lw=0.6, label=JOINTS[j])
        a.set_ylabel(lab)
    shade(ax[2], log)
    for j in range(6):
        ax[2].plot(t, log["true_d"][:, j], color=f"C{j + 1}", lw=0.5, alpha=0.4)
        ax[2].plot(t, log["d"][:, j], color=f"C{j + 1}", lw=1.0)
    ax[2].set_ylabel("disturbance d (rad/s^2)\nthin = truth, thick = KF")
    ax[0].legend(ncol=7, fontsize=7, loc="lower left")
    ax[2].set_xlabel("time (s)")
    fig.suptitle("Joint Kalman filter (encoders only): estimate - truth", y=0.995)
    fig.tight_layout()
    fig.savefig(out / f"{tag}_joint_kf.png", dpi=130)
    plt.close(fig)

    # cube KF
    fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    ok = ~np.isnan(log["cube"][:, 0]) & (log["phase"] <= PHASES.index("descend"))
    for i, a in enumerate(ax):
        shade(a, log)
        e = (log["cube"][:, i] - log["true_cube"][:, i]) * 1000
        s = log["cube_sig"][:, i] * 1000
        a.fill_between(t[ok], -2 * s[ok], 2 * s[ok], color="C0", alpha=0.2, label="KF 2 sigma")
        a.plot(t[ok], e[ok], color="C0", label="KF estimate - truth")
        if len(log.get("meas_t", [])):
            idx = np.searchsorted(t, log["meas_t"]).clip(0, len(t) - 1)
            a.plot(log["meas_t"], (log["meas"][:, i] - log["true_cube"][idx, i]) * 1000, ".", color="C3", ms=3,
                   label="locate_cube measurement - truth")
        a.set_ylabel(f"{'xyz'[i]} error (mm)")
        a.set_ylim(-8, 8)
    ax[0].legend(fontsize=7, loc="lower left")
    ax[2].set_xlabel("time (s)")
    fig.suptitle("Cube Kalman filter (base frame), up to the grasp", y=0.995)
    fig.tight_layout()
    fig.savefig(out / f"{tag}_cube_kf.png", dpi=130)
    plt.close(fig)

    # control
    fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    shade(ax[0], log)
    for j in range(6):
        ax[0].plot(t, log["q_ref"][:, j], lw=0.8, label=JOINTS[j])
    ax[0].set_ylabel("q_ref (rad)")
    ax[0].legend(ncol=6, fontsize=7)
    shade(ax[1], log)
    for j in range(6):
        ax[1].plot(t, (log["true_q"][:, j] - log["q_ref"][:, j]) * 1000, lw=0.7)
    ax[1].set_ylabel("tracking error\nq_true - q_ref (mrad)")
    shade(ax[2], log)
    for j in range(6):
        ax[2].plot(t, log["tau"][:, j] * 100, lw=0.7)
    ax[2].set_ylabel("torque (% of limit)")
    ax[2].set_xlabel("time (s)")
    fig.suptitle("PBVS reference, LQR tracking and effort", y=0.995)
    fig.tight_layout()
    fig.savefig(out / f"{tag}_control.png", dpi=130)
    plt.close(fig)
