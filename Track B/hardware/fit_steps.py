"""Lunch: the joint lag and encoder noise from the morning's step_*.npz and joints_*.npz -> config (tau, td,
sigma_q, sigma_qd). Saves a plot per step file in FOLDER/fits.

    python fit_steps.py ..\\..\\..\\hw-data\\2026-10-28
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import lfilter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from real import DT, update_config
from src.model import TAU_MIN

H = 0.001  # model simulated on a 1 ms grid


class Step:
    # one step_test.py file. model: command -> speedJ ramp at accel -> dead time td -> first-order lag tau -> qd
    def __init__(self, path):
        d = np.load(path)
        self.path, self.j, self.accel = Path(path), int(d["joint"]), float(d["accel"])
        self.ramp = not ("sim" in d.files and bool(d["sim"]))  # MuJoCoUR3e has no speedJ ramp
        self.t, self.cmd, self.qd = d["t"], d["cmd"][:, self.j], d["qd"][:, self.j]
        # each command goes out at the start of its cycle, one period before that cycle's logged read
        tc = self.t - np.median(np.diff(self.t))
        self.grid = np.arange(tc[0], self.t[-1] + H, H)
        c = self.cmd[np.clip(np.searchsorted(tc, self.grid, side="right") - 1, 0, None)]
        self.r = c.copy()
        if self.ramp:
            for k in range(1, len(c)):
                self.r[k] = self.r[k - 1] + np.clip(c[k] - self.r[k - 1], -self.accel * H, self.accel * H)

    def model(self, tau, td):
        r = np.interp(self.grid - td, self.grid, self.r)
        if tau > 0:
            a = 1 - np.exp(-H / tau)
            r = lfilter([a], [1, a - 1], r)
        return np.interp(self.t, self.grid, r)

    def residual(self, tau, td):
        return self.model(tau, td) - self.qd


def fit(steps):
    # coarse grid, then least squares from the best point, all of one joint's files together
    res = lambda p: np.concatenate([s.residual(*p) for s in steps])
    grid = [(tau, td) for tau in np.geomspace(0.002, 0.5, 25) for td in np.arange(0, 0.15, 0.005)]
    p0 = min(grid, key=lambda p: np.sum(res(p) ** 2))
    p = least_squares(res, p0, bounds=([TAU_MIN, 0], [1.0, 0.15]), diff_step=1e-3).x
    rms = lambda p: np.sqrt(np.mean(res(p) ** 2))
    return p, rms(p), rms((0, 0))


def noise(path):
    d = np.load(path)
    q, qd = d["q"], d["qd"]
    k = np.arange(len(q))
    # remove any slow drift so only the reading noise is left
    q = q - np.polynomial.polynomial.polyval(k, np.polynomial.polynomial.polyfit(k, q, 1)).T
    sq, sqd = q.std(0), qd.std(0)
    steps = [np.abs(np.diff(np.unique(d["q"][:, j]))) for j in range(6)]
    smallest = min((s.min() for s in steps if len(s)), default=None)
    flags = []
    for j in range(6):
        if len(steps[j]) < 4:
            # quantised: uniform rounding error, variance step^2 / 12
            step = steps[j].min() if len(steps[j]) else smallest
            if step is None:
                sq[j] = max(sq[j], 1e-6)
                flags.append(f"joint {j}: q never changed on any joint, sigma_q floored at 1e-6 rad")
            else:
                sq[j] = max(sq[j], step / np.sqrt(12))
                flags.append(f"joint {j}: {len(steps[j]) + 1} distinct q values, quantised, sigma_q = step / sqrt(12)")
        if sqd[j] < 1e-4:
            # an exact or near-exact reading at rest says nothing about qd noise in motion
            sqd[j] = 1e-4
            flags.append(f"joint {j}: qd constant at rest, sigma_qd floored at 1e-4 rad/s")
    return sq, sqd, flags


def is_sim(path):
    with np.load(path) as d:
        return "sim" in d.files and bool(d["sim"])


def real_only(paths, kind):
    # a --sim test run saved into the lab's folder must not be fitted together with the real recordings
    sim = [p for p in paths if is_sim(p)]
    if sim and len(sim) < len(paths):
        print(f"ignoring {len(sim)} sim {kind} file(s): {', '.join(p.name for p in sim)}")
        return [p for p in paths if p not in sim]
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--config", type=Path, help="default FOLDER/config.json")
    args = ap.parse_args()
    config = args.config or args.folder / "config.json"

    steps = [Step(p) for p in real_only(sorted(args.folder.glob("step_*.npz")), "step")]
    if not steps:
        raise SystemExit(f"no step_*.npz in {args.folder}")
    (args.folder / "fits").mkdir(exist_ok=True)
    tau, td, fitted = np.zeros(6), np.zeros(6), {}
    print("joint  files  tau (ms)  td (ms)  RMS lag model  RMS ramp only (mrad/s)")
    for j in sorted({s.j for s in steps}):
        mine = [s for s in steps if s.j == j]
        (tau[j], td[j]), rms, rms0 = fit(mine)
        fitted[j] = True
        note = "  <- tau < 2 periods: lag negligible, qd = u" if tau[j] < 2 * DT else ""
        print(f"{j:5d}  {len(mine):5d}  {1e3 * tau[j]:8.1f}  {1e3 * td[j]:7.1f}  {1e3 * rms:13.3f}  {1e3 * rms0:13.3f}{note}")
        for s in mine:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(s.t, s.cmd, "k:", lw=1, label="command")
            ax.plot(s.t, s.qd, ".", ms=2, label="measured qd")
            ax.plot(s.t, s.model(tau[j], td[j]), lw=1.2,
                    label=f"ramp + dead time {1e3 * td[j]:.0f} ms + lag {1e3 * tau[j]:.0f} ms")
            ax.plot(s.t, s.model(0, 0), lw=1, label="ramp only" if s.ramp else "command only (sim: no ramp)")
            ax.set(xlabel="t (s)", ylabel=f"joint {j} velocity (rad/s)", title=f"{s.path.name}, accel {s.accel:g}")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(args.folder / "fits" / f"{s.path.stem}.png", dpi=130)
            plt.close(fig)
    # joints without a step test take the nearest tested joint's values
    tested = np.array(sorted(fitted))
    source = [int(tested[np.argmin(np.abs(tested - j))]) for j in range(6)]
    for j in range(6):
        if source[j] != j:
            tau[j], td[j] = tau[source[j]], td[source[j]]
            print(f"joint {j}: no step test, using joint {source[j]}'s tau and td")
    print(f"plots in {args.folder / 'fits'}")

    keys = dict(tau=tau, td=td, tau_source=source)
    joints = sorted(real_only(sorted(args.folder.glob("joints_*.npz")), "joints"), key=lambda p: len(np.load(p)["t"]))
    if joints:
        sq, sqd, flags = noise(joints[-1])
        print(f"\nnoise from {joints[-1].name} ({len(np.load(joints[-1])['t'])} samples)")
        print("joint   sigma q (urad)  sigma qd (mrad/s)")
        for j in range(6):
            print(f"{j:5d}  {1e6 * sq[j]:14.3f}  {1e3 * sqd[j]:17.4f}")
        for f in flags:
            print(f)
        keys.update(sigma_q=sq, sigma_qd=sqd)
    else:
        print(f"\nno joints_*.npz in {args.folder}, sigma_q / sigma_qd not written")
    update_config(config, **keys)
