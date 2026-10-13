"""Randomised trials of the full pipeline. Ground truth is only used here, to score.

    python evaluate.py --study robust --trials 20
    python evaluate.py --study all --trials 8
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "results"

ROBUST = {
    "nominal": {},
    "cube yaw +/-45 deg": {"yaw": 45},
    "camera noise 3x": {"cam_noise": 3.0},
    "camera noise 10x": {"cam_noise": 10.0},
    "14-bit encoders": {"enc_bits": 14},
    "model masses +20%": {"mass_scale": 1.2},
    "30 N push while descending": {"push": ("descend", 0.3, [30.0, 0, 0], 0.2)},
    "40 N push while carrying": {"push": ("transfer", 0.3, [0, 40.0, 0], 0.3)},
    "cube nudged 3 cm on approach": {"nudge": ("approach", 0.4, 0.03)},
    "all of the above (3x noise)": {"yaw": 45, "cam_noise": 3.0, "enc_bits": 14, "mass_scale": 1.2,
                                    "push": ("transfer", 0.3, [0, 40.0, 0], 0.3), "nudge": ("approach", 0.4, 0.03)},
}
PUSH = {"push": ("transfer", 0.3, [0, 40.0, 0], 0.3)}
STUDIES = {
    "robust": ROBUST,
    "lqr": {f"R = {r:g}": {"lqr": {"R": r}, **PUSH} for r in [0.1, 0.3, 1.0, 3.0, 10.0]},
    "kf": {f"sigma_d = {s:g}": {"kf": {"sigma_d": s}, **PUSH} for s in [2.0, 5.0, 20.0, 50.0, 200.0]},
    "cube": {"CV model (original), nominal": {"cube_cv": True, "cube_q": 1e-6, "cube_R": 1.0},
             "CV model (original), nudged": {"cube_cv": True, "cube_q": 1e-6, "cube_R": 1.0,
                                             "nudge": ("approach", 0.4, 0.03)},
             "CP model, R x 0.1": {"cube_R": 0.1}, "CP model, R x 1": {"cube_R": 1.0},
             "CP model, R x 4 (default)": {}, "CP model, R x 10": {"cube_R": 10.0},
             "CP model, R x 4, nudged": {"nudge": ("approach", 0.4, 0.03)}},
    "dhat": {f"{name}, d-hat {'on' if on else 'off'}": {"use_dhat": on, **c}
             for name, c in [("nominal", {}), ("40 N push", PUSH), ("masses +20%", {"mass_scale": 1.2})]
             for on in (True, False)},
}


def score(res, log):
    from src.controller import PHASES
    e = np.stack([log["q"] - log["true_q"], log["qd"] - log["true_qd"]], -1)
    Pi = np.linalg.inv(log["P"][:, :2, :2])
    nees = np.einsum("kji,kil,kjl->kj", e, Pi, e)
    cam = (log["phase"] <= PHASES.index("descend")) & ~np.isnan(log["cube"][:, 0])
    ec = log["cube"][cam] - log["true_cube"][cam]
    cnees = np.einsum("ki,kil,kl->k", ec, np.linalg.inv(log["cube_P"][cam]), ec)
    trk = np.abs(log["true_q"] - log["q_ref"]).max(axis=1)
    g = (log["phase"] >= PHASES.index("grasp")) & (log["phase"] <= PHASES.index("lift"))
    return dict(lifted=bool(res["lifted"]), placed=bool(res["placed"]), t=res["events"].get("done", np.nan),
                grasp=1000 * res.get("grasp_err", np.nan), est=1000 * res.get("est_err", np.nan),
                q_rms=1e6 * np.sqrt((e[..., 0] ** 2).mean()), qd_rms=1e3 * np.sqrt((e[..., 1] ** 2).mean()),
                nees=nees.mean(), nees95=(nees < 5.99).mean(), cnees=cnees.mean(), rej=res["n_rej"],
                trk_rms=1e3 * np.sqrt((trk ** 2).mean()), trk_max=1e3 * trk[~g].max(initial=0), trk_grasp=1e3 * trk[g].max(initial=0),
                tau_rms=100 * np.sqrt((log["tau"] ** 2).mean()),
                rough=100 * np.abs(np.diff(log["tau"], axis=0)).mean())


def trial(job):
    seed, cfg = job
    from main import make_trial, run_trial
    rng = np.random.default_rng(seed)
    cfg = dict(cfg)
    model, data = make_trial(rng, motors=True, yaw_range=np.deg2rad(cfg.pop("yaw", 0)))
    if "nudge" in cfg:
        a = rng.uniform(0, 2 * np.pi)
        cfg["nudge"] = (*cfg["nudge"][:2], cfg["nudge"][2] * np.array([np.cos(a), np.sin(a)]))
    if "push" in cfg:
        cfg["push"] = (*cfg["push"][:2], np.array(cfg["push"][2]), cfg["push"][3])
    return score(*run_trial(model, data, rng, **cfg))


COLS = [("lifted", "picked", "{:.0%}"), ("placed", "placed", "{:.0%}"), ("grasp", "grasp offset mm", "{:.1f}"),
        ("est", "cube est err mm", "{:.2f}"), ("cnees", "cube NEES (3)", "{:.1f}"), ("rej", "NIS rejects", "{:.1f}"),
        ("q_rms", "q err urad", "{:.1f}"), ("qd_rms", "qd err mrad/s", "{:.2f}"), ("nees", "joint NEES (2)", "{:.2f}"),
        ("trk_rms", "track rms mrad", "{:.3f}"), ("trk_max", "track max mrad", "{:.2f}"),
        ("trk_grasp", "track max grasp/lift mrad", "{:.2f}"),
        ("tau_rms", "torque rms %", "{:.1f}"), ("rough", "torque rough %", "{:.3f}"), ("t", "time s", "{:.1f}")]


def table(rows):
    lines = ["| condition | " + " | ".join(c[1] for c in COLS) + " |", "|---" * (len(COLS) + 1) + "|"]
    for name, rs in rows.items():
        cells = []
        for key, _, fmt in COLS:
            v = np.array([r[key] for r in rs], float)
            cells.append(fmt.format(np.nanmean(v)) + (f" / {np.nanmax(v):.1f}" if key in ("grasp", "est") else ""))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_tradeoffs(study, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = {k: {c: np.nanmean([r[c] for r in rs]) for c in rs[0]} for k, rs in rows.items()}
    fig, ax = plt.subplots(figsize=(5, 4))
    if study == "lqr":
        x, y, xl, yl = "tau_rms", "trk_max", "torque rms (% of limit)", "peak tracking error (mrad)"
    elif study == "kf":
        x, y, xl, yl = "rough", "trk_max", "torque roughness, mean |dtau| per step (%)", "peak tracking error (mrad)"
    else:
        return
    for k, v in m.items():
        ax.plot(v[x], v[y], "o", color="C0")
        ax.annotate(k, (v[x], v[y]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(f"{study} trade-off (40 N push while carrying)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / f"tradeoff_{study}.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="robust", choices=[*STUDIES, "all"])
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 2) - 2)))
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    seeds = range(1000, 1000 + args.trials)
    with ProcessPoolExecutor(args.workers) as ex:
        for study in (STUDIES if args.study == "all" else [args.study]):
            conds = STUDIES[study]
            jobs = [(s, c) for c in conds.values() for s in seeds]
            out = list(ex.map(trial, jobs))
            rows = {name: out[i * args.trials:(i + 1) * args.trials] for i, name in enumerate(conds)}
            text = f"## {study} ({args.trials} randomised trials per row, same seeds in every row)\n\n" + table(rows)
            print(text + "\n", flush=True)
            (OUT / f"eval_{study}.md").write_text(text + "\n")
            plot_tradeoffs(study, rows)
