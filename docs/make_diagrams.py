"""Draws docs/figures/frames.png and docs/figures/pipeline.png.

    python docs/make_diagrams.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent / "figures"
SRC = {"model": "#1f77b4", "calibration": "#d95f02", "measured": "#1b9e77"}


def box(ax, x, y, text, w=0.15, h=0.075, fc="#f4f4f4", ec="#333333", size=9, weight="normal"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.008", fc=fc, ec=ec, lw=1.2))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, weight=weight)


def arrow(ax, a, b, label="", color="#333333", size=8, off=(0, 0.035), ha="center", rad=0.0):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6,
                                                     connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
    if label:
        ax.text((a[0] + b[0]) / 2 + off[0], (a[1] + b[1]) / 2 + off[1], label, ha=ha, va="center", fontsize=size,
                color=color)


def frames():
    fig, ax = plt.subplots(figsize=(13, 8.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.01, 0.97, "Simulation: UR5e, camera on the wrist", fontsize=12, weight="bold", va="top")
    y = 0.86
    xs = [0.07, 0.30, 0.53, 0.88]
    box(ax, xs[0], y, "world", w=0.1)
    box(ax, xs[1], y, "base\n(UR5e base body)", w=0.14)
    box(ax, xs[2], y, "wrist_3_link", w=0.13)
    box(ax, xs[3], y, "camera\nOpenCV: x right, y down,\nz forward (optical axis)", w=0.2, h=0.1)
    arrow(ax, (xs[0] + 0.05, y), (xs[1] - 0.07, y), "Rz(180 deg)\nur5e.xml", SRC["model"], off=(0, 0.045))
    arrow(ax, (xs[1] + 0.07, y), (xs[2] - 0.065, y), "FK(q_hat): six link\ntransforms, ur5e.xml\nq from 20-bit encoders",
          SRC["model"], off=(0, 0.055))
    arrow(ax, (xs[2] + 0.065, y), (xs[3] - 0.1, y), "T_wrist_cam: 0.10 m along y, 0.08 m\nalong z, optical axis "
          "along +y\nsimulator.py (MuJoCo -z view, flip y and z)", SRC["model"], off=(0, 0.06))
    ax.text(xs[3] - 0.02, y - 0.1, "K: calibrate.py on 40 checkerboard views\n(K_calib.npy, fx 433.16)", ha="center",
            fontsize=8, color=SRC["calibration"])

    ax.text(0.01, 0.725, "Hardware: UR3e, camera fixed in the cell (D455 on 30 Oct)", fontsize=12, weight="bold",
            va="top")
    yb, yc = 0.47, 0.585
    box(ax, 0.07, yb, "base\n(UR controller\nbase frame)", w=0.11, h=0.095)
    box(ax, 0.53, yc, "camera\nOpenCV axes", w=0.13)
    box(ax, 0.40, 0.40, "tool0 (flange)", w=0.12)
    box(ax, 0.70, 0.40, "TCP\n(panel setting)", w=0.11)
    arrow(ax, (0.125, yb + 0.03), (0.465, yc), "", SRC["calibration"])
    ax.text(0.17, 0.645, "T_base_cam = T_base_cam_fit (calib_handeye.py, ArUco marker,\n20 poses, RMS 0.80 px)",
            fontsize=8, color=SRC["calibration"], ha="left", va="center")
    ax.text(0.17, 0.605, "+ 0.020 m in base z (T_base_cam_dz, from the marker check)", fontsize=8,
            color=SRC["measured"], ha="left", va="center")
    ax.text(0.53, yc - 0.065, "K: factory intrinsics (RealSense)", ha="center", fontsize=8, color=SRC["calibration"])
    arrow(ax, (0.125, yb - 0.03), (0.34, 0.40), "FK(q): RTB UR3e URDF, start=\"base\"\nq from RTDE joint readings",
          SRC["model"], off=(-0.02, -0.05))
    arrow(ax, (0.46, 0.40), (0.645, 0.40), "T_tool0_tcp: z 0.204 m\n(panel TCP setting)", SRC["measured"],
          off=(0, 0.045))

    ax.text(0.01, 0.27, "Cube measurement, both", fontsize=12, weight="bold", va="top")
    yp = 0.15
    steps = [(0.09, "pixel (u, v)\n+ depth"), (0.29, "camera frame\nK^-1 (u, v, 1) depth"), (0.49, "base frame"),
             (0.69, "minus loc_offset\n(hardware only)"), (0.89, "cube KF\nmeasurement")]
    for x, t in steps:
        box(ax, x, yp, t, w=0.15, h=0.085)
    labels = [("locate_cube: colour mask,\ntop face, table-ring\ndepth correction", SRC["model"]),
              ("T_base_cam\n(sim: FK(q_hat) x T_wrist_cam)", SRC["model"]),
              ("eval_objects.py --set-offset:\nmean bias against TCP touches", SRC["measured"]), ("", "#333333")]
    for (x0, _), (x1, _), (lab, col) in zip(steps, steps[1:], labels):
        arrow(ax, (x0 + 0.075, yp), (x1 - 0.075, yp), lab, col, off=(0, -0.085))

    for i, (name, col) in enumerate(SRC.items()):
        ax.plot([0.62 + i * 0.13], [0.965], "s", color=col, ms=10)
        ax.text(0.632 + i * 0.13, 0.965, {"model": "model file / code", "calibration": "calibration",
                                          "measured": "measured"}[name], fontsize=9, va="center", color=col)
    fig.savefig(OUT / "frames.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def pipeline():
    fig, ax = plt.subplots(figsize=(13, 6.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, 0.5, 0.92, "pick state machine:  look > approach > refine > descend > grasp > lift > transfer > lower > "
        "release > retreat > home", w=0.9, h=0.07, fc="#fff4d6", size=10)

    y1 = 0.66
    box(ax, 0.08, y1, "RGB-D camera\nsim: wrist, 10 Hz\nhw: fixed D455", w=0.13, h=0.12)
    box(ax, 0.27, y1, "locate_cube\ncolour mask, depth,\ntable-ring correction", w=0.15, h=0.12)
    box(ax, 0.47, y1, "cube KF\nconstant position\nNIS gate 16.27", w=0.14, h=0.12)
    box(ax, 0.67, y1, "PBVS\nv = lambda e on the\ngrasp pose, DLS", w=0.14, h=0.12)
    box(ax, 0.88, y1, "q_ref, qd_ref\n(joint reference)", w=0.14, h=0.1)
    arrow(ax, (0.145, y1), (0.195, y1))
    arrow(ax, (0.345, y1), (0.40, y1), "p_cam", off=(0, 0.08))
    arrow(ax, (0.54, y1), (0.60, y1), "p_cube, P", off=(0, 0.08))
    arrow(ax, (0.74, y1), (0.81, y1))

    y2 = 0.33
    box(ax, 0.08, y2, "joint encoders\nsim: 20 bit, 500 Hz\nhw: RTDE, ~87 Hz", w=0.13, h=0.12)
    box(ax, 0.30, y2, "joint KF\nx = [q, qd, d]\nper joint", w=0.14, h=0.12)
    box(ax, 0.62, y2, "joint LQR\non [e, ed, integral e]\n+ d_hat feedforward", w=0.16, h=0.12)
    box(ax, 0.88, y2 + 0.04, "sim: computed torque\ntau = M(q_hat) v + h\n> UR5e torque motors", w=0.17, h=0.11,
        fc="#e8f1fb")
    box(ax, 0.88, y2 - 0.1, "hw: speedJ joint\nvelocity > UR3e", w=0.17, h=0.08, fc="#e8f6f0")
    arrow(ax, (0.145, y2), (0.23, y2), "q", off=(0, 0.03))
    arrow(ax, (0.37, y2), (0.54, y2), "q_hat, qd_hat, d_hat", off=(0, 0.03))
    arrow(ax, (0.88, y1 - 0.05), (0.66, y2 + 0.06), "", rad=-0.15)
    arrow(ax, (0.70, y2 + 0.02), (0.795, y2 + 0.04))
    arrow(ax, (0.70, y2 - 0.03), (0.795, y2 - 0.09))
    arrow(ax, (0.30, y2 + 0.06), (0.27, y1 - 0.06), "q_hat > T_base_cam (sim)", off=(-0.085, 0))
    arrow(ax, (0.36, y2 + 0.06), (0.64, y1 - 0.06), "q_hat > FK, Jacobian", off=(0.075, -0.03))
    for x in (0.47, 0.67):
        ax.annotate("", xy=(x, y1 + 0.06), xytext=(x, 0.885), arrowprops=dict(arrowstyle="<|-|>", color="#999999",
                                                                             lw=1, ls="--"))
    ax.text(0.5, 0.06, "Everything after the sensors runs on estimates; simulator ground truth only scores results "
            "(*_eval_only).", ha="center", fontsize=10, style="italic")
    fig.savefig(OUT / "pipeline.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    frames()
    pipeline()
