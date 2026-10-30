# Track B full pipeline: sim demo notes

> Working notes for the hardware-induction sim demo, drafted with Claude Code (Opus 5.5). Not project-page text:
> rewrite in your own words before anything goes on the page, and list the tool and what it did in the GenAI declaration.

## Run (PowerShell)

From the repo root, after setting up `.venv` there (see Fresh environment below):

```
cd "Track B"
..\.venv\Scripts\python.exe main.py                      # live demo: back-to-back randomised picks until the window is closed
..\.venv\Scripts\python.exe main.py --yaw 45 --nudge     # cube rotated +/-45 deg, and moved 3 cm while the arm approaches
..\.venv\Scripts\python.exe main.py --push               # 40 N shove on the wrist while carrying the cube
..\.venv\Scripts\python.exe main.py --cam-noise 10       # 10x depth noise (+/-2 cm wavy depth error)
..\.venv\Scripts\python.exe main.py --headless --seed 0  # one trial, prints the result, saves plots to results/
..\.venv\Scripts\python.exe model_analysis.py            # Part A numbers
..\.venv\Scripts\python.exe evaluate.py --study robust --trials 20   # Part E table (also: cube, lqr, kf, dhat, all)
..\.venv\Scripts\python.exe calibrate.py                 # Part B, unchanged
..\.venv\Scripts\python.exe eval_perception.py           # Part B, unchanged
```

Fresh environment, from the repo root: `py -3.12 -m venv .venv` then
`.venv\Scripts\python.exe -m pip install -r "Track B\requirements.txt"` (scipy and matplotlib were added).

In the viewer: green see-through box = cube KF estimate, blue dot = current PBVS target for the pinch point, top-left
text = phase, estimate, 1 sigma, truth (eval only) and error. The separate "wrist camera" window shows the noisy RGB
the pipeline sees, the red-mask outline (yellow) and the KF estimate projected back into the image (green cross).
Live disturbance: double-click a body to select it, then Ctrl + right-drag to push it (Ctrl + left-drag rotates).
Space pauses. The camera phases run at roughly half real time on this laptop (rendering + noise model), the rest
runs in real time.

## Pipeline

```
encoders, 20 bit, 500 Hz ─────────────> joint KF [q, qd, d] ──────────────┬───────────────────────────────┐
                                                                          │ q_hat                         │ q_hat, qd_hat, d_hat
wrist RGB-D, 10 Hz -> d435_noise -> locate_cube --T_base_cam(q_hat)--> cube KF (NIS gate) -> pick task    │
                                                                          │ target pose                   │
                                                                          v                               v
                                                     PBVS: v = lam * e  -> DLS -> q_ref, qd_ref, qdd_ref -> LQR + integral
                                                                                                          │ v - d_hat
                                                                    tau = M(q_hat) v + h(q_hat, qd_hat) <─┘
                                                                          │
                                                                     UR5e torque motors (MuJoCo)
```

Everything after the sensors runs on estimates. Ground truth only appears in `truth_eval_only` /
`cube_truth_eval_only` (main.py), which feed the log and the scoring in evaluate.py, never the loop. The controller's
kinematics and dynamics come from `RobotModel`, a separately compiled copy of the scene evaluated at q_hat.

## Frames

```
world --Rz(180 deg)--> base --FK(q_hat)--> wrist_3_link --T_wrist_cam--> camera (OpenCV)     [perception.py]
                       base --FK(q_hat)--> gripper/pinch (TCP)                                [RobotModel.tcp]
```

The cube estimate, PBVS error, Jacobian and drop zone are all in the base frame. The drop zone is a fixed cell
location given in world coordinates (`DROP_XY`) and converted to base once.

## What changed from main

| File | Change |
|---|---|
| `simulator.py` | `build(..., object_yaw, motors)`: optional cube yaw; torque motors (same +/-150 / +/-28 N m limits) instead of the Menagerie position servos. Default call unchanged, so calibrate.py / eval_perception.py behave as before. `DROP_XY` constant. |
| `src/model.py` | New. State-space model helpers (c2d, ctrb, obsv, Gramians, PBH) and `RobotModel` (M, h, TCP pose, Jacobian at q_hat). |
| `src/estimation.py` | `JointKF` added. `KalmanFilter3D` gets `constant_velocity` and `q_pos` options and `nis()`; its default behaviour is unchanged. |
| `src/controller.py` | New (was empty). Discrete LQR with integral action, computed torque, PBVS, pick state machine. |
| `src/perception.py` | `locate_cube(..., with_yaw=True)` also returns yaw (minAreaRect of the top face in the base xy plane). `d435_noise` caches its constant wave pattern: output is bit-identical, about 25 ms faster per frame. |
| `main.py` | Full pipeline `run_trial`, viewer overlays, headless plots. Original helpers kept. |
| `model_analysis.py`, `evaluate.py`, `src/plots.py` | New. |

## Design and parameter values

| Block | Choice | Values | Why |
|---|---|---|---|
| Plant model | computed torque makes each joint q'' = v | x = [q, qd] per joint, 12 states, C = [I 0] | M(q) has large off-diagonals (0.45, 0.74 kg m^2 at HOME), so the raw joints are coupled; after computed torque they are 6 identical double integrators |
| Rates | physics, control, joint KF / camera | 500 Hz / 10 Hz | UR e-Series controller rate; camera limited by render + noise model |
| Encoders | MuJoCo jointpos + noise + quantisation | 20 bit, sigma = LSB * sqrt(1 + 1/12) = 6.2 urad | quantisation plus 1 LSB of noise, as in read_encoders |
| Joint KF | x = [q, qd, d], d = everything computed torque gets wrong | sigma_a = 0.5 rad/s^2, sigma_d = 20 rad/s^2/sqrt(s) | sigma_d from the kf study: 5 left a 7 mrad wobble at grasp, 50 doubled torque roughness |
| Cube KF | constant position (random walk) | q_pos = 1e-7 m^2 per 0.1 s, R = 4 x the eval_perception R, NIS gate 16.27 (chi2 3 dof, 99.9 %), re-init after 3 rejections | the CV model drifted up to 12.8 mm once the cube left the view; R x 1 gave NEES about 10 (overconfident) |
| LQR | per joint on [e, ed, int e], dlqr | Q = diag(1e6, 1111, 4e8), R = 1 (Bryson: 1 mrad, 0.03 rad/s, 5e-5 rad s, 1 rad/s^2 correction) | poles -23.4 +/- 16.8j and -24 rad/s, about 10x the PBVS loop, well below the estimator (about 400 rad/s) |
| Disturbance feedforward | v - d_hat | on | kills the gripper and push transients (dhat study) |
| PBVS | v = lam e on the pinch pose, capped and rate limited, damped least squares | lam = 3 1/s, 0.25 m/s (0.08 descending), 1 rad/s, 1 m/s^2, damping 0.02 | outer loop, 10x slower than the joint loop |
| Grasp | gripper z down, fingers along a cube face normal | yaw from minAreaRect, averaged on the 90 deg circle | yaw error 0.23 deg mean, 0.55 deg max over 20 rotated cubes |

## Results

All from `python evaluate.py --study all` (20 trials per row, seeds 1000 to 1019, the same seeds in every row so rows
are paired; about 7 min), run on 30 Oct with the committed `K_calib.npy`. Full tables with every column are written
to `results/eval_*.md`; plots of one trial (estimate +/- 2 sigma against truth) to
`results/seed*_joint_kf.png`, `seed*_cube_kf.png`, `seed*_control.png`.

### Robustness (Part E), 20 trials per row, 200 in total

| condition | picked | placed | grasp offset mm | cube est err mm | cube NEES (3) | NIS rejects | q err urad | qd err mrad/s | joint NEES (2) | track rms mrad | track max mrad | track max grasp/lift mrad | torque rms % | torque rough % | time s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nominal | 100% | 100% | 0.3 / 0.4 | 0.68 / 0.9 | 2.9 | 0.0 | 5.5 | 2.51 | 1.58 | 0.276 | 0.14 | 2.13 | 8.3 | 0.688 | 17.0 |
| cube yaw +/-45 deg | 100% | 100% | 0.3 / 0.4 | 0.61 / 0.9 | 2.8 | 0.0 | 5.5 | 2.52 | 1.59 | 0.263 | 0.16 | 1.99 | 8.3 | 0.684 | 17.0 |
| camera noise 3x | 100% | 100% | 0.1 / 0.2 | 1.20 / 1.5 | 7.5 | 0.0 | 5.5 | 2.51 | 1.58 | 0.277 | 0.14 | 2.15 | 8.3 | 0.687 | 17.0 |
| camera noise 10x | 100% | 100% | 2.4 / 2.8 | 3.28 / 4.3 | 53.1 | 0.9 | 5.5 | 2.53 | 1.59 | 0.251 | 0.14 | 2.11 | 8.3 | 0.690 | 17.0 |
| 14-bit encoders | 100% | 100% | 0.3 / 0.4 | 0.68 / 0.9 | 2.9 | 0.0 | 219.6 | 22.07 | 1.41 | 1.226 | 0.83 | 14.93 | 8.7 | 2.759 | 17.0 |
| model masses +20% | 100% | 100% | 0.3 / 0.4 | 0.68 / 0.9 | 2.9 | 0.0 | 5.5 | 2.52 | 1.59 | 0.252 | 0.26 | 1.94 | 8.3 | 0.810 | 17.0 |
| 30 N push while descending | 100% | 100% | 0.3 / 0.4 | 0.65 / 0.9 | 2.9 | 0.0 | 5.5 | 2.51 | 1.58 | 0.279 | 0.59 | 2.12 | 8.3 | 0.691 | 17.0 |
| 40 N push while carrying | 100% | 100% | 0.3 / 0.4 | 0.68 / 0.9 | 2.9 | 0.0 | 5.5 | 2.52 | 1.58 | 0.285 | 0.98 | 2.13 | 8.3 | 0.694 | 17.0 |
| cube nudged 3 cm on approach | 100% | 100% | 0.3 / 0.4 | 0.66 / 0.9 | 87.7 | 3.0 | 5.5 | 2.51 | 1.58 | 0.276 | 0.14 | 2.13 | 8.3 | 0.689 | 17.0 |
| all of the above (3x noise) | 100% | 100% | 0.2 / 0.4 | 1.16 / 1.5 | 90.9 | 3.0 | 222.8 | 23.39 | 1.48 | 1.170 | 4.27 | 13.84 | 8.9 | 3.329 | 17.0 |

Notes: "track max" is the largest joint tracking error outside grasp/lift, the grasp/lift column is separate because
closing the gripper is the biggest disturbance in every trial. Cube NEES is about 3 nominally (consistent) but 7.5 at
3x and 53 at 10x noise, because R was not scaled with the injected noise; the nudge rows' NEES is the few frames
before three NIS rejections restart the filter. 14-bit encoders raise q error 40x and torque roughness 4x, but the KF
R is set from the bit count so it stays consistent (NEES 1.4).

### Trade-offs, 20 trials per row

Cube filter model and R (Part C):

| cube KF | picked | est. error at grasp, mean / max mm | grasp offset mm | NEES (expect 3) |
|---|---|---|---|---|
| constant velocity (original), R x 1 | 100% | 6.44 / 12.8 | 1.6 / 3.1 | 7.5 |
| constant velocity, cube nudged 3 cm | 90% | 6.72 / 13.9 | 1.7 / 5.6 | 33.3 |
| constant position, R x 0.1 | 100% | 0.81 / 0.9 | 0.3 / 0.5 | 81.5 |
| constant position, R x 1 | 100% | 0.69 / 0.9 | 0.3 / 0.5 | 9.5 |
| **constant position, R x 4 (used)** | 100% | 0.68 / 0.9 | 0.3 / 0.4 | 2.9 |
| constant position, R x 10 | 100% | 0.64 / 0.9 | 0.3 / 0.5 | 1.5 |
| constant position, R x 4, nudged | 100% | 0.66 / 0.9 | 0.3 / 0.4 | 87.7 |

LQR control weight R, with the 40 N push (Part D, `results/tradeoff_lqr.png`):

| R | track rms mrad | track max mrad | grasp/lift max mrad | torque roughness % |
|---|---|---|---|---|
| 0.1 | 0.137 | 0.58 | 1.69 | 0.904 |
| 0.3 | 0.192 | 0.75 | 1.58 | 0.776 |
| **1 (used)** | 0.285 | 0.98 | 2.13 | 0.694 |
| 3 | 0.446 | 1.22 | 5.35 | 0.646 |
| 10 | 0.436 | 1.58 | 3.47 | 0.603 |

Joint KF disturbance noise sigma_d, with the 40 N push (Part C, `results/tradeoff_kf.png`):

| sigma_d | qd err mrad/s | joint NEES (expect 2) | grasp/lift max mrad | torque roughness % |
|---|---|---|---|---|
| 2 | 2.36 | 2.93 | 7.32 | 0.196 |
| 5 | 1.92 | 1.84 | 7.50 | 0.308 |
| **20 (used)** | 2.52 | 1.58 | 2.13 | 0.694 |
| 50 | 3.85 | 1.72 | 1.63 | 1.443 |
| 200 | 15.25 | 5.63 | 0.65 | 10.810 |

Disturbance feedforward v - d_hat:

| case | d_hat | track rms mrad | track max mrad | grasp/lift max mrad | torque roughness % |
|---|---|---|---|---|---|
| nominal | on | 0.276 | 0.14 | 2.13 | 0.688 |
| nominal | off | 1.133 | 0.14 | 12.78 | 0.187 |
| 40 N push | on | 0.285 | 0.98 | 2.13 | 0.694 |
| 40 N push | off | 1.357 | 7.58 | 12.78 | 0.193 |
| model masses +20% | on | 0.252 | 0.26 | 1.94 | 0.810 |
| model masses +20% | off | 1.063 | 1.72 | 11.97 | 0.217 |

### Regression checks

- `eval_perception.py` in a fresh environment (30 Oct): 50/50 detected at 0x/1x/3x/10x, mean 0.98 / 1.19 / 1.28 /
  2.15 mm, the README table exactly.
- `evaluate.py` is deterministic on our PC for a given `K_calib.npy`: the same trials run serially, in parallel and repeated
  give bit-identical results.
- `calibrate.py` gives fx 433.16 (the README value), but the K it writes differs from the committed `K_calib.npy` by
  up to 0.0006 px. Every later trial reads that file, and the trials are sensitive enough that `evaluate.py`'s tables
  then move by up to about 0.15 mm (camera noise 3x: 1.20 mm mean estimate error with the committed K, 1.16 mm with
  the rewritten one). Run `git restore "Track B/K_calib.npy"` after `calibrate.py` to reproduce the tables above.

## Assumptions and limits

- The arm is torque controlled in sim. On the real UR the PBVS joint reference would go to the robot's own joint
  servo instead of the computed-torque loop.
- The drop zone is a known fixed location, not detected (it is clipped at the image edge from HOME).
- No camera latency or motion blur; the image is rendered at the instant it is used.
- The grasp is timed (0.8 s), there is no gripper force or object-in-hand sensing.
- Camera mount and link geometry known exactly (same as the perception assumptions in the README).
