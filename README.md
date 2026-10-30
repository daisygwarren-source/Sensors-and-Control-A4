# Sensors and Control A4: vision-guided pick (Track B)

41014 Sensors and Control, Assessment 4 integrated project.

**Group 9** (Canvas group "Project 9"): Charlie White, Sarah Gill, Daisy Warren.

## What it does

A robot arm finds a red cube on a table with an RGB-D camera, picks it up and moves it. The cube starts in a new random
place every trial. We built the system in simulation first, then ran it on the lab's real arm.

- **Simulation:** a UR5e (MuJoCo Menagerie model) with a Robotiq 2F-85 gripper and a wrist-mounted RGB-D camera, in
  MuJoCo. The camera adds RealSense D435-style depth noise. The arm carries the cube to a fixed drop zone.
- **Hardware:** the lab UR3e with an OnRobot RG2 gripper and a RealSense D435 fixed in the cell. The arm lifts the
  cube and puts it back where it was.

Both run the same loop:

1. Calibrated camera and joint encoders give the cube's position in the robot base frame (`src/perception.py`).
2. A Kalman filter tracks the cube position and rejects outlier measurements with an NIS gate (`src/estimation.py`).
3. A Kalman filter per joint estimates position, velocity and a lumped disturbance (`src/estimation.py`).
4. Position-based visual servoing (PBVS) drives the gripper to the grasp pose, and an LQR with integral action tracks
   the joint reference (`src/controller.py`). In simulation the LQR output goes through computed torque. On the UR3e
   it is a joint velocity command.
5. A state machine runs the task: look, approach, refine, descend, grasp, lift, transfer, lower, release, retreat.

The controller only ever uses estimates. Simulator ground truth is used only to score results, and every function
that reads it is named `*_eval_only`: `truth_eval_only` and `cube_truth_eval_only` in `Track B/main.py`, and
`in_base_eval_only` in `Track B/eval_perception.py`. `intrinsics()` in `main.py` (the true camera matrix) is only
printed by `calibrate.py` for comparison.

## Repository layout

| Path | Contents |
|---|---|
| `Track B/main.py` | Simulation demo and one full trial (`run_trial`) |
| `Track B/simulator.py` | MuJoCo scene: UR5e, gripper, wrist camera, table, cube |
| `Track B/src/` | Perception, estimation (Kalman filters), controller (LQR, PBVS, pick task), state-space model, plots |
| `Track B/model_analysis.py` | State-space model, controllability and observability (Part A) |
| `Track B/calibrate.py`, `Track B/eval_perception.py` | Camera calibration and perception evaluation (Part B) |
| `Track B/evaluate.py` | Randomised trials and trade-off studies (Parts C to E) |
| `Track B/hardware/` | UR3e scripts, see [`Track B/hardware/README.md`](Track%20B/hardware/README.md) |
| `Track B/results/` | Tables and plots written by the scripts. `results/hardware/<date>/` holds each lab day's config, logs and plots |
| `Track B/PIPELINE.md` | Pipeline design notes and the full simulation results tables |
| `Track B/universal_robots_ur5e/`, `Track B/robotiq_2f85/` | MuJoCo Menagerie models (licences in each folder) |

## Install

Tested on Windows 11 with Python 3.12, in PowerShell:

```
git clone https://github.com/daisygwarren-source/Sensors-and-Control-A4.git
cd Sensors-and-Control-A4
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r "Track B\requirements.txt"
```

If PowerShell blocks `Activate.ps1`, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first, or use
`.venv\Scripts\activate.bat` in Command Prompt. On Linux or macOS (not tested) use `python3.12 -m venv .venv` and
`source .venv/bin/activate`. MuJoCo's viewer comes with the `mujoco` package.

## Run

From the repo root, with the environment activated:

```
cd "Track B"
python main.py
```

This opens the MuJoCo viewer and a "wrist camera" window, and runs randomised picks back to back until the viewer is
closed. In the viewer, the green see-through box is the cube Kalman filter's estimate, the blue dot is the current
PBVS target, and the top-left text shows the phase, the estimate with its 1 sigma, the true position (evaluation
only) and the error. The wrist camera window shows the noisy image the pipeline sees, the red mask outline (yellow)
and the estimate projected back into the image (green cross). The camera phases run at about half real time.

Other commands, all from `Track B` (times measured on our PC):

| Command | What to expect |
|---|---|
| `python main.py --yaw 45 --nudge` | Cube turned up to +/-45 deg, then moved 3 cm while the arm approaches. The filter rejects the jump, restarts on the new position and the arm re-targets |
| `python main.py --slide` | Cube slides about 5 cm sideways along the table during the approach |
| `python main.py --push` | 40 N shove on the wrist while carrying the cube |
| `python main.py --cam-noise 10` | 10x depth noise (+/-2 cm wavy depth error) |
| `python main.py --headless --seed 0` | One trial with no window, about 10 s. Prints each phase, then a result line (`lifted`, `placed`, grasp and estimate errors), and saves `results/seed0_*.png` |
| `python model_analysis.py` | Part A: discrete model, controllability and observability ranks, PBH tests, closed-loop poles. About 1 s |
| `python calibrate.py` | Part B: calibrated K against the true K (fx 433.16 against 432.97). Overwrites `K_calib.npy`. About 3 s |
| `python eval_perception.py` | Part B: FK check, encoder sensitivity, 50 trials at four noise levels (the results table below). About 15 s |
| `python evaluate.py --study robust --trials 20` | Part E: 10 conditions x 20 trials, all picked and placed. About 2.5 min. Writes `results/eval_robust.md` |

`main.py --headless`, `calibrate.py` and `evaluate.py` overwrite committed files in `results/` and `K_calib.npy`.
`git restore .` puts them back.

## Hardware

The UR3e scripts need the lab robot, the camera and the lecturer's starter code. Setup and run order are in
[`Track B/hardware/README.md`](Track%20B/hardware/README.md). On 30 Oct, 29 of 32 pick trials succeeded
([`tier3_results.md`](Track%20B/results/hardware/2026-10-30/plots/tier3_results.md)): 15 of 17 with the camera
looking only while the arm is stopped, and 14 of 15 with the camera also tracking the cube while the arm moves. Two
failures were the script refusing a cube estimate outside the arm's reach or the work area before moving, and one was
a `UR speedJ command failed` error from the robot interface.

# Track B perception: calibration and cube localisation

Code: `Track B/src/perception.py`, `Track B/calibrate.py`, `Track B/eval_perception.py`. Input: one wrist RGB-D frame and the six joint-encoder readings. Output: the cube centre in the robot base frame, or `None` if the cube is not fully in view.

## Run

```
cd "Track B"
pip install -r requirements.txt
python calibrate.py                    # calibrate K from 40 checkerboard views, writes K_calib.npy
python calibrate.py --free-distortion  # same, with distortion coefficients free (prints only, does not save)
python main.py --headless              # one randomised trial: noisy capture -> encoders -> FK -> locate_cube
python eval_perception.py              # FK check, encoder sensitivity, 50 randomised trials at 0x, 1x, 3x and 10x noise
python eval_perception.py --study      # adds the band and erosion trade-off study
```

## Frames

```
world --Rz(180 deg)--> base --FK(q)--> wrist_3_link --T_wrist_cam--> camera (OpenCV: x right, y down, z forward)
```

- `base` is the UR5e base body. It is rotated 180 deg about z relative to world (`ur5e.xml:64`), so the table centre is at x = -0.5 m in base coordinates.
- `FK(q)` is the product of the six link transforms in `ur5e.xml`: a fixed offset (including the 90 deg body rotations at the shoulder-lift and wrist-1 links), then a rotation about the joint axis. `eval_perception.py` checks it against MuJoCo over 200 random configurations: the wrist pose matches to 1.0e-15 and the camera pose to 1.1e-15.
- `T_wrist_cam` is the fixed camera mount from `simulator.py`: 0.10 m along wrist y, 0.08 m along wrist z, optical axis along wrist +y. MuJoCo cameras look along -z with y up, so converting to the OpenCV convention flips y and z.
- `q` comes from MuJoCo `jointpos` sensors (`enc_*` in `simulator.py`), quantised to 20 bits with 1 LSB of Gaussian noise (`read_encoders` in `main.py`).

## Calibration

Zhang's method (OpenCV `calibrateCamera`) on 40 simulated views of a 9 x 7 checkerboard with 20 mm squares, with camera RGB noise applied. Each image is blurred (5 x 5 Gaussian) before corner detection, and views whose reprojection error is above twice the median are dropped before the final fit. The board is placed in front of the camera using the camera's simulated pose, which is scene setup only (like a person holding the board up); the calibration itself sees only the images.

| | fx | fy | cx | cy |
|---|---|---|---|---|
| Calibrated (25 of 40 views, RMS 0.115 px) | 433.16 | 433.21 | 319.53 | 239.81 |
| True (from MuJoCo `fovy`, evaluation only) | 432.97 | 432.97 | 320 | 240 |

Lens distortion is fixed at zero because MuJoCo renders an ideal pinhole camera. With the coefficients left free (`python calibrate.py --free-distortion`), the solver fits distortion the renderer doesn't have (k2 = 0.041, k3 = -0.127), and cx moves further from the truth (319.19, against 319.53 with distortion fixed). On a real camera, call `calibrate(..., fix_distortion=False)`.

## Camera noise model (`d435_noise`)

| Term | Value | Source |
|---|---|---|
| Random depth error | sigma = z^2 * subpixel / (f_ir * b) | Intel D400 tuning guide [1] |
| subpixel | 0.08 | Intel's typical value [1] |
| f_ir | 424 px | Intel's worked example: 848 px wide, 90 deg HFOV, f = 424 / tan(45 deg) [1] |
| b (stereo baseline) | 50 mm | D435, Intel's worked example [1] |
| Wavy systematic depth error | 2 mm amplitude, one period across the image | Assumed, to model the lecturer's "wavy wall" observation |
| Dropped depth at depth edges | 50% of edge pixels set to 0 (invalid) | Assumed |
| Depth quantisation | 1 mm | Assumed |
| RGB noise | sigma = 3 grey levels | Assumed |

At our working range (about 0.45 m) the random term is about 0.8 mm, Intel's best case for a well-textured surface. `eval_perception.py` also runs the model at 3x and 10x, scaling both `subpixel` and the wave. At 10x the wave is +/-2 cm (4 cm peak to peak) on top of 7.6 mm of random noise. That is our answer to the lecturer's warning that a RealSense can read a flat wall as a wavy surface up to about 5 cm thick.

## Localisation (`locate_cube`)

1. Threshold red in HSV, apply a morphological opening, and keep the largest connected component.
2. Reject the detection if the mask touches the image border, because the visible part of a clipped cube no longer centres on the cube.
3. Erode the mask by 1 px (3 x 3) to drop edge pixels, where depth is unreliable, and drop invalid (zero) depth.
4. Back-project a ring of table pixels 5 to 15 px around the cube into the base frame. Their median height minus the known table height is the local depth bias, which is removed from the cube's depth.
5. Back-project the cube pixels, find the top face with an iterated median height band (+/-5 mm), take the median x and y, and subtract half the cube size from z.

## Assumptions

- The table is flat, horizontal and at a known height (`table_z = 0.04` m, `TABLE_TOP_Z` in `simulator.py`). The depth-bias correction in step 4 only works if this holds.
- The cube is upright, resting on the table, of known size (`half = 0.02` m), and the only red object in view.
- The depth error varies smoothly over a few centimetres, so the table ring next to the cube has the same bias as the cube.
- The camera looks roughly straight down, as it does at `HOME`. The bias correction uses the z-component of the camera's optical axis.
- The camera mount and link geometry are known exactly (from `simulator.py` and `ur5e.xml`). No hand-eye calibration error is modelled.
- Encoder resolution is 20 bits. UR does not publish the figure; UR joints use RLS AksIM absolute encoders, a family rated up to 20 bits [2]. `eval_perception.py` measures the sensitivity: worst camera position error 0.011 mm at 20 bits, 0.139 mm at 16 bits and 0.608 mm at 14 bits.

## Results (`eval_perception.py`, 50 randomised trials, calibrated K)

| Noise | Detected | Mean error | Max error | Std x / y / z |
|---|---|---|---|---|
| none | 50/50 | 0.98 mm | 1.97 mm | 0.75 / 0.70 / 0.00 mm |
| 1x | 50/50 | 1.19 mm | 1.92 mm | 0.81 / 0.90 / 0.17 mm |
| 3x | 50/50 | 1.28 mm | 2.37 mm | 0.79 / 0.89 / 0.46 mm |
| 10x | 50/50 | 2.15 mm | 4.26 mm | 1.10 / 1.15 / 1.34 mm |

The 1x standard deviations give the measurement noise R for the Kalman filter. Ground truth (`cube_truth_eval_only` and `intrinsics` in `main.py`, `in_base_eval_only` in `eval_perception.py`) is used only to score these results.

## Band and erosion trade-off (`eval_perception.py --study`)

With no noise, the remaining error is a systematic bias, not noise: it points from the cube towards the point below the camera. Two settings in `locate_cube` cause most of it.

- **Top-face band.** Unless the cube is directly below the camera, the camera also sees one or two of its side faces. Side-face pixels within the band of the top surface are counted as top face and pull the estimate towards the camera, so a wider band means more bias.
- **Erosion.** Eroding the mask trims the top face's far edge, which borders the table, but not its near edge, which borders a side face in the same mask. The remaining top-face pixels are shifted towards the camera.

A narrow band has the opposite problem at high noise: too few pixels fall inside it and the iterated median wanders. Mean / max error in mm over the same 50 trials:

| Erosion | Band | 0x | 1x | 3x | 10x | Bias towards camera at 0x |
|---|---|---|---|---|---|---|
| 3 x 3 | +/-2 mm | 0.76 / 1.63 | 0.99 / 1.60 | 1.31 / 2.69 | 2.55 / 4.98 | 0.62 mm |
| 3 x 3 | +/-3 mm | 0.84 / 1.63 | 1.06 / 1.84 | 1.22 / 2.21 | 2.32 / 4.97 | 0.70 mm |
| **3 x 3** | **+/-5 mm (chosen)** | **0.98 / 1.97** | **1.19 / 1.92** | **1.28 / 2.37** | **2.15 / 4.26** | **0.86 mm** |
| 3 x 3 | +/-10 mm | 1.30 / 2.96 | 1.55 / 3.10 | 1.61 / 2.97 | 2.19 / 3.70 | 1.16 mm |
| 3 x 3 | +/-20 mm | 1.98 / 4.69 | 2.18 / 4.94 | 2.25 / 5.35 | 2.72 / 5.73 | 1.78 mm |
| none | +/-5 mm | 0.72 / 1.57 | 1.16 / 1.88 | 1.23 / 2.26 | 2.14 / 4.48 | 0.57 mm |
| 5 x 5 | +/-10 mm (previous) | 1.58 / 3.37 | 1.57 / 3.34 | 1.69 / 3.19 | 2.25 / 3.87 | 1.45 mm |

- **Band:** +/-5 mm gives the lowest mean error at 10x noise and stays within about 0.25 mm of the best band at lower noise. The previous +/-10 mm band with 5 x 5 erosion had a 1.58 mm mean error even with no noise.
- **Erosion:** no erosion gives slightly lower mean errors here (though a higher 10x max), because our noise model drops depth at edges (sets it to zero) instead of producing flying pixels, the mixed foreground/background depths that real stereo cameras return at object edges. A 3 x 3 erosion keeps one pixel of protection against those at a cost of about 0.3 mm of bias.
- **Unexplained part:** even with no erosion and a +/-2 mm band, 0.38 mm of bias towards the camera remains (full table from `--study`), which these two settings do not account for.

[1] Intel RealSense, "BKM for Tuning D435 and D415 Cameras" (webinar, rev 3): https://forums.intel.com/cipcp26785/attachments/cipcp26785/feeditem_no_topic/6363/1/BKM-For-Tuning-D435-and-D415-Cameras-Webinar_Rev3.pdf

[2] Renishaw / RLS, AksIM encoders in Universal Robots joints: https://www.renishaw.com/media/pdf/en/b439965d6aa648e0b5c0044c098c42e3.pdf
