# Sensors-and-Control-A4
This assessment 

# Track A, mobile robot:

 visual waypoint navigation. TurtleBot3 with 2D LiDAR and camera
in MuJoCo. Navigate a room with obstacles using the LiDAR and visually dock to the ArUco-marked
charging station. Required: state-space model of the platform; EKF fusing wheel-encoder odometry
with range and bearing, or pose, measurements from the camera; state-feedback or LQR trajectory
tracker. Constraint: the controller must use the estimated state, never ground truth.
# Track B, manipulator: 

Vision-guided pick. The brief names a UR3; this project uses a UR5e (MuJoCo Menagerie model) in MuJoCo with a wrist-mounted RGB-D
camera. Detect a target object on a tabletop, plan an approach and execute a pick using IBVS or
PBVS. Required: camera calibration; depth-based object localisation; state-space joint controller;
observer for unmeasured states. Constraint: object position is randomised between trials and the
system must demonstrate robustness.
Own project. Must include, at minimum, a state-space model with controllability and observability
analysis, at least one calibrated exteroceptive sensor, a Kalman filter or observer, and a feedback
controller

# Packages 

pip install -r "Track B/requirements.txt"

MuJoCo's viewer is built in (`mujoco.viewer`), so no separate viewer package is needed.

***Documentation for Mujoco:*** https://mujoco.readthedocs.io/en/stable/programming/simulation.html

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
