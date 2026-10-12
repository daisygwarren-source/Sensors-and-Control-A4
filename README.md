# Sensors-and-Control-A4
This assessment 

# Track A, mobile robot:

 visual waypoint navigation. TurtleBot3 with 2D LiDAR and camera
in MuJoCo. Navigate a room with obstacles using the LiDAR and visually dock to the ArUco-marked
charging station. Required: state-space model of the platform; EKF fusing wheel-encoder odometry
with range and bearing, or pose, measurements from the camera; state-feedback or LQR trajectory
tracker. Constraint: the controller must use the estimated state, never ground truth.
# Track B, manipulator: 

Vision-guided pick. UR3 in MuJoCo with a wrist-mounted RGB-D
camera. Detect a target object on a tabletop, plan an approach and execute a pick using IBVS or
PBVS. Required: camera calibration; depth-based object localisation; state-space joint controller;
1
41014 Sensors and Control in Mechatronics Systems Integrated Project Marking Guide
observer for unmeasured states. Constraint: object position is randomised between trials and the
system must demonstrate robustness.
Own project. Must include, at minimum, a state-space model with controllability and observability
analysis, at least one calibrated exteroceptive sensor, a Kalman filter or observer, and a feedback
controller

# Packages 

pip install mujoco mujoco-viewer
pip install mujoco

***Documentation for Mujoco:*** https://mujoco.readthedocs.io/en/stable/programming/simulation.html

# Track B perception: calibration and cube localisation

Code: `Track B/src/perception.py`, `Track B/calibrate.py`, `Track B/eval_perception.py`. Input: one wrist RGB-D frame and the six joint-encoder readings. Output: the cube centre in the robot base frame, or `None` if the cube is not fully in view.

## Run

```
cd "Track B"
pip install -r requirements.txt
python calibrate.py          # calibrate K from 40 checkerboard views, writes K_calib.npy
python main.py --headless    # one randomised trial: noisy capture -> encoders -> FK -> locate_cube
python eval_perception.py    # 50 randomised trials at 0x, 1x, 3x and 10x noise
```

## Frames

```
world --Rz(180 deg)--> base --FK(q)--> wrist_3_link --T_wrist_cam--> camera (OpenCV: x right, y down, z forward)
```

- `base` is the UR5e base body. It is rotated 180 deg about z relative to world (`ur5e.xml:64`), so the table centre is at x = -0.5 m in base coordinates.
- `FK(q)` is the product of the six link transforms in `ur5e.xml`: a fixed offset (including the 90 deg body rotations at the shoulder-lift and wrist-1 links), then a rotation about the joint axis. It matches MuJoCo's wrist pose to 1e-15 over 200 random configurations.
- `T_wrist_cam` is the fixed camera mount from `simulator.py`: 0.10 m along wrist y, 0.08 m along wrist z, optical axis along wrist +y. MuJoCo cameras look along -z with y up, so converting to the OpenCV convention flips y and z.
- `q` comes from MuJoCo `jointpos` sensors (`enc_*` in `simulator.py`), quantised to 20 bits with 1 LSB of Gaussian noise (`read_encoders` in `main.py`).

## Calibration

Zhang's method (OpenCV `calibrateCamera`) on 40 simulated views of a 9 x 7 checkerboard with 20 mm squares, with camera RGB noise applied. Each image is blurred (5 x 5 Gaussian) before corner detection, and views whose reprojection error is above twice the median are dropped before the final fit.

| | fx | fy | cx | cy |
|---|---|---|---|---|
| Calibrated (25 of 40 views, RMS 0.115 px) | 433.16 | 433.21 | 319.53 | 239.81 |
| True (from MuJoCo `fovy`, evaluation only) | 432.97 | 432.97 | 320 | 240 |

Lens distortion is fixed at zero because MuJoCo renders an ideal pinhole camera. With the coefficients left free, the solver overfit them (k3 = 0.29) and biased K. On a real camera, call `calibrate(..., fix_distortion=False)`.

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
2. Reject the detection if the mask touches the image border. A clipped cube gave errors of up to 35 mm.
3. Erode the mask by 2 px to drop edge pixels, where depth is unreliable, and drop invalid (zero) depth.
4. Back-project a ring of table pixels 5 to 15 px around the cube into the base frame. Their median height minus the known table height is the local depth bias, which is removed from the cube's depth.
5. Back-project the cube pixels, find the top face with an iterated median height band (+/-10 mm), take the median x and y, and subtract half the cube size from z.

## Assumptions

- The table is flat, horizontal and at a known height (`table_z = 0.04` m, `TABLE_TOP_Z` in `simulator.py`). The depth-bias correction in step 4 only works if this holds.
- The cube is upright, resting on the table, of known size (`half = 0.02` m), and the only red object in view.
- The depth error varies smoothly over a few centimetres, so the table ring next to the cube has the same bias as the cube.
- The camera looks roughly straight down, as it does at `HOME`. The bias correction uses the z-component of the camera's optical axis.
- The camera mount and link geometry are known exactly (from `simulator.py` and `ur5e.xml`). No hand-eye calibration error is modelled.
- Encoder resolution is 20 bits. UR does not publish the figure; UR joints use RLS AksIM absolute encoders, a family rated up to 20 bits [2]. Sensitivity: worst camera position error 0.018 mm at 20 bits, 0.16 mm at 16 bits and 0.57 mm at 14 bits.

## Results (`eval_perception.py`, 50 randomised trials, calibrated K)

| Noise | Detected | Mean error | Max error | Std x / y / z |
|---|---|---|---|---|
| none | 50/50 | 1.70 mm | 4.07 mm | 1.03 / 1.27 / 0.28 mm |
| 1x | 50/50 | 1.73 mm | 3.54 mm | 1.03 / 1.27 / 0.21 mm |
| 3x | 50/50 | 1.76 mm | 3.37 mm | 1.00 / 1.26 / 0.38 mm |
| 10x | 50/50 | 2.32 mm | 4.23 mm | 1.16 / 1.34 / 1.28 mm |

With the same noise model, the first version's max-height top-face filter reached 26 mm mean and 50 mm max error at 10x (44 trials, true K). The 1x standard deviations give the measurement noise R for the Kalman filter. Ground truth (`cube_truth_eval_only` and `intrinsics` in `main.py`) is used only to score these results.

[1] Intel RealSense, "BKM for Tuning D435 and D415 Cameras" (webinar, rev 3): https://forums.intel.com/cipcp26785/attachments/cipcp26785/feeditem_no_topic/6363/1/BKM-For-Tuning-D435-and-D415-Cameras-Webinar_Rev3.pdf

[2] Renishaw / RLS, AksIM encoders in Universal Robots joints: https://www.renishaw.com/media/pdf/en/b439965d6aa648e0b5c0044c098c42e3.pdf