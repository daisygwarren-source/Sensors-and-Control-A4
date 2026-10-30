# Track B hardware: UR3e pick

Scripts for the lab UR3e with an OnRobot RG2 gripper and a RealSense D435. They reuse the simulation's perception,
Kalman filters and LQR from `Track B/src`, with the UR3e's velocity-input joint model in place of computed torque.

## What you need

- The lab UR3e and RG2 on Ethernet, and the D435 on a USB 3 port.
- Windows with Python 3.12 and Git (what we used in the lab).
- The lecturer's starter code, [UTS-RI/UR3_RG2_rosbridge](https://github.com/UTS-RI/UR3_RG2_rosbridge), for the robot
  backends and the MuJoCo UR3e model used by `--sim`.

The scripts look for the starter code and write their data next to the repo folder, not inside it (`ROOT`, `STARTER`
and `--out` in `real.py`):

```
<parent folder>\
    Sensors-and-Control-A4\               this repo
    _reference\UR3_RG2_rosbridge\         starter code (main branch, commit 757d4a7)
    hw-data\<date>\                       captures, logs and config.json (default --out)
    hw-data\sim\config.json               placeholder config for --sim runs
```

## Install

From the repo root, in PowerShell:

```
git clone https://github.com/UTS-RI/UR3_RG2_rosbridge.git ..\_reference\UR3_RG2_rosbridge
git -C ..\_reference\UR3_RG2_rosbridge checkout 757d4a7
py -3.12 -m venv ..\ur3
..\ur3\Scripts\Activate.ps1
python -m pip install -r "Track B\hardware\requirements.txt"
```

`requirements.txt` has the versions from the lab laptop. It also installs two packages from Git, pinned by commit:
`realsense_lab` (the camera helpers on the starter repo's `RealSense-Lab` branch) and `onRobot` (the RG2 driver the
starter README installs). It is separate from `Track B\requirements.txt` because the starter code was run with
MuJoCo 3.17.

Set this in every new PowerShell window. Robotics Toolbox's URDF loader needs it, and `real.py` stops with a message
if it is missing:

```
$env:PYTHONUTF8 = "1"
```

## Checks without the robot

```
python "Track B\hardware\real.py"                          # ends with "kinematics OK"
python "Track B\hardware\real.py" --write-sim-config       # writes ..\hw-data\sim\config.json for the --sim runs
python "Track B\hardware\calib_handeye.py" --selftest      # four lines ending "pass"
python "Track B\hardware\real_joint_control.py" --sim      # RMS tracking errors under 1 mrad, about 25 s
python "Track B\hardware\real_pick.py" --sim --hover 0     # "1 of 1 trials finished", about 1 to 2 min
```

## Each lab session

```
$env:PYTHONUTF8 = "1"
$ip = "ROBOT_IP"                                     # from the label next to the arm
$d = "..\hw-data\$(Get-Date -Format yyyy-MM-dd)"     # the scripts' default --out
$cfg = "$d\config.json"
New-Item -ItemType Directory -Force $d
Copy-Item "Track B\results\hardware\2026-10-30\config.json" $cfg
```

The copied config keeps the values measured on earlier days: the panel's TCP (`T_tool0_tcp`), the cube size (`half`,
`object_width`), `grip_time`, the joint model (`tau`, `td`, `sigma_q`, `sigma_qd`) and the camera matrix `K`. Change
them if the tool, the object or the camera changes (`update_config` in `real.py` merges keys into the file). The joint
model comes from `step_test.py` on each joint and 30 s of `log_joints.py`, fitted by `fit_steps.py $d`. The steps
below overwrite the rest. Every script prints what it saved and which config keys it wrote.

Before any script moves the arm: panel in Remote mode, area clear, one hand on Ctrl+C and the panel's stop.

## Run order

**1. Hand-eye calibration**: where the camera is relative to the robot. ArUco marker `DICT_4X4_50` id 0, 100 mm. With
a fixed camera the marker goes on the gripper, with a wrist camera it goes on the table. Capture 15 or more poses with
large wrist rotations about different axes (Space saves a frame when the marker is found).

```
python "Track B\hardware\capture.py" --label handeye --ip $ip
python "Track B\hardware\calib_handeye.py" FOLDER --config $cfg --mount fixed
```

`FOLDER` is the folder `capture.py` printed. Writes `T_base_cam` (fixed) or `T_tool0_cam` (wrist), `mount` and
`handeye_rms_px`. Expect a reprojection RMS around 1 px or under (0.80 px on 30 Oct).

**2. Marker check**: separates hand-eye error from localisation error. Lay marker 0 flat on the table. In the capture
window press Space with the arm out of view, then `t` with the TCP touching the marker's centre. Repeat at several
places on the table.

```
python "Track B\hardware\capture.py" --label object --ip $ip
python "Track B\hardware\marker_check.py" FOLDER --config $cfg --marker 0.1
```

Prints where the camera puts the marker's centre against the TCP touch. No colour mask or depth is involved, so this
is the camera pose's error alone.

**3. Objects**: colour mask, localisation error, table height and work area. One capture run:

- The first capture is an image with the arm at the look pose (out of the camera's view for a fixed camera). This pose
  becomes `look_q`, where every pick starts and ends.
- For 6 or more cube positions spread over the table: Space for the image, then `t` with the TCP at the cube's centre.
- `b` with the fingertips touching the table.
- `c` at two opposite corners of the safe work area.

```
python "Track B\hardware\capture.py" --label object --ip $ip
python "Track B\hardware\eval_objects.py" FOLDER --config $cfg --set-offset --plot
```

It checks the kinematics against the robot's own TCP (wait for `kinematics OK`), opens the mask tuner (`n` / `p` next
and previous image, `s` save, Esc quits without saving), then prints the localisation error per pair, with the offset
fitted on half the pairs and scored on the other half. Writes `hsv`, `table_z`, `workspace`, `look_q` and
`loc_offset`, and the error plot to `FOLDER\eval_objects.png`.

**4. Pick**:

```
python "Track B\hardware\real_pick.py" --sim --config $cfg --hover 0 --out "$d\sim"         # sim on today's config first
python "Track B\hardware\real_pick.py" --ip $ip --config $cfg                                # dry run, stops 5 cm above the cube
python "Track B\hardware\real_pick.py" --ip $ip --config $cfg --hover 0 --trials 5          # picks
python "Track B\hardware\real_pick.py" --ip $ip --config $cfg --hover 0 --trials 3 --track  # camera also tracks while moving
```

The script refuses to start if a config key is missing or if its kinematics disagree with the robot's TCP by more than
5 mm. Each trial prints the phases, each look (`found 8/8, estimate ...`) and the grip width (`holding` or `missed`),
and the run ends with `N of N trials finished`. A cube estimate outside the arm's reach or the work area fails the
trial before the arm moves. `--track` only works with a fixed camera. Logs go to `$d\pick_<time>\`.

**5. Plots**:

```
python "Track B\hardware\plot_logs.py" $d
```

Writes the plots and `tier3_results.md` (one row per trial) to `$d\plots`. There is no ground truth on hardware, so
each plot says which reference it compares against.

Joint control alone (Tier 2, Part D): `python "Track B\hardware\real_joint_control.py" --ip $ip --config $cfg --delta 0
0 0.1 0 0 0.1` moves the arm out by `delta` and back with the KF and LQR, and logs the estimate and its uncertainty.

## Results (30 Oct)

In `Track B/results/hardware/2026-10-30/`:

- `plots/tier3_results.md`: every trial. 29 of 32 picks succeeded. The successful ones took 24.5 to 43.2 s, with the
  gripper closing at 59.9 to 61.2 mm on the 61 mm cube. Two failures were cube estimates past the arm's reach or off the work area,
  refused before moving. One was a `UR speedJ command failed` error.
- `object_20261030_120339/eval_objects.png`: localisation error against the TCP touches. Raw |xy| error 11.7 mm mean
  (26.0 mm max). With the offset subtracted, 6.6 mm in-sample and 10.6 mm held out.
- `plots/pick_20261030_124832_trial3_object.png`: a tracking pick where the cube was moved during the approach. The NIS
  gate rejects the jumps, the filter restarts on the new position, and the arm approaches again (3 times) and picks it.
- `plots/pick_20261030_135113_trial1_*.png`: one pick from a 5-of-5 run.
- `plots/joint_control_20261030_134701_*.png`: Tier 2 joint tracking and the joint KF.
- `config.json`: the calibration used that day, including `handeye_rms_px` = 0.80 and the factory `K`.

## Limitations

- The grasp check uses width only, because the RG2 driver has no grip-detected signal. `real_pick.py` calls a grasp
  good when the closed width is more than 5 mm above the grip target and less than 10 mm over the object width. That
  catches fingers closing on nothing, stopping on a corner, or not closing. It does not check again after the lift, so
  a cube that slips out later is not detected.
- Hardware has no ground truth. Localisation error is measured against TCP touches, so it includes the touch error and
  the robot's own kinematics.
- The starter code and data folders sit at fixed places next to the repo folder (see the layout above).
- `requirements.txt` was taken from the lab laptop's environment. It has not been installed from scratch on a second
  machine.
