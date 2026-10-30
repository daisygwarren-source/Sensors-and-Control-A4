# Track B hardware: UR3e pick

Scripts for a lab UR3e with an OnRobot RG2 gripper and a RealSense camera. We used a D435 on 28 Oct and a D455 on a
different UR3e on 30 Oct, both fixed in the cell. The scripts reuse the simulation's perception, Kalman filters and
LQR from `Track B/src`, with the UR3e's velocity-input joint model in place of computed torque.

## What you need

- A lab UR3e and RG2 on Ethernet, and the RealSense camera on a USB 3 port.
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
    hw-data\sim\                          placeholder config and logs for --sim runs
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
starter README installs), plus `pycurl`, which `onRobot` imports but doesn't declare. It is separate from
`Track B\requirements.txt` because the starter code was run with MuJoCo 3.17.

Set this in every new PowerShell window. Robotics Toolbox's URDF loader needs it, and `real.py` stops with a message
if it is missing:

```
$env:PYTHONUTF8 = "1"
```

## Checks without the robot

```
python "Track B\hardware\real.py"                                              # ends with "kinematics OK"
python "Track B\hardware\real.py" --write-sim-config                           # writes ..\hw-data\sim\config.json for the --sim runs
python "Track B\hardware\calib_handeye.py" --selftest                          # four lines ending "pass"
python "Track B\hardware\real_joint_control.py" --sim --out ..\hw-data\sim     # RMS tracking errors under 1 mrad, about 25 s
python "Track B\hardware\real_pick.py" --sim --hover 0 --out ..\hw-data\sim    # "1 of 1 trials finished", about 1 to 2 min
```

Without `--out`, the `--sim` runs write into today's `hw-data\<date>` folder, next to the real logs.

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

That config was saved with `sigma_t` = 0, but most picks on 30 Oct from 125141 onwards (all except 133640 and the
134123 dry run) ran with 0.0009. That cut the share of clipped control cycles from 13 to 82% to under 2%. Each pick
log records the config it used. To pick with it:

```
python -c "import sys; sys.path.insert(0, 'Track B/hardware'); from real import update_config; update_config(sys.argv[1], sigma_t=0.0009)" $cfg
```

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
`handeye_rms_px`. For a fixed camera it also stores the fit as `T_base_cam_fit` and resets `T_base_cam_dz` to 0
(step 3). Expect a reprojection RMS around 1 px or under (0.80 px on 30 Oct).

**2. Marker check**: separates hand-eye error from localisation error. Lay marker 0 flat on the table. In the capture
window press Space with the arm out of view, then `t` with the fingertips touching the marker's centre. Repeat at
several places on the table.

```
python "Track B\hardware\capture.py" --label object --ip $ip
python "Track B\hardware\marker_check.py" FOLDER --config $cfg --marker 0.1
```

Prints where the camera puts the marker's centre against the TCP at each touch, and the mean (`marker bias`). No
colour mask or depth is involved, so this is the camera pose's error alone. The z bias should be minus the distance
from the TCP down to whatever touched the marker. On 30 Oct that was -10 mm (fingertips at 214 mm, TCP at 204 mm).

**3. Camera height correction**, if the marker check's z bias is off. On 30 Oct it read -29 mm against -10 mm
expected, so the hand-eye fit had placed the camera about 20 mm too low. A separate check of the table depth around
the cube also gave -20 mm. Because `locate_cube` corrects depth along the camera ray, that height error became a
-36 mm y bias in the cube estimates. Raising the camera by `DZ` mm (here 20):

```
python -c "import json, sys; sys.path.insert(0, 'Track B/hardware'); from real import update_config; c = json.load(open(sys.argv[1])); T0 = c.get('T_base_cam_fit', c['T_base_cam']); T = [r[:] for r in T0]; T[2][3] += float(sys.argv[2]) / 1000; print('camera z', round(T0[2][3], 4), '->', round(T[2][3], 4)); update_config(sys.argv[1], T_base_cam=T, T_base_cam_fit=T0, T_base_cam_dz=float(sys.argv[2]) / 1000)" $cfg 20
```

`DZ` = expected z bias - measured z bias, positive raises the camera. It always shifts `T_base_cam_fit`, the
hand-eye result, so running it again with a new value replaces the correction instead of adding to it. On 30 Oct it
took the raw y bias from -36 to -10 mm and the held-out localisation error from 15.0 to 10.6 mm (step 4).

**4. Objects**: colour mask, localisation error, table height and work area. One capture run:

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

**5. Pick**:

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

**6. Plots**:

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
  gripper closing at 59.9 to 61.2 mm on the 61 mm cube. Two failures were cube estimates past the arm's reach or off
  the work area, refused before moving. In the third (`pick_20261030_124107`, trial 3), the cube was moved during the
  descent, after the last camera check. A finger hit it, the robot made a protective stop (C157), and the gripper never
  closed (it read 101 mm). The grasp check at the time counted that as holding, and the next motion command failed
  (`UR speedJ command failed`). The current check rejects 101 mm (see Limitations).
- `object_20261030_120339/eval_objects.png`: localisation error against the TCP touches, with the 20 mm camera height
  correction. Raw |xy| error 11.7 mm mean (26.0 mm max). With the offset subtracted, 6.6 mm in-sample and 10.6 mm
  held out.
- `plots/pick_20261030_124832_trial3_object.png`: a tracking pick where the cube was pushed four times during the
  approach. The NIS gate rejects 22 measurements, the filter restarts twice, and the arm approaches again (3 times, the
  limit) and picks it.
- `plots/pick_20261030_135113_trial1_*.png`: one pick from a 5-of-5 run.
- `plots/joint_control_20261030_134344_*.png`: Tier 2 baseline, joint tracking and the joint KF. The other Tier 2 runs
  that day: 134411 without d̂ feedforward, 134633 and 134701 with the measured velocity fed to the KF (σ_qd 1e-4 and
  1e-3), and 134434, aborted when the RTDE data stopped.
- `config.json`: the calibration used that day, including `handeye_rms_px` = 0.80, the factory `K` and the camera
  height correction (`T_base_cam_fit`, `T_base_cam_dz` = 0.02).

## Limitations

- The grasp check uses width only, because the RG2 driver has no grip-detected signal. `real_pick.py` calls a grasp
  good when the closed width is more than 5 mm above the grip target and less than 10 mm over the object width. That
  catches fingers closing on nothing, stopping on a corner, or not closing. It does not check again after the lift, so
  a cube that slips out later is not detected.
- Hardware has no ground truth. Localisation error is measured against TCP touches, so it includes the touch error and
  the robot's own kinematics.
- The camera height correction is a single number taken from the marker check, not part of the hand-eye fit.
- The starter code and data folders sit at fixed places next to the repo folder (see the layout above).
- `requirements.txt` was taken from the lab laptop's environment. `pip install --dry-run` resolves it, but it has not
  been fully installed from scratch on a second machine.
