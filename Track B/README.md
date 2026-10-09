# Task Description

**Platform:** UR5 6-DOF Manipulator with wrist RGB-D camera.
**Task:** Detect tabletop target, plan visual approach, and execute pick via IBVS/PBVS.
## Key Modules:
* Camera intrinsic calibration & 3D object detection/localization.
* State-space joint controller + observer for unmeasured joint velocities.
* Image-Based or Position-Based Visual Servoing (IBVS / PBVS).

**Constraint:** Target position randomized; system must prove robustness.


**Teams are encouraged to extend either track:**
* Add sensor noise.
* Implement active obstacle avoidance or velocity profiling.
* Compare LQR vs. PID or EKF vs. Unscented Kalman Filter (UKF).
* Explore different features in the perception space.