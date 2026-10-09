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