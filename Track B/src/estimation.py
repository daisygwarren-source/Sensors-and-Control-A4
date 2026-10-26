"""
State estimator for Track B.

Constant-velocity Kalman filter.

State:
    x = [px, py, pz, vx, vy, vz]

Measurement:
    z = [px, py, pz]

Uses localisation statistics measured in eval_perception.py.
"""

import numpy as np


class KalmanFilter3D:
    def __init__(self, dt=0.02, constant_velocity=True, q_pos=1e-6):
        self.dt = dt

        # State vector
        self.x = np.zeros((6, 1))

        # State transition matrix
        self.A = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])
        # static cube: position is a random walk, velocity states are left out of the prediction
        if not constant_velocity:
            self.A[:3, 3:] = 0

        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ])

        # Process noise
        self.Q = np.diag([
            q_pos,
            q_pos,
            q_pos,
            1e-4,
            1e-4,
            1e-4,
        ])

        # Measurement noise from eval_perception.py
        sigma_x = 0.00081
        sigma_y = 0.00090
        sigma_z = 0.00017

        self.R = np.diag([
            sigma_x**2,
            sigma_y**2,
            sigma_z**2,
        ])

        # Covariance matrix
        self.P = np.eye(6)

        self.initialized = False

    def initialize(self, position):
        position = np.asarray(position).reshape(3)

        self.x = np.array([
            [position[0]],
            [position[1]],
            [position[2]],
            [0.0],
            [0.0],
            [0.0],
        ])

        self.initialized = True

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, measurement):
        z = np.asarray(measurement).reshape(3, 1)

        innovation = z - self.H @ self.x

        S = self.H @ self.P @ self.H.T + self.R

        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ innovation

        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P

    def step(self, measurement):
        """
        One predict/update cycle.

        measurement:
            [x,y,z] from locate_cube()
            or None if cube not detected
        """

        if measurement is None:
            if self.initialized:
                self.predict()
                return self.x.flatten()
            return None

        if not self.initialized:
            self.initialize(measurement)
            return self.x.flatten()

        self.predict()
        self.update(measurement)

        return self.x.flatten()

    def nis(self, measurement):
        z = np.asarray(measurement).reshape(3, 1)
        S = self.H @ self.P @ self.H.T + self.R
        v = z - self.H @ self.x
        return (v.T @ np.linalg.solve(S, v)).item()

    @property
    def position(self):
        return self.x[:3, 0]

    @property
    def velocity(self):
        return self.x[3:, 0]


class JointKF:
    # per joint x = [q, qd, d] with q'' = u + d. u is the accel the computed torque asked for, d is
    # whatever it got wrong (payload, pushes, gripper reaction, mass error). only q is measured.
    # all joints have the same model and noise so one 3x3 P covers them, X is 3 x 6
    def __init__(self, q0, dt, sigma_enc, sigma_a=0.5, sigma_d=20.0):
        self.F = np.array([[1, dt, dt**2 / 2], [0, 1, dt], [0, 0, 1]])
        self.G = np.array([dt**2 / 2, dt, 0])
        self.H = np.array([1.0, 0, 0])
        self.Q = sigma_a**2 * np.outer(self.G, self.G) + np.diag([0, 0, sigma_d**2 * dt])
        self.R = sigma_enc**2
        self.X = np.vstack([q0, np.zeros(len(q0)), np.zeros(len(q0))])
        self.P = np.diag([sigma_enc**2, 1e-4, 1.0])

    def predict(self, u):
        self.X = self.F @ self.X + np.outer(self.G, u)
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, y):
        S = self.H @ self.P @ self.H + self.R
        K = self.P @ self.H / S
        self.X = self.X + np.outer(K, y - self.X[0])
        self.P = self.P - np.outer(K, self.H @ self.P)

    @property
    def sigma(self):
        return np.sqrt(np.diag(self.P))


def controllability_rank():
    """
    Used for the report.
    """

    A = np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ])

    B = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])

    Ctrb = B

    for i in range(1, 6):
        Ctrb = np.hstack((Ctrb, np.linalg.matrix_power(A, i) @ B))

    return np.linalg.matrix_rank(Ctrb)


def observability_rank():
    """
    Used for the report.
    """

    A = np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ])

    H = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
    ])

    Obsv = H

    for i in range(1, 6):
        Obsv = np.vstack((Obsv, H @ np.linalg.matrix_power(A, i)))

    return np.linalg.matrix_rank(Obsv)


if __name__ == "__main__":
    print("Controllability rank =", controllability_rank())
    print("Observability rank =", observability_rank())