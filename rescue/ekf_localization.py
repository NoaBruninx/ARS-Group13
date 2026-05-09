import numpy as np
import math

from config import *


class EKFLocalization:

    def __init__(self, x, y, theta):

        self.mu = np.array([
            x,
            y,
            theta
        ])

        self.sigma = np.eye(3) * 100.0

    def predict_odometry(self, dx, dy, dtheta):

        x, y, theta = self.mu

        theta_mid = theta + dtheta / 2.0

        x_new = x + dx
        y_new = y + dy
        theta_new = theta + dtheta

        self.mu = np.array([
            x_new,
            y_new,
            theta_new % (2 * math.pi)
        ])

    def update(
            self,
            observations,
            landmarks
    ):

        for (lm_idx, z_dist, z_bearing) in observations:

            lx, ly = landmarks[lm_idx]

            x, y, theta = self.mu

            expected_dist = math.hypot(
                lx - x,
                ly - y
            )

            expected_bearing = (
                    math.atan2(ly - y, lx - x)
                    - theta
            )

            z = np.array([
                z_dist,
                z_bearing
            ])

            z_hat = np.array([
                expected_dist,
                expected_bearing
            ])

            dx = lx - x
            dy = ly - y

            if expected_dist < 1e-9:
                continue

            C = np.array([
                [-dx / expected_dist,
                 -dy / expected_dist,
                 0],

                [dy / (expected_dist ** 2),
                 -dx / (expected_dist ** 2),
                 -1]
            ])

            Q = np.diag([
                SENSOR_NOISE_STD ** 2,
                BEARING_NOISE_STD ** 2
            ])

            S = C @ self.sigma @ C.T + Q

            K = self.sigma @ C.T @ np.linalg.inv(S)

            innovation = z - z_hat

            innovation[1] = (
                    (innovation[1] + math.pi)
                    % (2 * math.pi)
                    - math.pi
            )

            self.mu = self.mu + K @ innovation

            self.sigma = (
                                 np.eye(3)
                                 - K @ C
                         ) @ self.sigma