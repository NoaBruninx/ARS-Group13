"""
ekf_localization.py
EKF pose tracking using velocity controls and range-bearing landmark measurements.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from world import wrap_angle, distance


# Noise parameters. These are deliberately simple and easy to explain.
LANDMARK_RANGE = 260.0
RANGE_NOISE_STD = 4.0
BEARING_NOISE_STD = math.radians(3.0)

PROCESS_NOISE = np.diag([1.2, 1.2, math.radians(2.0)]) ** 2
MEASUREMENT_NOISE = np.diag([RANGE_NOISE_STD, BEARING_NOISE_STD]) ** 2


class EKFLocalizer:
    """Extended Kalman Filter for state [x, y, theta]."""

    def __init__(self, initial_pose):
        self.mu = np.array(initial_pose, dtype=float)
        self.sigma = np.diag([18.0, 18.0, math.radians(10.0)]) ** 2

    def predict(self, v: float, w: float, dt: float) -> None:
        """Prediction step using a nonlinear velocity motion model."""
        x, y, th = self.mu

        self.mu = np.array([
            x + v * dt * math.cos(th),
            y + v * dt * math.sin(th),
            wrap_angle(th + w * dt),
        ], dtype=float)

        # Jacobian of motion model w.r.t. state.
        G = np.array([
            [1.0, 0.0, -v * dt * math.sin(th)],
            [0.0, 1.0,  v * dt * math.cos(th)],
            [0.0, 0.0, 1.0],
        ])
        self.sigma = G @ self.sigma @ G.T + PROCESS_NOISE

    def correct_with_landmarks(self, true_pose, landmarks: List[Tuple[float, float]], rng) -> int:
        """Simulate visible landmark measurements and apply EKF correction.

        In a real robot the measurement would come from sensors. In simulation,
        noisy range-bearing measurements are generated from the true pose.
        """
        tx, ty, tth = true_pose
        visible_count = 0

        for lm in landmarks:
            lx, ly = lm
            r_true = distance((tx, ty), (lx, ly))
            if r_true > LANDMARK_RANGE:
                continue

            visible_count += 1
            bearing_true = wrap_angle(math.atan2(ly - ty, lx - tx) - tth)
            z = np.array([
                r_true + rng.gauss(0.0, RANGE_NOISE_STD),
                wrap_angle(bearing_true + rng.gauss(0.0, BEARING_NOISE_STD)),
            ])
            self._correct_one_landmark(lm, z)

        return visible_count

    def _correct_one_landmark(self, landmark, z) -> None:
        x, y, th = self.mu
        lx, ly = landmark
        dx = lx - x
        dy = ly - y
        q = dx * dx + dy * dy
        if q < 1e-9:
            return

        sq = math.sqrt(q)
        z_hat = np.array([sq, wrap_angle(math.atan2(dy, dx) - th)])

        # Jacobian of measurement model h(x) = [range, bearing].
        H = np.array([
            [-dx / sq, -dy / sq, 0.0],
            [ dy / q,  -dx / q, -1.0],
        ])

        S = H @ self.sigma @ H.T + MEASUREMENT_NOISE
        K = self.sigma @ H.T @ np.linalg.inv(S)

        innovation = z - z_hat
        innovation[1] = wrap_angle(innovation[1])

        self.mu = self.mu + K @ innovation
        self.mu[2] = wrap_angle(self.mu[2])
        self.sigma = (np.eye(3) - K @ H) @ self.sigma

    def position_error(self, true_pose) -> float:
        return distance((self.mu[0], self.mu[1]), (true_pose[0], true_pose[1]))
