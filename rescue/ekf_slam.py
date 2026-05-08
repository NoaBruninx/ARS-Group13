"""
ekf_slam.py
EKF-SLAM with known landmark correspondence.

State vector:
    X = [x, y, theta, l1_x, l1_y, l2_x, l2_y, ...]^T

The robot pose is estimated together with the landmark positions. The
occupancy grid map is still built separately from the range sensors; EKF-SLAM
is used for pose estimation and for maintaining a feature-based landmark map.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from world import wrap_angle, distance


# Simple, report-friendly noise parameters.
LANDMARK_RANGE = 320.0
RANGE_NOISE_STD = 4.0
BEARING_NOISE_STD = math.radians(3.0)

PROCESS_NOISE = np.diag([1.2, 1.2, math.radians(2.0)]) ** 2
MEASUREMENT_NOISE = np.diag([RANGE_NOISE_STD, BEARING_NOISE_STD]) ** 2


class EKFSLAM:
    """Extended Kalman Filter SLAM for range-bearing landmarks.

    Data association is assumed to be known, which keeps the implementation
    close to the course setting and easy to explain. A landmark is initialized
    the first time it is observed, then later observations update both the
    robot pose and the landmark position through the joint covariance matrix.
    """

    def __init__(self, initial_pose):
        self.state = np.array(initial_pose, dtype=float)  # full SLAM state after landmark initialization
        self.sigma = np.diag([18.0, 18.0, math.radians(10.0)]) ** 2
        self.num_landmarks = 0
        self.seen_landmarks: List[bool] = []
        self.landmark_range = LANDMARK_RANGE

    @property
    def mu(self) -> np.ndarray:
        """Compatibility property: returns the estimated robot pose [x, y, theta]."""
        return self.state[:3]

    def _ensure_landmark_slots(self, n_landmarks: int) -> None:
        if self.num_landmarks == n_landmarks:
            return
        if self.num_landmarks != 0:
            raise ValueError("Changing the number of landmarks during a run is not supported.")

        old_dim = 3
        new_dim = 3 + 2 * n_landmarks
        new_state = np.zeros(new_dim, dtype=float)
        new_state[:old_dim] = self.state[:old_dim]

        new_sigma = np.eye(new_dim, dtype=float) * 1e6
        new_sigma[:old_dim, :old_dim] = self.sigma

        self.state = new_state
        self.sigma = new_sigma
        self.num_landmarks = n_landmarks
        self.seen_landmarks = [False] * n_landmarks

    def predict(self, v: float, w: float, dt: float) -> None:
        """Prediction step using nonlinear velocity motion model.

        Only the robot pose changes during prediction. Landmark coordinates are
        static, but their correlation with the robot pose is carried in the full
        covariance matrix.
        """
        x, y, th = self.state[:3]
        self.state[0] = x + v * dt * math.cos(th)
        self.state[1] = y + v * dt * math.sin(th)
        self.state[2] = wrap_angle(th + w * dt)

        dim = len(self.state)
        G = np.eye(dim)
        G[:3, :3] = np.array([
            [1.0, 0.0, -v * dt * math.sin(th)],
            [0.0, 1.0,  v * dt * math.cos(th)],
            [0.0, 0.0, 1.0],
        ])
        self.sigma = G @ self.sigma @ G.T
        self.sigma[:3, :3] += PROCESS_NOISE
        self._symmetrize_covariance()

    def correct_with_landmarks(self, true_pose, landmarks: List[Tuple[float, float]], rng) -> int:
        """Simulate range-bearing observations and apply EKF-SLAM correction.

        In this simulated assignment, the measurement is generated from the true
        pose with Gaussian noise. The filter itself only receives the noisy
        range-bearing measurement and the landmark id, i.e. known correspondence.
        """
        self._ensure_landmark_slots(len(landmarks))

        tx, ty, tth = true_pose
        visible_count = 0

        for lm_id, (lx, ly) in enumerate(landmarks):
            r_true = distance((tx, ty), (lx, ly))
            if r_true > LANDMARK_RANGE:
                continue

            visible_count += 1
            bearing_true = wrap_angle(math.atan2(ly - ty, lx - tx) - tth)
            z = np.array([
                r_true + rng.gauss(0.0, RANGE_NOISE_STD),
                wrap_angle(bearing_true + rng.gauss(0.0, BEARING_NOISE_STD)),
            ], dtype=float)

            if not self.seen_landmarks[lm_id]:
                self._initialize_landmark(lm_id, z)

            self._correct_one_landmark(lm_id, z)

        return visible_count

    def _landmark_slice(self, lm_id: int) -> slice:
        start = 3 + 2 * lm_id
        return slice(start, start + 2)

    def _initialize_landmark(self, lm_id: int, z: np.ndarray) -> None:
        """Initialize landmark position from current robot pose and first observation."""
        x, y, th = self.state[:3]
        r, bearing = float(z[0]), float(z[1])
        global_bearing = wrap_angle(th + bearing)
        lx = x + r * math.cos(global_bearing)
        ly = y + r * math.sin(global_bearing)

        sl = self._landmark_slice(lm_id)
        self.state[sl] = np.array([lx, ly], dtype=float)

        # After first observation, uncertainty is large but finite. Cross-covariances
        # start at zero in this simple implementation; later EKF updates create the
        # correlations between robot and landmark estimates.
        idx = sl.start
        self.sigma[idx:idx + 2, idx:idx + 2] = np.diag([120.0, 120.0]) ** 2
        self.seen_landmarks[lm_id] = True

    def _correct_one_landmark(self, lm_id: int, z: np.ndarray) -> None:
        x, y, th = self.state[:3]
        sl = self._landmark_slice(lm_id)
        lx, ly = self.state[sl]

        dx = lx - x
        dy = ly - y
        q = dx * dx + dy * dy
        if q < 1e-9:
            return
        sq = math.sqrt(q)

        z_hat = np.array([sq, wrap_angle(math.atan2(dy, dx) - th)], dtype=float)

        H = np.zeros((2, len(self.state)), dtype=float)
        # Derivatives w.r.t. robot pose [x, y, theta].
        H[0, 0] = -dx / sq
        H[0, 1] = -dy / sq
        H[0, 2] = 0.0
        H[1, 0] = dy / q
        H[1, 1] = -dx / q
        H[1, 2] = -1.0

        # Derivatives w.r.t. landmark position [lx, ly].
        idx = sl.start
        H[0, idx] = dx / sq
        H[0, idx + 1] = dy / sq
        H[1, idx] = -dy / q
        H[1, idx + 1] = dx / q

        S = H @ self.sigma @ H.T + MEASUREMENT_NOISE
        K = self.sigma @ H.T @ np.linalg.inv(S)

        innovation = z - z_hat
        innovation[1] = wrap_angle(innovation[1])

        self.state = self.state + K @ innovation
        self.state[2] = wrap_angle(self.state[2])

        I = np.eye(len(self.state))
        self.sigma = (I - K @ H) @ self.sigma
        self._symmetrize_covariance()

    def _symmetrize_covariance(self) -> None:
        self.sigma = 0.5 * (self.sigma + self.sigma.T)

    def estimated_landmarks(self, seen_only: bool = True) -> List[Tuple[int, float, float]]:
        """Return landmark estimates as (id, x, y)."""
        if self.num_landmarks == 0:
            return []
        out = []
        for i in range(self.num_landmarks):
            if seen_only and not self.seen_landmarks[i]:
                continue
            lx, ly = self.state[self._landmark_slice(i)]
            out.append((i, float(lx), float(ly)))
        return out

    def num_mapped_landmarks(self) -> int:
        return sum(self.seen_landmarks)

    def mean_landmark_error(self, true_landmarks: List[Tuple[float, float]]) -> float:
        errors = []
        for i, lx, ly in self.estimated_landmarks(seen_only=True):
            if i < len(true_landmarks):
                errors.append(distance((lx, ly), true_landmarks[i]))
        return float(np.mean(errors)) if errors else 0.0

    def position_error(self, true_pose) -> float:
        return distance((self.state[0], self.state[1]), (true_pose[0], true_pose[1]))


# Backwards-compatible alias for code that previously imported EKFLocalizer.
EKFLocalizer = EKFSLAM
