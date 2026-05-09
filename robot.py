"""
robot.py
Robot model for Scout and Rescue robots.
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple

import numpy as np

try:
    import pygame
except ImportError:
    pygame = None

from controller import Genome, SearchRescueController
from ekf_slam import EKFSLAM
from world import (
    BLUE, PURPLE, WHITE, BLACK, GRAY, LIGHT_GRAY, GREEN, YELLOW,
    DT, ROBOT_RADIUS, SENSOR_REL_ANGLES, MAX_SENSOR_RANGE,
    distance, wrap_angle,
)


class Robot:
    def __init__(self, robot_id: int, role: str, pose, genome: Genome, color, seed: int = 0):
        self.robot_id = robot_id
        self.role = role  # "scout" or "rescue"
        self.true_pose = np.array(pose, dtype=float)
        self.ekf = EKFSLAM(pose)
        self.controller = SearchRescueController(genome)
        self.color = color
        self.rng = random.Random(seed)

        self.sensor_distances = [MAX_SENSOR_RANGE] * len(SENSOR_REL_ANGLES)
        self.sensor_endpoints: List[Tuple[float, float]] = []
        self.sensor_hits: List[bool] = []

        self.true_trajectory = []
        self.estimated_trajectory = []

        self.collisions = 0
        self.repeated_visits = 0
        self.visible_landmarks = 0
        # Simple anti-stuck bookkeeping used by the controller.
        self.stuck_counter = 0
        self._last_position = (float(self.true_pose[0]), float(self.true_pose[1]))

    def sense(self, world) -> None:
        self.sensor_distances = []
        self.sensor_endpoints = []
        self.sensor_hits = []

        x, y, th = self.true_pose
        for rel_angle in SENSOR_REL_ANGLES:
            d, endpoint, hit = world.cast_ray((x, y), th + rel_angle)
            self.sensor_distances.append(d)
            self.sensor_endpoints.append(endpoint)
            self.sensor_hits.append(hit)

    def update(self, world, grid, robots, dt: float = DT) -> None:
        # Sensor readings are generated from the simulated true pose.
        self.sense(world)

        # Controller uses EKF pose and occupancy grid.
        v, w = self.controller.control(self, grid, world, robots, self.rng)

        # Apply motion to true robot in the world.
        old_x, old_y = float(self.true_pose[0]), float(self.true_pose[1])
        self._move(world, v, w, dt)
        moved_dist = distance((old_x, old_y), (float(self.true_pose[0]), float(self.true_pose[1])))
        if moved_dist < 0.15:
            self.stuck_counter += 1
        else:
            self.stuck_counter = max(0, self.stuck_counter - 3)

        # EKF prediction/correction.
        self.ekf.predict(v, w, dt)
        self.visible_landmarks = self.ekf.correct_with_landmarks(self.true_pose, world.landmarks, self.rng)

        # Mapping uses estimated pose, not true pose.
        grid.update_from_scan(self.ekf.mu, self.sensor_endpoints, self.sensor_hits)

        # Victim interaction.
        self._update_victim_tasks(world)

        # Store trajectories for visualization.
        self.true_trajectory.append((float(self.true_pose[0]), float(self.true_pose[1])))
        self.estimated_trajectory.append((float(self.ekf.mu[0]), float(self.ekf.mu[1])))
        self.true_trajectory = self.true_trajectory[-1000:]
        self.estimated_trajectory = self.estimated_trajectory[-1000:]

    def _move(self, world, v: float, w: float, dt: float) -> None:
        x, y, th = self.true_pose
        new_th = wrap_angle(th + w * dt)
        dx = v * dt * math.cos(new_th)
        dy = v * dt * math.sin(new_th)

        nx, ny = x + dx, y + dy
        if not world.collides(nx, ny):
            self.true_pose = np.array([nx, ny, new_th], dtype=float)
            return

        # Simple wall sliding: try x-only and y-only movement.
        moved = False
        if not world.collides(x + dx, y):
            x += dx
            moved = True
        if not world.collides(x, y + dy):
            y += dy
            moved = True
        if not moved:
            self.collisions += 1

        self.true_pose = np.array([x, y, new_th], dtype=float)

    def _update_victim_tasks(self, world) -> None:
        x, y = float(self.true_pose[0]), float(self.true_pose[1])
        for victim in world.victims:
            d = distance((x, y), victim["pos"])
            # Both robots can visually detect nearby victim locations. The Scout
            # mainly discovers them through exploration, while the Rescue robot can
            # also discover victims that are directly in its corridor.
            if d < 85.0:
                victim["detected"] = True
            if d < 28.0 and self.role == "rescue":
                if not victim["rescued"]:
                    victim["rescued"] = True
                else:
                    # Count a repeated visit only occasionally, not every frame
                    # while the robot is still standing next to the same victim.
                    if len(self.true_trajectory) % 60 == 0:
                        self.repeated_visits += 1

    def _draw_covariance_ellipse(self, screen) -> None:
        """Draw a 2-sigma ellipse from the EKF-SLAM pose covariance.

        The ellipse uses the x/y block of the covariance matrix. It is drawn
        around the estimated pose, so it visually shows how localization
        uncertainty changes after motion prediction and landmark correction.
        """
        if pygame is None:
            return

        cov = np.asarray(self.ekf.sigma[:2, :2], dtype=float)
        if not np.all(np.isfinite(cov)):
            return

        # Numerical safety: force symmetric covariance before eigendecomposition.
        cov = 0.5 * (cov + cov.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 0.0)

        # Sort so the first eigenvalue is the major axis.
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        n_std = 2.0
        width = int(2 * n_std * math.sqrt(eigvals[0]))
        height = int(2 * n_std * math.sqrt(eigvals[1]))

        # Avoid huge or invisible ellipses that make the demo unreadable.
        if width < 3 or height < 3 or width > 350 or height > 350:
            return

        angle = math.degrees(math.atan2(eigvecs[1, 0], eigvecs[0, 0]))
        ex, ey = int(self.ekf.mu[0]), int(self.ekf.mu[1])

        ellipse_surf = pygame.Surface((width + 6, height + 6), pygame.SRCALPHA)
        ellipse_color = (*self.color, 95)
        pygame.draw.ellipse(ellipse_surf, ellipse_color, (3, 3, width, height), 2)
        rotated = pygame.transform.rotate(ellipse_surf, -angle)
        rect = rotated.get_rect(center=(ex, ey))
        screen.blit(rotated, rect)

    def _draw_visible_landmark_lines(self, screen, world) -> None:
        """Draw green observation lines to landmarks currently in sensor range."""
        if pygame is None or not hasattr(world, "landmarks"):
            return

        x, y, _ = self.true_pose
        landmark_range = getattr(self.ekf, "landmark_range", None)
        if landmark_range is None:
            # Fallback to the constant used by EKFSLAM if the attribute is not present.
            try:
                from ekf_slam import LANDMARK_RANGE
                landmark_range = LANDMARK_RANGE
            except Exception:
                landmark_range = MAX_SENSOR_RANGE

        for lx, ly in world.landmarks:
            if distance((x, y), (lx, ly)) <= landmark_range:
                pygame.draw.line(screen, GREEN, (int(x), int(y)), (int(lx), int(ly)), 2)
                pygame.draw.circle(screen, GREEN, (int(lx), int(ly)), 4)

    def localization_error(self) -> float:
        return self.ekf.position_error(self.true_pose)

    def draw(self, screen, world=None, show_trajectories: bool = True, show_sensors: bool = True) -> None:
        if pygame is None:
            return

        x, y, th = self.true_pose
        ex, ey, eth = self.ekf.mu

        # Visualize EKF-SLAM pose uncertainty as a 2-sigma covariance ellipse.
        self._draw_covariance_ellipse(screen)

        if show_trajectories and len(self.true_trajectory) > 2:
            pygame.draw.lines(screen, self.color, False, [(int(a), int(b)) for a, b in self.true_trajectory], 2)
        if show_trajectories and len(self.estimated_trajectory) > 2:
            draw_dotted_line(screen, [(int(a), int(b)) for a, b in self.estimated_trajectory], GRAY)

        if show_sensors:
            for endpoint, hit in zip(self.sensor_endpoints, self.sensor_hits):
                color = (150, 220, 160) if hit else (210, 230, 210)
                pygame.draw.line(screen, color, (int(x), int(y)), (int(endpoint[0]), int(endpoint[1])), 1)

            # Assignment visualization: draw a green line from the robot to each
            # landmark/feature currently inside the landmark sensor range. These
            # are the same observations used for EKF-SLAM correction.
            if world is not None:
                self._draw_visible_landmark_lines(screen, world)

        # EKF estimated pose.
        pygame.draw.circle(screen, GRAY, (int(ex), int(ey)), int(ROBOT_RADIUS), 1)
        pygame.draw.line(screen, GRAY, (int(ex), int(ey)),
                         (int(ex + 22 * math.cos(eth)), int(ey + 22 * math.sin(eth))), 2)

        # EKF-SLAM estimated landmark map (small dots instead of large crosses).
        for _, lx, ly in self.ekf.estimated_landmarks(seen_only=True):
            lx_i, ly_i = int(lx), int(ly)
            pygame.draw.circle(screen, self.color, (lx_i, ly_i), 3)

        # True robot body.
        pygame.draw.circle(screen, self.color, (int(x), int(y)), int(ROBOT_RADIUS))
        pygame.draw.circle(screen, BLACK, (int(x), int(y)), int(ROBOT_RADIUS), 1)
        pygame.draw.line(screen, WHITE, (int(x), int(y)),
                         (int(x + 24 * math.cos(th)), int(y + 24 * math.sin(th))), 3)

        label = "S" if self.role == "scout" else "R"
        draw_text(screen, label, int(x - 5), int(y - 8), WHITE, 14)


def draw_dotted_line(screen, points, color, dash: int = 7) -> None:
    for p0, p1 in zip(points[:-1], points[1:]):
        x0, y0 = p0
        x1, y1 = p1
        length = math.hypot(x1 - x0, y1 - y0)
        if length <= 1e-6:
            continue
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        d = 0.0
        while d < length:
            if int(d / dash) % 2 == 0:
                a = (int(x0 + ux * d), int(y0 + uy * d))
                b = (int(x0 + ux * min(d + dash, length)), int(y0 + uy * min(d + dash, length)))
                pygame.draw.line(screen, color, a, b, 2)
            d += dash


_FONT_CACHE = {}


def draw_text(screen, text: str, x: int, y: int, color, size: int = 18) -> None:
    if pygame is None:
        return
    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = pygame.font.SysFont("Arial", size)
    screen.blit(_FONT_CACHE[size].render(str(text), True, color), (x, y))