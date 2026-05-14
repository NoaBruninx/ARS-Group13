"""
occupancy_grid.py
Occupancy grid mapping with log-odds updates.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Tuple

import numpy as np

try:
    import pygame
except ImportError:
    pygame = None

from world import WIDTH, HEIGHT, clamp


LOG_ODDS_FREE = -0.35
LOG_ODDS_OCCUPIED = 0.85
LOG_ODDS_MIN = -5.0
LOG_ODDS_MAX = 5.0


Point = Tuple[float, float]



def bresenham(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    """Grid cells intersected by a line."""
    cells = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0

    while True:
        cells.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return cells


class OccupancyGrid:
    def __init__(self, width: int = WIDTH, height: int = HEIGHT, cell_size: int = 20):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.cols = math.ceil(width / cell_size)
        self.rows = math.ceil(height / cell_size)
        self.log_odds = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.evidence = np.zeros((self.rows, self.cols), dtype=np.int32)
        # Persistent "ever seen" mask so explored_fraction never decreases.
        self._ever_seen = np.zeros((self.rows, self.cols), dtype=bool)
        self.pheromone = np.zeros((self.height, self.width), dtype=float)
        self.pheromone_decay = 0.999
        self.pheromone_deposit = 1
        self.max_pheromone = 99999

    def get_pheromone(self, x: float, y: float) -> float:
        cx, cy = self.world_to_cell(x, y)

        if not self.in_bounds(cx, cy):
            return 0.0

        return float(self.pheromone[cy, cx])
    def deposit_pheromone(self, x: float, y: float, amount: float = 5):
        cx, cy = self.world_to_cell(x, y)

        if not self.in_bounds(cx, cy):
            return

        self.pheromone[cy, cx] += amount

        # clamp to prevent runaway hotspots
        if self.pheromone[cy, cx] > self.max_pheromone:
            self.pheromone[cy, cx] = self.max_pheromone

    def spread_pheromones(self, diffusion_rate: float = 0.0015):
        """
        Diffuse pheromones to neighboring cells (4-neighbour smoothing).
        This creates realistic spreading trails instead of only point deposits.
        """

        p = self.pheromone

        # shifted versions (fast convolution-like diffusion)
        up    = np.roll(p, -1, axis=0)
        down  = np.roll(p,  1, axis=0)
        left  = np.roll(p, -1, axis=1)
        right = np.roll(p,  1, axis=1)

        # average neighbors
        neighbors = (up + down + left + right) * 0.25

        # diffusion step
        self.pheromone = (1 - diffusion_rate) * p + diffusion_rate * neighbors

        # optional: keep bounds stable
        np.clip(self.pheromone, 0.0, self.max_pheromone, out=self.pheromone)
    def decay_pheromones(self):
        self.pheromone *= self.pheromone_decay

    def pheromone_gradient(self, x: float, y: float):
        cx, cy = self.world_to_cell(x, y)

        if not self.in_bounds(cx, cy):
            return (0.0, 0.0)

        def v(a, b):
            if self.in_bounds(a, b):
                return self.pheromone[b, a]
            return self.pheromone[cy, cx]

        # Sobel-style smoothing (stable vs 4-neighbor noise)
        gx = (
                v(cx + 1, cy - 1) + 2 * v(cx + 1, cy) + v(cx + 1, cy + 1)
                - v(cx - 1, cy - 1) - 2 * v(cx - 1, cy) - v(cx - 1, cy + 1)
        )

        gy = (
                v(cx - 1, cy + 1) + 2 * v(cx, cy + 1) + v(cx + 1, cy + 1)
                - v(cx - 1, cy - 1) - 2 * v(cx, cy - 1) - v(cx + 1, cy - 1)
        )

        norm = math.hypot(gx, gy)
        if norm < 1e-9:
            return (0.0, 0.0)

        return gx / norm, gy / norm

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        return int(x // self.cell_size), int(y // self.cell_size)

    def cell_to_world(self, cx: int, cy: int) -> Point:
        return (cx + 0.5) * self.cell_size, (cy + 0.5) * self.cell_size

    def in_bounds(self, cx: int, cy: int) -> bool:
        return 0 <= cx < self.cols and 0 <= cy < self.rows

    def probabilities(self) -> np.ndarray:
        odds = np.exp(self.log_odds)
        return odds / (1.0 + odds)

    def update_from_scan(self, estimated_pose, endpoints: List[Point], hits: List[bool]) -> None:
        """Update map using robot estimated pose and sensor endpoints."""
        sx, sy = self.world_to_cell(float(estimated_pose[0]), float(estimated_pose[1]))
        if not self.in_bounds(sx, sy):
            return

        for endpoint, hit in zip(endpoints, hits):
            ex, ey = self.world_to_cell(endpoint[0], endpoint[1])
            if not self.in_bounds(ex, ey):
                continue

            ray_cells = bresenham(sx, sy, ex, ey)
            if not ray_cells:
                continue

            free_cells = ray_cells[:-1] if hit else ray_cells
            for cx, cy in free_cells:
                if self.in_bounds(cx, cy):
                    self.log_odds[cy, cx] = clamp(
                        float(self.log_odds[cy, cx] + LOG_ODDS_FREE), LOG_ODDS_MIN, LOG_ODDS_MAX
                    )
                    self.evidence[cy, cx] += 1
                    self._ever_seen[cy, cx] = True

            if hit:
                cx, cy = ray_cells[-1]
                if self.in_bounds(cx, cy):
                    self.log_odds[cy, cx] = clamp(
                        float(self.log_odds[cy, cx] + LOG_ODDS_OCCUPIED), LOG_ODDS_MIN, LOG_ODDS_MAX
                    )
                    self.evidence[cy, cx] += 1
                    self._ever_seen[cy, cx] = True

    def merge_from(self, other: "OccupancyGrid") -> None:
        """Simple shared-map merge: add log-odds evidence."""
        if self.log_odds.shape != other.log_odds.shape:
            raise ValueError("Cannot merge maps with different shapes")
        self.log_odds = np.clip(self.log_odds + other.log_odds, LOG_ODDS_MIN, LOG_ODDS_MAX)
        self.evidence = self.evidence + other.evidence
        self._ever_seen |= other._ever_seen

    def explored_fraction(self) -> float:
        """Fraction of cells that have ever received sensor evidence (never decreases)."""
        return float(np.mean(self._ever_seen))

    def occupied_fraction(self) -> float:
        p = self.probabilities()
        return float(np.mean(p > 0.65))

    def is_cell_known(self, cx: int, cy: int) -> bool:
        """Return True if a grid cell has ever received any sensor evidence."""
        return self.in_bounds(cx, cy) and self._ever_seen[cy, cx]

    def is_point_known(self, x: float, y: float) -> bool:
        """Return True if a world position lies in an already observed cell."""
        cx, cy = self.world_to_cell(x, y)
        return self.is_cell_known(cx, cy)

    def get_frontiers(self, max_points: int = 300) -> List[Point]:
        """Return unknown cells next to free cells."""
        p = self.probabilities()
        free = p < 0.35
        unknown = self.evidence == 0
        frontiers: List[Point] = []

        for cy in range(1, self.rows - 1):
            for cx in range(1, self.cols - 1):
                if not unknown[cy, cx]:
                    continue
                if np.any(free[cy - 1:cy + 2, cx - 1:cx + 2]):
                    frontiers.append(self.cell_to_world(cx, cy))
                    if len(frontiers) >= max_points:
                        return frontiers
        return frontiers

    def draw(self, screen) -> None:
        if pygame is None:
            return

        # Lazily create and cache the overlay surface — never recreate it each frame.
        if not hasattr(self, "_overlay") or self._overlay is None:
            self._overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            self._overlay.fill((0, 0, 0, 0))
            self._drawn_evidence = np.zeros_like(self.evidence)

        # Only redraw cells whose evidence count has changed since last draw.
        changed = self._ever_seen & (self.evidence != self._drawn_evidence)
        if np.any(changed):
            p = self.probabilities()
            ys, xs = np.where(changed)
            for cy, cx in zip(ys, xs):
                prob = float(p[cy, cx])
                x = cx * self.cell_size
                y = cy * self.cell_size
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                # Clear old pixel first
                self._overlay.fill((0, 0, 0, 0), rect)
                if prob > 0.65:
                    shade = int(255 * (1.0 - prob))
                    color = (shade, shade, shade, 150)
                elif prob < 0.35:
                    color = (205, 230, 255, 90)
                else:
                    color = (180, 180, 180, 60)
                pygame.draw.rect(self._overlay, color, rect)
            self._drawn_evidence = self.evidence.copy()

        screen.blit(self._overlay, (0, 0))