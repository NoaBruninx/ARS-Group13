import math

import numpy as np
from config import GRID_ROWS, GRID_COLS


class PheromoneMap:

    def __init__(self):
        self.grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)

    def add(self, r, c, amount=1.0):
        if 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS:
            self.grid[r, c] += amount

    def decay(self, dt, decay_rate=5):
        self.grid *= math.exp(-decay_rate * dt)

    def get(self, r, c):
        if 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS:
            return self.grid[r, c]
        return 0.0

    # -------------------------------------------------
    # WALL-AWARE BEST DIRECTION
    # -------------------------------------------------
    def best_direction(self, r, c, wall_grid=None):

        if not (0 <= r < GRID_ROWS and 0 <= c < GRID_COLS):
            return (0, 0)

        best_score = -1e9
        best_dir = (0, 0)

        directions = [
            (-1, 0), (1, 0),
            (0, -1), (0, 1),
            (-1, -1), (-1, 1),
            (1, -1), (1, 1)
        ]

        for dr, dc in directions:

            nr, nc = r + dr, c + dc

            if not (0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS):
                continue

            pheromone = self.grid[nr, nc]

            # ---------------- WALL COST ----------------
            wall_cost = 0.0
            if wall_grid is not None:
                if wall_grid[nr, nc] == 1:
                    wall_cost = 1000.0  # strong penalty for walls

            # ---------------- FINAL SCORE ----------------
            score = pheromone - wall_cost

            if score > best_score:
                best_score = score
                best_dir = (dr, dc)

        return best_dir