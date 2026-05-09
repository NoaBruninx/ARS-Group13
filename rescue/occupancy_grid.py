import numpy as np
import math

from config import *


class OccupancyGridMap:

    def __init__(self):

        self.log_odds_grid = np.zeros(
            (GRID_ROWS, GRID_COLS),
            dtype=np.float32
        )

    def clear(self):

        self.log_odds_grid[:] = 0.0

    def update(
            self,
            pose,
            distances
    ):

        robot_x, robot_y, robot_theta = pose

        max_reading = (
                MAX_SENSOR_DISTANCE
                - CIRCLE_RADIUS
        )

        robot_row, robot_col = (
            int(robot_y // MAP_CELL_SIZE),
            int(robot_x // MAP_CELL_SIZE)
        )

        if (
                0 <= robot_row < GRID_ROWS
                and 0 <= robot_col < GRID_COLS
        ):

            self.log_odds_grid[
                robot_row,
                robot_col
            ] += L_FREE

        for i, dist in enumerate(distances):

            angle = (
                    robot_theta
                    + i * SENSOR_ANGLE_STEP
            )

            sx = (
                    robot_x
                    + math.cos(angle)
                    * CIRCLE_RADIUS
            )

            sy = (
                    robot_y
                    + math.sin(angle)
                    * CIRCLE_RADIUS
            )

            measured_dist = max(
                0.0,
                min(float(dist), max_reading)
            )

            hit_detected = (
                    measured_dist
                    < (max_reading - 1.0)
            )

            ex = (
                    sx
                    + math.cos(angle)
                    * measured_dist
            )

            ey = (
                    sy
                    + math.sin(angle)
                    * measured_dist
            )

            dx = ex - sx
            dy = ey - sy

            length = math.hypot(dx, dy)

            if length < 1e-9:
                continue

            step = max(
                1.0,
                MAP_CELL_SIZE / 2.0
            )

            steps = max(
                1,
                int(length / step)
            )

            cells = []

            last = None

            for s in range(steps + 1):

                t = s / steps

                x = sx + t * dx
                y = sy + t * dy

                row = int(y // MAP_CELL_SIZE)
                col = int(x // MAP_CELL_SIZE)

                if (
                        0 <= row < GRID_ROWS
                        and
                        0 <= col < GRID_COLS
                        and
                        (row, col) != last
                ):

                    cells.append((row, col))
                    last = (row, col)

            if not cells:
                continue

            free_cells = (
                cells[:-1]
                if hit_detected
                else cells
            )

            for row, col in free_cells:

                self.log_odds_grid[
                    row,
                    col
                ] += (
                        L_FREE
                        - L_PRIOR
                )

            if hit_detected:

                er, ec = cells[-1]

                self.log_odds_grid[
                    er,
                    ec
                ] += (
                        L_OCC
                        - L_PRIOR
                )

            np.clip(
                self.log_odds_grid,
                LOG_ODDS_MIN,
                LOG_ODDS_MAX,
                out=self.log_odds_grid
            )