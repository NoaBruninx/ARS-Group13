import pygame
import math

from config import *

# -------------------------------------------------
# COLORS (global constants — correct place)
# -------------------------------------------------

GREEN = (30, 180, 60)
RED = (220, 60, 60)
GRAY = (120, 120, 120)


class Renderer:

    def __init__(self, font_small, font_med):

        self.font_small = font_small
        self.font_med = font_med

    # -------------------------------------------------
    # ROBOT
    # -------------------------------------------------

    def draw_robot(self, surface, cx, cy, theta):

        pygame.draw.circle(
            surface,
            RED,
            (int(cx), int(cy)),
            CIRCLE_RADIUS
        )

        hx = cx + math.cos(theta) * CIRCLE_RADIUS
        hy = cy + math.sin(theta) * CIRCLE_RADIUS

        pygame.draw.line(
            surface,
            BLACK,
            (int(cx), int(cy)),
            (int(hx), int(hy)),
            3
        )

    # -------------------------------------------------
    # OCCUPANCY GRID
    # -------------------------------------------------

    def draw_occupancy_grid(self, surface, occupancy):

        import numpy as np

        probs = 1.0 - 1.0 / (1.0 + np.exp(occupancy.log_odds_grid))

        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):

                p = probs[row, col]

                if p > 0.55:
                    shade = max(0, min(255, int(255 * (1.0 - p))))
                elif p < 0.45:
                    shade = 245
                else:
                    shade = 210

                rect = pygame.Rect(
                    col * MAP_CELL_SIZE,
                    row * MAP_CELL_SIZE,
                    MAP_CELL_SIZE,
                    MAP_CELL_SIZE
                )

                pygame.draw.rect(surface, (shade, shade, shade), rect)

    # -------------------------------------------------
    # WALLS
    # -------------------------------------------------

    def draw_walls(self, surface, world):

        for (x1, y1, x2, y2) in world.internal_walls:

            pygame.draw.line(
                surface,
                BLACK,
                (x1, y1),
                (x2, y2),
                4
            )

        pygame.draw.rect(
            surface,
            BLACK,
            (0, 0, WIDTH, HEIGHT),
            4
        )

    # -------------------------------------------------
    # SENSORS
    # -------------------------------------------------

    def draw_sensors(self, surface, cx, cy, theta, distances, hit_points):

        for i, (dist, pt) in enumerate(zip(distances, hit_points)):

            angle = theta + i * SENSOR_ANGLE_STEP

            sx = cx + math.cos(angle) * CIRCLE_RADIUS
            sy = cy + math.sin(angle) * CIRCLE_RADIUS

            pygame.draw.line(
                surface,
                GRAY,
                (int(sx), int(sy)),
                (int(pt[0]), int(pt[1])),
                1
            )

    # -------------------------------------------------
    # LANDMARKS
    # -------------------------------------------------

    def draw_landmarks(self, surface, landmarks, cx, cy, observations):

        for i, (lx, ly) in enumerate(landmarks):

            pygame.draw.circle(surface, BLACK, (lx, ly), 6)

            visible = any(obs[0] == i for obs in observations)

            if visible:
                pygame.draw.line(
                    surface,
                    GREEN,
                    (int(cx), int(cy)),
                    (lx, ly),
                    1
                )

    # -------------------------------------------------
    # EKF
    # -------------------------------------------------

    def draw_kf_estimate(self, surface, mu, sigma):

        pygame.draw.circle(
            surface,
            BLUE,
            (int(mu[0]), int(mu[1])),
            8,
            2
        )

        sx = math.sqrt(abs(sigma[0, 0])) * 3
        sy = math.sqrt(abs(sigma[1, 1])) * 3

        rect = pygame.Rect(
            int(mu[0] - sx),
            int(mu[1] - sy),
            int(2 * sx),
            int(2 * sy)
        )

        pygame.draw.ellipse(surface, BLUE, rect, 1)

        hx = mu[0] + math.cos(mu[2]) * 14
        hy = mu[1] + math.sin(mu[2]) * 14

        pygame.draw.line(
            surface,
            BLUE,
            (int(mu[0]), int(mu[1])),
            (int(hx), int(hy)),
            2
        )

    # -------------------------------------------------
    # TRAJECTORY
    # -------------------------------------------------

    def draw_dotted_trajectory(self, surface, trajectory, color, width=2):

        if len(trajectory) < 2:
            return

        for i in range(0, len(trajectory) - 1, 2):

            x1, y1 = trajectory[i]
            x2, y2 = trajectory[i + 1]

            pygame.draw.line(
                surface,
                color,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                width
            )

    # -------------------------------------------------
    # HUD
    # -------------------------------------------------

    def draw_hud(self, surface, v_left, v_right, theta_deg,
                 num_obs, map_mode, show_map):

        lines = [
            f"Left motor  : {v_left:+.1f}",
            f"Right motor : {v_right:+.1f}",
            f"Heading     : {theta_deg:.1f}",
            f"Landmarks   : {num_obs}",
            f"Mapping     : {map_mode}",
            f"Map visible : {show_map}",
        ]

        y = 10

        for line in lines:

            txt = self.font_med.render(line, True, BLACK)
            surface.blit(txt, (10, y))
            y += 22

    # -------------------------------------------------
    # VICTIMS (FIXED)
    # -------------------------------------------------

    def draw_victims(self, surface, world):

        for v in world.victims:

            if v.saved:
                color = GRAY

            elif v.carried_by is not None:
                color = GREEN

            else:
                color = RED

            pygame.draw.circle(
                surface,
                color,
                (int(v.x), int(v.y)),
                8
            )

            pygame.draw.circle(
                surface,
                BLACK,
                (int(v.x), int(v.y)),
                8,
                1
            )
    def draw_pheromones(self, surface, pheromone_map):
        import numpy as np

        grid = pheromone_map.grid

        # normalize for visualization
        max_val = np.max(grid) if np.max(grid) > 0 else 1.0

        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):

                p = grid[r, c] / max_val  # 0 → 1

                if p <= 0.01:
                    continue  # skip empty cells

                intensity = int(255 * p)

                color = (intensity, 0, 255 - intensity)  # purple-red heatmap

                rect = pygame.Rect(
                    c * MAP_CELL_SIZE,
                    r * MAP_CELL_SIZE,
                    MAP_CELL_SIZE,
                    MAP_CELL_SIZE
                )

                pygame.draw.rect(surface, color, rect)
    def draw_robots(self, surface, robots):

        for robot in robots:
            pygame.draw.circle(
                surface,
                RED,
                (int(robot.x), int(robot.y)),
                CIRCLE_RADIUS
            )

            hx = robot.x + math.cos(robot.theta) * CIRCLE_RADIUS
            hy = robot.y + math.sin(robot.theta) * CIRCLE_RADIUS

            pygame.draw.line(
                surface,
                BLACK,
                (int(robot.x), int(robot.y)),
                (int(hx), int(hy)),
                3
            )