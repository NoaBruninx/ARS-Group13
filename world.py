"""
world.py
Search-and-rescue world, geometry helpers, ray casting, and drawing.

Scenario: Search and Rescue / damaged building.
The map represents a damaged building with rooms, corridors, rubble piles,
fixed landmarks for EKF-SLAM, hidden victims, and an optional collapsed passage.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

try:
    import pygame
except ImportError:  # Allows non-visual experiment runs to import this file.
    pygame = None


# ---------------------------------------------------------------------
# Global simulation constants
# ---------------------------------------------------------------------

WIDTH, HEIGHT = 1100, 700
FPS = 60
DT = 1.0 / FPS

ROBOT_RADIUS = 14.0
NUM_SENSORS = 12
MAX_SENSOR_RANGE = 190.0
SENSOR_REL_ANGLES = [2.0 * math.pi * i / NUM_SENSORS for i in range(NUM_SENSORS)]

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
LIGHT_GRAY = (210, 210, 210)
GREEN = (45, 150, 75)
DARK_GREEN = (25, 105, 55)
BLUE = (45, 110, 230)
PURPLE = (135, 70, 190)
ORANGE = (235, 145, 35)
RED = (210, 65, 65)
CYAN = (50, 170, 190)
YELLOW = (240, 210, 60)
UNKNOWN_DARK = (32, 36, 34)
KNOWN_FLOOR = (245, 250, 242)
RUBBLE = (210, 120, 40)
DANGER_RED = (180, 65, 55)


Point = Tuple[float, float]
Segment = Tuple[Point, Point]
RectTuple = Tuple[float, float, float, float]


# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def wrap_angle(a: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def point_to_segment_distance(p: Point, a: Point, b: Point) -> float:
    px, py = p
    ax, ay = a
    bx, by = b
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    denom = abx * abx + aby * aby
    if denom <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = clamp((apx * abx + apy * aby) / denom, 0.0, 1.0)
    cx = ax + t * abx
    cy = ay + t * aby
    return math.hypot(px - cx, py - cy)


def circle_intersects_rect(cx: float, cy: float, radius: float, rect: RectTuple) -> bool:
    rx, ry, rw, rh = rect
    closest_x = clamp(cx, rx, rx + rw)
    closest_y = clamp(cy, ry, ry + rh)
    return math.hypot(cx - closest_x, cy - closest_y) <= radius


def point_in_rect(x: float, y: float, rect: RectTuple) -> bool:
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh


def rect_to_segments(rect: RectTuple) -> List[Segment]:
    x, y, w, h = rect
    return [
        ((x, y), (x + w, y)),
        ((x + w, y), (x + w, y + h)),
        ((x + w, y + h), (x, y + h)),
        ((x, y + h), (x, y)),
    ]


def ray_segment_intersection(origin: Point, angle: float, a: Point, b: Point) -> Optional[Tuple[float, float, float]]:
    """Return (distance, ix, iy) if a ray intersects a segment; otherwise None."""
    ox, oy = origin
    dx, dy = math.cos(angle), math.sin(angle)
    ax, ay = a
    bx, by = b
    sx, sy = bx - ax, by - ay

    denom = dx * sy - dy * sx
    if abs(denom) < 1e-9:
        return None

    qx, qy = ax - ox, ay - oy
    t = (qx * sy - qy * sx) / denom
    u = (qx * dy - qy * dx) / denom

    if t >= 0.0 and 0.0 <= u <= 1.0:
        return t, ox + t * dx, oy + t * dy
    return None


# ---------------------------------------------------------------------
# Search-and-rescue world
# ---------------------------------------------------------------------


class SearchRescueWorld:
    """A simple 2D damaged-building layout.

    Static obstacles are internal walls, collapsed rubble, and blocked rooms.
    Victim locations are hidden target points. Landmarks are known beacon
    positions used for EKF-SLAM correction.
    """

    def __init__(self, dynamic_block: bool = False):
        self.dynamic_block_enabled = dynamic_block
        self.blockage_active = False

        self.outer_walls: List[Segment] = []
        self.obstacle_rects: List[RectTuple] = []
        self.dynamic_blocks: List[RectTuple] = []
        self.walls: List[Segment] = []

        # Victim dictionaries are updated during simulation.
        self.victims = []
        # Backward-compatible alias so older helper code still works if needed.
        self.plants = self.victims
        self.landmarks: List[Point] = []
        self.build()

    def build(self) -> None:
        margin = 40
        self.outer_walls = [
            ((margin, margin), (WIDTH - margin, margin)),
            ((WIDTH - margin, margin), (WIDTH - margin, HEIGHT - margin)),
            ((WIDTH - margin, HEIGHT - margin), (margin, HEIGHT - margin)),
            ((margin, HEIGHT - margin), (margin, margin)),
        ]

        # Damaged building layout: collapsed internal walls / rubble barriers.
        # The barriers create long corridors and several cross-passages, so the
        # robots have to explore, map, and switch between rooms. This keeps the
        # navigation challenging but still reliable for the demo.
        self.obstacle_rects = [
            # Each wall line is split into three rubble segments.
            # The two horizontal gaps create cross-corridors between rooms.
            (180, 90, 50, 105), (180, 320, 50, 80), (180, 525, 50, 85),
            (340, 90, 50, 105), (340, 320, 50, 80), (340, 525, 50, 85),
            (500, 90, 50, 105), (500, 320, 50, 80), (500, 525, 50, 85),
            (660, 90, 50, 105), (660, 320, 50, 80), (660, 525, 50, 85),
            (820, 90, 50, 105), (820, 320, 50, 80), (820, 525, 50, 85),
        ]

        # Optional collapsed passage. It is only inserted when activated.
        self.dynamic_blocks = []

        # Hidden victims are in rooms/corridors, not inside obstacles.
        victim_positions = [
            (95, 130), (290, 165), (450, 125), (625, 155), (790, 130), (985, 155),
            (100, 350), (285, 365), (455, 345), (615, 365), (790, 345), (985, 365),
            (95, 585), (285, 585), (455, 580), (620, 585), (790, 575), (985, 585),
        ]
        self.victims = [
            {"id": i, "pos": p, "detected": False, "rescued": False}
            for i, p in enumerate(victim_positions)
        ]
        self.plants = self.victims  # keep alias synchronized

        # Sparse beacon-like landmarks used by EKF-SLAM.
        self.landmarks = [
            (70, 70), (1030, 70), (70, 630), (1030, 630),
            (285, 75), (445, 625), (605, 75), (765, 625), (975, 75),
            (285, 350), (605, 350), (975, 350),
        ]
        self.rebuild_segments()

    def reset_victims(self) -> None:
        for victim in self.victims:
            victim["detected"] = False
            victim["rescued"] = False

    # Backward-compatible name used in older experiments.
    def reset_plants(self) -> None:
        self.reset_victims()

    def activate_dynamic_blockage(self) -> None:
        """Collapse one corridor section to test adaptation to map changes."""
        if self.blockage_active:
            return
        self.blockage_active = True
        # A small rubble block that closes a doorway/corridor section.
        self.dynamic_blocks.append((575, 315, 90, 60))
        self.rebuild_segments()

    def rebuild_segments(self) -> None:
        self.walls = list(self.outer_walls)
        for rect in self.obstacle_rects + self.dynamic_blocks:
            self.walls.extend(rect_to_segments(rect))

    def cast_ray(self, origin: Point, angle: float, max_range: float = MAX_SENSOR_RANGE) -> Tuple[float, Point, bool]:
        best_d = max_range
        end = (origin[0] + max_range * math.cos(angle), origin[1] + max_range * math.sin(angle))
        hit = False

        for a, b in self.walls:
            result = ray_segment_intersection(origin, angle, a, b)
            if result is None:
                continue
            d, ix, iy = result
            if d < best_d:
                best_d = d
                end = (ix, iy)
                hit = True

        return best_d, end, hit

    def collides(self, x: float, y: float, radius: float = ROBOT_RADIUS) -> bool:
        # Outer boundaries.
        if x < 40 + radius or x > WIDTH - 40 - radius or y < 40 + radius or y > HEIGHT - 40 - radius:
            return True

        # Internal walls, rubble, and dynamic blockages.
        for rect in self.obstacle_rects + self.dynamic_blocks:
            if circle_intersects_rect(x, y, radius, rect):
                return True
        return False

    def nearest_unrescued_victim(self, pos: Point) -> Optional[dict]:
        candidates = [v for v in self.victims if not v["rescued"]]
        if not candidates:
            return None
        return min(candidates, key=lambda v: distance(pos, v["pos"]))

    # Backward-compatible name used in older helper code.
    def nearest_uncared_plant(self, pos: Point) -> Optional[dict]:
        return self.nearest_unrescued_victim(pos)

    def num_detected(self) -> int:
        return sum(1 for v in self.victims if v["detected"])

    def num_rescued(self) -> int:
        return sum(1 for v in self.victims if v["rescued"])

    # Backward-compatible name used in older helper code.
    def num_cared(self) -> int:
        return self.num_rescued()

    def draw(self, screen, grid=None, show_ground_truth: bool = False) -> None:
        """Draw either the full world or only the discovered part of the world.

        Normal assignment mode uses ``grid`` and hides all cells that have not
        received sensor evidence yet. This gives the intended behaviour: the
        damaged-building map starts unknown and is revealed only by robot motion.

        ``show_ground_truth=True`` is only a debug/recording option. It draws
        the complete building layout and should not be used as the main demo.
        """
        if pygame is None:
            return

        if show_ground_truth or grid is None:
            self._draw_full_world(screen)
        else:
            self._draw_discovered_world(screen, grid)

    def _draw_full_world(self, screen) -> None:
        """Debug view: draw complete damaged-building ground truth."""
        pygame.draw.rect(screen, KNOWN_FLOOR, (40, 40, WIDTH - 80, HEIGHT - 80))

        for rect in self.obstacle_rects:
            pygame.draw.rect(screen, RUBBLE, pygame.Rect(*rect), border_radius=4)
            pygame.draw.rect(screen, GRAY, pygame.Rect(*rect), width=2, border_radius=4)

        for rect in self.dynamic_blocks:
            pygame.draw.rect(screen, DANGER_RED, pygame.Rect(*rect), border_radius=2)
            pygame.draw.rect(screen, BLACK, pygame.Rect(*rect), width=2, border_radius=2)

        for a, b in self.outer_walls:
            pygame.draw.line(screen, BLACK, a, b, 4)

        for lm in self.landmarks:
            pygame.draw.circle(screen, BLACK, (int(lm[0]), int(lm[1])), 5)
            pygame.draw.circle(screen, CYAN, (int(lm[0]), int(lm[1])), 8, 1)

        for victim in self.victims:
            self._draw_victim(screen, victim)

    def _draw_discovered_world(self, screen, grid) -> None:
        """Assignment view: draw only cells that the robots have observed."""
        pygame.draw.rect(screen, UNKNOWN_DARK, (40, 40, WIDTH - 80, HEIGHT - 80))
        pygame.draw.rect(screen, BLACK, (40, 40, WIDTH - 80, HEIGHT - 80), 3)

        # Reveal only cells that have sensor evidence. We use the ground-truth
        # geometry only to color already observed cells for visualization.
        for cy in range(grid.rows):
            for cx in range(grid.cols):
                if not grid.is_cell_known(cx, cy):
                    continue
                wx, wy = grid.cell_to_world(cx, cy)
                rect = pygame.Rect(cx * grid.cell_size, cy * grid.cell_size, grid.cell_size, grid.cell_size)

                color = KNOWN_FLOOR
                if any(point_in_rect(wx, wy, obstacle) for obstacle in self.obstacle_rects):
                    color = RUBBLE
                if any(point_in_rect(wx, wy, block) for block in self.dynamic_blocks):
                    color = DANGER_RED
                pygame.draw.rect(screen, color, rect)

        # Draw landmarks only after their cell has been observed.
        for lm in self.landmarks:
            if grid.is_point_known(lm[0], lm[1]):
                pygame.draw.circle(screen, BLACK, (int(lm[0]), int(lm[1])), 5)
                pygame.draw.circle(screen, CYAN, (int(lm[0]), int(lm[1])), 8, 1)

        # Draw victims only after discovery/rescue. This avoids revealing all
        # target locations at time zero.
        for victim in self.victims:
            if victim["detected"] or victim["rescued"]:
                self._draw_victim(screen, victim)

    def _draw_victim(self, screen, victim) -> None:
        x, y = victim["pos"]
        if victim["rescued"]:
            color = GREEN
        elif victim["detected"]:
            color = YELLOW
        else:
            color = ORANGE
        pygame.draw.circle(screen, color, (int(x), int(y)), 10)
        pygame.draw.circle(screen, BLACK, (int(x), int(y)), 10, 1)
        # Small cross makes victims visually different from simple landmarks.
        pygame.draw.line(screen, BLACK, (int(x - 5), int(y)), (int(x + 5), int(y)), 1)
        pygame.draw.line(screen, BLACK, (int(x), int(y - 5)), (int(x), int(y + 5)), 1)


# Backward compatibility: older imports still work, but the scenario is now search-and-rescue.
GreenhouseWorld = SearchRescueWorld