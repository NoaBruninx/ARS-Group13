from __future__ import annotations

import math
import random
from collections import deque
from typing import List, Optional, Tuple

try:
    import pygame
except ImportError:
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


class SearchRescueWorld:
    """Randomized search-and-rescue world with reachable random victims."""

    def __init__(
        self,
        dynamic_block: bool = False,
        seed: Optional[int] = None,
        num_victims: int = 15,
    ):
        self.dynamic_block_enabled = dynamic_block
        self.blockage_active = False
        self.seed = seed
        self.rng = random.Random(seed)
        self.num_victims = num_victims

        self.outer_walls: List[Segment] = []
        self.obstacle_rects: List[RectTuple] = []
        self.dynamic_blocks: List[RectTuple] = []
        self.walls: List[Segment] = []

        self.victims: List[dict] = []
        self.plants = self.victims
        self.landmarks: List[Point] = []

        self.start_hint: Point = (95, 100)
        self.build()

    def build(self) -> None:
        margin = 40
        self.outer_walls = [
            ((margin, margin), (WIDTH - margin, margin)),
            ((WIDTH - margin, margin), (WIDTH - margin, HEIGHT - margin)),
            ((WIDTH - margin, HEIGHT - margin), (margin, HEIGHT - margin)),
            ((margin, HEIGHT - margin), (margin, margin)),
        ]

        self.landmarks = [
            (70, 70), (1030, 70), (70, 630), (1030, 630),
            (285, 75), (445, 625), (605, 75), (765, 625), (975, 75),
            (285, 350), (605, 350), (975, 350),
        ]

        self.dynamic_blocks = []
        self.blockage_active = False
        self.victims = []
        self.plants = self.victims

        self._generate_random_map()

        victim_positions = self._generate_random_victim_positions(self.num_victims)
        self.victims = [
            {"id": i, "pos": p, "detected": False, "rescued": False}
            for i, p in enumerate(victim_positions)
        ]
        self.plants = self.victims

        if self.dynamic_block_enabled and self.rng.random() < 0.5:
            self.activate_dynamic_blockage()

        self.rebuild_segments()

    def regenerate(self) -> None:
        self.blockage_active = False
        self.build()

    def _generate_random_map(self, max_tries: int = 250) -> None:
        for _ in range(max_tries):
            candidate = self._make_random_obstacles()
            if self._start_has_reachable_space(candidate):
                self.obstacle_rects = candidate
                return
        raise RuntimeError("Could not generate a valid random map.")

    def _start_has_reachable_space(self, rects: List[RectTuple]) -> bool:
        reachable = self._reachable_cells(rects, [])
        return len(reachable) >= max(80, self.num_victims * 6)

    def _make_random_obstacles(self) -> List[RectTuple]:
        rects: List[RectTuple] = []

        x_positions = [180, 340, 500, 660, 820]
        wall_w = 50

        for x in x_positions:
            y_min, y_max = 90, 610

            possible_gaps = [
                (205, 315),
                (400, 520),
            ]
            if self.rng.random() < 0.5:
                g0 = self.rng.randint(120, 170)
                g1 = self.rng.randint(180, 235)
                if g1 > g0:
                    possible_gaps.append((g0, g1))
            if self.rng.random() < 0.5:
                g0 = self.rng.randint(540, 565)
                g1 = self.rng.randint(585, 610)
                if g1 > g0:
                    possible_gaps.append((g0, g1))

            possible_gaps.sort()
            gaps: List[List[int]] = []
            for g0, g1 in possible_gaps:
                g0, g1 = max(y_min, g0), min(y_max, g1)
                if g1 - g0 < 35:
                    continue
                if not gaps or g0 > gaps[-1][1]:
                    gaps.append([g0, g1])
                else:
                    gaps[-1][1] = max(gaps[-1][1], g1)

            current = y_min
            for g0, g1 in gaps:
                if g0 - current >= 35:
                    rects.append((x, current, wall_w, g0 - current))
                current = g1
            if y_max - current >= 35:
                rects.append((x, current, wall_w, y_max - current))

        extra_blocks = self.rng.randint(3, 8)
        attempts = 0
        while extra_blocks > 0 and attempts < 300:
            attempts += 1
            rw = self.rng.randint(30, 70)
            rh = self.rng.randint(30, 70)
            rx = self.rng.randint(80, WIDTH - 80 - rw)
            ry = self.rng.randint(80, HEIGHT - 80 - rh)
            rect = (rx, ry, rw, rh)

            if self._rect_is_safe_to_place(rect, rects):
                rects.append(rect)
                extra_blocks -= 1

        return rects

    def _rect_is_safe_to_place(self, rect: RectTuple, existing: List[RectTuple]) -> bool:
        rx, ry, rw, rh = rect

        protected_points = [self.start_hint] + self.landmarks

        inflated = (rx - 20, ry - 20, rw + 40, rh + 40)
        for px, py in protected_points:
            if point_in_rect(px, py, inflated):
                return False

        for other in existing:
            ox, oy, ow, oh = other
            overlaps = not (
                rx + rw < ox - 15 or
                ox + ow < rx - 15 or
                ry + rh < oy - 15 or
                oy + oh < ry - 15
            )
            if overlaps:
                return False

        return True

    def _reachable_cells(self, obstacle_rects: List[RectTuple], dynamic_rects: List[RectTuple]) -> List[Tuple[int, int]]:
        cell = 20
        cols = WIDTH // cell
        rows = HEIGHT // cell

        def cell_center(cx: int, cy: int) -> Point:
            return (cx * cell + cell / 2, cy * cell + cell / 2)

        blocked = [[False for _ in range(rows)] for _ in range(cols)]

        for cx in range(cols):
            for cy in range(rows):
                wx, wy = cell_center(cx, cy)

                if wx < 40 + ROBOT_RADIUS or wx > WIDTH - 40 - ROBOT_RADIUS:
                    blocked[cx][cy] = True
                    continue
                if wy < 40 + ROBOT_RADIUS or wy > HEIGHT - 40 - ROBOT_RADIUS:
                    blocked[cx][cy] = True
                    continue

                for rect in obstacle_rects + dynamic_rects:
                    if circle_intersects_rect(wx, wy, ROBOT_RADIUS + 4, rect):
                        blocked[cx][cy] = True
                        break

        sx = int(self.start_hint[0] // cell)
        sy = int(self.start_hint[1] // cell)
        sx = max(0, min(cols - 1, sx))
        sy = max(0, min(rows - 1, sy))

        if blocked[sx][sy]:
            return []

        q = deque([(sx, sy)])
        seen = {(sx, sy)}

        while q:
            cx, cy = q.popleft()
            for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                if 0 <= nx < cols and 0 <= ny < rows and not blocked[nx][ny] and (nx, ny) not in seen:
                    seen.add((nx, ny))
                    q.append((nx, ny))

        return list(seen)

    def _generate_random_victim_positions(self, count: int = 15) -> List[Point]:
        cell = 20

        def cell_center(cell_pos: Tuple[int, int]) -> Point:
            cx, cy = cell_pos
            return (cx * cell + cell / 2, cy * cell + cell / 2)

        reachable_cells = self._reachable_cells(self.obstacle_rects, self.dynamic_blocks)
        if not reachable_cells:
            raise RuntimeError("No reachable free area found for victim placement.")

        candidate_points: List[Point] = []
        for rc in reachable_cells:
            p = cell_center(rc)

            if distance(p, self.start_hint) < 45:
                continue
            if any(distance(p, lm) < 28 for lm in self.landmarks):
                continue
            if self.collides(p[0], p[1], radius=20):
                continue

            candidate_points.append(p)

        self.rng.shuffle(candidate_points)

        chosen: List[Point] = []
        for p in candidate_points:
            if any(distance(p, other) < 45 for other in chosen):
                continue
            chosen.append(p)
            if len(chosen) >= count:
                return chosen

        raise RuntimeError("Could not place all random victims on reachable free cells.")

    def _all_victims_reachable_with_dynamic(self, candidate_dynamic: List[RectTuple]) -> bool:
        cell = 20

        def point_to_cell(p: Point) -> Tuple[int, int]:
            cx = int(p[0] // cell)
            cy = int(p[1] // cell)
            return cx, cy

        seen = set(self._reachable_cells(self.obstacle_rects, candidate_dynamic))
        if not seen:
            return False

        for victim in self.victims:
            if point_to_cell(victim["pos"]) not in seen:
                return False
        return True

    def reset_victims(self) -> None:
        for victim in self.victims:
            victim["detected"] = False
            victim["rescued"] = False

    def reset_plants(self) -> None:
        self.reset_victims()

    def activate_dynamic_blockage(self) -> None:
        if self.blockage_active:
            return

        possible_blocks = [
            (575, 315, 90, 60),
            (255, 315, 70, 60),
            (735, 315, 70, 60),
        ]
        self.rng.shuffle(possible_blocks)

        for block in possible_blocks:
            candidate_dynamic = self.dynamic_blocks + [block]
            if self._all_victims_reachable_with_dynamic(candidate_dynamic):
                self.dynamic_blocks.append(block)
                self.blockage_active = True
                self.rebuild_segments()
                return

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
        if x < 40 + radius or x > WIDTH - 40 - radius or y < 40 + radius or y > HEIGHT - 40 - radius:
            return True

        for rect in self.obstacle_rects + self.dynamic_blocks:
            if circle_intersects_rect(x, y, radius, rect):
                return True
        return False

    def nearest_unrescued_victim(self, pos: Point) -> Optional[dict]:
        candidates = [v for v in self.victims if not v["rescued"]]
        if not candidates:
            return None
        return min(candidates, key=lambda v: distance(pos, v["pos"]))

    def nearest_uncared_plant(self, pos: Point) -> Optional[dict]:
        return self.nearest_unrescued_victim(pos)

    def num_detected(self) -> int:
        return sum(1 for v in self.victims if v["detected"])

    def num_rescued(self) -> int:
        return sum(1 for v in self.victims if v["rescued"])

    def num_cared(self) -> int:
        return self.num_rescued()

    def draw(self, screen, grid=None, show_ground_truth: bool = False) -> None:
        if pygame is None:
            return

        if show_ground_truth or grid is None:
            self._draw_full_world(screen)
        else:
            self._draw_discovered_world(screen, grid)

    def _draw_full_world(self, screen) -> None:
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
        pygame.draw.rect(screen, UNKNOWN_DARK, (40, 40, WIDTH - 80, HEIGHT - 80))
        pygame.draw.rect(screen, BLACK, (40, 40, WIDTH - 80, HEIGHT - 80), 3)

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

        for lm in self.landmarks:
            if grid.is_point_known(lm[0], lm[1]):
                pygame.draw.circle(screen, BLACK, (int(lm[0]), int(lm[1])), 5)
                pygame.draw.circle(screen, CYAN, (int(lm[0]), int(lm[1])), 8, 1)

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
        pygame.draw.line(screen, BLACK, (int(x - 5), int(y)), (int(x + 5), int(y)), 1)
        pygame.draw.line(screen, BLACK, (int(x), int(y - 5)), (int(x), int(y + 5)), 1)


GreenhouseWorld = SearchRescueWorld