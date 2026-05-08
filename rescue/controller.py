"""
controller.py
Behaviour-based controller with evolved weight vector (Genome).

Controller architecture (report-friendly explanation):
  The controller combines multiple behaviour vectors, each producing a
  direction in 2D space. The Genome is a real-valued weight vector that
  controls the contribution of each behaviour — conceptually identical to
  the weights of a linear output layer in an ANN. GENITOR evolves these
  weights to maximise rescue performance.

  Behaviours (= "neurons" in the behaviour layer):
    1. Obstacle avoidance   — repels from nearby walls
    2. Victim attraction    — pulls toward detected unrescued victims
    3. Frontier exploration — pulls toward unknown map regions
    4. Robot separation     — keeps robots spread apart
    5. Corridor following   — aligns with open passages
    6. Corridor switching   — moves between corridor levels
    7. Visited repulsion    — avoids already-rescued locations
    8. Zone coverage        — systematic column-by-column sweep

  This is equivalent to a single-layer linear network whose inputs are
  behaviour vectors and whose output is the desired heading direction.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Set

from world import (
    WIDTH, HEIGHT,
    MAX_SENSOR_RANGE,
    SENSOR_REL_ANGLES,
    distance,
    wrap_angle,
    clamp,
)


Vector = Tuple[float, float]


def _unit_from_heading(heading: float) -> Vector:
    return math.cos(heading), math.sin(heading)


def _add(v1: Vector, v2: Vector, scale: float = 1.0) -> Vector:
    return v1[0] + scale * v2[0], v1[1] + scale * v2[1]


def _norm(v: Vector) -> float:
    return math.hypot(v[0], v[1])


def _normalize(v: Vector) -> Vector:
    n = _norm(v)
    if n <= 1e-9:
        return 0.0, 0.0
    return v[0] / n, v[1] / n


def _heading_from_vector(v: Vector) -> Optional[float]:
    if _norm(v) <= 1e-8:
        return None
    return math.atan2(v[1], v[0])


@dataclass
class Genome:
    # Motion parameters
    forward_speed: float = 56.0
    turn_speed: float = 2.7
    obstacle_threshold: float = 95.0

    # Behaviour weights, evolved by the GA
    victim_attraction_weight: float = 1.2
    frontier_weight: float = 0.85
    wall_avoidance_weight: float = 3.0
    robot_separation_weight: float = 0.9
    visited_victim_penalty: float = 1.0
    corridor_following_weight: float = 0.65
    unknown_area_weight: float = 0.9
    corridor_switching_weight: float = 4.0

    @staticmethod
    def random(rng: random.Random) -> "Genome":
        return Genome(
            forward_speed=rng.uniform(35.0, 95.0),
            turn_speed=rng.uniform(1.0, 3.8),
            obstacle_threshold=rng.uniform(65.0, 125.0),
            victim_attraction_weight=rng.uniform(0.1, 4.0),
            frontier_weight=rng.uniform(0.1, 4.0),
            wall_avoidance_weight=rng.uniform(0.2, 4.0),
            robot_separation_weight=rng.uniform(0.0, 4.0),
            visited_victim_penalty=rng.uniform(0.0, 4.0),
            corridor_following_weight=rng.uniform(0.0, 3.0),
            unknown_area_weight=rng.uniform(0.1, 4.0),
            corridor_switching_weight=rng.uniform(0.0, 4.0),
        )

    def mutate(self, rng: random.Random, sigma: float = 0.10) -> "Genome":
        ranges = {
            "forward_speed": (25.0, 105.0),
            "turn_speed": (0.8, 4.2),
            "obstacle_threshold": (55.0, 135.0),
            "victim_attraction_weight": (0.0, 5.0),
            "frontier_weight": (0.0, 5.0),
            "wall_avoidance_weight": (0.0, 5.0),
            "robot_separation_weight": (0.0, 5.0),
            "visited_victim_penalty": (0.0, 5.0),
            "corridor_following_weight": (0.0, 4.0),
            "unknown_area_weight": (0.0, 5.0),
            "corridor_switching_weight": (0.0, 5.0),
        }
        data = asdict(self)
        for key, (lo, hi) in ranges.items():
            data[key] = clamp(data[key] + rng.gauss(0.0, sigma * (hi - lo)), lo, hi)
        return Genome(**data)

    @staticmethod
    def crossover(a: "Genome", b: "Genome", rng: random.Random) -> "Genome":
        da, db = asdict(a), asdict(b)
        child = {}
        for key in da:
            child[key] = da[key] if rng.random() < 0.5 else db[key]
        return Genome(**child)

    def to_dict(self) -> dict:
        return asdict(self)


# Search-and-rescue corridor x-positions used for coverage targets
CORRIDOR_X_POSITIONS = [90, 265, 425, 585, 745, 905, 1010]

# Cross-corridor y-bands used to move between rooms
CROSS_CORRIDOR_Y = [255.0, 455.0]


class SearchRescueController:
    """Behaviour-based vector controller.

    Each behaviour produces a vector in world coordinates. The evolved genome
    controls how strongly each vector contributes to the final desired heading.
    """

    def __init__(self, genome: Genome):
        self.g = genome
        self.last_turn_direction = 1.0
        self.last_heading: Optional[float] = None
        self.frontier_target: Optional[Vector] = None
        self.victim_target_id: Optional[int] = None
        self.recovery_timer = 0
        self.corridor_switch_direction = 1.0
        self.corridor_switch_timer = 0

        # --- Anti-loop / systematic coverage state ---
        self.visited_zones: Set[int] = set()
        self.zone_timer = 0
        self.zone_stuck_timer = 0
        self.last_zone_x = 0.0
        self.last_zone_y = 0.0
        self.forced_target: Optional[Vector] = None
        self.forced_timer = 0

        # Rescue robot explore mode (when no known victims to go to)
        self.no_new_victim_timer = 0
        self.explore_mode = False

    def control(self, robot, grid, world, robots, rng: random.Random):
        """Return linear and angular velocity commands (v, w)."""
        x, y, th = robot.ekf.mu

        # Update systematic coverage tracking
        self._update_zone_tracking(robot, world)

        # Recovery from stuck
        if getattr(robot, "stuck_counter", 0) > 25:
            recovery_len = 100 if robot.role == "rescue" else 70
            self.recovery_timer = recovery_len
            self.last_turn_direction *= -1.0
            self.frontier_target = None
            self.victim_target_id = None
            self.forced_target = None
            robot.stuck_counter = 0

        if self.recovery_timer > 0:
            self.recovery_timer -= 1
            half = self.recovery_timer > (50 if robot.role == "rescue" else 35)
            v = -0.28 * self.g.forward_speed if half else 0.35 * self.g.forward_speed
            w = self.last_turn_direction * self.g.turn_speed * 1.2
            return v, w

        # Emergency obstacle avoidance
        front = self._front_distance(robot)
        if front < 0.95 * self.g.obstacle_threshold:
            left = min(robot.sensor_distances[1], robot.sensor_distances[2], robot.sensor_distances[3])
            right = min(robot.sensor_distances[-1], robot.sensor_distances[-2], robot.sensor_distances[-3])
            self.last_turn_direction = 1.0 if left > right else -1.0
            v = -0.16 * self.g.forward_speed if front < 0.45 * self.g.obstacle_threshold else 0.18 * self.g.forward_speed
            return v, self.last_turn_direction * self.g.turn_speed

        # Forced target (anti-loop override)
        # Cancel forced_target for rescue robot when it has known victims to visit
        if self.forced_target is not None and robot.role == "rescue":
            unrescued_now = [p for p in world.victims if p["detected"] and not p["rescued"]]
            if unrescued_now:
                self.forced_target = None
                self.forced_timer = 0

        if self.forced_target is not None and self.forced_timer > 0:
            self.forced_timer -= 1
            fx, fy = self.forced_target
            d = distance((x, y), (fx, fy))
            if d < 60.0 or self.forced_timer == 0:
                self.forced_target = None
                self.forced_timer = 0
            else:
                heading = math.atan2(fy - y, fx - x)
                heading_error = wrap_angle(heading - th)
                w = clamp(self.g.turn_speed * heading_error, -self.g.turn_speed, self.g.turn_speed)
                obstacle_risk = clamp((self.g.obstacle_threshold - front) / max(self.g.obstacle_threshold, 1.0), 0.0, 1.0)
                v = self.g.forward_speed * clamp(1.0 - 0.65 * obstacle_risk, 0.25, 1.0)
                return v, w

        # Compute all behaviour vectors
        victim_vec = self._victim_vector(robot, world)
        frontier_vec = self._frontier_vector(robot, grid, rng)
        obstacle_vec = self._obstacle_avoidance_vector(robot)
        separation_vec = self._separation_vector(robot, robots)
        corridor_follow_vec = self._corridor_following_vector(robot)
        corridor_switch_vec = self._corridor_switching_vector(robot)
        visited_vec = self._visited_victim_repulsion_vector(robot, world)
        zone_vec = self._zone_coverage_vector(robot, grid)

        # Role-dependent scaling
        if robot.role == "scout":
            role_victim = 0.45
            role_frontier = 1.45
            role_zone = 2.5
        else:
            unrescued = [p for p in world.victims if p["detected"] and not p["rescued"]]
            if not unrescued or self.explore_mode:
                # No victims known: explore aggressively
                role_victim = 0.3
                role_frontier = 1.8
                role_zone = 2.5
            else:
                # Has targets: strongly pursue victims, cancel zone/corridor pulls
                role_victim = 6.0
                role_frontier = 0.1
                role_zone = 0.0
                # Force rescue to head directly to nearest victim
                x, y, th = robot.ekf.mu
                nearest = min(unrescued, key=lambda v: distance((x, y), v["pos"]))
                nx, ny = nearest["pos"]
                if distance((x, y), (nx, ny)) > 30.0:
                    heading = math.atan2(ny - y, nx - x)
                    heading_error = wrap_angle(heading - th)
                    front = self._front_distance(robot)

                    # If stuck_counter is high, wall-follow instead of going straight
                    if getattr(robot, "stuck_counter", 0) > 10 or self.recovery_timer > 0:
                        pass  # fall through to normal vector sum so obstacle avoidance helps
                    else:
                        obstacle_risk = clamp(
                            (self.g.obstacle_threshold - front) / max(self.g.obstacle_threshold, 1.0),
                            0.0, 1.0)
                        # If wall directly ahead but not stuck yet, blend toward open side
                        if obstacle_risk > 0.6:
                            left  = min(robot.sensor_distances[2], robot.sensor_distances[3], robot.sensor_distances[4])
                            right = min(robot.sensor_distances[8], robot.sensor_distances[9], robot.sensor_distances[10])
                            side_dir = 1.0 if left > right else -1.0
                            w = side_dir * self.g.turn_speed
                            v = self.g.forward_speed * 0.3
                            return v, w
                        w = clamp(self.g.turn_speed * heading_error * 1.5,
                                  -self.g.turn_speed, self.g.turn_speed)
                        v = self.g.forward_speed * clamp(1.0 - 0.6 * obstacle_risk, 0.2, 1.0)
                        return v, w

        total = (0.0, 0.0)
        total = _add(total, _unit_from_heading(th), 0.35)
        total = _add(total, victim_vec, self.g.victim_attraction_weight * role_victim)
        total = _add(total, frontier_vec, self.g.frontier_weight * self.g.unknown_area_weight * role_frontier)
        total = _add(total, obstacle_vec, self.g.wall_avoidance_weight)
        total = _add(total, separation_vec, self.g.robot_separation_weight)
        total = _add(total, corridor_follow_vec, self.g.corridor_following_weight)
        total = _add(total, corridor_switch_vec, self.g.corridor_switching_weight)
        total = _add(total, visited_vec, self.g.visited_victim_penalty)
        total = _add(total, zone_vec, role_zone * 0.75)  # tuned: balance with frontier

        desired_heading = _heading_from_vector(total)
        if desired_heading is None:
            desired_heading = wrap_angle(th + 0.4 * self.last_turn_direction)

        self.last_heading = desired_heading
        heading_error = wrap_angle(desired_heading - th)
        self.last_turn_direction = 1.0 if heading_error >= 0 else -1.0

        w = clamp(self.g.turn_speed * heading_error, -self.g.turn_speed, self.g.turn_speed)

        front = self._front_distance(robot)
        obstacle_risk = clamp((self.g.obstacle_threshold - front) / max(self.g.obstacle_threshold, 1.0), 0.0, 1.0)
        turn_risk = clamp(abs(heading_error) / 1.6, 0.0, 1.0)
        speed_factor = clamp(1.0 - 0.65 * obstacle_risk - 0.45 * turn_risk, 0.18, 1.0)
        v = self.g.forward_speed * speed_factor
        return v, w

    # ------------------------------------------------------------------
    # Anti-loop zone tracking
    # ------------------------------------------------------------------

    def _update_zone_tracking(self, robot, world) -> None:
        """Track zones visited and force movement to unexplored zones."""
        x, y, th = robot.ekf.mu

        # Track current x-zone
        closest_zone = min(range(len(CORRIDOR_X_POSITIONS)),
                           key=lambda i: abs(CORRIDOR_X_POSITIONS[i] - x))
        self.visited_zones.add(closest_zone)
        self.zone_timer += 1

        # Detect if robot is moving very little (looping)
        if abs(x - self.last_zone_x) < 8.0 and abs(y - self.last_zone_y) < 8.0:
            self.zone_stuck_timer += 1
        else:
            self.zone_stuck_timer = max(0, self.zone_stuck_timer - 2)
            self.last_zone_x = x
            self.last_zone_y = y

        # Stuck in one small area -> force move to unexplored zone
        if self.zone_stuck_timer > 300 and self.forced_target is None:
            self.zone_stuck_timer = 0
            target = self._pick_forced_target(robot, world)
            if target is not None:
                self.forced_target = target
                self.forced_timer = 400
                self.frontier_target = None
                self.victim_target_id = None

        # Every ~15 sim-seconds, check if there are unvisited zones to push to
        if self.zone_timer > 900 and self.forced_target is None:
            self.zone_timer = 0
            unvisited = [i for i in range(len(CORRIDOR_X_POSITIONS))
                         if i not in self.visited_zones]
            if unvisited:
                target_zone = max(unvisited, key=lambda i: abs(CORRIDOR_X_POSITIONS[i] - x))
                tx = CORRIDOR_X_POSITIONS[target_zone]
                ty = 120.0 if y > HEIGHT / 2 else 580.0
                self.forced_target = (tx, ty)
                self.forced_timer = 500
                self.frontier_target = None
                self.victim_target_id = None

        # Rescue robot explore mode when no victims are known
        if robot.role == "rescue":
            unrescued = [p for p in world.victims if p["detected"] and not p["rescued"]]
            if not unrescued:
                self.no_new_victim_timer += 1
            else:
                self.no_new_victim_timer = 0
                self.explore_mode = False
            if self.no_new_victim_timer > 600:
                self.explore_mode = True

    def _pick_forced_target(self, robot, world) -> Optional[Vector]:
        """Pick a forced navigation target prioritizing undetected/unrescued victims."""
        x, y, th = robot.ekf.mu

        # For rescue robot: go directly to a detected unrescued victim
        if robot.role == "rescue":
            unrescued = [p for p in world.victims if p["detected"] and not p["rescued"]]
            if unrescued:
                target = min(unrescued, key=lambda p: distance((x, y), p["pos"]))
                return target["pos"]

        # Head toward undetected victim positions (approximate corridor positions)
        undetected = []
        for victim in world.victims:
            if not victim["detected"]:
                px, py = victim["pos"]
                undetected.append((px, py))
        if undetected:
            return max(undetected, key=lambda p: distance((x, y), p))

        # All detected: go to unvisited zone corners
        unvisited = [i for i in range(len(CORRIDOR_X_POSITIONS))
                     if i not in self.visited_zones]
        if unvisited:
            target_zone = max(unvisited, key=lambda i: abs(CORRIDOR_X_POSITIONS[i] - x))
            tx = CORRIDOR_X_POSITIONS[target_zone]
            ty = 90.0 if y > HEIGHT / 2 else 610.0
            return (tx, ty)

        corners = [(90, 90), (1010, 90), (90, 610), (1010, 610)]
        return max(corners, key=lambda c: distance((x, y), c))

    # ------------------------------------------------------------------
    # Zone coverage vector (key anti-loop fix)
    # ------------------------------------------------------------------

    def _zone_coverage_vector(self, robot, grid) -> Vector:
        """Pull robot toward least-explored zones (columns AND top/bottom rows).

        Combines horizontal pull (toward unexplored column) with vertical pull
        (toward top or bottom of corridor) to ensure full coverage including
        victims at y~135 and y~575 that cross-corridors alone cannot reach.
        """
        x, y, th = robot.ekf.mu

        # Don't override rescue robot heading to a specific victim
        if self.victim_target_id is not None and robot.role == "rescue":
            return 0.0, 0.0

        # --- Horizontal: find least-explored column ---
        zone_scores = []
        for i, ax in enumerate(CORRIDOR_X_POSITIONS):
            unexplored = 0
            total_cells = 0
            for ay in range(80, HEIGHT - 80, 40):
                cx, cy = grid.world_to_cell(ax, ay)
                if grid.in_bounds(cx, cy):
                    total_cells += 1
                    if not grid.is_cell_known(cx, cy):
                        unexplored += 1
            unexplored_ratio = unexplored / total_cells if total_cells > 0 else 1.0
            d = abs(ax - x)
            score = unexplored_ratio * 1.5 - d / 600.0
            if abs(ax - x) < 80:
                score -= 1.0
            zone_scores.append((score, ax))

        vx = 0.0
        if zone_scores:
            zone_scores.sort(reverse=True)
            best_score, best_ax = zone_scores[0]
            if best_score >= 0.15:
                dx = best_ax - x
                if abs(dx) >= 50:
                    strength = clamp(abs(dx) / 400.0, 0.2, 1.2)
                    vx = (1.0 if dx > 0 else -1.0) * strength

        # --- Vertical: push toward top (y~90) or bottom (y~620) when in middle ---
        # Top row victims at y=135, bottom row victims at y=575.
        # Robots looping in cross-corridor band (y=200-500) never reach them.
        # Count how much of the top and bottom bands are unexplored.
        top_unexplored = 0
        bot_unexplored = 0
        top_total = 0
        bot_total = 0
        for ax in range(80, WIDTH - 80, 30):
            cx, cy_top = grid.world_to_cell(ax, 120)
            cx, cy_bot = grid.world_to_cell(ax, 580)
            if grid.in_bounds(cx, cy_top):
                top_total += 1
                if not grid.is_cell_known(cx, cy_top):
                    top_unexplored += 1
            if grid.in_bounds(cx, cy_bot):
                bot_total += 1
                if not grid.is_cell_known(cx, cy_bot):
                    bot_unexplored += 1

        top_ratio = top_unexplored / top_total if top_total > 0 else 1.0
        bot_ratio = bot_unexplored / bot_total if bot_total > 0 else 1.0

        vy = 0.0
        # Only push vertically if we are in the middle band
        if y > 150 and y < 550:
            if top_ratio > bot_ratio and top_ratio > 0.2:
                # Push up toward top row
                dy = 90.0 - y
                vy = clamp(dy / 250.0, -1.0, -0.3)
            elif bot_ratio > 0.2:
                # Push down toward bottom row
                dy = 620.0 - y
                vy = clamp(dy / 250.0, 0.3, 1.0)

        if abs(vx) < 0.01 and abs(vy) < 0.01:
            return 0.0, 0.0
        return vx, vy

    # ------------------------------------------------------------------
    # Behaviour vectors (unchanged from original)
    # ------------------------------------------------------------------

    def _point_roughly_visible(self, robot, point: Vector, margin: float = 18.0) -> bool:
        x, y, th = robot.ekf.mu
        px, py = point
        d = distance((x, y), (px, py))
        rel = wrap_angle(math.atan2(py - y, px - x) - th)
        best_idx = min(range(len(SENSOR_REL_ANGLES)), key=lambda i: abs(wrap_angle(SENSOR_REL_ANGLES[i] - rel)))
        return robot.sensor_distances[best_idx] + margin >= d

    def _victim_vector(self, robot, world) -> Vector:
        """Attract toward detected unrescued victims.

        For rescue robot: always target the nearest unrescued victim.
        For scout: use angle-weighted scoring to prefer victims ahead.
        """
        x, y, th = robot.ekf.mu
        is_rescue = (robot.role == "rescue")

        # Check if current target is still valid
        if self.victim_target_id is not None:
            for victim in world.victims:
                if victim["id"] == self.victim_target_id and victim["detected"] and not victim["rescued"]:
                    px, py = victim["pos"]
                    d = distance((x, y), (px, py))
                    if d > 22.0:
                        # Rescue robot: switch to closer victim if one appears
                        if is_rescue:
                            closer = [v for v in world.victims
                                      if v["detected"] and not v["rescued"]
                                      and distance((x, y), v["pos"]) < d - 50.0]
                            if closer:
                                self.victim_target_id = None
                                break
                        heading = math.atan2(py - y, px - x)
                        strength = clamp(300.0 / max(d, 25.0), 0.4, 3.0)
                        ux, uy = _unit_from_heading(heading)
                        return ux * strength, uy * strength
                    self.victim_target_id = None
                    break
            self.victim_target_id = None

        candidates = []
        for victim in world.victims:
            if victim["rescued"] or not victim["detected"]:
                continue
            px, py = victim["pos"]
            d = distance((x, y), (px, py))
            if d < 18.0:
                continue
            heading = math.atan2(py - y, px - x)
            if is_rescue:
                # Rescue: purely nearest first, ignore angle
                score = d
            else:
                # Scout: prefer victims roughly ahead
                angle_penalty = abs(wrap_angle(heading - th))
                score = d + 45.0 * angle_penalty
            candidates.append((score, victim["id"], d, heading))

        if not candidates:
            self.victim_target_id = None
            return 0.0, 0.0
        candidates.sort(key=lambda item: item[0])
        _, victim_id, d, heading = candidates[0]
        self.victim_target_id = victim_id
        strength = clamp(300.0 / max(d, 25.0), 0.4, 3.0)
        ux, uy = _unit_from_heading(heading)
        return ux * strength, uy * strength

    def _frontier_vector(self, robot, grid, rng: random.Random) -> Vector:
        x, y, th = robot.ekf.mu
        frontiers = grid.get_frontiers(max_points=350)
        if not frontiers:
            return 0.0, 0.0

        near_cross_corridor = self._near_cross_corridor(y)
        if self.frontier_target is not None and not near_cross_corridor:
            fx, fy = self.frontier_target
            d = distance((x, y), (fx, fy))
            if d > 45.0 and self._point_roughly_visible(robot, (fx, fy), margin=30.0):
                heading = math.atan2(fy - y, fx - x)
                ux, uy = _unit_from_heading(heading)
                strength = clamp(240.0 / max(d, 35.0), 0.30, 2.1)
                return ux * strength, uy * strength
            self.frontier_target = None
        elif near_cross_corridor:
            self.frontier_target = None

        best = None
        for fx, fy in frontiers:
            d = distance((x, y), (fx, fy))
            if d < 45.0:
                continue
            if not self._point_roughly_visible(robot, (fx, fy), margin=30.0):
                continue
            heading = math.atan2(fy - y, fx - x)
            angle_penalty = abs(wrap_angle(heading - th))
            lateral_bonus = min(abs(fx - x), 260.0)
            cross_corridor_bonus = 80.0 if self._near_cross_corridor(fy) else 0.0
            # Bonus for frontiers in unvisited zones
            zone_idx = min(range(len(CORRIDOR_X_POSITIONS)),
                           key=lambda i: abs(CORRIDOR_X_POSITIONS[i] - fx))
            unvisited_bonus = 120.0 if zone_idx not in self.visited_zones else 0.0
            score = (0.70 * d + 90.0 * angle_penalty
                     - 0.45 * lateral_bonus
                     - 1.3 * cross_corridor_bonus
                     - 1.5 * unvisited_bonus)
            if best is None or score < best[0]:
                best = (score, d, heading, (fx, fy))

        if best is None:
            return 0.0, 0.0
        _, d, heading, target = best
        self.frontier_target = target
        strength = clamp(240.0 / max(d, 35.0), 0.30, 2.1)
        ux, uy = _unit_from_heading(heading)
        return ux * strength, uy * strength

    def _obstacle_avoidance_vector(self, robot) -> Vector:
        x, y, th = robot.ekf.mu
        total = (0.0, 0.0)
        for rel_angle, d in zip(SENSOR_REL_ANGLES, robot.sensor_distances):
            if d >= self.g.obstacle_threshold:
                continue
            sensor_heading = th + rel_angle
            repel_heading = sensor_heading + math.pi
            risk = clamp((self.g.obstacle_threshold - d) / max(self.g.obstacle_threshold, 1.0), 0.0, 1.0)
            ux, uy = _unit_from_heading(repel_heading)
            front_bonus = 1.8 if abs(wrap_angle(rel_angle)) < math.radians(65) else 1.0
            total = _add(total, (ux, uy), front_bonus * 3.0 * risk * risk)
        return total

    def _separation_vector(self, robot, robots) -> Vector:
        x, y = robot.ekf.mu[0], robot.ekf.mu[1]
        total = (0.0, 0.0)
        for other in robots:
            if other is robot:
                continue
            ox, oy = other.ekf.mu[0], other.ekf.mu[1]
            d = distance((x, y), (ox, oy))
            if d > 140.0 or d <= 1e-6:
                continue
            heading_away = math.atan2(y - oy, x - ox)
            strength = clamp((140.0 - d) / 140.0, 0.0, 1.0)
            ux, uy = _unit_from_heading(heading_away)
            total = _add(total, (ux, uy), 2.2 * strength)
        return total

    def _corridor_following_vector(self, robot) -> Vector:
        th = robot.ekf.mu[2]
        preferred = [0, 1, 2, len(robot.sensor_distances) - 2, len(robot.sensor_distances) - 1]
        best_idx = max(preferred, key=lambda i: robot.sensor_distances[i])
        best_distance = robot.sensor_distances[best_idx]

        if best_distance < 0.55 * MAX_SENSOR_RANGE:
            side = [3, 4, len(robot.sensor_distances) - 4, len(robot.sensor_distances) - 3]
            best_idx = max(side, key=lambda i: robot.sensor_distances[i])
            best_distance = robot.sensor_distances[best_idx]

        if best_distance < 0.40 * MAX_SENSOR_RANGE:
            return 0.0, 0.0
        heading = th + SENSOR_REL_ANGLES[best_idx]
        strength = clamp(best_distance / MAX_SENSOR_RANGE, 0.0, 1.0)
        ux, uy = _unit_from_heading(heading)
        return ux * strength, uy * strength

    def _near_cross_corridor(self, y: float) -> bool:
        return abs(y - 255.0) < 85.0 or abs(y - 455.0) < 85.0

    def _corridor_switching_vector(self, robot) -> Vector:
        x, y, th = robot.ekf.mu
        if getattr(robot, "role", "scout") == "scout":
            default_dir = 1.0
        else:
            default_dir = -1.0

        if x > WIDTH - 170:
            default_dir = -1.0
        elif x < 170:
            default_dir = 1.0

        if self._near_cross_corridor(y):
            self.corridor_switch_timer = 95
            self.corridor_switch_direction = default_dir

        if self.corridor_switch_timer > 0:
            self.corridor_switch_timer -= 1
            return (self.corridor_switch_direction * 3.0, 0.0)

        cross_targets = [255.0, 455.0]
        target_y = min(cross_targets, key=lambda yy: abs(yy - y))
        dy = target_y - y
        if abs(dy) < 35.0:
            return 0.0, 0.0
        return (0.0, clamp(dy / 90.0, -1.4, 1.4))

    def _visited_victim_repulsion_vector(self, robot, world) -> Vector:
        x, y = robot.ekf.mu[0], robot.ekf.mu[1]
        total = (0.0, 0.0)
        for victim in world.victims:
            if not victim["rescued"]:
                continue
            px, py = victim["pos"]
            d = distance((x, y), (px, py))
            if d > 120.0 or d <= 1e-6:
                continue
            heading = math.atan2(y - py, x - px)
            strength = clamp((120.0 - d) / 120.0, 0.0, 1.0)
            ux, uy = _unit_from_heading(heading)
            total = _add(total, (ux, uy), strength)
        return total

    def _front_distance(self, robot) -> float:
        sensor = robot.sensor_distances
        return min(sensor[0], sensor[1], sensor[-1])

# Backward compatibility for older imports.
GreenhouseController = SearchRescueController