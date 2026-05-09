import math
import random
from config import *


class RobotPhysics:

    # ---------------- STATES ----------------
    SEARCH = 0
    TO_VICTIM = 1
    RETURN_HOME = 2

    def __init__(self, world, x_start, y_start):

        self.world = world

        # pose
        self.x = x_start
        self.y = y_start
        self.theta = 0.0

        # control
        self.v_left = 0.0
        self.v_right = 0.0

        # home
        self.home_x = x_start
        self.home_y = y_start

        # SAR state
        self.state = RobotPhysics.SEARCH
        self.carrying = None

        # perception memory
        self.seen_victims = {}
        self.target_victim = None

        # EKF tracking
        self.actual_dx = 0.0
        self.actual_dy = 0.0
        self.actual_dtheta = 0.0

    # =================================================
    # CONTROL HELPERS
    # =================================================

    def wheel_speeds_to_v_omega(self, v_left, v_right):
        v = (v_right + v_left) / 2.0
        omega = (v_right - v_left) / WHEEL_BASE
        return v, omega

    # =================================================
    # MAIN UPDATE
    # =================================================

    def update(self, dt):

        old_x, old_y, old_theta = self.x, self.y, self.theta

        # perception + decision
        self.perceive_victims()
        self.update_state()
        self.apply_behavior()

        # physics motion
        v, omega = self.wheel_speeds_to_v_omega(self.v_left, self.v_right)

        self.theta += omega * dt
        self.theta %= (2 * math.pi)

        dx = v * math.cos(self.theta) * dt
        dy = v * math.sin(self.theta) * dt

        self.x, self.y = self._resolve_collision(self.x, self.y, dx, dy)

        # pickup/drop
        self.try_pickup()
        self.try_drop()

        # EKF delta
        self.actual_dx = self.x - old_x
        self.actual_dy = self.y - old_y
        self.actual_dtheta = self.theta - old_theta

    # =================================================
    # LANDMARK OBSERVATIONS (FOR EKF)
    # =================================================

    def get_landmark_observations(self, landmarks):

        observations = []

        for i, (lx, ly) in enumerate(landmarks):

            dx = lx - self.x
            dy = ly - self.y

            dist = math.hypot(dx, dy)

            if dist <= LANDMARKS_RANGE:

                bearing = math.atan2(dy, dx) - self.theta

                noisy_dist = dist + random.gauss(0, SENSOR_NOISE_STD)
                noisy_bearing = bearing + random.gauss(0, BEARING_NOISE_STD)

                observations.append((i, noisy_dist, noisy_bearing))

        return observations

    # =================================================
    # PERCEPTION (VICTIMS)
    # =================================================

    def perceive_victims(self):

        for i, v in enumerate(self.world.victims):

            if v.saved:
                continue

            dist = math.hypot(v.x - self.x, v.y - self.y)

            if dist < LANDMARKS_RANGE:
                self.seen_victims[i] = (v.x, v.y)

    # =================================================
    # STATE MACHINE
    # =================================================

    def update_state(self):

        if self.carrying:
            self.state = RobotPhysics.RETURN_HOME
            return

        if self.seen_victims and self.state == RobotPhysics.SEARCH:
            self.target_victim = next(iter(self.seen_victims))
            self.state = RobotPhysics.TO_VICTIM

    # =================================================
    # BEHAVIOR
    # =================================================

    def apply_behavior(self):

        if self.state == RobotPhysics.SEARCH:
            return

        if self.state == RobotPhysics.TO_VICTIM:

            vx, vy = self.seen_victims[self.target_victim]
            target = math.atan2(vy - self.y, vx - self.x)
            self.set_heading_control(target)

        elif self.state == RobotPhysics.RETURN_HOME:

            target = math.atan2(
                self.home_y - self.y,
                self.home_x - self.x
            )

            self.set_heading_control(target)

    # =================================================
    # CONTROL
    # =================================================

    def set_heading_control(self, target_angle):

        error = (target_angle - self.theta + math.pi) % (2 * math.pi) - math.pi

        K = 2.0

        self.v_left = LINEAR_SPEED - K * error
        self.v_right = LINEAR_SPEED + K * error

    # =================================================
    # PICKUP / DROP
    # =================================================

    def try_pickup(self):

        if self.carrying:
            return

        for i, v in enumerate(self.world.victims):

            if v.saved or v.carried_by:
                continue

            if math.hypot(v.x - self.x, v.y - self.y) < 25:

                v.carried_by = self
                self.carrying = v
                self.state = RobotPhysics.RETURN_HOME
                break

    def try_drop(self):

        if not self.carrying:
            return

        if math.hypot(self.x - self.home_x, self.y - self.home_y) < 30:

            self.carrying.saved = True
            self.carrying.carried_by = None
            self.carrying = None

            self.state = RobotPhysics.SEARCH
            self.seen_victims = {}
            self.target_victim = None

    # =================================================
    # COLLISION
    # =================================================

    def _resolve_collision(self, x, y, dx, dy, steps=10):

        sx = dx / steps
        sy = dy / steps

        nx, ny = x, y

        for _ in range(steps):

            tx = nx + sx
            ty = ny + sy

            if not self._collides(tx, ty):
                nx, ny = tx, ty
            else:
                break

        return nx, ny

    def _collides(self, x, y):

        for ax, ay, bx, by in self.world.wall_segments:
            if self._dist_point_seg(x, y, ax, ay, bx, by) < CIRCLE_RADIUS:
                return True

        return False

    def _dist_point_seg(self, px, py, ax, ay, bx, by):

        dx = bx - ax
        dy = by - ay

        if dx == 0 and dy == 0:
            return math.hypot(px - ax, py - ay)

        t = max(0, min(1,
                       ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
                       ))

        cx = ax + t * dx
        cy = ay + t * dy

        return math.hypot(px - cx, py - cy)