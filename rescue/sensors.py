import math

from config import *


class SensorSystem:

    def __init__(self, world):

        self.world = world

    def line_segment_intersection(
            self,
            ax,
            ay,
            bx,
            by,
            cx,
            cy,
            dx,
            dy
    ):

        denom = (
                (ax - bx) * (cy - dy)
                -
                (ay - by) * (cx - dx)
        )

        if abs(denom) < 1e-10:
            return None

        t = (
                (
                        (ax - cx) * (cy - dy)
                        -
                        (ay - cy) * (cx - dx)
                )
                / denom
        )

        u = -(
                (
                        (ax - bx) * (ay - cy)
                        -
                        (ay - by) * (ax - cx)
                )
                / denom
        )

        if (
                0.0 <= t <= 1.0
                and
                0.0 <= u <= 1.0
        ):

            ix = ax + t * (bx - ax)
            iy = ay + t * (by - ay)

            return ix, iy

        return None

    def get_sensor_readings(
            self,
            cx,
            cy,
            theta
    ):

        distances = []
        hit_points = []

        for i in range(NUM_SENSORS):

            angle = (
                    theta
                    + i * SENSOR_ANGLE_STEP
            )

            sx = (
                    cx
                    + math.cos(angle)
                    * CIRCLE_RADIUS
            )

            sy = (
                    cy
                    + math.sin(angle)
                    * CIRCLE_RADIUS
            )

            ex = (
                    cx
                    + math.cos(angle)
                    * MAX_SENSOR_DISTANCE
            )

            ey = (
                    cy
                    + math.sin(angle)
                    * MAX_SENSOR_DISTANCE
            )

            closest_dist = (
                    MAX_SENSOR_DISTANCE
                    - CIRCLE_RADIUS
            )

            closest_pt = (ex, ey)

            for (wx1, wy1, wx2, wy2) in self.world.wall_segments:

                pt = self.line_segment_intersection(
                    sx, sy, ex, ey,
                    wx1, wy1, wx2, wy2
                )

                if pt is not None:

                    d = math.hypot(
                        pt[0] - sx,
                        pt[1] - sy
                    )

                    if d < closest_dist:

                        closest_dist = d
                        closest_pt = pt

            distances.append(closest_dist)
            hit_points.append(closest_pt)

        return distances, hit_points