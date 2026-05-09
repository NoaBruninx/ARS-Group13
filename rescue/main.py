import random

import pygame
import sys
import math
import numpy as np

from world import World, world_to_grid
from robot_physics import RobotPhysics
from sensors import SensorSystem
from ekf_localization import EKFLocalization
from occupancy_grid import OccupancyGridMap
from renderer import Renderer
from config import *
from pheromone_map import PheromoneMap

pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("ARS Assignment - Part 3 - Group 13")

font_small = pygame.font.SysFont(None, 16)
font_med = pygame.font.SysFont(None, 22)


def main():

    clock = pygame.time.Clock()

    world = World()
    wall_grid = build_wall_grid(world)

    # ---------------- ROBOTS ----------------
    robots = [
        RobotPhysics(world, 40, 50),
        RobotPhysics(world, 80, 100),
        RobotPhysics(world, 40, 70),
    ]

    # ---------------- EKF PER ROBOT ----------------
    ekfs = [
        EKFLocalization(r.x, r.y, r.theta)
        for r in robots
    ]

    trajectories = [[] for _ in robots]

    # ---------------- SYSTEMS ----------------
    sensors = SensorSystem(world)
    occupancy = OccupancyGridMap()

    explore_pheromone = PheromoneMap()
    home_pheromone = PheromoneMap()

    renderer = Renderer(font_small, font_med)

    use_aco = False
    use_ekf_pose_for_mapping = True
    show_map = True

    running = True

    while running:

        dt = clock.tick(240) / 1000.0
        dt = min(dt, 0.05)

        # ---------------- EVENTS ----------------
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:
                    running = False

                if event.key == pygame.K_t:
                    for r in robots:
                        r.teleport_valid_location()
                    trajectories = [[] for _ in robots]

                if event.key == pygame.K_c:
                    occupancy.clear()

                if event.key == pygame.K_m:
                    use_aco = not use_aco

                if event.key == pygame.K_g:
                    show_map = not show_map

        keys = pygame.key.get_pressed()

        # ---------------- CONTROL ----------------
        for robot in robots:



            if use_aco:

                        # ---------------- STUCK RECOVERY ----------------
                if getattr(robot, "is_stuck", False):

                    # rotate in place randomly
                    turn = random.choice([-1, 1])

                    robot.v_left = -LINEAR_SPEED * turn
                    robot.v_right = LINEAR_SPEED * turn

                    continue
                r, c = world_to_grid(robot.x, robot.y)

                if robot.carrying is not None:
                    dr, dc = home_pheromone.best_direction(r, c, wall_grid)
                else:
                    dr, dc = explore_pheromone.best_direction(r, c, wall_grid)



                # ---------------- CONVERT GRID DIR → VECTOR ----------------
                dir_x = dc
                dir_y = dr

                norm = math.hypot(dir_x, dir_y)

                if norm < 1e-6:
                    continue

                dir_x /= norm
                dir_y /= norm

                target_angle = math.atan2(dir_y, dir_x)
                robot.set_heading_control(target_angle)

            else:
                robot.v_left = 0.0
                robot.v_right = 0.0
                if keys[pygame.K_UP]:
                    robot.v_left += LINEAR_SPEED
                    robot.v_right += LINEAR_SPEED

                if keys[pygame.K_DOWN]:
                    robot.v_left -= LINEAR_SPEED
                    robot.v_right -= LINEAR_SPEED

                if keys[pygame.K_LEFT]:
                    robot.v_left += LINEAR_SPEED * 0.5
                    robot.v_right -= LINEAR_SPEED * 0.5

                if keys[pygame.K_RIGHT]:
                    robot.v_left -= LINEAR_SPEED * 0.5
                    robot.v_right += LINEAR_SPEED * 0.5

        # ---------------- SENSOR DATA ----------------
        sensor_data = []

        for robot in robots:
            distances, hits = sensors.get_sensor_readings(
                robot.x,
                robot.y,
                robot.theta
            )
            sensor_data.append((distances, hits))

        # ---------------- UPDATE ROBOTS + EKF ----------------
        for i, robot in enumerate(robots):

            robot.update(dt)
            movement = math.hypot(robot.actual_dx, robot.actual_dy)
            #some stuff so robot doesnt go into wals
            robot.is_stuck = movement < 0.5

            robot.perceive_victims()

            r, c = world_to_grid(robot.x, robot.y)

            # -------- EXPLORATION PHEROMONE --------
            explore_pheromone.add(r, c, 1.0)

            # -------- HOME PHEROMONE --------
            if robot.carrying is not None:
                hr, hc = world_to_grid(robot.home_x, robot.home_y)
                home_pheromone.add(hr, hc, 2.0)

            # ---------------- EKF ----------------
            ekf = ekfs[i]

            v, omega = robot.wheel_speeds_to_v_omega(
                robot.v_left,
                robot.v_right
            )

            ekf.predict_odometry(
                robot.actual_dx,
                robot.actual_dy,
                robot.actual_dtheta
            )



            if robot.actual_dx == 0 and abs(v) > 0:
                ekf.sigma *= 1.01

            obs = robot.get_landmark_observations(world.landmarks)
            if obs:
                ekf.update(obs, world.landmarks)

            trajectories[i].append((robot.x, robot.y))

        # ---------------- DECAY ----------------
        explore_pheromone.decay(dt)
        home_pheromone.decay(dt)

        # ---------------- MAP UPDATE ----------------
        for i, robot in enumerate(robots):

            distances, _ = sensor_data[i]
            ekf = ekfs[i]

            if use_ekf_pose_for_mapping:
                pose = ekf.mu
                map_mode = "Swarm EKF"
            else:
                pose = np.array([robot.x, robot.y, robot.theta])
                map_mode = "True pose"

            occupancy.update(pose, distances)

        # ---------------- RENDER ----------------
        screen.fill(WHITE)

        if show_map:
            renderer.draw_occupancy_grid(screen, occupancy)

        renderer.draw_pheromones(screen, explore_pheromone)
        renderer.draw_pheromones(screen, home_pheromone)

        renderer.draw_walls(screen, world)
        renderer.draw_victims(screen, world)
        renderer.draw_robots(screen, robots)

        for traj in trajectories:
            if len(traj) > 1:
                pygame.draw.lines(screen, GRAY, False, traj, 2)

        for ekf in ekfs:
            renderer.draw_kf_estimate(screen, ekf.mu, ekf.sigma)

        for i, robot in enumerate(robots):

            distances, hits = sensor_data[i]

            renderer.draw_sensors(
                screen,
                robot.x,
                robot.y,
                robot.theta,
                distances,
                hits
            )

        avg_v_left = sum(r.v_left for r in robots) / len(robots)
        avg_v_right = sum(r.v_right for r in robots) / len(robots)

        renderer.draw_hud(
            screen,
            avg_v_left,
            avg_v_right,
            math.degrees(robots[0].theta),
            len(world.landmarks),
            map_mode,
            show_map
        )

        pygame.display.flip()

    pygame.quit()
    sys.exit()

def build_wall_grid(world):
    grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.int8)

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):

            # convert grid cell → world position
            x = c * MAP_CELL_SIZE
            y = r * MAP_CELL_SIZE

            for ax, ay, bx, by in world.wall_segments:

                # simple distance-to-segment check
                dx, dy = bx - ax, by - ay
                if dx == 0 and dy == 0:
                    continue

                t = max(0, min(1,
                               ((x - ax) * dx + (y - ay) * dy) / (dx*dx + dy*dy)
                               ))

                cx = ax + t * dx
                cy = ay + t * dy

                if math.hypot(x - cx, y - cy) < CIRCLE_RADIUS:
                    grid[r, c] = 1
                    break

    return grid
if __name__ == "__main__":
    main()