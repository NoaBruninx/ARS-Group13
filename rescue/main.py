"""
main.py

Run:
    python main.py

Keys:
    R      reset
    SPACE  pause
    M      toggle occupancy map
    T      toggle trajectories
    S      toggle sensors
    B      activate collapsed passage
    G      debug: show/hide full ground-truth damaged building map
    E      run short GENITOR evolution and load best genome
    ESC    quit
"""

from __future__ import annotations

import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pygame

from controller import Genome
from evolution import evolve_genitor
from occupancy_grid import OccupancyGrid
from robot import Robot, draw_text
from world import (
    WIDTH, HEIGHT, FPS, DT,
    WHITE, BLACK, BLUE, PURPLE, GREEN, ORANGE, RED, UNKNOWN_DARK, YELLOW, LIGHT_GRAY,
    SearchRescueWorld,
)

# Log file setup
LOG_PATH = Path("results/simulation_log.csv")
LOG_PATH.parent.mkdir(exist_ok=True)


class SimLogger:
    """Simple CSV logger that flushes after every write."""
    def __init__(self, path: Path):
        self._f = open(path, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self._w.writerow(["step", "time_s", "rescued", "detected",
                          "explored_pct", "slam_error_px",
                          "collisions", "repeated_visits", "blockage_active"])
        self._f.flush()

    def write(self, step, t, world, grid, robots):
        mean_error = float(np.mean([r.localization_error() for r in robots]))
        self._w.writerow([
            step, round(t, 2),
            world.num_rescued(), world.num_detected(),
            round(100 * grid.explored_fraction(), 2),
            round(mean_error, 2),
            sum(r.collisions for r in robots),
            sum(r.repeated_visits for r in robots),
            int(world.blockage_active),
        ])
        self._f.flush()

    def close(self):
        self._f.close()


def load_best_or_default() -> Genome:
    path = Path("results/best_genome.json")
    if path.exists():
        try:
            import json
            data = json.loads(path.read_text(encoding="utf-8"))
            defaults = Genome().to_dict()
            defaults.update({k: v for k, v in data.items() if k in defaults})
            g = Genome(**defaults)
            print(f"Loaded evolved genome from {path}")
            return g
        except Exception as e:
            print(f"Could not load genome: {e} — using default")
    return Genome()


def create_simulation(genome: Genome, seed: int = 7):
    world = SearchRescueWorld(dynamic_block=True)
    grid = OccupancyGrid(cell_size=20)
    rng = random.Random(seed)
    robots = [
        Robot(1, "scout", (90, 120, math.pi / 2), genome, BLUE, rng.randint(0, 10_000_000)),
        Robot(2, "rescue", (995, 600, -math.pi / 2), genome, PURPLE, rng.randint(0, 10_000_000)),
    ]
    return world, grid, robots


def draw_hud(screen, world, grid, robots, paused: bool) -> None:
    """Minimal HUD: only victim counts. All details go to the log file."""
    total   = len(world.victims)
    rescued = world.num_rescued()
    detected = world.num_detected()

    panel = pygame.Surface((260, 70), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 160))
    screen.blit(panel, (10, 10))

    draw_text(screen, f"Victims rescued:  {rescued}/{total}", 20, 18, GREEN,  16)
    draw_text(screen, f"Victims detected: {detected}/{total}", 20, 42, ORANGE, 16)
    if paused:
        draw_text(screen, "PAUSED", WIDTH // 2 - 35, 10, WHITE, 18)


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("ARS Final - Search and Rescue Robots")
    clock = pygame.time.Clock()

    genome = load_best_or_default()
    world, grid, robots = create_simulation(genome)

    logger = SimLogger(LOG_PATH)
    print(f"Logging to {LOG_PATH.resolve()}")

    paused = False
    show_map = True
    show_trajectories = True
    show_sensors = True
    show_ground_truth = False
    mission_complete = False
    running = True
    step = 0
    sim_time = 0.0
    LOG_INTERVAL = 60  # write to log every 60 steps (~1 second)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_m:
                    show_map = not show_map
                elif event.key == pygame.K_t:
                    show_trajectories = not show_trajectories
                elif event.key == pygame.K_s:
                    show_sensors = not show_sensors
                elif event.key == pygame.K_g:
                    show_ground_truth = not show_ground_truth
                elif event.key == pygame.K_b:
                    world.activate_dynamic_blockage()
                elif event.key == pygame.K_r:
                    world, grid, robots = create_simulation(genome, seed=random.randint(0, 99999))
                    mission_complete = False
                    step = 0
                    sim_time = 0.0
                elif event.key == pygame.K_e:
                    print("Running short evolution. The Pygame window may freeze for a while...")
                    genome = evolve_genitor(population_size=12, evaluations=30, steps_per_eval=1000, episodes_per_genome=1)
                    world, grid, robots = create_simulation(genome, seed=random.randint(0, 99999))
                    mission_complete = False
                    step = 0
                    sim_time = 0.0

        if not paused:
            for robot in robots:
                robot.update(world, grid, robots, DT)
            step += 1
            sim_time += DT

            # Write to log every LOG_INTERVAL steps
            if step % LOG_INTERVAL == 0 and step > 0:
                logger.write(step, sim_time, world, grid, robots)

            # Auto-pause when all victims are rescued.
            if not mission_complete and world.num_rescued() == len(world.victims) and len(world.victims) > 0:
                paused = True
                mission_complete = True
                logger.write(step, sim_time, world, grid, robots)

        screen.fill(UNKNOWN_DARK)

        world.draw(screen, grid=grid, show_ground_truth=show_ground_truth)
        if show_map:
            grid.draw(screen)
        for robot in robots:
            robot.draw(screen, world=world, show_trajectories=show_trajectories, show_sensors=show_sensors)
        draw_hud(screen, world, grid, robots, paused)

        # Victory overlay.
        if mission_complete:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 140))
            screen.blit(overlay, (0, 0))

            box_w, box_h = 480, 200
            bx = (WIDTH - box_w) // 2
            by = (HEIGHT - box_h) // 2
            box = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
            box.fill((30, 120, 60, 230))
            screen.blit(box, (bx, by))
            pygame.draw.rect(screen, GREEN, (bx, by, box_w, box_h), 3)

            draw_text(screen, "MISSION COMPLETE!", bx + 80, by + 30, WHITE, 34)
            draw_text(screen, f"All {world.num_rescued()} victims rescued!", bx + 100, by + 85, YELLOW, 22)
            draw_text(screen, f"Explored: {100 * grid.explored_fraction():.1f}%   Collisions: {sum(r.collisions for r in robots)}", bx + 60, by + 120, LIGHT_GRAY, 18)
            draw_text(screen, "Press R to restart", bx + 155, by + 158, LIGHT_GRAY, 16)

        pygame.display.flip()
        clock.tick(FPS)

    logger.close()
    pygame.quit()


if __name__ == "__main__":
    main()