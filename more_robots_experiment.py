from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import numpy as np

from controller import Genome
from occupancy_grid import OccupancyGrid
from robot import Robot
from world import DT, BLUE, PURPLE, SearchRescueWorld


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)



@dataclass
class ExperimentResult:
    scouts: int
    rescue: int
    total_robots: int
    seed: int

    rescued_victims: int
    detected_victims: int
    explored_fraction: float

    collisions: int
    repeated_visits: int
    mean_localization_error: float

    completion_time: float
    mission_completed: bool


@dataclass
class StepSnapshot:
    scouts: int
    rescue: int
    seed: int
    step: int

    rescued_victims: int
    detected_victims: int
    explored_fraction: float
    collisions: int


def append_csv_row(path: str, row: dict):
    file_exists = Path(path).exists()

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def append_snapshots(path: str, snapshots: List[dict]):
    if not snapshots:
        return

    file_exists = Path(path).exists()

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=snapshots[0].keys())

        if not file_exists:
            writer.writeheader()

        writer.writerows(snapshots)


# SAFE RANDOM SPAWN

def random_free_pose(world: SearchRescueWorld, rng: random.Random, max_tries: int = 200):
    for _ in range(max_tries):
        x = rng.uniform(60, 1040)
        y = rng.uniform(60, 640)
        th = rng.uniform(-math.pi, math.pi)

        if not world.collides(x, y):
            return (x, y, th)

    return (100, 100, 0.0)


# TEAM CREATION


def create_team(world, genome, num_scouts, num_rescue, seed):
    rng = random.Random(seed)

    robots = []
    rid = 1

    for _ in range(num_scouts):
        pose = random_free_pose(world, rng)
        robots.append(
            Robot(rid, "scout", pose, genome, BLUE, rng.randint(0, 10_000_000))
        )
        rid += 1

    for _ in range(num_rescue):
        pose = random_free_pose(world, rng)
        robots.append(
            Robot(rid, "rescue", pose, genome, PURPLE, rng.randint(0, 10_000_000))
        )
        rid += 1

    return robots


# SINGLE EXPERIMENT

def run_experiment(
        genome: Genome,
        num_scouts: int,
        num_rescue: int,
        seed: int,
        max_steps: int = 5000,
):

    # FIXED MAP (same every run)
    world_seed = 452132

    world = SearchRescueWorld(
        dynamic_block=True,
        seed=world_seed,
    )

    grid = OccupancyGrid(cell_size=20)

    # DIFFERENT ROBOT RANDOMNESS
    robots = create_team(world, genome, num_scouts, num_rescue, seed)

    localization_errors = []

    completion_time = max_steps * DT
    mission_completed = False

    snapshots: List[StepSnapshot] = []

    for step in range(max_steps):

        if step == max_steps // 2:
            world.activate_dynamic_blockage()

        for robot in robots:
            robot.update(world, grid, robots, DT)
            localization_errors.append(robot.localization_error())

        grid.decay_pheromones()
        grid.spread_pheromones()

        # ======================================================
        # LIVE MILESTONE PRINTING (NEW)
        # ======================================================
        if step % 500 == 0 and step > 0:
            print(
                f"[t={step}] "
                f"rescued={world.num_rescued()} | "
                f"detected={world.num_detected()} | "
                f"explored={grid.explored_fraction():.3f} | "
                f"collisions={sum(r.collisions for r in robots)}"
            )

            snapshots.append(
                StepSnapshot(
                    scouts=num_scouts,
                    rescue=num_rescue,
                    seed=seed,
                    step=step,

                    rescued_victims=world.num_rescued(),
                    detected_victims=world.num_detected(),
                    explored_fraction=grid.explored_fraction(),
                    collisions=sum(r.collisions for r in robots),
                )
            )

        if world.num_rescued() == len(world.victims):
            completion_time = step * DT
            mission_completed = True
            break

    return ExperimentResult(
        scouts=num_scouts,
        rescue=num_rescue,
        total_robots=num_scouts + num_rescue,
        seed=seed,

        rescued_victims=world.num_rescued(),
        detected_victims=world.num_detected(),

        explored_fraction=grid.explored_fraction(),

        collisions=sum(r.collisions for r in robots),
        repeated_visits=sum(r.repeated_visits for r in robots),

        mean_localization_error=float(np.mean(localization_errors))
        if localization_errors else 0.0,

        completion_time=completion_time,
        mission_completed=mission_completed,
    ), snapshots


# EXPERIMENT DRIVER

class ScalabilityExperiment:

    def __init__(self, runs_per_config: int = 10, max_steps: int = 5000):
        self.runs_per_config = runs_per_config
        self.max_steps = max_steps

        self.genome = self.load_best_or_default()

    def load_best_or_default(self) -> Genome:
        path = Path("best_genome.json")

        if path.exists():
            import json
            data = json.loads(path.read_text(encoding="utf-8"))

            defaults = Genome().to_dict()
            defaults.update({k: v for k, v in data.items() if k in defaults})

            return Genome(**defaults)

        return Genome()


    def run(self):

        configs = []

        for rescue in range(1, 5):
            for scouts in range(0, 5):
                configs.append((scouts, rescue))

        for scouts, rescue in configs:

            print(f"\nCONFIG: scouts={scouts}, rescue={rescue}")

            for seed in range(self.runs_per_config):

                result, snaps = run_experiment(
                    self.genome,
                    scouts,
                    rescue,
                    seed,
                    self.max_steps,
                )

                # =========================
                # SAVE FINAL RESULT NOW
                # =========================
                append_csv_row(
                    "results/scalability_final.csv",
                    asdict(result)
                )

                # =========================
                # SAVE SNAPSHOTS NOW
                # =========================
                append_snapshots(
                    "results/scalability_timeseries.csv",
                    [asdict(s) for s in snaps]
                )

                print(
                    f"run={seed} | "
                    f"rescued={result.rescued_victims} | "
                    f"explored={result.explored_fraction:.3f} | "
                    f"collisions={result.collisions}"
                )




if __name__ == "__main__":

    exp = ScalabilityExperiment(
        runs_per_config=5,
        max_steps=5000,
    )

    exp.run()