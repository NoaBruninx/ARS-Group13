"""
evolution.py
GENITOR-inspired steady-state genetic algorithm.

This is the evolutionary component for the damaged building scenario. It evolves
interpretable real-valued controller parameters rather than a black-box model.
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np

from controller import Genome
from occupancy_grid import OccupancyGrid
from robot import Robot
from world import BLUE, PURPLE, DT, SearchRescueWorld, distance


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


@dataclass
class EvalResult:
    fitness: float
    explored_fraction: float
    detected_victims: int
    rescued_victims: int
    collisions: int
    repeated_visits: int
    mean_localization_error: float
    robot_overlap_ratio: float


@dataclass
class ScoredGenome:
    genome: Genome
    result: EvalResult


# ---------------------------------------------------------------------
# Team creation and evaluation
# ---------------------------------------------------------------------


def make_robots(genome: Genome, seed: int, one_robot: bool = False) -> List[Robot]:
    rng = random.Random(seed)
    if one_robot:
        return [Robot(1, "rescue", (995, 600, -math.pi / 2), genome, PURPLE, rng.randint(0, 10_000_000))]

    return [
        Robot(1, "scout",  (90,  120, math.pi / 2),  genome, BLUE,   rng.randint(0, 10_000_000)),
        Robot(2, "rescue", (995, 600, -math.pi / 2), genome, PURPLE, rng.randint(0, 10_000_000)),
        ]


def merge_metrics_from_maps(maps: List[OccupancyGrid]) -> float:
    if len(maps) == 1:
        return maps[0].explored_fraction()
    evidence_union = np.zeros_like(maps[0].evidence, dtype=bool)
    for m in maps:
        evidence_union |= m.evidence > 0
    return float(np.mean(evidence_union))


def evaluate_genome(
    genome: Genome,
    steps: int = 1800,
    seed: int = 0,
    shared_map: bool = True,
    one_robot: bool = False,
    dynamic_block: bool = False,
    cell_size: int = 20,
) -> EvalResult:
    """Run one simulation episode and compute fitness."""
    world = SearchRescueWorld(dynamic_block=dynamic_block, seed=seed)
    robots = make_robots(genome, seed=seed, one_robot=one_robot)

    if shared_map:
        shared_grid = OccupancyGrid(cell_size=cell_size)
        maps = [shared_grid for _ in robots]
    else:
        maps = [OccupancyGrid(cell_size=cell_size) for _ in robots]

    localization_errors = []
    close_together_count = 0

    for step in range(steps):
        if dynamic_block and step == steps // 2:
            world.activate_dynamic_blockage()

        for i, robot in enumerate(robots):
            robot.update(world, maps[i], robots, DT)
            localization_errors.append(robot.localization_error())

        for m in {id(m): m for m in maps}.values():
            m.decay_pheromones()
            m.spread_pheromones()

        if len(robots) > 1:
            if distance((robots[0].true_pose[0], robots[0].true_pose[1]),
                        (robots[1].true_pose[0], robots[1].true_pose[1])) < 70.0:
                close_together_count += 1

    explored_fraction = merge_metrics_from_maps(maps)
    detected = world.num_detected()
    rescued = world.num_rescued()
    collisions = sum(r.collisions for r in robots)
    repeated_visits = sum(r.repeated_visits for r in robots)
    mean_loc_error = float(np.mean(localization_errors)) if localization_errors else 0.0
    overlap_ratio = close_together_count / max(1, steps)

    # Group-level fitness. The weights are intentionally simple and reportable.
    fitness = (
        350.0 * rescued
        + 40.0 * detected
        + 850.0 * explored_fraction
        - 6.0 * collisions
        - 3.0 * repeated_visits
        - 1.2 * mean_loc_error
        - 120.0 * overlap_ratio
    )

    return EvalResult(
        fitness=float(fitness),
        explored_fraction=float(explored_fraction),
        detected_victims=int(detected),
        rescued_victims=int(rescued),
        collisions=int(collisions),
        repeated_visits=int(repeated_visits),
        mean_localization_error=float(mean_loc_error),
        robot_overlap_ratio=float(overlap_ratio),
    )


# ---------------------------------------------------------------------
# GENITOR-style evolutionary algorithm
# ---------------------------------------------------------------------


def genome_vector(g: Genome) -> np.ndarray:
    d = g.to_dict()
    return np.array([d[k] for k in sorted(d.keys())], dtype=float)


def population_diversity(population: List[ScoredGenome]) -> float:
    if len(population) < 2:
        return 0.0
    vectors = [genome_vector(sg.genome) for sg in population]
    dists = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            dists.append(float(np.linalg.norm(vectors[i] - vectors[j])))
    return float(np.mean(dists)) if dists else 0.0


def tournament_select(population: List[ScoredGenome], rng: random.Random, k: int = 4) -> Genome:
    candidates = rng.sample(population, k=min(k, len(population)))
    candidates.sort(key=lambda item: item.result.fitness, reverse=True)
    return candidates[0].genome


def evolve_genitor(
    population_size: int = 18,
    evaluations: int = 80,
    seed: int = 42,
    steps_per_eval: int = 1600,
    episodes_per_genome: int = 2,
    output_csv: str = "results/evolution_log.csv",
) -> Genome:
    """GENITOR-inspired steady-state GA.

    Each iteration creates one child. If the child is better than the worst
    individual, it replaces the worst individual. This is close to the GENITOR
    idea discussed in class: parents and children coexist, with rank/fitness
    based replacement.
    """
    rng = random.Random(seed)

    def evaluate_average(g: Genome, base_seed: int) -> EvalResult:
        results = []
        for ep in range(episodes_per_genome):
            results.append(evaluate_genome(g, steps=steps_per_eval, seed=base_seed + ep * 1000))
        return EvalResult(
            fitness=float(np.mean([r.fitness for r in results])),
            explored_fraction=float(np.mean([r.explored_fraction for r in results])),
            detected_victims=int(round(np.mean([r.detected_victims for r in results]))),
            rescued_victims=int(round(np.mean([r.rescued_victims for r in results]))),
            collisions=int(round(np.mean([r.collisions for r in results]))),
            repeated_visits=int(round(np.mean([r.repeated_visits for r in results]))),
            mean_localization_error=float(np.mean([r.mean_localization_error for r in results])),
            robot_overlap_ratio=float(np.mean([r.robot_overlap_ratio for r in results])),
        )

    population: List[ScoredGenome] = []
    for i in range(population_size):
        g = Genome.random(rng)
        res = evaluate_average(g, seed * 10_000 + i * 100)
        population.append(ScoredGenome(g, res))

    log_rows = []

    for evaluation in range(evaluations):
        population.sort(key=lambda item: item.result.fitness, reverse=True)
        best = population[0]
        avg_fitness = float(np.mean([sg.result.fitness for sg in population]))
        diversity = population_diversity(population)

        log_rows.append({
            "evaluation": evaluation,
            "best_fitness": best.result.fitness,
            "avg_fitness": avg_fitness,
            "diversity": diversity,
            "best_explored_fraction": best.result.explored_fraction,
            "best_detected_victims": best.result.detected_victims,
            "best_rescued_victims": best.result.rescued_victims,
            "best_collisions": best.result.collisions,
            "best_mean_localization_error": best.result.mean_localization_error,
            "best_overlap_ratio": best.result.robot_overlap_ratio,
        })

        print(
            f"Eval {evaluation:03d} | best={best.result.fitness:.1f} | "
            f"avg={avg_fitness:.1f} | rescued={best.result.rescued_victims} | "
            f"explored={best.result.explored_fraction:.3f} | div={diversity:.2f}"
        )

        # Create one child.
        parent_a = tournament_select(population, rng)
        parent_b = tournament_select(population, rng)
        child = Genome.crossover(parent_a, parent_b, rng).mutate(rng, sigma=0.10)
        child_result = evaluate_average(child, seed * 50_000 + evaluation * 100)

        # Replace worst if child is better.
        population.sort(key=lambda item: item.result.fitness, reverse=True)
        if child_result.fitness > population[-1].result.fitness:
            population[-1] = ScoredGenome(child, child_result)

    population.sort(key=lambda item: item.result.fitness, reverse=True)
    best = population[0]

    Path(output_csv).parent.mkdir(exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    with open("results/best_genome.json", "w", encoding="utf-8") as f:
        json.dump(best.genome.to_dict(), f, indent=2)

    print("\nBest genome saved to results/best_genome.json")
    print(json.dumps(best.genome.to_dict(), indent=2))
    return best.genome


if __name__ == "__main__":
    evolve_genitor(population_size=14, evaluations=40, steps_per_eval=900, episodes_per_genome=1)
