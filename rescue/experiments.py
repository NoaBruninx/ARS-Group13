"""
experiments.py
Batch experiments and plotting for the report.

Quick run:
    python experiments.py --quick

Fuller run:
    python experiments.py --full
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from controller import Genome
from evolution import evaluate_genome, evolve_genitor


RESULTS = Path("results")
FIGURES = Path("figures")
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def load_best_genome() -> Genome:
    path = RESULTS / "best_genome.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))

        # Compatibility with older greenhouse genomes after the scenario rename.
        rename_map = {
            "plant_attraction_weight": "victim_attraction_weight",
            "visited_plant_penalty": "visited_victim_penalty",
            "aisle_switching_weight": "corridor_switching_weight",
        }
        for old_key, new_key in rename_map.items():
            if old_key in data and new_key not in data:
                data[new_key] = data[old_key]

        defaults = Genome().to_dict()
        allowed = set(defaults.keys())
        defaults.update({k: v for k, v in data.items() if k in allowed})
        return Genome(**defaults)
    return Genome()


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def mean_metric(rows: List[dict], metric: str) -> float:
    return float(np.mean([r[metric] for r in rows])) if rows else 0.0


def result_to_row(condition: str, seed: int, result) -> dict:
    return {
        "condition": condition,
        "seed": seed,
        "fitness": result.fitness,
        "explored_fraction": result.explored_fraction,
        "detected_victims": result.detected_victims,
        "rescued_victims": result.rescued_victims,
        "collisions": result.collisions,
        "repeated_visits": result.repeated_visits,
        "mean_localization_error": result.mean_localization_error,
        "robot_overlap_ratio": result.robot_overlap_ratio,
    }


def plot_bar_summary(rows: List[dict], metric: str, title: str, output: Path) -> None:
    conditions = sorted(set(r["condition"] for r in rows))
    means = []
    stds = []
    for c in conditions:
        values = [r[metric] for r in rows if r["condition"] == c]
        means.append(float(np.mean(values)))
        stds.append(float(np.std(values)))

    plt.figure(figsize=(7, 4))
    plt.bar(conditions, means, yerr=stds, capsize=4)
    plt.title(title)
    plt.ylabel(metric.replace("_", " "))
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def plot_evolution_log(csv_path: Path = RESULTS / "evolution_log.csv") -> None:
    if not csv_path.exists():
        print("No evolution_log.csv found. Run evolution first.")
        return

    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    if data.size == 0:
        return

    plt.figure(figsize=(7, 4))
    plt.plot(data["evaluation"], data["best_fitness"], label="best")
    plt.plot(data["evaluation"], data["avg_fitness"], label="average")
    plt.title("Fitness over GENITOR evaluations")
    plt.xlabel("Evaluation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "fitness_over_evaluations.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(data["evaluation"], data["diversity"])
    plt.title("Population diversity over evaluations")
    plt.xlabel("Evaluation")
    plt.ylabel("Average pairwise genome distance")
    plt.tight_layout()
    plt.savefig(FIGURES / "diversity_over_evaluations.png", dpi=160)
    plt.close()


# ---------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------


def experiment_baseline_vs_evolved(repetitions: int, steps: int) -> List[dict]:
    baseline = Genome()
    evolved = load_best_genome()
    rows = []
    for seed in range(repetitions):
        rows.append(result_to_row("baseline", seed, evaluate_genome(baseline, steps=steps, seed=100 + seed)))
        rows.append(result_to_row("evolved", seed, evaluate_genome(evolved, steps=steps, seed=200 + seed)))
    write_csv(RESULTS / "baseline_vs_evolved.csv", rows)
    plot_bar_summary(rows, "rescued_victims", "Baseline vs evolved: rescued victims", FIGURES / "baseline_vs_evolved_rescued_victims.png")
    plot_bar_summary(rows, "fitness", "Baseline vs evolved: fitness", FIGURES / "baseline_vs_evolved_fitness.png")
    return rows


def experiment_one_vs_two(repetitions: int, steps: int) -> List[dict]:
    g = load_best_genome()
    rows = []
    for seed in range(repetitions):
        rows.append(result_to_row("one_rescue_robot", seed, evaluate_genome(g, steps=steps, seed=300 + seed, one_robot=True)))
        rows.append(result_to_row("scout_plus_rescue", seed, evaluate_genome(g, steps=steps, seed=400 + seed, one_robot=False)))
    write_csv(RESULTS / "one_vs_two.csv", rows)
    plot_bar_summary(rows, "rescued_victims", "One robot vs heterogeneous team", FIGURES / "one_vs_two_rescued_victims.png")
    plot_bar_summary(rows, "explored_fraction", "One robot vs team: explored area", FIGURES / "one_vs_two_explored.png")
    return rows


def experiment_shared_map(repetitions: int, steps: int) -> List[dict]:
    g = load_best_genome()
    rows = []
    for seed in range(repetitions):
        rows.append(result_to_row("independent_maps", seed, evaluate_genome(g, steps=steps, seed=500 + seed, shared_map=False)))
        rows.append(result_to_row("shared_map", seed, evaluate_genome(g, steps=steps, seed=600 + seed, shared_map=True)))
    write_csv(RESULTS / "shared_map_vs_no_shared.csv", rows)
    plot_bar_summary(rows, "explored_fraction", "Shared map vs independent maps", FIGURES / "shared_map_explored.png")
    plot_bar_summary(rows, "robot_overlap_ratio", "Shared map: robot overlap ratio", FIGURES / "shared_map_overlap.png")
    return rows


def experiment_collapsed_passage(repetitions: int, steps: int) -> List[dict]:
    g = load_best_genome()
    rows = []
    for seed in range(repetitions):
        rows.append(result_to_row("normal_building", seed, evaluate_genome(g, steps=steps, seed=700 + seed, dynamic_block=False)))
        rows.append(result_to_row("collapsed_passage", seed, evaluate_genome(g, steps=steps, seed=800 + seed, dynamic_block=True)))
    write_csv(RESULTS / "collapsed_passage.csv", rows)
    plot_bar_summary(rows, "rescued_victims", "Normal vs collapsed passage: rescued victims", FIGURES / "collapsed_passage_rescued.png")
    plot_bar_summary(rows, "collisions", "Normal vs collapsed passage: collisions", FIGURES / "collapsed_passage_collisions.png")
    return rows


def experiment_map_resolution(repetitions: int, steps: int) -> List[dict]:
    g = load_best_genome()
    rows = []
    for seed in range(repetitions):
        for cell_size in [12, 20, 32, 40]:
            result = evaluate_genome(g, steps=steps, seed=900 + seed * 10 + cell_size, cell_size=cell_size)
            rows.append(result_to_row(f"cell_{cell_size}", seed, result))
    write_csv(RESULTS / "map_resolution.csv", rows)
    plot_bar_summary(rows, "explored_fraction", "Map resolution: explored fraction", FIGURES / "map_resolution_explored.png")
    plot_bar_summary(rows, "collisions", "Map resolution: collisions", FIGURES / "map_resolution_collisions.png")
    return rows


def run_all(quick: bool = True) -> None:
    if quick:
        repetitions = 3
        steps = 800
        print("Running QUICK experiments. Use --full for stronger report data.")
    else:
        repetitions = 8
        steps = 1600
        print("Running FULL experiments. This may take a while.")

    # Create/refresh an evolved genome if none exists.
    if not (RESULTS / "best_genome.json").exists():
        print("No best genome found. Running a short evolution first.")
        evolve_genitor(
            population_size=12 if quick else 18,
            evaluations=25 if quick else 80,
            steps_per_eval=600 if quick else 1200,
            episodes_per_genome=1 if quick else 2,
        )

    plot_evolution_log()
    experiment_baseline_vs_evolved(repetitions, steps)
    experiment_one_vs_two(repetitions, steps)
    experiment_shared_map(repetitions, steps)
    experiment_collapsed_passage(repetitions, steps)
    experiment_map_resolution(repetitions, steps)

    print("Done. Check results/ and figures/.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run short experiments")
    parser.add_argument("--full", action="store_true", help="Run longer experiments")
    args = parser.parse_args()
    run_all(quick=not args.full)
