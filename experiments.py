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
        "pheromone_changed_frontier_rate": result.pheromone_changed_frontier_rate,
        "frontier_pheromone_coverage": result.frontier_pheromone_coverage,
    }


def plot_experiment_dashboard(
    rows: List[dict],
    conditions: List[str],
    condition_labels: List[str],
    panels: List[tuple],
    title: str,
    output: Path,
    single_row: bool = False,
) -> None:
    """One summary figure per experiment: 2×2 boxplot grid, one metric per panel.
    Use single_row=True for a 1×N horizontal layout."""
    n = len(panels)
    if single_row:
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        axes = list(axes)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        axes = axes.flatten()

    for ax, (metric, panel_title, ylabel) in zip(axes, panels):
        data = [
            [float(r[metric]) for r in rows if r["condition"] == condition]
            for condition in conditions
        ]
        ax.boxplot(data, tick_labels=condition_labels, showmeans=True)
        ax.set_title(panel_title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output, dpi=180)
    plt.close()


def _condition_values(rows: List[dict], condition: str, metric: str) -> List[float]:
    return [float(r[metric]) for r in rows if r["condition"] == condition]


def _normalise_metric_means(
    rows: List[dict],
    conditions: List[str],
    metrics: List[str],
) -> Dict[str, List[float]]:
    """
    Compute mean values for each condition and metric, then normalise each
    metric independently to [0, 1].

    This makes metrics with different scales comparable in the same figure.
    """
    means_by_metric = {}

    for metric in metrics:
        means = []
        for condition in conditions:
            values = _condition_values(rows, condition, metric)
            means.append(float(np.mean(values)) if values else 0.0)
        means_by_metric[metric] = means

    normalised = {condition: [] for condition in conditions}

    for metric in metrics:
        means = means_by_metric[metric]
        min_value = min(means)
        max_value = max(means)

        for condition, value in zip(conditions, means):
            if max_value == min_value:
                normalised[condition].append(0.5)
            else:
                normalised[condition].append((value - min_value) / (max_value - min_value))

    return normalised


def plot_normalised_grouped_summary(
    rows: List[dict],
    conditions: List[str],
    condition_labels: List[str],
    metrics: List[str],
    metric_labels: List[str],
    title: str,
    output: Path,
) -> None:
    """
    Grouped bar chart with metrics normalised to [0, 1].

    This is useful when plotting metrics with different scales, such as
    fitness, explored fraction, rescued victims, and collisions.
    """
    normalised = _normalise_metric_means(rows, conditions, metrics)

    x = np.arange(len(metrics))
    width = 0.34

    plt.figure(figsize=(9.5, 4.8))

    for i, condition in enumerate(conditions):
        offset = (i - (len(conditions) - 1) / 2) * width
        plt.bar(
            x + offset,
            normalised[condition],
            width,
            label=condition_labels[i],
        )

    plt.title(title)
    plt.ylabel("Normalised mean value")
    plt.xticks(x, metric_labels, rotation=20, ha="right")
    plt.ylim(0, 1.08)
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def plot_single_metric_boxplot(
    rows: List[dict],
    conditions: List[str],
    condition_labels: List[str],
    metric: str,
    title: str,
    ylabel: str,
    output: Path,
) -> None:
    """
    Boxplot for one metric across conditions.
    Useful to show variability across random seeds.
    """
    data = [
        _condition_values(rows, condition, metric)
        for condition in conditions
    ]

    plt.figure(figsize=(6.5, 4.2))
    plt.boxplot(data, tick_labels=condition_labels, showmeans=True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()

def plot_evolution_log(csv_path: Path = RESULTS / "evolution_log.csv") -> None:
    if not csv_path.exists():
        print("No evolution_log.csv found. Run evolution first.")
        return

    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    if data.size == 0:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(data["evaluation"], data["best_fitness"], label="best")
    ax1.plot(data["evaluation"], data["avg_fitness"], label="average")
    ax1.set_ylabel("Fitness")
    ax1.set_title("GENITOR Evolution Progress")
    ax1.legend()
    ax1.grid(alpha=0.25)

    ax2.plot(data["evaluation"], data["diversity"], color="tab:orange")
    ax2.set_ylabel("Population Diversity")
    ax2.set_xlabel("Evaluation")
    ax2.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(FIGURES / "evolution_progress.png", dpi=160)
    plt.close()


# ---------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------


def experiment_baseline_vs_evolved(repetitions: int, steps: int) -> List[dict]:
    baseline = Genome()
    evolved = load_best_genome()
    rows = []

    for seed in range(repetitions):
        scenario_seed = 100 + seed
        rows.append(result_to_row("baseline", seed, evaluate_genome(baseline, steps=steps, seed=scenario_seed)))
        rows.append(result_to_row("evolved",  seed, evaluate_genome(evolved,   steps=steps, seed=scenario_seed)))

    write_csv(RESULTS / "baseline_vs_evolved.csv", rows)

    plot_experiment_dashboard(
        rows=rows,
        conditions=["baseline", "evolved"],
        condition_labels=["Baseline", "Evolved"],
        panels=[
            ("rescued_victims",  "Rescued Victims", "Victims"),
            ("detected_victims", "Detected Victims","Victims"),
            ("explored_fraction","Explored Area",   "Fraction"),
            ("repeated_visits",  "Repeated Visits", "Count"),
        ],
        title="Baseline vs Evolved Controller",
        output=FIGURES / "baseline_vs_evolved_dashboard.png",
        single_row=True,
    )

    return rows


def experiment_one_vs_two(repetitions: int, steps: int) -> List[dict]:
    g = load_best_genome()
    rows = []

    for seed in range(repetitions):
        scenario_seed = 300 + seed
        rows.append(result_to_row("one_rescue_robot",  seed, evaluate_genome(g, steps=steps, seed=scenario_seed, one_robot=True)))
        rows.append(result_to_row("scout_plus_rescue", seed, evaluate_genome(g, steps=steps, seed=scenario_seed, one_robot=False)))

    write_csv(RESULTS / "one_vs_two.csv", rows)
    plot_experiment_dashboard(
        rows=rows,
        conditions=["one_rescue_robot", "scout_plus_rescue"],
        condition_labels=["Single Rescue", "Scout + Rescue"],
        panels=[
            ("rescued_victims",  "Rescued Victims", "Victims"),
            ("detected_victims", "Detected Victims","Victims"),
            ("explored_fraction","Explored Area",   "Fraction"),
        ],
        title="Single Rescue Robot vs Scout + Rescue Team",
        output=FIGURES / "one_vs_team_dashboard.png",
        single_row=True,
    )
    return rows



def experiment_shared_map(repetitions: int, steps: int) -> List[dict]:
    g = load_best_genome()
    rows = []
    for seed in range(repetitions):
        scenario_seed = 500 + seed
        rows.append(result_to_row("independent_maps", seed, evaluate_genome(g, steps=steps, seed=scenario_seed, shared_map=False)))
        rows.append(result_to_row("shared_map",       seed, evaluate_genome(g, steps=steps, seed=scenario_seed, shared_map=True)))
    write_csv(RESULTS / "shared_map_vs_no_shared.csv", rows)
    _plot_shared_map_figure(rows, FIGURES / "shared_map_dashboard.png")
    return rows


def _plot_shared_map_figure(rows: List[dict], output: Path) -> None:
    conditions = ["independent_maps", "shared_map"]
    labels     = ["Independent Maps", "Shared Map"]
    colors     = ["#2980b9", "#27ae60"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    # Panel 1: grouped bar — rescued + detected
    metrics       = ["rescued_victims", "detected_victims"]
    metric_labels = ["Rescued", "Detected"]
    x     = np.arange(len(metrics))
    width = 0.35
    for i, (cond, label, color) in enumerate(zip(conditions, labels, colors)):
        means = [float(np.mean([r[m] for r in rows if r["condition"] == cond])) for m in metrics]
        stds  = [float(np.std ([r[m] for r in rows if r["condition"] == cond])) for m in metrics]
        axes[0].bar(x + (i - 0.5) * width, means, width,
                    label=label, color=color, yerr=stds, capsize=4, alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metric_labels)
    axes[0].set_ylabel("Victims")
    axes[0].set_title("Rescued & Detected Victims", fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    # Panels 2-4: boxplots
    for ax, (metric, title, ylabel) in zip(axes[1:], [
        ("explored_fraction",   "Explored Area",      "Fraction"),
        ("repeated_visits",     "Repeated Visits",     "Count"),
        ("robot_overlap_ratio", "Robot Overlap Ratio", "Ratio"),
    ]):
        data = [[r[metric] for r in rows if r["condition"] == c] for c in conditions]
        ax.boxplot(data, tick_labels=labels, showmeans=True)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Shared Map vs Independent Maps", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output, dpi=180)
    plt.close()


def experiment_collapsed_passage(repetitions: int, steps: int) -> List[dict]:
    g = load_best_genome()
    rows = []
    for seed in range(repetitions):
        scenario_seed = 700 + seed
        rows.append(result_to_row("normal_building",   seed, evaluate_genome(g, steps=steps, seed=scenario_seed, dynamic_block=False)))
        rows.append(result_to_row("collapsed_passage", seed, evaluate_genome(g, steps=steps, seed=scenario_seed, dynamic_block=True)))
    write_csv(RESULTS / "collapsed_passage.csv", rows)
    plot_experiment_dashboard(
        rows=rows,
        conditions=["normal_building", "collapsed_passage"],
        condition_labels=["Normal Building", "Collapsed Passage"],
        panels=[
            ("rescued_victims",  "Rescued Victims",  "Victims"),
            ("detected_victims", "Detected Victims", "Victims"),
            ("explored_fraction","Explored Area",    "Fraction"),
        ],
        title="Normal Building vs Collapsed Passage",
        output=FIGURES / "collapsed_passage_dashboard.png",
        single_row=True,
    )
    return rows


def experiment_map_resolution(repetitions: int, steps: int) -> List[dict]:
    g = load_best_genome()
    rows = []
    for seed in range(repetitions):
        scenario_seed = 900 + seed
        for cell_size in [12, 20, 32, 40]:
            result = evaluate_genome(g, steps=steps, seed=scenario_seed, cell_size=cell_size)
            rows.append(result_to_row(f"cell_{cell_size}", seed, result))
    write_csv(RESULTS / "map_resolution.csv", rows)
    plot_experiment_dashboard(
        rows=rows,
        conditions=["cell_12", "cell_20", "cell_32", "cell_40"],
        condition_labels=["12 px", "20 px", "32 px", "40 px"],
        panels=[
            ("rescued_victims",  "Rescued Victims", "Victims"),
            ("explored_fraction","Explored Area",   "Fraction"),
        ],
        title="Map Resolution Comparison",
        output=FIGURES / "map_resolution_dashboard.png",
        single_row=True,
    )
    return rows


def plot_pheromone_ablation_figure(rows: List[dict], output: Path) -> None:
    """1×4: bar (frontier_pheromone_coverage) | boxplot (changed_rate) | boxplot (repeated_visits) | boxplot (rescued_victims)."""
    conditions = ["with_pheromone", "no_pheromone"]
    labels     = ["With Pheromone", "No Pheromone"]
    colors     = ["#2980b9", "#e67e22"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    # ── Panel 1: bar chart — frontier_pheromone_coverage ─────────────────
    means = [float(np.mean([r["frontier_pheromone_coverage"] for r in rows if r["condition"] == c]))
             for c in conditions]
    stds  = [float(np.std ([r["frontier_pheromone_coverage"] for r in rows if r["condition"] == c]))
             for c in conditions]
    axes[0].bar(labels, means, color=colors, alpha=0.85, yerr=stds, capsize=5, width=0.5)
    axes[0].set_ylim(0, 1.15)
    axes[0].set_ylabel("Fraction of Frontiers with Pheromone > 0")
    axes[0].set_title("Frontier Pheromone Coverage", fontsize=12, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.25)

    # ── Panels 2-4: boxplots ──────────────────────────────────────────────
    for ax, (metric, title, ylabel) in zip(axes[1:], [
        ("pheromone_changed_frontier_rate", "Frontier Choice Changed",  "Fraction of Decisions Changed"),
        ("repeated_visits",                 "Repeated Visits",          "Count"),
        ("rescued_victims",                 "Rescued Victims",          "Victims"),
    ]):
        data = [[r[metric] for r in rows if r["condition"] == c] for c in conditions]
        ax.boxplot(data, tick_labels=labels, showmeans=True)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Pheromone Ablation: With vs Without Pheromone Layer",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output, dpi=180)
    plt.close()


def experiment_pheromone_ablation(repetitions: int, steps: int) -> List[dict]:
    g = load_best_genome()
    rows = []
    phero_steps = max(steps, 3200)
    for seed in range(repetitions):
        scenario_seed = 1100 + seed
        rows.append(result_to_row("with_pheromone", seed,
                                  evaluate_genome(g, steps=phero_steps, seed=scenario_seed, use_pheromone=True)))
        rows.append(result_to_row("no_pheromone",   seed,
                                  evaluate_genome(g, steps=phero_steps, seed=scenario_seed, use_pheromone=False)))
    write_csv(RESULTS / "pheromone_ablation.csv", rows)
    plot_pheromone_ablation_figure(rows, FIGURES / "pheromone_ablation_dashboard.png")
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
            steps_per_eval=600 if quick else 1600,
            episodes_per_genome=1 if quick else 2,
        )

    plot_evolution_log()
    experiment_baseline_vs_evolved(repetitions, steps)
    experiment_one_vs_two(repetitions, steps)
    experiment_shared_map(repetitions, steps)
    experiment_collapsed_passage(repetitions, steps)
    experiment_map_resolution(repetitions, steps)
    experiment_pheromone_ablation(repetitions, steps)

    print("Done. Check results/ and figures/.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run short experiments")
    parser.add_argument("--full", action="store_true", help="Run longer experiments")
    args = parser.parse_args()
    run_all(quick=not args.full)
