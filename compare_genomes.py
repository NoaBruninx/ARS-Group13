"""
compare_genomes.py
مقایسه سریع baseline vs evolved بدون نیاز به pygame.

اجرا:
    python compare_genomes.py
"""

import json
import numpy as np
from pathlib import Path
from controller import Genome
from evolution import evaluate_genome

STEPS = 1800   # تعداد قدم‌های شبیه‌سازی (~30 ثانیه)
SEEDS = [1, 2, 3]  # چند seed مختلف برای مقایسه منصفانه


def load_evolved() -> Genome:
    path = Path("results/best_genome.json")
    if not path.exists():
        print("هیچ genome evolved پیدا نشد. فقط baseline اجرا می‌شود.")
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    defaults = Genome().to_dict()
    defaults.update({k: v for k, v in data.items() if k in defaults})
    return Genome(**defaults)


def run_condition(genome: Genome, label: str, seeds: list) -> dict:
    print(f"\n--- {label} ---")
    results = []
    for seed in seeds:
        r = evaluate_genome(genome, steps=STEPS, seed=seed)
        results.append(r)
        print(f"  seed={seed}: rescued={r.rescued_victims}, "
              f"explored={r.explored_fraction:.2f}, "
              f"fitness={r.fitness:.0f}")

    return {
        "label": label,
        "rescued_mean": float(np.mean([r.rescued_victims for r in results])),
        "rescued_std":  float(np.std([r.rescued_victims for r in results])),
        "explored_mean": float(np.mean([r.explored_fraction for r in results])),
        "fitness_mean": float(np.mean([r.fitness for r in results])),
        "fitness_std":  float(np.std([r.fitness for r in results])),
        "collisions_mean": float(np.mean([r.collisions for r in results])),
        "loc_error_mean": float(np.mean([r.mean_localization_error for r in results])),
    }


def main():
    baseline = Genome()
    evolved  = load_evolved()

    results = []
    results.append(run_condition(baseline, "Baseline (default genome)", SEEDS))
    if evolved:
        results.append(run_condition(evolved, "Evolved genome", SEEDS))

    print("\n" + "=" * 55)
    print(f"{'متریک':<28} {'Baseline':>12} {'Evolved':>12}")
    print("=" * 55)
    metrics = [
        ("Fitness (mean ± std)",
         f"{results[0]['fitness_mean']:.0f} ± {results[0]['fitness_std']:.0f}",
         f"{results[1]['fitness_mean']:.0f} ± {results[1]['fitness_std']:.0f}" if len(results) > 1 else "—"),
        ("Rescued victims (mean ± std)",
         f"{results[0]['rescued_mean']:.1f} ± {results[0]['rescued_std']:.1f}",
         f"{results[1]['rescued_mean']:.1f} ± {results[1]['rescued_std']:.1f}" if len(results) > 1 else "—"),
        ("Explored fraction",
         f"{results[0]['explored_mean']:.3f}",
         f"{results[1]['explored_mean']:.3f}" if len(results) > 1 else "—"),
        ("Collisions",
         f"{results[0]['collisions_mean']:.1f}",
         f"{results[1]['collisions_mean']:.1f}" if len(results) > 1 else "—"),
        ("Mean localization error (px)",
         f"{results[0]['loc_error_mean']:.1f}",
         f"{results[1]['loc_error_mean']:.1f}" if len(results) > 1 else "—"),
    ]
    for name, b, e in metrics:
        print(f"{name:<28} {b:>12} {e:>12}")
    print("=" * 55)

    # ذخیره نتیجه
    out = Path("results/genome_comparison.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nنتایج در {out} ذخیره شد.")


if __name__ == "__main__":
    main()