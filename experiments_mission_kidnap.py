from __future__ import annotations
import math
import random
import sys
from copy import deepcopy
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from world import (SearchRescueWorld, WIDTH, HEIGHT, DT, ROBOT_RADIUS, BLUE, PURPLE)
from controller import Genome
from occupancy_grid import OccupancyGrid

#  Robot wrapper
class SimRobot:
    """wrapper around Robot with teleport() methode -> for experiment purposes only!!!!!"""
    def __init__(self, robot_id: int, role: str, pose, color, rng_seed: int):
        from robot import Robot
        self.rng    = random.Random(rng_seed)
        self._robot = Robot(robot_id, role, pose, Genome(), color, rng_seed)
        self.role   = role

    @property
    def true_pose(self):
        return self._robot.true_pose

    @property
    def collisions(self):
        return self._robot.collisions

    @property
    def ekf(self):
        return self._robot.ekf

    def update(self, world, grid, robots, dt: float = DT):
        self._robot.update(world, grid, [r._robot for r in robots], dt)

    def teleport(self, new_pose):
        """
        teleport bot + reset EKF with high uncertainty and wrong prior
        -> not aware of teleport 
        """
        self._robot.true_pose = np.array(new_pose, dtype=float)

        offset_x = self.rng.uniform(200, 300) * self.rng.choice([-1, 1])
        offset_y = self.rng.uniform(150, 250) * self.rng.choice([-1, 1])
        margin   = 40 + ROBOT_RADIUS + 2
        wrong_x  = max(margin, min(WIDTH  - margin, float(new_pose[0]) + offset_x))
        wrong_y  = max(margin, min(HEIGHT - margin, float(new_pose[1]) + offset_y))
        wrong_th = float(new_pose[2])

        ekf  = self._robot.ekf
        n_lm = ekf.num_landmarks
        dim  = 3 + 2 * n_lm

        new_state      = np.zeros(dim, dtype=float)
        new_state[0]   = wrong_x
        new_state[1]   = wrong_y
        new_state[2]   = wrong_th

        new_sigma       = np.eye(dim, dtype=float) * 1e6
        new_sigma[0, 0] = 80_000.0
        new_sigma[1, 1] = 80_000.0
        new_sigma[2, 2] = (math.pi / 4) ** 2

        ekf.state          = new_state
        ekf.sigma          = new_sigma
        ekf.seen_landmarks = [False] * n_lm

#  Experiment-parameters
FIXED_SEED   = 42
NUM_VICTIMS  = 10     
SIM_DURATION = 180.0
TELEPORT_T   = 15.0 #teleport after 15s
N_SEEDS      = 30
SNAPSHOT_T   = [15.0, 60.0, 120.0, 180.0]

# Teleport-poses
# Scout start:   ( 90, 120) 
# Rescuer start: (995, 600) 
_TELEPORT_POOL = [
    (550.0, 350.0,  0.5),    
    ( 90.0, 600.0,  0.0),    
    (995.0, 120.0,  math.pi),
    (300.0, 500.0,  1.0),
    (750.0, 150.0, -1.0),
]

def _pick_teleport_seeded(world, seed: int):
    rng = random.Random(seed + 9999)
    pool = list(_TELEPORT_POOL)
    rng.shuffle(pool)
    for pose in pool:
        if not world.collides(pose[0], pose[1]):
            return pose
    margin = 40 + ROBOT_RADIUS + 5
    for _ in range(2000):
        x = rng.uniform(margin, WIDTH  - margin)
        y = rng.uniform(margin, HEIGHT - margin)
        if not world.collides(x, y):
            return (x, y, rng.uniform(-math.pi, math.pi))
    return (550.0, 350.0, 0.0)

def run_trial(condition: str, seed: int, world_template: SearchRescueWorld) -> dict:
    world = deepcopy(world_template)
    grid  = OccupancyGrid(cell_size=20)

    robots = [
        SimRobot(1, "scout",  ( 90, 120,  math.pi / 2), BLUE,   seed),
        SimRobot(2, "rescue", (995, 600, -math.pi / 2), PURPLE, seed + 1),
    ]

    KIDNAP_INDICES = {
        "baseline":      [],
        "scout_kidnap":  [0],
        "rescue_kidnap": [1],
    }
    kidnap_ids    = KIDNAP_INDICES[condition]
    teleport_dest = _pick_teleport_seeded(world, seed)

    steps      = int(SIM_DURATION / DT)
    teleport_s = int(TELEPORT_T / DT)
    snap_steps = {int(t / DT): t for t in SNAPSHOT_T}

    rescued_at:  dict[float, int] = {}
    detected_at: dict[float, int] = {}
    post_kidnap_collisions = 0
    mission_complete_t     = None
    first_rescue_t         = None
    rescued_at_kidnap      = 0
    detected_at_kidnap     = 0

    for step in range(steps):
        t = step * DT
        if step == teleport_s:
            rescued_at_kidnap  = world.num_rescued()
            detected_at_kidnap = world.num_detected()
            for idx in kidnap_ids:
                robots[idx].teleport(teleport_dest)

        pre_cols = [robots[idx].collisions for idx in kidnap_ids]

        for robot in robots:
            robot.update(world, grid, robots, DT)

        if step >= teleport_s and kidnap_ids:
            post_kidnap_collisions += sum(
                max(0, robots[idx].collisions - pre_cols[i])
                for i, idx in enumerate(kidnap_ids)
            )

        num_rescued  = world.num_rescued()
        num_detected = world.num_detected()

        if first_rescue_t is None and num_rescued > 0:
            first_rescue_t = t

        if step in snap_steps:
            rescued_at[snap_steps[step]]  = num_rescued
            detected_at[snap_steps[step]] = num_detected

        if mission_complete_t is None and num_rescued == len(world.victims):
            mission_complete_t = t

    for t in SNAPSHOT_T:
        if t not in rescued_at:
            rescued_at[t]  = world.num_rescued()
            detected_at[t] = world.num_detected()

    final_res = world.num_rescued()
    final_det = world.num_detected()

    return {
        "rescued_at":             rescued_at,
        "detected_at":            detected_at,
        "mission_complete_t":     mission_complete_t,
        "first_rescue_t":         first_rescue_t,
        "explored_pct":           100.0 * grid.explored_fraction(),
        "post_kidnap_collisions": post_kidnap_collisions,
        "total_victims":          len(world.victims),
        "rescued_at_kidnap":      rescued_at_kidnap,
        "detected_at_kidnap":     detected_at_kidnap,
        "rescued_after_kidnap":   max(0, final_res - rescued_at_kidnap),
        "detected_after_kidnap":  max(0, final_det - detected_at_kidnap),
    }
CONDITIONS = ["baseline", "scout_kidnap", "rescue_kidnap"]
LABELS = {
    "baseline":      "Baseline\n(no kidnap)",
    "scout_kidnap":  "Scout\nkidnapped",
    "rescue_kidnap": "Rescuer\nkidnapped",
}

#create variations between trials via seeded world gen
def _make_world(seed: int) -> SearchRescueWorld:
    return SearchRescueWorld(dynamic_block=False, seed=seed * 7 + 13, num_victims=NUM_VICTIMS)

TOTAL_VICTIMS = NUM_VICTIMS
results: dict[str, list[dict]] = {c: [] for c in CONDITIONS}

for cond in CONDITIONS:
    print(f"Running condition: {cond} ...")
    for s in range(N_SEEDS):
        world_s = _make_world(s)
        r = run_trial(cond, seed=s, world_template=world_s)
        results[cond].append(r)
        rescued_last = r["rescued_at"][SNAPSHOT_T[-1]]
        comp_str = (
            f"{r['mission_complete_t']:.0f}s"
            if r["mission_complete_t"] is not None else "DNF"
        )
        print(
            f"  seed={s:2d}  "
            f"@kidnap={r['rescued_at_kidnap']}/{TOTAL_VICTIMS}  "
            f"after={r['rescued_after_kidnap']}  "
            f"det_after={r['detected_after_kidnap']}  "
            f"@180s={rescued_last}/{TOTAL_VICTIMS}  "
            f"complete={comp_str}"
        )

#Aggregation per condition
def agg(cond: str) -> dict:
    trials = results[cond]
    n      = len(trials)

    rescued_curve  = {t: np.mean([tr["rescued_at"][t]  for tr in trials]) for t in SNAPSHOT_T}
    detected_curve = {t: np.mean([tr["detected_at"][t] for tr in trials]) for t in SNAPSHOT_T}

    complete_times  = [tr["mission_complete_t"] for tr in trials if tr["mission_complete_t"] is not None]
    completion_rate = len(complete_times) / n
    mean_complete   = float(np.mean(complete_times)) if complete_times else float("nan")

    first_times = [tr["first_rescue_t"] for tr in trials if tr["first_rescue_t"] is not None]
    mean_first  = float(np.mean(first_times)) if first_times else float("nan")

    return {
        "rescued_curve":          rescued_curve,
        "detected_curve":         detected_curve,
        "completion_rate":        completion_rate,
        "mean_complete_t":        mean_complete,
        "mean_first_t":           mean_first,
        "mean_explored":          float(np.mean([tr["explored_pct"] for tr in trials])),
        "mean_collisions":        float(np.mean([tr["post_kidnap_collisions"] for tr in trials])),
        "mean_rescued_at_kidnap": float(np.mean([tr["rescued_at_kidnap"] for tr in trials])),
        "mean_rescued_after":     float(np.mean([tr["rescued_after_kidnap"] for tr in trials])),
        "mean_detected_after":    float(np.mean([tr["detected_after_kidnap"] for tr in trials])),
    }

aggs = {c: agg(c) for c in CONDITIONS}

#Summary table -> completely generated by AI
W = 105
print("=" * W)
print(
    f"{'Condition':<22} "
    f"{'@kidnap':>7} {'after':>6} {'det+':>5} "
    f"{'@180s':>6} {'First':>7} "
    f"{'Comp%':>6} {'AvgT':>8} "
    f"{'Expl%':>6} {'Collis':>7}"
)
print("-" * W)
for cond in CONDITIONS:
    a  = aggs[cond]
    rc = a["rescued_curve"]
    first_t = f"{a['mean_first_t']:.1f}s"    if not math.isnan(a["mean_first_t"])    else "  n/a"
    avg_t   = f"{a['mean_complete_t']:.1f}s"  if not math.isnan(a["mean_complete_t"]) else "   DNF"
    print(
        f"{LABELS[cond].replace(chr(10),' '):<22} "
        f"{a['mean_rescued_at_kidnap']:>7.1f} "
        f"{a['mean_rescued_after']:>6.1f} "
        f"{a['mean_detected_after']:>5.1f} "
        f"{rc[180.0]:>6.1f} "
        f"{first_t:>7} "
        f"{100*a['completion_rate']:>5.0f}% "
        f"{avg_t:>8} "
        f"{a['mean_explored']:>5.1f}% "
        f"{a['mean_collisions']:>7.1f}"
    )
print("=" * W)

#Plots (2 x 3) -> also fully generated by AI
C = {
    "baseline":      "#5A9E6F",
    "scout_kidnap":  "#4A7FB5",
    "rescue_kidnap": "#C05050",
}
colors     = [C[c] for c in CONDITIONS]
bar_labels = [LABELS[c] for c in CONDITIONS]
x_pos      = np.arange(len(CONDITIONS))

fig = plt.figure(figsize=(16, 11))
fig.patch.set_facecolor("#F8F8F6")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.46, wspace=0.34)


# -- 1. Rescue rate curve
ax = fig.add_subplot(gs[0, 0])
ax.set_facecolor("#F8F8F6")
for cond in CONDITIONS:
    rc = aggs[cond]["rescued_curve"]
    xs = [0.0] + SNAPSHOT_T
    ys = [0.0] + [rc[t] for t in SNAPSHOT_T]
    ax.plot(xs, ys, color=C[cond], lw=2.2, marker="o", ms=5,
            label=LABELS[cond].replace("\n", " "))
ax.axvline(TELEPORT_T, color="#888", lw=1.2, ls="--", alpha=0.7)
ax.text(TELEPORT_T + 2, TOTAL_VICTIMS * 0.05, "kidnap", fontsize=7.5, color="#555")
ax.set_xlabel("Simulation time (s)", fontsize=9)
ax.set_ylabel("Avg victims rescued", fontsize=9)
ax.set_title("Rescue rate over time", fontsize=10, fontweight="bold")
ax.set_xlim(0, SIM_DURATION)
ax.set_ylim(0, TOTAL_VICTIMS * 1.15)
ax.legend(fontsize=7.5, framealpha=0.45)
ax.grid(True, alpha=0.25)


# -- 2. Detection rate curve
ax = fig.add_subplot(gs[0, 1])
ax.set_facecolor("#F8F8F6")
for cond in CONDITIONS:
    dc = aggs[cond]["detected_curve"]
    xs = [0.0] + SNAPSHOT_T
    ys = [0.0] + [dc[t] for t in SNAPSHOT_T]
    ax.plot(xs, ys, color=C[cond], lw=2.2, marker="s", ms=4,
            ls="--", label=LABELS[cond].replace("\n", " "))
ax.axvline(TELEPORT_T, color="#888", lw=1.2, ls="--", alpha=0.7)
ax.text(TELEPORT_T + 2, TOTAL_VICTIMS * 0.05, "kidnap", fontsize=7.5, color="#555")
ax.set_xlabel("Simulation time (s)", fontsize=9)
ax.set_ylabel("Avg victims detected", fontsize=9)
ax.set_title("Detection rate over time",
             fontsize=10, fontweight="bold")
ax.set_xlim(0, SIM_DURATION)
ax.set_ylim(0, TOTAL_VICTIMS * 1.15)
ax.legend(fontsize=7.5, framealpha=0.45)
ax.grid(True, alpha=0.25)

# -- 3. Rescued voor vs na kidnap
ax = fig.add_subplot(gs[0, 2])
ax.set_facecolor("#F8F8F6")
before = [aggs[c]["mean_rescued_at_kidnap"] for c in CONDITIONS]
after  = [aggs[c]["mean_rescued_after"]     for c in CONDITIONS]
w = 0.35
bars_b = ax.bar(x_pos - w/2, before, width=w, color=colors,
                edgecolor="white", lw=0.8, label=f"Before kidnap (t<={TELEPORT_T:.0f}s)")
bars_a = ax.bar(x_pos + w/2, after,  width=w, color=colors,
                edgecolor="black",  lw=0.9, alpha=0.5, label="After kidnap")
for bar, v in zip(bars_b, before):
    if v > 0.05:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{v:.1f}", ha="center", fontsize=8.5)
for bar, v in zip(bars_a, after):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f"{v:.1f}", ha="center", fontsize=8.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(bar_labels, fontsize=8)
ax.set_ylabel("Avg victims rescued", fontsize=9)
ax.set_title("Rescued before vs after\nkidnap event  <- key metric",
             fontsize=10, fontweight="bold")
ax.set_ylim(0, TOTAL_VICTIMS * 0.95)
ax.legend(fontsize=7.5, framealpha=0.45)
ax.grid(True, axis="y", alpha=0.25)


# -- 4. Mission completion rate
ax = fig.add_subplot(gs[1, 0])
ax.set_facecolor("#F8F8F6")
comp_pcts = [100 * aggs[c]["completion_rate"] for c in CONDITIONS]
bars = ax.bar(bar_labels, comp_pcts, color=colors, width=0.55,
              edgecolor="white", lw=0.8)
for bar, pct in zip(bars, comp_pcts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f"{pct:.0f}%", ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("Trials completing mission (%)", fontsize=9)
ax.set_title(f"Mission completion rate\n(within {SIM_DURATION:.0f}s)",
             fontsize=10, fontweight="bold")
ax.set_ylim(0, 118)
ax.grid(True, axis="y", alpha=0.25)


# -- 5. Completion time + first rescue
ax = fig.add_subplot(gs[1, 1])
ax.set_facecolor("#F8F8F6")
comp_times  = [aggs[c]["mean_complete_t"] for c in CONDITIONS]
first_times = [aggs[c]["mean_first_t"]    for c in CONDITIONS]
w = 0.35
for i, (t, f_t, col) in enumerate(zip(comp_times, first_times, colors)):
    if math.isnan(t):
        ax.bar(i - w/2, SIM_DURATION * 0.95, width=w,
               color=col, edgecolor="white", lw=0.8, alpha=0.35, hatch="//")
        ax.text(i - w/2, SIM_DURATION * 0.97, "DNF",
                ha="center", fontsize=9, color="#555")
    else:
        ax.bar(i - w/2, t, width=w, color=col, edgecolor="white", lw=0.8)
        ax.text(i - w/2, t + 1.5, f"{t:.0f}s", ha="center", fontsize=8.5)
    if not math.isnan(f_t):
        ax.bar(i + w/2, f_t, width=w, color=col, edgecolor="black", lw=0.7, alpha=0.55)
        ax.text(i + w/2, f_t + 1.5, f"{f_t:.0f}s", ha="center", fontsize=8.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(bar_labels, fontsize=8)
ax.set_ylabel("Time (s)", fontsize=9)
ax.set_title("Mean completion time (solid)\nvs time to first rescue (faded)",
             fontsize=10, fontweight="bold")
ax.set_ylim(0, SIM_DURATION * 1.1)
ax.grid(True, axis="y", alpha=0.25)


# -- 6. Collisions + exploration
ax  = fig.add_subplot(gs[1, 2])
ax2 = ax.twinx()
ax.set_facecolor("#F8F8F6")
collisions = [aggs[c]["mean_collisions"] for c in CONDITIONS]
explored   = [aggs[c]["mean_explored"]   for c in CONDITIONS]
alpha_cols = [
    (int(h[1:3],16)/255, int(h[3:5],16)/255, int(h[5:7],16)/255, 0.45)
    for h in [C[c] for c in CONDITIONS]
]
ax.bar( x_pos - 0.18, collisions, width=0.33, color=colors,
        edgecolor="white", lw=0.8, label="Post-kidnap collisions (left)")
ax2.bar(x_pos + 0.18, explored,   width=0.33, color=alpha_cols,
        edgecolor="white", lw=0.8, label="Explored % (right)")
ax.set_xticks(x_pos)
ax.set_xticklabels(bar_labels, fontsize=8)
ax.set_ylabel("Avg post-kidnap collisions", fontsize=9)
ax2.set_ylabel("Map explored at end (%)", fontsize=9)
ax.set_title("Post-kidnap collisions\nvs map exploration",
             fontsize=10, fontweight="bold")
ax.grid(True, axis="y", alpha=0.2)
l1, b1 = ax.get_legend_handles_labels()
l2, b2 = ax2.get_legend_handles_labels()
ax.legend(l1 + l2, b1 + b2, fontsize=7.5, framealpha=0.45, loc="upper left")


fig.suptitle(
    f"Kidnapped Robot — Mission Impact  "
    f"(N={N_SEEDS} seeds · map seed={FIXED_SEED} · {TOTAL_VICTIMS} victims · "
    f"kidnap @ t={TELEPORT_T:.0f}s, seeded pose per trial)",
    fontsize=11, fontweight="bold", y=1.01,
)

OUT = Path("experiments_mission_results2.png")
plt.savefig(str(OUT), dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nSaved -> {OUT.resolve()}")
