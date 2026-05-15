import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from world import DT  # IMPORTANT




df = pd.read_csv("result/scalability_timeseries.csv")
df = df.sort_values(["step", "scouts", "rescue"])

df["total_robots"] = df["scouts"] + df["rescue"]




scouts_vals = sorted(df["scouts"].unique())
rescue_vals = sorted(df["rescue"].unique())
steps = sorted(df["step"].unique())



def build_matrix(step, metric):
    sub = df[df["step"] == step]

    mat = np.full((len(rescue_vals), len(scouts_vals)), np.nan)

    for i, r in enumerate(rescue_vals):
        for j, s in enumerate(scouts_vals):

            vals = sub[
                (sub["scouts"] == s) &
                (sub["rescue"] == r)
                ][metric].values

            if len(vals) > 0:
                mat[i, j] = np.mean(vals)

    return mat




def compute_global_vmin_vmax(metric):
    vals = []

    for step in steps:
        mat = build_matrix(step, metric)
        vals.append(mat)

    all_vals = np.concatenate([m.flatten() for m in vals])
    all_vals = all_vals[~np.isnan(all_vals)]

    return np.min(all_vals), np.max(all_vals)




def plot_all_steps(metric, title, save=False, out_dir="plots"):

    if save:
        Path(out_dir).mkdir(exist_ok=True)

    vmin, vmax = compute_global_vmin_vmax(metric)

    for step in steps:

        mat = build_matrix(step, metric)

        plt.figure(figsize=(6, 5))

        im = plt.imshow(
            mat,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax
        )

        plt.colorbar(im, label=metric)

        plt.xticks(range(len(scouts_vals)), scouts_vals)
        plt.yticks(range(len(rescue_vals)), rescue_vals)

        plt.xlabel("Scouts")
        plt.ylabel("Rescue")
        plt.title(f"{title} | step={step}")

        plt.tight_layout()

        if save:
            plt.savefig(f"{out_dir}/{metric}_step_{step}.png", dpi=150)
            plt.close()
        else:
            plt.show()




def compute_completion_df():

    grouped = df.groupby(["scouts", "rescue", "seed"]).agg({
        "step": "max"
    }).reset_index()

    grouped["completion_time"] = grouped["step"] * DT

    return grouped.groupby(["scouts", "rescue"]).agg({
        "completion_time": "mean"
    }).reset_index()


def plot_completion_heatmap(save=False, out_dir="plots"):

    mean_df = compute_completion_df()

    mat = np.full((len(rescue_vals), len(scouts_vals)), np.nan)

    for _, row in mean_df.iterrows():
        i = rescue_vals.index(row["rescue"])
        j = scouts_vals.index(row["scouts"])
        mat[i, j] = row["completion_time"]

    plt.figure(figsize=(6, 5))

    im = plt.imshow(
        mat,
        origin="lower",
        aspect="auto",
        cmap="plasma_r"
    )

    plt.colorbar(im, label="Avg Completion Time (s)")

    plt.xticks(range(len(scouts_vals)), scouts_vals)
    plt.yticks(range(len(rescue_vals)), rescue_vals)

    plt.xlabel("Scouts")
    plt.ylabel("Rescue")
    plt.title("Average Completion Time Heatmap")

    plt.tight_layout()

    if save:
        Path(out_dir).mkdir(exist_ok=True)
        plt.savefig(f"{out_dir}/completion_time_heatmap.png", dpi=150)
        plt.close()
    else:
        plt.show()



# collapse per run (removes time dimension)
final_df = df.groupby(["scouts", "rescue", "seed"]).agg({
    "explored_fraction": "max",
    "collisions": "sum",
    "step": "max",
    "rescued_victims": "max"
}).reset_index()

final_df["total_robots"] = final_df["scouts"] + final_df["rescue"]
final_df["mission_completed"] = final_df["rescued_victims"] > 0
final_df["completion_time"] = final_df["step"] * DT



def plot_success_rate():
    g = final_df.groupby("total_robots")["mission_completed"].mean()

    plt.figure()
    (g * 100).plot(marker="o")
    plt.ylabel("Success Rate (%)")
    plt.xlabel("Total Robots")
    plt.title("Mission Success Rate vs Team Size")
    plt.grid()
    plt.show()




def plot_exploration():
    g = final_df.groupby("total_robots")["explored_fraction"].mean()
    std = final_df.groupby("total_robots")["explored_fraction"].std()

    x = g.index

    plt.figure()
    plt.plot(x, g.values, marker="o")
    plt.fill_between(x, g - std, g + std, alpha=0.2)

    plt.ylabel("Explored Fraction")
    plt.xlabel("Total Robots")
    plt.title("Exploration vs Team Size")
    plt.grid()
    plt.show()



def plot_collisions():
    g = final_df.groupby("total_robots")["collisions"].mean()
    std = final_df.groupby("total_robots")["collisions"].std()

    x = g.index

    plt.figure()
    plt.plot(x, g.values, marker="o")
    plt.fill_between(x, g - std, g + std, alpha=0.2)

    plt.ylabel("Collisions")
    plt.xlabel("Total Robots")
    plt.title("Collisions vs Team Size")
    plt.grid()
    plt.show()




def plot_completion_time():
    g = final_df.groupby("total_robots")["completion_time"].mean()
    std = final_df.groupby("total_robots")["completion_time"].std()

    x = g.index

    plt.figure()
    plt.plot(x, g.values, marker="o")
    plt.fill_between(x, g - std, g + std, alpha=0.2)

    plt.ylabel("Completion Time (s)")
    plt.xlabel("Total Robots")
    plt.title("Completion Time vs Team Size")
    plt.grid()
    plt.show()




def plot_efficiency():
    g = final_df.groupby("total_robots")["explored_fraction"].mean()

    efficiency = g / g.index

    plt.figure()
    efficiency.plot(marker="o")

    plt.ylabel("Exploration per Robot")
    plt.xlabel("Total Robots")
    plt.title("Marginal Efficiency of Robots")
    plt.grid()
    plt.show()

def compute_victims_df():
    grouped = df.groupby(["scouts", "rescue", "seed"]).agg({
        "rescued_victims": "max",
        "detected_victims": "max"
    }).reset_index()

    return grouped.groupby(["scouts", "rescue"]).agg({
        "rescued_victims": "mean",
        "detected_victims": "mean"
    }).reset_index()


def plot_victims_heatmap(save=False, out_dir="plots"):

    mean_df = compute_victims_df()

    rescued_mat = np.full((len(rescue_vals), len(scouts_vals)), np.nan)
    detected_mat = np.full((len(rescue_vals), len(scouts_vals)), np.nan)

    for _, row in mean_df.iterrows():
        i = rescue_vals.index(row["rescue"])
        j = scouts_vals.index(row["scouts"])

        rescued_mat[i, j] = row["rescued_victims"]
        detected_mat[i, j] = row["detected_victims"]


    plt.figure(figsize=(6, 5))

    im = plt.imshow(
        rescued_mat,
        origin="lower",
        aspect="auto",
        cmap="YlGnBu"
    )

    plt.colorbar(im, label="Avg Rescued Victims")

    plt.xticks(range(len(scouts_vals)), scouts_vals)
    plt.yticks(range(len(rescue_vals)), rescue_vals)

    plt.xlabel("Scouts")
    plt.ylabel("Rescue")
    plt.title("Average Rescued Victims (Scouts vs Rescue)")

    plt.tight_layout()

    if save:
        Path(out_dir).mkdir(exist_ok=True)
        plt.savefig(f"{out_dir}/rescued_victims_heatmap.png", dpi=150)
        plt.close()
    else:
        plt.show()


    plt.figure(figsize=(6, 5))

    im = plt.imshow(
        detected_mat,
        origin="lower",
        aspect="auto",
        cmap="magma"
    )

    plt.colorbar(im, label="Avg Detected Victims")

    plt.xticks(range(len(scouts_vals)), scouts_vals)
    plt.yticks(range(len(rescue_vals)), rescue_vals)

    plt.xlabel("Scouts")
    plt.ylabel("Rescue")
    plt.title("Average Detected Victims (Scouts vs Rescue)")

    plt.tight_layout()

    if save:
        plt.savefig(f"{out_dir}/detected_victims_heatmap.png", dpi=150)
        plt.close()
    else:
        plt.show()


def compute_rescued_victims_df():
    grouped = df.groupby(["scouts", "rescue", "seed"]).agg({
        "rescued_victims": "max"
    }).reset_index()

    return grouped.groupby(["scouts", "rescue"]).agg({
        "rescued_victims": "mean"
    }).reset_index()


def plot_rescued_victims_heatmap(save=False, out_dir="plots"):

    mean_df = compute_rescued_victims_df()

    mat = np.full((len(rescue_vals), len(scouts_vals)), np.nan)

    for _, row in mean_df.iterrows():
        i = rescue_vals.index(row["rescue"])
        j = scouts_vals.index(row["scouts"])
        mat[i, j] = row["rescued_victims"]

    plt.figure(figsize=(6, 5))

    im = plt.imshow(
        mat,
        origin="lower",
        aspect="auto",
        cmap="YlGnBu"
    )

    plt.colorbar(im, label="Avg Rescued Victims")

    plt.xticks(range(len(scouts_vals)), scouts_vals)
    plt.yticks(range(len(rescue_vals)), rescue_vals)

    plt.xlabel("Scouts")
    plt.ylabel("Rescue")
    plt.title("Average Rescued Victims (Scouts vs Rescue)")

    plt.tight_layout()

    if save:
        Path(out_dir).mkdir(exist_ok=True)
        plt.savefig(f"{out_dir}/rescued_victims_heatmap.png", dpi=150)
        plt.close()
    else:
        plt.show()
# ============================================================
# RUN EVERYTHING
# ============================================================
def print_best_completion_time():
    g = final_df.groupby("total_robots")["completion_time"].mean()

    best_team = g.idxmin()
    best_time = g.min()

    print("\n==== BEST AVERAGE COMPLETION TIME ====")
    print(f"Best team size: {best_team} robots")
    print(f"Average completion time: {best_time:.2f} s")

    print("\nAll averages:")
    for k, v in g.items():
        print(f"{k} robots -> {v:.2f} s")
plot_all_steps(
    metric="explored_fraction",
    title="Exploration Over Time (Scouts vs Rescue)",
    save=False
)

plot_all_steps(
    metric="rescued_victims",
    title="Rescue Progress Over Time (Scouts vs Rescue)",
    save=False
)

plot_completion_heatmap(save=False)
plot_victims_heatmap(save=False)
plot_rescued_victims_heatmap(save=False)
# NEW ANALYSIS PLOTS
plot_success_rate()
plot_exploration()
plot_collisions()
plot_completion_time()
plot_efficiency()
print_best_completion_time()