import math, random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

LANDMARKS_RANGE   = 200
MOTION_NOISE_STD  = 2.0
SENSOR_NOISE_STD  = 5.0
BEARING_NOISE_STD = 0.05

# Custom landmark layout
EXP_LANDMARKS = [
    (300, 300),   
    (200, 200),   
    (400, 200),  
    (200, 400),   
    (400, 400),  
]

# Motion parameters 
V     = 20.0   # linear speed  (px/s)
OMEGA = 2.0    # angular speed (rad/s),  orbit radius = V/OMEGA = 10 px


CONDITIONS = {
    "0 landmarks": (900, 400, 0.0),
    "1 landmark":  ( 80, 200, 0.0),
    "2 landmarks": ( 80, 300, 0.0),
    "3 landmarks": (300, 150, 0.0),
}

COLORS = {
    "0 landmarks": "#AAAAAA",
    "1 landmark":  "#D4893A",
    "2 landmarks": "#4A7FB5",
    "3 landmarks": "#5A9E6F",
}

N_SEEDS = 20
N_STEPS = 150   
DT      = 1 / 30.0

#  Motion and sensor model functions 

def update_true_pose(rx, ry, rth, v, omega, dt):
    """True robot motion: exact velocity model, no noise."""
    if abs(omega) < 1e-6:
        rx_new  = rx + v * math.cos(rth) * dt
        ry_new  = ry + v * math.sin(rth) * dt
        rth_new = rth
    else:
        r       = v / omega
        rx_new  = rx - r * math.sin(rth) + r * math.sin(rth + omega * dt)
        ry_new  = ry + r * math.cos(rth) - r * math.cos(rth + omega * dt)
        rth_new = rth + omega * dt
    return rx_new, ry_new, rth_new % (2 * math.pi)


def get_obs(cx, cy, theta):
    obs = []
    for i, (lx, ly) in enumerate(EXP_LANDMARKS):
        dist = math.hypot(lx - cx, ly - cy)
        if dist <= LANDMARKS_RANGE:
            bearing = math.atan2(ly - cy, lx - cx) - theta
            obs.append((i,
                dist    + random.gauss(0, SENSOR_NOISE_STD),
                bearing + random.gauss(0, BEARING_NOISE_STD)))
    return obs


def kf_predict(mu, sigma, v, omega, dt):
    """EKF predict step: velocity motion model + linearised Jacobian."""
    x, y, theta = mu
    if abs(omega) < 1e-6:
        xn  = x + v * math.cos(theta) * dt
        yn  = y + v * math.sin(theta) * dt
        thn = theta
        A = np.array([
            [1, 0, -v * math.sin(theta) * dt],
            [0, 1,  v * math.cos(theta) * dt],
            [0, 0,  1],
        ])
    else:
        r   = v / omega
        xn  = x - r * math.sin(theta) + r * math.sin(theta + omega * dt)
        yn  = y + r * math.cos(theta) - r * math.cos(theta + omega * dt)
        thn = theta + omega * dt
        A = np.array([
            [1, 0, -r * math.cos(theta) + r * math.cos(theta + omega * dt)],
            [0, 1, -r * math.sin(theta) + r * math.sin(theta + omega * dt)],
            [0, 0,  1],
        ])
    R = np.diag([MOTION_NOISE_STD**2, MOTION_NOISE_STD**2, 0.01])
    return np.array([xn, yn, thn % (2 * math.pi)]), A @ sigma @ A.T + R


def kf_update(mu, sigma, obs):
    """EKF update step: sequential correction for each observation."""
    for (lm_idx, z_dist, z_bearing) in obs:
        lx, ly = EXP_LANDMARKS[lm_idx]
        x, y, theta = mu
        exp_dist = math.hypot(lx - x, ly - y)
        if exp_dist < 1e-6:
            continue
        exp_bearing = math.atan2(ly - y, lx - x) - theta

        inn    = np.array([z_dist - exp_dist, z_bearing - exp_bearing])
        inn[1] = (inn[1] + math.pi) % (2 * math.pi) - math.pi   # wrap bearing

        C = np.array([
            [-(lx - x) / exp_dist,    -(ly - y) / exp_dist,    0],
            [ (ly - y) / exp_dist**2, -(lx - x) / exp_dist**2, -1],
        ])
        Q = np.diag([SENSOR_NOISE_STD**2, BEARING_NOISE_STD**2])
        K = sigma @ C.T @ np.linalg.inv(C @ sigma @ C.T + Q)

        mu    = mu + K @ inn
        sigma = (np.eye(3) - K @ C) @ sigma
    return mu, sigma


# Simulation trial 

def run_trial(sx, sy, sth, v, omega, n_steps, seed):
    """
    Run one EKF trial with a moving robot.
    The filter starts with a 50-px position offset to simulate an uncertain
    initial guess, we then observe how fast different triangulation conditions
    allow the filter to correct.
    Returns: covariance trace, position error, eigenvalue ratio per step.
    """
    random.seed(seed)
    np.random.seed(seed)

    rx, ry, rth = sx, sy, sth
    mu    = np.array([sx + 50.0, sy + 50.0, sth + 0.1], dtype=float)
    sigma = np.eye(3) * 500.0

    traces, errors, ratios, obs_counts = [], [], [], []

    for _ in range(n_steps):
        rx, ry, rth   = update_true_pose(rx, ry, rth, v, omega, DT)
        mu, sigma     = kf_predict(mu, sigma, v, omega, DT)
        obs           = get_obs(rx, ry, rth)
        obs_counts.append(len(obs))
        if obs:
            mu, sigma = kf_update(mu, sigma, obs)

        pos_cov = sigma[:2, :2]
        traces.append(np.trace(pos_cov))
        errors.append(math.hypot(mu[0] - rx, mu[1] - ry))

        eigvals = np.sort(np.linalg.eigvalsh(pos_cov))
        ratios.append(eigvals[-1] / (eigvals[0] + 1e-6))

    return traces, errors, ratios, obs_counts


# Run all conditions

results = {}
for label, (sx, sy, sth) in CONDITIONS.items():
    actual_n = sum(1 for (lx, ly) in EXP_LANDMARKS
                   if math.hypot(lx - sx, ly - sy) <= LANDMARKS_RANGE)
    print(f"{label:15s}: {actual_n} landmarks visible at start from ({sx:4d}, {sy:4d})")

    all_t, all_e, all_r, all_c = [], [], [], []
    for s in range(N_SEEDS):
        tr, er, ra, co = run_trial(sx, sy, sth, V, OMEGA, N_STEPS, seed=s)
        all_t.append(tr); all_e.append(er)
        all_r.append(ra); all_c.append(co)

    # Verify landmark visibility is stable across the entire trajectory
    counts = np.array(all_c)          
    c_min  = int(counts.min())
    c_max  = int(counts.max())
    c_mean = counts.mean()
    stable = "OK" if c_min == c_max else "WARNING: count varies!"
    print(f"  landmark count across trajectory: "
          f"min={c_min}  max={c_max}  mean={c_mean:.2f}  [{stable}]")

    results[label] = {
        "traces":     np.mean(all_t, axis=0),
        "traces_std": np.std(all_t,  axis=0),
        "errors":     np.mean(all_e, axis=0),
        "errors_std": np.std(all_e,  axis=0),
        "ratios":     np.mean(all_r, axis=0),
        "pos":        (sx, sy),
    }

# ── Plotting ───────────────────────────────────────────────────────────────────

time_axis = np.arange(N_STEPS) / 30.0

fig = plt.figure(figsize=(14, 11))
fig.patch.set_facecolor("white")
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.46, wspace=0.32)

# Panel 1: covariance trace (log scale) 
ax = fig.add_subplot(gs[0, 0])
ax.set_facecolor("white")
for label, data in results.items():
    mean, std = data["traces"], data["traces_std"]
    ax.plot(time_axis, mean, color=COLORS[label], lw=2, label=label)
    ax.fill_between(time_axis, np.maximum(mean - std, 1e-2), mean + std,
                    color=COLORS[label], alpha=0.18)
ax.set_xlabel("Time (s)", fontsize=9)
ax.set_ylabel("Trace of position covariance  (px²)", fontsize=9)
ax.set_title("Position uncertainty over time\n"
             "per number of visible landmarks  (±1σ band)", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, framealpha=0.45)
ax.set_yscale("log")
ax.grid(True, alpha=0.25, which="both")

# Panel 2: position error over time 
ax = fig.add_subplot(gs[0, 1])
ax.set_facecolor("white")
for label, data in results.items():
    mean, std = data["errors"], data["errors_std"]
    ax.plot(time_axis, mean, color=COLORS[label], lw=2, label=label)
    ax.fill_between(time_axis, np.maximum(mean - std, 0), mean + std,
                    color=COLORS[label], alpha=0.18)
ax.set_xlabel("Time (s)", fontsize=9)
ax.set_ylabel("Position error  (px)", fontsize=9)
ax.set_title("Estimation error over time\n"
             "(filter starts 50 px off, ±1σ band)", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, framealpha=0.45)
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.25)

# Panel 3: covariance ellipse elongation 
ax = fig.add_subplot(gs[1, 0])
ax.set_facecolor("white")
for label, data in results.items():
    ax.plot(time_axis, data["ratios"], color=COLORS[label], lw=2, label=label)
ax.axhline(1.0, color="#555", lw=1, ls="--", alpha=0.6)
ax.text(time_axis[-1] * 0.95, 1.15, "isotropic", ha="right", fontsize=7.5, color="#555")
ax.set_xlabel("Time (s)", fontsize=9)
ax.set_ylabel("Eigenvalue ratio  (major / minor)", fontsize=9)
ax.set_title("Covariance ellipse elongation\n"
             "(ratio = 1 means isotropic uncertainty)",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=8, framealpha=0.45)
ax.set_yscale("log")
ax.grid(True, alpha=0.25, which="both")

# Panel 4: experiment layout map 
ax = fig.add_subplot(gs[1, 1])
ax.set_facecolor("white")
ax.set_xlim(-20, 1020)
ax.set_ylim(-20, 720)
ax.invert_yaxis()
ax.set_xlabel("x  (px)", fontsize=9)
ax.set_ylabel("y  (px)", fontsize=9)
ax.set_title("Experiment layout\n"
             "(● landmarks, × start, dashed = sensor range)",
             fontsize=10, fontweight="bold")

# Landmarks
for i, (lx, ly) in enumerate(EXP_LANDMARKS):
    ax.plot(lx, ly, "ko", ms=7, zorder=5)
    ax.annotate(f"L{i}", (lx, ly), textcoords="offset points",
                xytext=(5, -12), fontsize=7)

# Robot start positions and sensor range circles
for label, data in results.items():
    sx, sy = data["pos"]
    c = COLORS[label]

    ax.plot(sx, sy, "x", color=c, ms=11, mew=2.5, zorder=6)

    ax.add_patch(patches.Circle((sx, sy), LANDMARKS_RANGE,
                                fill=False, edgecolor=c, lw=1.5, ls="--", alpha=0.7))

    ax.annotate(label, (sx, sy), textcoords="offset points",
                xytext=(6, -13), fontsize=7.5, color=c, fontweight="bold")

ax.grid(True, alpha=0.2)

fig.suptitle(
    "Effect of Visible Landmarks on EKF Localization (Moving Robot)\n",
    fontsize=12, fontweight="bold", y=1.01,
)

plt.savefig("triangulation_results.png", dpi=160,
            bbox_inches="tight", facecolor=fig.get_facecolor())
print("Saved triangulation_results.png")
