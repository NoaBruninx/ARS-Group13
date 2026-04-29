import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# ARS Assignment - Part 3 - Group 13
# Experiment: How does landmark visibility range affect mapping quality?


# World parameters 
WIDTH, HEIGHT = 1000, 700
CELL = 10
COLS = math.ceil(WIDTH / CELL)
ROWS = math.ceil(HEIGHT / CELL)


INTERNAL_WALLS = [
    (200, 100, 200, 600),
    (200, 450, 340, 450),
    (400, 100, 400, 350),
    (200, 100, 600, 100),
    (600, 100, 600, 400),
    (600, 400, 800, 400),
    (800, 200, 800, 700),
    (800, 450, 1000, 450),
    (200, 700, 550, 700),
    (650, 700, 800, 700),
]

ALL_WALLS = INTERNAL_WALLS + [
    (0, 0, WIDTH - 1, 0),
    (WIDTH - 1, 0, WIDTH - 1, HEIGHT - 1),
    (WIDTH - 1, HEIGHT - 1, 0, HEIGHT - 1),
    (0, HEIGHT - 1, 0, 0),
]

LANDMARKS = [
    (150, 200),
    (500, 300),
    (750, 500),
    (300, 600),
    (850, 150),
]

# Sensor / robot parameters
NUM_SENSORS = 12
ANG_STEP = 2 * math.pi / NUM_SENSORS
MAX_SENSOR_RANGE = 200
CIRCLE_RADIUS = 18

MOTION_NOISE_STD = 2.0
SENSOR_NOISE_STD = 5.0
BEARING_NOISE_STD = 0.05

# Occupancy grid log-odds constants
P_OCC = 0.75
P_FREE = 0.35
L_PRIOR = math.log(0.5 / 0.5)        
L_OCC = math.log(P_OCC / (1 - P_OCC))
L_FREE = math.log(P_FREE / (1 - P_FREE))
LOG_MIN, LOG_MAX = -5.0, 5.0

CONF_THRESH = 0.5   # log-odds threshold for occupied/free classification

# Landmark visibility ranges to test
LANDMARK_RANGES = [50, 100, 150, 200, 300, 500]


# Geometry helpers
def _intersect(ax, ay, bx, by, cx, cy, dx, dy):
    d = (ax - bx) * (cy - dy) - (ay - by) * (cx - dx)
    if abs(d) < 1e-10:
        return None
    t = ((ax - cx) * (cy - dy) - (ay - cy) * (cx - dx)) / d
    u = -((ax - bx) * (ay - cy) - (ay - by) * (ax - cx)) / d
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return ax + t * (bx - ax), ay + t * (by - ay)
    return None


def raycast(x, y, th):
    dists = []
    for i in range(NUM_SENSORS):
        a = th + i * ANG_STEP
        ex = x + math.cos(a) * MAX_SENSOR_RANGE
        ey = y + math.sin(a) * MAX_SENSOR_RANGE
        best = MAX_SENSOR_RANGE - CIRCLE_RADIUS
        for x1, y1, x2, y2 in ALL_WALLS:
            hit = _intersect(x, y, ex, ey, x1, y1, x2, y2)
            if hit:
                dist = math.hypot(hit[0] - x, hit[1] - y)
                if dist < best:
                    best = dist
        dists.append(best)
    return dists

# EKF helpers
def ekf_predict(mu, sigma, v, omega, dt):
    x, y, th = mu
    if abs(omega) < 1e-6:
        x_new = x + v * math.cos(th) * dt
        y_new = y + v * math.sin(th) * dt
        th_new = th
        A = np.array([
            [1, 0, -v * math.sin(th) * dt],
            [0, 1,  v * math.cos(th) * dt],
            [0, 0,  1],
        ])
    else:
        r = v / omega
        x_new  = x - r * math.sin(th) + r * math.sin(th + omega * dt)
        y_new  = y + r * math.cos(th) - r * math.cos(th + omega * dt)
        th_new = th + omega * dt
        A = np.array([
            [1, 0, -r * math.cos(th) + r * math.cos(th + omega * dt)],
            [0, 1, -r * math.sin(th) + r * math.sin(th + omega * dt)],
            [0, 0,  1],
        ])
    R = np.diag([MOTION_NOISE_STD**2, MOTION_NOISE_STD**2, 0.01])
    mu_bar = np.array([x_new, y_new, th_new % (2 * math.pi)])
    sigma_bar = A @ sigma @ A.T + R
    return mu_bar, sigma_bar


def ekf_update(mu_bar, sigma_bar, observations):
    mu = mu_bar.copy()
    sigma = sigma_bar.copy()
    for lm_idx, z_dist, z_bearing in observations:
        lx, ly = LANDMARKS[lm_idx]
        x, y, th = mu
        exp_dist = math.hypot(lx - x, ly - y)
        if exp_dist < 1e-6:
            continue
        exp_bearing = math.atan2(ly - y, lx - x) - th

        z     = np.array([z_dist, z_bearing])
        z_hat = np.array([exp_dist, exp_bearing])

        C = np.array([
            [-(lx - x) / exp_dist, -(ly - y) / exp_dist,  0],
            [ (ly - y) / exp_dist**2, -(lx - x) / exp_dist**2, -1],
        ])
        Q = np.diag([SENSOR_NOISE_STD**2, BEARING_NOISE_STD**2])
        K = sigma @ C.T @ np.linalg.inv(C @ sigma @ C.T + Q)

        innov = z - z_hat
        innov[1] = (innov[1] + math.pi) % (2 * math.pi) - math.pi

        # Skip update if innovation is implausibly large (kidnap-like outlier)
        if abs(innov[0]) > 150:
            continue

        mu    = mu + K @ innov
        sigma = (np.eye(3) - K @ C) @ sigma

    return mu, sigma


def get_observations(true_x, true_y, true_th, lm_range):
    obs = []
    for i, (lx, ly) in enumerate(LANDMARKS):
        dist = math.hypot(lx - true_x, ly - true_y)
        if dist <= lm_range:
            bearing = math.atan2(ly - true_y, lx - true_x) - true_th
            obs.append((
                i,
                dist + random.gauss(0, SENSOR_NOISE_STD),
                bearing + random.gauss(0, BEARING_NOISE_STD),
            ))
    return obs

# Occupancy grid helpers
def _point_to_segment_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-10:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def collides_at_position(cx, cy):
    for (ax, ay, bx, by) in INTERNAL_WALLS:
        if _point_to_segment_dist(cx, cy, ax, ay, bx, by) < CIRCLE_RADIUS:
            return True
    return False



def make_grid():
    return np.zeros((ROWS, COLS), dtype=np.float32)


def to_grid(x, y):
    return int(y // CELL), int(x // CELL)


def in_grid(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS


def cells_along_ray(x0, y0, x1, y1):
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length < 1e-9:
        r, c = to_grid(x0, y0)
        return [(r, c)] if in_grid(r, c) else []
    step = max(1.0, CELL / 2.0)
    n = max(1, int(length / step))
    cells, last = [], None
    for s in range(n + 1):
        t = s / n
        r, c = to_grid(x0 + t * dx, y0 + t * dy)
        if in_grid(r, c) and (r, c) != last:
            cells.append((r, c))
            last = (r, c)
    return cells

# Update log-odds grid from EKF pose and sensor distances

def update_grid(grid, pose, dists):
    rx, ry, rth = pose
    max_reading = MAX_SENSOR_RANGE - CIRCLE_RADIUS

    rr, rc = to_grid(rx, ry)
    if in_grid(rr, rc):
        grid[rr, rc] += L_FREE

    for i, d in enumerate(dists):
        a = rth + i * ANG_STEP
        sx = rx + math.cos(a) * CIRCLE_RADIUS
        sy = ry + math.sin(a) * CIRCLE_RADIUS
        measured = max(0.0, min(float(d), max_reading))
        hit = measured < (max_reading - 1.0)
        ex = sx + math.cos(a) * measured
        ey = sy + math.sin(a) * measured

        ray = cells_along_ray(sx, sy, ex, ey)
        if not ray:
            continue
        free_cells = ray[:-1] if hit else ray
        for r, c in set(free_cells):
            grid[r, c] += L_FREE - L_PRIOR
        if hit:
            r, c = ray[-1]
            if in_grid(r, c):
                grid[r, c] += L_OCC - L_PRIOR

    np.clip(grid, LOG_MIN, LOG_MAX, out=grid)

# Ground-truth occupied cells 
def build_ground_truth():
    gt = np.zeros((ROWS, COLS), dtype=bool)
    for x1, y1, x2, y2 in ALL_WALLS:
        n = max(1, int(math.hypot(x2 - x1, y2 - y1)))
        for i in range(n):
            x = x1 + (x2 - x1) * i / n
            y = y1 + (y2 - y1) * i / n
            r, c = to_grid(x, y)
            if in_grid(r, c):
                gt[r, c] = True
    return gt


GT = build_ground_truth()


# Metrics

def compute_metrics(grid):
    occ = grid > CONF_THRESH
    tp = np.sum(occ & GT)
    fp = np.sum(occ & ~GT)
    fn = np.sum(~occ & GT)
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


# Robot path: lawnmower covering the whole arena
def lawnmower_path(stripe_step=60, margin=40):
    wps, left = [], True
    y = margin
    while y <= HEIGHT - margin:
        wps += [(margin, y), (WIDTH - margin, y)] if left else [(WIDTH - margin, y), (margin, y)]
        y += stripe_step
        left = not left
    return wps

# Run one simulation for a given landmark range
SPEED = 120.0
DT    = 0.05
REACH = 15.0

def run(lm_range):
    grid = make_grid()
    wps  = lawnmower_path()

    # Start position
    tx, ty, tth = float(wps[0][0]), float(wps[0][1]), 0.0

    # EKF state (initially well-aligned with true pose)
    mu    = np.array([tx, ty, tth])
    sigma = np.eye(3) * 100.0

    wi = 1
    true_trail = [(tx, ty)]
    ekf_trail  = [(mu[0], mu[1])]

    while wi < len(wps):
        gx, gy = wps[wi]
        if math.hypot(gx - tx, gy - ty) < REACH:
            wi += 1
            continue

        # Desired heading towards waypoint
        desired_th = math.atan2(gy - ty, gx - tx)
        dist       = math.hypot(gx - tx, gy - ty)
        v          = min(SPEED, dist / DT)
        omega      = (desired_th - tth + math.pi) % (2 * math.pi) - math.pi
        omega     /= DT   

        # Clamp omega so the robot turns smoothly
        omega = max(-3.0, min(3.0, omega))

        # Move true robot (no noise in position for fair comparison)
        tx_new = tx + v * math.cos(tth) * DT
        ty_new = ty + v * math.sin(tth) * DT
        tth    = (tth + omega * DT) % (2 * math.pi)

        # Move freely (lawnmower keeps full coverage)
        tx_new = max(CIRCLE_RADIUS + 2, min(WIDTH  - CIRCLE_RADIUS - 2, tx_new))
        ty_new = max(CIRCLE_RADIUS + 2, min(HEIGHT - CIRCLE_RADIUS - 2, ty_new))
        tx, ty = tx_new, ty_new

        true_trail.append((tx, ty))

        # EKF predict
        mu, sigma = ekf_predict(mu, sigma, v, omega, DT)

        # EKF update with landmark observations from true position
        obs = get_observations(tx, ty, tth, lm_range)
        if obs:
            mu, sigma = ekf_update(mu, sigma, obs)

        ekf_trail.append((mu[0], mu[1]))

        # Only take sensor readings from valid (non-wall) positions
        if not collides_at_position(tx, ty):
            dists = raycast(tx, ty, tth)
            update_grid(grid, mu, dists)

    return grid, np.array(true_trail), np.array(ekf_trail)

# Run all conditions

print("Running landmark range experiment...")
print(f"{'Range':>6} | {'Precision':>9} | {'Recall':>6} | {'F1':>6}")
print("-" * 38)

NUM_SEEDS = 5
SEEDS = [42, 7, 123, 99, 256]

results = []
for lm_range in LANDMARK_RANGES:
    seed_metrics = []
    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        grid, true_trail, ekf_trail = run(lm_range)
        seed_metrics.append(compute_metrics(grid))
    # Average metrics across seeds
    m = {k: float(np.mean([s[k] for s in seed_metrics])) for k in ("precision", "recall", "f1")}
    m_std = {k: float(np.std([s[k]  for s in seed_metrics])) for k in ("precision", "recall", "f1")}
    results.append((lm_range, grid, true_trail, ekf_trail, m, m_std))
    print(f"{lm_range:>6} | {m['precision']:>9.3f} | {m['recall']:>6.3f} | {m['f1']:>6.3f}  (±{m_std['f1']:.3f})")


# Plot 1: metrics vs landmark range
ranges  = [r[0] for r in results]
prec    = [r[4]["precision"] for r in results]
rec     = [r[4]["recall"]    for r in results]
f1s     = [r[4]["f1"]        for r in results]
prec_e  = [r[5]["precision"] for r in results]
rec_e   = [r[5]["recall"]    for r in results]
f1s_e   = [r[5]["f1"]        for r in results]

fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(ranges, prec, yerr=prec_e, marker="o", capsize=4, label="Precision")
ax.errorbar(ranges, rec,  yerr=rec_e,  marker="o", capsize=4, label="Recall")
ax.errorbar(ranges, f1s,  yerr=f1s_e,  marker="o", capsize=4, label="F1")
ax.set_xlabel("Landmark visibility range (px)")
ax.set_ylabel("Score")
ax.set_title("Occupancy-grid quality vs landmark visibility range\n(EKF pose used for mapping)")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("landmark_range_metrics.png", dpi=150)
plt.show()


# Plot 2: visual map comparison for each range
n = len(results)
ncols = 3
nrows = math.ceil(n / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 4 + 1))
fig.suptitle("Occupancy maps — different landmark visibility ranges", fontsize=11)

legend_patches = [
    mpatches.Patch(color=[0.15, 0.55, 0.25], label="True positive (wall found)"),
    mpatches.Patch(color=[0.85, 0.25, 0.20], label="False positive (phantom wall)"),
    mpatches.Patch(color=[0.95, 0.65, 0.10], label="False negative (missed wall)"),
    mpatches.Patch(color=[0.95, 0.95, 0.95], label="Free space"),
    mpatches.Patch(color=[0.75, 0.75, 0.75], label="Unknown"),
    mlines.Line2D([], [], color="grey",    linewidth=1.2, alpha=0.6, label="True path"),
    mlines.Line2D([], [], color="#4488cc", linewidth=1.2, alpha=0.8, label="EKF estimated path"),
]

for idx, (lm_range, grid, true_trail, ekf_trail, m, _) in enumerate(results):
    ax = axes[idx // ncols][idx % ncols] if nrows > 1 else axes[idx]

    occ  = grid > CONF_THRESH
    free = grid < -CONF_THRESH

    img = np.ones((ROWS, COLS, 3)) * 0.75   
    img[free] = [0.95, 0.95, 0.95]        

    img[occ  &  GT] = [0.15, 0.55, 0.25]   
    img[occ  & ~GT] = [0.85, 0.25, 0.20]   
    img[~occ &  GT] = [0.95, 0.65, 0.10]   

    ax.imshow(img, origin="upper", extent=[0, WIDTH, HEIGHT, 0])

    if len(true_trail) > 1:
        ax.plot(true_trail[:, 0], true_trail[:, 1],
                color="grey", linewidth=0.4, alpha=0.4, label="True path")
    if len(ekf_trail) > 1:
        ax.plot(ekf_trail[:, 0], ekf_trail[:, 1],
                color="#4488cc", linewidth=0.5, alpha=0.6, label="EKF path")

    ax.set_title(
        f"Range={lm_range} px\nP={m['precision']:.2f}  R={m['recall']:.2f}  F1={m['f1']:.2f}",
        fontsize=9,
    )
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(HEIGHT, 0)
    ax.set_xticks([])
    ax.set_yticks([])

for idx in range(n, nrows * ncols):
    axes[idx // ncols][idx % ncols].set_visible(False)

fig.legend(handles=legend_patches, loc="lower center", ncol=4, fontsize=8,
           bbox_to_anchor=(0.5, 0.0), frameon=True)

plt.tight_layout(rect=[0, 0.08, 1, 0.96], h_pad=3.0)
plt.savefig("landmark_range_maps.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nSaved: landmark_range_metrics.png  landmark_range_maps.png")
