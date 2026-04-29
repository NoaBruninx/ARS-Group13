import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


WIDTH, HEIGHT = 1000, 700
CELL = 10

NUM_SENSORS = 12
ANG_STEP = 2 * math.pi / NUM_SENSORS
MAX_RANGE = 200


P_OCC = 0.75
P_FREE = 0.35

L_OCC = math.log(P_OCC / (1 - P_OCC))
L_FREE = math.log(P_FREE / (1 - P_FREE))

CONF_THRESH = 0.5


SIGMA = 15


WALLS = [
    # interior walls
    (200,100,200,600),
    (200,450,340,450),
    (400,100,400,350),
    (200,100,600,100),
    (600,100,600,400),
    (600,400,800,400),
    (800,200,800,700),
    (800,450,1000,450),
    (200,700,550,700),
    (650,700,800,700),
    # boundary walls (edges of the map)
    (0, 0, WIDTH-1, 0),
    (WIDTH-1, 0, WIDTH-1, HEIGHT-1),
    (WIDTH-1, HEIGHT-1, 0, HEIGHT-1),
    (0, HEIGHT-1, 0, 0),
]

ALL_WALLS = WALLS


def intersect(ax, ay, bx, by, cx, cy, dx, dy):
    d = (ax-bx)*(cy-dy) - (ay-by)*(cx-dx)
    if abs(d) < 1e-10:
        return None
    t = ((ax-cx)*(cy-dy) - (ay-cy)*(cx-dx)) / d
    u = -((ax-bx)*(ay-cy) - (ay-by)*(ax-cx)) / d
    if 0 <= t <= 1 and 0 <= u <= 1:
        return ax + t*(bx-ax), ay + t*(by-ay)
    return None


def raycast(x, y, th):
    dists = []
    for i in range(NUM_SENSORS):
        a = th + i * ANG_STEP
        sx, sy = x, y
        ex = x + math.cos(a) * MAX_RANGE
        ey = y + math.sin(a) * MAX_RANGE
        best = MAX_RANGE
        for x1, y1, x2, y2 in ALL_WALLS:
            hit = intersect(sx, sy, ex, ey, x1, y1, x2, y2)
            if hit:
                dist = math.hypot(hit[0]-sx, hit[1]-sy)
                best = min(best, dist)
        dists.append(best)
    return dists


def add_noise(distances, sigma):
    return [max(0, d + random.gauss(0, sigma)) for d in distances]


def move_toward(x, y, th, tx, ty, speed, dt):
    """Move directly toward target at constant speed, update heading."""
    desired = math.atan2(ty - y, tx - x)
    dist = math.hypot(tx - x, ty - y)
    step = min(speed * dt, dist)
    x += math.cos(desired) * step
    y += math.sin(desired) * step
    return x, y, desired


def make_grid():
    r = math.ceil(HEIGHT / CELL)
    c = math.ceil(WIDTH / CELL)
    return np.zeros((r, c), dtype=np.float32)


def to_grid(x, y):
    return int(y // CELL), int(x // CELL)


def update_grid(grid, x, y, th, dists):
    rows, cols = grid.shape
    for i, d in enumerate(dists):
        a = th + i * ANG_STEP
        steps = max(1, int(d / CELL))
        for s in range(steps):
            fx = x + math.cos(a) * (s * CELL)
            fy = y + math.sin(a) * (s * CELL)
            gx, gy = to_grid(fx, fy)
            if 0 <= gx < rows and 0 <= gy < cols:
                grid[gx, gy] += L_FREE
        if d < MAX_RANGE:
            ex = x + math.cos(a) * d
            ey = y + math.sin(a) * d
            gx, gy = to_grid(ex, ey)
            if 0 <= gx < rows and 0 <= gy < cols:
                grid[gx, gy] += L_OCC
    np.clip(grid, -5, 5, out=grid)


GT_DILATION = 0

def build_ground_truth(grid_shape):
    gt = np.zeros(grid_shape, dtype=bool)
    rows, cols = grid_shape
    for x1, y1, x2, y2 in WALLS:
        steps = int(math.hypot(x2-x1, y2-y1))
        for i in range(max(1, steps)):
            x = x1 + (x2-x1)*i/steps
            y = y1 + (y2-y1)*i/steps
            r, c = to_grid(x, y)
            for dr in range(-GT_DILATION, GT_DILATION + 1):
                for dc in range(-GT_DILATION, GT_DILATION + 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        gt[nr, nc] = True
    return gt



def lawnmower_path(stripe_step=40, margin=20):
    waypoints = []
    y = margin
    left_to_right = True
    while y <= HEIGHT - margin:
        if left_to_right:
            waypoints.append((margin,          y))
            waypoints.append((WIDTH - margin,  y))
        else:
            waypoints.append((WIDTH - margin,  y))
            waypoints.append((margin,          y))
        y += stripe_step
        left_to_right = not left_to_right
    return waypoints



SPEED = 120.0
DT    = 0.1

grid = make_grid()
wp   = lawnmower_path(stripe_step=40, margin=20)

x, y, th = wp[0][0], wp[0][1], 0.0
wi = 1
robot_trail = [(x, y)]

REACH_DIST = 15.0

while wi < len(wp):
    tx, ty = wp[wi]
    if math.hypot(tx - x, ty - y) < REACH_DIST:
        wi += 1
        continue

    x, y, th = move_toward(x, y, th, tx, ty, SPEED, DT)
    robot_trail.append((x, y))

    true_dists  = raycast(x, y, th)
    noisy_dists = add_noise(true_dists, SIGMA)
    update_grid(grid, x, y, th, noisy_dists)

gt = build_ground_truth(grid.shape)


rows, cols = grid.shape
trail_arr  = np.array(robot_trail)

gt_img = np.ones((rows, cols, 3))
gt_img[gt] = [0.1, 0.1, 0.1]

est_img = np.ones((rows, cols, 3)) * 0.75

free_mask = grid < -CONF_THRESH
occ_mask  = grid >  CONF_THRESH

est_img[free_mask] = [0.95, 0.95, 0.95]
est_img[occ_mask]  = [0.10, 0.10, 0.10]

tp_mask = occ_mask &  gt
fp_mask = occ_mask & ~gt
fn_mask = ~occ_mask & gt

est_img[tp_mask] = [0.15, 0.55, 0.25]
est_img[fp_mask] = [0.85, 0.25, 0.20]
est_img[fn_mask] = [0.95, 0.65, 0.10]

# -----------------------------
# Metrics
# -----------------------------
tp = np.sum(tp_mask)
fp = np.sum(fp_mask)
fn = np.sum(fn_mask)
precision = tp / (tp + fp + 1e-9)
recall    = tp / (tp + fn + 1e-9)
f1        = 2 * precision * recall / (precision + recall + 1e-9)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Occupancy grid mapping — lawn-mower path, σ = {SIGMA}", fontsize=13)

ax = axes[0]
ax.imshow(gt_img, origin="upper", extent=[0, WIDTH, HEIGHT, 0])
ax.plot(trail_arr[:, 0], trail_arr[:, 1],
        color="#4488cc", linewidth=0.5, alpha=0.4, label="Robot path")
ax.scatter(*wp[0], color="#4488cc", s=50, zorder=5, label="Start")
for x1, y1, x2, y2 in WALLS:
    ax.plot([x1, x2], [y1, y2], color="#222", linewidth=1.2)
ax.add_patch(plt.Rectangle((0, 0), WIDTH, HEIGHT,
             edgecolor="#222", facecolor="none", linewidth=1.5))
ax.set_title("Ground truth")
ax.set_xlim(0, WIDTH)
ax.set_ylim(HEIGHT, 0)
ax.set_xlabel("x (px)")
ax.set_ylabel("y (px)")
ax.legend(fontsize=8, loc="upper right")

ax = axes[1]
ax.imshow(est_img, origin="upper", extent=[0, WIDTH, HEIGHT, 0])
ax.plot(trail_arr[:, 0], trail_arr[:, 1],
        color="#4488cc", linewidth=0.5, alpha=0.4)
ax.scatter(*wp[0], color="#4488cc", s=50, zorder=5)

patches = [
    mpatches.Patch(color=[0.15, 0.55, 0.25], label="True positive"),
    mpatches.Patch(color=[0.85, 0.25, 0.20], label="False positive"),
    mpatches.Patch(color=[0.95, 0.65, 0.10], label="False negative"),
    mpatches.Patch(color=[0.95, 0.95, 0.95], label="Free space"),
    mpatches.Patch(color=[0.75, 0.75, 0.75], label="Unknown"),
]
ax.legend(handles=patches, fontsize=7.5, loc="upper right")
ax.set_title(f"Estimated map  |  P={precision:.2f}  R={recall:.2f}  F1={f1:.2f}")
ax.set_xlim(0, WIDTH)
ax.set_ylim(HEIGHT, 0)
ax.set_xlabel("x (px)")

plt.tight_layout()
plt.savefig(f"map_sigma{SIGMA}.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Waypoints: {len(wp)}  |  Steps: {len(robot_trail)}")
print(f"P={precision:.3f} | R={recall:.3f} | F1={f1:.3f}")
