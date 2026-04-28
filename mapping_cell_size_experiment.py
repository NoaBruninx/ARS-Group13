import math
import random
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

#taken from Mapping.py 
WIDTH, HEIGHT = 1000, 700
CIRCLE_RADIUS  = 18
LINEAR_SPEED   = 80.0
NUM_SENSORS    = 12
SENSOR_ANGLE_STEP = 2 * math.pi / NUM_SENSORS
MAX_SENSOR_DISTANCE = 200

P_PRIOR = 0.5
P_OCC   = 0.75
P_FREE  = 0.35
L_PRIOR = math.log(P_PRIOR / (1.0 - P_PRIOR))
L_OCC   = math.log(P_OCC   / (1.0 - P_OCC))
L_FREE  = math.log(P_FREE  / (1.0 - P_FREE))
LOG_ODDS_MIN = -5.0
LOG_ODDS_MAX  =  5.0

INTERNAL_WALLS = [
    #   Left room vertical wall              
    (200, 100, 200, 600),
    #   Left room horizontal opening (gap 200-280)     
    (200, 450, 340, 450),
    #   Right side of left room               
    (400, 100, 400, 350),
    #   Top of left room                  
    (200, 100, 600, 100),
    #   Middle vertical divider               
    (600, 100, 600, 400),
    #   Middle horizontal shelf               
    (600, 400, 800, 400),
    #   Right vertical wall                 
    (800, 200, 800, 700),
    #   Right room horizontal opening           
    (800, 450, 1000, 450),
    #   Bottom corridor                   
    (200, 700, 550, 700),
    (650, 700, 800, 700),
    # #   Small inner box (obstacle)             
    # (450, 550, 550, 550),
    # (450, 550, 450, 650),
    # (450, 650, 550, 650),
    # (550, 550, 550, 650),
]

#internal + outer walls 
ALL_WALL_SEGMENTS = (
    list(INTERNAL_WALLS)
    + [(0, 0, WIDTH, 0), (WIDTH, 0, WIDTH, HEIGHT),
       (WIDTH, HEIGHT, 0, HEIGHT), (0, HEIGHT, 0, 0)]
)

# Ground-truth occupied pixels: pre-rasterise walls at 1-pixel resolution
# so we can compare the grid to reality later.
def rasterise_walls(resolution=2):
    """Return a set of (row, col) pixel-level occupied positions."""
    occupied = set()
    for (x1, y1, x2, y2) in ALL_WALL_SEGMENTS:
        length = math.hypot(x2 - x1, y2 - y1)
        steps = max(1, int(length / resolution))
        for s in range(steps + 1):
            t = s / steps
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            occupied.add((int(y), int(x)))
    return occupied

WALL_PIXELS = rasterise_walls()

# Pure-Python geometry helpers without pygame, for the simulation and mapping logic
def line_segment_intersection(ax, ay, bx, by, cx, cy, dx, dy):

    denom = (ax - bx) * (cy - dy) - (ay - by) * (cx - dx)
    if abs(denom) < 1e-10:
        return None   # parallel / coincident

    t = ((ax - cx) * (cy - dy) - (ay - cy) * (cx - dx)) / denom
    u = -((ax - bx) * (ay - cy) - (ay - by) * (ax - cx)) / denom

    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        ix = ax + t * (bx - ax)
        iy = ay + t * (by - ay)
        return ix, iy
    return None


def get_sensor_readings(cx, cy, theta):
    
    
    wall_segs = ALL_WALL_SEGMENTS
    distances  = []
    hit_points = []

    for i in range(NUM_SENSORS):
        angle = theta + i * SENSOR_ANGLE_STEP
        # Sensor start point (at robot edge)
        sx = cx + math.cos(angle) * CIRCLE_RADIUS
        sy = cy + math.sin(angle) * CIRCLE_RADIUS
        
        # Ray endpoint (far end)
        ex = cx + math.cos(angle) * MAX_SENSOR_DISTANCE
        ey = cy + math.sin(angle) * MAX_SENSOR_DISTANCE

        closest_dist = MAX_SENSOR_DISTANCE - CIRCLE_RADIUS
        closest_pt   = (ex, ey)

        for (wx1, wy1, wx2, wy2) in wall_segs:
            pt = line_segment_intersection(sx, sy, ex, ey,
                                           wx1, wy1, wx2, wy2)
            if pt is not None:
                d = math.hypot(pt[0] - sx, pt[1] - sy)
                if d < closest_dist:
                    closest_dist = d
                    closest_pt   = pt

        distances.append(closest_dist)
        hit_points.append(closest_pt)

    return distances, hit_points


#easier version just to simulate the robot's movement and collision checking, collision yes or no 
def point_to_segment_distance(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-10:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))

def collides(cx, cy):
    for (ax, ay, bx, by) in ALL_WALL_SEGMENTS:
        if point_to_segment_distance(cx, cy, ax, ay, bx, by) < CIRCLE_RADIUS:
            return True
    return False

def update_pose(x, y, theta, v, omega, dt):
    if abs(omega) < 1e-6:
        return x + v * math.cos(theta) * dt, y + v * math.sin(theta) * dt, theta
    r = v / omega
    x_new = x - r * math.sin(theta) + r * math.sin(theta + omega * dt)
    y_new = y + r * math.cos(theta) - r * math.cos(theta + omega * dt)
    return x_new, y_new, (theta + omega * dt) % (2 * math.pi)

# Occupancy grid helpers, parameterised by cell_size
def make_grid(cell_size):
    rows = math.ceil(HEIGHT / cell_size)
    cols = math.ceil(WIDTH  / cell_size)
    return np.zeros((rows, cols), dtype=np.float32), rows, cols

def world_to_grid(x, y, cell_size):
    return int(y // cell_size), int(x // cell_size)

def in_grid(row, col, rows, cols):
    return 0 <= row < rows and 0 <= col < cols

def cells_along_line(x0, y0, x1, y1, cell_size, rows, cols):
    """
    Return grid cells touched by a line segment from (x0, y0) to (x1, y1) -> sensor ray 
    """
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length < 1e-9:
        r, c = world_to_grid(x0, y0, cell_size)
        return [(r, c)] if in_grid(r, c, rows, cols) else []
    step = max(1.0, cell_size / 2.0)
    steps = max(1, int(length / step))
    cells, last = [], None
    for s in range(steps + 1):
        t = s / steps
        r, c = world_to_grid(x0 + t * dx, y0 + t * dy, cell_size)
        if in_grid(r, c, rows, cols) and (r, c) != last:
            cells.append((r, c))
            last = (r, c)
    return cells

def update_grid(log_odds, pose, distances, cell_size, rows, cols):
    """
    use cells from cells_along_line to update the occupancy grid
    """
    rx, ry, rtheta = pose
    max_r = MAX_SENSOR_DISTANCE - CIRCLE_RADIUS
    rr, rc = world_to_grid(rx, ry, cell_size)
    if in_grid(rr, rc, rows, cols):
        log_odds[rr, rc] += L_FREE
    for i, dist in enumerate(distances):
        angle = rtheta + i * SENSOR_ANGLE_STEP
        sx = rx + math.cos(angle) * CIRCLE_RADIUS
        sy = ry + math.sin(angle) * CIRCLE_RADIUS
        meas = max(0.0, min(float(dist), max_r))
        hit  = meas < (max_r - 1.0)
        ex = sx + math.cos(angle) * meas
        ey = sy + math.sin(angle) * meas
        ray = cells_along_line(sx, sy, ex, ey, cell_size, rows, cols)
        if not ray:
            continue
        for r, c in set(ray[:-1] if hit else ray):
            log_odds[r, c] += L_FREE - L_PRIOR
        if hit:
            er, ec = ray[-1]
            if in_grid(er, ec, rows, cols):
                log_odds[er, ec] += L_OCC - L_PRIOR
    np.clip(log_odds, LOG_ODDS_MIN, LOG_ODDS_MAX, out=log_odds)

# Metrics
CONF_THRESH = 1.0  # log-odds threshold to call a cell "confident"

def compute_metrics(log_odds, cell_size, rows, cols):
    total = rows * cols
    occ_mask  = log_odds >  CONF_THRESH #confidently occupied
    free_mask = log_odds < -CONF_THRESH #confidently free
    unk_mask  = ~occ_mask & ~free_mask #uncertain

    frac_occ  = occ_mask.sum()  / total
    frac_free = free_mask.sum() / total
    frac_unk  = unk_mask.sum()  / total

    #entropy per cell (using p_occ)
    p = 1.0 - 1.0 / (1.0 + np.exp(log_odds)) #p(occupied) from log-odds
    eps = 1e-9
    entropy = -p * np.log2(p + eps) - (1 - p) * np.log2(1 - p + eps) #shannon entropy formula
    mean_entropy = entropy.mean()

    # Average absolute log-odds (overall map confidence)
    mean_abs_lo = np.abs(log_odds).mean()

    # Memory in bytes
    mem_bytes = log_odds.nbytes

    # Ground-truth comparison: for every wall pixel, check the grid cell it belongs to
    # TP = occupied cell that overlaps a wall pixel
    # We compare at the grid resolution
    wall_cells = set()
    for (wy, wx) in WALL_PIXELS:
        r = int(wy // cell_size)
        c = int(wx // cell_size)
        if in_grid(r, c, rows, cols):
            wall_cells.add((r, c))

    tp = sum(1 for (r, c) in wall_cells if log_odds[r, c] > CONF_THRESH)
    precision = tp / max(1, occ_mask.sum())
    recall    = tp / max(1, len(wall_cells))
    f1 = 2 * precision * recall / max(1e-9, precision + recall)

    return {
        "frac_occ":    frac_occ,
        "frac_free":   frac_free,
        "frac_unk":    frac_unk,
        "mean_entropy": mean_entropy,
        "mean_abs_lo": mean_abs_lo,
        "mem_bytes":   mem_bytes,
        "precision":   precision,
        "recall":      recall,
        "f1":          f1,
    }

# We work here with a scripted robot path for this experiment (a figure-8-ish tour that covers the whole maze)
def generate_waypoints():
    """Waypoints the robot will drive toward, covering the whole map."""
    return [
        (100, 400), (100, 200), (300, 200), (490, 200),
        (490, 300), (300, 300), (100, 300), (100, 550),
        (300, 550), (490, 550), (490, 650), (300, 650),
        (100, 650), (100, 400),
        # right side
        (700, 150), (700, 300), (700, 500), (900, 500),
        (900, 300), (900, 150), (700, 150),
    ]


def steer_to(rx, ry, rtheta, tx, ty, dt, speed=LINEAR_SPEED):
    """Simple proportional controller to steer toward a target point."""
    desired = math.atan2(ty - ry, tx - rx)
    err = (desired - rtheta + math.pi) % (2 * math.pi) - math.pi
    omega = 3.0 * err           # proportional heading control
    omega = max(-4.0, min(4.0, omega))
    v = speed * max(0.1, 1.0 - abs(err))
    nx, ny, nth = update_pose(rx, ry, rtheta, v, omega, dt)
    if collides(nx, ny):
        nx, ny = rx, ry
    return nx, ny, nth


CELL_SIZES = [5, 10, 20, 40]          # pixels, the variable we sweep
SIM_DURATION = 60.0                   # seconds of simulated time per run
DT = 1 / 30.0                         # simulation timestep
METRIC_INTERVAL = 1.0                 # record metrics every N seconds

COLORS = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a"]
LABELS = [f"{cs}px cell" for cs in CELL_SIZES]


def run_experiment(cell_size, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    log_odds, rows, cols = make_grid(cell_size)
    rx, ry, rtheta = 100.0, 400.0, 0.0
    t = 0.0
    waypoints = generate_waypoints()
    wp_idx = 0

    metric_times = []
    metric_history = defaultdict(list)

    next_metric_t = 0.0
    start_wall = time.time()

    while t < SIM_DURATION:
        # Navigate toward next waypoint
        tx, ty = waypoints[wp_idx]
        dist_to_wp = math.hypot(tx - rx, ty - ry)
        if dist_to_wp < 20:
            wp_idx = (wp_idx + 1) % len(waypoints)

        rx, ry, rtheta = steer_to(rx, ry, rtheta, tx, ty, DT)

        # Sensor readings (ground truth, no noise for mapping clarity)
        distances, _ = get_sensor_readings(rx, ry, rtheta)

        # Update occupancy grid
        update_grid(log_odds, [rx, ry, rtheta], distances, cell_size, rows, cols)

        # Record metrics at intervals
        if t >= next_metric_t:
            m = compute_metrics(log_odds, cell_size, rows, cols)
            metric_times.append(t)
            for k, v in m.items():
                metric_history[k].append(v)
            next_metric_t += METRIC_INTERVAL

        t += DT

    wall_time = time.time() - start_wall
    final_metrics = compute_metrics(log_odds, cell_size, rows, cols)
    final_metrics["wall_time_s"] = wall_time
    final_metrics["total_cells"] = rows * cols

    return metric_times, dict(metric_history), final_metrics, log_odds, rows, cols


# ---------------------------------------------------------------------------
# Plotting -> this is completely generated by genAI 
# ---------------------------------------------------------------------------

def render_grid_image(log_odds, rows, cols):
    """Convert log-odds grid to a greyscale RGB image array."""
    p = 1.0 - 1.0 / (1.0 + np.exp(log_odds))
    img = np.zeros((rows, cols, 3), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            po = p[r, c]
            if po > 0.55:
                shade = max(0, int(255 * (1 - po)))
            elif po < 0.45:
                shade = 245
            else:
                shade = 210
            img[r, c] = (shade, shade, shade)
    return img


def make_plots(results):
    """
    results: list of (cell_size, times, history, final, log_odds, rows, cols)
    """
    fig = plt.figure(figsize=(20, 22), facecolor="#0d1117")

    gs = gridspec.GridSpec(
        5, len(results),
        figure=fig,
        hspace=0.55, wspace=0.35,
        top=0.94, bottom=0.04,
        left=0.07, right=0.97
    )

    ax_col = {
        "entropy":   0,
        "conf":      1,
        "f1":        2,
        "mem":       3,
    }

    # ---- Row 0: final rendered grids ----
    for j, (cs, times, hist, final, log_odds, rows, cols) in enumerate(results):
        ax = fig.add_subplot(gs[0, j])
        img = render_grid_image(log_odds, rows, cols)
        ax.imshow(img, aspect="auto", origin="upper",
                  extent=[0, WIDTH, HEIGHT, 0])
        ax.set_title(f"cell = {cs}px\n{rows}×{cols} = {rows*cols:,} cells",
                     color="white", fontsize=10)
        ax.axis("off")

    # ---- Row 1: mean entropy over time ----
    ax_ent = fig.add_subplot(gs[1, :])
    ax_ent.set_facecolor("#161b22")
    for j, (cs, times, hist, final, *_) in enumerate(results):
        ax_ent.plot(times, hist["mean_entropy"], color=COLORS[j],
                    linewidth=2, label=LABELS[j])
    ax_ent.set_title("Mean Cell Entropy Over Time  (lower = more confident map)",
                     color="white")
    ax_ent.set_xlabel("Simulated time (s)", color="#aaa")
    ax_ent.set_ylabel("Mean entropy (bits)", color="#aaa")
    ax_ent.legend(facecolor="#0d1117", labelcolor="white", fontsize=9)
    ax_ent.tick_params(colors="#aaa")
    for sp in ax_ent.spines.values():
        sp.set_color("#333")

    # ---- Row 2: fraction confident over time ----
    ax_conf = fig.add_subplot(gs[2, :])
    ax_conf.set_facecolor("#161b22")
    for j, (cs, times, hist, final, *_) in enumerate(results):
        frac_conf = [o + f for o, f in zip(hist["frac_occ"], hist["frac_free"])]
        ax_conf.plot(times, frac_conf, color=COLORS[j],
                     linewidth=2, label=LABELS[j])
    ax_conf.set_title("Fraction of Cells with High Confidence  (free + occupied > threshold)",
                      color="white")
    ax_conf.set_xlabel("Simulated time (s)", color="#aaa")
    ax_conf.set_ylabel("Confident cell fraction", color="#aaa")
    ax_conf.legend(facecolor="#0d1117", labelcolor="white", fontsize=9)
    ax_conf.tick_params(colors="#aaa")
    for sp in ax_conf.spines.values():
        sp.set_color("#333")

    # ---- Row 3: F1 over time ----
    ax_f1 = fig.add_subplot(gs[3, :])
    ax_f1.set_facecolor("#161b22")
    for j, (cs, times, hist, final, *_) in enumerate(results):
        ax_f1.plot(times, hist["f1"], color=COLORS[j],
                   linewidth=2, label=LABELS[j])
    ax_f1.set_title("Wall Detection F1-Score Over Time  (vs. ground-truth wall pixels)",
                    color="white")
    ax_f1.set_xlabel("Simulated time (s)", color="#aaa")
    ax_f1.set_ylabel("F1 score", color="#aaa")
    ax_f1.set_ylim(0, 1)
    ax_f1.legend(facecolor="#0d1117", labelcolor="white", fontsize=9)
    ax_f1.tick_params(colors="#aaa")
    for sp in ax_f1.spines.values():
        sp.set_color("#333")

    # ---- Row 4: bar charts of final metrics ----
    metric_keys = ["mean_entropy", "f1", "mem_bytes", "wall_time_s"]
    titles = [
        "Final Mean Entropy\n(lower = more certain)",
        "Final Wall F1-Score\n(higher = more accurate)",
        "Grid Memory Usage\n(bytes)",
        "Computation Time\n(seconds for full run)",
    ]
    for k_idx, (key, title) in enumerate(zip(metric_keys, titles)):
        ax = fig.add_subplot(gs[4, k_idx])
        ax.set_facecolor("#161b22")
        vals = [r[3][key] for r in results]
        bars = ax.bar(LABELS, vals, color=COLORS, edgecolor="#333", linewidth=0.5)
        ax.set_title(title, color="white", fontsize=9)
        ax.tick_params(colors="#aaa", labelsize=7)
        for sp in ax.spines.values():
            sp.set_color("#333")
        # value labels on bars
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f"{v:.3g}", ha="center", va="bottom",
                    color="white", fontsize=7)

    out = "map_cell_size_experiment.png"
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    results = []
    for cs in CELL_SIZES:
        print(f"Running cell_size={cs}px ...", end=" ", flush=True)
        times, hist, final, log_odds, rows, cols = run_experiment(cs)
        print(f"done  (F1={final['f1']:.3f}, entropy={final['mean_entropy']:.3f},"
              f" mem={final['mem_bytes']/1024:.1f}KB, t={final['wall_time_s']:.1f}s)")
        results.append((cs, times, hist, final, log_odds, rows, cols))

    print("\nFinal summary:")
    print(f"{'cell_size':>12} {'rows×cols':>14} {'F1':>7} {'entropy':>10} "
          f"{'precision':>10} {'recall':>8} {'mem(KB)':>9} {'time(s)':>8}")
    for cs, times, hist, final, lo, rows, cols in results:
        print(f"{cs:>10}px {rows:4}×{cols:<6} "
              f"{final['f1']:>7.3f} {final['mean_entropy']:>10.4f} "
              f"{final['precision']:>10.3f} {final['recall']:>8.3f} "
              f"{final['mem_bytes']/1024:>9.1f} {final['wall_time_s']:>8.2f}")

    make_plots(results)