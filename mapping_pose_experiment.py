import math
import random
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# World / robot constants
# ---------------------------------------------------------------------------
WIDTH, HEIGHT = 1000, 700
CIRCLE_RADIUS = 18
LINEAR_SPEED = 80.0
WHEEL_BASE = 2 * CIRCLE_RADIUS

NUM_SENSORS = 12
SENSOR_ANGLE_STEP = 2 * math.pi / NUM_SENSORS
MAX_SENSOR_DISTANCE = 200

MAP_CELL_SIZE = 10
GRID_COLS = math.ceil(WIDTH / MAP_CELL_SIZE)
GRID_ROWS = math.ceil(HEIGHT / MAP_CELL_SIZE)

P_PRIOR = 0.5
P_OCC = 0.75
P_FREE = 0.35
L_PRIOR = math.log(P_PRIOR / (1.0 - P_PRIOR))
L_OCC = math.log(P_OCC / (1.0 - P_OCC))
L_FREE = math.log(P_FREE / (1.0 - P_FREE))
LOG_ODDS_MIN = -5.0
LOG_ODDS_MAX = 5.0

MOTION_NOISE_STD = 2.0
SENSOR_NOISE_STD = 5.0
BEARING_NOISE_STD = 0.05
KIDNAP_THRESHOLD = 150

LANDMARKS = [
    (150, 200),
    (500, 300),
    (750, 500),
    (300, 600),
    (850, 150),
]
LANDMARKS_RANGE = 200

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

ALL_WALL_SEGMENTS = list(INTERNAL_WALLS) + [
    (0, 0, WIDTH, 0),
    (WIDTH, 0, WIDTH, HEIGHT),
    (WIDTH, HEIGHT, 0, HEIGHT),
    (0, HEIGHT, 0, 0),
]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def line_segment_intersection(ax, ay, bx, by, cx, cy, dx, dy):
    denom = (ax - bx) * (cy - dy) - (ay - by) * (cx - dx)
    if abs(denom) < 1e-10:
        return None
    t = ((ax - cx) * (cy - dy) - (ay - cy) * (cx - dx)) / denom
    u = -((ax - bx) * (ay - cy) - (ay - by) * (ax - cx)) / denom
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return ax + t * (bx - ax), ay + t * (by - ay)
    return None


def point_to_segment_distance(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-10:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def collides(cx, cy):
    for seg in ALL_WALL_SEGMENTS:
        if point_to_segment_distance(cx, cy, *seg) < CIRCLE_RADIUS:
            return True
    return False


# ---------------------------------------------------------------------------
# Sensor model
# ---------------------------------------------------------------------------
def get_sensor_readings(cx, cy, theta):
    distances = []
    for i in range(NUM_SENSORS):
        angle = theta + i * SENSOR_ANGLE_STEP
        sx = cx + math.cos(angle) * CIRCLE_RADIUS
        sy = cy + math.sin(angle) * CIRCLE_RADIUS
        ex = cx + math.cos(angle) * MAX_SENSOR_DISTANCE
        ey = cy + math.sin(angle) * MAX_SENSOR_DISTANCE
        closest_dist = MAX_SENSOR_DISTANCE - CIRCLE_RADIUS

        for seg in ALL_WALL_SEGMENTS:
            pt = line_segment_intersection(sx, sy, ex, ey, *seg)
            if pt is not None:
                d = math.hypot(pt[0] - sx, pt[1] - sy)
                if d < closest_dist:
                    closest_dist = d

        distances.append(closest_dist)
    return distances


# ---------------------------------------------------------------------------
# Motion model
# ---------------------------------------------------------------------------
def update_pose(x, y, theta, v, omega, dt):
    if abs(omega) < 1e-6:
        return x + v * math.cos(theta) * dt, y + v * math.sin(theta) * dt, theta
    r = v / omega
    x_new = x - r * math.sin(theta) + r * math.sin(theta + omega * dt)
    y_new = y + r * math.cos(theta) - r * math.cos(theta + omega * dt)
    return x_new, y_new, (theta + omega * dt) % (2 * math.pi)


# ---------------------------------------------------------------------------
# Landmark observations with noise
# ---------------------------------------------------------------------------
def get_landmark_observations(cx, cy, theta):
    observations = []
    for i, (lx, ly) in enumerate(LANDMARKS):
        dist = math.hypot(lx - cx, ly - cy)
        if dist <= LANDMARKS_RANGE:
            bearing = math.atan2(ly - cy, lx - cx) - theta
            noisy_dist = dist + random.gauss(0, SENSOR_NOISE_STD)
            noisy_bearing = bearing + random.gauss(0, BEARING_NOISE_STD)
            observations.append((i, noisy_dist, noisy_bearing))
    return observations


# ---------------------------------------------------------------------------
# EKF
# ---------------------------------------------------------------------------
def kalman_predict(mu, sigma, v, omega, dt):
    x, y, theta = mu

    if abs(omega) < 1e-6:
        x_new = x + v * math.cos(theta) * dt
        y_new = y + v * math.sin(theta) * dt
        theta_new = theta
        A = np.array([
            [1, 0, -v * math.sin(theta) * dt],
            [0, 1,  v * math.cos(theta) * dt],
            [0, 0, 1],
        ])
    else:
        r = v / omega
        x_new = x - r * math.sin(theta) + r * math.sin(theta + omega * dt)
        y_new = y + r * math.cos(theta) - r * math.cos(theta + omega * dt)
        theta_new = theta + omega * dt
        A = np.array([
            [1, 0, -r * math.cos(theta) + r * math.cos(theta + omega * dt)],
            [0, 1, -r * math.sin(theta) + r * math.sin(theta + omega * dt)],
            [0, 0, 1],
        ])

    mu_bar = np.array([x_new, y_new, theta_new % (2 * math.pi)])
    R = np.diag([MOTION_NOISE_STD**2, MOTION_NOISE_STD**2, 0.01])
    sigma_bar = A @ sigma @ A.T + R
    return mu_bar, sigma_bar


def kalman_update(mu_bar, sigma_bar, observations):
    mu = mu_bar.copy()
    sigma = sigma_bar.copy()

    for (lm_idx, z_dist, z_bearing) in observations:
        lx, ly = LANDMARKS[lm_idx]
        x, y, theta = mu

        expected_dist = math.hypot(lx - x, ly - y)
        expected_bearing = math.atan2(ly - y, lx - x) - theta

        z = np.array([z_dist, z_bearing])
        z_hat = np.array([expected_dist, expected_bearing])

        C = np.array([
            [-(lx - x) / expected_dist, -(ly - y) / expected_dist, 0],
            [(ly - y) / expected_dist**2, -(lx - x) / expected_dist**2, -1],
        ])

        Q = np.diag([SENSOR_NOISE_STD**2, BEARING_NOISE_STD**2])
        K = sigma @ C.T @ np.linalg.inv(C @ sigma @ C.T + Q)

        innovation = z - z_hat
        innovation[1] = (innovation[1] + math.pi) % (2 * math.pi) - math.pi

        if abs(innovation[0]) > KIDNAP_THRESHOLD:
            sigma = np.eye(3) * 50000.0
            mu[0] = lx - z_dist * math.cos(z_bearing + mu[2])
            mu[1] = ly - z_dist * math.sin(z_bearing + mu[2])
            continue

        mu = mu + K @ innovation
        sigma = (np.eye(3) - K @ C) @ sigma

    return mu, sigma


# ---------------------------------------------------------------------------
# Occupancy grid
# ---------------------------------------------------------------------------
def create_occupancy_grid():
    return np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)


def world_to_grid(x, y):
    return int(y // MAP_CELL_SIZE), int(x // MAP_CELL_SIZE)


def in_grid(row, col):
    return 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS


def cells_along_line(x0, y0, x1, y1):
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)

    if length < 1e-9:
        r, c = world_to_grid(x0, y0)
        return [(r, c)] if in_grid(r, c) else []

    step = max(1.0, MAP_CELL_SIZE / 2.0)
    steps = max(1, int(length / step))
    cells = []
    last = None

    for s in range(steps + 1):
        t = s / steps
        r, c = world_to_grid(x0 + t * dx, y0 + t * dy)
        if in_grid(r, c) and (r, c) != last:
            cells.append((r, c))
            last = (r, c)

    return cells


def update_occupancy_grid(log_odds, pose, distances):
    rx, ry, rtheta = pose
    max_r = MAX_SENSOR_DISTANCE - CIRCLE_RADIUS

    rr, rc = world_to_grid(rx, ry)
    if in_grid(rr, rc):
        log_odds[rr, rc] += L_FREE

    for i, dist in enumerate(distances):
        angle = rtheta + i * SENSOR_ANGLE_STEP
        sx = rx + math.cos(angle) * CIRCLE_RADIUS
        sy = ry + math.sin(angle) * CIRCLE_RADIUS

        meas = max(0.0, min(float(dist), max_r))
        hit = meas < (max_r - 1.0)

        ex = sx + math.cos(angle) * meas
        ey = sy + math.sin(angle) * meas
        ray = cells_along_line(sx, sy, ex, ey)

        if not ray:
            continue

        for r, c in set(ray[:-1] if hit else ray):
            log_odds[r, c] += L_FREE - L_PRIOR

        if hit:
            er, ec = ray[-1]
            if in_grid(er, ec):
                log_odds[er, ec] += L_OCC - L_PRIOR

    np.clip(log_odds, LOG_ODDS_MIN, LOG_ODDS_MAX, out=log_odds)


# ---------------------------------------------------------------------------
# Waypoint navigation
# ---------------------------------------------------------------------------
def generate_waypoints():
    return [
        (100, 400), (100, 200), (300, 200), (490, 200),
        (490, 300), (300, 300), (100, 300), (100, 550),
        (300, 550), (490, 550), (490, 650), (300, 650),
        (100, 650), (100, 400),
        (700, 150), (700, 300), (700, 500), (900, 500),
        (900, 300), (900, 150), (700, 150),
    ]


def steer_to(rx, ry, rtheta, tx, ty, dt, speed=LINEAR_SPEED):
    desired = math.atan2(ty - ry, tx - rx)
    err = (desired - rtheta + math.pi) % (2 * math.pi) - math.pi
    omega = max(-4.0, min(4.0, 3.0 * err))
    v = speed * max(0.1, 1.0 - abs(err))
    nx, ny, nth = update_pose(rx, ry, rtheta, v, omega, dt)

    if collides(nx, ny):
        nx, ny = rx, ry

    return nx, ny, nth, v, omega


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------
SIM_DURATION = 60.0
DT = 1 / 30.0
SEED = 42


def run_simulation():
    random.seed(SEED)
    np.random.seed(SEED)

    true_log_odds = create_occupancy_grid()
    ekf_log_odds = create_occupancy_grid()

    rx, ry, rtheta = 100.0, 400.0, 0.0

    kf_mu = np.array([rx, ry, rtheta])
    kf_sigma = np.eye(3) * 100.0

    waypoints = generate_waypoints()
    wp_idx = 0
    t = 0.0

    while t < SIM_DURATION:
        tx, ty = waypoints[wp_idx]
        if math.hypot(tx - rx, ty - ry) < 20:
            wp_idx = (wp_idx + 1) % len(waypoints)
            tx, ty = waypoints[wp_idx]

        rx, ry, rtheta, v, omega = steer_to(rx, ry, rtheta, tx, ty, DT)

        kf_mu, kf_sigma = kalman_predict(kf_mu, kf_sigma, v, omega, DT)

        obs = get_landmark_observations(rx, ry, rtheta)
        if obs:
            kf_mu, kf_sigma = kalman_update(kf_mu, kf_sigma, obs)

        distances = get_sensor_readings(rx, ry, rtheta)

        update_occupancy_grid(true_log_odds, [rx, ry, rtheta], distances)
        update_occupancy_grid(ekf_log_odds, kf_mu, distances)

        t += DT

    return true_log_odds, ekf_log_odds


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def logodds_to_probability(log_odds):
    return 1.0 / (1.0 + np.exp(-log_odds))


def make_plots(true_log_odds, ekf_log_odds):
    true_prob = logodds_to_probability(true_log_odds)
    ekf_prob = logodds_to_probability(ekf_log_odds)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Occupancy Grid Maps: True Pose vs EKF Pose", fontsize=14)

    axes[0].imshow(true_prob, cmap="gray", origin="upper", extent=[0, WIDTH, HEIGHT, 0])
    axes[0].set_title("Map using ground-truth pose")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    axes[1].imshow(ekf_prob, cmap="gray", origin="upper", extent=[0, WIDTH, HEIGHT, 0])
    axes[1].set_title("Map using EKF-estimated pose")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    plt.tight_layout()
    plt.savefig("pose_maps.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    print("Running simulation...")
    true_lo, ekf_lo = run_simulation()
    make_plots(true_lo, ekf_lo)