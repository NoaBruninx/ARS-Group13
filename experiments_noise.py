import math, random
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# WORLD SETTINGS (same as Pygame)
# -------------------------
WIDTH, HEIGHT = 1000, 700
LINEAR_SPEED = 80.0

LANDMARKS = [(150,200),(500,300),(750,500),(300,600),(850,150)]
LANDMARKS_RANGE = 200

MAX_STEPS = 6000
DT = 1/30
N_SEEDS = 400


BASE_MOTION_NOISE = 2.0
BASE_SENSOR_NOISE = 5.0
BASE_BEARING_NOISE = 0.05



def update_pose(x, y, theta, v, omega, dt):
    if abs(omega) < 1e-6:
        return (
            x + v * math.cos(theta) * dt,
            y + v * math.sin(theta) * dt,
            theta
        )

    r = v / omega
    return (
        x - r * math.sin(theta) + r * math.sin(theta + omega * dt),
        y + r * math.cos(theta) - r * math.cos(theta + omega * dt),
        (theta + omega * dt) % (2 * math.pi)
    )



def get_landmark_observations(cx, cy, theta, sensor_std, bearing_std):
    obs = []
    for i, (lx, ly) in enumerate(LANDMARKS):
        dist = math.hypot(lx - cx, ly - cy)
        if dist <= LANDMARKS_RANGE:
            bearing = math.atan2(ly - cy, lx - cx) - theta

            obs.append((
                i,
                dist + random.gauss(0, sensor_std),
                bearing + random.gauss(0, bearing_std)
            ))
    return obs



def kalman_predict(mu, sigma, v, omega, dt, motion_std):
    x, y, theta = mu

    if abs(omega) < 1e-6:
        xn = x + v * math.cos(theta) * dt
        yn = y + v * math.sin(theta) * dt
        thn = theta

        A = np.array([
            [1, 0, -v * math.sin(theta) * dt],
            [0, 1,  v * math.cos(theta) * dt],
            [0, 0, 1]
        ])
    else:
        r = v / omega
        xn = x - r * math.sin(theta) + r * math.sin(theta + omega * dt)
        yn = y + r * math.cos(theta) - r * math.cos(theta + omega * dt)
        thn = theta + omega * dt

        A = np.array([
            [1, 0, -r * math.cos(theta) + r * math.cos(theta + omega * dt)],
            [0, 1, -r * math.sin(theta) + r * math.sin(theta + omega * dt)],
            [0, 0, 1]
        ])

    R = np.diag([motion_std**2, motion_std**2, 0.01])
    sigma = A @ sigma @ A.T + R

    return np.array([xn, yn, thn]), sigma



def kalman_update(mu, sigma, obs, sensor_std, bearing_std):
    for lm_idx, z_dist, z_bearing in obs:

        lx, ly = LANDMARKS[lm_idx]
        x, y, theta = mu

        dx = lx - x
        dy = ly - y
        q = dx**2 + dy**2

        if q < 1e-6:
            continue

        exp_dist = math.sqrt(q)
        if exp_dist < 1e-6:
            continue

        exp_bearing = math.atan2(dy, dx) - theta

        z = np.array([z_dist, z_bearing])
        z_hat = np.array([exp_dist, exp_bearing])

        C = np.array([
            [-dx / exp_dist, -dy / exp_dist, 0],
            [ dy / q,       -dx / q,        -1]
        ])

        Q = np.diag([sensor_std**2, bearing_std**2])

        S = C @ sigma @ C.T + Q
        S += np.eye(2) * 1e-6

        try:
            K = sigma @ C.T @ np.linalg.solve(S, np.eye(2))
        except np.linalg.LinAlgError:
            continue

        innovation = z - z_hat
        innovation[1] = (innovation[1] + math.pi) % (2 * math.pi) - math.pi

        mu = mu + K @ innovation
        sigma = (np.eye(3) - K @ C) @ sigma

    return mu, sigma



def run_trial(noise, seed=0, randomize=False):

    random.seed(seed)
    np.random.seed(seed)

    NOISE_FACTOR = noise

    motion_std  = BASE_MOTION_NOISE * NOISE_FACTOR
    sensor_std  = BASE_SENSOR_NOISE * NOISE_FACTOR
    bearing_std = BASE_BEARING_NOISE * NOISE_FACTOR

    # TRUE STATE
    if randomize:
        rx = random.uniform(100, WIDTH-100)
        ry = random.uniform(100, HEIGHT-100)
        rth = random.uniform(0, 2*math.pi)
    else:
        rx, ry, rth = 100.0, 400.0, 0.0


    mu = np.array([rx, ry, rth])
    sigma = np.eye(3) * 1e-6

    v = LINEAR_SPEED * (0.5 + 0.1 * random.random())
    omega = 0.3 + 0.1 * random.random()

    errors = []

    for _ in range(MAX_STEPS):

        rx, ry, rth = update_pose(rx, ry, rth, v, omega, DT)

        rx  += random.gauss(0, motion_std)
        ry  += random.gauss(0, motion_std)
        rth += random.gauss(0, BASE_BEARING_NOISE * NOISE_FACTOR)

        rth = (rth + math.pi) % (2 * math.pi) - math.pi

        mu, sigma = kalman_predict(mu, sigma, v, omega, DT, motion_std)

        obs = get_landmark_observations(rx, ry, rth, sensor_std, bearing_std)
        if obs:
            mu, sigma = kalman_update(mu, sigma, obs, sensor_std, bearing_std)

        errors.append(math.hypot(mu[0] - rx, mu[1] - ry))

    return np.array(errors)


# EXPERIMENT
NOISE_LEVELS = [0, 0.2, 0.5, 1, 2, 4]

def run_experiment(randomize):
    curves = {}
    auc_scores = {}

    for n in NOISE_LEVELS:
        runs = []
        aucs = []

        for s in range(N_SEEDS):
            e = run_trial(n, seed=s, randomize=randomize)
            runs.append(e)
            aucs.append(np.trapz(e))

        curves[n] = np.mean(runs, axis=0)
        auc_scores[n] = np.mean(aucs)

    return curves, auc_scores


# RUN
fixed_curves, fixed_auc = run_experiment(False)
rand_curves, rand_auc = run_experiment(True)

t = np.arange(MAX_STEPS) * DT
# PRINT AUC RESULTS
print("\n=== FIXED START AUC ===")
for n in NOISE_LEVELS:
    print(f"noise={n:>4}: AUC = {fixed_auc[n]:.4f}")

print("\n=== RANDOM START AUC ===")
for n in NOISE_LEVELS:
    print(f"noise={n:>4}: AUC = {rand_auc[n]:.4f}")



# PLOTS
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
for n in NOISE_LEVELS:
    plt.plot(t, fixed_curves[n], label=f"noise={n}")
plt.title("Fixed start")
plt.xlabel("Time (s)")
plt.ylabel("Position error")
plt.legend()
plt.grid()

plt.subplot(1,2,2)
plt.bar([str(n) for n in NOISE_LEVELS],
        [fixed_auc[n] for n in NOISE_LEVELS])
plt.title("Fixed AUC")
plt.xlabel("Noise level")
plt.ylabel("Total error (AUC)")
plt.grid()

plt.tight_layout()
plt.show()


plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
for n in NOISE_LEVELS:
    plt.plot(t, rand_curves[n], label=f"noise={n}")
plt.title("Random start")
plt.xlabel("Time (s)")
plt.ylabel("Position error")
plt.legend()
plt.grid()

plt.subplot(1,2,2)
plt.bar([str(n) for n in NOISE_LEVELS],
        [rand_auc[n] for n in NOISE_LEVELS])
plt.title("Random AUC")
plt.xlabel("Noise level")
plt.ylabel("Total error (AUC)")
plt.grid()

plt.tight_layout()
plt.show()