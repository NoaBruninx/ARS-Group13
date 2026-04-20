import math, random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

WIDTH, HEIGHT       = 1000, 700
CIRCLE_RADIUS       = 18
LINEAR_SPEED        = 80.0
WHEEL_BASE          = 2 * CIRCLE_RADIUS
MOTION_NOISE_STD    = 2.0
SENSOR_NOISE_STD    = 5.0
BEARING_NOISE_STD   = 0.05
LANDMARKS_RANGE     = 200
CONVERGE_THRESH     = 40

LANDMARKS = [(150,200),(500,300),(750,500),(300,600),(850,150)]

def get_landmark_observations(cx, cy, theta):
    """
    measure distances and bearings to nearby landmarks, with noise
    output: list of (landmark_index, distance, bearing) tuples
    """
    obs = []
    for i,(lx,ly) in enumerate(LANDMARKS):
        dist = math.hypot(lx-cx, ly-cy)
        if dist <= LANDMARKS_RANGE:
            bearing = math.atan2(ly-cy, lx-cx) - theta
            obs.append((i,
                dist    + random.gauss(0, SENSOR_NOISE_STD),
                bearing + random.gauss(0, BEARING_NOISE_STD)))
    return obs

def count_landmarks(cx, cy):
    "how many landmarks are within visible range of the given position?"
    return sum(1 for lx,ly in LANDMARKS if math.hypot(lx-cx,ly-cy) <= LANDMARKS_RANGE)

def update_pose(x, y, theta, v, omega, dt):
    "motion model: given current pose and control inputs, compute new pose after dt seconds"
    if abs(omega) < 1e-6:
        return x+v*math.cos(theta)*dt, y+v*math.sin(theta)*dt, theta
    r = v/omega
    return (x - r*math.sin(theta) + r*math.sin(theta+omega*dt),
            y + r*math.cos(theta) - r*math.cos(theta+omega*dt),
            (theta+omega*dt)%(2*math.pi))

def kalman_predict(mu, sigma, v, omega, dt):
    """
    PREDICT STEP OF EKF
    mu: current mean pose (x,y,theta)
    sigma: current covariance matrix (3x3), uncertainty of the pose estimate
    v, omega: control inputs (linear and angular velocity)
    dt: time step duration
    A: Jacobian to use for covariance update
    R: motion noise covariance to add to uncertainty
    returns: (mu_bar, sigma_bar) predicted mean and covariance after motion
    """
    x,y,theta = mu
    if abs(omega)<1e-6:
        xn,yn,thn = x+v*math.cos(theta)*dt, y+v*math.sin(theta)*dt, theta
        A = np.array([[1,0,-v*math.sin(theta)*dt],[0,1,v*math.cos(theta)*dt],[0,0,1]])
    else:
        r=v/omega
        xn=x-r*math.sin(theta)+r*math.sin(theta+omega*dt)
        yn=y+r*math.cos(theta)-r*math.cos(theta+omega*dt)
        thn=theta+omega*dt
        A=np.array([[1,0,-r*math.cos(theta)+r*math.cos(theta+omega*dt)],
                    [0,1,-r*math.sin(theta)+r*math.sin(theta+omega*dt)],[0,0,1]])
    R=np.diag([MOTION_NOISE_STD**2,MOTION_NOISE_STD**2,0.01])
    return np.array([xn,yn,thn%(2*math.pi)]), A@sigma@A.T+R

def kalman_update(mu_bar, sigma_bar, obs, threshold):
    mu, sigma = mu_bar.copy(), sigma_bar.copy()
    for (lm_idx, z_dist, z_bearing) in obs:
        #take matching landmark
        lx,ly = LANDMARKS[lm_idx]
        x,y,theta = mu
        #compute expected measurement and Jacobian for this landmark
        exp_dist    = math.hypot(lx-x,ly-y)
        exp_bearing = math.atan2(ly-y,lx-x)-theta
        z     = np.array([z_dist, z_bearing])
        z_hat = np.array([exp_dist, exp_bearing])
        #Jacobian of measurement function w.r.t pose
        C=np.array([[-(lx-x)/exp_dist,-(ly-y)/exp_dist,0],
                    [(ly-y)/exp_dist**2,-(lx-x)/exp_dist**2,-1]])
        #measurement noise covariance sensor 
        Q=np.diag([SENSOR_NOISE_STD**2,BEARING_NOISE_STD**2])
        #compute Kalman gain
        K=sigma@C.T@np.linalg.inv(C@sigma@C.T+Q)
        #compute innovation and update mean/covariance
        inn=z-z_hat
        inn[1]=(inn[1]+math.pi)%(2*math.pi)-math.pi
        #kidnapped detection: if the innovation is too large, ignore the update and reinitialize around the landmark
        if abs(inn[0])>threshold:
            sigma=np.eye(3)*50000.0
            mu[0]=lx-z_dist*math.cos(z_bearing+mu[2])
            mu[1]=ly-z_dist*math.sin(z_bearing+mu[2])
            continue
        mu=mu+K@inn
        sigma=(np.eye(3)-K@C)@sigma
    return mu, sigma

def run_trial(threshold, tx, ty, tth, max_steps=600, dt=1/30, seed=0):
    """
    run one complete simulation trial of the kidnapped robot problem, starting with a teleport to (tx,ty,tth)
    EKF gets wrong initial guess (mu) and high uncertainty (sigma) to reflect the kidnap, then tries to recover using the observations
    returns: list of position errors at each step, and the step at which convergence was achieved (or None if never converged)
    """
    random.seed(seed); np.random.seed(seed)
    rx,ry,rth = float(tx),float(ty),float(tth)
    mu    = np.array([rx+300.0, ry+200.0, 0.0])
    sigma = np.eye(3)*100.0
    errors, conv_step = [], None
    v, omega = LINEAR_SPEED*0.5, 0.3
    for step in range(max_steps):
        #simulate motion with noise
        rx,ry,rth = update_pose(rx,ry,rth,v,omega,dt)
        rx=max(CIRCLE_RADIUS+2,min(WIDTH-CIRCLE_RADIUS-2,rx))
        ry=max(CIRCLE_RADIUS+2,min(HEIGHT-CIRCLE_RADIUS-2,ry))
        #EKF predict and update
        mu,sigma=kalman_predict(mu,sigma,v,omega,dt)
        obs=get_landmark_observations(rx,ry,rth)
        #possibly update with observations, and check error for convergence
        if obs: mu,sigma=kalman_update(mu,sigma,obs,threshold)
        err=math.hypot(mu[0]-rx,mu[1]-ry)
        errors.append(err)
        if conv_step is None and err<CONVERGE_THRESH:
            conv_step=step
    return errors, conv_step

N_SEEDS=15

#Experiment 1: threshold sensitivity 
THRESHOLDS=[30,75,150,300,500]
TX,TY,TTH=500,400,0.5 #static teleport location for thid experiment
thresh_curves, thresh_conv = {}, {}
for thr in THRESHOLDS:
    errs_all, convs = [], []
    #for all thresholds, run multiple trials with different random seeds to average out noise effects
    for s in range(N_SEEDS):
        e,c = run_trial(thr,TX,TY,TTH,seed=s)
        errs_all.append(e)
        convs.append(c if c is not None else 600)
    thresh_curves[thr]=np.mean(errs_all,axis=0) #avg error curve
    thresh_conv[thr]=np.mean(convs)/30 #avg convergence time

#Experiment 2: landmark count (0/1/2) sensitivity
# teleport locations with number of landmarks in range: 0→(50,400), 1→(50,80), 2→(305,260)
LM_CONDITIONS = {
    "0 landmarks": (50,  400, 0.0),
    "1 landmark":  (50,   80, 0.0),
    "2 landmarks": (305, 260, 0.0),
}
FIXED_THR=150
lm_curves, lm_conv, lm_n = {}, {}, {}
for label,(tx,ty,tth) in LM_CONDITIONS.items():
    lm_n[label]=count_landmarks(tx,ty)
    errs_all, convs = [], []
    for s in range(N_SEEDS):
        e,c=run_trial(FIXED_THR,tx,ty,tth,seed=s)
        errs_all.append(e)
        convs.append(c if c is not None else 600)
    lm_curves[label]=np.mean(errs_all,axis=0)
    lm_conv[label]=np.mean(convs)/30

# ── Plotting ──────────────────────────────────────────────────────────────────
time_axis=np.arange(600)/30

C_THR=["#C05050","#D4893A","#4A7FB5","#5A9E6F","#8B5FA0"]
C_LM =["#AAAAAA","#D4893A","#4A7FB5"]

fig=plt.figure(figsize=(14,10))
fig.patch.set_facecolor("#F8F8F6")
gs=gridspec.GridSpec(2,2,figure=fig,hspace=0.42,wspace=0.32)

# 1a error curves, per threshold avg postion error over time
ax=fig.add_subplot(gs[0,0]); ax.set_facecolor("#F8F8F6")
for thr,c in zip(THRESHOLDS,C_THR):
    ax.plot(time_axis,thresh_curves[thr],color=c,lw=1.8,label=f"τ = {thr} px")
ax.axhline(CONVERGE_THRESH,color="#333",lw=1,ls="--",alpha=0.5)
ax.text(time_axis[-1]*0.97,CONVERGE_THRESH+4,"convergence\nthreshold",
        ha="right",fontsize=7.5,color="#555")
ax.set_xlabel("Time after kidnap (s)",fontsize=9)
ax.set_ylabel("Position error (px)",fontsize=9)
ax.set_title("Exp 1 — Position error over time\nper detection threshold τ",fontsize=10,fontweight="bold")
ax.legend(fontsize=8,framealpha=0.45); ax.set_xlim(0,20); ax.grid(True,alpha=0.25)

# 1b convergence bar, per threshold avg time for convergence 
ax=fig.add_subplot(gs[0,1]); ax.set_facecolor("#F8F8F6")
bars=ax.bar([str(t) for t in THRESHOLDS],[thresh_conv[t] for t in THRESHOLDS],
            color=C_THR,width=0.55,edgecolor="white",lw=0.8)
for bar,thr in zip(bars,THRESHOLDS):
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.05,
            f"{thresh_conv[thr]:.1f}s",ha="center",fontsize=9)
ax.set_xlabel("Detection threshold τ (px)",fontsize=9)
ax.set_ylabel("Avg time to convergence (s)",fontsize=9)
ax.set_title("Exp 1 — Convergence speed\nvs detection threshold",fontsize=10,fontweight="bold")
ax.grid(True,axis="y",alpha=0.25)

# 2a error curves, per landmark condition -> avg error curve over time 
ax=fig.add_subplot(gs[1,0]); ax.set_facecolor("#F8F8F6")
for (label,curve),c in zip(lm_curves.items(),C_LM):
    ax.plot(time_axis,curve,color=c,lw=1.8,label=f"{label} (n={lm_n[label]})")
ax.axhline(CONVERGE_THRESH,color="#333",lw=1,ls="--",alpha=0.5)
ax.set_xlabel("Time after kidnap (s)",fontsize=9)
ax.set_ylabel("Position error (px)",fontsize=9)
ax.set_title("Exp 2 — Position error over time\nvs number of visible landmarks",fontsize=10,fontweight="bold")
ax.legend(fontsize=8,framealpha=0.45); ax.set_xlim(0,20); ax.grid(True,alpha=0.25)

# 2b convergence bar, per landmark condition -> avg convergence time
ax=fig.add_subplot(gs[1,1]); ax.set_facecolor("#F8F8F6")
labels=list(lm_conv.keys())
vals=[lm_conv[l] for l in labels]
bars=ax.bar(labels,vals,color=C_LM,width=0.45,edgecolor="white",lw=0.8)
for bar,val,label in zip(bars,vals,labels):
    txt=f"{val:.1f}s" if val<20 else "no conv."
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.05,
            txt,ha="center",fontsize=9)
ax.set_xlabel("Teleport location",fontsize=9)
ax.set_ylabel("Avg time to convergence (s)",fontsize=9)
ax.set_title("Exp 2 — Convergence speed\nvs number of visible landmarks",fontsize=10,fontweight="bold")
ax.grid(True,axis="y",alpha=0.25)

fig.suptitle("Kidnapped Robot Problem — EKF Recovery Experiments\n",
             fontsize=12,fontweight="bold",y=1.01)

plt.savefig("experiments_results.png",
            dpi=160,bbox_inches="tight",facecolor=fig.get_facecolor())