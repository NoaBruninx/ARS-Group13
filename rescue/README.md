# ARS Final Assignment - Search and Rescue

## Scenario

This project implements the **Search and Rescue** application from the ARS assignment list. A heterogeneous team of robots explores a damaged building, maps the unknown environment, detects hidden victims, and sends a rescue robot to reach and assist them.

The damaged building contains collapsed internal walls, narrow corridors, cross-passages, fixed landmarks, hidden victim locations, and an optional collapsed passage. In the normal visualization mode, the map starts hidden. Only cells observed by the robots' distance sensors are revealed through the occupancy grid as the robots move. The robots must estimate their pose, build an occupancy grid map, and use the generated map for navigation.

## Robots

- **Scout Robot**: explores the damaged building, discovers victim locations, and improves map coverage.
- **Rescue Robot**: uses the shared map and detected victim information to reach victims and mark them as rescued/assisted.

## Implemented components

- 2D Pygame simulator
- Distance sensors with ray casting
- Collision handling with simple wall sliding
- EKF-SLAM using known range-bearing landmark correspondence
- Occupancy grid mapping with log-odds
- Shared occupancy map between robots
- Evolved behaviour-based vector controller
- GENITOR-inspired steady-state genetic algorithm
- Batch experiments and plots

## Run the interactive simulation

```bash
pip install pygame numpy matplotlib
python main.py
```

### Keys

- `R` reset
- `SPACE` pause
- `M` toggle occupancy map
- `T` toggle trajectories
- `S` toggle sensor rays
- `B` activate collapsed passage
- `G` debug only: show/hide full ground-truth building map
- `E` run short GENITOR evolution and load best genome
- `ESC` quit

## Run evolution only

```bash
python evolution.py
```

Outputs:

- `results/evolution_log.csv`
- `results/best_genome.json`

## Run experiments

Quick version:

```bash
python experiments.py --quick
```

Fuller version:

```bash
python experiments.py --full
```

Outputs:

- CSV files in `results/`
- plots in `figures/`

## Suggested report experiments

1. Baseline controller vs evolved controller
2. One rescue robot vs Scout + Rescue robot team
3. Shared map vs independent maps
4. Normal damaged building vs collapsed passage
5. Map resolution comparison

## Important implementation note

The occupancy grid is updated using the **EKF-SLAM estimated pose**, not the ground-truth pose. This is important because the final assignment requires the robot to use self-localization and the generated map for navigation.

The full ground-truth map is **not shown by default**. Press `G` only for debugging or for a report screenshot that explicitly compares the discovered map with the real environment. For the final demo video, use the normal hidden-map view so it is clear that the map is built during exploration.

## EKF-SLAM note

The project uses `ekf_slam.py` for localization. The SLAM state is:

```text
X = [x, y, theta, l1_x, l1_y, l2_x, l2_y, ...]^T
```

The robot pose and landmark locations are estimated together. Landmark data association is assumed to be known, which matches the simplified assignment setting and keeps the implementation explainable. The occupancy grid map is still built from range sensor beams, while EKF-SLAM provides the estimated pose used for mapping and navigation.

In the visualization, the dotted trajectory is the EKF-SLAM estimated robot path. Small colored crosses are the robot's estimated landmark positions from EKF-SLAM, not the ground-truth landmark map.

## Localization visualization

The visualization includes two assignment-specific localization cues:

- A **2-sigma covariance ellipse** around the EKF-SLAM estimated robot pose. The ellipse is computed from the x/y block of the SLAM covariance matrix.
- A **green observation line** between the robot and each landmark currently inside the landmark sensor range. These are the same range-bearing landmark observations used for EKF-SLAM correction.

These cues help demonstrate when the robot is only predicting from motion and when it is correcting its pose estimate using visible features.

## Controller summary

The controller uses an evolved behaviour-based vector policy. Victim attraction, frontier exploration, obstacle avoidance, robot separation, corridor following, corridor switching, and visited-victim avoidance each produce a vector. The genome controls the weights of these behaviours, so the controller remains explainable while still being optimized by evolution.

## Victim marker legend

- Orange circle/cross: victim exists in the ground-truth/debug map but has not been discovered yet. This is only visible when pressing `G`.
- Yellow circle/cross: victim has been detected but has not yet been rescued.
- Green circle/cross: victim has been reached and rescued/assisted by the Rescue robot.

For the final video, keep the normal hidden-map view. In that mode, victim locations appear only after detection or rescue, so the robots do not start with oracle knowledge of all victim positions.

## Compact-map version

This version uses a laptop-friendly world size (`1100 x 700`). The damaged-building barriers are split into segments with cross-passages so the robots can move between corridors instead of getting trapped in one lane.

## External resources

Mention all external resources in the final report, including Python libraries and any use of generative AI.
