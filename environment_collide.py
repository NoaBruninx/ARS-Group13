import pygame
import sys
import math


# ARS Assignment - Part 1 - Group 13: Mobile Robot Simulator



pygame.init()

#   Display 
WIDTH, HEIGHT = 1000, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("ARS Assignment - Part 1 - Group 13: Mobile Robot Simulator")

#   Colors 
RED    = (220,  60,  60)
WHITE  = (255, 255, 255)
BLACK  = (  0,   0,   0)
GRAY   = (200, 200, 200)
BLUE   = ( 50, 100, 220)
GREEN  = ( 30, 180,  60)

#   Robot parameters 
CIRCLE_RADIUS   = 18          # pixels  (physical body radius)
LINEAR_SPEED    = 80.0        # pixels/second  (forward speed)


#  Sensor parameters 
NUM_SENSORS          = 12
SENSOR_ANGLE_STEP    = 2 * math.pi / NUM_SENSORS   # radians
MAX_SENSOR_DISTANCE  = 200   # pixels

#   Font  
font_small = pygame.font.SysFont(None, 16)
font_med   = pygame.font.SysFont(None, 22)


#  MAZE  – defined as a list of (x1, y1, x2, y2) line segments


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

ALL_WALL_SEGMENTS = None 

def get_wall_segments():
    global ALL_WALL_SEGMENTS
    if ALL_WALL_SEGMENTS is None:
        ALL_WALL_SEGMENTS = list(INTERNAL_WALLS) + [
            (0,      0,      WIDTH, 0     ),   # top
            (WIDTH,  0,      WIDTH, HEIGHT),   # right
            (WIDTH,  HEIGHT, 0,     HEIGHT),   # bottom
            (0,      HEIGHT, 0,     0     ),   # left
        ]
    return ALL_WALL_SEGMENTS


def draw_walls(surface):
    
    
    for (x1, y1, x2, y2) in INTERNAL_WALLS:
        pygame.draw.line(surface, BLACK, (x1, y1), (x2, y2), 4)
    # Outer boundary
    pygame.draw.rect(surface, BLACK, (0, 0, WIDTH, HEIGHT), 4)



#  MOTION MODEL  –  velocity-based (ICC / circular arc)


def update_pose_velocity_model(x, y, theta, v, omega, dt):
    """
    Noise-free velocity motion model.
    Inputs:
        x, y     : current position (pixels)
        theta    : current heading (radians)
        v        : linear velocity (pixels/s)
        omega    : angular velocity (rad/s)
        dt       : timestep (s)
    Returns:
        x', y', theta'
    """
    if abs(omega) < 1e-6:
        # Straight-line motion (omega ≈ 0)
        x_new     = x + v * math.cos(theta) * dt
        y_new     = y + v * math.sin(theta) * dt
        theta_new = theta
    else:
        # Circular arc motion
        r = v / omega                          # radius of curvature
        # Instantaneous Center of Curvature (ICC)
        x_icc = x - r * math.sin(theta)
        y_icc = y + r * math.cos(theta)

        dtheta = omega * dt

        x_new     = (x_icc
                     + (x - x_icc) * math.cos(dtheta)
                     - (y - y_icc) * math.sin(dtheta))
        y_new     = (y_icc
                     + (x - x_icc) * math.sin(dtheta)
                     + (y - y_icc) * math.cos(dtheta))
        theta_new = theta + dtheta

    return x_new, y_new, theta_new % (2 * math.pi)



#  SENSOR MODEL  –  12 distance sensors (line intersection)


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
    
    
    wall_segs = get_wall_segments()
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



#  COLLISION HANDLING  –  wall-sliding


def point_to_segment_distance_and_normal(px, py, ax, ay, bx, by):
  
    
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-10:
        # Degenerate segment – treat as point
        dist = math.hypot(px - ax, py - ay)
        if dist < 1e-10:
            return 0.0, 0.0, 0.0
        return dist, (px - ax) / dist, (py - ay) / dist

    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    closest_x = ax + t * dx
    closest_y = ay + t * dy
    diff_x = px - closest_x
    diff_y = py - closest_y
    dist = math.hypot(diff_x, diff_y)
    if dist < 1e-10:
        return 0.0, 0.0, 0.0
    return dist, diff_x / dist, diff_y / dist

# Check if robot at (cx, cy) collides with any wall (considering robot radius)
def collides_at_position(cx,cy):
    
    for (ax, ay, bx, by) in get_wall_segments():
        dist, _, _ = point_to_segment_distance_and_normal(cx, cy, ax, ay, bx, by)
        if dist < CIRCLE_RADIUS:
            return True
    return False





def resolve_collisions(x, y, dx, dy, substeps=12):

    step_dx = dx / substeps # small incremental steps for better collision handling
    step_dy = dy / substeps # small incremental steps for better collision handling
    
    new_x, new_y = x, y # start at current position
    for _ in range(substeps):
        # Try to move by one small step
        trial_x = new_x + step_dx
        trial_y = new_y + step_dy
        
        if not collides_at_position(trial_x, trial_y):
            # No collision, move freely
            new_x, new_y = trial_x, trial_y
            continue
        
        
        best_dist = float('inf')
        best_nx, best_ny = 0.0, 0.0
        for (ax,ay,bx,by) in get_wall_segments():
            dist, nx, ny = point_to_segment_distance_and_normal(new_x, new_y, ax, ay, bx, by)
            if dist < best_dist:
                best_dist = dist
                best_nx ,best_ny = nx, ny
        dot = step_dx * best_nx + step_dy * best_ny
        
        slide_dx = step_dx - dot * best_nx
        slide_dy = step_dy - dot * best_ny
        
        trial_x = new_x + slide_dx
        trial_y = new_y + slide_dy
        
        
        if not collides_at_position(trial_x, trial_y):
            new_x, new_y = trial_x, trial_y
      
            
    # Clamp to stay within bounds (accounting for robot radius and a small margin)
    new_x = max(CIRCLE_RADIUS + 2, min(WIDTH  - CIRCLE_RADIUS - 2, new_x))
    new_y = max(CIRCLE_RADIUS + 2, min(HEIGHT - CIRCLE_RADIUS - 2, new_y))

    return new_x, new_y


# DRAWING HELPERS : Draw robot body and heading indicator

def draw_robot(surface, cx, cy, theta):
    
    pygame.draw.circle(surface, RED, (int(cx), int(cy)), CIRCLE_RADIUS)
    # Heading line
    hx = cx + math.cos(theta) * CIRCLE_RADIUS
    hy = cy + math.sin(theta) * CIRCLE_RADIUS
    pygame.draw.line(surface, BLACK,
                     (int(cx), int(cy)), (int(hx), int(hy)), 3)


# Draw sensor rays and distance values
def draw_sensors(surface, cx, cy, theta, distances, hit_points):
    
    for i, (dist, pt) in enumerate(zip(distances, hit_points)):
        angle = theta + i * SENSOR_ANGLE_STEP
        
        # Sensor start point (at robot edge)
        sx = cx + math.cos(angle) * CIRCLE_RADIUS
        sy = cy + math.sin(angle) * CIRCLE_RADIUS
        
        
        pygame.draw.line(surface, GRAY,
                         (int(sx), int(sy)), (int(pt[0]), int(pt[1])), 1)
        # Distance label near the tip
        label_x = pt[0] + math.cos(angle) * 14
        label_y = pt[1] + math.sin(angle) * 14
        txt = font_small.render(str(int(dist)), True, BLUE)
        rect = txt.get_rect(center=(int(label_x), int(label_y)))
        surface.blit(txt, rect)


def draw_hud(surface, v_left, v_right, theta_deg):

    lines = [
        f"Left motor  : {v_left:+.1f} px/s",
        f"Right motor : {v_right:+.1f} px/s",
        f"Heading     : {theta_deg % 360:.1f} deg",
    ]
    for i, line in enumerate(lines):
        txt = font_med.render(line, True, BLACK)
        surface.blit(txt, (10, 10 + i * 22))



#  DIFFERENTIAL-DRIVE  –  convert wheel speeds to v / omega

WHEEL_BASE = 2 * CIRCLE_RADIUS   # distance between wheels (pixels)
# Standard differential-drive kinematics
def wheel_speeds_to_v_omega(v_left, v_right):
   
    v     = (v_right + v_left) / 2.0
    omega = (v_right - v_left) / WHEEL_BASE
    return v, omega



#  MAIN LOOP


def main():
    clock = pygame.time.Clock()

    # Initial pose  (start in the upper-left open area)
    robot_x     = 100.0
    robot_y     = 400.0
    robot_theta = 0.0      # radians, 0 = facing right

    running = True
    while running:
        dt_actual = clock.tick(60) / 1000.0   # real elapsed time (s)
        dt_actual = min(dt_actual, 0.05)       # cap to avoid huge jumps

        #   Event handling                   
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        #   Keyboard → wheel speeds              
        keys = pygame.key.get_pressed()

        v_left  = 0.0
        v_right = 0.0

        if keys[pygame.K_UP]:
            v_left  += LINEAR_SPEED
            v_right += LINEAR_SPEED
        if keys[pygame.K_DOWN]:
            v_left  -= LINEAR_SPEED
            v_right -= LINEAR_SPEED
        if keys[pygame.K_LEFT]:
            # Rotate left: right wheel faster
            v_left  += LINEAR_SPEED * 0.5
            v_right -= LINEAR_SPEED * 0.5
        if keys[pygame.K_RIGHT]:
            # Rotate right: left wheel faster
            v_left  -= LINEAR_SPEED * 0.5
            v_right += LINEAR_SPEED * 0.5

        #   Velocity-based motion model            
        v, omega = wheel_speeds_to_v_omega(v_left, v_right)

        new_x, new_y, new_theta = update_pose_velocity_model(
            robot_x, robot_y, robot_theta, v, omega, dt_actual)

        #   Collision resolution (wall sliding)        
        dx = new_x - robot_x
        dy = new_y - robot_y
        robot_x, robot_y = resolve_collisions(robot_x, robot_y, dx, dy)
        robot_theta = new_theta   # orientation is never blocked

        #   Sensor readings                   
        distances, hit_points = get_sensor_readings(
            robot_x, robot_y, robot_theta)

        #   Rendering                      
        screen.fill(WHITE)
        draw_walls(screen)
        draw_sensors(screen, robot_x, robot_y, robot_theta,
                     distances, hit_points)
        draw_robot(screen, robot_x, robot_y, robot_theta)
        draw_hud(screen, v_left, v_right,
                 math.degrees(robot_theta))

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
