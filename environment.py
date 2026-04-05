import pygame
import sys
import math

pygame.init()

# Setup display
WIDTH, HEIGHT = 1200, 900
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("ARS assignment")
RED = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

CIRCLE_RADIUS = 25
CIRCLE_SPEED = 5
ROTATION_SPEED = 5 
MARGIN = 30

WALLS = [ #x, y, width, height -> from walls in frame
    (200, 100, 5, 500),  
    (200, 450, 200, 5),  
    (400, 100, 5, 350),  
    (400, 100, 200, 5),   
    (800, 200, 5, 500), 
    (800, 450, 200, 5),  
]

NUM_SENSORS = 12
SENSOR_ANGLE_STEP = 360 / NUM_SENSORS
MAX_SENSOR_DISTANCE = 200
font = pygame.font.SysFont(None, 16)

def draw_walls(screen):
    pygame.draw.line(screen, BLACK, (200, 100), (200, 600), 5)  
    pygame.draw.line(screen, BLACK, (200, 450), (400, 450), 5)  
    pygame.draw.line(screen, BLACK, (400, 450), (400, 100), 5)  
    pygame.draw.line(screen, BLACK, (400, 100), (600, 100), 5)

    pygame.draw.line(screen, BLACK, (800, 200), (800, 700), 5)
    pygame.draw.line(screen, BLACK, (800, 450), (1000, 450), 5)

def get_wall_segments():
    segments = []
    for (x, y, w, h) in WALLS:
        segments.append((x, y, x+w, y))       # top
        segments.append((x, y, x, y+h))       # left
        segments.append((x+w, y, x+w, y+h))   # right
        segments.append((x, y+h, x+w, y+h))   # bottom

    segments.append((0, 0, WIDTH, 0))         # top
    segments.append((WIDTH, 0, WIDTH, HEIGHT)) # right
    segments.append((WIDTH, HEIGHT, 0, HEIGHT)) # bottom
    segments.append((0, HEIGHT, 0, 0))        # left
    return segments

def line_intersection(
        line1_x1, line1_y1, line1_x2, line1_y2,
        line2_x1, line2_y1, line2_x2, line2_y2
):
    # Calculate denominator to check if lines are parallel
    denominator = (line1_x1 - line1_x2) * (line2_y1 - line2_y2) -(line1_y1 - line1_y2) * (line2_x1 - line2_x2)
    if denominator == 0:
        return None  # Lines are parallel or overlapping

    # Calculate where the intersection happens along each line
    t = ((line1_x1 - line2_x1) * (line2_y1 - line2_y2) -(line1_y1 - line2_y1) * (line2_x1 - line2_x2)) / denominator

    u = -((line1_x1 - line1_x2) * (line1_y1 - line2_y1) - (line1_y1 - line1_y2) * (line1_x1 - line2_x1)) / denominator

    # Check if intersection is within both line segments
    if 0 <= t <= 1 and 0 <= u <= 1:
        intersection_x = line1_x1 + t * (line1_x2 - line1_x1)
        intersection_y = line1_y1 + t * (line1_y2 - line1_y1)
        return intersection_x, intersection_y

    return None

def get_sensor_readings(circle_x, circle_y, angle_degrees):
    sensor_distances = []
    wall_segments = get_wall_segments()

    for sensor_index in range(NUM_SENSORS):
        # Calculate sensor angle relative to robot orientation
        angle_degrees_rel = angle_degrees + sensor_index * SENSOR_ANGLE_STEP
        angle_radians = math.radians(angle_degrees_rel)

        # End point of the sensor ray
        sensor_end_x = circle_x + math.cos(angle_radians) * MAX_SENSOR_DISTANCE
        sensor_end_y = circle_y + math.sin(angle_radians) * MAX_SENSOR_DISTANCE

        # Start with max distance (no wall detected yet)
        closest_distance = MAX_SENSOR_DISTANCE

        # Check intersection with each wall segment
        for (wall_x1, wall_y1, wall_x2, wall_y2) in wall_segments:
            intersection = line_intersection(
                circle_x, circle_y,
                sensor_end_x, sensor_end_y,
                wall_x1, wall_y1,
                wall_x2, wall_y2
            )

            if intersection:
                hit_x, hit_y = intersection

                distance_to_wall = math.sqrt(
                    (hit_x - circle_x) ** 2 +
                    (hit_y - circle_y) ** 2
                )

                # Keep the closest wall hit
                if distance_to_wall < closest_distance:
                    closest_distance = distance_to_wall

        sensor_distances.append(closest_distance)

    return sensor_distances

def is_forward_blocked(sensors):
    front = sensors[0:3] + sensors[9:12]  # S0, S1, S2, S9, S10, S11
    return any(dist <= MARGIN for dist in front)

def is_backward_blocked(sensors):
    back = sensors[4:9]  # S4, S5, S6, S7, S8
    return any(dist <= MARGIN for dist in back)


def draw_sensor_values(screen, cx, cy, sensors, angle_degrees):
    for i, dist in enumerate(sensors):
        angle = math.radians(angle_degrees + i * SENSOR_ANGLE_STEP)

        # Position where the text SHOULD be centered
        pos_x = cx + math.cos(angle) * (CIRCLE_RADIUS + 20)
        pos_y = cy + math.sin(angle) * (CIRCLE_RADIUS + 20)

        text = font.render(str(int(dist)), True, (0, 0, 0))
        text_rect = text.get_rect(center=(pos_x, pos_y)) 

        screen.blit(text, text_rect)

def draw_robot(screen, cx, cy, angle_degrees):
    pygame.draw.circle(screen, RED, (int(cx), int(cy)), CIRCLE_RADIUS)

    # Draw orientation line showing the robot's pose
    radians = math.radians(angle_degrees)
    line_end_x = cx + math.cos(radians) * (CIRCLE_RADIUS)
    line_end_y = cy + math.sin(radians) * (CIRCLE_RADIUS)
    pygame.draw.line(screen, BLACK, (int(cx), int(cy)), (int(line_end_x), int(line_end_y)), 3)


def main():
    # Initialize circle position and orientation
    circle_x = WIDTH // 2
    circle_y = HEIGHT // 2
    angle_degrees = 0
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get state of all keys
        keys = pygame.key.get_pressed()

        # Rotate left/right to change pose
        if keys[pygame.K_LEFT]:
            angle_degrees = (angle_degrees - ROTATION_SPEED) % 360
        if keys[pygame.K_RIGHT]:
            angle_degrees = (angle_degrees + ROTATION_SPEED) % 360

        # Move forward/backward based on current orientation
        move_dx = 0
        move_dy = 0
        sensor_readings = get_sensor_readings(circle_x, circle_y, angle_degrees)

        if keys[pygame.K_UP]:
            if not is_forward_blocked(sensor_readings):
                move_dx += math.cos(math.radians(angle_degrees)) * CIRCLE_SPEED
                move_dy += math.sin(math.radians(angle_degrees)) * CIRCLE_SPEED

        if keys[pygame.K_DOWN]:
            if not is_backward_blocked(sensor_readings):
                move_dx -= math.cos(math.radians(angle_degrees)) * CIRCLE_SPEED
                move_dy -= math.sin(math.radians(angle_degrees)) * CIRCLE_SPEED

        new_x = circle_x + move_dx
        new_y = circle_y + move_dy

        # Keep circle within screen bounds with margin
        circle_x = max(CIRCLE_RADIUS + MARGIN, min(WIDTH - CIRCLE_RADIUS - MARGIN, new_x))
        circle_y = max(CIRCLE_RADIUS + MARGIN, min(HEIGHT - CIRCLE_RADIUS - MARGIN, new_y))

        screen.fill(WHITE)

        draw_walls(screen)
        draw_robot(screen, circle_x, circle_y, angle_degrees)

        # Reuse sensor readings from movement logic for drawing
        draw_sensor_values(screen, circle_x, circle_y, sensor_readings, angle_degrees)

        pygame.display.flip()
        pygame.time.Clock().tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()