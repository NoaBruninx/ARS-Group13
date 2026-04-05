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
MARGIN = 5

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

def circle_rect_collision(cx, cy, radius, rx, ry, rw, rh):
    # Find closest point on rect to circle
    closest_x = max(rx, min(cx, rx + rw))
    closest_y = max(ry, min(cy, ry + rh))
    distance = math.sqrt((cx - closest_x)**2 + (cy - closest_y)**2)
    return distance < radius

def check_collision(cx, cy, radius):
    for obs in WALLS:
        if circle_rect_collision(cx, cy, radius, *obs):
            return True
    return False

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

def get_sensor_readings(circle_x, circle_y):
    sensor_distances = []
    wall_segments = get_wall_segments()

    for sensor_index in range(NUM_SENSORS):
        # Calculate sensor angle
        angle_degrees = sensor_index * SENSOR_ANGLE_STEP
        angle_radians = math.radians(angle_degrees)

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

def draw_sensor_values(screen, cx, cy, readings):
    for i, dist in enumerate(readings):
        angle = math.radians(i * SENSOR_ANGLE_STEP)

        # Position where the text SHOULD be centered
        pos_x = cx + math.cos(angle) * (CIRCLE_RADIUS + 20)
        pos_y = cy + math.sin(angle) * (CIRCLE_RADIUS + 20)

        text = font.render(str(int(dist)), True, (0, 0, 0))
        text_rect = text.get_rect(center=(pos_x, pos_y))  # 👈 FIX

        screen.blit(text, text_rect)

def main():
    # Initialize circle position
    circle_x = WIDTH // 2
    circle_y = HEIGHT // 2
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get state of all keys
        keys = pygame.key.get_pressed()

        # Move circle based on arrow keys
        new_x = circle_x
        new_y = circle_y
        if keys[pygame.K_LEFT]:
            new_x -= CIRCLE_SPEED
        if keys[pygame.K_RIGHT]:
            new_x += CIRCLE_SPEED
        if keys[pygame.K_UP]:
            new_y -= CIRCLE_SPEED
        if keys[pygame.K_DOWN]:
            new_y += CIRCLE_SPEED

        # Check collision with obstacles
        if not check_collision(new_x, new_y, CIRCLE_RADIUS + MARGIN):
            circle_x = new_x
            circle_y = new_y

        # Keep circle within screen bounds with margin
        circle_x = max(CIRCLE_RADIUS + MARGIN, min(WIDTH - CIRCLE_RADIUS - MARGIN, circle_x))
        circle_y = max(CIRCLE_RADIUS + MARGIN, min(HEIGHT - CIRCLE_RADIUS - MARGIN, circle_y))

        screen.fill(WHITE)

        draw_walls(screen)
        pygame.draw.circle(screen, RED, (circle_x, circle_y), CIRCLE_RADIUS)

        sensor_readings = get_sensor_readings(circle_x, circle_y)
        draw_sensor_values(screen, circle_x, circle_y, sensor_readings)

        pygame.display.flip()
        pygame.time.Clock().tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()