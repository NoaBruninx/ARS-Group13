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

        pygame.display.flip()
        pygame.time.Clock().tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()