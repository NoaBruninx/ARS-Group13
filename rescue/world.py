import random
from config import *


class Victim:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.carried_by = None
        self.saved = False


class World:

    def __init__(self, num_victims=10):

        # ---------------- WALLS ----------------
        self.internal_walls = [
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

        # ---------------- LANDMARKS ----------------
        self.landmarks = [
            (150, 200),
            (500, 300),
            (750, 500),
            (300, 600),
            (850, 150),
        ]

        self.wall_segments = (
                list(self.internal_walls)
                + [
                    (0, 0, WIDTH, 0),
                    (WIDTH, 0, WIDTH, HEIGHT),
                    (WIDTH, HEIGHT, 0, HEIGHT),
                    (0, HEIGHT, 0, 0),
                ]
        )

        # ---------------- ENTITIES ----------------
        self.robots = []
        self.victims = []

        # 🔥 AUTO SPAWN HERE (NO MANUAL WORK EVER AGAIN)
        self.spawn_victims(num_victims)

    # -------------------------------------------------
    # ROBOTS
    # -------------------------------------------------

    def add_robot(self, robot):
        self.robots.append(robot)

    # -------------------------------------------------
    # VICTIM SPAWNING (SAFE + AUTOMATIC)
    # -------------------------------------------------

    def spawn_victims(self, n):

        self.victims = []

        for _ in range(n):

            for _attempt in range(100):  # prevents infinite loops

                x = random.randint(50, WIDTH - 50)
                y = random.randint(50, HEIGHT - 50)

                # avoid spawning inside walls (simple but effective)
                safe = True

                for ax, ay, bx, by in self.internal_walls:
                    if abs(x - ax) < 25 and abs(y - ay) < 25:
                        safe = False
                        break

                if safe:
                    self.victims.append(Victim(x, y))
                    break

    # -------------------------------------------------
    # QUERY
    # -------------------------------------------------

    def get_alive_victims(self):
        return [v for v in self.victims if not v.saved]

def world_to_grid(x, y): return int(y // MAP_CELL_SIZE), int(x // MAP_CELL_SIZE)