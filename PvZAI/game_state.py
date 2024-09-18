import numpy as np
from PlantsVsZombies.source import constants as c

class RowState:
    def __init__(self):
        self.plants = 0
        self.sunflower_count = 0
        self.attacker_count = 0
        self.defensive_count = 0
        self.zombie_proximity = 0  # 0: No zombies, 1: Far, 2: Medium, 3: Close
        self.zombie_count = 0
        self.has_car = 0

    def to_feature_vector(self):
        return np.array([
            self.sunflower_count,
            self.zombie_proximity,
            (self.attacker_count > 0),
            (self.defensive_count > 0)
        ])

class GameState:
    def __init__(self, sun_count, rows=c.ROWS_NUMBER):
        self.sun_count = sun_count
        self.rows = [RowState() for _ in range(rows)]

    def get_row_selector_state(self):
        return np.array([row.zombie_proximity**2 + row.zombie_count**1.5 - row.plants**2 for row in self.rows])

    def get_row_features(self):
        return np.array([
            [
                -row.sunflower_count,
                -row.attacker_count,
                -row.defensive_count,
                row.zombie_count,
                row.zombie_proximity,
                -row.has_car
            ] for row in self.rows
        ])


# We'll keep these classes for compatibility, but they won't be used in the main GameState
class Plant:
    def __init__(self, plant_type, position, health, state, cost):
        self.type = plant_type
        self.position = position
        self.health = health
        self.state = state
        self.cost = cost

class Zombie:
    def __init__(self, zombie_type, position, health, state):
        self.type = zombie_type
        self.position = position
        self.health = health
        self.state = state
