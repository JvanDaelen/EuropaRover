### Code to generate a numpy array based map containing randomly generated objects
import numpy as np

class Map():
    #Grid label constants
    EMPTY = 0
    ROCK = 1
    CLIFF = 2

    def __init__(self, grid_width : int, grid_height : int) -> None:
        self.grid_width = grid_width
        self.grid_height = grid_height

    def generate_terrain(self, nr_obstacles: int = 10, object_radius: int = 2) -> None:
        self.terrain = np.zeros((self.grid_width, self.grid_height))

        for _ in range(nr_obstacles):
            rand_x, rand_y = self.get_random_coordinate(self.grid_width, self.grid_height)
            self.terrain[
                max(0, rand_x - object_radius + 1):min(self.grid_width, rand_x + object_radius),
                max(0, rand_y - object_radius + 1):min(self.grid_height, rand_y + object_radius),
                ] = self.ROCK
        
        # Pick a target location that is not in an obstacle
        self.target_coords = None
        while self.target_coords is None:
            rand_x, rand_y = self.get_random_coordinate(self.grid_width, self.grid_height)
            if self.terrain[rand_x, rand_y] == self.EMPTY:
                self.target_coords = (rand_x, rand_y)

    @staticmethod
    def get_random_coordinate(max_width : int, max_height : int):
        rand_x = np.random.randint(0, max_width)
        rand_y = np.random.randint(0, max_height)
        return rand_x, rand_y