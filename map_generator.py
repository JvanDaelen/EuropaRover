### Code to generate a numpy array based map containing randomly generated objects
import numpy as np

class Map():
    #Grid label constants
    TERRAIN_TYPES = {
        'EMPTY' : 0, 
        'CLIFF' : 1, 
        'END' : 2, 
        'PASSED' : 3, 
        'CURRENT' : 4
        }
    EXPLORED = 1

    def __init__(self, 
                 grid_width = 10, 
                 grid_height = 10, 
                 nr_cliffs = 3, 
                 cliff_length_max = 4,
                 cliff_length_min = 2
                 ) -> None:
        # make class vars
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.nr_cliffs = nr_cliffs
        self.cliff_length_max = cliff_length_max
        self.cliff_length_min = cliff_length_min
        self.terrain = np.zeros((grid_width, grid_height))
        print(self.terrain)
        self._generate_cliffs()
        print(self.terrain)
    
    def _generate_cliffs(self):
        for cliff_nr in range(self.nr_cliffs):
            print(f'{cliff_nr = }')
            # Pick random point
            center_x, center_y = self.get_random_coordinate()
            # self.terrain[center_x, center_y] = self.TERRAIN_TYPES['CLIFF']

            # Pick random direction (between + and - 90 to use cos and sin)
            direction_deg = np.random.rand() * 180 - 90
            # pick random length in range
            random_length = np.random.default_rng().integers(
                self.cliff_length_min, 
                self.cliff_length_max, 
                endpoint=True
                ) 
            # Walk length/2 in both + and - direction
            x_walk = np.cos(np.deg2rad(direction_deg))
            y_walk = np.sin(np.deg2rad(direction_deg))
            for step_nr in range(int(np.ceil(random_length/2)) + 1):
                print(f'{step_nr = }')
                if 0 <= center_x + x_walk * step_nr < self.grid_width and 0 <= center_y + y_walk * step_nr < self.grid_height:
                    print(f'{center_x + x_walk * step_nr = }')
                    self.terrain[int(center_x + x_walk * step_nr), int(center_y + y_walk * step_nr)] = self.TERRAIN_TYPES['CLIFF']
                if 0 <= center_x - x_walk * step_nr < self.grid_width and 0 <= center_y - y_walk * step_nr < self.grid_height:
                    self.terrain[int(center_x - x_walk * step_nr), int(center_y - y_walk * step_nr)] = self.TERRAIN_TYPES['CLIFF']

    def get_random_coordinate(self):
        rand_x = np.random.randint(0, self.grid_width)
        rand_y = np.random.randint(0, self.grid_height)
        return rand_x, rand_y


if __name__ == '__main__':
    map = Map()