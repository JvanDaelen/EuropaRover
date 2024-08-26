### Code to generate a numpy array based map containing randomly generated objects
import numpy as np
import matplotlib.pyplot as plt
from support_functions import save_file_path

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
                 nr_cliffs = 4, 
                 cliff_length_max = 4,
                 cliff_length_min = 1,
                 seed = None
                 ) -> None:
        # make class vars
        if seed:
            self.seed = seed
            np.random.seed(seed)
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.nr_cliffs = nr_cliffs
        self.cliff_length_max = cliff_length_max
        self.cliff_length_min = cliff_length_min
        self.terrain = np.zeros((grid_width, grid_height))
        self._generate_cliffs()
        self._add_end_point()
        self._set_start_point()
    
    def _generate_cliffs(self):
        for cliff_nr in range(self.nr_cliffs):
            # Pick random point
            center_x, center_y = self.get_random_coordinate()
            # self.terrain[center_x, center_y] = self.TERRAIN_TYPES['CLIFF']

            # Pick random direction (between + and - 90 to use cos and sin)
            direction_deg = np.random.rand() * 180 - 90
            # pick random length in range
            random_length = np.random.randint(self.cliff_length_min, self.cliff_length_max + 1) 
            # Walk length/2 in both + and - direction
            x_walk = np.cos(np.deg2rad(direction_deg))
            y_walk = np.sin(np.deg2rad(direction_deg))
            for step_nr in range(int(np.ceil(random_length/2)) + 1):
                if 0 <= center_x + x_walk * step_nr < self.grid_width and 0 <= center_y + y_walk * step_nr < self.grid_height:
                    self.terrain[int(center_x + x_walk * step_nr), int(center_y + y_walk * step_nr)] = self.TERRAIN_TYPES['CLIFF']
                if 0 <= center_x - x_walk * step_nr < self.grid_width and 0 <= center_y - y_walk * step_nr < self.grid_height:
                    self.terrain[int(center_x - x_walk * step_nr), int(center_y - y_walk * step_nr)] = self.TERRAIN_TYPES['CLIFF']

    def _add_end_point(self):
        rand_x, rand_y = self.get_random_coordinate()
        if self.terrain[rand_x, rand_y] == self.TERRAIN_TYPES['EMPTY']:
            self.end_x = rand_x
            self.end_y = rand_y
            self.terrain[rand_x, rand_y] = self.TERRAIN_TYPES['END']
        else:
            self._add_end_point()

    def _set_start_point(self):
        rand_x, rand_y = self.get_random_coordinate()
        if self.terrain[rand_x, rand_y] == self.TERRAIN_TYPES['EMPTY']:
            if np.sqrt((rand_x - self.end_x)**2 + (rand_y - self.end_y)**2) > self.grid_width / 2.5:
                self.start_x = rand_x
                self.start_y = rand_y
                self.current_x = rand_x
                self.current_y = rand_y
                self.terrain[rand_x, rand_y] = self.TERRAIN_TYPES['CURRENT']
            else:
                self._set_start_point()
        else:
            self._set_start_point()
    
    def reset(self):
        self.terrain[self.terrain == self.TERRAIN_TYPES['PASSED']] = self.TERRAIN_TYPES['EMPTY']
        self.terrain[self.terrain == self.TERRAIN_TYPES['CURRENT']] = self.TERRAIN_TYPES['EMPTY']
        self.terrain[self.start_x, self.start_y] = self.TERRAIN_TYPES['CURRENT']
        self.current_x = self.start_x
        self.current_y = self.start_y


    def move_rover(self, delta_x, delta_y):
        new_x = self.current_x + delta_x
        new_y = self.current_y + delta_y
        if 0 <= new_x < self.grid_width and 0 <= new_y < self.grid_height:
            terrain_type_at_new_location = self.terrain[new_x, new_y]
            
            if terrain_type_at_new_location != self.TERRAIN_TYPES['CLIFF'] and terrain_type_at_new_location != self.TERRAIN_TYPES['END']:
                self.terrain[new_x, new_y] = self.TERRAIN_TYPES['CURRENT']
                self.terrain[self.current_x, self.current_y] = self.TERRAIN_TYPES['EMPTY']
                self.current_x = new_x
                self.current_y = new_y
        else:
            return False
        return terrain_type_at_new_location
    
    def get_random_coordinate(self):
        rand_x = np.random.randint(0, self.grid_width)
        rand_y = np.random.randint(0, self.grid_height)
        return rand_x, rand_y

def test_map_generation(seed = None):
    for seed in range(20):
        map = Map(seed = seed)
        fig, ax = plt.subplots(figsize=(6,6))
        plt.imshow(map.terrain, cmap="tab10", vmin = -0.25, vmax = 4.75)
        # plt.axis('off')
        plt.colorbar()
        plt.grid(True)
        save_path = save_file_path('figures/terrain-fixed-seed-test', f'terrain_test_{map.grid_width}x{map.grid_height}_s{seed}', 'png')
        plt.savefig(save_path)

def test_map_reset():
    map = Map()
    for i in range(20):
        dx = np.random.randint(-1, 2)
        dy = np.random.randint(-1, 2)
        fig, ax = plt.subplots(figsize=(6,6))
        plt.imshow(map.terrain, cmap="tab10", vmin = -0.25, vmax = 4.75)
        # plt.axis('off')
        plt.colorbar()
        plt.grid(True)
        ter_type = map.move_rover(dx, dy)
        save_path = save_file_path('figures/reset_test', 'reset_test', 'png')
        plt.savefig(save_path)
        if ter_type != Map().TERRAIN_TYPES['EMPTY'] or ter_type is False:
            map.reset()

if __name__ == '__main__':
    test_map_generation()