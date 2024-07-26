### Code to generate a numpy array based map containing randomly generated objects
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
from support_functions import save_file_path


class GlobalWander(gym.Env):
    """
    Custom environment for simulating a rover avoiding obstacles in a 
    discretized grid world.
    """
    metadata = {'render.modes': ['console','rgb_array']}
    #Direction constants
    n_actions = 4 #3 possible steps each turn
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    #Grid label constants
    EMPTY = 0
    OBJECT = 2
    PASSED = 1
    TARGET = 3
    CURRENT = 4

    #Rewards
    REACH_END_REWARD = 1.5
    EXISTS_REWARD = 1
    DIED_REWARD = -1
    DIRECTION_REWARD = 0.2

    #Map Settings
    nr_cliffs = 10
    cliff_length_max = 10
    cliff_length_min = 2

    def __init__(self) -> None:
        super().__init__()
        self.stepnum = 0
        self.grid_size = (50, 50)
        self.generate_global_map()
        # self.generate_starting_position()

        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = gym.spaces.Dict(
            spaces={
                "grid": gym.spaces.Box(low=0, high=4, shape=self.grid_size, dtype=np.int32)
            })
        
        self._get_obs()
    
    def step(self, action):
        #Get direction from action
        step = (0, 0)
        if action == self.UP:
            step = (0, 1) # move up (+1 in y direction)
        elif action == self.DOWN:
            step = (0, -1) # move down (-1 in y direction)
        elif action == self.LEFT:
            step = (-1, 0) # move left (-1 in x direction)
        elif action == self.RIGHT:
            step = (1, 0) # move right (+1 in x direction)
        else:
            raise ValueError(f"{action = } is not part of the action space")
        
        # Return the appropriate reward
        cell_value = self.terrain[self.current_x + step[1], self.current_y + step[0]]
        if cell_value == self.OBJECT:
            reward = self.DIED_REWARD
            done = True
        elif cell_value == self.TARGET:
            reward = self.REACH_END_REWARD
            done = True
        else:
            reward = self.EXISTS_REWARD
            done = False

        self.terrain[self.current_x + step[1], self.current_y + step[0]] = self.CURRENT
        self.terrain[self.current_x, self.current_y] = self.PASSED
        self.current_x = self.current_x + step[1]
        self.current_y = self.current_y + step[0]

        # Check if in right direction
        dot = np.dot(step, self.direction)
        if self.direction[0] * self.direction[1] < 0:
            dot = -dot
        if dot > 0:
            reward += self.DIRECTION_REWARD
        else:
            reward -= self.DIRECTION_REWARD * 1.1
        
        return  self._get_obs(), reward, done, {}, {}
    
    def _get_obs(self):
        direction_x = self.target_x - self.current_x
        direction_y = self.target_y - self.current_y
        direction = np.array([direction_x, direction_y], dtype=np.float32)
        norm = np.linalg.norm(direction)
        if norm == 0:
            self.direction = direction
        else:
            self.direction = direction / np.linalg.norm(direction)
        local_terrain = self.terrain[self.current_x-1:self.current_x+2, self.current_y-1:self.current_y+2]
        return {"grid": local_terrain, "direction": direction}
        
    def generate_global_map(self):
        # initialize as empty grid
        self.terrain = np.zeros(self.grid_size)
        # Add border
        self.terrain[:,0] = self.OBJECT
        self.terrain[:,-1] = self.OBJECT
        self.terrain[0,:] = self.OBJECT
        self.terrain[-1,:] = self.OBJECT

        # Add other features
        self._generate_cliffs()
        self._add_end_point()
        self._set_start_point()
    
    def _generate_cliffs(self):
        for cliff_nr in range(self.nr_cliffs):
            # Pick random point
            center_x, center_y = self.get_random_coordinate()

            # Pick random direction (between + and - 90 to use cos and sin)
            direction_deg = np.random.rand() * 180 - 90
            # pick random length in range
            random_length = np.random.randint(self.cliff_length_min, self.cliff_length_max + 1) 
            # Walk length/2 in both + and - direction
            x_walk = np.cos(np.deg2rad(direction_deg))
            y_walk = np.sin(np.deg2rad(direction_deg))
            for step_nr in range(int(np.ceil(random_length/2)) + 1):
                if 0 <= center_x + x_walk * step_nr < self.grid_size[0] and 0 <= center_y + y_walk * step_nr < self.grid_size[1]:
                    self.terrain[int(center_x + x_walk * step_nr), int(center_y + y_walk * step_nr)] = self.OBJECT
                if 0 <= center_x - x_walk * step_nr < self.grid_size[0] and 0 <= center_y - y_walk * step_nr < self.grid_size[1]:
                    self.terrain[int(center_x - x_walk * step_nr), int(center_y - y_walk * step_nr)] = self.OBJECT

    def _add_end_point(self):
        rand_x, rand_y = self.get_random_coordinate()
        if self.terrain[rand_x, rand_y] == self.EMPTY:
            self.target_x = rand_x
            self.target_y = rand_y
            self.terrain[rand_x, rand_y] = self.TARGET
        else:
            self._add_end_point()

    def _set_start_point(self):
        rand_x, rand_y = self.get_random_coordinate()
        if self.terrain[rand_x, rand_y] == self.EMPTY:
            if np.sqrt((rand_x - self.target_x)**2 + (rand_y - self.target_y)**2) > self.grid_size[0] / 2.5:
                self.start_x = rand_x
                self.start_y = rand_y
                self.current_x = rand_x
                self.current_y = rand_y
                self.terrain[rand_x, rand_y] = self.CURRENT
            else:
                self._set_start_point()
        else:
            self._set_start_point()

    def render_map(self):
        plt.figure(figsize=(6,6), layout="constrained")
        # fig, ax = plt.subplots()
        plt.imshow(self.terrain, origin='lower', cmap='YlGn')
        # Loop over data dimensions and create text annotations.
        if self.grid_size[0] < 10:
            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    text = plt.text(j, i, self.terrain[i, j],
                                ha="center", va="center")
        plt.scatter(self.current_y, self.current_x, s=50, marker='s', label='current')
        plt.scatter(self.target_y, self.target_x, s=50, marker='s', label='target')
        plt.legend()
        # plt.show()

    def get_random_coordinate(self):
        rand_x = np.random.randint(0, self.grid_size[0])
        rand_y = np.random.randint(0, self.grid_size[0])
        return rand_x, rand_y


def test_map_generation():
    env = GlobalWander()
    env.render_map()

def test_get_obs():
    env = GlobalWander()
    obs = env._get_obs()
    env.render_map()
    fig, ax = plt.subplots()
    ax.imshow(obs['grid'], origin='lower', cmap='YlGn')
    # Loop over data dimensions and create text annotations.
    for i in range(3):
        for j in range(3):
            text = ax.text(j, i, obs['grid'][i, j],
                        ha="center", va="center")
    plt.show()

def test_step():
    env = GlobalWander()
    for s in range(4):
        for i in range(1):
            out = env.step(s)
            print(env.direction)
            print(out[1])
    env.render_map()
    


if __name__ == '__main__':
    # test_map_generation()    
    # np.random.seed(42)
    # main()
    # test_manual()
    # test_get_obs()
    test_step()