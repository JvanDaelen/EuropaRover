import numpy as np
import gymnasium as gym
from gymnasium import spaces

from map_generator import Map

class EuropaRover(gym.Env):
    #Direction constants
    n_actions = 4
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    #Rewards
    REACH_END_REWARD = 10
    EXISTS_REWARD = -0.5
    DIED_REWARD = -10
    CLOSER_REWARD = .9

    def __init__(self):
        super().__init__()
        self.map = Map()

        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = gym.spaces.Dict(
        spaces={
            # "grid": gym.spaces.Box(low=0, high=4, shape=(20,20), dtype=np.int32)
            "position": gym.spaces.Box(low=0, high=10, shape=(2,), dtype=np.int32)
            # "rover_x": gym.spaces.Box(low=(-self.grid_size[0]+1), high=(self.grid_size[0]-1), shape=(1,), dtype=np.int32),
            # "rover_y": gym.spaces.Box(low=(-self.grid_size[1]+1), high=(self.grid_size[1]-1), shape=(1,), dtype=np.int32),
        })

    def step(self, action):
        if action == self.UP:
            terrain_type = self.map.move_rover( 0,  1)
        elif action == self.DOWN:
            terrain_type = self.map.move_rover( 0, -1)
        elif action == self.LEFT:
            terrain_type = self.map.move_rover(-1,  0)
        elif action == self.RIGHT:
            terrain_type = self.map.move_rover( 1,  0)

        reward = self.EXISTS_REWARD
        done = False
        if terrain_type == self.map.TERRAIN_TYPES['CLIFF'] or terrain_type is False:
            reward += self.DIED_REWARD
            done = True
        elif terrain_type == self.map.TERRAIN_TYPES['END']:
            reward += self.REACH_END_REWARD
            done = True
        new_dist = np.sqrt((self.map.current_x - self.map.end_x)**2 + (self.map.current_y - self.map.end_y)**2)
        if new_dist < self.distance:
            reward += self.CLOSER_REWARD
        self.distance = new_dist

        return self._get_obs(), reward, done, {}, {}
    
    def reset(self, seed = None, options = None):
        if seed:
            np.random.seed(seed)
        self.map.reset()
        self.distance = np.sqrt((self.map.current_x - self.map.end_x)**2 + (self.map.current_y - self.map.end_y)**2)
        return self._get_obs(), {}

    def hard_reset(self, seed = None):
        self.map = Map(seed = seed)
        self.distance = np.sqrt((self.map.current_x - self.map.end_x)**2 + (self.map.current_y - self.map.end_y)**2)

    def render(self):
        return self.map.terrain

    def _get_obs(self):
        # return {"grid": self.map.terrain}
        return {"position": np.array([[self.map.current_x, self.map.current_y]], dtype=float)}