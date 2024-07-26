# Tutorial links
# training: https://gymnasium.farama.org/tutorials/training_agents/reinforce_invpend_gym_v26/#sphx-glr-tutorials-training-agents-reinforce-invpend-gym-v26-py
# game environment: https://github.com/guszejnovdavid/custom_game_reinforcement_learning/blob/main/custom_game_reinforcement_learning.ipynb
# additional: https://blog.paperspace.com/getting-started-with-openai-gym/


import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import torch


class Centered3x3Environment(gym.Env):
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
    REACH_END_REWARD = 2
    EXISTS_REWARD = 0.5
    DIED_REWARD = -3
    DIRECTION_REWARD = 0.5

    def __init__(self) -> None:
        super().__init__()
        self.stepnum = 0
        self.grid_size = (3, 3)
        self.generate_terrain()
        # self.generate_starting_position()

        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = gym.spaces.Dict(
            spaces={
                "grid": gym.spaces.Box(low=0, high=4, shape=self.grid_size, dtype=np.int32),
                "direction": gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
            })
    
    def generate_terrain(self) -> None:
        # initialize as empty grid
        self.terrain = np.zeros(self.grid_size)

        # add the current position value to the center
        self.terrain[1, 1] = self.CURRENT

        target_assigned = False
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                if(x, y) != (1, 1):
                    epsilon = np.random.random()
                    if epsilon < 0.4:
                        self.terrain[x, y] = self.OBJECT
                        if epsilon < 0.1 and not target_assigned:
                            self.terrain[x, y] = self.TARGET 
                            direction_x = x
                            direction_y = y
        
        if all([
            self.terrain[1, 0] == self.OBJECT,
            self.terrain[0, 1] == self.OBJECT,
            self.terrain[2, 1] == self.OBJECT,
            self.terrain[1, 2] == self.OBJECT
        ]):
            self.generate_terrain()

        if not target_assigned:
            direction_x = np.random.randint(-10, 11)
            direction_y = np.random.randint(-10, 11)
        direction = np.array([direction_x, direction_y], dtype=np.float32)
        norm = np.linalg.norm(direction)
        if norm == 0:
            self.direction = direction
        else:
            self.direction = direction / np.linalg.norm(direction)

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
        cell_value = self.terrain[1 + step[1], 1 + step[0]]
        if cell_value == self.OBJECT:
            reward = self.DIED_REWARD
        elif cell_value == self.TARGET:
            reward = self.REACH_END_REWARD
        else:
            reward = self.EXISTS_REWARD

        self.terrain[1 + step[1], 1 + step[0]] = self.CURRENT
        self.terrain[1, 1] = self.PASSED

        # Check if in right direction
        dot = np.dot(step, self.direction)
        if self.direction[0] * self.direction[1] < 0:
            dot = -dot
        if dot > 0:
            reward += self.DIRECTION_REWARD
        else:
            reward -= self.DIRECTION_REWARD * 1.1
        
        done = True
        return  self._get_obs(), reward, done, {}, {}

    def reset(self, seed = None, options = None) -> dict:
        self.generate_terrain()
        obs = self._get_obs()  
        return obs, dict()
    
    def render(self, mode: str = 'rgb_array'):
        """Render in console or rgb_array mode"""
        if mode == 'console':
            print(self._get_obs())
        elif mode == 'rgb_array':
            return self.terrain
        else:
            raise NotImplementedError(f"rendering mode {mode = } not implemented")
        pass
        
    def _get_obs(self) -> dict:
            #return observation in the format of self.observation_space
            return {"grid": self.terrain, "direction": self.direction}
   
def test_manual():
    import matplotlib.animation as animation
    from time import sleep   
    env = Centered3x3Environment()
    print(env.terrain)
    action = np.random.randint(0, 3)
    print(f"{action = }")
    out = env.step(action)
    reward = out[1]
    print(f"{reward = }")
    fig, ax = plt.subplots()
    ax.imshow(out[0]['grid'], origin='lower', cmap='YlGn')
    # Loop over data dimensions and create text annotations.
    for i in range(3):
        for j in range(3):
            text = ax.text(j, i, env.terrain[i, j],
                        ha="center", va="center")
    plt.show()

def test_direction():
    for _ in range(10):
        env = Centered3x3Environment()
        obs = env._get_obs()
        print(f"{obs['direction'] = }")
        step = np.random.randint(0, 4)
        env.step(np.random.randint(0, 4))
        print(step, env.dot)

if __name__ == "__main__":
    # np.random.seed(42)
    # main()
    test_manual()
    # test_direction()