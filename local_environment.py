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
from support_functions import save_file_path
import csv


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
    # EMPTY = 0
    # OBJECT = 2
    # PASSED = 1
    # TARGET = 3
    # CURRENT = 4

    #Grid labels set 2
    EMPTY = 2
    OBJECT = -1
    PASSED = 1 #EMPTY #
    TARGET = 3
    CURRENT = 0

    #Rewards
    # REACH_END_REWARD = 2
    # EXISTS_REWARD = 0.5
    # DIED_REWARD = -3
    # DIRECTION_REWARD = 0.5

    # REACH_END_REWARD = 2.
    # EXISTS_REWARD = 1.
    # DIED_REWARD = 0.
    # DIRECTION_REWARD = 0.12501

    REACH_END_REWARD = 1.
    EXISTS_REWARD = 0.0
    DIED_REWARD = -1.0
    DIRECTION_REWARD = 0.125

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
        self.terrain = np.ones(self.grid_size) * self.EMPTY
        # print("reset grid")
        # print(f"{self.terrain[0,0] :>4.1f},{self.terrain[0,1] :>4.1f},{self.terrain[0,2] :>4.1f}")
        # print(f"{self.terrain[1,0] :>4.1f},{self.terrain[1,1] :>4.1f},{self.terrain[1,2] :>4.1f}")
        # print(f"{self.terrain[2,0] :>4.1f},{self.terrain[2,1] :>4.1f},{self.terrain[2,2] :>4.1f}")

        # add the current position value to the center
        self.terrain[1, 1] = self.CURRENT
        if 3 in self.terrain.astype(int):
            print(f"{self.terrain[0,0] :>4.1f},{self.terrain[0,1] :>4.1f},{self.terrain[0,2] :>4.1f}")
            print(f"{self.terrain[1,0] :>4.1f},{self.terrain[1,1] :>4.1f},{self.terrain[1,2] :>4.1f}")
            print(f"{self.terrain[2,0] :>4.1f},{self.terrain[2,1] :>4.1f},{self.terrain[2,2] :>4.1f}")
            raise RuntimeError('Random target in terrain')

        target_assigned = False
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                if(x, y) != (1, 1):
                    epsilon = np.random.random()
                    if epsilon < 0.4:
                        self.terrain[x, y] = self.OBJECT
                        if epsilon < 0.1 and not target_assigned:
                            target_assigned = True
                            self.terrain[x, y] = self.TARGET 
                            direction_x = x - 1
                            direction_y = y - 1
                    else:
                        epsilon = np.random.random()
                        if epsilon < 0.4:
                            self.terrain[x, y] = self.PASSED
        
        if all([
            self.terrain[1, 0] == self.OBJECT,
            self.terrain[0, 1] == self.OBJECT,
            self.terrain[2, 1] == self.OBJECT,
            self.terrain[1, 2] == self.OBJECT
        ]):
            self.generate_terrain()
        else:
            # if not target_assigned:
            # print("random direction check grid")
            # print(f"{self.terrain[0,0] :>4.1f},{self.terrain[0,1] :>4.1f},{self.terrain[0,2] :>4.1f}")
            # print(f"{self.terrain[1,0] :>4.1f},{self.terrain[1,1] :>4.1f},{self.terrain[1,2] :>4.1f}")
            # print(f"{self.terrain[2,0] :>4.1f},{self.terrain[2,1] :>4.1f},{self.terrain[2,2] :>4.1f}")
            if 3 not in self.terrain.astype(int):
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
        dir_inverse = False
        if action == self.UP:
            step = (-1, 0) # move up (+1 in y direction)
            # dir_inverse = True
        elif action == self.DOWN:
            step = (1, 0) # move down (-1 in y direction)
            # dir_inverse = True
        elif action == self.LEFT:
            step = (0, -1) # move left (-1 in x direction)
        elif action == self.RIGHT:
            step = (0, 1) # move right (+1 in x direction)
        else:
            raise ValueError(f"{action = } is not part of the action space")
        
        # Return the appropriate reward
        cell_value = self.terrain[1 + step[0], 1 + step[1]]
        if cell_value == self.OBJECT:
            reward = self.DIED_REWARD
        elif cell_value == self.TARGET:
            reward = self.REACH_END_REWARD
        else:
            reward = self.EXISTS_REWARD
        # Check if in right direction
        step_unit = np.array([step[0], step[1]]) / np.linalg.norm(step)
        dot = np.dot(step, self.direction)
        # if self.direction[0] * self.direction[1] < 0 or dir_inverse:
        #     dot = -dot
        # if dot >= 0:
        if np.arccos(dot) <= np.pi/2:
            reward += self.DIRECTION_REWARD
        else:
            reward -= self.DIRECTION_REWARD * 1.1

        if reward == 0.8625:
            print(f"Error case: {cell_value = }, {step = }, {action = }, {self.direction = }, {np.arccos(dot) = }")
            print(f"{self.terrain[0,0] :>4.1f},{self.terrain[0,1] :>4.1f},{self.terrain[0,2] :>4.1f}")
            print(f"{self.terrain[1,0] :>4.1f},{self.terrain[1,1] :>4.1f},{self.terrain[1,2] :>4.1f}")
            print(f"{self.terrain[2,0] :>4.1f},{self.terrain[2,1] :>4.1f},{self.terrain[2,2] :>4.1f}")
        # else:
        #     print(f"Norma case: {cell_value = }, {step = }, {action = }, {self.direction = }, {np.arccos(dot) = }")

        self.terrain[1 + step[1], 1 + step[0]] = self.CURRENT
        self.terrain[1, 1] = self.PASSED

        
        if action == self.UP:
            self.terrain[0,:] = self.terrain[1,:]
            self.terrain[1,:] = self.terrain[2,:]
            self.terrain[2,:] = self.EMPTY
            target_assigned = self.TARGET in self.terrain
            # print(f"{target_assigned}")
            if target_assigned:
                direction_x, direction_y = np.where(self.terrain == self.TARGET)
                direction_x = direction_x[0]
                direction_y = direction_y[0]
                # print(f"{direction_x = }, {direction_y = }")
            for x in [2]:
                for y in [0, 1, 2]:
                    if(x, y) != (1, 1):
                        epsilon = np.random.random()
                        if epsilon < 0.4:
                            self.terrain[x, y] = self.OBJECT
                            if epsilon < 0.1 and not target_assigned:
                                target_assigned = True
                                self.terrain[x, y] = self.TARGET 
                                direction_x = x
                                direction_y = y
                    else:
                        epsilon = np.random.random()
                        if epsilon < 0.4:
                            self.terrain[x, y] = self.PASSED

        elif action == self.DOWN:
            self.terrain[2,:] = self.terrain[1,:]
            self.terrain[1,:] = self.terrain[0,:]
            self.terrain[0,:] = self.EMPTY
            target_assigned = self.TARGET in self.terrain
            if target_assigned:
                direction_x, direction_y = np.where(self.terrain == self.TARGET)
                direction_x = direction_x[0]
                direction_y = direction_y[0]
                # print(direction_x, direction_y)
            for x in [0]:
                for y in [0, 1, 2]:
                    if(x, y) != (1, 1):
                        epsilon = np.random.random()
                        if epsilon < 0.4:
                            self.terrain[x, y] = self.OBJECT
                            if epsilon < 0.1 and not target_assigned:
                                target_assigned = True
                                self.terrain[x, y] = self.TARGET 
                                direction_x = x
                                direction_y = y
                    else:
                        epsilon = np.random.random()
                        if epsilon < 0.4:
                            self.terrain[x, y] = self.PASSED
        elif action == self.LEFT:
            self.terrain[:,2] = self.terrain[:,1]
            self.terrain[:,1] = self.terrain[:,0]
            self.terrain[:,0] = self.EMPTY
            target_assigned = self.TARGET in self.terrain
            if target_assigned:
                direction_x, direction_y = np.where(self.terrain == self.TARGET)
                direction_x = direction_x[0]
                direction_y = direction_y[0]
                # print(direction_x, direction_y)
            for x in [0,1,2]:
                for y in [0]:
                    if(x, y) != (1, 1):
                        epsilon = np.random.random()
                        if epsilon < 0.4:
                            self.terrain[x, y] = self.OBJECT
                            if epsilon < 0.1 and not target_assigned:
                                target_assigned = True
                                self.terrain[x, y] = self.TARGET 
                                direction_x = x
                                direction_y = y
                    else:
                        epsilon = np.random.random()
                        if epsilon < 0.4:
                            self.terrain[x, y] = self.PASSED
        elif action == self.RIGHT:
            self.terrain[:,0] = self.terrain[:,1]
            self.terrain[:,1] = self.terrain[:,2]
            self.terrain[:,2] = self.EMPTY
            target_assigned = self.TARGET in self.terrain
            if target_assigned:
                direction_x, direction_y = np.where(self.terrain == self.TARGET)
                direction_x = direction_x[0]
                direction_y = direction_y[0]
                # print(direction_x, direction_y)
            for x in [0,1,2]:
                for y in [2]:
                    if(x, y) != (1, 1):
                        epsilon = np.random.random()
                        if epsilon < 0.4:
                            self.terrain[x, y] = self.OBJECT
                            if epsilon < 0.1 and not target_assigned:
                                target_assigned = True
                                self.terrain[x, y] = self.TARGET 
                                direction_x = x
                                direction_y = y
                    else:
                        epsilon = np.random.random()
                        if epsilon < 0.4:
                            self.terrain[x, y] = self.PASSED
        else:
            raise ValueError(f"{action = } is not part of the action space")
        
        if target_assigned:
            direction = np.array([direction_x, direction_y], dtype=np.float32)
            norm = np.linalg.norm(direction)
            if norm == 0:
                self.direction = direction
            else:
                self.direction = direction / np.linalg.norm(direction)
        
        done = True
        return  self.get_obs(from_step=True), reward, done, {}, {}

    def reset(self, seed = None, options = None) -> dict:
        self.generate_terrain()
        obs = self.get_obs()  
        return obs, dict()
    
    def render(self, mode: str = 'rgb_array'):
        """Render in console or rgb_array mode"""
        if mode == 'console':
            print(self.get_obs())
        elif mode == 'rgb_array':
            return self.terrain
        else:
            raise NotImplementedError(f"rendering mode {mode = } not implemented")
        pass
        
    def get_obs(self, from_step = False) -> dict:
            #return observation in the format of self.observation_space
            if int(self.terrain[1, 1]) != self.CURRENT and not from_step:
                print(self.terrain)
                print(self.direction)
                print(f"{from_step = }")
                raise ValueError('local env observation does not have current in the center')
            return {"grid": self.terrain, "direction": self.direction}
   
def test_manual(): 
    for i in range(10):
        env = Centered3x3Environment()
        # print(env.terrain)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.imshow(env.terrain, origin='lower', cmap='YlGn')
        # Loop over data dimensions and create text annotations.
        for i in range(3):
            for j in range(3):
                c = 'k'
                if env.terrain[i, j] > 1.5:
                    c = 'w'
                text = ax.text(j, i, env.terrain[i, j],
                            ha="center", va="center",fontsize=40,c=c)
        # action = np.random.randint(0, 3)
        # print(f"{action = }")
        # out = env.step(action)
        # reward = out[1]
        # print(f"{reward = }")
        # print(f"{out[0]['direction'] = }")
        # ax[1].imshow(out[0]['grid'], origin='lower', cmap='YlGn')
        # Loop over data dimensions and create text annotations.
        # for i in range(3):
        #     for j in range(3):
        #         text = ax[1].text(j, i, env.terrain[i, j],
        #                     ha="center", va="center")
        # print(reward)
        # grid = env.terrain
        # target_assigned = env.TARGET in grid
        # if target_assigned:
        #     tar_loc = np.where(grid == env.TARGET)
        #     print(f"{tar_loc[0][0]},{tar_loc[1][0]}")

        plt.savefig(save_file_path('MapTesting/target_distribution_test/', f'test_env_{i}', 'png'))
        plt.close('all')

def test_map_to_csv():
    file_path = save_file_path('MapTesting/all_distribution_test/', 'all_dist', 'csv')
    with open(file_path, 'w', newline="") as file:
        writer = csv.writer(file)
        row = ['cell_0x0', 'cell_0x1', 'cell_0x2', 'cell_1x0', 'cell_1x1', 'cell_1x2', 'cell_2x0', 'cell_2x1', 'cell_2x2', 'dir_x', 'dir_y']
        writer.writerow(row)
    for i in range(100_000):
        print(f"{i:<6}", end='\r')
        env = Centered3x3Environment()
        obs = env.get_obs()

        with open(file_path, 'a', newline="") as file:
            writer = csv.writer(file)
            row = [
                int(obs['grid'][0, 0]), int(obs['grid'][0, 1]), int(obs['grid'][0, 2]),
                int(obs['grid'][1, 0]), int(obs['grid'][1, 1]), int(obs['grid'][1, 2]),
                int(obs['grid'][2, 0]), int(obs['grid'][2, 1]), int(obs['grid'][2, 2]),
                obs['direction'][0], obs['direction'][1]
                ]
            writer.writerow(row)

def test_direction():
    for _ in range(20):
        env = Centered3x3Environment()
        obs = env.get_obs()
        print(f"{obs['grid'] = }")
        step = np.random.randint(0, 4)
        
        print(step, env.step(step))

def test_direction_reward():
    env = Centered3x3Environment()
    env.terrain = np.array([[2., 3., 2.],
                            [2., 0., 2.],
                            [2., 2., 2.]])
    print(env.terrain[0, 1])
    env.direction = np.array([-1., 0])
    print(env.step(env.UP))
    env.terrain = np.array([[2., 2., 2.],
                            [2., 0., 2.],
                            [2., 3., 2.]])
    env.direction = np.array([1., 0.])
    print(env.step(env.DOWN))
    env.terrain = np.array([[2., 2., 2.],
                            [2., 0., 3.],
                            [2., 2., 2.]])
    env.direction = np.array([0., 1.])
    print(env.step(env.RIGHT))
    env.terrain = np.array([[2., 2., 2.],
                            [3., 0., 2.],
                            [2., 2., 2.]])
    env.direction = np.array([0., -1.])
    print(env.step(env.LEFT))

if __name__ == "__main__":
    # np.random.seed(42)
    # main()
    # test_manual()
    # test_direction()
    # test_direction_reward()
    test_map_to_csv()