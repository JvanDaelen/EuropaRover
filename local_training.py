import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from local_environment import Centered3x3Environment
from support_functions import save_file_path

class EvalueNetwork(nn.Module):
    """Parametrized Policy Network."""

    def __init__(self):
        """
        TODO
        """
        super().__init__()

        # Hyperparameters
        nr_input_channels = 1
        nr_kernels_conv_layer_1 = 100
        sz_kernels_conv_layer_1 = 2
        nr_kernels_conv_layer_2 = 50
        sz_kernels_conv_layer_2 = 2
        nr_out_dense_layer = 128

        # Dependent hyper parameters
        sz_hor_output_conv_layer_1 = 3 - sz_kernels_conv_layer_1 + 1
        nr_outputs_conv_layer_2 = nr_kernels_conv_layer_2 * \
                                (sz_hor_output_conv_layer_1 - sz_kernels_conv_layer_2 + 1)**2
        print(nr_outputs_conv_layer_2)

        # Shared Network
        self.evalue_net = nn.Sequential(
            nn.Conv2d(1, nr_kernels_conv_layer_1, sz_kernels_conv_layer_1, dtype=torch.float,),
            nn.BatchNorm1d(3 - sz_kernels_conv_layer_1 + 1),
            nn.ReLU(),
            nn.Conv2d(nr_kernels_conv_layer_1, nr_kernels_conv_layer_2, sz_kernels_conv_layer_2, dtype=torch.float,),
            nn.BatchNorm1d(sz_hor_output_conv_layer_1 - sz_kernels_conv_layer_2 + 1),
            nn.ReLU(),
            nn.Flatten(0, -1),
            nn.Linear(nr_outputs_conv_layer_2, nr_out_dense_layer),
            nn.ReLU(),
            nn.Linear(nr_out_dense_layer, 4),
            nn.ReLU()
        )

        self.direction_net = nn.Sequential(
            nn.Linear(6, 4),
            nn.ReLU()
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        TODO
        """
        grid = obs['grid']
        direction_tensor = torch.from_numpy(obs['direction'])
        grid = grid.reshape((3, 3, 1))
        grid = torch.from_numpy(grid)
        grid = grid.permute(2, 0, 1)
        evalue_net_output = self.evalue_net(grid.float())
        direction_net_input = torch.cat([evalue_net_output, direction_tensor])
        direction_net_output = self.direction_net(direction_net_input)
        return direction_net_output


class Trainer():
    def __init__(self):
        """Initializes an agent that learns a policy via REINFORCE algorithm [1]
        to solve the task at hand (Inverted Pendulum v4).

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """

        # Hyperparameters
        self.learning_rate = 1e-3  # Learning rate for policy optimization
        self.gamma = 0.99  # Discount factor

        self.env = Centered3x3Environment()
        self.net = EvalueNetwork()
        # self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.learning_rate)
    
    def train(self, max_eps, epsilon = 0):
        self.rewards = []
        self.actions = []
        self.states = []
        self.losses = []
        self.ep_nrs = []
        self.colors = []

        for ep in range(max_eps):
            if ep > 50e+3:
                epsilon = 0
            obs, _ = self.env.reset()
            action_tensor = self.net.forward(obs)
            gamma = np.random.random()
            if gamma < epsilon or torch.max(action_tensor) < 0.01:
                action = np.random.randint(0, 4)
                self.colors.append('r')
            else:
                action = int(torch.argmax(action_tensor))
                self.colors.append('b')
            old_q = action_tensor[action]
            new_state, reward, _, _, _ = self.env.step(action)
            new_prediction = self.net.forward(new_state)
            new_q = new_prediction[int(torch.argmax(new_prediction))]
            gamma = 0.1
            loss = (reward + gamma * new_q - old_q)**2
            # loss = (old_q - reward)**2
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.ep_nrs.append(ep)
            self.losses.append(loss.detach().numpy())
            self.rewards.append(reward)
            self.actions.append(action)
            self.states.append(obs['grid'])

def test_net():
    env = Centered3x3Environment()
    net = EvalueNetwork()
    print(f"{env.get_obs()['grid'] = }")
    obs = env.get_obs()
    print(f"{net.forward(obs) = }")

def take_action():
    env = Centered3x3Environment()
    net = EvalueNetwork()
    obs = env.get_obs()
    action_tensor = net.forward(obs)
    action = int(torch.argmax(action_tensor))
    old_q = action_tensor[action]
    new_state, reward, _, _, _ = env.step(action)
    print(f"{action_tensor = }")
    print(f"{torch.argmax(action_tensor) = }")
    print(f"{env.step(action) = }")
    print(f"{new_state = }")
    new_prediction = net.forward(new_state)
    new_q = new_prediction[int(torch.argmax(new_prediction))]
    gamma = 0.9
    loss = (reward + gamma * new_q - old_q)**2
    print(f"{old_q = }")
    print(f"{new_q = }")
    print(f"{loss = }")

def test_training():
    trainer = Trainer()
    max_eps = 200000
    epsilon = 0.01
    trainer.train(max_eps, epsilon)
    plt.figure(figsize=(6,6), layout="constrained")
    plt.subplot(311)
    plt.title(f'C3x3, nr episodes = {max_eps}, epsilon = {epsilon :.2f}, lr=1e-3')
    plt.ylabel(r'Losses $(r_t + \gamma * q_{t+1} - q_t)^2$')
    plt.plot(trainer.losses)
    plt.subplot(312)
    plt.ylabel('Rewards')
    plt.scatter(trainer.ep_nrs, trainer.rewards, 2, alpha=0.5, c='gray')
    window = 15
    average_x = []
    average_data = []
    for ind in range(len(trainer.rewards) - window + 1):
        average_x.append(np.mean(trainer.ep_nrs[ind:ind+window]) + window)
        average_data.append(np.mean(trainer.rewards[ind:ind+window]))
    plt.plot(average_data)
    plt.subplot(313)
    plt.ylabel('Action')
    plt.scatter(trainer.ep_nrs, trainer.actions, 2, c=trainer.colors)
    plt.xlabel('Episode')
    # plt.show()
    plt.savefig(save_file_path(f'results/LocalTraining/centered_directional_3x3_greedy_{str(epsilon).replace('.','_')}_standard_loss_rewards_1E_1_5RE_0_2D_SGD_long_dyn_epsilon', f'training_loss', 'png'))

if __name__ == '__main__':
    for i in range(5):
        test_training()
    # test_net()