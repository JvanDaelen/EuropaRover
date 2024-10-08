import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from local_environment import Centered3x3Environment
from support_functions import save_file_path
from global_environment import GlobalWander
import csv

class EvalueNetwork(nn.Module):
    """Parametrized Policy Network."""

    def __init__(self):
        """
        TODO
        """
        super().__init__()

        # Hyperparameters
        nr_input_channels = 1

        # net2
        # nr_kernels_conv_layer_1 = 20
        # sz_kernels_conv_layer_1 = 2
        # nr_kernels_conv_layer_2 = 100
        # sz_kernels_conv_layer_2 = 2
        # nr_out_dense_layer = 128
        
        # net3
        # nr_kernels_conv_layer_1 = 4
        # sz_kernels_conv_layer_1 = 1
        # nr_kernels_conv_layer_2 = 100
        # sz_kernels_conv_layer_2 = 3
        # nr_out_dense_layer = 128

        # net4
        # nr_kernels_conv_layer_1 = 100
        # sz_kernels_conv_layer_1 = 3
        # nr_kernels_conv_layer_2 = 100
        # sz_kernels_conv_layer_2 = 3
        # nr_out_dense_layer = 128

        # net5
        # nr_kernels_conv_layer_1 = 100
        # sz_kernels_conv_layer_1 = 3
        # nr_kernels_conv_layer_2 = 0
        # sz_kernels_conv_layer_2 = 0
        # nr_out_dense_layer = 128
        # nr_out_dense_layer2 = 60

        # net6
        # nr_kernels_conv_layer_1 = 20
        # sz_kernels_conv_layer_1 = 2
        # nr_kernels_conv_layer_2 = 20
        # sz_kernels_conv_layer_2 = 2
        # nr_out_dense_layer = 128
        # nr_out_dense_layer2 = 60

        # net7
        # nr_kernels_conv_layer_1 = 20
        # sz_kernels_conv_layer_1 = 2
        # nr_kernels_conv_layer_2 = 10
        # sz_kernels_conv_layer_2 = 2
        # nr_out_dense_layer = 40
        # nr_out_dense_layer2 = 20

        # # Dependent hyper parameters
        # sz_hor_output_conv_layer_1 = 3 - sz_kernels_conv_layer_1 + 1
        # nr_outputs_conv_layer_1 = sz_hor_output_conv_layer_1 * nr_kernels_conv_layer_1
        # # nr_outputs_conv_layer_2 = nr_outputs_conv_layer_1
        # nr_outputs_conv_layer_2 = nr_kernels_conv_layer_2 * \
        #                         (sz_hor_output_conv_layer_1 - sz_kernels_conv_layer_2 + 1)**2
        # print(nr_outputs_conv_layer_2)

        # # Shared Network
        # self.evalue_net = nn.Sequential(
        #     nn.Conv2d(1, nr_kernels_conv_layer_1, sz_kernels_conv_layer_1, dtype=torch.float,),
        #     nn.BatchNorm1d(3 - sz_kernels_conv_layer_1 + 1),
        #     nn.ReLU(),
        #     nn.Conv2d(nr_kernels_conv_layer_1, nr_kernels_conv_layer_2, sz_kernels_conv_layer_2, dtype=torch.float,),
        #     nn.BatchNorm1d(sz_hor_output_conv_layer_1 - sz_kernels_conv_layer_2 + 1),
        #     nn.ReLU(),
        #     nn.Flatten(0, -1),
        #     # nn.Linear(nr_outputs_conv_layer_2, nr_out_dense_layer),
        #     # nn.ReLU(),
        #     # nn.Linear(nr_out_dense_layer, 4),
        #     # nn.ReLU(),
        #     # nn.Softmax(-1)
        # )

        # self.direction_net = nn.Sequential(
        #     nn.Linear(nr_outputs_conv_layer_2 + 2, nr_out_dense_layer),
        #     nn.ReLU(),
        #     nn.Linear(nr_out_dense_layer, nr_out_dense_layer2),
        #     nn.ReLU(),
        #     nn.Linear(nr_out_dense_layer2, 4),
        #     nn.ReLU()
        #     # nn.Linear(nr_out_dense_layer + 2, 6),
        #     # nn.ReLU(),
        #     # nn.Linear(6, 4),
        #     # nn.ReLU()
        # )

        # Net 8
        # nr_kernels_conv_layer_1 = 5
        # sz_kernels_conv_layer_1 = 1
        # nr_kernels_conv_layer_2 = 10
        # sz_kernels_conv_layer_2 = 3
        # nr_out_dense_layer = 10
        # nr_out_dense_layer2 = nr_out_dense_layer

        # Net 9
        nr_kernels_conv_layer_1 = 3
        sz_kernels_conv_layer_1 = 1
        nr_kernels_conv_layer_2 = 10
        sz_kernels_conv_layer_2 = 3
        nr_out_dense_layer = 10
        nr_out_dense_layer2 = nr_out_dense_layer

        # Dependent hyper parameters
        sz_hor_output_conv_layer_1 = 3 - sz_kernels_conv_layer_1 + 1
        nr_outputs_conv_layer_1 = sz_hor_output_conv_layer_1 * nr_kernels_conv_layer_1
        # nr_outputs_conv_layer_2 = nr_outputs_conv_layer_1
        nr_outputs_conv_layer_2 = nr_kernels_conv_layer_2 * \
                                (sz_hor_output_conv_layer_1 - sz_kernels_conv_layer_2 + 1)**2
        print(nr_outputs_conv_layer_2)

        # Shared Network
        self.evalue_net = nn.Sequential(
            nn.Conv2d(1, nr_kernels_conv_layer_1, sz_kernels_conv_layer_1, dtype=torch.float,),
            # nn.BatchNorm1d(3 - sz_kernels_conv_layer_1 + 1), # Deactivate for net 10
            # nn.ReLU(), # Deactivate for net 10
            nn.Conv2d(nr_kernels_conv_layer_1, nr_kernels_conv_layer_2, sz_kernels_conv_layer_2, dtype=torch.float,),
            # nn.BatchNorm1d(sz_hor_output_conv_layer_1 - sz_kernels_conv_layer_2 + 1),
            # nn.Sigmoid(), # Sigmoid for net 10 relu nor net <10
            nn.ReLU(),
            nn.Flatten(0, -1),
            nn.Linear(nr_outputs_conv_layer_2, nr_out_dense_layer),
            # nn.ReLU(),
            # nn.Linear(nr_out_dense_layer, 4),
            # nn.ReLU(),
            # nn.Softmax(-1)
        )

        self.direction_net = nn.Sequential(
            # nn.Linear(nr_outputs_conv_layer_2 + 2, nr_out_dense_layer),
            # nn.LeakyReLU(), # Sigmoid for net10. leaky relu for net 11
            # nn.Linear(nr_out_dense_layer, nr_out_dense_layer2),
            # nn.ReLU(),
            nn.Linear(nr_out_dense_layer2 + 2, 4),
            # nn.LeakyReLU() # Sigmoid for net10. leaky relu for net 11
            # nn.Linear(nr_out_dense_layer + 2, 6),
            # nn.ReLU(),
            # nn.Linear(6, 4),
            # nn.ReLU()
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
        grid_float = grid.float()
        evalue_net_output = self.evalue_net(grid_float)
        if abs(torch.max(evalue_net_output)) < 0.01:
                # action = np.random.randint(0, 4)
                print('no output eval')
                print(grid)
                print(evalue_net_output)
        direction_net_input = torch.cat([evalue_net_output, direction_tensor])
        direction_net_output = self.direction_net(direction_net_input)
        # if torch.max(direction_net_output) < 0.01:
        #         # action = np.random.randint(0, 4)
        #         print('no output eval')
        #         print(grid)
        #         print(direction_tensor)
        #         print(direction_net_output)
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
        self.learning_rate = 1e-2  # Learning rate for policy optimization

        self.env = Centered3x3Environment()
        self.net = EvalueNetwork()
        # self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.learning_rate)
        # self.optimizer = torch.optim.Adadelta(self.net.parameters())
    
    def train(self, max_eps, epsilon = 0, initial_ep_nr = 0):
        self.rewards = []
        self.actions = []
        self.states = []
        self.losses = []
        self.ep_nrs = []
        self.colors = []

        for ep in range(max_eps):
            if ep + initial_ep_nr > 100e+3: # pepsi
                epsilon = 0
            obs, _ = self.env.reset()
            
            action_tensor = self.net.forward(obs)
            gamma = np.random.random()
            if gamma < epsilon:
                action = np.random.randint(0, 4)
                self.colors.append('r')
            elif torch.max(torch.abs(action_tensor)) < 0.01:
            #     # action = np.random.randint(0, 4)
                action = int(torch.argmax(action_tensor))
                print(action_tensor)
                self.colors.append('g')
            else:
                action = int(torch.argmax(action_tensor))
                self.colors.append('b')
            old_q = action_tensor[action]
            new_state, reward, _, _, _ = self.env.step(action)
            if reward == 0.8625:
                print(f"Error case: {action = }")
                print("obs grid")
                print(f"{obs['grid'][0,0] :>4.1f},{obs['grid'][0,1] :>4.1f},{obs['grid'][0,2] :>4.1f}")
                print(f"{obs['grid'][1,0] :>4.1f},{obs['grid'][1,1] :>4.1f},{obs['grid'][1,2] :>4.1f}")
                print(f"{obs['grid'][2,0] :>4.1f},{obs['grid'][2,1] :>4.1f},{obs['grid'][2,2] :>4.1f}")
                print("new state")
                obs = new_state
                print(f"{obs['grid'][0,0] :>4.1f},{obs['grid'][0,1] :>4.1f},{obs['grid'][0,2] :>4.1f}")
                print(f"{obs['grid'][1,0] :>4.1f},{obs['grid'][1,1] :>4.1f},{obs['grid'][1,2] :>4.1f}")
                print(f"{obs['grid'][2,0] :>4.1f},{obs['grid'][2,1] :>4.1f},{obs['grid'][2,2] :>4.1f}")
                raise RuntimeError('faulty observation')
            
            new_prediction = self.net.forward(new_state)
            new_q = new_prediction[int(torch.argmax(new_prediction))]
            gamma = 0.9
            # loss = (reward + gamma * new_q - old_q)**2
            loss = (reward - old_q)**2
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
    gamma = 0.5
    # loss = (reward + gamma * new_q - old_q)**2
    loss = (reward - old_q)**2
    print(f"{old_q = }")
    print(f"{new_q = }")
    print(f"{loss = }")

def test_training(training_nr = 0):
    epsilon = 0.0
    # REACH_END_REWARD = 1
    # EXISTS_REWARD = 0.5
    # DIED_REWARD = 0
    # DIRECTION_REWARD = 0.125
    id = 108
    test_folder = f'results/GlobalTesting/tr{id}-{training_nr}_gtlo_n1d0DR_1d0RE_0d0ER_0d125DiR_plt0d1p_lset2_SGD'
    # test_folder = f'results/GlobalTesting/tr{id}-{training_nr}_gtlo_0d0DR_2d0RE_1d0ER_0d125DiR_plt10p_lset2_SGD'
    trainer = Trainer()
    max_eps = 20_000
    tot_eps = 0
    eps_batch_size = int(max_eps / 10)
    memory = {
        "rewards": [],
        "actions": [],
        "states": [],
        "losses": [],
        "ep_nrs": [],
        "colors": [],
        "test_rewards": [],
        "test_eps_number": [],
        "test_mean_rewards": [],
        "test_mean_eps": []
    }
    while tot_eps < max_eps:
        print(tot_eps)
        trainer.train(eps_batch_size, epsilon)
        memory["rewards"].extend(trainer.rewards)
        memory["actions"].extend(trainer.actions)
        memory["states"].extend(trainer.states)
        memory["losses"].extend(trainer.losses)
        memory["ep_nrs"].extend(list(np.asarray(trainer.ep_nrs) + tot_eps))
        memory["colors"].extend(trainer.colors)
        tot_eps += eps_batch_size
        plt.figure(figsize=(6,6), layout="constrained")
        plt.subplot(311)
        plt.title(f'C3x3, nr episodes = {max_eps}, epsilon = {epsilon :.2f}, lr=1e-2')
        # plt.ylabel(r'Losses $(r_t + \gamma * q_{t+1} - q_t)^2$')
        plt.ylabel(r'Losses $(r_t - q_t)^2$')
        plt.scatter(memory["ep_nrs"], memory["losses"], 2, alpha=0.5, c=memory["colors"])
        window = 500
        average_x = []
        average_data = []
        for ind in range(len(memory["losses"]) - window + 1):
            average_x.append(np.mean(memory["ep_nrs"][ind:ind+window]) + window)
            average_data.append(np.mean(memory["losses"][ind:ind+window]))
        plt.plot(average_data, c='r')
        plt.subplot(312)
        plt.ylabel('Rewards')
        plt.scatter(memory["ep_nrs"], memory["rewards"], 2, alpha=0.5, c=memory["colors"])
        window = 500
        average_x = []
        average_data = []
        for ind in range(len(memory["rewards"]) - window + 1):
            average_x.append(np.mean(memory["ep_nrs"][ind:ind+window]) + window)
            average_data.append(np.mean(memory["rewards"][ind:ind+window]))
        plt.plot(average_data, c='r')
        plt.subplot(313)
        plt.ylabel('Action')
        plt.scatter(memory["ep_nrs"], memory["actions"], 2, c=memory["colors"])
        plt.xlabel('Episode')
        # plt.show()
        plt.savefig(save_file_path(test_folder, f'training_results', 'png'))

        # Global wander test
        mean_rewards = []
        for run_id in range(20):
            test_env = GlobalWander()
            obs = test_env.get_obs()
            done = False
            tot_reward = 0
            test_ep = 0
            while not done and test_ep < 200:
                test_ep += 1
                action_tensor = trainer.net.forward(obs)
                if torch.max(action_tensor) < 0.01:
                    action = np.random.randint(0, 4)
                else:
                    action = int(torch.argmax(action_tensor))
                obs, reward, done, _, _ = test_env.step(action)
                tot_reward += reward
            memory["test_rewards"].append(tot_reward)
            memory["test_eps_number"].append(tot_eps)
            mean_rewards.append(tot_reward)
            test_env.render_map()
            plt.savefig(save_file_path(test_folder + '/rendered_maps/', f'test_env_result_ep{tot_eps}_run{run_id}', 'png'))
            plt.close('all')

        memory["test_mean_rewards"].append(np.mean(mean_rewards))
        memory["test_mean_eps"].append(tot_eps)

        plt.figure(figsize=(6,2), layout="constrained")
        plt.subplot(111)
        plt.title(f'C3x3, nr episodes = {max_eps}, epsilon = {epsilon :.2f}, lr=1e-3')
        plt.ylabel(r'$Reward_{total}$')
        plt.plot(memory["test_mean_eps"], memory["test_mean_rewards"])
        plt.scatter(memory["test_eps_number"], memory["test_rewards"], 5, alpha=0.5, c='gray')
        plt.xlabel('Training Episode')
        # plt.show()
        plt.savefig(save_file_path(test_folder, f'test_rewards', 'png'))

    ## Save memory to csv
    with open(f'results/csvs/id{id}-training_run{training_nr}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for key, item in memory.items():
            if key == 'states':
                continue
            writer.writerow([key] + item)


        

            

if __name__ == '__main__':
    for i in range(5):
        test_training(i+1)
    # test_net()