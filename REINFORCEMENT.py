from __future__ import annotations

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from support_functions import save_file_path

import gymnasium as gym

from environment import EuropaRover

plt.rcParams["figure.figsize"] = (10, 5)


class Policy_Network(nn.Module):
    def __init__(self, obs_space_dims: int, action_space_dims: int):
        super().__init__()

        nr_kernels_1 = 64
        nr_kernels_2 = 2*nr_kernels_1
        nr_kernels_3 = nr_kernels_2
        hidden_space = 100
        self.shared_net = nn.Sequential(
            # nn.Conv2d(1, nr_kernels_1, 2, dtype=torch.float),
            # nn.Sigmoid(),
            # nn.Conv2d(nr_kernels_1, nr_kernels_2, 2, dtype=torch.float),
            # nn.Sigmoid(),
            # nn.Conv2d(nr_kernels_2, nr_kernels_3, 2, stride=1, dtype=torch.float),
            # nn.Sigmoid(),
            # nn.Flatten(0, -1),
            # nn.Linear(nr_kernels_3*7*7, hidden_space, dtype=torch.float),
            # nn.Tanh()
            nn.Flatten(0, -1),
            nn.Linear(2, 100),
            nn.Sigmoid(),
            nn.Linear(100, hidden_space),
            nn.Sigmoid()
        )

        self.l1 = nn.Sequential(
            nn.Flatten(0, -1),
        )

        self.l2 = nn.Sequential(
            nn.Linear(10*10, 200),
            nn.ReLU(),
            nn.Linear(200, 150),
        )

        self.l3 = nn.Sequential(
            nn.ReLU()
        )

        # Policy Mean specific Linear Layer
        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_space, action_space_dims, dtype=torch.float)
        )

        # Policy Std Dev specific Linear Layer
        self.policy_stddev_net = nn.Sequential(
            nn.Linear(hidden_space, action_space_dims, dtype=torch.float)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Conditioned on the observation, returns the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            x: Observation from the environment

        Returns:
            action_means: predicted mean of the normal distribution
            action_stddevs: predicted standard deviation of the normal distribution
        """
        # x = nn.functional.normalize(x)
        shared_features = self.shared_net(x.float())
        if any(torch.isnan(shared_features)):
            print(x.float())
            print('l1:', self.l1(x.float()))
            print('l2:', self.l2(self.l1(x.float())))
            print('l3:', self.l3(self.l2(self.l1(x.float()))))
            print(shared_features)
        action_means = self.policy_mean_net(shared_features)
        action_stddevs = torch.log(
            1 + torch.exp(self.policy_stddev_net(shared_features))
        )

        return action_means, action_stddevs

class REINFORCE:
    """REINFORCE algorithm."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes an agent that learns a policy via REINFORCE algorithm [1]
        to solve the task at hand (Inverted Pendulum v4).

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """

        # Hyperparameters
        self.learning_rate = .1#2e-2  # Learning rate for policy optimization
        self.gamma = 0.99  # Discount factor
        self.eps = 1e-4  # small number for mathematical stability

        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards

        self.net = Policy_Network(obs_space_dims, action_space_dims)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

    def sample_action(self, state: np.ndarray) -> float:
        """Returns an action, conditioned on the policy and observation.

        Args:
            state: Observation from the environment

        Returns:
            action: Action to be performed
        """
        state = torch.tensor(np.array([state]))
        action_means, action_stddevs = self.net(state)

        # create a normal distribution from the predicted
        #   mean and standard deviation and sample an action
        # try:
        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        
        action = distrib.sample((4,))
        prob = distrib.log_prob(action)
        # print(action_means, action, sep= '|')
        

        action = int(torch.argmax(action))

        self.probs.append(prob)
        return action

    def update(self):
        """Updates the policy network's weights."""
        running_g = 0
        gs = []

        # Discounted return (backwards) - [::-1] will return an array in reverse
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)

        loss = 0
        # minimize -1 * prob * reward obtained
        for log_prob, delta in zip(self.probs, deltas):
            loss += log_prob.mean() * delta * (-1)

        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.rewards = []

# Create and wrap the environment
print('starting')
env = EuropaRover()
wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward

total_num_episodes = int(2000)  # Total number of episodes og: 5e3
nr_epochs = int(1)
# Observation-space of InvertedPendulum-v4 (4)
obs_space_dims = 20*20 #env.observation_space.shape[0]
# Action-space of InvertedPendulum-v4 (1)
action_space_dims = 4 #env.action_space.shape[0]
rewards_over_seeds = []


agent = REINFORCE(obs_space_dims, action_space_dims)
for epoch in range(nr_epochs):  # Fibonacci seeds # 
    # set seed
    seed = 15 # np.random.randint(0,1000)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    env.hard_reset()

    # Reinitialize agent every seed
    reward_over_episodes = []
    top_reward = -9000
    nan_crash = False
    for episode in range(total_num_episodes):
        # gymnasium v26 requires users to set seed while resetting the environment
        obs, info = wrapped_env.reset(seed=seed)

        done = False
        nan_crash = False
        nr_steps = 0
        total_reward = 0
        while not done:
            state = [obs["position"]]
            # try:
            action = agent.sample_action(state)
            # Take most probable action if probability
            # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
            # These represent the next observation, the reward from the step,
            # if the episode is terminated, if the episode is truncated and
            # additional info from the step
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            if np.nan in obs:
                print(obs)
            total_reward += reward
            agent.rewards.append(reward)
            # except:
            #     done = True
            #     nan_crash = True

            # End the episode when either truncated or terminated is true
            #  - truncated: The episode duration reaches max number of timesteps
            #  - terminated: Any of the state space values is no longer finite.
            if nr_steps > 60:
                truncated = True
            done = terminated or truncated
            nr_steps += 1

        # if total_reward > top_reward:
        #     top_reward = total_reward
        #     fig, ax = plt.subplots(figsize=(6,6))
        #     plt.imshow(env.map.terrain, cmap="tab10", vmin = -0.25, vmax = 4.75)
        #     plt.colorbar()
        #     plt.title(f"ts: {total_reward}, s: {seed}, ep: {episode}")
        #     plt.grid(True)
        #     save_path = save_file_path('figures/top_scores_long_single_seed/', f'top_score', 'png')
        #     plt.savefig(save_path)
        #     plt.close()
        
        # if not nan_crash:
        reward_over_episodes.append(wrapped_env.return_queue[-1])
        agent.update()

        # if seed * total_num_episodes + episode % 500 == 0:
        #     fig, ax = plt.subplots(figsize=(6,6))
        #     plt.imshow(env.map.terrain, cmap="tab10", vmin = -0.25, vmax = 4.75)
        #     plt.colorbar()
        #     plt.title(f"ts: {total_reward}, s: {seed}, ep: {episode}")
        #     plt.grid(True)
        #     save_path = save_file_path('figures/intermediate_long_single_seed/', f'intermediate_run_{seed * total_num_episodes + episode}', 'png')
        #     plt.savefig(save_path)
        #     plt.close()
        if episode % 100 == 0:
            fig, ax = plt.subplots(figsize=(6,6))
            plt.imshow(env.map.terrain, cmap="tab10", vmin = -0.25, vmax = 4.75)
            plt.colorbar()
            plt.title(f"ts: {total_reward}, s: {seed}, ep: {episode}")
            plt.grid(True)
            save_path = save_file_path(f'figures/iter_results/seed{seed}/', f'epoch{epoch}seed{seed}episode{episode}', 'png')
            plt.savefig(save_path)
            plt.close()

        if episode % 10 == 0:
            avg_reward = np.mean(wrapped_env.return_queue)
            print(f"epoch {epoch:<6} so {((epoch)*total_num_episodes+episode)/(nr_epochs*total_num_episodes):>4.1%}\t | seed {seed:>4}| \t {episode = :>5} so {episode/total_num_episodes:>4.1%} \t | \t {avg_reward = :<20.3f}", end='\r')


    rewards_over_seeds.append(reward_over_episodes)

# torch.save(model.state_dict(), PATH)

rewards_to_plot = [[reward[0] for reward in rewards] for rewards in rewards_over_seeds]
df1 = pd.DataFrame(rewards_to_plot).melt()
df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
sns.set(style="darkgrid", context="talk", palette="rainbow")
sns.lineplot(x="episodes", y="reward", data=df1).set(
    title=f"REINFORCE for {total_num_episodes}e{nr_epochs}s"
)
plt.savefig(save_file_path('results/combined', 'result_sigmoid', 'png'))
plt.show()

rewards_to_plot = []
iteration = []
for rewards in rewards_over_seeds:
    for reward in rewards:
        rewards_to_plot.append(reward[0])
df1 = pd.DataFrame([rewards_to_plot]).melt()
# print(f"{df1 = }")
df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
sns.set(style="darkgrid", context="talk", palette="rainbow")
sns.scatterplot(x="episodes", y="reward", data=df1).set(
    title="REINFORCE for RoverSimulaton"
)
plt.savefig(save_file_path('results/scatter', 'result_sigmoid', 'png'))
plt.show()

