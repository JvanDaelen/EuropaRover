import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import random

import gymnasium as gym

class RegressionNetwork(nn.Module):
    def __init__(self, in_dimention, out_dimention) -> None:
        super().__init__()

        hidden_dimention = 2
        self.refnet = nn.Sequential(
            nn.Linear(in_dimention, hidden_dimention, dtype=float),
            nn.ReLU(),
            nn.Linear(hidden_dimention, out_dimention, dtype=float)
        )
    
    def forward(self, x: torch.Tensor):
        return self.refnet(x)
    


def reference_function(x):
    if x < 0:
        return -x
    else:
        return x
    
input = np.arange(-10, 10, 0.1)
reference_response = [reference_function(x) for x in input]

network = RegressionNetwork(1, 1)
initial_network_response = [network(torch.tensor([x])).detach().numpy()[0] for x in input]

learning_rate = 0.001
loss_fn = nn.SmoothL1Loss()
optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate, amsgrad=True)

num_epochs = 1
loss_values = []

for epoch in range(num_epochs):
    # Set optimuzer to zero
    optimizer.zero_grad()
    rand_x = np.random.random(1) * 20 - 10
    correct_response = [reference_function(x) for x in rand_x]
    correct_response = torch.tensor([correct_response], dtype=torch.double)
    pred = network(torch.tensor([rand_x], dtype=torch.double))
    # # print(f"{correct_response = }")
    # # print(f"{pred = }")
    loss = loss_fn(pred, torch.tensor([np.double(correct_response)]))
    # # print(f"{loss = }")
    loss_values.append(loss.item())
    # # print(f"{loss_values = }")
    loss.backward()
    optimizer.step()


trained_network_response = [network(torch.tensor([x])).detach().numpy()[0] for x in input]
plt.figure('outputs')
plt.plot(input,  reference_response, label='reference')
plt.scatter(input,  initial_network_response, label='initial')
plt.scatter(input,  trained_network_response, label='trained')
plt.legend()

plt.figure('loss')
plt.plot(loss_values)
plt.show()