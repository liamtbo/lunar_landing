import numpy as np
import torch
import gymnasium as gym
from collections import namedtuple, deque
import random

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math

import train
import action_functions
import plots

# env = gym.make('LunarLander-v2', render_mode="human")
env = gym.make('LunarLander-v2')

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class DQN(nn.Module): 
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # fully connected layers of nn
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, output_dim)
    # x is state input q(s,a)
    # output is q(s,a) for all action vals
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

functions = {
    "env": env,
    "policy_nn": None,
    "target_nn": None,
    "loss_function": nn.SmoothL1Loss(),
    "optimizer": None,
    "device": device,
    "select_action": action_functions.e_greedy # softmax or e_greedy
}
# hyperparameters
hp = {
    "episodes": 300,
    "graph_increment": 10,
    "replay_steps": 4,
    "ReplayBuffer_capacity": 10000,
    "minibatch_size": 128,
    "eps_end": 0.05,
    "eps_start": 0.9,
    "eps_decay": 1000,
    "plot": True,
    "lr": 1e-4,
    "tau": 0.005,
}

num_episodes = 100

# resetting model params
functions["policy_nn"] = DQN(state_dim, action_dim).to(device)
functions["target_nn"] = DQN(state_dim, action_dim).to(device)
functions["target_nn"].load_state_dict(functions["policy_nn"].state_dict())
functions["optimizer"] = optim.Adam(functions["policy_nn"].parameters())


reward = []
for episode in range(num_episodes):
    reward.append(train.training(functions, hp))
    if hp["plot"]:
        plots.plot_rewards(reward)

env.close()
