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

env = gym.make('LunarLander-v2', render_mode="human")
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
        self.layer1 = nn.Linear(input_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, output_dim)
    # x is state input q(s,a)
    # output is q(s,a) for all action vals
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# torch.save(policy_nn.state_dict(), "lunar_landing/policy_nn_weights.pth")
# policy_nn.load_state_dict(torch.load("lunar_landing/policy_nn_weights.pth"))
# policy_nn.train()

# target_nn.eval()

functions = {
    "env": env,
    "policy_nn": None,
    "target_nn": None,
    "loss_function": nn.SmoothL1Loss(),
    "optimizer": None,
    "device": device,
    "select_action": action_functions.softmax # softmax or e_greedy
}

hp = {
    "episodes": 100,
    "graph_increment": 10,
    "replay_steps": 4,
    "ReplayBuffer_capacity": 10000,
    "minibatch_size": 128,
    "eps_end": 0.05,
    "eps_start": 0.9,
    "eps_decay": 1000,
    "plot": True,
    "learning_rate": None,
    "tau": None,
    "seed": None
}

lr_range = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
tau_range = np.array([0.001, 0.01, 0.1, 1.0])
num_runs = 10
num_episodes = 100

reward_tracker = torch.zeros(size=(len(lr_range), len(tau_range), num_runs, num_episodes))

for lr_i in range(len(lr_range)):

    for tau_i in range(len(tau_range)):
        print(f"lr: {lr_range[lr_i]}\ttau: {tau_range[tau_i]}")

        for run in range(num_runs):
            # resetting model params
            functions["policy_nn"] = DQN(state_dim, action_dim).to(device)
            functions["target_nn"] = DQN(state_dim, action_dim).to(device)
            functions["target_nn"].load_state_dict(functions["policy_nn"].state_dict())
            functions["optimizer"] = optim.Adam(functions["policy_nn"].parameters(), lr=lr_range[lr_i])
            hp["learning_rate"] = lr_range[lr_i]
            hp["tau"] = tau_range[tau_i]
            hp["sees"] = run

            reward = []
            for episode in range(num_episodes):
                reward.append(train.training(functions, hp))
                if hp["plot"]:
                    plots.plot_rewards(reward)
            reward_tracker[lr_i, tau_i, run] = torch.tensor([reward])
        
        print(f"\tavg last 20 reward -> {(reward_tracker[lr_i, tau_i][:,-20:].sum(1) / 20).sum() / num_runs}")

print("here")