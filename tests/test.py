import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import math
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import namedtuple, deque

from lunar_landing.new_main import e_greedy
import unittest

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
# env = gym.make('LunarLander-v2', render_mode="human")
env = gym.make('LunarLander-v2')

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
policy_nn = DQN(state_dim, action_dim).to(device)
target_nn = DQN(state_dim, action_dim).to(device)
target_nn.load_state_dict(policy_nn.state_dict())
lr = 0.99

functions = {
    "env": env,
    "policy_nn": policy_nn,
    "target_nn": target_nn,
    "loss_function": nn.SmoothL1Loss(),
    "optimizer": optim.Adam(policy_nn.parameters(), lr=lr), # TODO amsgrad?, ADAMW?
    "device": device,
    "select_action": e_greedy # softmax or e_greedy
}
hyperparameters = {
    "episodes": 300,
    "graph_increment": 10,
    "replay_steps": 20,
    "learning_rate": lr,
    "tau": 0.001,
    "replay_buffer_capacity": 50000,
    "minibatch_size": 128,
    "state_dim": state_dim,
    "action_dim": action_dim,
    "eps_end": 0.05,
    "eps_start": 0.9,
    "eps_decay": 1000
}

steps_done = 0


class TestEpsilonGreedy(unittest.TestCase):
    def test_basics(self):
        qvalues1 = torch.tensor([1,2,3,4])
        qvalues2 = torch.tensor([4,1,2,3])
        qvalues3 = torch.tensor([3,4,1,2])
        qvalues4 = torch.tensor([2,3,4,1])
        self.assert_Equal(e_greedy(qvalues1, functions, hyperparameters), 3)
        self.assert_Equal(e_greedy(qvalues2, functions, hyperparameters), 0)
        self.assert_Equal(e_greedy(qvalues3, functions, hyperparameters), 1)
        self.assert_Equal(e_greedy(qvalues4, functions, hyperparameters), 2)


if __name__ == "__main__":
    unittest.main()