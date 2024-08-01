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
from itertools import count

import unittest

from main import DQN

env = gym.make('LunarLander-v2')
torch.manual_seed(1) # for torch.randint in e_greedy

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

from optimum import  ReplayMemory
from main import optimize_network
class TestOptimizeNet(unittest.TestCase):
    def test_basics(self):
        # Get number of actions from gym action space
        n_actions = env.action_space.n
        # Get the number of state observations
        state, info = env.reset()
        n_observations = len(state)
        
        policy_net = DQN(n_observations, n_actions).to(device)
        # policy_net.load_state_dict(torch.load("lunar_landing/policy_nn_weights"))
        target_net = DQN(n_observations, n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        
        optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
        memory = ReplayMemory(10_000)
        
    
        state, _ = env.reset(seed=1) # gets us the same env every time
        state = torch.tensor(state, dtype=torch.float32, device=device)
        for t in count():
            # print(state)
            with torch.no_grad():
                state_qvalues = policy_nn(state)
            action = e_greedy(state_qvalues, functions, hp)

            next_state, reward, terminated, truncated, _ = env.step(action)
            replay.push(state.tolist(), action, reward, next_state, terminated)
            state = torch.tensor(next_state, dtype=torch.float32, device=device)

            optimize_network(functions, hp, replay)

            if terminated or truncated:
                break
        
        for name, param in policy_nn.named_parameters():
            print(f"name: {name}\n\tparam: {param}")

        
        