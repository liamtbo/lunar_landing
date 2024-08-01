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

# torch.optim

import unittest

from main import DQN

env = gym.make('LunarLander-v2')
torch.manual_seed(1) # for torch.randint in e_greedy

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
# torch.cuda.manual_seed(1)
# torch.cuda.manual_seed_all(1)
# torch.cuda.seed(1)


functions = {
    "env": env,
    "policy_nn": None,
    "target_nn": None,
    "loss_function": nn.SmoothL1Loss(),
    "optimizer": None,
    "device": device,
    "select_action": None # softmax or e_greedy
}
hp = {
    "episodes": 300,
    "graph_increment": 10,
    "replay_steps": 20,
    "tau": 0.001,
    "ReplayBuffer_capacity": 10000,
    "minibatch_size": 128,
    "eps_end": 0.05,
    "eps_start": 0.9,
    "eps_decay": 1000
}

random.seed(1)

# E - these tests never did random sample
from main import e_greedy
class TestEpsilonGreedy(unittest.TestCase):
    def test_basics(self):
        hp["steps_done"] = 1500
        qvalues1 = torch.tensor([1,2,3,4])
        qvalues2 = torch.tensor([4,1,2,3])
        qvalues3 = torch.tensor([3,4,1,2])
        qvalues4 = torch.tensor([2,3,4,1])
        self.assertEqual(e_greedy(qvalues1, functions, hp), 3) # unfirom random choice with seed 1
        self.assertEqual(e_greedy(qvalues2, functions, hp), 0)
        self.assertEqual(e_greedy(qvalues3, functions, hp), 1)
        self.assertEqual(e_greedy(qvalues4, functions, hp), 2)

from main import softmax
class TestSoftMax(unittest.TestCase):
    def test_basics(self):
        qvalues1 = torch.tensor([1,2,3,10], dtype=torch.float)
        qvalues2 = torch.tensor([10,1,2,3], dtype=torch.float)
        qvalues3 = torch.tensor([3,10,1,2], dtype=torch.float)
        qvalues4 = torch.tensor([2,3,10,1], dtype=torch.float)
        self.assertEqual(softmax(qvalues1, functions, hp), 3) # unfirom random choice with seed 1
        self.assertEqual(softmax(qvalues2, functions, hp), 0)
        self.assertEqual(softmax(qvalues3, functions, hp), 1)
        self.assertEqual(softmax(qvalues4, functions, hp), 2)

from main import  ReplayBuffer
from main import optimize_network
class TestOptimizeNet(unittest.TestCase):
    def test_basics(self):


        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        policy_nn = DQN(state_dim, action_dim).to(device)
        # torch.save(policy_nn.state_dict(), "policy_params.pth")
        policy_nn.load_state_dict(torch.load("policy_params.pth"))
        functions["policy_nn"] = policy_nn
        target_nn = DQN(state_dim, action_dim).to(device)
        target_nn.load_state_dict(policy_nn.state_dict())
        functions["target_nn"] = target_nn
        functions["optimizer"] = optim.AdamW(policy_nn.parameters(), lr=1e-4, amsgrad=True)
        replay = ReplayBuffer(10_000)

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

        
        


if __name__ == "__main__":
    unittest.main()