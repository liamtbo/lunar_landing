import gymnasium as gym
import torch
from itertools import count
import torch.nn as nn
import torch.nn.functional as F
import math


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

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

env = gym.make('LunarLander-v2', render_mode="human")
policy_nn = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
policy_nn.load_state_dict(torch.load("learned_policy.pth"))

state, _ = env.reset()
for t in count():
    state = torch.tensor(state, dtype=torch.float32, device=device)
    state_qvalues = policy_nn(state)
    action = state_qvalues.max(0).indices.item()
    next_state, reward, terminated, truncated, info = env.step(action)

    state = next_state

    if terminated or truncated:
        break