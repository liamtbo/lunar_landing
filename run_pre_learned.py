import gymnasium as gym
import torch
from itertools import count
import torch.nn as nn
import torch.nn.functional as F


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

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

def softmax(state_qvalues):
    state_qvalues_probabilities = torch.softmax(state_qvalues, dim=0)
    state_qvalues_dis = torch.distributions.Categorical(state_qvalues_probabilities)
    action = state_qvalues_dis.sample().item()
    return action

env = gym.make('LunarLander-v2', render_mode="human")
policy_nn = torch.load("lunar_landing/policy_nn.pth")
state, _ = env.reset()
for t in count():
    state = torch.tensor(state, dtype=torch.float32, device=device)
    state_qvalues = policy_nn(state)
    action = softmax(state_qvalues)
    next_state, reward, terminated, truncated, info = env.step(action)

    state = next_state

    if terminated or truncated:
        break