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

def softmax(state_qvalues):
    state_qvalues_probabilities = torch.softmax(state_qvalues, dim=0)
    state_qvalues_dis = torch.distributions.Categorical(state_qvalues_probabilities)
    action = state_qvalues_dis.sample().item()
    return action

# def e_greedy(q_values, functions, hp):
#     eps_threshold = hp["eps_end"] + (hp["eps_start"] - hp["eps_end"]) \
#                     * math.exp(-1. * hp["steps_done"] / hp["eps_decay"])
#     hp["steps_done"] += 1
#     sample = torch.rand(1).item()
#     # print(f"sample: {sample}")
#     # sample = eps_threshold + 0.00001
#     if sample > eps_threshold:
#         with torch.no_grad():
#             return q_values.max(0).indices.item()
#     else:
#         action = torch.randint(0, env.action_space.n, size=(1,)).item()
#         # print(f"action: {action}")
#         return torch.tensor([action], device=device, dtype=torch.long).item()
    

env = gym.make('LunarLander-v2', render_mode="human")
policy_nn = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
policy_nn.load_state_dict(torch.load("learned_policy_nn.pth"))

state, _ = env.reset()
for t in count():
    state = torch.tensor(state, dtype=torch.float32, device=device)
    state_qvalues = policy_nn(state)
    action = softmax(state_qvalues)
    next_state, reward, terminated, truncated, info = env.step(action)

    state = next_state

    if terminated or truncated:
        break