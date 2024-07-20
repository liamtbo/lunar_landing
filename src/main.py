import gymnasium as gym
import torch
import torch.nn as nn # actin-value nn
import torch.optim as optim # optimizer
import torch.nn.functional as F # activates and loss functions
from tqdm import tqdm
import time

# deep Q-network
## nn.Module is a base class that provides functionality to organize and manage 
## the parameters of a neural network.
class DQN(nn.Module): 
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # fully connected layers of nn
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    
    # x is state input q(s,a)
    # output is q(s,a) for all action vals
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
env = gym.make('LunarLander-v2', render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
model = DQN(state_dim, action_dim)
optimizer = optim.Adam(model.parameters())

for episode in tqdm(range(1000)):
    state = (env.reset())[0]
    env.render()
    terminated = False

    while not terminated:

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_values = model.forward(state_tensor)
        action_probabilies = torch.softmax(action_values, dim=1) # dim is dimension to compute softmax on
        action_dis = torch.distributions.Categorical(action_probabilies)
        action = action_dis.sample().item()

        next_state, reward, terminated, _, _ = env.step(action)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        # print(next_state_tensor)

        # expected sarsa
        next_state_action_values = model.forward(next_state_tensor)
        next_state_action_probabilites = torch.softmax(next_state_action_values, dim=1)
        sum_next_state_actions = (next_state_action_values * next_state_action_probabilites).sum()

        # TD error
        td_target = torch.FloatTensor(reward + 0.99 * sum_next_state_actions)
        predicted_value = torch.FloatTensor(action_values[0, action])
        loss = F.smooth_l1_loss(predicted_value, td_target)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state

