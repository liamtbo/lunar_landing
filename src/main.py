import gymnasium as gym
import torch
import torch.nn as nn # actin-value nn
import torch.optim as optim # optimizer
import torch.nn.functional as F # activates and loss functions
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# deep Q-network
## nn.Module is a base class that provides functionality to organize and manage 
## the parameters of a neural network.
class DQN(nn.Module): 
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # fully connected layers of nn
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    # x is state input q(s,a)
    # output is q(s,a) for all action vals
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    

class replay_buffer():
    def __init__(self, buffer_size: int, minibatch_size: int):
        self.buffer = np.array([])
        self.buffer_size = buffer_size
        self.minibatch_size = minibatch_size
        self.current_size = 0

    def add(self, state: list[float]):
        if self.buffer_size == self.current_size:
            del self.buffer[0]
        else:
            self.current_size += 1
        self.buffer = np.append(self.buffer, state)
    
    def sample(self):
        return [self.buffer[i] for i in np.random.choice(np.arange(self.buffer_size), size=self.minibatch_size)]

    
def train_loop(model, env, loss_function, optimizer, episodes, graph_increment) -> list[float]:
    reward_tracker = []
    rewards = 0
    model.train()

    for episode in tqdm(range(episodes)):
        state = (env.reset())[0]
        # env.render()
        terminated = False

        while not terminated:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_values = model(state_tensor)
            action_probabilies = torch.softmax(action_values, dim=1) # dim is dimension to compute softmax on
            action_dis = torch.distributions.Categorical(action_probabilies)
            action = action_dis.sample().item()

            next_state, reward, terminated, _, _ = env.step(action)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

            # expected sarsa
            next_state_action_values = model(next_state_tensor)
            next_state_action_probabilites = torch.softmax(next_state_action_values, dim=1)
            sum_next_state_actions = (next_state_action_values * next_state_action_probabilites).sum()

            # TD error
            td_target = torch.FloatTensor((reward + 0.99 * sum_next_state_actions))
            predicted_value = torch.FloatTensor(action_values[0, action])
            loss = loss_function(predicted_value, td_target)

            # backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            rewards += reward
            state = next_state

        if episode % graph_increment  == 0 and episode != 0:
            reward_tracker.append(rewards / graph_increment)
            rewards = 0
    return reward_tracker

def plot_reward(episodes: int, graph_increment: int, reward_tracker: list[float]):
    x = [i for i in range(int(episodes / graph_increment - 1))]
    y = reward_tracker

    fig, ax = plt.subplots()
    ax.plot(x,y,label="reward data", marker='o')
    ax.set_title("simple line plot")
    ax.set_xlabel("epsidoes")
    ax.set_ylabel("num of rewards")
    ax.legend()
    plt.show()

def main():
    # env = gym.make('LunarLander-v2', render_mode="human")
    env = gym.make('LunarLander-v2')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model = DQN(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters())
    episodes = 500
    graph_increment = 10
    
    reward_tracker = train_loop(model, env, F.smooth_l1_loss, optimizer, episodes, graph_increment)

    plot_reward(episodes, graph_increment, reward_tracker)

    env.close()

if __name__ == "__main__":
    main()
