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
    

def train_loop(train_loop_init, optimize_net_init, replay_buffer_init) -> list[float]:
    model = train_loop_init["model"]
    env = train_loop_init["env"]
    episodes = train_loop_init["episodes"]
    graph_increment = train_loop_init["graph_increment"]
    tau = optimize_net_init["tau"]

    loss_function = optimize_net_init["loss_function"]
    optimizer = optimize_net_init["optimizer"]
    gamma = optimize_net_init["gamma"]

    reward_tracker = []
    reward_sum = 0
    model.train()

    for episode in tqdm(range(episodes)):
        state = (env.reset())[0]
        state_tensor = torch.tensor(state, dtype=torch.float, requires_grad=True)
        # env.render()
        terminated = False
        steps = 0
        while not terminated:
            states_qvalues = model(state_tensor)
            # softmax
            with torch.no_grad():
                action_probabilies = torch.softmax(states_qvalues, dim=0)# dim is dimension to compute softmax on
                action_dis = torch.distributions.Categorical(action_probabilies)
                action = action_dis.sample().item()

            next_state, reward, terminated, _, _ = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float, requires_grad=False)

            with torch.no_grad():
                next_states_qvalues = model(next_state)
                next_state_qvals_probs = torch.softmax(next_states_qvalues, dim=0)
                sum_next_states_qvals_times_probs = torch.sum(next_states_qvalues * next_state_qvals_probs, dim=0)
            
                # # TD error
                td_targets = reward + gamma * sum_next_states_qvals_times_probs * (1 - terminated)
                predicted_values = states_qvalues[action]

            loss = loss_function(predicted_values, td_targets)
        
            # # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            reward_sum += reward
            state = next_state
            # complete batch updater here
   
        if episode % graph_increment  == 0 and episode != 0:
            reward_tracker.append(reward_sum / graph_increment)
            reward_sum = 0
            # print(reward_tracker)

    return reward_tracker

def plot_reward(train_loop_init, reward_tracker: list[float]):
    episodes = train_loop_init["episodes"]
    graph_increment = train_loop_init["graph_increment"]
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

    # TODO
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # torch.set_default_device(device)

    train_loop_init = {
        "env": env,
        "model": model,
        "episodes":1500, # make sure graph_incremenet | episodes
        "graph_increment": 10,
        "timeout": 500
    }
    optimize_net_init = {
        "loss_function": F.mse_loss,
        "optimizer": optim.Adam(model.parameters(), lr=1e-3, betas = (0.9, 0.999), eps = 1e-8),
        "replay_steps": 20,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "gamma": 0.99,
        "tau": 0.001
    }
    replay_buffer_init = {
        "buffer_capacity": 50000,
        "minibatch_size": 8
    }
    reward_tracker = train_loop(train_loop_init, optimize_net_init, replay_buffer_init)

    plot_reward(train_loop_init, reward_tracker)

    env.close()

if __name__ == "__main__":
    main()
