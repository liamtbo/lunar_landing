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
    def __init__(self, replay_buffer_init):
        self.buffer = []
        self.buffer_size = replay_buffer_init["buffer_size"]
        self.minibatch_size = replay_buffer_init["minibatch_size"]
        self.current_size = 0

    def add(self, state: list[float], action: int, reward: float, next_state: list[float], terminal: bool):
        if self.buffer_size == self.current_size:
            del self.buffer[0]
        else:
            self.current_size += 1
        self.buffer.append([state, action, reward, next_state, terminal])
    
    def sample(self):
        minibatch_i = np.random.choice(len(self.buffer), self.minibatch_size, replace=False)
        states, actions, reward, next_state, terminal = zip(*[self.buffer[i] for i in minibatch_i])
        return np.stack(states), np.array(actions), np.array(reward), np.stack(next_state), np.array(terminal)


def get_td_target(minibatch):
    # for each element in minibatch compute td error
    pass

"""
model = DQN(state_dim, action_dim)
target_model = DQN(state_dim, action_dim)
target_model.load_state_dict(model.state_dict())
"""
def optimize_network(replay, model, optimize_net_init):
    loss_function = optimize_net_init["loss_function"]
    optimizer = optimize_net_init["optimizer"]
    replay_steps = optimize_net_init["replay_steps"]

    # target_model = DQN(optimize_net_init["state_dim"], optimize_net_init["action_dim"])
    # target_model.load_state_dict(model.state_dict())

    # for step in range(replay_steps):
    states, actions, rewards, next_states, terminals = replay.sample()
    states = torch.tensor(states, dtype=torch.float)
    actions = torch.tensor(actions, dtype=torch.int)
    rewards = torch.tensor(rewards, dtype=torch.float)
    next_states = torch.tensor(next_states, dtype=torch.float)
    terminals = torch.tensor(terminals, dtype=torch.float)

    # TODO with_no_grad?
    states_qvalues = model(states)
    next_states_qvalues = model(next_states)

    # # expected sarsa
    next_state_qvals_probs = torch.softmax(next_states_qvalues, dim=1)
    sum_next_states_qvals_times_probs = torch.sum(next_states_qvalues * next_state_qvals_probs, dim=1)

    # # TD error
    td_targets = rewards + 0.99 * sum_next_states_qvals_times_probs * (1 - terminals)
    predicted_values = states_qvalues[torch.arange(states_qvalues.size(0)), actions]
    loss = loss_function(predicted_values, td_targets)

    # # backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# TODO this needs to be udpatetd
# TODO hyper params need to be added and adjusted
def train_loop(train_loop_init, optimize_net_init, replay_buffer_init) -> list[float]:
    model = train_loop_init["model"]
    env = train_loop_init["env"]
    episodes = train_loop_init["episodes"]
    graph_increment = train_loop_init["graph_increment"]

    reward_tracker = []
    reward_sum = 0
    model.train()
    replay = replay_buffer(replay_buffer_init)

    for episode in tqdm(range(episodes)):
        state = (env.reset())[0]
        state_tensor = torch.tensor(state, dtype=torch.float)
        # env.render()
        terminated = False

        while not terminated:
            action_values = model(state_tensor)
            action_probabilies = torch.softmax(action_values, dim=0)# dim is dimension to compute softmax on
            action_dis = torch.distributions.Categorical(action_probabilies)
            action = action_dis.sample().item()

            next_state, reward, terminated, _, _ = env.step(action)

            replay.add(state, action, reward, next_state, terminated)
            reward_sum += reward
            state = next_state
            # complete batch updater here
            if replay.buffer_size == replay.current_size:
                optimize_network(replay, model, optimize_net_init)

        if episode % graph_increment  == 0 and episode != 0:
            reward_tracker.append(reward_sum / graph_increment)
            reward_sum = 0

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
    # device = (
    #     "cuda" if torch.cuda.is_available()
    #     else "mps" if torch.backends.mps.is_available()
    #     else "cpu"
    # )
    
    train_loop_init = {
        "env": env,
        "model": model,
        "episodes": 500,
        "graph_increment": 10 
    }
    optimize_net_init = {
        "loss_function": F.smooth_l1_loss,
        "optimizer": optim.Adam(model.parameters(), lr=1e-3),
        "replay_steps": 20,
        "state_dim": state_dim,
        "action_dim": action_dim
    }
    replay_buffer_init = {
        "buffer_size": 50,
        "minibatch_size": 50
    }
    reward_tracker = train_loop(train_loop_init, optimize_net_init, replay_buffer_init)

    plot_reward(train_loop_init, reward_tracker)

    env.close()

if __name__ == "__main__":
    main()
