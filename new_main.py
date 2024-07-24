import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import namedtuple, deque

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
    
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "terminated"))

class replay_buffer():
    def __init__(self, replay_buff_capacity):
        self.replay_buffer = deque([], maxlen=replay_buff_capacity)

    def push(self, *args):
        self.replay_buffer.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.replay_buffer, batch_size)
    
    def __len__(self):
        return len(self.replay_buffer)

def optimize_network(functions, hyperparameters, replay: replay_buffer):
    if len(replay) < hyperparameters["minibatch_size"]:
        return
    device = functions["device"]
    policy_nn = functions["policy_nn"]
    target_nn = functions["target_nn"]
    lr = hyperparameters["learning_rate"]
    loss_function = functions["loss_function"]
    optimizer = functions["optimizer"]

    batch = replay.sample(hyperparameters["minibatch_size"])
    batch = Transition(*zip(*batch))
    states = torch.tensor(np.array(batch.state), dtype=torch.float32, device=device)
    actions = torch.tensor(batch.action, dtype=torch.long, device=device).unsqueeze(1) # need long for gather
    rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device)
    next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=device)
    terminated = torch.tensor(batch.terminated, dtype=int, device=device)

    predicted  = policy_nn(states).gather(1, actions).squeeze(1)
    with torch.no_grad():
        next_states_qvalues = torch.max(target_nn(next_states), dim=1).values

    # TODO PE with working with terminated q_vals
    target = rewards + lr * next_states_qvalues * (1 - terminated)
    loss = loss_function(predicted, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def softmax(state_qvalues):
    state_qvalues_probabilities = torch.softmax(state_qvalues, dim=0)
    state_qvalues_dis = torch.distributions.Categorical(state_qvalues_probabilities)
    action = state_qvalues_dis.sample().item()
    return action

def training_loop(functions, hyperparameters):
    env = functions["env"]
    policy_nn = functions["policy_nn"]
    target_nn = functions["target_nn"]
    device = functions["device"]
    select_action = functions["select_action"]
    graph_increment = hyperparameters["graph_increment"]
    replay = replay_buffer(hyperparameters["replay_buffer_capacity"])
    reward_tracker = []
    reward_sum = 0

    for episode in tqdm(range(hyperparameters["episodes"])):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device)
        terminated = False

        while not terminated:
            with torch.no_grad():
                state_qvalues = policy_nn(state)
            action = select_action(state_qvalues)
            next_state, reward, terminated, _, _ = env.step(action) # TODO PE truncated, look at cart ex
            # TODO PE possibly need to make next_state = none when terminal
            reward_sum += reward
            replay.push(state.tolist(), action, reward, next_state, terminated)
            state = torch.tensor(next_state, dtype=torch.float32, device=device)

            optimize_network(functions, hyperparameters, replay)

            target_nn.load_state_dict(policy_nn.state_dict())
        
        if episode % graph_increment  == 0 and episode != 0:
            reward_tracker.append(reward_sum / graph_increment)
            reward_sum = 0
            print(reward_tracker)

    return reward_tracker


def plot_reward(hyperparameters, reward_tracker: list[float]):
    episodes = hyperparameters["episodes"]
    graph_increment = hyperparameters["graph_increment"]

    if episodes % graph_increment != 0:
        print("graph increment must divide episodes for graph/data to be displayed")
        return

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
        "select_action": softmax,
    }
    hyperparameters = {
        "episodes": 300,
        "graph_increment": 10,
        "replay_steps": 20,
        "learning_rate": lr,
        "tau": 0.001,
        "replay_buffer_capacity": 50000,
        "minibatch_size": 128
    }

    reward_tracker = training_loop(functions, hyperparameters)
    plot_reward(hyperparameters, reward_tracker)

    env.close()

if __name__ == "__main__":
    main()
