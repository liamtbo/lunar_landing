import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import math
import random
import numpy as np
from tqdm import tqdm
from itertools import count
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque

random.seed(1)

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


# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

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
        next_states_qvalues = target_nn(next_states) * (1 - terminated).unsqueeze(dim=1)
        # # expected sarsa
        next_state_qvals_probs = torch.softmax(next_states_qvalues, dim=1)
        sum_next_states_qvals_times_probs = torch.sum(next_states_qvalues * next_state_qvals_probs, dim=1)

    target = rewards + 0.99 * sum_next_states_qvals_times_probs
    loss = loss_function(predicted, target)
    optimizer.zero_grad()
    loss.backward()

    # torch.nn.utils.clip_grad_value_(policy_nn.parameters(), 100)

    optimizer.step()

steps_done = 0

def e_greedy(q_values, functions, hyperparameters):
    env = functions["env"]
    device = functions["device"]
    global steps_done
    eps_threshold = hyperparameters["eps_end"] + (hyperparameters["eps_start"] - hyperparameters["eps_end"]) \
                    * math.exp(-1. * steps_done / hyperparameters["eps_decay"])
    steps_done += 1
    sample = random.random() # TODO 
    if sample > eps_threshold:
        with torch.no_grad():
            return q_values.max(0).indices.item()
    else:
        return torch.tensor([env.action_space.sample()], device=device, dtype=torch.long).item()

def softmax(state_qvalues, functions, hyperparameters):
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
    replay = replay_buffer(hyperparameters["replay_buffer_capacity"])
    reward_sum = 0

    for episode in range(hyperparameters["episodes"]):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device)
        terminated = False

        for t in count():
            with torch.no_grad():
                state_qvalues = policy_nn(state)
            action = select_action(state_qvalues, functions, hyperparameters)

            next_state, reward, terminated, truncated, info = env.step(action)
            reward_sum += reward
            replay.push(state.tolist(), action, reward, next_state, terminated)
            state = torch.tensor(next_state, dtype=torch.float32, device=device)

            for _ in range(hyperparameters["replay_steps"]):
                optimize_network(functions, hyperparameters, replay)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            # TAU = 0.005
            # target_net_state_dict = target_nn.state_dict()
            # policy_net_state_dict = policy_nn.state_dict()
            # for key in policy_net_state_dict:
                # target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            # target_nn.load_state_dict(target_net_state_dict)
        
            target_nn.load_state_dict(policy_nn.state_dict())

            if terminated or truncated:
                episode_durations.append(reward_sum) # TODO
                reward_sum = 0
                plot_durations()
                break

    print('Complete')
    plot_durations(show_result=True)
    plt.ioff()
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
    # torch.save(policy_nn.state_dict(), "lunar_landing/policy_nn_weights.pth")
    # policy_nn.load_state_dict(torch.load("lunar_landing/policy_nn_weights.pth"))
    target_nn = DQN(state_dim, action_dim).to(device)
    target_nn.load_state_dict(policy_nn.state_dict())
    lr = 1e-4

    functions = {
        "env": env,
        "policy_nn": policy_nn,
        "target_nn": target_nn,
        "loss_function": nn.SmoothL1Loss(),
        "optimizer": optim.Adam(policy_nn.parameters(), lr=lr),
        "device": device,
        "select_action": softmax # softmax or e_greedy
    }
    hyperparameters = {
        "episodes": 100,
        "graph_increment": 10,
        "replay_steps": 4,
        "learning_rate": lr,
        "tau": 0.001,
        "replay_buffer_capacity": 50000,
        "minibatch_size": 8,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "eps_end": 0.05,
        "eps_start": 0.9,
        "eps_decay": 1000,
    }

    training_loop(functions, hyperparameters)
    torch.save(policy_nn.state_dict(), "learned_policy_nn.pth")

    env.close()

if __name__ == "__main__":
    main()