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
import sys

class DQN(nn.Module): 
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "terminated"))

class ReplayBuffer():
    def __init__(self, replay_buff_capacity):
        self.ReplayBuffer = deque([], maxlen=replay_buff_capacity)

    def push(self, *args):
        self.ReplayBuffer.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.ReplayBuffer, batch_size)
    
    def __len__(self):
        return len(self.ReplayBuffer)


# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

episode_rewards = []

def plot_rewards(show_result=False):
    plt.figure(1)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('reward')
    plt.plot(rewards_t.numpy())
    # Take 100 episode averages and plot them too
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def optimize_network(functions, hp, replay: ReplayBuffer):
    if len(replay) < hp["minibatch_size"]:
        return
    device = functions["device"]
    policy_nn = functions["policy_nn"]
    target_nn = functions["target_nn"]
    loss_function = functions["loss_function"]
    optimizer = functions["optimizer"]

    batch = replay.sample(hp["minibatch_size"])
    batch = Transition(*zip(*batch))

    states = torch.tensor(np.array(batch.state), dtype=torch.float32, device=device)
    actions = torch.tensor(batch.action, dtype=torch.long, device=device).unsqueeze(1)
    rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device)
    next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=device)
    terminated = torch.tensor(batch.terminated, dtype=int, device=device)

    predicted  = policy_nn(states).gather(1, actions).squeeze(1)
    with torch.no_grad():
        next_states_qvalues = torch.max(target_nn(next_states), dim=1).values * (1 - terminated)
    target = rewards + 0.99 * next_states_qvalues
    loss = loss_function(predicted, target)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_nn.parameters(), 1)

    optimizer.step()


def e_greedy(q_values, functions, hp):
    env = functions["env"]
    device = functions["device"]
    global steps_done
    eps_threshold = hp["eps_end"] + (hp["eps_start"] - hp["eps_end"]) \
                    * math.exp(-1. * hp["steps_done"] / hp["eps_decay"])
    hp["steps_done"] += 1
    sample = torch.rand(1).item()
    # print(f"sample: {sample}")
    # sample = eps_threshold + 0.00001
    if sample > eps_threshold:
        with torch.no_grad():
            return q_values.max(0).indices.item()
    else:
        action = torch.randint(0, env.action_space.n, size=(1,)).item()
        # print(f"action: {action}")
        return torch.tensor([action], device=device, dtype=torch.long).item()

def softmax(state_qvalues, functions, hp):
    state_qvalues_probabilities = torch.softmax(state_qvalues, dim=0)
    state_qvalues_dis = torch.distributions.Categorical(state_qvalues_probabilities)
    action = state_qvalues_dis.sample().item()
    return action

def training_loop(functions, hp):
    env = functions["env"]
    policy_nn = functions["policy_nn"]
    target_nn = functions["target_nn"]
    device = functions["device"]
    select_action = functions["select_action"]
    graph_increment = hp["graph_increment"]
    replay = ReplayBuffer(hp["ReplayBuffer_capacity"])
    reward_sum = 0

    for episode in range(hp["episodes"]):
        state, _ = env.reset()


        # TODO tests - del
        # state = np.zeros(shape=(8,))

        state = torch.tensor(state, dtype=torch.float32, device=device)
        terminated = False

        for t in count():
            with torch.no_grad():
                state_qvalues = policy_nn(state)
            action = select_action(state_qvalues, functions, hp)
            next_state, reward, terminated, truncated, _ = env.step(action)
            reward_sum += reward
            replay.push(state.tolist(), action, reward, next_state, terminated)
            state = torch.tensor(next_state, dtype=torch.float32, device=device)

            optimize_network(functions, hp, replay)

            # soft update
            TAU = 0.005
            target_net_state_dict = target_nn.state_dict()
            policy_net_state_dict = policy_nn.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_nn.load_state_dict(target_net_state_dict)
        
            if terminated or truncated:
                episode_rewards.append(reward_sum) # TODO
                reward_sum = 0
                plot_rewards()
                break
            
    print('Complete')
    plot_rewards(show_result=True)
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
        "select_action": e_greedy # softmax or e_greedy
    }
    hp = {
        "episodes": 1000,
        "graph_increment": 10,
        "replay_steps": 1,
        "learning_rate": lr,
        "tau": 0.001,
        "ReplayBuffer_capacity": 10000,
        "minibatch_size": 32,
        "eps_end": 0.05,
        "eps_start": 0.9,
        "eps_decay": 1000,
        "steps_done": 0,
    }

    training_loop(functions, hp)
    torch.save(policy_nn.state_dict(), "learned_policy.pth")
    env.close()

if __name__ == "__main__":
    main()