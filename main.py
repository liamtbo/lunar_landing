import torch
import torch.nn as nn # actin-value nn
import torch.optim as optim # optimizer
import torch.nn.functional as F # activates and loss functions

import random
from tqdm import tqdm
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import namedtuple, deque

# deep Q-network
## nn.Module is a base class that provides functionality to organize and manage 
## the parameters of a neural network.
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
    
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

class replay_buffer():
    def __init__(self, replay_buff_capacity):
        self.replay_buffer = deque([], maxlen=replay_buff_capacity)

    def push(self, *args):
        self.replay_buffer.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.replay_buffer, batch_size)
    
    def __len__(self):
        return len(self.replay_buffer)


def optimize_network(sample, model, target_model, optimize_net_init):

    loss_function = optimize_net_init["loss_function"]
    optimizer = optimize_net_init["optimizer"]
    gamma = optimize_net_init["gamma"]

    batch = Transition(*zip(*sample)) # "*" unpacks sample, zip puts cols together, * unpacks zip
    states = torch.tensor(batch.state, dtype=torch.float)

    actions = torch.tensor(batch.action, dtype=torch.int)
    rewards = torch.tensor(batch.reward, dtype=torch.float)
    next_states = torch.tensor(batch.next_state, dtype=torch.float)
    done = torch.tensor(batch.done, dtype=torch.float)

    # TODO with_no_grad? ALSO, calculating states qvals twice ! once here one in train loop
    states_qvalues = model(states) 
    next_states_qvalues = target_model(next_states)

    # # expected sarsa
    next_state_qvals_probs = torch.softmax(next_states_qvalues, dim=1)
    sum_next_states_qvals_times_probs = torch.sum(next_states_qvalues * next_state_qvals_probs, dim=1)

    # # TD error
    td_targets = rewards + gamma * sum_next_states_qvals_times_probs * (1 - done)
    predicted_values = states_qvalues[torch.arange(states_qvalues.size(0)), actions]
    loss = loss_function(predicted_values, td_targets)

    # # backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    

def train_loop(train_loop_init, optimize_net_init) -> list[float]:
    policy_nn = train_loop_init["policy_nn"]
    target_nn = train_loop_init["target_nn"]

    env = train_loop_init["env"]
    graph_increment = train_loop_init["graph_increment"]
    tau = optimize_net_init["tau"]
    reward_tracker = []
    reward_sum = 0
    policy_nn.train()
    replay = replay_buffer(optimize_net_init["replay_buff_capacity"])

    for episode in tqdm(range(train_loop_init["episodes"])):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float)
        # env.render()
        done = False

        while not done:
            action_values = policy_nn(state)
            # softmax
            action_probabilies = torch.softmax(action_values / tau, dim=0)# dim is dimension to compute softmax on
            action_dis = torch.distributions.Categorical(action_probabilies)
            action = action_dis.sample().item()
            # e-greedy
            # if np.random.rand() < .9:
            #     action = torch.argmax(action_values).item()
            # else:
            #     action = torch.randint(0,len(action_values), (1,)).item()

            next_state, reward, done, _, _ = env.step(action)

            replay.push(state.tolist(), action, reward, next_state, done)
            reward_sum += reward
            state = torch.tensor(next_state)
            # complete batch updater here
            if len(replay) > optimize_net_init["minibatch_size"]:
                target_nn.load_state_dict(policy_nn.state_dict())

                for _ in range(optimize_net_init["replay_steps"]):
                    sample = replay.sample(optimize_net_init["minibatch_size"])
                    optimize_network(sample, policy_nn, target_nn, optimize_net_init)

        if episode % graph_increment  == 0 and episode != 0:
            reward_tracker.append(reward_sum / graph_increment)
            reward_sum = 0
            print(reward_tracker)

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
    policy_nn = DQN(state_dim, action_dim)
    target_nn = DQN(state_dim, action_dim)
    target_nn.load_state_dict(policy_nn.state_dict())

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    train_loop_init = {
        "env": env,
        "policy_nn": policy_nn,
        "target_nn": target_nn,
        "episodes":300, # make sure graph_incremenet | episodes
        "graph_increment": 10,
        "timeout": 500,
        "device": device
    }
    optimize_net_init = {
        "loss_function": F.mse_loss,
        "optimizer": optim.Adam(policy_nn.parameters(), lr=1e-3, betas = (0.9, 0.999), eps = 1e-8),
        "replay_steps": 20,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "gamma": 0.99,
        "tau": 0.001,
        "replay_buff_capacity": 50000,
        "minibatch_size": 128
    }
    reward_tracker = train_loop(train_loop_init, optimize_net_init)

    plot_reward(train_loop_init, reward_tracker)

    env.close()

if __name__ == "__main__":
    main()
