import gymnasium as gym

import torch

import math
import random
import numpy as np
from itertools import count

from collections import namedtuple, deque
from torch.nn.utils import clip_grad_norm_

import plots

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

def L2_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def optimize_network(functions, hp, replay: ReplayBuffer):

    device = functions["device"]
    policy_nn = functions["policy_nn"]
    target_nn = functions["target_nn"]
    loss_function = functions["loss_function"]
    optimizer = functions["optimizer"]

    batch = replay.sample(hp["minibatch_size"])
    batch = Transition(*zip(*batch))
    states = torch.tensor(np.array(batch.state), dtype=torch.float32, device=device)
    actions = torch.tensor(batch.action, dtype=torch.long, device=device).unsqueeze(1) # need long for gather
    rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device)
    next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=device)
    terminated = torch.tensor(batch.terminated, dtype=int, device=device)


    predicted  = policy_nn(states).gather(1, actions).squeeze(1)
    # print(f"predicted: {predicted}")
    with torch.no_grad():
        next_states_qvalues = target_nn(next_states) * (1 - terminated).unsqueeze(dim=1)
        # # expected sarsa
        next_state_qvals_probs = torch.softmax(next_states_qvalues, dim=1)
        sum_next_states_qvals_times_probs = torch.sum(next_states_qvalues * next_state_qvals_probs, dim=1)

    # print(f"next_states_qvals: {next_states_qvalues}")
    # TODO PE with working with terminated q_vals
    target = rewards + 0.99 * sum_next_states_qvals_times_probs
    # print(f"target: {target}")
    loss = loss_function(predicted, target)
    # print(f"\tloss: {loss}")
    # print(f"loss: {loss}")
    optimizer.zero_grad()
    loss.backward()

    # compute L2 norm
    ## divicde by replay steps to get avg grad norm per episode
    # clip_grad_norm_(policy_nn.parameters(), max_norm=0.5)
    torch.nn.utils.clip_grad_value_(policy_nn.parameters(), 100)

    avg_norm = L2_norm(policy_nn) / hp["replay_steps"]

    optimizer.step()
    return avg_norm

def training(functions, hp):
    replay = ReplayBuffer(hp["ReplayBuffer_capacity"])
    random.seed(hp["seed"])
    reward_sum = 0
    L2_norm_sum = 0
    # print(f"Episode: {episode}")
    state, _ = functions["env"].reset()
    state = torch.tensor(state, dtype=torch.float32, device=functions["device"])
    terminated = False

    for t in count():
        with torch.no_grad():
            state_qvalues = functions["policy_nn"](state)
        # print(f"\tstate_qvalues: {state_qvalues}")
        action = functions["select_action"](state_qvalues, functions, hp)


        next_state, reward, terminated, truncated, info = functions["env"].step(action)
        reward_sum += reward
        replay.push(state.tolist(), action, reward, next_state, terminated)
        state = torch.tensor(next_state, dtype=torch.float32, device=functions["device"])

        if len(replay) >= hp["minibatch_size"]:
            for _ in range(hp["replay_steps"]):
                L2_norm_sum += optimize_network(functions, hp, replay)

        TAU = hp["tau"]
        target_net_state_dict = functions["target_nn"].state_dict()
        policy_net_state_dict = functions["policy_nn"].state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        functions["target_nn"].load_state_dict(target_net_state_dict)

        if terminated or truncated:
            break
    return reward_sum
