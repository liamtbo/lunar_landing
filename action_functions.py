import torch
import random
import math


def e_greedy(q_values, functions, hp):
    env = functions["env"]
    device = functions["device"]
    global steps_done
    eps_threshold = hp["eps_end"] + (hp["eps_start"] - hp["eps_end"]) \
                    * math.exp(-1. * steps_done / hp["eps_decay"])
    steps_done += 1
    sample = random.random()
    if sample > eps_threshold:
        with torch.no_grad():
            return q_values.max(0).indices.item()
    else:
        return torch.tensor([env.action_space.sample()], device=device, dtype=torch.long).item()

def softmax(state_qvalues, functions, hp):
    tau = hp["tau"]
    state_qvalues_probabilities = torch.softmax(state_qvalues / tau, dim=0)
    # print(f"\tstate_qvalues_probabilites: {state_qvalues_probabilities}")
    state_qvalues_dis = torch.distributions.Categorical(state_qvalues_probabilities)
    action = state_qvalues_dis.sample().item()
    return action