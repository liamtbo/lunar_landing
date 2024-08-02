import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
import random

def softmax(state_qvalues, functions, hp):
    state_qvalues_probabilities = torch.softmax(state_qvalues, dim=0)
    state_qvalues_dis = torch.distributions.Categorical(state_qvalues_probabilities)
    action = state_qvalues_dis.sample().item()
    return action

torch.manual_seed(1)


for _ in range(5):
    print(softmax(torch.tensor([2.,2.,2.,2.]), 5, 6))