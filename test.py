import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
import random

t1 = torch.tensor([[1,1],
                   [2,2]])
t2 = torch.tensor([[0],[1]])

print(t1 * t2)