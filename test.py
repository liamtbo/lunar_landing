import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
import random

l = []
for i in range(5):
    l.append(np.ones(shape=(5,)))
    l[i][i] = 9
l = np.array(l)
l = torch.tensor(l)
print(torch.max(l, dim=1).values)
