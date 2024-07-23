import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

qvals = torch.FloatTensor([[1,2,3,4],
                           [5,6,7,8]])
actions = torch.tensor([1,3], dtype=torch.int)

print(qvals[torch.arange(qvals.size(0)), actions])