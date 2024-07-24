import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
import random

def f(x):
    print(x)

d = {
    "f": f()
}

d["f"](3)