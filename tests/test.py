import random as r
import numpy as np
import torch

from src.main import replay_buffer


replay = replay_buffer(10, 5)
for i in range(10):
    s = np.random.rand(5)
    next_s = np.random.rand(5)
    replay.add(s, 1, 2, next_s)
print(replay.sample())




