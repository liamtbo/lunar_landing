import torch
import numpy as np

tmp1 = torch.zeros(size=(5,5))
tmp1[1] = torch.tensor([1,2,3,4,5])
print(tmp1)
print(tmp1[:,-2:].sum(1).sum()  / 5)