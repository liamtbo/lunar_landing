import torch
import numpy as np

l = [1,2,3,4]

test = {
    "l": l
}
def f(test):
    test["l"][2] = 99999

f(test)
print(l)

