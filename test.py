import torch
alpha = torch.tensor([1,2,3,4,5])
beta = alpha.numpy()
print(alpha)
print(beta)
if torch.cuda.is_available():
    alpha = alpha.to('cuda')

print(alpha)
