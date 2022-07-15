import torch

x = torch.rand((1, 16, 10))

y = torch.flatten(x, start_dim = 1)

print(f'initial shape: {x.shape}')
print(f'flatten shape: {y.shape}')
