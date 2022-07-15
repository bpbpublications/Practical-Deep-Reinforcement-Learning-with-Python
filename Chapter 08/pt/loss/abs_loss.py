import torch

a = torch.tensor([1, 2]).float()
b = torch.tensor([1, 5]).float()

abs_loss = torch.nn.L1Loss()
abs_error = abs_loss(a, b)

print(f'abs: {abs_error.item()}')
