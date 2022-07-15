import torch

a = torch.tensor([1, 2]).float()
b = torch.tensor([1, 5]).float()

mse_loss = torch.nn.MSELoss()
mse_error = mse_loss(a, b)

print(f'mse: {mse_error.item()}')
