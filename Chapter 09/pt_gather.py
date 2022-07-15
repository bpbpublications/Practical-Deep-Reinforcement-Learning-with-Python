import torch

params = torch.tensor([[3, 4]])
idx = torch.tensor([[1]])
out = torch.gather(params, dim = 1, index = idx)

print(out.numpy())
# [[1]
#  [4]]
