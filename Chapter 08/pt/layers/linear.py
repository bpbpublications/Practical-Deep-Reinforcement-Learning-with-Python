import torch

x = torch.tensor(data = [1, 2, 3]).float()

linear = torch.nn.Linear(3, 2)

linear.weight = torch.nn.Parameter(
    torch.tensor([[0, 2, 5], [1, 0, 2]]).float()
)
linear.bias = torch.nn.Parameter(
    torch.tensor([1, 1]).float()
)

y = linear(x)

print(y.tolist())
