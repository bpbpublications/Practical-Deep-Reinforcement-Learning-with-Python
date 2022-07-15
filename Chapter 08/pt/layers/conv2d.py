import torch
from torch.nn.parameter import Parameter

A = torch.tensor([[[[1, 2, 0, 1],
                    [-1, 0, 3, 2],
                    [1, 3, 0, 1],
                    [2, -2, 1, 0]]]]).float()

# Convolution Layer
conv2d = torch.nn.Conv2d(1, 1, kernel_size = 2, bias = False)

# Setting 2x2 Kernel to Convolution Layer
conv2d.weight = Parameter(torch.tensor([[[[1, -1], [-1, 1]]]]).float())

# Executing Convolution
output = conv2d(A)

print(output.tolist())
