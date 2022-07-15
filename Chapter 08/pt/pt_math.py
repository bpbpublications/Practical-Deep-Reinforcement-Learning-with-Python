import torch

x = torch.ones((1, 2))  # (1, 1)

y = torch.range(0, 1)  # (0, 1)

# implicit addition
z = x + y  # (1, 2)

# explicit addition
w = z.add(y)  # (1, 3)

# implicit multiplication
k = w * -1  # (-1, -3)

# absolute value
a = k.abs()  # (1, 3)

# implicit division
b = a / 2  # (0.5, 1.5)

# Rounding to nearest integer lower than
c = b.floor()  # (0, 1)

# Rounding to nearest integer greater than
d = b.ceil()  # (1, 2)

# Computes element-wise equality
eq = c.eq(d)  # (False, False)

# Mean tensor value
avg = d.mean()  # 1.5

# Max tensor value
mx = d.max()  # 2

# Min tensor value
mn = d.min()  # 1

# Sum of all tensor values
sm = d.sum()  # 3
