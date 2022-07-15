import torch
import matplotlib.pyplot as plt

x = torch.linspace(-10, 10)

# ReLU
relu = torch.nn.ReLU()
plt.title('ReLU')
plt.plot(x.tolist(), relu(x).tolist())
plt.show()

# Sigmoid
sigmoid = torch.nn.Sigmoid()
plt.title('Sigmoid')
plt.plot(x.tolist(), sigmoid(x).tolist())
plt.show()

# Tanh
tanh = torch.nn.Tanh()
plt.title('Tanh')
plt.plot(x.tolist(), tanh(x).tolist())
plt.show()
