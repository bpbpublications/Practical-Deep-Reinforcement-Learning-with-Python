import torch

player_characteristics = torch.Tensor([
    [9, 7, 9],  # Player 1
    [10, 9, 10],  # Player 2
    [8, 10, 8]  # Player 3
])

# Total Characteristics
player_total = player_characteristics.sum(dim = 1)

softmax = torch.nn.Softmax()

player_prob = softmax(player_total)

# Probabilities
print(player_prob.numpy())
