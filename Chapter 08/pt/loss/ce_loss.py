import torch

prob_vector = torch.tensor([[0.85, 0.15]]).float()

# number of correct class
correct_class = torch.tensor([0])

ce_loss = torch.nn.CrossEntropyLoss()
ce_error = ce_loss(prob_vector, correct_class)

print(f'ce: {ce_error.item()}')
