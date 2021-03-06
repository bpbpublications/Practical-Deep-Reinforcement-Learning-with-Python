import torch
from torch.nn import Dropout

torch.manual_seed(1)

t = torch.randint(10, (5,)).float()
print(f'Initial Tensor: {t.numpy()}')

dropout = Dropout(p = .5)

dropout.train()
r = dropout(t)
print(f'Dropout Train: {r.numpy()}')

dropout.eval()
r = dropout(t)
print(f'Dropout Eval: {r.numpy()}')
