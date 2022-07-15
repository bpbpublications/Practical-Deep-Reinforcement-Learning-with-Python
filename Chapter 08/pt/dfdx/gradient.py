from torch.autograd import grad

from ch8.pt.dfdx.function import get_function

f, params = get_function(2, 4, 3, 5)

# Gradient as Tensor List
dy_dx = grad(outputs = f, inputs = params.values())

# Converting to Numpy List
g = [d.item() for d in dy_dx]

print(g)
