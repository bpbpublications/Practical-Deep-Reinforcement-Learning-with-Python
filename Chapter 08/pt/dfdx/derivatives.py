from torch.autograd import grad

from ch8.pt.dfdx.function import get_function

f, params = get_function(2, 4, 3, 5)

df_dx1 = grad(outputs = f, inputs = [params['x1']])[0]

print(df_dx1.item())
