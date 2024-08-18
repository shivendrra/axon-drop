# from drop import Scalar

# x1, x2 = Scalar(2), Scalar(3)
# x3, x4 = Scalar(5), Scalar(10)
# x5, x6 = Scalar(1), Scalar(4)
# x7 = Scalar(-2)

# a1 = x1 + x2
# a2 = x3 - x4
# a3 = a1 * a2
# a4 = a3 ** 2
# a5 = x5 * x6
# a6 = a5.sigmoid()
# a7 = x7.tanh()
# a8 = a4 + a6
# a9 = a8 + a7
# y = a9.relu()

# y.backward()

# print(x1)
# print(x2)
# print(x3)
# print(x4)
# print(x5)
# print(x6)
# print(x7)
# print(y)

from drop import tensor

a = tensor([[2, 4, 5, -4], [-3, 0, 9, -1]])
b = tensor([[1, 0, -2, 0], [-1, 10, -2, 4]])

c = a + b
d = c.tanh()
e = d.silu()
f = e ** 2
g = f.sigmoid()
h = g.sum()

h.backward()

print("a.grad:")
print(a.grad)
print("\nb.grad:")
print(b.grad)
print("\nc.grad:")
print(c.grad)
print("\nd.grad:")
print(d.grad)
print("\ne.grad:")
print(e.grad)
print("\nf.grad:")
print(f.grad)
print("\ng.grad:")
print(g.grad)
print("\nh.grad:")
print(h.grad)

import torch

a = torch.tensor([[2.0, 4.0, 5.0, -4.0], [-3.0, 0.0, 9.0, -1.0]], requires_grad=True)
b = torch.tensor([[1.0, 0.0, -2.0, 0.0], [-1.0, 10.0, -2.0, 4.0]], requires_grad=True)

c = a + b
d = torch.tanh(c)
e = torch.nn.functional.silu(d)
f = e ** 2
g = torch.sigmoid(f)
h = g.sum()

c.retain_grad()
d.retain_grad()
e.retain_grad()
f.retain_grad()
g.retain_grad()
h.retain_grad()
h.backward()

print("a.grad:")
print(a.grad)
print("\nb.grad:")
print(b.grad)
print("\nc.grad:")
print(c.grad)
print("\nd.grad:")
print(d.grad)
print("\ne.grad:")
print(e.grad)
print("\nf.grad:")
print(f.grad)
print("\ng.grad:")
print(g.grad)
print("\nh.grad:")
print(h.grad)