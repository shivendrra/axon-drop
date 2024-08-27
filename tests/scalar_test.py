"""
  doesn't has assert functions, just manually match the outputs to the rest
"""

import torch

x1, x2 = torch.tensor([2.0], requires_grad=True), torch.tensor([3.0], requires_grad=True)
x3, x4 = torch.tensor([5.0], requires_grad=True), torch.tensor([10.0], requires_grad=True)
x5, x6 = torch.tensor([1.0], requires_grad=True), torch.tensor([4.0], requires_grad=True)
x7 = torch.tensor([-2.0], requires_grad=True)

a1 = x1 + x2
a2 = x3 - x4
a3 = a1 * a2
a4 = a3 ** 2
a5 = x5 * x6
a6 = a5.sigmoid()
a7 = x7.tanh()
a8 = a4 + a6
a9 = a8 + a7
y = a9.relu()

a1.retain_grad()
a2.retain_grad()
a3.retain_grad()
a4.retain_grad()
a5.retain_grad()
a6.retain_grad()
a7.retain_grad()
a8.retain_grad()
a9.retain_grad()
y.retain_grad()
y.backward()

print(x1.data, x1.grad)
print(x2.data, x2.grad)
print(x3.data, x3.grad)
print(x4.data, x4.grad)
print(x5.data, x5.grad)
print(x6.data, x6.grad)
print(x7.data, x7.grad)
print(y.data, y.grad)

from drop import scalar

x1, x2 = scalar(2), scalar(3)
x3, x4 = scalar(5), scalar(10)
x5, x6 = scalar(1), scalar(4)
x7 = scalar(-2)

a1 = x1 + x2
a2 = x3 - x4
a3 = a1 * a2
a4 = a3 ** 2
a5 = x5 * x6
a6 = a5.sigmoid()
a7 = x7.tanh()
a8 = a4 + a6
a9 = a8 + a7
y = a9.relu()

y.backward()

print(x1)
print(x2)
print(x3)
print(x4)
print(x5)
print(x6)
print(x7)
print(y)