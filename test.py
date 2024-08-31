# import drop
# from drop import tensor

# a = tensor([[2, 4, 5, -4], [-3, 0, 9, -1]], dtype=drop.float32)
# b = tensor([[1, 0, -2, 0], [-1, 10, -2, 4]], dtype=drop.float32)

# c = a + b
# d = c.tanh()
# e = d.silu()
# f = e ** 2
# g = f.sigmoid()
# h = g.sum()

# h.backward()

# print("a.grad:")
# print(a.grad)
# print("\nb.grad:")
# print(b.grad)
# print("\nc.grad:")
# print(c.grad)
# print("\nd.grad:")
# print(d.grad)
# print("\ne.grad:")
# print(e.grad)
# print("\nf.grad:")
# print(f.grad)
# print("\ng.grad:")
# print(g.grad)
# print("\nh.grad:")
# print(h.grad)

import drop
import drop.nn as nn

xs = drop.tensor([
  [2.0, 3.0, -1.0],
  [3.0, 0.0, -0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0]
])
ys = drop.tensor([1.0, -1.0, -1.0, 1.0])

class MLP(nn.Module):
  def __init__(self, _in, _hidden, _out) -> None:
    super().__init__()
    self.layer1 = nn.Linear(_in, _hidden, bias=False)
    self.layer2 = nn.Linear(_hidden, _out, bias=False)
  
  def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    return out

model = MLP(3, 10, 1)

# out = model(xs)
# loss = (((ys - out) ** 2 )).sum()
# loss.backward()

# model.zero_grad()
# print(model)
# for p in model.parameters():
#   p.data -= p.grad * 0.1
# print(model)

epochs = 100
for k in range(epochs):
  out = model(xs)
  loss = (((ys - out) ** 2 )).sum()

  model.zero_grad()
  loss.backward()
  
  for p in model.parameters():
    p.data -= p.grad * 0.001
  print(k, " -> ", loss)