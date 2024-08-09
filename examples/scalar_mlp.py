from drop import Scalar, nn

xs = [
  [2.0, 3.0, -1.0],
  [3.0, 0.0, -0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0]
]
ys = [1.0, -1.0, -1.0, 1.0]
model = nn.MLP(3, [4, 4, 1])

epochs = 100
for k in range(epochs):
  out = [model(x) for x in xs]
  loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, out))

  model.zero_grad()
  loss.backward()
  
  for p in model.parameters():
    p.data -= 0.001 * p.grad
  print(k, " -> ", loss.data)