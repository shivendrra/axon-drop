import drop
from drop.dropy import tensor
import drop.nn as nn

# Input and target tensors
xs = tensor([
  [2.0, 3.0, -1.0],
  [3.0, 0.0, -0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0]
], requires_grad=True)
ys = tensor([1.0, -1.0, -1.0, 1.0], requires_grad=True)

# Define the model
class MLP(nn.Module):
  def __init__(self, _in, _hidden, _out) -> None:
    super().__init__()
    self.layer1 = nn.Linear(_in, _hidden, bias=False)
    self.tanh = nn.Tanh()
    self.layer2 = nn.Linear(_hidden, _out, bias=False)
  
  def forward(self, x):
    out = self.layer1(x)
    out = self.tanh(out)
    out = self.layer2(out)
    return out

# Initialize model
model = MLP(3, 10, 1)
epochs = 100
learning_rate = 0.01

# Training loop
for k in range(epochs):
  out = model(xs)
  loss = (((ys - out) ** 2 ).sum()) / 2
  model.zero_grad()
  loss.backward()
  print(f"Epoch {k}: Loss = {loss.data}")
  for p in model.parameters():
    p.data -= p.grad * learning_rate