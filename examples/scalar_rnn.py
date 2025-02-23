import drop
import drop.nn as nn

class RNN(nn.Module):
  def __init__(self, _in, _hidden, _out) -> None:
    super().__init__()
    self.hidden_size = _hidden
    self.i2h = nn.Linear(_in, _hidden, bias=False)
    self.h2h = nn.Linear(_hidden, _hidden, bias=False)
    self.h2o = nn.Linear(_hidden, _out, bias=False)
    self.tanh = nn.Tanh()
  
  def forward(self, x):
    batch_size = x.shape[0]
    h = drop.zeros(shape=(batch_size, self.hidden_size))
    for t in range(x.shape[1]):
      h = self.tanh(self.i2h(x[:, t]) + self.h2h(h))
    out = self.h2o(h)
    return out

xs = drop.tensor([
  [[2.0, 3.0, -1.0]],
  [[3.0, 0.0, -0.5]],
  [[0.5, 1.0, 1.0]],
  [[1.0, 1.0, -1.0]]
], requires_grad=True)  # Shape: [batch_size, time_steps, input_features]

ys = drop.tensor([[1.0], [-1.0], [-1.0], [1.0]], requires_grad=True)  # Shape: [batch_size, output_features]
model = RNN(3, 10, 1)
epochs = 100
learning_rate = 0.01

# Training loop
for k in range(epochs):
  out = model(xs)
  loss = (((ys - out) ** 2 ).sum()) / 2
  model.zero_grad()
  loss.backward()
  print(f"Epoch {k}: Loss = {loss.data}")
  
  # Update parameters
  for p in model.parameters():
    p.data -= p.grad * learning_rate