from .scalar import Scalar
import pickle
import random

class Module:
  def zero_grad(self):
    for p in self.parameters():
      p.grad = 0

  def parameters(self):
    return []
  
  def forward(self, *inputs, **kwargs):
    raise NotImplementedError('forward not written')

  def __call__(self, *inputs, **kwargs):
    return self.forward(*inputs, **kwargs)
  
  def save(self, filename='model.pickle'):
    with open(filename, 'wb') as f:
      pickle.dump(self.save_dict(), f)

  def load(self, filename='model.pickle'):
    with open(filename, 'rb') as f:
      state = pickle.load(f)
    self.load_dict(state)

class Neuron(Module):
  def __init__(self, _in, nonlin=True):
    self.wei = [Scalar(random.uniform(-1, 1)) for _ in range(_in)]
    self.b = Scalar(0)
    self.nonlin = nonlin
  
  def __call__(self, x):
    act = sum((wi * xi for wi, xi in zip(self.wei, x)), self.b)
    return act.sigmoid() if self.nonlin else act

  def parameters(self):
    return self.wei + [self.b]
  
  def __repr__(self) -> str:
    return f"{'Sigmoid' if self.nonlin else 'Linear'}Neuron({len(self.wei)})"

class Layer(Module):
  def __init__(self, n_in, n_out, **kwargs):
    self.neurons = [Neuron(n_in, **kwargs) for _ in range(n_out)]
  
  def __call__(self, x):
    out = [n(x) for n in self.neurons]
    return out[0] if len(out) == 1 else out
  
  def parameters(self):
    return [p for n in self.neurons for p in n.parameters()]
  
  def __repr__(self) -> str:
    neurons = ',\n\t\t'.join([str(n) for n in self.neurons])
    return f"Layer of [{neurons}]"

class MLP(Module):
  def __init__(self, n_in, n_out):
    sz = [n_in] + n_out
    self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(n_out)-1) for i in range(len(n_out))]

  def __call__(self, x):
    for layers in self.layers:
      x = layers(x)
    return x
  
  def parameters(self):
    return [p for layers in self.layers for p in layers.parameters()]

  def __repr__(self):
    layers = ',\n\t'.join([str(layer) for layer in self.layers])
    return f"MLP of [{layers}]"

class RNNCell(Module):
  def __init__(self, input_size, hidden_size, nonlin=True):
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.input_neuron = Neuron(input_size, nonlin=False)
    self.hidden_neuron = Neuron(hidden_size, nonlin=False)
    self.nonlin = nonlin
    self.input_neuron.b = Scalar(0.1)
    self.hidden_neuron.b = Scalar(0.1)

  def __call__(self, x, h):
    wx = sum((wi * xi for wi, xi in zip(self.input_neuron.wei, x)), self.input_neuron.b)
    wh = sum((wi * hi for wi, hi in zip(self.hidden_neuron.wei, h)), self.hidden_neuron.b)
    act = wx + wh
    return act.relu() if self.nonlin else act

  def parameters(self):
    return self.input_neuron.parameters() + self.hidden_neuron.parameters()

  def __repr__(self):
    return f"{'ReLU' if self.nonlin else 'Linear'}RNNCell({self.input_size}, {self.hidden_size})"

class RNN(Module):
  def __init__(self, input_size, hidden_size, output_size, num_layers=1):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.rnn_cells = [RNNCell(input_size, hidden_size) if i == 0 else RNNCell(hidden_size, hidden_size) for i in range(num_layers)]
    self.output_layer = Layer(hidden_size, output_size)

  def __call__(self, x, h=None):
    if h is None:
      h = [Scalar(0) for _ in range(self.hidden_size)]
    for rnn_cell in self.rnn_cells:
      h = [rnn_cell(x, h)]
    return self.output_layer(h)

  def parameters(self):
    return [p for rnn_cell in self.rnn_cells for p in rnn_cell.parameters()] + self.output_layer.parameters()

  def __repr__(self):
    return f"RNN of [{', '.join(str(rnn_cell) for rnn_cell in self.rnn_cells)}, {self.output_layer}]"