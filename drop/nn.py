from ._tensor import tensor
from collections import OrderedDict
import pickle
from ._helpers import flatten
from ._utils import _randn

class Module:
  def __init__(self) -> None:
    self._modules = OrderedDict()
    self._params = OrderedDict()
    self._grads = OrderedDict()
    self.training = True
  
  def forward(self, *inputs, **kwargs):
    raise NotImplementedError("forward not written")
  
  def __call__(self, *inputs, **kwargs):
    return self.forward(*inputs, **kwargs)
  
  def train(self):
    self.training = True
    for param in self.parameters():
      param.requires_grad = True
  
  def eval(self):
    self.training = True
    for param in self.parameters():
      param.requires_grad = False
  
  def modules(self):
    yield from self._modules.values()

  def zero_grad(self):
    for param in self.parameters():
      # print("param: ", param.grad)
      param.zero_grad()

  def parameters(self):
    params = []
    for param in self._params.values():
      params.append(param)
    for module in self._modules.values():
      params.extend(module.parameters())
    return params

  def __setattr__(self, key, value):
    if isinstance(value, Module):
      self._modules[key] = value
    elif isinstance(value, Parameter):
      self._params[key] = value
    super().__setattr__(key, value)

  def __repr__(self):
    module_str = self.__class__.__name__ + '(\n'
    for key, module in self._modules.items():
      module_str += '  (' + key + '): ' + repr(module) + '\n'
    for key, param in self._params.items():
      module_str += '  (' + key + '): Parameters: ' + str(param.tolist()) + '\n'
    module_str += ')'
    return module_str
  
  def __str__(self):
    def format_param(param):
      param_str = "Parameter containing:\n"
      if isinstance(param, tensor):
        param_str += str(param)
      else:
        param_str += str(param.tolist())
      return param_str
    
    lines = []
    for key, param in self._params.items():
      lines.append(f"{key}: {format_param(param)}")
    for key, module in self._modules.items():
      lines.append(f"{key}:\n{module}")
    return self.__repr__() + "\n" + "\n".join(lines)

  def n_param(self):
    total = 0
    for param in self.parameters():
      total += param.numel()
    return total

  def save_dict(self):
    state = OrderedDict()
    for name, param in self._params.items():
      state[name] = param.tolist()
    for name, module in self._modules.items():
      state[name] = module.save_dict()
    return state

  def save(self, filename='model.pickle'):
    with open(filename, 'wb') as f:
      pickle.dump(self.save_dict(), f)

  def load(self, filename='model.pickle'):
    with open(filename, 'rb') as f:
      state = pickle.load(f)
    self.load_dict(state)

  def load_dict(self, state):
    for name, value in state.items():
      if isinstance(value, dict):
        self._modules[name].load_dict(value)
      else:
        self._params[name].data = value

class Parameter(tensor):
  def __init__(self, shape) -> None:
    data = _randn(domain=(-1, 1), shape=shape)
    super().__init__(data)
  
  def zero_grad(self) -> None:
    self.grad.zero_grad()
  
  def tolist(self) -> list:
    return super().tolist()
  
  def numel(self) -> int:
    return len(flatten(self.data))
  
  def __repr__(self) -> str:
    return super().__repr__()
  
  def __str__(self) -> str:
    return "\nParameter containing:\n" + super().__repr__()

class Linear(Module):
  def __init__(self, _in, _out, bias=False):
    super(Linear, self).__init__()
    self.wei = Parameter(shape=(_in, _out))
    if bias:
      self.bias = Parameter(shape=(1, _out))
    else:
      self.bias = None

  def __call__(self, x):
    return self.forward(x)

  def forward(self, x):
    x = x if isinstance(x, tensor) else tensor(x, requires_grad=True)
    out = x @ self.wei
    if self.bias is not None:
      out = out + self.bias
    return out

  def parameters(self):
    params = [self.wei]
    if self.bias is not None:
      params.append(self.bias)
    return params
  
  def __repr__(self):
    return f"<LinearLayer in_features={self.wei.shape[0]} out_features={self.wei.shape[1]}>"