# Drop

Drop is a part of the Axon project. It is an autograd library that uses scalar-level autograd instead of tensor-level autograd, which is essentially a Python tensor class wrapper over scalar value classes. The core scalar operations are implemented in C/C++, making it faster and more efficient while supporting additional functions.

``tensor`` class is a wrapper over ``scalar`` class written in pure python, though the actual c-version implementation exist in *tensor.cpp* which works properly and can be used for same purpose but would work faster than the python version.

### ***FunFact***: this project is developed almost (90%) by GPT-4o and o3-mini as an experiment to check the understanding, reasoning and ability of language models for big and logically complex projects like this one

## Bounty:
Solve the bug for the reward. More info in this [Git Issue](https://github.com/shivendrra/axon-drop/issues/19)

## Features

- **Basic Arithmetic Operations**: Addition, subtraction, multiplication, division, exponentiation.
- **Common Mathematical Functions**: ReLU, sigmoid, tanh, SiLU, and more.
- **Automatic Gradient Computation**: Supports backpropagation for both scalar and tensor operations.
- **Efficient and Fast**: Core operations implemented in C/C++.

## Installation

Install library from PyPI.org:

```
pip install axon-drop 
```

Clone this repository and build the library:

```bash
git clone https://github.com/shivendrra/axon-drop.git
cd drop
```

## Scalar
The Scalar library is a simple implementation of scalar operations with automatic gradient computation. It supports basic operations like addition, multiplication, exponentiation, and common functions such as ReLU, sigmoid, and tanh. The library also includes backpropagation functionality for gradient updates.

### Usage

Here's a simple example demonstrating how to use the Scalar library:

```python
from drop import scalar

# Initialize scalars
x1 = scalar(2)
x2 = scalar(3)

# Perform operations
a1 = x1 + x2
a2 = x1 - x2
y = (a1 * a2).tanh()

# Perform backpropagation
y.backward()

# Print gradients
print(x1.grad)  # Gradient of x1
print(x2.grad)  # Gradient of x2
```

## Tensor

The Tensor class extends the capabilities of the Scalar class to support multi-dimensional arrays, similar to PyTorch's `Tensor` class. It allows for more complex operations and is essential for implementing neural networks or any machine learning models that require multi-dimensional data.

### Usage

Here's a simple example demonstrating how to use the Tensor class:

```python
from drop import tensor

# Initialize tensors
a = tensor([[2, 4, 5, -4], [-3, 0, 9, -1]])
b = tensor([[1, 0, -2, 0], [-1, 10, -2, 4]])

# Perform operations
c = a + b
d = c.tanh()
e = d.silu()
f = e ** 2
g = f.sigmoid()
h = g.sum()

# Perform backpropagation
h.backward()

# Print gradients
print("Gradients of a:\n", a.grad)
print("Gradients of b:\n", b.grad)
```

### Explanation:

- **Tensor Initialization**: Tensors are initialized with multi-dimensional arrays, and gradients are automatically set up for each operation.
- **Operations**: The example demonstrates basic operations (`+`, `**`, etc.), as well as more advanced functions (`tanh`, `silu`, `sigmoid`).
- **Backpropagation**: The `.backward()` function computes gradients for all tensors involved in the computation graph.

## Contributing

Feel free to open issues or submit pull requests if you have any improvements or bug fixes!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.