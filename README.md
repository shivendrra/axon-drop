# Drop

Drop is a part of Axon project. it is an autograd library that uses scalar level autograd instead of tensor level autograd, which is basically a python tensor class wrapper over scalar value class which is micrograd code written in c/c++ but has more functions & is faster.

## Features

- Basic arithmetic operations: addition, subtraction, multiplication, division, exponentiation
- Common mathematical functions: ReLU, sigmoid, tanh
- Automatic gradient computation and backpropagation

## Installation

Clone this repository and build the library:

```bash
git clone https://github.com/shivendrra/axon-drop.git
cd drop
```

## Scalar
The Scalar library is a simple implementation of scalar operations with automatic gradient computation. It supports basic operations like addition, multiplication, exponentiation, and common functions such as ReLU, sigmoid, and tanh. The library also includes backpropagation functionality for gradient updates.

## Usage

Here's a simple example demonstrating how to use the Scalar library:

```python
from scalar import Scalar

# Initialize scalars
x1 = Scalar(2)
x2 = Scalar(3)

# Perform operations
a1 = x1 + x2
a2 = x3 - x4
y = (a1 * a2).tanh()

# Perform backpropagation
y.backward()

# Print results
print(x1)
print(x2)
print(y)
```

## Contributing

Feel free to open issues or submit pull requests if you have any improvements or bug fixes!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.