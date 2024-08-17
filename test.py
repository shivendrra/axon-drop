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

# print(a.grad)
# print(b.grad)
# print(c.grad)
# print(d.grad)

def backward(data):
  def __back(item):
    if isinstance(item, list):
      return [__back(row) for row in item]
    else:
      print('prev: ', item.prev[0].prev)
      return item
  return __back(data)

print(backward(d.data))
# d.backward()
# print(a.grad)
# print(b.grad)
# print(c.grad)
# print(d.grad)