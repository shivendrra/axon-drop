# from drop import Tensor

# a, b = [[1, 5, -6], [1, 6, -3]], [[-2, 0, -6], [7, -2, 0]]
# a, b = Tensor(a), Tensor(b)

# c = a + b
# d = c.relu()
# e = c.tanh()

# e.backward()

# print(c)
# print(d)
# print(e)

# print(c.grad)
# print(d.grad)
# print(e.grad)

from drop import Scalar

x1, x2 = Scalar(2), Scalar(3)
x3, x4 = Scalar(5), Scalar(10)
x5, x6 = Scalar(1), Scalar(4)
x7 = Scalar(-2)

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