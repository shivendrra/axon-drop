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

a = Scalar(2.0)
b = Scalar(3.0)

c = a + b
d = a * b
# e = d.relu()
# f = d.tanh()
# h = d.sigmoid()

print(a)
print(b)
print(c)
print(d)
# print(e)
# print(f)
# print(h)

d.backward()

print(a.grad)
print(b.grad)
print(c.grad)
print(d.grad)
# print(e.value.contents._prev_size)
# print(f.value.contents._prev_size)
# print(h.value.contents._prev_size)