from drop import Tensor

a, b = [[1, 5, -6], [1, 6, -3]], [[-2, 0, -6], [7, -2, 0]]
a, b = Tensor(a), Tensor(b)

c = a + b
d = c.relu()
e = c.tanh()
f = c.swiglu()

print(c)
print(d)
print(e)
print(f)