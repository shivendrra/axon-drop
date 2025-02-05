from drop import tensor

a = tensor([[2, 4, 5, -4], [-3, 0, 9, -1]])
b = tensor([[1, 0, -2, 0], [-1, 10, -2, 4]])

c = a + b
d = c.tanh()
e = d.silu()
f = e ** 2
g = f.sigmoid()
h = g.sum()

h.backward()

print("a.grad:")
print(a.grad)
print("\nb.grad:")
print(b.grad)
print("\nc.grad:")
print(c.grad)
print("\nd.grad:")
print(d.grad)
print("\ne.grad:")
print(e.grad)
print("\nf.grad:")
print(f.grad)
print("\ng.grad:")
print(g.grad)
print("\nh.grad:")
print(h.grad)