import drop

x1, x2 = drop.Scalar(2), drop.Scalar(3)
x3, x4 = drop.Scalar(5), drop.Scalar(10)
x5, x6 = drop.Scalar(1), drop.Scalar(4)
x7 = drop.Scalar(-2)

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

print("x1: ", x1)
print("x2: ", x2)
print("x3: ", x3)
print("x4: ", x4)
print("x5: ", x5)
print("x6: ", x6)
print("x7: ", x7)
print("y: ",y)

print(x1)
x1.data = 9
print(x1)