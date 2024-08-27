import drop

x1, x2 = drop.scalar(2, dtype=drop.int16), drop.scalar(3, dtype=drop.int16)
x3, x4 = drop.scalar(5, dtype=drop.int16), drop.scalar(10, dtype=drop.int16)
x5, x6 = drop.scalar(1, dtype=drop.int16), drop.scalar(4, dtype=drop.int16)
x7 = drop.scalar(-2, dtype=drop.int16)

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