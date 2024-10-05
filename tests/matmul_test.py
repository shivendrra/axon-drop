import timeit
import drop
import numpy as np

# Generate random matrices
a_axon = drop.randn(shape=(100, 100))
a = drop.tensor(a_axon, dtype=drop.int8)

# Measure timeit for axon matrix multiplication
start_timeit_axon = timeit.default_timer()
result_axon = drop.matmul(a_axon, a_axon)
# result_axon = a @ a
end_timeit_axon = timeit.default_timer()

# Measure timeit for numpy matrix multiplication
start_timeit_numpy = timeit.default_timer()
result_numpy = np.matmul(a_axon, a_axon)
end_timeit_numpy = timeit.default_timer()

# Print the results
print("Axon matmul result:")
print(result_axon.shape)

print("\nNumpy matmul result:")
print(result_numpy.shape)

print("\ntimeit taken by Axon matmul: {:.6f} seconds".format(end_timeit_axon - start_timeit_axon))
print("timeit taken by Numpy matmul: {:.6f} seconds".format(end_timeit_numpy - start_timeit_numpy))