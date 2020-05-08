import numpy as np

A = np.random.randint(0, 10, 4)
B = np.random.randint(0, 10, 5)

print(A)
print(B)

Ax, Bx = np.meshgrid(A, B)
Ax = Ax.flatten()
Bx = Bx.flatten()
print(Ax)
print(Bx)

print(np.inf)