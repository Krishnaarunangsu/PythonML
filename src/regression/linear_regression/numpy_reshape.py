import numpy as np

z = np.array([[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12]])
print(z)
print(z.shape)
print(z.reshape(1, -1))
print(z.reshape(-1,1))