# Python Program illustrating
# numpy.ravel() method

import numpy as np

array = np.arange(15).reshape(3, 5)
print("Original array : \n", array)

# Output comes like [ 0  1  2 ..., 12 13 14]
# as it is a long output, so it is the way of
# showing output in Python
print("\nravel() : ", array.ravel())

# This shows array.ravel is equivalent to reshape(-1, order=order).
print("\nnumpy.ravel() == numpy.reshape(-1)")
print("Reshaping array : ", array.reshape(-1))