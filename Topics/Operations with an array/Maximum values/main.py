import numpy as np

int1, int2, int3 = int(input()), int(input()), int(input())
array = np.array([int1, int2, int3])
print(array.max(), array.argmax(), sep='\n')
