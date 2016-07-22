import scipy.optimize as optimize
import numpy as np

def f(c):
    return np.sqrt(c[0]**2 + c[1]**2 + c[2]**2)

result = optimize.minimize(, [[1,1,1], [1,1,1],[1,1,1]], bounds=((0, 2), (0, 2), (0, 2), (0, 2), (0, 2), (0, 2), (0, 2), (0, 2), (0, 2)))
print(result)
