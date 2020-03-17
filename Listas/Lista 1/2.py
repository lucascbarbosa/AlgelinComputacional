import numpy as np


a = np.array([1,2,3,4]).reshape((2,2))

def converge(a):
    n = len(a)
    for i in range(n):
        for j in range(n):
            if a[i,j] < np.sum(np.delete(a[i,j])) or a[i][j] < np.sum(np.delete(a[j],j))