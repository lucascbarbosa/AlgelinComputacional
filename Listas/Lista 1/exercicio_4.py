from exercicio_1 import *
import numpy as np
from math import ceil

matA = np.zeros((10,10))
matB = np.array([4,0,8,0,12,0,8,0,4,0])
mult = 1
offset = 16
#edit matA
for i in range(10):
    if i == 3:
        mult = -1
        offset = 22
    for j in range(10):
        matA[i,j] = 10-abs(i-j)
        matA[i,i] = offset+mult*i
print(matA)
x = decomp_LU(matA,matB)
print('x = %s'%np.array2string(x,separator=' '))
print('Prova Real')
print(np.around(matA.dot(x),decimals = 0))