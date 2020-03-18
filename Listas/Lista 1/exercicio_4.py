from exercicio_1 import *
import numpy as np
from math import ceil

def main():
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
    f = open('results/result_4.txt','w',encoding='utf-8')
    results = []
    print(matA)
    results.append('A=\n%s'%np.array2string(matA,separator=' '))
    print('Resolvendo por LU')
    results.append('Resolvendo por LU')
    if check_singularity(matA): 
        print('Não é possivel resolver pois matriz possui singularidade.') 
        results.append('Não é possivel resolver pois matriz possui singularidade.') 
    else:
        x_lu = decomp_LU(matA,matB)
        print('X_T = %s'%np.array2string(x_lu,separator=' '))
        results.append('X_T = %s'%np.array2string(x_lu.transpose(),separator=' '))
    print('Resolvendo por Cholesky')
    results.append('Resolvendo por Cholesky')
    x_chol = decomp_Cholesky(matA,matB)
    if type(x_chol) is str:
        print(x_chol)
        results.append(x_chol)
    else:
        print('X_T = %s'%np.array2string(x_chol.transpose(),separator=' '))
        results.append('X_T = %s'%np.array2string(x_chol.transpose(),separator=' '))
    
    f.write('\n'.join(results))
    f.close()
if __name__ == "__main__":
    main()