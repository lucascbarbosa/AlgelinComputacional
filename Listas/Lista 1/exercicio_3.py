from exercicio_1 import *
from exercicio_2 import *

def main():
    f = open('results/result_3.txt','w',encoding='utf-8')  
    results = []
    matA = np.array([5,-4,1,0,-4,6,-4,1,1,-4,6,-4,0,1,-4,5]).reshape((4,4))
    matB = np.array([-1,2,1,3])
    print('a) Resolver o Sistema: ')
    results.append('a) Resolver o Sistema: ')
    print('1) Eliminação de Gauss')
    results.append('1) Eliminação de Gauss')
    if check_singularity(matA):
        print('Não é possível, pois a matriz possui singularidade.')
        results.append('Não é possível, pois a matriz possui singularidade.')
    else:
        m,u = elim(matA)
        x = resolve(u,m.dot(matB))
        print('X_T = %s'%np.array2string(x.transpose(),separator=' '))
        results.append('X_T = %s'%np.array2string(x.transpose(),separator=' '))
    print('2) Eliminação de Gauss-Jordan')
    results.append('2) Eliminação de Gauss-Jordan')
    if check_singularity(matA):
        print('Não é possível, pois a matriz possui singularidade.')
        results.append('Não é possível, pois a matriz possui singularidade.')
    else:
        m1,u = elim(matA)
        m2,diag = elim(u,mode='lower')
        x = resolve(diag,m2.dot(m1.dot(matB)))
        print('X_T = %s'%np.array2string(x.transpose(),separator=' '))
        results.append('X_T = %s'%np.array2string(x.transpose(),separator=' '))

    print('3) Decomposição LU')
    results.append('3) Decomposição LU')
    if check_singularity(matA):
        print('Não é possível, pois a matriz possui singularidade.')
        results.append('Não é possível, pois a matriz possui singularidade.')
    else:
        x = decomp_LU(matA,matB)
        print('X_T = %s'%np.array2string(x.transpose(),separator=' '))
        results.append('X_T = %s'%np.array2string(x.transpose(),separator=' '))

    print('4) Decomposição Cholesky')
    results.append('4) Decomposição Cholesky')
    x = decomp_Cholesky(matA,matB)
    if type(x) is str:
        print(x)
        results.append(x)
    else:
        print('X_T = %s'%np.array2string(x.transpose(),separator=' '))
        results.append('X_T = %s'%np.array2string(x.transpose(),separator=' '))

    print('5) Modo Iterativo de Jacobi')
    results.append('5) Modo Iterativo de Jacobi')
    if not converge_jacobi(matA):
        print('Não é possível, pois a matriz não converge para Jacobi.') 
        results.append('Não é possível, pois a matriz não converge para Jacobi.') 
    else:
        x = jacobi(matA,matB)
        print('X_T = %s'%np.array2string(x.transpose(),separator=' '))
        results.append('X_T = %s'%np.array2string(x.transpose(),separator=' '))

    print('6) Modo Iterativo Gauss-Seidel')
    results.append('6) Modo Iterativo Gauss-Seidel')
    if not converge_gauss(matA):
        print('Não é possível, pois a matriz não converge para Gauss-Seidel.') 
        results.append('Não é possível, pois a matriz não converge para Gauss-Seidel.') 
    else:
        x = gauss_seidel(matA,matB,tol=0.00001)
        print('X_T = %s'%np.array2string(x.transpose(),separator=' '))
        results.append('X_T = %s'%np.array2string(x.transpose(),separator=' '))

    print('b) Obter A-1 através do método de Gauss- Jordan')
    results.append('b) Obter A-1 através do método de Gauss- Jordan')
    if check_singularity(matA):
        print('Não é possível, pois a matriz possui singularidade.') 
        results.append('Não é possível, pois a matriz possui singularidade.') 
    else:

        m1,u = elim(matA)
        m2,diag = elim(u,mode='lower')
        m3 = np.identity(len(diag)).dot(np.linalg.inv(diag))
        a_inv = m3.dot(m2.dot(m1))
        print(a_inv)
        results.append('A-1 =\n%s'%np.array2string(a_inv,separator=' '))
        print('Prova Real A-1 * A = I')
        results.append('Prova Real A-1 * A = I')
        print('%s' %np.array2string(np.around(a_inv.dot(matA),decimals=0),separator = ' '))
        results.append('%s' %np.array2string(np.around(a_inv.dot(matA),decimals=0),separator = ' '))
    f.write('\n'.join(results))
    f.close()
if __name__ == "__main__":
    main()