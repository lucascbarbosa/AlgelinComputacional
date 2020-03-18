import numpy as np
from math import sqrt

a = np.array([3,-1,-1,-1,3,-1,-1,-1,3]).reshape((3,3))
b = np.array([1,2,1])

def converge_jacobi(a):
    n = len(a)
    for i in range(n):
        for j in range(n):
            if abs(a[i,i]) < abs(np.sum(np.delete(a[i],i))) or abs(a[i][i]) < abs(np.sum(np.delete(a[i],i))): return False
            else:
                pass
    return True

def converge_gauss(a):
    if np.all(np.linalg.eigvals(a) > 0):
        return True
    else:
        if converge_jacobi(a):
            return True
        else:
            return False
    

def jacobi(a,b,tol=0.01):
    n = len(a)
    x= np.ones(n) #we use b as a initial vector x
    x_new = np.array(list(x)) #the next vector x
    r = 1
    iteration_round = 0
    while r > tol:
        for i in range(n):
            a_except = np.delete(a[i],i)# array ai except the aii
            x_except = np.delete(x,i)#array x except xi
            s = a_except.dot(x_except)#the sum of the product aij*xj for j [1,n], except j = i
            x_new[i] = (b[i] - s)/a[i,i] 
        r = np.around((np.linalg.norm(x_new-x)/np.linalg.norm(x_new)),decimals=10)
        x = np.array(list(x_new))
        print('iter = %d x = %s r = %f'%(iteration_round,np.array2string(x),r))
        iteration_round+=1 
    x_new = np.around(x_new,decimals=3)
    return x_new


def gauss_seidel(a,b,tol=0.01):
    n = len(a)
    x = np.ones(n) #we use b as a initial vector x
    x_new = np.array(list(x))#the next vector x
    r = 1
    iteration_round = 0
    while r > tol:
        for i in range(n):
            a_except = np.delete(a[i],i)# array ai except the aii
            x_except = np.delete(x_new,i)#array x_new except xi,instead of x
            s = a_except.dot(x_except)#the sum of the product aij*xj for j [1,n], except j = i
            x_new[i] = (b[i] - s)/a[i,i]
        r = np.linalg.norm(x_new-x)/np.linalg.norm(x_new)
        print('iter = %d x = %s r = %f'%(iteration_round,np.array2string(x),r))
        x = np.array(list(x_new))
        iteration_round += 1
    x_new = np.around(x_new,decimals=2)
    return x_new


def main():
    f = open('results/result_2.txt','w',encoding='utf-8')
    results = []
    a = input('Digite a matriz A linearizada com os valores separados por , : ').split(',')
    size = int(sqrt(len(a)))
    a = list(map(lambda x: int(x),a))
    a = np.array(a).reshape((size,size))
    b = input('Digite a matriz B linearizada com os valores separados por , : ').split(',')
    b = list(map(lambda x: int(x),b))
    b = np.array(b)
    print('Resolução por Jacobi')
    results.append('Resolução por Jacobi')
    if not converge_jacobi(a): 
        print ('Erro. Não converge para Jacobi.') 
        results.append('Erro. Não converge para Jacobi.')
    else:
        x = jacobi(a,b,tol=0.00001)
        print('x = %s'%np.array2string(x.transpose(),separator=' '))
        results.append('x = %s'%np.array2string(x.transpose(),separator=' '))
    print('Resolução por Gauss-Seidel')
    results.append('Resolução por Gauss-Seidel')
    if not converge_gauss(a): 
        print('Erro. Não converge para Gauss-Seidel.')
        print('Erro. Não converge para Gauss-Seidel.')
    else:
       
        x = gauss_seidel(a,b,tol=0.00001)
        print('x = %s'%np.array2string(x.transpose(),separator=' '))
        results.append('x = %s'%np.array2string(x.transpose(),separator=' '))
    f.write('\n'.join(results))

if __name__ == "__main__":
    main()