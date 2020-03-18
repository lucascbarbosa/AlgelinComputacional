import numpy as np


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

def jacobi(a,b,x,tol=0.01):
    n = len(a)
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
    x_new = np.around(x_new,decimals=2)
    return x_new


def gauss_seidel(a,b,x,tol=0.01):
    n = len(a)
    print(x)
    x_new = np.array(list(x))#the next vector x
    r = 1
    iteration_round = 0
    while r > tol:
        for i in range(n):
            a_except = np.delete(a[i],i)# array ai except the aii
            x_except = np.delete(x_new,i)#array x_new except xi,instead of x
            s = a_except.dot(x_except)#the sum of the product aij*xj for j [1,n], except j = i
            x_new[i] = (b[i] - s)/a[i,i]
        print(np.linalg.norm(x_new-x),np.linalg.norm(x_new))
        r = np.linalg.norm(x_new-x)/np.linalg.norm(x_new)
        x = np.array(list(x_new))
        print('iter = %d x = %s r = %f'%(iteration_round,np.array2string(x),r))
        iteration_round += 1
    x_new = np.around(x_new,decimals=2)
    return x_new

print('JACOBI')
print(jacobi(a,b,np.ones(len(a)),0.0001))
print('GAUSS-SEIDEL')
print(gauss_seidel(a,b,np.ones(len(a)),0.0001))