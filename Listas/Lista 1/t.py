import numpy as np

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


a = np.array([5,-4,1,0,-4,6,-4,1,1,-4,6,-4,0,1,-4,5]).reshape((4,4)) 
b = np.array([-1,2,1,3])
print(a)
print(b)
print(gauss_seidel(a,b,tol=0.00001))
