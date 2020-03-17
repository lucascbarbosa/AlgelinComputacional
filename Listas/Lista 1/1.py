import numpy as np
from math import sqrt,pow
def check_singularidade(a):

    return np.linalg.det(a) == 0

def resolve(a,b,mode): #AX = B return X
    n = len(b)
    try:
        x_shape = (a.shape[1],b.shape[1])
    except IndexError: #in case b is 1-d b.shape = (l,) than its not possible to get number of collumns
        x_shape = (a.shape[1],1)
    x = np.zeros(x_shape) # A's size = mxn ,X = nxo, B =  mxo, 
    if mode == 'front':
        for i in range(n):
            limit = i
            x[i] = (b[i] - a[i,:limit].dot(x[:limit]))/a[i,i]
    if mode == 'back':
        for i in range(n-1,-1,-1):
            limit = i+1
            x[i] = (b[i]-a[i,limit:].dot(x[limit:]))/a[i,i]
    x = np.around(x,decimals = 2) 
    return x
 

def decomp_LU(a,b):
    #get L and U from matrix A
    n = len(a)
    m_ = np.identity(n)
    for j in range(n-1):
        m = np.identity(n)
        for i in range(j+1,n):
            m[i,j] = -a[i,j]/a[j,j]
        m_ = m.dot(m_)
        a = m.dot(a)
    l = np.linalg.inv(m_)
    l = np.around(l,decimals = 2)  
    a = np.around(a,decimals = 2)  
    y = resolve(l,b,'front')#now resolve LY = B
    x = resolve(a,y,'back')#then resolve UX = Y
    return x

def decomp_Cholesky(a,b):
    n = len(a)
    l = np.zeros((n,n))
    for i in range(n):
        s = 0
        s = l[i,:i].dot(l[i,:i]) #same as computing the norm of the vector
        l[i,i] = sqrt(a[i,i]-s)

        for j in range(i+1,n):
            s = 0
            for k in range(i):
                s += (l[i,k]*l[j,k])
            l[j,i] = (a[j,i]-s)/l[i,i]
    l = np.around(l,decimals= 2)
    y = resolve(l,b,'front')#resolve LY=B
    x = resolve(l.transpose(),y,'back')#than LtX=|Y
    return x


def main():
    a = input('Digite a matriz A linearizada com os valores separados por , : ').split(',')
    size = int(sqrt(len(a)))
    a = list(map(lambda x: int(x),a))
    a = np.array(a).reshape((size,size))
    b = input('Digite a matriz B linearizada com os valores separados por , : ').split(',')
    b = list(map(lambda x: int(x),b))
    b = np.array(b)
    if check_singularidade(a): print ('Erro. Há singularidade na matriz informada.') 
    else:
        print('Resolução por LU')
        x = decomp_LU(a,b)
        print('x = %s'%np.array2string(x,separator=' '))
        print('Resolução por Cholesky')
        x = decomp_Cholesky(a,b)
        print('x = %s'%np.array2string(x,separator=' '))

if __name__ == "__main__":
    main()