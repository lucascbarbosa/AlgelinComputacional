import numpy as np
import math 
def check_singularity(a):

    return np.linalg.det(a) == 0 

def resolve(a,b,mode='back'): #AX = B return X
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

def elim(a,mode='upper'):
    n = len(a)
    m_ = np.identity(n)
    if mode ==  'upper':#get the upper triangle matrix
        for j in range(n-1):
            m = np.identity(n)
            for i in range(j+1,n):
                m[i,j] = -a[i,j]/a[j,j]
            m_ = m.dot(m_)
            a = m.dot(a)
        m_ = np.around(m_,decimals = 2)  
        a = np.around(a,decimals = 2)  
    if mode == 'lower':#get the lowe triangle matrix
        for j in range(n-1,-1,-1):
            m = np.identity(n)
            for i in range(j-1,-1,-1):
                m[i,j] = -a[i,j]/a[j,j]
            m_ = m.dot(m_)
            a = m.dot(a)
        m_ = np.around(m_,decimals = 2)  
        a = np.around(a,decimals = 2) 
    return m_,a

def decomp_LU(a,b):
    #get L and U from matrix A
    m,u = elim(a)
    l = np.linalg.inv(m)  
    y = resolve(l,b,'front')#now resolve LY = B
    x = resolve(u,y,'back')#then resolve UX = Y
    x = np.around(x,decimals = 1)
    return x

def decomp_Cholesky(a,b):
    n = len(a)
    l = np.zeros((n,n))
    for i in range(n): #algotihm
        s = 0
        s = np.inner(l[i,:i],l[i,:i]) #same as computing the norm of the vector
        try:
            l[i,i] = math.sqrt(a[i,i]-s)
        except ValueError:
            return 'Não foi possível realizar a decomposição de Cholensky'
        for j in range(i+1,n):
            s = 0
            for k in range(i):
                s += (l[i,k]*l[j,k])
            l[j,i] = (a[j,i]-s)/l[i,i]
    l = np.around(l,decimals= 2)
    u = l.transpose()
    y = resolve(l,b,'front')#resolve LY=B
    x = resolve(u,y,'back')#than LtX=|Y
    x = np.around(x,decimals = 1)
    return x



def main():
    f = open('results/result_1.txt','w',encoding='utf-8')
    results = []
    matA = input('Digite a matriz A linearizada com os valores separados por , : ').split(',')
    size = int(math.sqrt(len(matA)))
    matA = list(map(lambda x: float(x),matA)) 
    matA = np.array(matA).reshape((size,size))
    print(matA)
    results.append('A =\n%s'%np.array2string(matA,separator= ' '))
    matB = input('Digite a matriz B linearizada com os valores separados por , : ').split(',')
    matB = list(map(lambda x: float(x),matB))
    matB = np.array(matB)
    results.append('B_T = %s'%np.array2string(matB,separator= ' '))
    if check_singularity(matA): 
        print('Erro. Há singularidade na matriz informada.') 
        results.append('Erro. Há singularidade na matriz informada.') 
    else:
        print('Resolução por LU')
        results.append('Resolução por LU')
        x = decomp_LU(matA,matB)
        print('X_T = %s'%np.array2string(x.transpose(),separator=' '))
        results.append('x = %s'%np.array2string(x.transpose(),separator=' '))
        print('Resolução por Cholesky')
        results.append('Resolução por Cholesky')
        x = decomp_Cholesky(matA,matB)
        if type(x) is str:
            print(x)
            results.append(x)
        else:
            print('X_T = %s'%np.array2string(x.transpose(),separator=' '))
            results.append('x = %s'%np.array2string(x.transpose(),separator=' '))
    f.write('\n'.join(results))
    f.close()

if __name__ == "__main__":
    main()