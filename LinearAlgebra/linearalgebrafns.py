import numpy as np

def multiplyvectors(v1,v2):
    if not (v1.ndim==v2.ndim==1) or not (v1.shape[0]==v2.shape[0]):
        return "can't do it"
    total=0
    for i in range(v1.shape[0]):
        total+=v1[i]*v2[i]
    return total

def multiplymatrices(matrix1,matrix2):
    if not (matrix1.ndim==matrix2.ndim==2) or not (matrix1.shape[1]==matrix2.shape[0]):
        return "can't do it"
    numrows=matrix1.shape[0]
    numcolumns=matrix2.shape[1]
    soln=np.zeros((numrows,numcolumns))
    for i in range(numrows):
        for j in range(numcolumns):
            soln[i,j]=multiplyvectors(matrix1[i,:],matrix2[:,j])
    return soln

def GE(matrix, vector):
    numrows=matrix.shape[0]
    matrix=matrix.astype(float)
    vector=vector.astype(float)
    for i in range(numrows):
        k=i+1
        while matrix[i,i]==0:
            matrix[[i,k]]=matrix[[k,i]]
            vector[[i,k]]=vector[[k,i]]
            k+=1
        matrix[i,:], vector[i] = matrix[i,:]/matrix[i,i], vector[i]/matrix[i,i]
        for j in range(i+1, numrows):
            matrix[j,:], vector[j] = matrix[j,:]-(matrix[j,i]*matrix[i,:]), vector[j]-(matrix[j,i]*vector[i])
    return matrix, vector

def BS(matrix, vector): #only apply to results of GE!! must be upper triangular with 1s on the diagonal
    numrows=matrix.shape[0]
    solns=np.zeros(numrows)
    for i in np.arange(numrows)[::-1]: 
        rightsum=0
        for j in range(i+1,numrows):
            rightsum+=matrix[i,j]*solns[j]
        solns[i]=vector[i]-rightsum
    return solns

def LU(matrix): #this will not work if a 0 appears on the diagonal
    numrows=matrix.shape[0]
    U=matrix
    L=np.identity(numrows)
    for i in range(numrows):
        Li=np.identity(numrows)
        Li[i,i]=Li[i,i]/U[i,i]
        L[i,i]=L[i,i]*U[i,i]
        for j in range(i+1, numrows):
            Li[j,i]=-U[j,i]/U[i,i]
            L[j,i]=U[j,i]
        U=multiplymatrices(Li,U)
    return L,U

def FS(matrix, vector): #only apply to lower triangular matrix
    numrows=matrix.shape[0]
    solns=np.zeros(numrows)
    for i in np.arange(numrows): 
        rightsum=0
        for j in range(0,i):
            rightsum+=matrix[i,j]*solns[j]
        solns[i]=(vector[i]-rightsum)/matrix[i,i]
    return solns

def solve_LU(L,U,v):
    y=FS(L,v)
    x=BS(U,y)
    return x