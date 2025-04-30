import numpy as np

def eulers_arr(diffeq, t0, y0s, deltat=1, n=10):
    if type(y0s)!=np.ndarray:
        y0s=np.array([y0s])
    points=np.zeros((1+y0s.shape[0],n+1))
    points[0,:]=t0 + np.arange(0, n+1) * deltat
    points[1:,0]=y0s
    i=1
    while i<=n:
        points[1:,i]=points[1:,i-1]+deltat*diffeq(points[0,i-1], points[1:,i-1])
        i+=1
    return points


def RK4(diffeq, t0, y0s, deltat=1, n=10):
    if type(y0s)!=np.ndarray:
        y0s=np.array([y0s])
    points=np.zeros((1+y0s.shape[0],n+1)) 
    points[0,:]=t0 + np.arange(0, n+1) * deltat #this is the array of ts
    points[1:,0]=y0s
    deltat2=deltat/2
    i=1
    while i<=n:
        t=points[0,i-1]
        ys=points[1:,i-1]
        k1=diffeq(t, ys)
        k2=diffeq(t+deltat2,ys+deltat2*k1)
        k3=diffeq(t+deltat2,ys+deltat2*k2)
        k4=diffeq(t+deltat,ys+deltat*k3)
        points[1:,i]=points[1:,i-1]+(deltat/6)*(k1+2*k2+2*k3+k4)
        i+=1
    return points


def rmse(estimates,actuals,tol=0.000000000000001):
    errors=estimates-actuals
    errors2=errors**2
    errors2s=np.zeros(errors2.shape[0])
    errors2sm=errors2.mean(axis=1)
    rmse=errors2sm**0.5
    for i in range(rmse.shape[0]):
        if rmse[i]<tol:
            rmse[i]=np.NaN
    return rmse


def maxerror(estimates,actuals,tol=0.000000000000001):
    errors=estimates-actuals
    maxes=np.max(np.abs(errors), axis=1)
    for i in range(maxes.shape[0]):
        if maxes[i]<tol:
            maxes[i]=np.NaN
    return maxes


def errorvsdeltat(solver, errorestimator, diffeq, actualfn, t0, y0s, n=10, ndt=10, dti=1):
    if type(y0s)!=np.ndarray:
        y0s=np.array([y0s])
    deltats=np.zeros(ndt)
    for i in range(deltats.shape[0]):
        if i==0:
            deltats[i]=dti
        else:
            deltats[i]=deltats[i-1]/2
    normerrors=np.zeros((y0s.shape[0],ndt))
    for i in range(deltats.shape[0]):
        estimates=solver(diffeq, t0, y0s, deltats[i], n)
        actuals=actualfn(estimates[0],y0s)
        errors=errorestimator(estimates[1:],actuals)
        normerrors[:,i]=errors/deltats[i]
    print(normerrors)
    factor=np.zeros((y0s.shape[0],ndt-1))
    for i in range(factor.shape[1]):
        factor[:,i]=normerrors[:,i]/normerrors[:,i+1]
    return factor