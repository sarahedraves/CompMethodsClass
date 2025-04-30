import numpy as np

def Kernel(p1,p2,sigma): #both p1 and p2 need to be a 1x2 array (so they have x and y values)
    distance=((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5 #pythag
    return np.exp(-(distance**2)/(2*sigma**2)) #kernel function

def pdot(p,Ps,K,sigma): #current point p has dimension 1x2 and other points Ps have dimension nx2
    numerator=0
    denominator=0
    n=Ps.shape[0]
    for i in range(n):
        pi=Ps[i]
        numerator+=K(p,pi,sigma)*pi
        denominator+=K(p,pi,sigma)
    return (numerator/denominator)-p

def Pdot(Ps,K,sigma):
    n=Ps.shape[0]
    deltas=np.zeros((n,2))
    for i in range(n):
        deltas[i]=pdot(Ps[i],np.delete(Ps,i,axis=0),K,sigma)
    return deltas

def meanshiftevolution(P0s,K,sigma,numt):
    t=0
    newPs=P0s
    while t<numt:
        newPs=newPs+Pdot(newPs,Kernel,sigma)
        t+=1
    return newPs