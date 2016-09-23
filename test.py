#!/usr/bin/python
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

def solvr(y, t, K,D,M,L, in_f):
    nq= len(y)/2
    pos,vel= y[0:nq:1],y[nq:]

    vel_dot= np.linalg.inv(M)*(K*pos.reshape(nq,1) +D*vel.reshape(nq,1)- L*in_f(t))

    out= np.row_stack((vel.reshape(nq,1),vel_dot.reshape(nq,1)))
    return np.squeeze(np.asarray(out))

def main():
    k= 100
    d= 10
    m= 10 

    #A= np.matrix([[0,1],[-k/m,-c/m]])
    #B= np.zeros([2,1])

    K= k*np.matrix([[-2,1],[1,-2]])
    D= d*np.matrix([[-2,1],[1,-2]])
    M= m*np.eye(2)

    L=np.zeros([2,1])

    in_f= lambda t: 0

    tspan = np.arange(0, 10.0, 0.01)
    x0= np.array([1,0,0,0])
    asol = integrate.odeint(solvr, x0, tspan,args=(K,D,M,L,in_f))
    C= np.zeros([4,1])
    C[0]=1 
    plt.plot(tspan,np.dot(asol,C))
    plt.show()

if __name__ == '__main__':
    main()
