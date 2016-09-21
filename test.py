#!/usr/bin/python
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

def solvr(Y, t, A, B, in_f):
    out= A*Y.reshape(2,1) + B*in_f(t)
    return np.squeeze(np.asarray(out))

def main():
    k= 100
    c= 10
    m= 10 

    A= np.matrix([[0,1],[-k/m,-c/m]])
    B= np.zeros([2,1])

    in_f= lambda t: 0

    tspan = np.arange(0, 10.0, 0.01)
    x0= np.array([1,0])
    asol = integrate.odeint(solvr, x0, tspan,args=(A,B,in_f))
    plt.plot(tspan,asol)
    plt.show()


if __name__ == '__main__':
    main()
