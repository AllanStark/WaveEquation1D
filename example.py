#!/usr/bin/python
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from WaveEquation1D import WaveEquation1D as Wave1D

def main():
    n=12-2
    L= np.zeros([n,1])

    WaveObj= Wave1D(n+2,1,1,1,0.01,L)

    in_f= lambda t: 0
    tmp= np.linspace(0,1,n+2)
    x0= np.zeros([n*2])
    x0[0:n:1]= np.sin(np.pi*tmp[1:n+1:1])
    WaveObj.solve(in_f,x0,plot_flag=1)

    #plt.plot(np.arange(0,10.0,0.01),WaveObj.x_sol[:,5])
    #plt.show()

    #print WaveObj.x_sol.shape   
    #print WaveObj.x_sol[0,0:n:1].reshape(1,10).shape
    #plt.ion()

    #for i in range(WaveObj.x_sol.shape[0]):
    #    plt.axis([0, 1, -1, 1])
    #    out= np.c_[0,WaveObj.x_sol[i,0:n:1].reshape(1,n),0]
    #    plt.plot(np.linspace(0,1,12), out.reshape(12,1))
    #    plt.pause(0.01)
    #    plt.cla()

    #while True:
    #    plt.pause(0.01)



if __name__ == '__main__':
    main()

