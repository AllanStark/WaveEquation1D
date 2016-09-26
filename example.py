#!/usr/bin/python
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from WaveEquation1D import WaveEquation1D as Wave1D

def angreifPunkte(x,mue,sigma):
    tmp= -(x-mue)**2/(2*sigma**2)
    amp= (1/(sigma*np.sqrt(2*np.pi)*len(x)))
    return amp*np.exp(tmp)

def main():
    n=102-2
    tmp= np.linspace(0,1,n+2)
    #L= np.zeros([n,1])
    L= angreifPunkte(tmp[1:-1],0.2,0.05)

    WaveObj= Wave1D(n+2,1,1,1,0.01,L)

    amp= 100
    in_force= lambda t: (t>0.1)*amp*np.sin(2*np.pi*0.5*t)
    in_time= np.arange(0, 1.0, 0.01)

    x0= np.zeros([n*2])
    #x0[0:n:1]= np.sin(np.pi*tmp[1:n+1:1])
    WaveObj.solve(in_force,in_time,x0,plot_flag=1)


if __name__ == '__main__':
    main()

