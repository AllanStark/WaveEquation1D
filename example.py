#!/usr/bin/python
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from WaveEquation1D import WaveEquation1D as Wave1D

def main():
    n=22-2
    L= np.zeros([n,1])

    WaveObj= Wave1D(n+2,1,1,1,0.01,L)

    in_f= lambda t: 0
    tmp= np.linspace(0,1,n+2)
    x0= np.zeros([n*2])
    x0[0:n:1]= np.sin(np.pi*tmp[1:n+1:1])
    WaveObj.solve(in_f,x0,plot_flag=1)


if __name__ == '__main__':
    main()

