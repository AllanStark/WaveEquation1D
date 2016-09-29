#!/usr/bin/python
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from WaveEquation1D import WaveEquation1D as Wave1D
from scipy.signal import chirp

def angreifPunkte(x,mue,sigma):
    tmp= -(x-mue)**2/(2*sigma**2)
    amp= (1/(sigma*np.sqrt(2*np.pi)*len(x)))
    return amp*np.exp(tmp)

def main():
    n=102-2
    t_end= 10
    tmp= np.linspace(0,1,n+2)

    #L= np.zeros([n,1])
    L= angreifPunkte(tmp[1:-1],0.2,0.05)

    amp= 100
    #in_force= lambda t: (t>0.1)*amp*np.sin(2*np.pi*0.5*t)
    in_force= lambda t: amp*chirp(t,f0=0.5,f1=10,t1=t_end,method='linear')
    in_time= np.linspace(0, t_end, 5e3)

    #plt.plot(in_time,in_force(in_time))
    #plt.show()

    x0= np.zeros([n*2])
    #x0[0:n:1]= np.sin(np.pi*tmp[1:n+1:1])

    WaveObj= Wave1D(n+2,1,1,1,0.01,L)
    WaveObj.solve(in_force,in_time,x0,plot_flag=1)

if __name__ == '__main__':
    main()

