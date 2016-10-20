#!/usr/bin/python
import numpy as np
from scipy import integrate
import scipy
import matplotlib.pyplot as plt

def main():
    k= 2000
    d= 20
    m= 10 

    #A= np.matrix([[0,1],[-k/m,-c/m]])
    #B= np.zeros([2,1])

    K= k*np.matrix([[-1]])
    D= d*np.matrix([[-1]])
    M= m*np.eye(1)

    L=np.ones([1])
    C= np.zeros([2,1])
    C[0]=1 
    x0= np.array([0,0])

    #K= k*np.matrix([[-2,1],[1,-2]])
    #D= d*np.matrix([[-2,1],[1,-2]])
    #M= m*np.eye(2)

    #L=np.zeros([2,1])
    #C= np.zeros([4,1])
    #C[0]=1 
    #x0= np.array([1,0,0,0])

    #in_f= lambda t: ImpulseForce(1,100,0.05,t) 
    F0=100
    driveFreq= 0.5*np.sqrt(k/m)/(2*np.pi)
    in_f= lambda t: F0*np.sin(2*np.pi*driveFreq*t) 

    tspan = np.arange(0, 20 , (1.0/driveFreq)*1e-4)
    asol = integrate.odeint(solvr, x0, tspan,args=(K,D,M,L,in_f))
    outSig= np.dot(asol,C)

    #plt.subplot(211)
    #plt.plot(tspan,in_f(tspan))
    #plt.subplot(212)
    #plt.plot(tspan,outSig,'r');plt.show()

    plt.plot(tspan,in_f(tspan)/F0,tspan,outSig/max(outSig),'r');plt.show()

    #print 1.0/d
    #Y_ana= Mobility_analytical(k,d,m,driveFreq*2*np.pi)
    #R_Mob_act= Calc_R_Mob(F0,outSig,driveFreq,tspan)
    #print 'hhh',np.angle(Y_ana)
    #print 'R Mob Ana=',phase(Y_ana),',R Mob_act=',R_Mob_act
    #print 'eta_ana=',eta_analytical(k,d,m,driveFreq*2*np.pi)

    #plt.plot(tspan,ImpulseForce(1,100,0.05,tspan))
    #plt.show()

if __name__ == '__main__':
    main()
