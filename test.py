#!/usr/bin/python
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

def ImpulseForce(off,scale,sig,x):
    return scale*np.exp(-((x-off)**2)/(2*sig**2))

# quick and dirty way
def Calc_SteadyStateVelAmp(ResponseSig):
    tmp= ResponseSig
    return np.max(np.abs(tmp[np.int(len(tmp)*(1-0.2)):]))

def Calc_R_Mob(F0,ResponseSig):
    out= Calc_SteadyStateVelAmp(ResponseSig)/F0
    return out

def Calc_EnergyDissipation(F0,ResponseSig):
    R_Mob= Calc_R_Mob(F0,ResponseSig)
    return (F0**2*R_Mob)/2

def solvr(y, t, K,D,M,L, in_f):
    nq= len(y)/2
    pos,vel= y[0:nq:1],y[nq:]

    vel_dot= np.linalg.inv(M)*(K*pos.reshape(nq,1) +D*vel.reshape(nq,1)+ L*in_f(t))

    out= np.row_stack((vel.reshape(nq,1),vel_dot.reshape(nq,1)))
    return np.squeeze(np.asarray(out))

def main():
    k= 100
    d= 10
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
    driveFreq=10
    in_f= lambda t: F0*np.sin(2*np.pi*driveFreq*t) 

    tspan = np.arange(0, 25.0, 0.0001)
    asol = integrate.odeint(solvr, x0, tspan,args=(K,D,M,L,in_f))
    outSig= np.dot(asol,C)

    EnergyDissipation= Calc_EnergyDissipation(F0,outSig)

    plt.plot(tspan,outSig)
    plt.title(EnergyDissipation)
    plt.show()

    #plt.plot(tspan,ImpulseForce(1,100,0.05,tspan))
    #plt.show()

if __name__ == '__main__':
    main()
