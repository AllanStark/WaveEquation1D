#!/usr/bin/python
import numpy as np
from scipy import integrate
import scipy
import matplotlib.pyplot as plt

def ImpulseForce(off,scale,sig,x):
    return scale*np.exp(-((x-off)**2)/(2*sig**2))

# quick and dirty way
def cutSig(sig):
    return sig[np.int(len(sig)*(1-0.2)):]

# quick and dirty way
def Calc_SteadyStateAmp(ResponseSig):
    #print np.max(np.abs(tmp[np.int(len(tmp)*(1-0.2)):]))
    return np.max(np.abs(cutSig(ResponseSig)))

def Calc_Complex_Mob(F0,ResponseSig,f,tspan):
    out=Calc_R_Mob(F0,ResponseSig)+ 1j*(Calc_Im_Mob(F0,ResponseSig,f,tspan))
    return out

def Calc_Mag_Mob(F0,ResponseSig):
    out= Calc_SteadyStateAmp(ResponseSig)/F0
    print out
    return out

def Calc_Phase(f,tspan,ResSig):
    T= 1/f
    delta= np.mean(np.diff(tspan))
    if tspan[-1]>T:
        ResSig= ResSig[tspan<(T/2.0)]
    ResSig_f= scipy.fft(ResSig)
    b= np.cos(2*np.pi*f*tspan[tspan<(T/2.0)])
    #bf= scipy.fft(b)
    #c= scipy.ifft(ResSig_f * scipy.conj(bf))

    #time_shift = np.argmax(abs(c))

    time_shift= tspan[np.argmax(abs(ResSig))]
    #print time_shift
    return time_shift*2*np.pi*f-np.pi/2

def Calc_Im_Mob(F0,ResponseSig,f,tspan):
    ResponseSig_Mod= cutSig(ResponseSig)/max(cutSig(ResponseSig)) 
    tspan_Mod= cutSig(tspan)-(cutSig(tspan)[0])
    out=Calc_Mag_Mob(F0,ResponseSig)*np.sin(Calc_Phase(f,tspan_Mod,ResponseSig_Mod))
    #print out
    return out

def Calc_R_Mob(F0,ResponseSig,f,tspan):
    ResponseSig_Mod= cutSig(ResponseSig)/max(cutSig(ResponseSig)) 
    tspan_Mod= cutSig(tspan)-(cutSig(tspan)[0])
    theta= Calc_Phase(f,tspan_Mod,ResponseSig_Mod)
    print theta
    out= Calc_Mag_Mob(F0,ResponseSig)*np.cos(theta)
    #print out
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

def Mobility_analytical(k,c,m,w):
    Y= ((pow(w,2)*c)+1j*(k*w-pow(w,3)*m))/(pow((k-pow(w,2)*m),2)+pow((w*c),2))
    return Y

def eta_analytical(k,c,m,w):
    Y= Mobility_analytical(k,c,m,w) 
    eta_an= Y.real/(m*w*pow(abs(Y),2))
    return eta_an

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
