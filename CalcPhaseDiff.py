#!/usr/bin/python
import numpy as np
from scipy import integrate
import scipy
import matplotlib.pyplot as plt

def Calc_Phase(f,tspan,delta,a,b):
    T= 1/f
    if tspan[-1]>T:
        print len(a)
        a= a[tspan<T]
        b= b[tspan<T]
        print len(a)
    af = scipy.fft(a)
    bf = scipy.fft(b)
    c = scipy.ifft(af * scipy.conj(bf))

    #plt.plot(c)
    #plt.show()

    time_shift = np.argmax(abs(c))
    return time_shift*2*np.pi*f*delta   

def main():
    f= 1.0
    delta=1/f*1e-4
    tspan= np.arange(0,23,delta)
    phase= -np.pi/6

    soll_sig= np.sin(2*np.pi*f*tspan)
    ist_sig= np.sin(2*np.pi*f*tspan+phase)

    #plt.plot(tspan,soll_sig,tspan,ist_sig)
    #plt.show()

    #T= 1/f
    #t1= tspan[np.argmax(abs(soll_sig[tspan<=(T/2.0)]))]
    #t2= tspan[np.argmax(abs(ist_sig[tspan<=(T/2.0)]))]
    #phase_ist= 2*np.pi*f*(t1-t2)
    phase_fft= Calc_Phase(f,tspan,delta,soll_sig,ist_sig)
    #print 'phase ist=',t1-t2,'  phase_soll=',phase,'phase exact',phase_fft
    print '  phase_soll=',phase,'phase exact',phase_fft

if __name__ == '__main__':
    main()
