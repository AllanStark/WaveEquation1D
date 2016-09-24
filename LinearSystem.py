#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint 

class LinearSystem(object):
    def create_TridiagMatrix(self,n,e=1):
        A= e*(np.eye(n,k=1)-2*np.eye(n)+np.eye(n,k=-1))
        return A 

    def solve(self,in_f,x0,plot_flag=0):
        tspan = np.arange(0, 10.0, 0.01)
        invM= self.invM
        K= self.K
        D= self.D
        L= self.L
        self.x_sol= odeint(self.solvr, x0, tspan,args=(in_f,invM,K,D,L))

        if plot_flag:
            if not self.C:
                C= np.zeros([self.n*2,1])
            self.x_out= np.dot(self.x_sol,C)
            self.t_out= tspan
            plt.plot(sel.t_out,self.x_out)


    def solvr(self,x, t, in_f,invM,K,D,L):
        nq= len(x)/2
        pos,vel= x[0:nq:1],x[nq:]

        vel_dot= np.dot(invM,(np.dot(K,pos.reshape(nq,1)) +np.dot(D,vel.reshape(nq,1))-np.dot(L,in_f(t))))

        out= np.row_stack((vel.reshape(nq,1),vel_dot.reshape(nq,1)))
        return np.squeeze(np.asarray(out))
