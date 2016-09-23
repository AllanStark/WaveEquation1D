#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

class LinearSystem:
    def create_TridiagMatrix(self,n,e=1):
        A= e*(np.eye(n,k=1)-2*np.eye(n)+np.eye(n,k=-1))
        return A 

    def solve(self,in_f,x0,plot_flag=0):
        tspan = np.arange(0, 10.0, 0.01)
        self.x_sol = integrate.odeint(solvr, x0, tspan,args=(in_f,invM,K,D,L))

        if plot_flag:
            if not self.C:
                C= np.zeros([self.n*2,1])
            self.x_out= np.dot(self.x_sol,C)
            self.t_out= tspan
            plt.plot(sel.t_out,self.x_out)


    def solvr(x, t, K,D,M,L, in_f):
        nq= len(x)/2
        pos,vel= y[0:nq:1],y[nq:]

        vel_dot= invM*(K*pos.reshape(nq,1) +D*vel.reshape(nq,1)- L*in_f(t))

        out= np.row_stack((vel.reshape(nq,1),vel_dot.reshape(nq,1)))
        return np.squeeze(np.asarray(out))
