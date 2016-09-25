#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from LinearSystem import LinearSystem

class WaveEquation1D(LinearSystem):
    def __init__(self,n=10,DomainLength=1,k=1,m=1,alpha=0.01,*L):
        if not L:
            L= np.zeros([n,1])
            L[3:6:1]=[[0.2],[0.6],[0.2]]
        self.n_wbn=n;
        self.n= self.n_wbn-2;
        self.DomainLength=DomainLength;
        self.k=k;
        self.m=m;
        self.alpha=alpha;
        self.L= L

        self.s= np.linspace(0,self.DomainLength,self.n_wbn)
        self.dx= np.mean(np.diff(self.s))

        self.calcMatrices()

    def calcMatrices(self):
        self.M= self.m*np.eye(self.n)
        self.invM= np.linalg.inv(self.M)
        DL= self.DomainLength
        k=self.k
        dx= self.dx
        n= self.n
        ScaleFac= (-np.sqrt(k*DL**2)/dx)**2.0
        self.K= self.create_TridiagMatrix(n,ScaleFac)
        self.D= self.alpha*self.K

    def solve(self,in_f,x0,plot_flag=0):
        super(WaveEquation1D,self).solve(in_f,x0,0)

        if plot_flag:
            self.plot_response()

    def plot_response(self):
        plt.ion()
        n= self.n

        for i in range(self.x_sol.shape[0]):
            plt.axis([0, 1, -1, 1])
            out= np.c_[0,self.x_sol[i,0:n:1].reshape(1,n),0]
            plt.plot(np.linspace(0,1,n+2), out.reshape(n+2,1))
            plt.pause(0.01)
            plt.cla()

        while True:
            plt.pause(0.01)
        
