import numpy as np
from ..base import ERS
from ..utils import compute_squared_distances

class StoVol(ERS):

    def __init__(self, dimension, alpha, beta, sv):
        super().__init__(dimension)
        
        self.alpha = alpha
        self.beta = beta
        self.sv = sv
        self.ss = sv/np.sqrt(1-alpha**2)
        
    def random_grid(self, N, T, y):
        x=np.zeros((T,N,self.dimension))
        for t in np.arange(T):
            x[t]=np.log(y[t]**2)-np.log(self.beta**2)-np.log(np.random.randn(N,self.dimension)**2)
        return x
    
    def backwardsampling(self, x, filter_state):
        T = x.shape[0]
        N = x.shape[1]
        
        icand = np.zeros(T, int)
        backfilter = np.zeros(N)
        transition = np.zeros(N)

        icand[T-1] = np.random.choice(N,size=1, replace=True, p=filter_state[-1])

        for t in np.arange(0, T-1)[::-1]:
            x1 = x[t+1,icand[t+1],:]
            x2=self.alpha*x[t] 
            #x2 = np.moveaxis(np.einsum('ij,kj', self.alpha, x[t]),0,1)
            
            transition = np.exp(-np.sum((x1- x2)**2, axis=-1)/(2*self.sv**2))/self.sv
            backfilter = filter_state[t]*transition.squeeze()
            backfilter = backfilter/np.sum(backfilter) 
            icand[t]   = np.random.choice(N, size=1, replace= True, p=backfilter)
        return icand
        
    def w_func(self, x, t):
        x1=x[t]
        #x2 = np.moveaxis(np.einsum('ij,kj', self.alpha, x[t-1]),0,1)
        x2=self.alpha*x[t-1]  
        dists = compute_squared_distances(x1,x2)

        logw = dists / (2.*self.sv**2)
        logwmin=np.min(logw)

        w = np.exp(-logwmin)*np.exp(-logw+logwmin)/self.sv

        return w

    def _w_init_func(self, x):
        return np.exp(-np.sum(x[0,:,:]**2, axis=-1)/(2*self.ss**2))/self.ss

    def _bound_initw(self, winit, icand):
        winit[icand[0]] = 1./self.ss
        return winit
    
    def _w_bound_func(self, w, icand, t):
        w[icand[t],:] = 1./self.sv
        w[:, icand[t-1]] = 1./self.sv
        return w
