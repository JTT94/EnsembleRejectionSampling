import numpy as np
from ..base import ERS
from ..utils import compute_squared_distances


class HardObstacle(ERS):

    def __init__(self, dimension, sv):
        super().__init__(dimension)
        self.sv = sv
        
    def random_grid(self, N, T, y):
        x=np.random.random((T,N,self.dimension))
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
            x2 = x[t]
            
            transition = np.exp(-np.sum((x1- x2)**2, axis=-1)/(2*self.sv**2))/self.sv
            backfilter = filter_state[t]*transition.squeeze()
            backfilter = backfilter/np.sum(backfilter) 
            icand[t]   = np.random.choice(N, size=1, replace= True, p=backfilter)
        return icand
        
    def w_func(self, x, t):
        x1 = x[t]
        x2= x[t-1]
        
        dists = compute_squared_distances(x1,x2)

        logw = dists / (2.*self.sv**2)
        logwmin=np.min(logw)

        w = np.exp(-logwmin)*np.exp(-logw+logwmin)/self.sv

        return w

    def _w_init_func(self, x):
        N = x.shape[1]
        return np.ones(N)

    def _bound_initw(self, winit, icand):
        winit[icand[0]] = 1.
        return winit
    
    def _w_bound_func(self, w, icand, t):
        w[icand[t],:] = 1./self.sv
        w[:, icand[t-1]] = 1./self.sv
        return w


