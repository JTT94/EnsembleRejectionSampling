import numpy as np
from ..base import ERS
from ..utils import compute_squared_distances

class NonLinearAR(ERS):

    def __init__(self, dimension, alpha, sv, sw):
        super().__init__(dimension)
        
        if np.isscalar(alpha):
            alpha = np.eye(dimension) * alpha

        self.alpha = alpha
        self.sv = sv
        self.sw = sw
        
        
    def random_grid(self, N, T, y):
        x=np.zeros((T,N,self.dimension))
        for t in range(T):
            x[t] = y[t] + self.sw * np.random.randn(N,self.dimension)
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
            # x2 = self.alpha * np.tanh(x[t])
            x2 = np.moveaxis(np.einsum('ij,kj', self.alpha, np.tanh(x[t])),0,1)
            
            transition = np.exp(-np.sum((x1- x2)**2, axis=-1)/(2*self.sv**2))/self.sv
            backfilter = filter_state[t]*transition.squeeze()
            backfilter = backfilter/np.sum(backfilter) 
            icand[t]   = np.random.choice(N, size=1, replace= True, p=backfilter)
        return icand
        
    def w_func(self, x, t):
        x1 = x[t]
        #x2 = self.alpha * np.tanh(x[t])
        x2=np.moveaxis(np.einsum('ij,kj', self.alpha, np.tanh(x[t-1])),0,1)
        
        dists = compute_squared_distances(x1,x2)

        logw = dists / (2.*self.sv**2)
        logwmin=np.min(logw)

        w = np.exp(-logwmin)*np.exp(-logw+logwmin)/self.sv

        return w

    def _w_init_func(self, x):
        return np.exp(-np.sum(x[0]**2, axis=-1)/2)

    def _bound_initw(self, winit, icand):
        winit[icand[0]] = 1.
        return winit
    
    def _w_bound_func(self, w, icand, t):
        w[icand[t],:] = 1./self.sv
        w[:, icand[t-1]] = 1./self.sv
        return w

    def generate_x(self, T):
        # true hidden state
        d = self.dimension
        xtrue = np.zeros((T,d))
        xtrue[0,:] = np.random.randn(d)
        for t in range(1,T):
            x2 = np.einsum('ij,j', self.alpha, np.tanh(xtrue[t-1]))
            xtrue[t] = x2 + self.sv * np.random.randn(d)
        return xtrue
    
    def generate_y(self,x):
        T = x.shape[0]
        d = self.dimension
        y = np.zeros((T,d))
        y[0] = x[0]+ self.sw*np.random.randn(1)

        for t in range(2,T):
            y[t] = x[t] + self.sw * np.random.randn(d)
        return y