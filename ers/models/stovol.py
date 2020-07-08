import numpy as np
from ..base import ERS
from ..utils import compute_squared_distances, bound_initw, bound_w

class StoVol(ERS):
    def __init__(self, dimension, alpha, beta, sv):
        super().__init__(dimension)
        self.beta = beta
        self.alpha = alpha
        self.sv = sv
        self.ss = sv/np.sqrt(1-alpha**2)
        
    
    def random_grid(self, N, T, y):
        x=np.zeros((T,N,self.dimension))
        for t in np.arange(T):
            x[t]=np.log(y[t]**2)-np.log(self.beta**2)-np.log(np.random.randn(N,self.dimension)**2)
        return x
    
    
    def _step(self, x, predlike, filter_state, llike, t):
        x1=x[t]
        x2=self.alpha*x[t-1]  
        dists = compute_squared_distances(x1,x2)

        logw = dists / (2.*self.sv**2)
        logwmin=np.min(logw)

        w = np.exp(-logwmin)*np.exp(-logw+logwmin)/self.sv

        filter_state[t]=w.dot(filter_state[t-1])
        predlike[t]=np.sum(filter_state[t])
        filter_state[t]=filter_state[t]/predlike[t]
        llike=llike+np.log(predlike[t])
        return llike, predlike, filter_state
    
    
    def _bound_step(self, x, predlike, filter_state, llike, wbar, icand, t):
        x1=x[t]
        x2=self.alpha*x[t-1]  
        dists = compute_squared_distances(x1,x2)

        logw = dists / (2.*self.sv**2)
        logwmin=np.min(logw)

        w = np.exp(-logwmin)*np.exp(-logw+logwmin)/self.sv
        w = bound_w(w, wbar, icand, t)

        filter_state[t]=w.dot(filter_state[t-1])
        predlike[t]=np.sum(filter_state[t])
        filter_state[t]=filter_state[t]/predlike[t]
        llike=llike+np.log(predlike[t])
        return llike, predlike, filter_state
    

    def forwardHMM(self, x):
        T = x.shape[0]
        N = x.shape[1]
        
        predlike=np.zeros(T)  
        filter_state=np.zeros((T,N)) 

        #init
        winit = np.exp(-x[0,:,0]**2./(2*self.ss**2))/self.ss
        filter_state[0] = winit
        predlike[0] = np.sum(filter_state[0])
        llike = np.log(predlike[0])
        filter_state[0] = filter_state[0]/predlike[0]

        llike=0. 
        for t in range(1,T):
            llike, predlike, filter_state = self._step(x, predlike, filter_state, llike, t)
        return llike, predlike, filter_state
    
    
    def forwardHMMbound(self, x, icand):
        
        T = x.shape[0]
        N = x.shape[1]
        
        wbar = self.wbar(T)
            
        predlike=np.zeros(T)  
        filter_state=np.zeros((T,N)) 

        #init
        winit = np.exp(-x[0,:,0]**2./(2*self.ss**2))/self.ss
        winit = bound_initw(winit, wbar, icand)

        filter_state[0] = winit
        predlike[0] = np.sum(filter_state[0])
        llike = np.log(predlike[0])
        filter_state[0] = filter_state[0]/predlike[0]

        llike=0. 

        for t in range(1,T):
            llike, predlike, filter_state = self._bound_step(x, predlike, filter_state, llike, wbar, icand, t)
        return llike, predlike, filter_state
    
    def backwardsampling(self, x, filter_state):
        T = x.shape[0]
        N = x.shape[1]
        
        icand = np.zeros(T, int)
        backfilter = np.zeros(N)
        transition = np.zeros(N)

        icand[T-1] = np.random.choice(N,size=1, replace=True, p=filter_state[-1])

        for t in np.arange(0, T-1)[::-1]:
            transition = np.exp(-(x[t+1,icand[t+1],:]-self.alpha*x[t])**2/(2*self.sv**2))/self.sv
            backfilter = filter_state[t]*transition.squeeze()
            backfilter = backfilter/np.sum(backfilter) 
            icand[t]   = np.random.choice(N, size=1, replace= True, p=backfilter)
        return icand
        
    def wbar(self, T):
        wbar        = np.zeros(T)
        wbar[0]   = 1./self.ss
        wbar[1:T] = 1./self.sv
        return wbar