import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm

class ERS(ABC):
    def __init__(self, dimension):
        self.dimension = dimension
        


    def _step(self, x, predlike, filter_state, llike, t, icand =None, bound=False):


        w = self.w_func(x,t)

        if bound:
            w = self._w_bound_func(w, icand, t)

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
        winit = self._w_init_func(x)

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
        
            
        predlike=np.zeros(T)  
        filter_state=np.zeros((T,N)) 

        # bound init w
        winit = self._w_init_func(x)
        winit = self._bound_initw(winit, icand)

        filter_state[0] = winit
        predlike[0] = np.sum(filter_state[0])
        llike = np.log(predlike[0])
        filter_state[0] = filter_state[0]/predlike[0]

        llike=0. 

        for t in range(1,T):
            llike, predlike, filter_state = self._step(x, predlike, filter_state, llike, t, icand, bound=True)
        return llike, predlike, filter_state

    @abstractmethod
    def random_grid(self, n_particles, T, y):
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def backwardsampling(self, x, filter_state):
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def w_func(self, x, t):
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def _w_init_func(self, x):
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def _bound_initw(self, winit, icand):
        raise NotImplementedError("Please Implement this method")
    
    @abstractmethod
    def _w_bound_func(self, w, icand, t):
        raise NotImplementedError("Please Implement this method")

    def sample_one(self, n_particles, T, y):
        cand_x = []
        accepted = False
        n_trial = 0
        while not accepted:   
                n_trial += 1
                xcand=np.zeros((T,self.dimension))
                x = self.random_grid(n_particles, T, y)
                logZ, _, filter_state = self.forwardHMM(x)
                icand = self.backwardsampling(x, filter_state)
                logZbar, _, _ = self.forwardHMMbound(x, icand)
                pacc= np.exp(logZ-logZbar)
                
                for t in range(T):
                    xcand[t] = x[t,icand[t],:]
                
                cand_x.append(xcand.copy())
                u = np.random.random()
                if u < pacc:
                    xacc = xcand.copy()
                    accepted = True
        
        return xacc, cand_x, n_trial

    def sample_n(self, n_samples, n_particles, T, y):

        
        acccepted_x = []
        candidates_x = []
        n_trials = 0

        for n_acc_samples in tqdm(range(n_samples)):
            accepted = False
            xacc, cand_x, n_trial = self.sample_one(n_particles, T, y)
            acccepted_x.append(xacc)
            candidates_x.append(cand_x)
            n_trials += n_trial
            print(n_acc_samples)

        acccepted_x = np.array(acccepted_x)
        candidates_x = np.concatenate(candidates_x)
        return acccepted_x, candidates_x, n_trials



    
    
    
