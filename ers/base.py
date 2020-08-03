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
        paccs = []
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
                paccs.append(pacc)

                u = np.random.random()
                if u < pacc:
                    accepted = True
        
        return cand_x, n_trial, paccs

    def sample_n(self, n_samples, n_particles, T, y, verbose=False):

        
        accepted_x_indices = []
        candidates_x = []
        paccss = []
        n_trials = 0

        for i in tqdm(range(n_samples)):
            if verbose:
                print('\n Iteration: {0}'.format(i))

            cand_x, n_trial, paccs = self.sample_one(n_particles, T, y)
            candidates_x.append(cand_x)
            paccss.append(paccs)
            n_trials += n_trial
            accepted_x_indices.append(n_trials-1)
            
            

        candidates_x = np.concatenate(candidates_x)
        paccs = np.concatenate(paccss)
        accepted_x_indices = np.array(accepted_x_indices)
        return accepted_x_indices, candidates_x, n_trials, paccs



    
    
    
