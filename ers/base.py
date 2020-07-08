import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm

class ERS(ABC):
    def __init__(self, dimension):
        self.dimension = dimension
        
    @abstractmethod
    def random_grid(self, n_particles, T, y):
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def forwardHMM(self, x):
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def backwardsampling(self, x, filter_state):
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def forwardHMMbound(self, x, icand):
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def wbar(self, T):
        raise NotImplementedError("Please Implement this method")

    def __call__(self, n_samples, n_particles, T, y):

        
        acccepted_x = []
        cand_x = []
        n_trial = 0

        for n_acc_samples in tqdm(range(n_samples)):
            accepted = False

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
                    acccepted_x.append(xacc)
                    accepted = True
                if n_trial % 50 == 0:
                    print("Trial {0}".format(n_trial))
        
        acccepted_x = np.array(acccepted_x)
        cand_x = np.array(cand_x)
        return acccepted_x, cand_x, n_trial



    
    
    
