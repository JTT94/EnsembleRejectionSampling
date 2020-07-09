
import numpy as np

def compute_squared_distances(x, y):
    x2 = np.expand_dims(np.sum(x**2, axis=1),-1)
    y2 = np.expand_dims(np.sum(y**2, axis=1),-1)
    dists = -2 * np.dot(x, y.T) + y2.T + x2
    return dists

def bound_initw(winit, wbar, icand):
    winit[icand[0]] = wbar[0]
    return winit
    
def bound_w(w, wbar, icand, t):
    w[icand[t],:] = wbar[t]
    w[:, icand[t-1]] = wbar[t]
    return w

