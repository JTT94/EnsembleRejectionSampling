
import numpy as np

def compute_squared_distances(x, y):
    x2 = np.expand_dims(np.sum(x**2, axis=1),-1)
    y2 = np.expand_dims(np.sum(y**2, axis=1),-1)
    dists = -2 * np.dot(x, y.T) + y2.T + x2
    return dists

