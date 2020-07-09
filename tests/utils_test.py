import numpy as np
import os, sys

sys.path.append("..")
from ers.utils import compute_squared_distances

def test_compute_distances_multivariate():
    T = 10
    N = 1000
    d = 5
    x = np.random.random((T, N, d))

    x1 = x[1]
    x2 = x[2]

    out = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            out[i,j] = np.sum((x1[i]-x2[j])**2)

    dists = compute_squared_distances(x1,x2)
    diffs = np.abs(out - dists)
    tol = 0.001
    assert np.sum(diffs) < tol


def test_compute_distances_univariate():
    T = 10
    N = 1000
    d = 1
    x = np.random.random((T, N, d))

    x1 = x[1]
    x2 = x[2]

    out = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            out[i,j] = np.sum((x1[i]-x2[j])**2)

    dists = compute_squared_distances(x1,x2)
    diffs = np.abs(out - dists)
    tol = 0.001
    assert np.sum(diffs) < tol


