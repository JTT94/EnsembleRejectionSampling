import math

import jax.numpy as jnp
import jax.random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ers import ers
from ers.base import BoundedLogWeightModel


# Model def
class HardObstacle(BoundedLogWeightModel):
    def __init__(self, dimension, sv, T):
        self.sv = sv
        self.dimension = dimension
        self.T = T

    def grid_sampler(self, key, N):
        T, dimension = self.T, self.dimension
        return jax.random.uniform(key, (T, N, dimension), minval=-1., maxval=1.)

    @property
    def log_w0_upper(self):
        return 0.

    def log_wt_partial_upper(self, x_t, _left):
        return -jnp.log(self.sv)

    @property
    def log_wt_upper(self):
        return -jnp.log(self.sv) + self.dimension * jnp.log(2.)

    def log_w0(self, x0):
        return jnp.log(jnp.sum(x0 ** 2) <= 1)

    def log_wt(self, x_t_1, x_t):
        Mt_part = self.log_Mt(x_t_1, x_t)
        Gt_part = jnp.log(jnp.sum(x_t ** 2) <= 1)
        return Gt_part + Mt_part + self.dimension * jnp.log(2.)

    def log_Mt(self, x_t_1, x_t):
        scaled_diff = (x_t - x_t_1) / self.sv
        dists = jnp.sum(scaled_diff ** 2)
        return -0.5 * dists - self.dimension * jnp.log(self.sv) - 0.5 * self.dimension * math.log(2 * math.pi)


# Dirty config

key = jax.random.PRNGKey(42)

n_samples = 10_000
N = 250
T = 50
sv = 0.2
d = 2

log_weight_model = HardObstacle(dimension=d, sv=sv, T=T)

acceptance_probas, samples, n_trials = ers.ers(key, log_weight_model, N, n_samples, vmapped=False)

plt.hist(acceptance_probas, bins=25)
plt.show()
plt.hist(n_trials, density=True)
plt.show()

for t in np.arange(0, T, 5):
    fig = plt.figure()
    sns.jointplot(x=samples[:, t, 0], y=samples[:, t, 1], kind="hex")
    plt.show()
