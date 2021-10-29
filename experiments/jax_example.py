import jax.numpy as jnp
import jax.random
import numpy as np

from ers import ers
from ers.base import BoundedLogWeightModel
import matplotlib.pyplot as plt

# Model def
class StoVol(BoundedLogWeightModel):
    def __init__(self, dimension, alpha, beta, sv, data):
        self.alpha = alpha
        self.beta = beta
        self.sv = sv
        self.ss = sv / np.sqrt(1 - alpha ** 2)
        self.dimension = dimension
        self.data = data
        self.T = self.data.shape[0]

    def grid_sampler(self, key, N):
        T, dimension = self.T, self.dimension
        eps = jax.random.normal(key, (T, N, dimension))
        return jnp.log(self.data[:, None, :] ** 2) - jnp.log(self.beta ** 2) - jnp.log(eps ** 2)

    @property
    def log_w0_upper(self):
        return -jnp.log(self.ss)

    def log_wt_partial_upper(self, x_t, _left):
        return -jnp.log(self.sv)

    @property
    def log_wt_upper(self):
        return -jnp.log(self.sv)

    def log_w0(self, x0):
        return -jnp.sum(x0 ** 2, axis=-1) / (2 * self.ss ** 2) - jnp.log(self.ss)

    def log_wt(self, x_t_1, x_t):
        return -jnp.sum((x_t - self.alpha * x_t_1) ** 2, axis=-1) / (2 * self.sv ** 2) - jnp.log(self.sv)

    def log_Mt(self, x_t_1, x_t):
        return self.log_wt(x_t_1, x_t)

# Dirty config


key = jax.random.PRNGKey(42)
np.random.seed(0)

n_samples = 50
N = 100
T = 25
alpha = 0.95
beta = 0.7
sv = 0.3
d = 1

ss = sv / np.sqrt(1 - alpha ** 2)

xtrue = np.zeros((T, d))
xtrue[0] = ss * np.random.randn(d)

y = np.zeros((T, d))
y[0] = beta * np.exp(xtrue[0, 0] / 2) * np.random.randn(d)
for t in np.arange(1, T):
    xtrue[t] = alpha * xtrue[t - 1] + sv * np.random.randn(d)
    y[t] = beta * np.exp(xtrue[t] / 2) * np.random.randn(d)

log_weight_model = StoVol(dimension=d, alpha=alpha, beta=beta, sv=sv, data=y)

acceptance_probas, samples, n_trials = ers.ers(key, log_weight_model, N, n_samples, vmapped=True)

plt.hist(acceptance_probas)
plt.hist(n_trials)
fig, ax = plt.subplots()
ax.plot(samples[..., 0].T, alpha=0.5, linestyle="--")
ax.plot(xtrue[:, 0])
plt.show()

