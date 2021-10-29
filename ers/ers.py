#  MIT License
#
#  Copyright (c) 2021 Adrien Corenflos
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax.experimental.host_callback import id_print
from jax.scipy.special import logsumexp

from ers.base import LogWeightModel, BoundedLogWeightModel


def ers(key,
        log_weight_model: BoundedLogWeightModel,
        N, M,
        vmapped: bool = True):
    """
    Forward filtering backward sampling.

    Parameters
    ----------
    key: PRNGKey
        the random JAX key used as an initialisation of the algorithm.
    log_weight_model: LogWeightModel
        Model to sample from.
    N: int
        Number of particles used for the filtering pass
    M: int
        Number of trajectories to sample
    vmapped: bool, optional
        Vmap the ERS sampler, otherwise loop. In practice most iterations will have roughly the same average
        acceptance probability, so that vmapping kinda makes sense.
    Returns
    -------
    ...
    """

    sampler = log_weight_model.grid_sampler
    log_w0 = log_weight_model.log_w0
    log_w0_upper = log_weight_model.log_w0_upper
    log_wt = log_weight_model.log_wt
    log_wt_partial_upper = log_weight_model.log_wt_partial_upper
    log_wt_upper = log_weight_model.log_wt_upper
    log_Mt = log_weight_model.log_Mt

    keys = jax.random.split(key, M)
    key_ers = lambda k: _ers(k, sampler, log_w0, log_w0_upper, log_wt, log_wt_partial_upper,
                             log_wt_upper, log_Mt, N)
    if vmapped:
        result = jax.vmap(key_ers)(keys)

    else:

        _, result = jax.lax.scan(lambda _, k: (None, key_ers(k)), None, keys)

    return result


@partial(jax.jit, static_argnums=(1, 2, 4, 5, 7, 8))
def _ers(key, sampler, log_w0_fn, log_w0_upper, log_wt_fn, log_wt_partial_upper, log_wt_upper, log_Mt, N):
    init_key, loop_key = jax.random.split(key, 2)
    init_sample = sampler(init_key, 1)[:, 0]  # this will only be used to initialise the algorithm

    def cond(carry):
        return carry[0]

    def body(carry):
        _rejected, i, avg_pacc, op_key, sample = carry
        sample_key, rejection_key, next_key = jax.random.split(op_key, 3)
        ell, upper_bound, proposed_sample = _ers_one(sample_key, sampler, log_w0_fn, log_w0_upper, log_wt_fn,
                                                     log_wt_partial_upper, log_wt_upper, log_Mt, N)

        pacc = jnp.exp(ell - upper_bound)
        avg_pacc = (avg_pacc * i + pacc) / (i + 1)
        u = jax.random.uniform(rejection_key)
        rejected = u > pacc
        # id_print(pacc, what="pacc")
        # id_print(rejected, what="rejected")
        return rejected, i + 1, avg_pacc, next_key, proposed_sample

    _, n_trials, average_acceptance_proba, _, sample = jax.lax.while_loop(cond, body,
                                                                          (True, 0, 0., loop_key, init_sample))
    return average_acceptance_proba, sample, n_trials


def _ers_one(key, sampler, log_w0_fn, log_w0_upper, log_wt_fn, log_wt_partial_upper,
             log_wt_upper, log_Mt, N):
    grid_key, backward_sampler_key = jax.random.split(key, 2)
    grid_samples = sampler(grid_key, N)
    filtering_log_weights, ell = _forward_filtering(grid_samples, log_w0_fn, log_wt_fn, False)
    proposed_sample, sampler_indices = _backward_sampling(backward_sampler_key, grid_samples, filtering_log_weights,
                                                          log_Mt)
    _, upper_bound = _forward_filtering(grid_samples, log_w0_fn, log_wt_fn, True, log_w0_upper, log_wt_partial_upper,
                                        log_wt_upper, sampler_indices)

    return ell, upper_bound, proposed_sample


def _forward_filtering(grid_samples,
                       log_w0_fn, log_wt_fn, bounding: bool,
                       log_w0_upper: float = None, log_wt_partial_upper: Callable = None,
                       log_wt_upper: float = None, traj_indices: jnp.ndarray = None
                       ):
    # initialisation
    x_0 = grid_samples[0]
    log_w0 = jax.vmap(log_w0_fn)(x_0)
    ell_init = logsumexp(log_w0)
    if bounding:
        ind_0 = traj_indices[0]
        log_w0 = log_w0.at[ind_0].set(log_w0_upper)
        traj_indices = traj_indices[1:]
    else:
        traj_indices = ind_0 = None

    # given two arrays x_t_1 (N,) and x_t (M,), log_wt_vmapped(x_t_1, x_t) will return the (N, M) array of pairwise
    # log weights, therefore, it has to be PRE multiplied by the running filtering weights to compute the new filtering
    # weights.
    log_wt_vmapped = jax.vmap(jax.vmap(log_wt_fn,
                                       [0, None]),
                              [None, 0])

    def log_weight_fn(x_t_1, ind_t_1, x_t, ind_t):
        log_wt_mat = log_wt_vmapped(x_t_1, x_t)
        if bounding:
            log_wt_partial_upper_vmapped = jax.vmap(log_wt_partial_upper, [0, None])
            log_wt_mat = log_wt_mat.at[ind_t_1].set(log_wt_partial_upper_vmapped(x_t, False))
            log_wt_mat = log_wt_mat.at[:, ind_t].set(log_wt_partial_upper_vmapped(x_t_1, True))
            log_wt_mat = log_wt_mat.at[ind_t_1, ind_t].set(log_wt_upper)
        return log_wt_mat

    def body(carry, inputs):
        ind_t, x_t = inputs
        ind_t_1, x_t_1, log_w_t_1, ell_t_1 = carry
        log_w_t_mat = log_weight_fn(x_t_1, ind_t_1, x_t, ind_t)  # update weight matrix
        log_w_t = _log_matvec(log_w_t_mat, log_w_t_1, transpose_a=False)
        ell_inc = logsumexp(log_w_t)
        log_w_t = log_w_t - ell_inc
        return (ind_t, x_t, log_w_t, ell_t_1 + ell_inc), log_w_t

    (*_, ell), filtering_log_weights = jax.lax.scan(body, (ind_0, x_0, log_w0, ell_init),
                                                    (traj_indices, grid_samples[1:]))

    filtering_log_weights = jnp.concatenate([log_w0[None, :], filtering_log_weights])
    return filtering_log_weights, ell


def _backward_sampling(key,
                       grid_samples,
                       filtering_log_weights,
                       log_Mt):
    T, N, *_ = grid_samples.shape
    keys = jax.random.split(key, T)
    log_w_T = filtering_log_weights[-1]

    log_Mt_vmapped = jax.vmap(log_Mt, [0, None])

    ind_T = jax.random.choice(keys[0], N, (), p=jnp.exp(log_w_T))
    x_T = grid_samples[-1, ind_T]

    def backward_sampling_body(x_t, inputs):
        x_t_1, log_w_t_1, op_key = inputs
        log_w_t = log_w_t_1 + log_Mt_vmapped(x_t_1, x_t)
        log_w_t = log_w_t - logsumexp(log_w_t)
        ind_t_1 = jax.random.choice(op_key, N, (), p=jnp.exp(log_w_t))
        x_t_1 = x_t_1[ind_t_1]
        return x_t_1, (x_t_1, ind_t_1)

    _, (traj, indices) = jax.lax.scan(backward_sampling_body,
                                      x_T,
                                      (grid_samples[:-1],
                                       filtering_log_weights[:-1],
                                       keys[:-1]
                                       ),
                                      reverse=True
                                      )

    traj = jnp.concatenate([traj, x_T[None, :]])
    indices = jnp.concatenate([indices, jnp.atleast_1d(ind_T)])

    return traj, indices


@partial(jax.jit, static_argnums=(2,))
def _log_matvec(log_A, log_b, transpose_a=False):
    """
    Examples
    --------
    >>> import numpy as np
    >>> log_A = np.random.randn(50, 3)
    >>> log_b = np.random.randn(3)
    >>> np.max(np.abs(np.exp(log_matvec(log_A, log_b)) - np.exp(log_A) @ np.exp(log_b))) < 1e-5
    True
    """
    if transpose_a:
        log_A = log_A.T
    Amax = jnp.max(log_A)
    bmax = jnp.max(log_b)
    A = jnp.exp(log_A - Amax)
    b = jnp.exp(log_b - bmax)
    return Amax + bmax + jnp.log(A @ b)
