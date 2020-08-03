import numpy as np
import time
import os
from pathos.multiprocessing import ProcessingPool as Pool
from .models import NonLinearAR, HardObstacle, StoVol
from .utils import pickle_obj, unpickle_obj

accepted_fn = lambda output_dir, rank: os.path.join(output_dir, 'accepted_x_{0}.txt'.format(rank))
candidates_fn = lambda output_dir, rank: os.path.join(output_dir, 'candidates_x_{0}.txt'.format(rank))
paccs_fn = lambda output_dir, rank: os.path.join(output_dir, 'paccs_{0}.txt'.format(rank))

NONLINEARAR_TAG = 'nonLinearAR'
HARDOBSTACLE_TAG = 'hardobstacle'
STOVOL_TAG = 'stovol'

def model_factory(model_tag, params):
    if model_tag == NONLINEARAR_TAG:
        return NonLinearAR(dimension=params['d'], 
                           alpha=params['alpha'], 
                           sv=params['sv'], 
                           sw=params['sw'])

    if model_tag == HARDOBSTACLE_TAG:
        return HardObstacle(dimension=params['d'], 
                           sv=params['sv'])

    if model_tag == STOVOL_TAG:
        return StoVol(dimension=params['d'], 
                           alpha=params['alpha'], 
                           beta=params['beta'], 
                           sv=params['sw'])

def run_func(model, n_samples, n_particles, T, y, output_dir):
    def func(rank):
        np.random.seed(rank)
        accepted_x_i, cand_x, n_trial, paccs = model.sample_n(n_samples=n_samples, 
                                                     n_particles=n_particles,
                                                     T=T, 
                                                     y=y)
        pickle_obj(accepted_x_i, accepted_fn(output_dir, rank))
        pickle_obj(cand_x, candidates_fn(output_dir, rank))
        pickle_obj(paccs, paccs_fn(output_dir, rank))
    return func

def run_parallel(num_workers, model, n_samples, n_particles, T, y, output_dir):
    f = run_func(model, n_samples, n_particles, T, y, output_dir)
    if num_workers > 1:
        with Pool(num_workers) as p:
            p.map(f, range(num_workers))
    else:
        f(0)

def process_outputs(num_workers, output_dir):
    accepted_xs_lst = [unpickle_obj(accepted_fn(output_dir, rank)) for rank in range(num_workers)]
    candidate_xs_lst = [unpickle_obj(candidates_fn(output_dir, rank)) for rank in range(num_workers)]
    paccs_lst = [unpickle_obj(paccs_fn(output_dir, rank)) for rank in range(num_workers)]

    accepted_xs = np.concatenate([cand_x[acc_x] for acc_x, cand_x in zip(accepted_xs_lst, candidate_xs_lst)])
    candidate_xs = np.concatenate(candidate_xs_lst)
    paccs = np.concatenate(paccs_lst)

    return accepted_xs, candidate_xs, paccs
