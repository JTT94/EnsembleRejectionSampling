import mlflow
import numpy as np
import os
import time
import tqdm
from mlflow import log_metric
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

        accepted_x_indices = []
        candidates_x = []
        paccss = []
        n_trials = 0
        start_time = time.time()
        for i in tqdm.tqdm(range(n_samples)):
            cand_x, n_trial, paccs = model.sample_one(n_particles, T, y)
            candidates_x.append(cand_x)
            paccss.append(paccs)
            n_trials += n_trial
            accepted_x_indices.append(n_trials - 1)

            pacc_std = np.sqrt(np.var(paccs))
            pacc_avg = np.mean(paccs)

            run_time = time.time() - start_time
            log_metric('w{0}-iter'.format(rank), i)
            log_metric('w{0}-run_time'.format(rank), run_time, i)
            log_metric('w{0}-pacc_avg'.format(rank), pacc_avg, i)
            log_metric('w{0}-pacc_std'.format(rank), pacc_std, i)

        candidates_x = np.concatenate(candidates_x)
        paccs = np.concatenate(paccss)
        accepted_x_i = np.array(accepted_x_indices)

        pickle_obj(accepted_x_i, accepted_fn(output_dir, rank))
        pickle_obj(candidates_x, candidates_fn(output_dir, rank))
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
