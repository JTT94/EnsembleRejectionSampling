import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from mlflow import log_metric, log_param, log_artifacts, log_artifact

matplotlib.use('Agg')

sys.path.append('../..')
from ers.models import NonLinearAR
from ers.runners import run_parallel, process_outputs, model_factory
from ers.utils import pickle_obj, unpickle_obj
import argparse

# if __name__ == "__main__":
# get arguments
parser = argparse.ArgumentParser()
parser.add_argument('--T', action='store', type=int, default=50)
parser.add_argument('--alpha', action='store', type=int, default=0.9)
parser.add_argument('--beta', action='store', type=int, default=0.7)
parser.add_argument('--sv', action='store', type=int, default=0.3)
parser.add_argument('--sw', action='store', type=int, default=0.1)
parser.add_argument('--d', action='store', type=int, default=1)
parser.add_argument('--n_samples', action='store', type=int, default=10)
parser.add_argument('--n_particles', action='store', type=int, default=500)
parser.add_argument('--seed', action='store', type=int, default=16)
parser.add_argument('--n_workers', action='store', type=int, default=2)
parser.add_argument('--out_dir', action='store', type=str, default='./')
parser.add_argument('--model_tag', action='store', type=str, default='hardobstacle')
args = parser.parse_args()
# args = parser.parse_args([])

# General Params
T = args.T
d = args.d
seed = args.seed
n_samples = args.n_samples
n_particles = args.n_particles
out_dir = args.out_dir
n_workers = args.n_workers
model_tag = args.model_tag

params = args.__dict__
np.random.seed(seed=seed)

# init model
model = model_factory(model_tag, params)

if hasattr(model, 'generate_x'):
    xtrue = model.generate_x(T)
    y = model.generate_y(xtrue)

    # log true x, y
    fp = os.path.join(out_dir, 'xtrue.pkl')
    pickle_obj(xtrue, fp)
    log_artifact(fp)

    fp = os.path.join(out_dir, 'observations.pkl')
    pickle_obj(y, fp)
    log_artifact(fp)
else:
    xtrue = None
    y = None

# run sampler
run_parallel(n_workers, model, n_samples, n_particles, T, y, out_dir)
accepted_xs, candidate_xs, paccs = process_outputs(n_workers, out_dir)

fp = os.path.join(out_dir, 'candidate_xs.pkl')
pickle_obj(candidate_xs, fp)
log_artifact(fp)

fp = os.path.join(out_dir, 'accepted_xs.pkl')
pickle_obj(accepted_xs, fp)
log_artifact(fp)

pacc_std = np.sqrt(np.var(paccs))
pacc_avg = np.mean(paccs)
log_metric('pacc_std', pacc_std)
log_metric('pacc_avg', pacc_avg)

fig, axs = plt.subplots(d, gridspec_kw={'hspace': 0.4})
for d_i in range(d):
    if d > 1:
        ax = axs[d_i]
    else:
        ax = axs
    ax.set_title('Dimension {0}'.format(d_i))
    for i in range(candidate_xs.shape[0]):
        ax.plot(candidate_xs[i, :, d_i], color='gray')

    for i in range(accepted_xs.shape[0]):
        ax.plot(accepted_xs[i, :, d_i], color='blue')

    average_x = np.mean(accepted_xs, axis=0)
    if xtrue is not None:
        ax.plot(xtrue[:, d_i], color='red')
    ax.plot(average_x[:, d_i], color='pink')

fp = os.path.join(out_dir, 'summary_plot.png')
plt.savefig(fp)
log_artifact(fp)
