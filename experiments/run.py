import mlflow
import os, sys
import time
dir_path = os.path.dirname(os.path.realpath(__file__))

run_ids = []
param_list = []
synchronous=True

T=100
n_samples=500
n_workers=1


for dimension in [1]:
    for T in [100, 250, 500]:
        for num_particles in [T, 2*T, 5*T]:
    
            params = {'model_tag':'nonLinearAR',
                    'T' : T,
                    'dimension' : dimension,
                    'n_samples': n_samples//n_workers ,
                    'n_particles' : num_particles,
                    'n_workers' : n_workers}
            param_list.append(params)
    
for id, params in enumerate(param_list):
    
    run = mlflow.projects.run(
        os.path.join(dir_path,'./run_model'),
        backend='local',
        synchronous=synchronous,
        parameters = params)

    run_ids.append(run.run_id)

obj = mlflow.tracking.MlflowClient()

def runs_finished(run_ids):
    return all(obj.get_run(run_id).info.status == 'FINISHED' for run_id in run_ids)

while not runs_finished(run_ids):
    time.sleep(.1)

