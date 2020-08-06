import mlflow
import os, sys
import time
dir_path = os.path.dirname(os.path.realpath(__file__))

print(dir_path)

run_ids = []



param_list = []

param_list.append({'model_tag':'hardobstacle',
                 'T' : 250,
                 'dimension' : 1,
                 'n_samples': 500,
                 'n_particles' : 250,
                 'n_workers' : 1})
param_list.append({'model_tag':'hardobstacle',
                 'T' : 250,
                 'dimension' : 1,
                 'n_samples': 500,
                 'n_particles' : 500,
                 'n_workers' : 1})
param_list.append({'model_tag':'hardobstacle',
                 'T' : 250,
                 'dimension' : 1,
                 'n_samples': 500,
                 'n_particles' : 1250,
                 'n_workers' : 1})

param_list.append({'model_tag':'nonLinearAR',
                 'T' : 500,
                 'dimension' : 1,
                 'n_samples': 500,
                 'n_particles' : 500,
                 'n_workers' : 1})
param_list.append({'model_tag':'nonLinearAR',
                 'T' : 500,
                 'dimension' : 1,
                 'n_samples': 500,
                 'n_particles' : 1000,
                 'n_workers' : 1})
param_list.append({'model_tag':'nonLinearAR',
                 'T' : 500,
                 'dimension' : 1,
                 'n_samples': 500,
                 'n_particles' : 2000,
                 'n_workers' : 1})    

for id, params in enumerate(param_list):
    if params['T'] == 500:
        run = mlflow.projects.run(
            os.path.join(dir_path,'./run_model'),
            backend='local',
            synchronous=False,
            parameters = params)
        run_ids.append(run.run_id)

obj = mlflow.tracking.MlflowClient()

def runs_finished(run_ids):
    return all(obj.get_run(run_id).info.status == 'FINISHED' for run_id in run_ids)

#while not runs_finished(run_ids):
#    time.sleep(1.)

#obj.get_run(run1.run_id)

#obj.get_metric_history(run1.run_id, 'w0-iter')