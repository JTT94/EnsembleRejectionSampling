import mlflow
import os, sys
import time
dir_path = os.path.dirname(os.path.realpath(__file__))

print(dir_path)

run_ids = []

run1 = mlflow.projects.run(
    os.path.join(dir_path,'./run_model'),
    backend='local',
    synchronous=False,
    parameters = {'T' : 100,
                 'dimension' : 1,
                 'n_samples': 25,
                 'n_particles' : 1000,
                 'n_workers' : 20})
run_ids.append(run1.run_id)

run2 = mlflow.projects.run(
    os.path.join(dir_path,'./run_model'),
    backend='local',
    synchronous=False,
    parameters = {'T' : 500,
                 'dimension' : 1,
                 'n_samples': 25,
                 'n_particles' : 2000,
                 'n_workers' : 20})
obj = mlflow.tracking.MlflowClient()

run_ids.append(run2.run_id)

def runs_finished(run_ids):
    return all(obj.get_run(run_id).info.status == 'FINISHED' for run_id in run_ids)

while not runs_finished(run_ids):
    time.sleep(1.)

