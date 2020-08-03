import mlflow
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

print(dir_path)

mlflow.projects.run(
    os.path.join(dir_path,'./run_model'),
    backend='local',
    synchronous=False,
    parameters = {'T' : 50,
                 'dimension' : 1,
                 'n_samples': 2,
                 'n_particles' : 2000,
                 'n_workers' : 1})


mlflow.projects.run(
    os.path.join(dir_path,'./run_model'),
    backend='local',
    synchronous=True,
    parameters = {'T' : 200,
                 'dimension' : 1,
                 'n_samples': 1,
                 'n_particles' : 2000,
                 'n_workers' : 1})


