import mlflow

mlflow.projects.run(
    './run_model',
    backend='local',
    synchronous=False,
    parameters = {'T' : 500,
                 'dimension' : 1,
                 'n_samples': 50,
                 'n_particles' : 1000,
                 'n_workers' : 10})

