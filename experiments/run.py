import mlflow

mlflow.projects.run(
    './run_model',
    backend='local',
    synchronous=False,
    parameters = {'T' : 10,
                 'n_workers' : 1})

