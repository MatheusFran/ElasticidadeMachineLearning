import mlflow


def save_mlflow(model, params, results, host='localhost', port=5000):
    tracking_uri = f"http://{host}:{port}"
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = "Elasticidade Machine Learning"
    mlflow.set_experiment(experiment_name)
    registered_model_name = "elasticidade_model"

    with mlflow.start_run(run_name=experiment_name):
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=registered_model_name
        )

        for key, value in params.items():
            mlflow.log_param(key, value)

        for key, value in results.items():
            mlflow.log_metric(key, value)

        print('Modelo, params e results, registrados com sucesso!')
