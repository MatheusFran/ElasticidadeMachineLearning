import mlflow


def save_mlflow(model, params, results, preprocess_pipeline=None, training_pipeline=None,
                host='localhost', port=5000):
    tracking_uri = f"http://{host}:{port}"
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = "Elasticidade Machine Learning"
    mlflow.set_experiment(experiment_name)
    registered_model_name = "elasticidade_model"

    with mlflow.start_run(run_name=experiment_name) as run:
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=registered_model_name
        )

        if preprocess_pipeline is not None:
            mlflow.sklearn.log_model(
                sk_model=preprocess_pipeline,
                artifact_path="preprocess_pipeline",
                registered_model_name=f"{registered_model_name}_preprocess"
            )

        if training_pipeline is not None:
            mlflow.sklearn.log_model(
                sk_model=training_pipeline,
                artifact_path="training_pipeline",
                registered_model_name=f"{registered_model_name}_training"
            )

        for key, value in params.items():
            mlflow.log_param(key, value)

        for key, value in results.items():
            mlflow.log_metric(key, value)

        print('Modelo, params, results e pipelines registrados com sucesso!')

        return run.info.run_id
