import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import pandas as pd
import joblib
from pathlib import Path

from src.model.model import ModelTrain

def pipeline_ml():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("elasticidade_experiment")

    with mlflow.start_run() as run:

        model_trainer = ModelTrain()
        model_trainer.fit()

        trained_model = model_trainer.model
        encoder = model_trainer.encoder
        metrics = model_trainer.metrics
        params = model_trainer.params

        X_train, X_test, y_train, y_test = model_trainer.load_and_preprocess()
        X_test_encoded = encoder.transform(X_test)
        y_pred = trained_model.predict(X_test_encoded)

        signature = infer_signature(X_test_encoded, y_pred)

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        mlflow.log_params(params)

        artifacts_path = Path('../../artifacts')
        artifacts_path.mkdir(parents=True, exist_ok=True)
        model_path = artifacts_path / 'model'
        encoder_path = artifacts_path / 'enconder'
        model_path.mkdir(exist_ok=True)
        encoder_path.mkdir(exist_ok=True)

        joblib.dump(trained_model, model_path / "model_v1.joblib")
        joblib.dump(encoder, encoder_path / "enconder_v1.joblib")

        mlflow.sklearn.save_model(
            sk_model=trained_model,
            path="model",
            signature=signature,
            input_example=X_test_encoded[:5].values if hasattr(X_test_encoded, 'values') else X_test_encoded[:5],
            pip_requirements=[
                "scikit-learn",
                "pandas",
                "numpy"
            ]
        )

        mlflow.log_artifact(str(encoder_path / "enconder_v1.joblib"), "preprocessing")

        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(
            model_uri=model_uri,
            name="elasticidade_model"
        )

        print(f"Modelo registrado no MLflow com run_id={run.info.run_id}")
        return run.info.run_id


if __name__ == '__main__':
    pipeline_ml()