# api/main.py
import sys
import os
from pathlib import Path

# resolver imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from fastapi import FastAPI, HTTPException
import mlflow.sklearn
import pandas as pd
import joblib
from pydantic import BaseModel
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.data.preprocessing_data import PreprocessingData
from src.utils.elasticidade import calculate_elasticity

app = FastAPI(title="Elasticidade API", version="1.0.0")


class ProductData(BaseModel):
    id: str
    date: str
    customer_id: int
    transaction_id: int
    sku_category: str
    sku: str
    sales_amount: float
    price_unit: float


class BatchData(BaseModel):
    products: List[ProductData]


mlflow.set_tracking_uri("http://mlflow:5000")

try:
    model_name = "elasticidade_model"
    model_version = "4"
    model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{model_version}")
    logger.info(f"Modelo carregado do MLflow: {model_name} versão {model_version}")

    try:

        client = mlflow.tracking.MlflowClient()
        model_version_details = client.get_model_version(model_name, model_version)
        run_id = model_version_details.run_id

        encoder_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{run_id}/preprocessing/enconder_v1.joblib"
        )
        encoder = joblib.load(encoder_path)
        logger.info("Encoder carregado do MLflow")
    except Exception as e:
        logger.warning(f"Não foi possível carregar encoder do MLflow: {e}")
        encoder_local_path = project_root / "artifacts" / "enconder" / "enconder_v1.joblib"
        if encoder_local_path.exists():
            encoder = joblib.load(encoder_local_path)
            logger.info("Encoder carregado do path local")
        else:
            encoder = None
            logger.warning("Encoder não encontrado")

except Exception as e:
    logger.error(f"Erro ao carregar modelo: {e}")
    # Fallback para modelo local se disponível
    try:
        model_local_path = project_root / "artifacts" / "model" / "model_v1.joblib"
        encoder_local_path = project_root / "artifacts" / "enconder" / "enconder_v1.joblib"

        if model_local_path.exists() and encoder_local_path.exists():
            model = joblib.load(model_local_path)
            encoder = joblib.load(encoder_local_path)
            logger.info("Modelo e encoder carregados dos paths locais")
        else:
            raise Exception("Nem modelo do MLflow nem local estão disponíveis")
    except Exception as local_error:
        logger.error(f"Erro ao carregar modelo local: {local_error}")
        model = None
        encoder = None


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "encoder_loaded": encoder is not None
    }


@app.post("/elasticidade")
def predict_elasticidade(batch: BatchData):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado")

    try:
        df = pd.DataFrame([p.model_dump() for p in batch.products])
        logger.info(f"Recebidos {len(df)} produtos para predição")

        preprocessor = PreprocessingData()
        df_processed = preprocessor.transform(df, dropUnique=False)
        logger.info(f"Colunas após preprocessing: {df_processed.columns.tolist()}")

        if encoder is not None:
            df_encoded = encoder.transform(df_processed)
            logger.info(f"Shape após encoder: {df_encoded.shape}")
        else:
            logger.warning("Encoder não disponível, usando dados sem encoding")
            df_encoded = df_processed

        predictions = model.predict(df_encoded)

        elasticity_result = calculate_elasticity(df_processed,price_col="price_unit",quantity_col=predictions)

        return {
            "predictions": predictions,
            "elasticity": elasticity_result,
            "status": "success"
        }




    except Exception as e:
        logger.error(f"Erro na predição: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
