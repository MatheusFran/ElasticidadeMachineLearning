from fastapi import FastAPI, Request
from mlflow import MlflowClient
from pydantic import BaseModel
from mlflow.sklearn import load_model
import mlflow
import pandas as pd
from src.data_ingestion import DataIngestion
from src.elasticidade import calculate_elasticity
import os
from dotenv import load_dotenv

app = FastAPI()

mlflow.set_tracking_uri("http://mlflow:5000")

model_name = "elasticidade_model"
model_version = "1"
model = load_model(model_uri=f"models:/{model_name}/{model_version}")


class ElasticidadeInput(BaseModel):
    promotion: bool
    name: str
    price: float


load_dotenv()

db_params={
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}

ingestion = DataIngestion(db_params)


def makePrediction(model, data: dict) -> float:
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return float(prediction[0])


@app.post("/prediction/elasticidade")
async def predict_elasticidade(request: ElasticidadeInput, fastapi_req: Request):
    product_info = ingestion.get_product_by_name(request.name)
    if product_info.empty:
        return {"error": f"Produto '{request.name}' não encontrado."}

    features = {
        "promotion": request.promotion,
        "name": request.name,
        "price": request.price,
        "receita": 0.0,
        "price_diff": 0.0
    }

    predicted_sales = makePrediction(model, features)

    features["receita"] = features["price"] * predicted_sales
    price_mean = product_info["price"].mean()
    features["price_diff"] = features["price"] - price_mean

    final_sales = makePrediction(model, features)

    df = pd.DataFrame([{
        "name": request.name,
        "price": features["price"],
        "sales_volume": final_sales,
        "promotion": request.promotion
    }])
    elasticity = calculate_elasticity(df, group_by=["promotion"])

    return {
        "product": request.name,
        "price": features["price"],
        "predicted_sales_volume": final_sales,
        "receita": features["receita"],
        "price_diff": features["price_diff"],
        "elasticity": elasticity.to_dict(orient="records")
    }
