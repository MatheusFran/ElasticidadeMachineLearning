from src.data.data_ingestion import IngestionFactory, DataIngestionContext
import pandas as pd


def load_data(db_params: dict) -> pd.DataFrame:
    strategy = IngestionFactory.create("all")
    context = DataIngestionContext(db_params, strategy)
    df = context.execute()
    return df
