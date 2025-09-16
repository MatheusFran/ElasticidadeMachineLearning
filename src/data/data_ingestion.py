import pandas as pd
import psycopg2
from abc import ABC, abstractmethod


class IngestionStrategy(ABC):
    @abstractmethod
    def read(self, conn):
        pass


class ReadAllProducts(IngestionStrategy):
    def read(self, conn):
        query = "SELECT * FROM store"
        return pd.read_sql(query, conn)


class ReadProductByID(IngestionStrategy):
    def __init__(self, product_id):
        self.product_id = product_id

    def read(self, conn):
        query = "SELECT * FROM store WHERE id = %s"
        return pd.read_sql(query, conn, params=(self.product_id,))


class DataIngestionContext:
    def __init__(self, db_params, strategy: IngestionStrategy):
        self.db_params = db_params
        self.strategy = strategy

    def _connection(self):
        return psycopg2.connect(**self.db_params)

    def execute(self) -> pd.DataFrame:
        conn = self._connection()
        try:
            return self.strategy.read(conn)
        except Exception as e:
            print("Erro na leitura:", e)
            return pd.DataFrame()
        finally:
            conn.close()


class IngestionFactory:
    @staticmethod
    def create(strategy_type: str, **kwargs) -> IngestionStrategy:
        if strategy_type == "all":
            return ReadAllProducts()
        elif strategy_type == "by_id":
            return ReadProductByID(kwargs.get("product_id"))
        else:
            raise ValueError(f"Ingestion desconhecido: {strategy_type}")


