import pandas as pd
import psycopg2


class DataIngestion:
    def __init__(self, db_params):
        self.df = pd.DataFrame()
        self.db_params = db_params

    def _connection(self):
        return psycopg2.connect(**self.db_params)

    def read_db(self):
        conn = self._connection()
        try:
            df = pd.read_sql('SELECT * FROM zara', conn)
            return df
        except Exception as e:
            print(e)
        finally:
            conn.close()

    def read_products(self):
        conn = self._connection()
        try:
            query = "SELECT name, price FROM zara"
            df = pd.read_sql(query, conn)
            return df
        finally:
            conn.close()

    def get_product_by_name(self, name: str) -> pd.DataFrame:
        conn = self._connection()
        try:
            query = "SELECT * FROM zara WHERE name = %s"
            df = pd.read_sql(query, conn, params=(name,))
            return df
        except Exception as e:
            print("Erro ao buscar produto por nome:", e)
            return pd.DataFrame()
        finally:
            conn.close()

    def get_product_by_id(self, product_id: int) -> pd.DataFrame:
        conn = self._connection()
        try:
            query = "SELECT * FROM zara WHERE id = %s"
            df = pd.read_sql(query, conn, params=(product_id,))
            return df
        except Exception as e:
            print("Erro ao buscar produto por ID:", e)
            return pd.DataFrame()
        finally:
            conn.close()
