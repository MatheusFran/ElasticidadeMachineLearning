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
