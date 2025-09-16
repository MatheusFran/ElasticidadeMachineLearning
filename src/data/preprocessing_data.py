import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PreprocessingData(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame,dropUnique:bool = True):
        df = X.copy()
        df['price_unit'] = df['sales_amount'] / df['quantity']

        cols_changetype = ['customer_id', 'transaction_id']
        col_date = 'date'
        col_drop = ['id']

        df[cols_changetype] = df[cols_changetype].astype(int)

        if not pd.api.types.is_datetime64_any_dtype(df[col_date]):
            df[col_date] = pd.to_datetime(df[col_date], errors='coerce')

        df['date_year'] = df[col_date].dt.year
        df['date_month'] = df[col_date].dt.month
        df['date_day'] = df[col_date].dt.day

        df.drop(columns=[col_date], inplace=True)


        df = df.drop(col_drop, axis=1)
        if dropUnique:
            unique_counts = df.nunique()
            cols_to_drop = unique_counts[unique_counts == 1].index.tolist()
            df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

        return df
