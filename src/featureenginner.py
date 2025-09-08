import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler


class ClassFeatureEngineering:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def create_data_col(self, col):
        df = self.df
        self.df[f'{col}_year'] = df[col].dt.year
        self.df[f'{col}_month'] = df[col].dt.month
        self.df[f'{col}_day'] = df[col].dt.day
        self.df[f'{col}_weekday'] = df[col].dt.weekday
        self.df[f'{col}_is_weekend'] = df[col].dt.weekday >= 5
        self.df[f'{col}_hour'] = df[col].dt.hour

    def encoding(self, cols):
        df_encoded = pd.get_dummies(self.df[cols], prefix=cols)
        self.df = pd.concat([self.df, df_encoded], axis=1)
        self.df.drop(columns=cols, inplace=True)

    def name_product(self):
        encoder = LabelEncoder()
        self.df['name'] = encoder.fit_transform(self.df['name']).astype(str)

    def price_sales(self):
        self.df['receita'] = self.df['price'] * self.df['sales_volume']
        self.df['price_diff'] = self.df['price'] - self.df['price'].mean()

    def standard_pad(self):
        scaler = StandardScaler()
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        scaled = scaler.fit_transform(self.df[num_cols])
        df_scaled = pd.DataFrame(scaled, columns=num_cols, index=self.df.index)
        self.df[num_cols] = df_scaled

    def __call__(self, date_cols, onehot_cols):
        if date_cols:
            for col in date_cols:
                self.create_data_col(col)
        self.encoding(onehot_cols)
        self.name_product()
        self.price_sales()
        self.standard_pad()
        return self.df
