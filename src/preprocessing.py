import pandas as pd
import numpy as np


class ClassPreprocessing:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def del_col_unique(self):
        unique_counts = self.df.nunique()
        cols_to_drop = unique_counts[unique_counts == 1].index.tolist()
        self.df.drop(columns=cols_to_drop, inplace=True)

    def na_value(self):
        mask = self.df['description'].isna() | self.df['description'].astype(str).str.fullmatch('NaN')
        self.df['description'] = np.where(mask, self.df['name'].fillna('produto sem nome'), self.df['description'])

    def drop_col(self, list_col=None):
        if list_col:
            self.df.drop(columns=list_col, inplace=True, errors='ignore')

    def __call__(self, drop_columns=None):
        self.del_col_unique()
        self.na_value()
        self.drop_col(list_col=drop_columns)
        return self.df
