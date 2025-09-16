from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


class LabelEnconding(BaseEstimator, TransformerMixin):
    def __init__(self, columns,unknown_value = -1):
        self.columns = columns
        self.encoders = {}
        self.unknown_value = unknown_value


    def fit(self, X, y=None):
        for col in self.columns:
            le = LabelEncoder()
            le.fit(X[col])
            self.encoders[col] = le
        return self

    def transform(self, X):
        X = X.copy()
        for col, le in self.encoders.items():
            mapping = {cls: idx for idx, cls in enumerate(le.classes_)}
            X[col] = X[col].map(mapping).fillna(self.unknown_value).astype(int)
        return X