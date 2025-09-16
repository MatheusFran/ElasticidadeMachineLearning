import os
import numpy as np
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.features.enconding import LabelEnconding
from src.data.load_data import load_data
from src.data.preprocessing_data import PreprocessingData

load_dotenv()

class ModelTrain:
    def __init__(self, model=None, encoder_cols=None, test_size=0.2, random_state=42):
        self.db_params = {
            "database": os.getenv("DB_NAME"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "host": os.getenv("DB_HOST"),
            "port": os.getenv("DB_PORT"),
        }
        self.test_size = test_size
        self.random_state = random_state
        self.encoder_cols = encoder_cols or ["sku", "sku_category"]

        self.model = model or RandomForestRegressor(
            random_state=random_state, n_estimators=100, n_jobs=-1
        )

        self.encoder = LabelEnconding(columns=self.encoder_cols)
        self.metrics = {}
        self.params = {}

        self._X_train = None
        self._X_test = None
        self._y_train = None
        self._y_test = None

    def load_and_preprocess(self):
        if self._X_train is None:
            df = load_data(self.db_params)
            df_processed = PreprocessingData().transform(df)

            X = df_processed.drop(columns=["quantity"], axis=1)
            y = df_processed["quantity"]

            self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )

        return self._X_train, self._X_test, self._y_train, self._y_test

    def fit(self):
        X_train, X_test, y_train, y_test = self.load_and_preprocess()

        X_train_encoded = self.encoder.fit_transform(X_train)
        X_test_encoded = self.encoder.transform(X_test)

        self.model.fit(X_train_encoded, y_train)

        y_pred = self.model.predict(X_test_encoded)
        self.metrics = self.evaluate(y_test, y_pred)
        self.params = self.model.get_params()

        return self

    def evaluate(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}
        return metrics

    def predict(self, X):
        X_encoded = self.encoder.transform(X)
        return self.model.predict(X_encoded)

    def get_sklearn_model(self):
        return self.model

    def get_encoder(self):
        return self.encoder