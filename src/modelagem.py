import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV


class Modeling:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.X = None
        self.y = None
        self.tunning = None
        self.best_model = None
        self.best_params = None
        self.best_score = None
        self.results = None

    def split_data(self):
        self.X = self.df.drop(columns=['sales_volume'])
        self.y = self.df['sales_volume']
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def feature_selection_rfe(self, model):
        X_train, x_test, y_train, y_test = self.split_data()
        rfe = RFE(model, n_features_to_select=5)
        rfe.fit(X_train, y_train)
        selected_features = X_train.columns[rfe.support_]
        return selected_features

    def tunning_hiper(self, model, params):
        tunning = RandomizedSearchCV(
            model,
            param_distributions=params,
            n_iter=10,
            cv=3,
            scoring='neg_mean_squared_error',
            random_state=42
        )
        tunning.fit(self.X, self.y)

        self.tunning = tunning
        self.best_model = tunning.best_estimator_
        self.best_params = tunning.best_params_
        self.best_score = -tunning.best_score_

        return self.best_model, self.best_params, self.best_score

    def evaluate_model(self, model):
        X_train, x_test, y_train, y_test = self.split_data()
        y_pred = model.predict(x_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results = {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
        }

        self.results = results
        return self.results
