# ml/model_registry.py

import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor  # Temporarily disabled due to libgomp dependency
from catboost import CatBoostRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import numpy as np

logger = logging.getLogger(__name__)

# Define simple wrapper classes for deep models
class SimpleLSTMWrapper:
    def __init__(self, input_shape=(10, 1), units=50):
        self.model = Sequential()
        self.model.add(LSTM(units, input_shape=input_shape))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self, X, y):
        X_reshaped = np.reshape(X.values, (X.shape[0], X.shape[1], 1))
        self.model.fit(X_reshaped, y.values, epochs=5, batch_size=16, verbose=0)

    def predict(self, X):
        X_reshaped = np.reshape(X.values, (X.shape[0], X.shape[1], 1))
        return self.model.predict(X_reshaped).flatten()

# Registry of real, production models
MODEL_REGISTRY = {
    "random_forest": RandomForestRegressor(n_estimators=100),
    "gradient_boosting": GradientBoostingRegressor(),
    "linear_regression": LinearRegression(),
    "ridge": Ridge(),
    "lasso": Lasso(),
    "svr": SVR(),
    "decision_tree": DecisionTreeRegressor(),
    "xgboost": XGBRegressor(verbosity=0),
    "xgboost_tuned": XGBRegressor(verbosity=0, n_estimators=200, max_depth=4, learning_rate=0.05),
    "catboost": CatBoostRegressor(verbose=0),
    "lstm": SimpleLSTMWrapper(),  # production LSTM wrapper
}

def get_model_by_name(name: str):
    model = MODEL_REGISTRY.get(name.lower())
    if model:
        logger.info(f"✅ Model '{name}' retrieved from registry.")
    else:
        logger.warning(f"⚠️ Model '{name}' not found in registry.")
    return model
