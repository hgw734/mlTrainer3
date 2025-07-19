import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import logging

from ml.models.lstm_model import train_lstm
from ml.models.xgboost_model import train_xgboost
from ml.models.transformer_model import train_transformer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def train_ensemble(X: np.ndarray, y: np.ndarray, window_size: int = 10):
    """
    Train base models (LSTM, XGBoost, Transformer) and a meta-model to combine predictions.
    Returns a dict with trained models and meta-learner.
    """
    logger.info("🤖 Training ensemble model (LSTM + XGBoost + Transformer)...")

    # Prepare train/test for ensemble training
    X_train, X_meta, y_train, y_meta = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train base models
    model_lstm = train_lstm(X_train, y_train, window_size)
    model_xgb = train_xgboost(X_train, y_train)
    model_tf = train_transformer(X_train, y_train, window_size)

    # Generate predictions for meta model
    preds_lstm = model_lstm.predict(X_meta).flatten()
    preds_xgb = model_xgb.predict(X_meta)
    preds_tf = model_tf.predict(X_meta).flatten()

    # Stack predictions
    stacked_preds = np.vstack([preds_lstm, preds_xgb, preds_tf]).T
    meta_model = LinearRegression()
    meta_model.fit(stacked_preds, y_meta)

    logger.info("✅ Ensemble model training complete.")
    return {
        "lstm": model_lstm,
        "xgboost": model_xgb,
        "transformer": model_tf,
        "meta_model": meta_model
    }
