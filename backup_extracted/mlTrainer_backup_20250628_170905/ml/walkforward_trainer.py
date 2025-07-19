import os
import json
import logging
import joblib
import numpy as np
import xgboost as xgb
from typing import Dict
from ml.dataset_builder import build_dataset

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def train_walkforward_model(ticker: str, config: Dict = None) -> Dict:
    """
    Trains a walk-forward model for the given ticker using engineered features.
    Supports: XGBoost (default)
    """
    if config is None:
        config = {
            "model_type": "xgboost",
            "horizon_days": 10,
            "target_column": "close"
        }

    try:
        X_train, X_val, y_train, y_val = build_dataset(
            ticker=ticker,
            target_column=config.get("target_column", "close"),
            horizon_days=config.get("horizon_days", 10)
        )

        if X_train is None:
            logger.warning(f"⚠️ No training data for {ticker}")
            return {}

        model_type = config.get("model_type", "xgboost")
        model = None

        if model_type == "xgboost":
            model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
            model.fit(X_train, y_train)

        # Save model
        model_path = os.path.join(MODEL_DIR, f"{ticker}_{model_type}_model.pkl")
        joblib.dump(model, model_path)

        # Evaluate
        preds = model.predict(X_val)
        mae = np.mean(np.abs(preds - y_val))
        directional_acc = np.mean((preds > 0) == (y_val > 0))

        logger.info(f"✅ {ticker} model trained. MAE={mae:.5f}, Directional Acc={directional_acc:.2%}")

        return {
            "ticker": ticker,
            "model_type": model_type,
            "model_path": model_path,
            "mae": mae,
            "directional_accuracy": directional_acc
        }

    except Exception as e:
        logger.error(f"❌ Training failed for {ticker}: {e}")
        return {}
