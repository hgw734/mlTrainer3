import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def train_xgboost(X_raw: np.ndarray, y_raw: np.ndarray):
    """
    Train an XGBoost regression model on the given data.
    Returns the trained model.
    """
    logger.info("📊 Training XGBoost model...")

    # Basic train-test split for internal validation
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        verbosity=1,
        random_state=42,
        tree_method="hist"  # Fast CPU method compatible with Replit
    )

    model.fit(X_train, y_train)

    # Evaluate performance
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    logger.info(f"✅ XGBoost training complete. RMSE on test set: {rmse:.4f}")

    return model
