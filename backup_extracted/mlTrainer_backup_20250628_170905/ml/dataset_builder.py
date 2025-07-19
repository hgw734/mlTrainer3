import os
import logging
import pandas as pd
from datetime import timedelta
from typing import Tuple

MERGED_DIR = "features/merged"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def build_dataset(
    ticker: str,
    target_column: str = "close",
    horizon_days: int = 10,
    feature_lag_days: int = 1,
    validation_ratio: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Builds ML-ready dataset with lagged features and future return target.
    """
    try:
        file_path = os.path.join(MERGED_DIR, f"{ticker}_matrix.csv")
        if not os.path.exists(file_path):
            logger.warning(f"⚠️ Merged matrix not found for {ticker}")
            return None, None, None, None

        df = pd.read_csv(file_path, parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Define target as % return N days ahead
        df["target"] = df[target_column].shift(-horizon_days) / df[target_column] - 1
        df = df.dropna()

        feature_cols = [
            col for col in df.columns
            if col not in ["date", "target", "ticker", "symbol", "open", "high", "low", "close"]
        ]

        X = df[feature_cols]
        y = df["target"]

        split_idx = int(len(df) * (1 - validation_ratio))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        logger.info(f"✅ Built dataset for {ticker}: {len(X_train)} train / {len(X_val)} val samples")
        return X_train, X_val, y_train, y_val

    except Exception as e:
        logger.error(f"❌ Failed to build dataset for {ticker}: {e}")
        return None, None, None, None
