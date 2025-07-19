import os
import logging
import pandas as pd
from datetime import datetime

QUIVER_DIR = "features/quiver"
PRICE_DIR = "data/price"
REGIME_FILE = "features/regime/regime_labels.csv"
MERGED_DIR = "features/merged"
os.makedirs(MERGED_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def merge_quiver_with_price(ticker: str, include_regime: bool = False):
    try:
        quiver_path = os.path.join(QUIVER_DIR, f"{ticker}_features.csv")
        price_path = os.path.join(PRICE_DIR, f"{ticker}_price.csv")

        if not os.path.exists(quiver_path) or not os.path.exists(price_path):
            logger.warning(f"⚠️ Missing files for {ticker}. Skipping.")
            return

        quiver_df = pd.read_csv(quiver_path, parse_dates=["date"])
        price_df = pd.read_csv(price_path, parse_dates=["date"])
        price_df = price_df.sort_values("date")

        merged = pd.merge(price_df, quiver_df, on="date", how="left")
        merged = merged.fillna(0)

        if include_regime and os.path.exists(REGIME_FILE):
            regime_df = pd.read_csv(REGIME_FILE, parse_dates=["date"])
            merged = pd.merge(merged, regime_df, on="date", how="left")
            merged["regime_score"] = merged["regime_score"].fillna(method="ffill").fillna(0)

        merged = merged.sort_values("date")
        out_path = os.path.join(MERGED_DIR, f"{ticker}_matrix.csv")
        merged.to_csv(out_path, index=False)
        logger.info(f"✅ Saved merged feature matrix for {ticker}: {out_path} ({len(merged)} rows)")

    except Exception as e:
        logger.error(f"❌ Failed to merge features for {ticker}: {e}")

def build_all_quiver_matrices():
    files = [f for f in os.listdir(QUIVER_DIR) if f.endswith("_features.csv")]
    tickers = [f.split("_")[0] for f in files]

    for ticker in tickers:
        merge_quiver_with_price(ticker)

    logger.info("🎯 All Quiver feature matrices generated.")
