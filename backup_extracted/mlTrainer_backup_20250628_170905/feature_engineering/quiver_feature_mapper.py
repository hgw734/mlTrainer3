import os
import glob
import logging
import pandas as pd
from datetime import datetime, timedelta

RAW_DIR = "staging/quiver_raw"
FEATURE_DIR = "features/quiver"
os.makedirs(FEATURE_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def _standardize_date(date_str):
    try:
        return pd.to_datetime(date_str).dt.date
    except Exception:
        return pd.NaT

def _normalize_amount_range(text):
    """
    Convert amount range string (e.g. "$15,001-$50,000") into average numeric value.
    """
    if not isinstance(text, str) or "-" not in text:
        return 0
    try:
        clean = text.replace("$", "").replace(",", "")
        low, high = clean.split("-")
        return (float(low) + float(high)) / 2
    except Exception:
        return 0

def process_quiver_csv(file_path: str) -> pd.DataFrame:
    """
    Load and transform a single Quiver CSV into normalized daily feature format.
    """
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.lower()
        df["date"] = pd.to_datetime(df["date"] if "date" in df.columns else df.get("transactiondate", pd.NaT), errors='coerce').dt.date
        df = df.dropna(subset=["date"])

        if "ticker" not in df.columns and "symbol" in df.columns:
            df["ticker"] = df["symbol"]

        if "ticker" not in df.columns:
            logger.warning(f"⚠️ No ticker column found in {file_path}, skipping")
            return pd.DataFrame()

        # Normalize key signal features
        if "amount" in df.columns:
            df["amount_score"] = df["amount"].apply(_normalize_amount_range)

        if "type" in df.columns:
            df["buy_flag"] = df["type"].str.lower().str.contains("buy").astype(int)
            df["sell_flag"] = df["type"].str.lower().str.contains("sell").astype(int)

        agg = df.groupby(["ticker", "date"]).agg({
            "buy_flag": "sum",
            "sell_flag": "sum",
            "amount_score": "sum"
        }).reset_index()

        # Add decay-weighted event count (e.g., trailing 7d activity)
        result_frames = []
        for ticker, group in agg.groupby("ticker"):
            group = group.sort_values("date")
            group["event_7d"] = group["buy_flag"].rolling(window=7, min_periods=1).sum()
            result_frames.append(group)

        result_df = pd.concat(result_frames).reset_index(drop=True)
        return result_df

    except Exception as e:
        logger.error(f"❌ Failed to process {file_path}: {e}")
        return pd.DataFrame()

def build_quiver_features():
    logger.info("🚀 Starting feature mapping from Quiver raw data...")
    all_files = glob.glob(os.path.join(RAW_DIR, "*.csv"))
    if not all_files:
        logger.warning("⚠️ No raw Quiver files found.")
        return

    combined = pd.DataFrame()
    for file in all_files:
        df = process_quiver_csv(file)
        if not df.empty:
            combined = pd.concat([combined, df], ignore_index=True)

    if combined.empty:
        logger.warning("⚠️ No features generated from Quiver data.")
        return

    # Save one feature file per ticker
    for ticker, group in combined.groupby("ticker"):
        group = group.sort_values("date")
        output_path = os.path.join(FEATURE_DIR, f"{ticker}_features.csv")
        group.to_csv(output_path, index=False)
        logger.info(f"✅ Saved Quiver feature file: {output_path} ({len(group)} rows)")

    logger.info("🎯 Feature mapping complete.")
