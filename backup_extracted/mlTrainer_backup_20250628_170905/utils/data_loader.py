import os
import json
import pandas as pd
from datetime import datetime
from core.immutable_gateway import verify_data_source

RECOMMENDATIONS_PATH = "data/live_recommendations.json"
HOLDINGS_PATH = "data/holdings.json"

def get_live_recommendations() -> pd.DataFrame:
    """Return live recommendations DataFrame, enforcing verified source"""
    if not os.path.exists(RECOMMENDATIONS_PATH):
        return pd.DataFrame()

    try:
        with open(RECOMMENDATIONS_PATH, "r") as f:
            raw_data = json.load(f)

        # Compliance check
        if not verify_data_source("polygon", "live_recommendations"):
            raise ValueError("Unverified recommendation source")

        df = pd.DataFrame(raw_data)
        expected_cols = [
            "Ticker", "Score", "Confidence", "Expected Profit %",
            "Current Price", "Recommended Entry", "Exit Target",
            "Expected Timeframe", "Strategy", "Market Regime"
        ]
        return df[expected_cols] if all(col in df.columns for col in expected_cols) else pd.DataFrame()

    except Exception as e:
        print(f"[data_loader] Error loading recommendations: {e}")
        return pd.DataFrame()

def get_current_holdings() -> pd.DataFrame:
    """Return current holdings DataFrame"""
    if not os.path.exists(HOLDINGS_PATH):
        return pd.DataFrame()

    try:
        with open(HOLDINGS_PATH, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        if "Progress %" not in df.columns:
            df["Progress %"] = 0.0
        return df
    except Exception as e:
        print(f"[data_loader] Error loading holdings: {e}")
        return pd.DataFrame()

def add_to_holdings(row: dict):
    """Add a recommended stock to holdings.json"""
    try:
        current = get_current_holdings()
        if row["Ticker"] in current["Ticker"].values:
            return  # Already added

        entry = {
            "Ticker": row["Ticker"],
            "Entry Price": row["Current Price"],
            "Date Selected": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Strategy at Entry": row["Strategy"],
            "Exit Target": row["Exit Target"],
            "Live Price": row["Current Price"],
            "Progress %": 0.0,
            "Days Held": 0
        }

        new_df = pd.concat([current, pd.DataFrame([entry])], ignore_index=True)
        with open(HOLDINGS_PATH, "w") as f:
            json.dump(new_df.to_dict(orient="records"), f, indent=2)

    except Exception as e:
        print(f"[data_loader] Error adding to holdings: {e}")

def remove_from_holdings(symbol: str):
    """Remove stock from holdings by Ticker"""
    try:
        current = get_current_holdings()
        updated = current[current["Ticker"] != symbol]
        with open(HOLDINGS_PATH, "w") as f:
            json.dump(updated.to_dict(orient="records"), f, indent=2)
    except Exception as e:
        print(f"[data_loader] Error removing from holdings: {e}")
