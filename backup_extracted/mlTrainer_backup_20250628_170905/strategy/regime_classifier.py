import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import logging

from data_sources.fred_api import fetch_vix, fetch_interest_rates, fetch_yield_spread

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def compute_regime_score(start: str, end: str) -> pd.DataFrame:
    """
    Computes a dynamic regime score (0–100) from verified FRED macro data.
    Score reflects market state: 0 = max fear, 100 = max greed.
    """

    logger.info(f"📈 Calculating regime score from {start} to {end}")

    # Fetch verified inputs
    vix = fetch_vix(start, end)
    rates = fetch_interest_rates(start, end)
    spread = fetch_yield_spread(start, end)

    # Merge all macro inputs
    df = pd.concat([vix, rates, spread], axis=1).dropna()
    df.columns = ["vix", "rates", "spread"]

    # Normalize all indicators (0–1) with MinMaxScaler
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(scaled_values, index=df.index, columns=df.columns)

    # Compute regime score: greed = low VIX + high spread + low rates
    # Weighted sum: adjust as needed
    df_scaled["score_raw"] = (
        (1 - df_scaled["vix"]) * 0.5 +
        df_scaled["spread"] * 0.3 +
        (1 - df_scaled["rates"]) * 0.2
    )

    # Final scaling to 0–100
    df_scaled["regime_score"] = (df_scaled["score_raw"] * 100).clip(0, 100)
    df_scaled.drop(columns=["score_raw"], inplace=True)

    logger.info("✅ Regime score calculation complete.")
    return df_scaled[["regime_score"]]
