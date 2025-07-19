import logging
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Dict

from strategy.regime_classifier import compute_regime_score
from strategy.strategy_router import select_strategy
from mlTrainer import MLTrainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def generate_date_ranges(start: str, end: str, training_months: int, validation_days: int) -> List[Dict[str, str]]:
    """
    Generate rolling walk-forward date windows.
    """
    ranges = []
    current_start = pd.to_datetime(start)

    while True:
        train_end = current_start + pd.DateOffset(months=training_months)
        val_end = train_end + pd.DateOffset(days=validation_days)

        if val_end > pd.to_datetime(end):
            break

        ranges.append({
            "train_start": current_start.strftime('%Y-%m-%d'),
            "train_end": train_end.strftime('%Y-%m-%d'),
            "val_end": val_end.strftime('%Y-%m-%d')
        })

        current_start += pd.DateOffset(weeks=4)

    return ranges

def run_walk_forward_trials(symbol: str, start_date: str, end_date: str):
    """
    Runs rolling walk-forward training trials with regime-aware model selection.
    """
    logger.info(f"🚀 Starting walk-forward regime-aware training for {symbol} ({start_date} to {end_date})")

    date_ranges = generate_date_ranges(start_date, end_date, training_months=6, validation_days=30)
    trainer = MLTrainer()

    results = []

    for window in date_ranges:
        regime_df = compute_regime_score(window["train_start"], window["train_end"])
        avg_regime_score = regime_df["regime_score"].mean()
        strategy = select_strategy(avg_regime_score)

        config = {
            "symbol": symbol,
            "start_date": window["train_start"],
            "end_date": window["train_end"],
            "model": strategy["model"],
            "optimization_target": strategy["optimization_target"]
        }

        logger.info(f"📦 Running trial: {config}")
        result = trainer.run_trial(config)
        result["regime_score"] = round(avg_regime_score, 2)
        result["train_window"] = f"{window['train_start']} to {window['train_end']}"
        results.append(result)

    logger.info("✅ Walk-forward trials complete.")
    return results
