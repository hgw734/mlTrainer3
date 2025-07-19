import os
import json
import logging
from typing import List, Dict
from ml.walkforward_trainer import train_walkforward_model

TICKER_LIST = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL"]  # Can be loaded from DB or S&P100 file
RESULTS_DIR = "ml/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def train_all_tickers(config: Dict = None) -> List[Dict]:
    results = []
    for ticker in TICKER_LIST:
        logger.info(f"🚀 Training model for {ticker}")
        result = train_walkforward_model(ticker, config)
        if result:
            results.append(result)
            _save_result(result)
    return results

def _save_result(result: Dict):
    try:
        path = os.path.join(RESULTS_DIR, f"{result['ticker']}_metrics.json")
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"✅ Saved metrics for {result['ticker']}")
    except Exception as e:
        logger.error(f"❌ Failed to save result for {result['ticker']}: {e}")
