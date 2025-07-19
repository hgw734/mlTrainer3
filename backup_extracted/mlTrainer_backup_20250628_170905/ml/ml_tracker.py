import os
import json
import logging
from typing import List, Dict

RESULTS_DIR = "ml/results"
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_all_model_metrics() -> List[Dict]:
    results = []
    for file in os.listdir(RESULTS_DIR):
        if file.endswith("_metrics.json"):
            try:
                with open(os.path.join(RESULTS_DIR, file), "r") as f:
                    results.append(json.load(f))
            except Exception as e:
                logger.error(f"❌ Failed to load {file}: {e}")
    return results

def print_summary_table():
    results = load_all_model_metrics()
    print(f"{'Ticker':<8} {'Model':<10} {'MAE':<10} {'Dir.Acc':<10}")
    print("-" * 40)
    for r in results:
        print(f"{r['ticker']:<8} {r['model_type']:<10} {r['mae']:<10.5f} {r['directional_accuracy']:<10.2%}")
