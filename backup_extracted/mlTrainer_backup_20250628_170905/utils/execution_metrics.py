import pandas as pd
from utils.data_loader import get_live_recommendations, get_current_holdings

def compute_live_metrics() -> dict:
    """Compute real-time dashboard metrics for recommendations and holdings"""
    metrics = {
        "active_recommendations": 0,
        "current_holdings": 0,
        "win_rate": 0.0,
        "avg_score": 0.0
    }

    try:
        recos = get_live_recommendations()
        holdings = get_current_holdings()

        if not recos.empty:
            metrics["active_recommendations"] = len(recos)
            metrics["avg_score"] = round(recos["Score"].mean(), 2)

        if not holdings.empty:
            metrics["current_holdings"] = len(holdings)

            # Calculate win rate from performance %
            win_trades = holdings[holdings["Progress %"] >= 100]
            if len(holdings) > 0:
                metrics["win_rate"] = round(100 * len(win_trades) / len(holdings), 2)

    except Exception as e:
        print(f"[execution_metrics] Error computing metrics: {e}")

    return metrics
