import pandas as pd
import os
import json
from datetime import datetime

TRAINING_LOG_PATH = "data/model_training_log.json"
ACCURACY_TABLE_PATH = "data/model_accuracy_table.json"

def get_training_performance_summary() -> dict:
    """Return overall training summary and model accuracy table"""
    summary = {
        "avg_accuracy": 0.0,
        "active_models": 0,
        "last_trained": "NA",
        "accuracy_table": pd.DataFrame()
    }

    if not os.path.exists(ACCURACY_TABLE_PATH):
        return summary

    try:
        with open(ACCURACY_TABLE_PATH, "r") as f:
            accuracy_data = json.load(f)

        df = pd.DataFrame(accuracy_data)
        summary["accuracy_table"] = df
        summary["avg_accuracy"] = round(df["Accuracy (%)"].mean(), 2)
        summary["active_models"] = df["Model"].nunique()

    except Exception as e:
        print(f"[performance] Error loading accuracy data: {e}")

    if os.path.exists(TRAINING_LOG_PATH):
        try:
            with open(TRAINING_LOG_PATH, "r") as f:
                logs = json.load(f)
            timestamps = [entry["timestamp"] for entry in logs]
            if timestamps:
                latest = max([datetime.fromisoformat(t) for t in timestamps])
                summary["last_trained"] = latest.strftime("%Y-%m-%d %H:%M")
        except Exception as e:
            print(f"[performance] Error loading training log: {e}")

    return summary

def get_model_variance_report(threshold: float = 15.0) -> pd.DataFrame:
    """Detects underperforming regime zones for further training focus"""
    if not os.path.exists(ACCURACY_TABLE_PATH):
        return pd.DataFrame()

    try:
        with open(ACCURACY_TABLE_PATH, "r") as f:
            accuracy_data = json.load(f)
        df = pd.DataFrame(accuracy_data)

        # Compute std dev of each model by regime
        grouped = df.groupby(["Model", "Regime"])["Accuracy (%)"].mean().reset_index()
        pivot = grouped.pivot(index="Model", columns="Regime", values="Accuracy (%)").fillna(0)

        # Flag regime zones with significant underperformance
        std_dev = pivot.std(axis=1)
        underperforming = pivot.loc[std_dev > threshold]

        return underperforming.reset_index()

    except Exception as e:
        print(f"[performance] Error generating variance report: {e}")
        return pd.DataFrame()
