import os
import logging
import joblib
import pandas as pd

MERGED_DIR = "features/merged"
MODEL_DIR = "models"
RECOMMENDATIONS_FILE = "signals/daily_recommendations.csv"
os.makedirs("signals", exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def generate_daily_recommendations():
    recommendations = []

    for file in os.listdir(MERGED_DIR):
        if not file.endswith("_matrix.csv"):
            continue

        ticker = file.split("_")[0]
        data_path = os.path.join(MERGED_DIR, file)
        model_path = os.path.join(MODEL_DIR, f"{ticker}_xgboost_model.pkl")

        if not os.path.exists(model_path):
            continue

        try:
            df = pd.read_csv(data_path)
            df = df.sort_values("date").tail(1)  # Get latest row
            model = joblib.load(model_path)

            feature_cols = [c for c in df.columns if c not in ["date", "open", "high", "low", "close", "ticker", "symbol"]]
            score = model.predict(df[feature_cols])[0]

            recommendations.append((ticker, round(score * 100, 2)))

        except Exception as e:
            logger.error(f"❌ Failed to score {ticker}: {e}")

    df_out = pd.DataFrame(recommendations, columns=["Ticker", "Expected_Return_%"])
    df_out = df_out.sort_values("Expected_Return_%", ascending=False)
    df_out.to_csv(RECOMMENDATIONS_FILE, index=False)
    logger.info(f"📊 Saved {len(df_out)} recommendations to {RECOMMENDATIONS_FILE}")
