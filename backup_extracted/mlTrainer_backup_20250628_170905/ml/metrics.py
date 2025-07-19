from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def evaluate_model(model, X_val, y_val, optimization_target: str = "rmse") -> dict:
    """
    Evaluate the model using the specified optimization target and return metrics.
    Supports regression and classification models.
    """

    try:
        predictions = model.predict(X_val)
    except Exception as e:
        logger.error(f"❌ Model prediction failed: {e}")
        return {"score": -1, "metrics": {}}

    # If predictions are 2D (e.g. Keras), flatten
    if predictions.ndim > 1:
        predictions = predictions.flatten()

    is_classification = optimization_target.lower() in ["f1", "accuracy", "precision", "recall"]

    metrics = {}

    if is_classification:
        preds_bin = (predictions >= 0.5).astype(int)
        y_val_bin = (y_val >= 0.5).astype(int)

        metrics["accuracy"] = accuracy_score(y_val_bin, preds_bin)
        metrics["precision"] = precision_score(y_val_bin, preds_bin, zero_division=0)
        metrics["recall"] = recall_score(y_val_bin, preds_bin, zero_division=0)
        metrics["f1_score"] = f1_score(y_val_bin, preds_bin, zero_division=0)
        score = metrics.get(optimization_target.lower(), 0.0)
    else:
        metrics["rmse"] = np.sqrt(mean_squared_error(y_val, predictions))
        metrics["mae"] = mean_absolute_error(y_val, predictions)
        metrics["r2"] = r2_score(y_val, predictions)
        score = -metrics.get(optimization_target.lower(), metrics["rmse"])  # Lower is better for loss

    logger.info(f"📊 Evaluation ({optimization_target}): {score:.4f}")
    return {
        "score": round(score, 4),
        "metrics": {k: round(v, 4) for k, v in metrics.items()}
    }
