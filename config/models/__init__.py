"""
Model Configuration Module
Split from monolithic models_config.py for better organization
"""

from .ml_models import ML_MODELS_CONFIG
from .deep_learning_models import DEEP_LEARNING_CONFIG
from .financial_models import FINANCIAL_MODELS_CONFIG
from .timeseries_models import TIMESERIES_CONFIG
from .ensemble_models import ENSEMBLE_CONFIG
from .experimental_models import EXPERIMENTAL_CONFIG

# Aggregate all model configurations
ALL_MODELS_CONFIG = {
**ML_MODELS_CONFIG,
**DEEP_LEARNING_CONFIG,
**FINANCIAL_MODELS_CONFIG,
**TIMESERIES_CONFIG,
**ENSEMBLE_CONFIG,
**EXPERIMENTAL_CONFIG,
}

# Model categories for easy access
MODEL_CATEGORIES = {
"machine_learning": list(ML_MODELS_CONFIG.keys()),
"deep_learning": list(DEEP_LEARNING_CONFIG.keys()),
"financial": list(FINANCIAL_MODELS_CONFIG.keys()),
"timeseries": list(TIMESERIES_CONFIG.keys()),
"ensemble": list(ENSEMBLE_CONFIG.keys()),
"experimental": list(EXPERIMENTAL_CONFIG.keys()),
}


def get_model_config(model_name: str) -> dict:
    """Get configuration for a specific model"""
    return ALL_MODELS_CONFIG.get(model_name, {})


    def get_models_by_category(category: str) -> list:
        """Get all models in a specific category"""
        return MODEL_CATEGORIES.get(category, [])
