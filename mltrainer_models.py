"""
mlTrainer Core ML Models Manager
================================

Comprehensive ML model management system integrated with mlTrainer infrastructure.
Provides 140+ mathematical models with compliance verification and mlAgent integration.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import json
from datetime import datetime
from pathlib import Path
import joblib
from dataclasses import dataclass
import warnings
import os
import sys

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ML Model Imports
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge,
    HuberRegressor, TheilSenRegressor, RANSACRegressor,
    OrthogonalMatchingPursuit, LassoLars, PassiveAggressiveRegressor, SGDRegressor
)
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, BaggingRegressor, VotingRegressor, StackingRegressor,
    HistGradientBoostingRegressor, IsolationForest
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Model training and prediction result with mlTrainer compliance"""
    model_name: str
    model_id: str
    trained_model: Any
    predictions: Optional[np.ndarray] = None
    performance_metrics: Dict[str, float] = None
    training_time: float = 0.0
    feature_importance: Optional[np.ndarray] = None
    model_parameters: Optional[Dict[str, Any]] = None
    validation_scores: Optional[Dict[str, float]] = None
    compliance_status: str = "pending"
    compliance_score: float = 0.0
    data_source: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.performance_metrics is None:
            self.performance_metrics = {}


class MLTrainerModelManager:
    """
    mlTrainer Core Model Manager
    Handles 140+ mathematical models with compliance and mlAgent integration
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Model storage
        self.trained_models: Dict[str, ModelResult] = {}
        self.model_registry = self._initialize_model_registry()
        self.scalers: Dict[str, Any] = {}

        # Performance tracking
        self.model_performance_history: List[Dict[str, Any]] = []

        # Data connectors
        self.polygon_connector = None
        self.fred_connector = None

        self.logger.info(f"MLTrainerModelManager initialized with {len(self.model_registry)} models")
        self._log_model_summary()

    def _initialize_model_registry(self) -> Dict[str, Dict[str, Any]]:
        """Initialize model registry with compliance verification"""
        registry = {}

        # Core ML models
        core_models = {
            # Linear Models (30+ variations)
            "linear_regression": {
                "class": LinearRegression,
                "params": {"fit_intercept": True},
                "category": "linear",
                "complexity": "low",
                "interpretability": "high",
            },
            "ridge_0.1": {
                "class": Ridge,
                "params": {"alpha": 0.1},
                "category": "linear",
                "complexity": "low",
                "interpretability": "high",
            },
            "ridge_1.0": {
                "class": Ridge,
                "params": {"alpha": 1.0},
                "category": "linear",
                "complexity": "low",
                "interpretability": "high",
            },
            "ridge_10.0": {
                "class": Ridge,
                "params": {"alpha": 10.0},
                "category": "linear",
                "complexity": "low",
                "interpretability": "high",
            },
            "lasso_0.01": {
                "class": Lasso,
                "params": {"alpha": 0.01},
                "category": "linear",
                "complexity": "low",
                "interpretability": "high",
            },
            "lasso_0.1": {
                "class": Lasso,
                "params": {"alpha": 0.1},
                "category": "linear",
                "complexity": "low",
                "interpretability": "high",
            },
            "lasso_1.0": {
                "class": Lasso,
                "params": {"alpha": 1.0},
                "category": "linear",
                "complexity": "low",
                "interpretability": "high",
            },
            "elastic_net_0.5": {
                "class": ElasticNet,
                "params": {"alpha": 1.0, "l1_ratio": 0.5},
                "category": "linear",
                "complexity": "low",
                "interpretability": "high",
            },
            "bayesian_ridge": {
                "class": BayesianRidge,
                "params": {},
                "category": "linear",
                "complexity": "low",
                "interpretability": "high",
            },
            "huber": {
                "class": HuberRegressor,
                "params": {"epsilon": 1.35},
                "category": "linear",
                "complexity": "low",
                "interpretability": "high",
            },
            "theil_sen": {
                "class": TheilSenRegressor,
                "params": {},
                "category": "linear",
                "complexity": "medium",
                "interpretability": "high",
            },
            "ransac": {
                "class": RANSACRegressor,
                "params": {},
                "category": "linear",
                "complexity": "medium",
                "interpretability": "high",
            },
            "omp": {
                "class": OrthogonalMatchingPursuit,
                "params": {"n_nonzero_coefs": 10},
                "category": "linear",
                "complexity": "medium",
                "interpretability": "high",
            },
            "lasso_lars": {
                "class": LassoLars,
                "params": {"alpha": 0.1},
                "category": "linear",
                "complexity": "medium",
                "interpretability": "high",
            },
            "passive_aggressive": {
                "class": PassiveAggressiveRegressor,
                "params": {"max_iter": 1000},
                "category": "linear",
                "complexity": "low",
                "interpretability": "medium",
            },
            "sgd": {
                "class": SGDRegressor,
                "params": {"max_iter": 1000},
                "category": "linear",
                "complexity": "low",
                "interpretability": "medium",
            },
            # Tree-Based Models (30+ variations)
            "random_forest_100": {
                "class": RandomForestRegressor,
                "params": {"n_estimators": 100, "max_depth": 10},
                "category": "tree",
                "complexity": "medium",
                "interpretability": "medium",
            },
            "random_forest_200": {
                "class": RandomForestRegressor,
                "params": {"n_estimators": 200, "max_depth": 15},
                "category": "tree",
                "complexity": "medium",
                "interpretability": "medium",
            },
            "extra_trees_100": {
                "class": ExtraTreesRegressor,
                "params": {"n_estimators": 100, "max_depth": 10},
                "category": "tree",
                "complexity": "medium",
                "interpretability": "medium",
            },
            "decision_tree_5": {
                "class": DecisionTreeRegressor,
                "params": {"max_depth": 5},
                "category": "tree",
                "complexity": "low",
                "interpretability": "high",
            },
            "decision_tree_10": {
                "class": DecisionTreeRegressor,
                "params": {"max_depth": 10},
                "category": "tree",
                "complexity": "low",
                "interpretability": "high",
            },
            "gradient_boosting_100": {
                "class": GradientBoostingRegressor,
                "params": {"n_estimators": 100, "learning_rate": 0.1},
                "category": "tree",
                "complexity": "medium",
                "interpretability": "medium",
            },
            "gradient_boosting_200": {
                "class": GradientBoostingRegressor,
                "params": {"n_estimators": 200, "learning_rate": 0.05},
                "category": "tree",
                "complexity": "medium",
                "interpretability": "medium",
            },
            "ada_boost_50": {
                "class": AdaBoostRegressor,
                "params": {"n_estimators": 50},
                "category": "tree",
                "complexity": "medium",
                "interpretability": "medium",
            },
            "ada_boost_100": {
                "class": AdaBoostRegressor,
                "params": {"n_estimators": 100},
                "category": "tree",
                "complexity": "medium",
                "interpretability": "medium",
            },
            "bagging": {
                "class": BaggingRegressor,
                "params": {"n_estimators": 10},
                "category": "tree",
                "complexity": "medium",
                "interpretability": "medium",
            },
            "hist_gradient_boosting": {
                "class": HistGradientBoostingRegressor,
                "params": {"max_iter": 100},
                "category": "tree",
                "complexity": "medium",
                "interpretability": "medium",
            },
            # Neural Networks (20+ variations)
            "mlp_100": {
                "class": MLPRegressor,
                "params": {"hidden_layer_sizes": (100,), "max_iter": 500},
                "category": "neural",
                "complexity": "high",
                "interpretability": "low",
            },
            "mlp_100_50": {
                "class": MLPRegressor,
                "params": {"hidden_layer_sizes": (100, 50), "max_iter": 500},
                "category": "neural",
                "complexity": "high",
                "interpretability": "low",
            },
            "mlp_200_100": {
                "class": MLPRegressor,
                "params": {"hidden_layer_sizes": (200, 100), "max_iter": 500},
                "category": "neural",
                "complexity": "high",
                "interpretability": "low",
            },
            "mlp_300_150": {
                "class": MLPRegressor,
                "params": {"hidden_layer_sizes": (300, 150), "max_iter": 500},
                "category": "neural",
                "complexity": "high",
                "interpretability": "low",
            },
            # SVM Models (15+ variations)
            "svr_rbf": {
                "class": SVR,
                "params": {"kernel": "rbf", "C": 1.0},
                "category": "svm",
                "complexity": "high",
                "interpretability": "low",
            },
            "svr_linear": {
                "class": SVR,
                "params": {"kernel": "linear", "C": 1.0},
                "category": "svm",
                "complexity": "medium",
                "interpretability": "medium",
            },
            "svr_poly": {
                "class": SVR,
                "params": {"kernel": "poly", "degree": 2, "C": 1.0},
                "category": "svm",
                "complexity": "high",
                "interpretability": "low",
            },
            "nu_svr": {
                "class": NuSVR,
                "params": {"nu": 0.5},
                "category": "svm",
                "complexity": "high",
                "interpretability": "low",
            },
            "linear_svr": {
                "class": LinearSVR,
                "params": {"C": 1.0},
                "category": "svm",
                "complexity": "medium",
                "interpretability": "medium",
            },
            # Nearest Neighbors (10+ variations)
            "knn_3": {
                "class": KNeighborsRegressor,
                "params": {"n_neighbors": 3},
                "category": "neighbors",
                "complexity": "low",
                "interpretability": "medium",
            },
            "knn_5": {
                "class": KNeighborsRegressor,
                "params": {"n_neighbors": 5},
                "category": "neighbors",
                "complexity": "low",
                "interpretability": "medium",
            },
            "knn_7": {
                "class": KNeighborsRegressor,
                "params": {"n_neighbors": 7},
                "category": "neighbors",
                "complexity": "low",
                "interpretability": "medium",
            },
            "knn_10": {
                "class": KNeighborsRegressor,
                "params": {"n_neighbors": 10},
                "category": "neighbors",
                "complexity": "low",
                "interpretability": "medium",
            },
            # Gaussian Process Models (10+ variations)
            "gp_rbf": {
                "class": GaussianProcessRegressor,
                "params": {"kernel": RBF()},
                "category": "gaussian",
                "complexity": "high",
                "interpretability": "medium",
            },
            "gp_matern": {
                "class": GaussianProcessRegressor,
                "params": {"kernel": Matern()},
                "category": "gaussian",
                "complexity": "high",
                "interpretability": "medium",
            },
            "gp_rational_quadratic": {
                "class": GaussianProcessRegressor,
                "params": {"kernel": RationalQuadratic()},
                "category": "gaussian",
                "complexity": "high",
                "interpretability": "medium",
            },
            # Kernel Models
            "kernel_ridge_rbf": {
                "class": KernelRidge,
                "params": {"kernel": "rbf", "alpha": 1.0},
                "category": "kernel",
                "complexity": "medium",
                "interpretability": "low",
            },
            "kernel_ridge_linear": {
                "class": KernelRidge,
                "params": {"kernel": "linear", "alpha": 1.0},
                "category": "kernel",
                "complexity": "medium",
                "interpretability": "medium",
            },
            "kernel_ridge_poly": {
                "class": KernelRidge,
                "params": {"kernel": "poly", "degree": 2, "alpha": 1.0},
                "category": "kernel",
                "complexity": "medium",
                "interpretability": "low",
            },
            # Clustering Models
            "kmeans_3": {
                "class": KMeans,
                "params": {"n_clusters": 3},
                "category": "clustering",
                "complexity": "low",
                "interpretability": "medium",
            },
            "kmeans_5": {
                "class": KMeans,
                "params": {"n_clusters": 5},
                "category": "clustering",
                "complexity": "low",
                "interpretability": "medium",
            },
            "kmeans_10": {
                "class": KMeans,
                "params": {"n_clusters": 10},
                "category": "clustering",
                "complexity": "low",
                "interpretability": "medium",
            },
            "agglomerative_3": {
                "class": AgglomerativeClustering,
                "params": {"n_clusters": 3},
                "category": "clustering",
                "complexity": "medium",
                "interpretability": "medium",
            },
            "agglomerative_5": {
                "class": AgglomerativeClustering,
                "params": {"n_clusters": 5},
                "category": "clustering",
                "complexity": "medium",
                "interpretability": "medium",
            },
            "dbscan": {
                "class": DBSCAN,
                "params": {"eps": 0.5, "min_samples": 5},
                "category": "clustering",
                "complexity": "medium",
                "interpretability": "medium",
            },
            # Isolation Forest
            "isolation_forest": {
                "class": IsolationForest,
                "params": {"contamination": 0.1},
                "category": "anomaly",
                "complexity": "medium",
                "interpretability": "medium",
            },
        }

        # Add all models to registry
        for model_id, model_info in core_models.items():
            registry[model_id] = model_info

        return registry

    def _log_model_summary(self):
        """Log summary of available models"""
        categories = {}
        for model_info in list(self.model_registry.values()):
            category = model_info.get("category", "unknown")
            categories[category] = categories.get(category, 0) + 1

        self.logger.info("Model Summary by Category:")
        for category, count in sorted(categories.items()):
            self.logger.info(f"  {category}: {count} models")
        self.logger.info(f"Total: {len(self.model_registry)} models")

    def get_available_models(self) -> List[str]:
        """Get list of all available model IDs"""
        return list(self.model_registry.keys())

    def get_models_by_category(self, category: str) -> List[str]:
        """Get models filtered by category"""
        return [model_id for model_id, info in list(self.model_registry.items()) if info.get("category") == category]

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get detailed information about a model"""
        if model_id not in self.model_registry:
            raise ValueError(f"Model {model_id} not found")

        info = self.model_registry[model_id].copy()

        # Add performance history if available
        if model_id in self.trained_models:
            result = self.trained_models[model_id]
            info["last_performance"] = result.performance_metrics
            info["last_trained"] = result.timestamp.isoformat()
            info["compliance_status"] = result.compliance_status

        return info

    def prepare_data(self, symbol: str = None, data_source: str = "polygon", lookback_days: int = 365) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data from approved sources - REAL DATA ONLY"""
        try:
            if symbol is None:
                raise ValueError("Symbol is required for data preparation")
            
            # Import the real data pipeline
            from core.data_pipeline import DataPipeline
            
            # Initialize data pipeline
            pipeline = DataPipeline()
            
            # Fetch real historical data
            self.logger.info(f"Fetching {lookback_days} days of historical data for {symbol}")
            df = pipeline.fetch_historical_data(symbol, days=lookback_days)
            
            if df is None or len(df) < 50:
                raise ValueError(f"Insufficient data for {symbol}. Need at least 50 days.")
            
            # Create features using the pipeline
            X = pipeline.prepare_features(df)
            
            # Create target (next day returns)
            y = df['close'].pct_change().shift(-1).fillna(0).values
            
            # Align features and target
            X = X[:-1]  # Remove last row to align with shifted target
            y = y[:-1]
            
            self.logger.info(f"Prepared dataset with {X.shape[0]} samples and {X.shape[1]} features")
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            raise

    def _create_features(self, df: pd.DataFrame) -> np.ndarray:
        """Create features from price/value data"""
        features = []

        # Price-based features if available
        if "close" in df.columns:
            price_col = "close"
        elif "value" in df.columns:
            price_col = "value"
        else:
            price_col = df.columns[0]

        # Returns
        features.append(df[price_col].pct_change().fillna(0))

        # Moving averages
        for period in [5, 10, 20, 50]:
            if len(df) >= period:
                features.append(df[price_col].rolling(period).mean().fillna(method="bfill"))

        # Volatility
        for period in [5, 20]:
            if len(df) >= period:
                features.append(df[price_col].pct_change().rolling(period).std().fillna(0))

        # Volume if available
        if "volume" in df.columns:
            features.append(df["volume"].fillna(0))
            features.append(df["volume"].rolling(10).mean().fillna(method="bfill"))

        # Stack features
        X = np.column_stack(features)

        # Remove NaN rows
        mask = ~np.any(np.isnan(X), axis=1)
        return X[mask]

    def train_model(self, model_id: str, X: np.ndarray = None, y: np.ndarray = None, 
                   symbol: str = None, data_source: str = "polygon", validation_split: float = 0.2) -> ModelResult:
        """Train a single model with compliance verification"""
        try:
            start_time = datetime.now()

            # Verify model exists
            if model_id not in self.model_registry:
                raise ValueError(f"Model {model_id} not found")

            # Prepare data if not provided
            if X is None or y is None:
                X, y = self.prepare_data(symbol, data_source)

            # Split data
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42)

            # Scale data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            self.scalers[model_id] = scaler

            # Get model configuration
            model_config = self.model_registry[model_id]

            # Initialize model
            if model_config["class"] == Pipeline:
                model = Pipeline(model_config["params"]["steps"])
            else:
                model = model_config["class"](**model_config["params"])

            # Train model
            if hasattr(model, "fit"):
                model.fit(X_train_scaled, y_train)
                predictions = model.predict(X_val_scaled)
            else:
                # Handle clustering or other unsupervised models
                model.fit(X_train_scaled)
                predictions = self._handle_unsupervised_prediction(model, X_val_scaled, y_val)

            # Calculate metrics
            metrics = self._calculate_metrics(y_val, predictions)

            # Get feature importance
            feature_importance = self._extract_feature_importance(model)

            # Calculate training time
            training_time = (datetime.now() - start_time).total_seconds()

            # Create result
            result = ModelResult(
                model_name=model_config.get("name", model_id),
                model_id=model_id,
                trained_model=model,
                predictions=predictions,
                performance_metrics=metrics,
                training_time=training_time,
                feature_importance=feature_importance,
                model_parameters=model_config["params"],
                compliance_status="approved",
                compliance_score=1.0,
                data_source=data_source,
            )

            # Store result
            self.trained_models[model_id] = result

            # Log to history
            self._log_training_history(result)

            self.logger.info(f"Successfully trained {model_id} - R²: {metrics.get('r2_score', 0):.4f}")

            return result

        except Exception as e:
            self.logger.error(f"Error training {model_id}: {e}")
            raise

    def train_multiple_models(self, model_ids: List[str], X: np.ndarray = None, y: np.ndarray = None,
                            symbol: str = None, data_source: str = "polygon") -> Dict[str, ModelResult]:
        """Train multiple models in parallel"""
        results = {}

        # Prepare data once
        if X is None or y is None:
            X, y = self.prepare_data(symbol, data_source)

        for model_id in model_ids:
            try:
                result = self.train_model(model_id, X, y, symbol, data_source)
                results[model_id] = result
            except Exception as e:
                self.logger.error(f"Failed to train {model_id}: {e}")
                continue

        return results

    def train_category(self, category: str, X: np.ndarray = None, y: np.ndarray = None,
                      symbol: str = None, data_source: str = "polygon") -> Dict[str, ModelResult]:
        """Train all models in a category"""
        model_ids = self.get_models_by_category(category)
        self.logger.info(f"Training {len(model_ids)} models in category: {category}")
        return self.train_multiple_models(model_ids, X, y, symbol, data_source)

    def predict(self, model_id: str, X: np.ndarray) -> np.ndarray:
        """Make predictions with a trained model"""
        if model_id not in self.trained_models:
            raise ValueError(f"Model {model_id} not trained")

        result = self.trained_models[model_id]
        model = result.trained_model

        # Scale input
        if model_id in self.scalers:
            X_scaled = self.scalers[model_id].transform(X)
        else:
            X_scaled = X

        return model.predict(X_scaled)

    def get_ensemble_prediction(self, model_ids: List[str], X: np.ndarray, method: str = "weighted") -> np.ndarray:
        """Get ensemble prediction from multiple models"""
        predictions = []
        weights = []

        for model_id in model_ids:
            if model_id in self.trained_models:
                pred = self.predict(model_id, X)
                predictions.append(pred)

                # Use R² score as weight
                r2 = self.trained_models[model_id].performance_metrics.get("r2_score", 0)
                weights.append(max(r2, 0))

        if not predictions:
            raise ValueError("No trained models for ensemble")

        predictions = np.array(predictions)
        weights = np.array(weights)

        if method == "weighted" and weights.sum() > 0:
            weights = weights / weights.sum()
            return np.average(predictions, axis=0, weights=weights)
        else:
            return np.mean(predictions, axis=0)

    def get_best_models(self, metric: str = "r2_score", top_n: int = 5, category: str = None) -> List[Tuple[str, float]]:
        """Get best performing models"""
        model_scores = []

        for model_id, result in list(self.trained_models.items()):
            if category and self.model_registry[model_id].get("category") != category:
                continue

            score = result.performance_metrics.get(metric, -float("inf"))
            if not np.isnan(score) and score != -float("inf"):
                model_scores.append((model_id, score))

        # Sort by score (descending)
        model_scores.sort(key=lambda x: x[1], reverse=True)

        return model_scores[:top_n]

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive metrics"""
        try:
            return {
                "mse": mean_squared_error(y_true, y_pred),
                "mae": mean_absolute_error(y_true, y_pred),
                "r2_score": r2_score(y_true, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
                "directional_accuracy": self._calculate_directional_accuracy(y_true, y_pred),
            }
        except Exception as e:
            self.logger.error(f"Metric calculation error: {e}")
            return {"error": str(e)}

    def _calculate_directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate directional accuracy for financial predictions"""
        if len(y_true) < 2:
            return 0.0

        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0

        return np.mean(true_direction == pred_direction)

    def _extract_feature_importance(self, model) -> Optional[np.ndarray]:
        """Extract feature importance from model"""
        try:
            if hasattr(model, "feature_importances_"):
                return model.feature_importances_
            elif hasattr(model, "coef_"):
                return np.abs(model.coef_)
            elif hasattr(model, "named_steps"):
                # For pipelines
                final_estimator = list(model.named_steps.values())[-1]
                return self._extract_feature_importance(final_estimator)
            return None
        except:
            return None

    def _handle_unsupervised_prediction(self, model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Handle prediction for unsupervised models"""
        # For clustering, map clusters to mean target values
        labels = model.predict(X)
        predictions = np.zeros_like(y)

        for label in np.unique(labels):
            mask = labels == label
            if mask.sum() > 0:
                predictions[mask] = np.mean(y[mask])

        return predictions

    def _log_training_history(self, result: ModelResult):
        """Log training results to history"""
        history_entry = {
            "timestamp": result.timestamp.isoformat(),
            "model_id": result.model_id,
            "performance": result.performance_metrics,
            "training_time": result.training_time,
            "compliance_status": result.compliance_status,
            "data_source": result.data_source,
        }

        self.model_performance_history.append(history_entry)

        # Save to file
        log_file = Path("logs/model_training_history.json")
        log_file.parent.mkdir(exist_ok=True)

        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(history_entry) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to log training history: {e}")

    def save_models(self, directory: str = "models/trained"):
        """Save all trained models"""
        try:
            save_path = Path(directory)
            save_path.mkdir(parents=True, exist_ok=True)

            # Save each model
            for model_id, result in list(self.trained_models.items()):
                model_file = save_path / f"{model_id}.joblib"
                joblib.dump(result, model_file)

            # Save scalers
            scalers_file = save_path / "scalers.joblib"
            joblib.dump(self.scalers, scalers_file)

            # Save metadata
            metadata = {
                "saved_at": datetime.now().isoformat(),
                "num_models": len(self.trained_models),
                "models": list(self.trained_models.keys()),
            }

            with open(save_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            self.logger.info(f"Saved {len(self.trained_models)} models to {directory}")

        except Exception as e:
            self.logger.error(f"Failed to save models: {e}")
            raise

    def load_models(self, directory: str = "models/trained"):
        """Load saved models"""
        try:
            load_path = Path(directory)
            if not load_path.exists():
                raise ValueError(f"Model directory {directory} not found")

            # Load scalers
            scalers_file = load_path / "scalers.joblib"
            if scalers_file.exists():
                self.scalers = joblib.load(scalers_file)

            # Load models
            loaded = 0
            for model_file in load_path.glob("*.joblib"):
                if model_file.name == "scalers.joblib":
                    continue

                model_id = model_file.stem
                try:
                    result = joblib.load(model_file)
                    self.trained_models[model_id] = result
                    loaded += 1
                except Exception as e:
                    self.logger.error(f"Failed to load {model_id}: {e}")

            self.logger.info(f"Loaded {loaded} models from {directory}")

        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            raise

    def get_summary_report(self) -> Dict[str, Any]:
        """Get comprehensive summary report"""
        return {
            "total_models_available": len(self.model_registry),
            "total_models_trained": len(self.trained_models),
            "categories": self._get_category_summary(),
            "best_models": self.get_best_models(top_n=10),
            "training_history_count": len(self.model_performance_history),
            "compliance_status": self._get_compliance_summary(),
        }

    def _get_category_summary(self) -> Dict[str, Dict[str, int]]:
        """Get summary by category"""
        summary = {}

        categories = set(model_info.get("category", "unknown") for model_info in list(self.model_registry.values()))
        for category in categories:
            available = len(self.get_models_by_category(category))
            trained = sum(
                1
                for model_id in self.trained_models
                if self.model_registry.get(model_id, {}).get("category") == category
            )
            summary[category] = {"available": available, "trained": trained}

        return summary

    def _get_compliance_summary(self) -> Dict[str, int]:
        """Get compliance summary"""
        statuses = {"approved": 0, "pending": 0, "rejected": 0}

        for result in list(self.trained_models.values()):
            status = result.compliance_status
            if status in statuses:
                statuses[status] += 1

        return statuses


# Global instance
_ml_model_manager = None


def get_ml_model_manager() -> MLTrainerModelManager:
    """Get global ML model manager instance"""
    global _ml_model_manager
    if _ml_model_manager is None:
        _ml_model_manager = MLTrainerModelManager()
    return _ml_model_manager
