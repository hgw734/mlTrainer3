"""
mlTrainer - Model Technical Registry
===================================

Purpose: Technical registry of available ML models and their implementation characteristics.
Provides mlTrainer with factual information about model capabilities and requirements.

This system provides mlTrainer with:
- Available models and their technical specifications
- Implementation requirements (data, compute, time)
- Model strengths and limitations
- Technical integration capabilities

All strategy decisions, combinations, and applications are made by mlTrainer.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

class ModelIntelligence:
    """Complete intelligence system for all models - uses centralized registry as source of truth"""
    
    def __init__(self):
        # Get centralized model registry - SINGLE SOURCE OF TRUTH
        from core.model_registry import get_model_registry
        self.model_registry = get_model_registry()
        
        # Build intelligence on top of centralized registry
        self.model_knowledge = self._build_intelligence_from_registry()
        self.combination_strategies = self._initialize_combination_strategies()
        self.market_condition_mappings = self._initialize_market_mappings()
        self.ensemble_recipes = self._initialize_ensemble_recipes()
        
        logger.info("ModelIntelligence initialized with complete model knowledge")
    
    def _build_intelligence_from_registry(self) -> Dict[str, Dict[str, Any]]:
        """Build intelligence layer on top of centralized model registry"""
        intelligence = {}
        
        # Get all models from centralized registry
        for model_name in self.model_registry.get_all_models():
            model_info = self.model_registry.get_model_info(model_name)
            
            # Add intelligence layer to registry data
            intelligence[model_name] = {
                "category": model_info.get("category", "Unknown"),
                "strength": self._get_model_strength(model_name, model_info),
                "weakness": self._get_model_weakness(model_name, model_info),
                "best_for": self._get_model_best_for(model_name, model_info),
                "market_conditions": self._get_market_conditions(model_name, model_info),
                "training_time": model_info.get("training_time", "medium"),
                "interpretability": model_info.get("interpretability", "medium"),
                "data_requirements": model_info.get("data_requirements", "medium"),
                "ensemble_weight": self._calculate_ensemble_weight(model_name, model_info)
            }
        
        return intelligence
    
    def _get_model_strength(self, model_name: str, model_info: Dict) -> str:
        """Get model strength based on type and category"""
        category = model_info.get("category", "")
        model_type = model_info.get("type", "")
        
        strength_map = {
            "RandomForest": "Robust to overfitting, handles mixed data types",
            "XGBoost": "High performance, handles missing values well",
            "LightGBM": "Fast training, memory efficient",
            "LSTM": "Captures long-term dependencies, sequential patterns",
            "ARIMA": "Statistical rigor, time series forecasting",
            "Prophet": "Handles seasonality, missing data robust"
        }
        
        return strength_map.get(model_name, f"General {category.lower()} capabilities")
    
    def _get_model_weakness(self, model_name: str, model_info: Dict) -> str:
        """Get model weakness based on type and category"""
        weakness_map = {
            "RandomForest": "Can overfit with very noisy data",
            "XGBoost": "Prone to overfitting without proper tuning",
            "LightGBM": "Sensitive to hyperparameters",
            "LSTM": "Requires large datasets, slower training",
            "ARIMA": "Requires stationarity assumptions",
            "Prophet": "Less flexible for complex patterns"
        }
        
        return weakness_map.get(model_name, "Requires proper tuning and validation")
    
    def _get_model_best_for(self, model_name: str, model_info: Dict) -> List[str]:
        """Get what the model is best for"""
        best_for_map = {
            "RandomForest": ["Medium-term predictions", "Feature importance analysis", "Stable markets"],
            "XGBoost": ["High accuracy requirements", "Competitions", "Complex patterns"],
            "LightGBM": ["Large datasets", "Fast training", "Memory constraints"],
            "LSTM": ["Time series prediction", "Long-term trends", "Sequential data"],
            "ARIMA": ["Traditional forecasting", "Statistical analysis", "Short-term predictions"],
            "Prophet": ["Business forecasting", "Holiday effects", "Trend analysis"]
        }
        
        return best_for_map.get(model_name, ["General machine learning tasks"])
    
    def _get_market_conditions(self, model_name: str, model_info: Dict) -> List[str]:
        """Get suitable market conditions for the model"""
        conditions_map = {
            "RandomForest": ["stable", "trending"],
            "XGBoost": ["volatile", "complex"],
            "LightGBM": ["trending", "stable"],
            "LSTM": ["trending", "cyclical"],
            "ARIMA": ["stable", "predictable"],
            "Prophet": ["trending", "seasonal"]
        }
        
        return conditions_map.get(model_name, ["general"])
    
    def _calculate_ensemble_weight(self, model_name: str, model_info: Dict) -> float:
        """Calculate ensemble weight based on model characteristics"""
        # Base weights by category
        category_weights = {
            "Tree-Based": 0.15,
            "Deep Learning": 0.20,
            "Time Series": 0.18,
            "Traditional ML": 0.12,
            "Ensemble": 0.25,
            "Advanced Models": 0.16
        }
        
        category = model_info.get("category", "Traditional ML")
        return category_weights.get(category, 0.15)
    
    def _initialize_model_knowledge_legacy(self) -> Dict[str, Dict[str, Any]]:
        """LEGACY: Now replaced by centralized registry. This function is deprecated."""
        # This function is now deprecated - all models come from centralized registry
        return {}
            # ===== TREE-BASED ENSEMBLE MODELS =====
            "RandomForest": {
                "category": "Tree-Based Ensemble",
                "strength": "Robust to overfitting, handles mixed data types",
                "weakness": "Can overfit with very noisy data",
                "best_for": ["Medium-term predictions", "Feature importance analysis", "Stable markets"],
                "market_conditions": ["stable", "trending"],
                "training_time": "fast",
                "interpretability": "high",
                "data_requirements": "medium",
                "ensemble_weight": 0.15
            },
            
            "XGBoost": {
                "category": "Tree-Based Ensemble",
                "strength": "High performance, gradient boosting optimization",
                "weakness": "Requires hyperparameter tuning",
                "best_for": ["Structured data", "Feature importance", "All market conditions"],
                "market_conditions": ["stable", "trending", "volatile"],
                "training_time": "medium",
                "interpretability": "medium",
                "data_requirements": "medium",
                "ensemble_weight": 0.20
            },
            
            "LightGBM": {
                "category": "Tree-Based Ensemble", 
                "strength": "Fast training, memory efficient",
                "weakness": "Can overfit with small datasets",
                "best_for": ["Large datasets", "Fast predictions", "Real-time trading"],
                "market_conditions": ["trending", "volatile"],
                "training_time": "fast",
                "interpretability": "medium",
                "data_requirements": "large",
                "ensemble_weight": 0.18
            },
            
            "CatBoost": {
                "category": "Tree-Based Ensemble",
                "strength": "Handles categorical features automatically",
                "weakness": "Slower than LightGBM",
                "best_for": ["Mixed data types", "Categorical features", "Robust predictions"],
                "market_conditions": ["stable", "irregular"],
                "training_time": "slow",
                "interpretability": "medium",
                "data_requirements": "medium",
                "ensemble_weight": 0.17
            },
            
            "GradientBoosting": {
                "category": "Tree-Based Ensemble",
                "strength": "Sequential error correction",
                "weakness": "Prone to overfitting",
                "best_for": ["Small to medium datasets", "Careful tuning"],
                "market_conditions": ["stable"],
                "training_time": "medium",
                "interpretability": "medium",
                "data_requirements": "small",
                "ensemble_weight": 0.12
            },
            
            "DecisionTree": {
                "category": "Tree-Based Ensemble",
                "strength": "Highly interpretable, simple rules",
                "weakness": "Prone to overfitting",
                "best_for": ["Rule extraction", "Interpretable decisions", "Simple patterns"],
                "market_conditions": ["stable"],
                "training_time": "fast",
                "interpretability": "very_high",
                "data_requirements": "small",
                "ensemble_weight": 0.08
            },
            
            # ===== DEEP LEARNING MODELS =====
            "LSTM": {
                "category": "Deep Learning",
                "strength": "Captures long-term dependencies, sequential patterns",
                "weakness": "Requires large datasets, slower training",
                "best_for": ["Time series prediction", "Long-term trends", "Sequential data"],
                "market_conditions": ["trending", "cyclical"],
                "training_time": "slow",
                "interpretability": "low",
                "data_requirements": "large",
                "ensemble_weight": 0.22
            },
            
            "GRU": {
                "category": "Deep Learning",
                "strength": "Faster than LSTM, good for shorter sequences",
                "weakness": "Less powerful than LSTM for long sequences",
                "best_for": ["Medium-term predictions", "Faster training", "Resource constraints"],
                "market_conditions": ["trending", "volatile"],
                "training_time": "medium",
                "interpretability": "low",
                "data_requirements": "medium",
                "ensemble_weight": 0.19
            },
            
            "Transformer": {
                "category": "Deep Learning",
                "strength": "Attention mechanism, parallel processing",
                "weakness": "Requires very large datasets",
                "best_for": ["Complex patterns", "Multi-feature attention", "Advanced pattern recognition"],
                "market_conditions": ["complex", "multi_factor"],
                "training_time": "very_slow",
                "interpretability": "very_low",
                "data_requirements": "very_large",
                "ensemble_weight": 0.25
            },
            
            "CNN_LSTM": {
                "category": "Deep Learning",
                "strength": "Combines spatial and temporal features",
                "weakness": "Complex architecture, hard to tune",
                "best_for": ["Multi-dimensional time series", "Complex patterns"],
                "market_conditions": ["volatile", "complex"],
                "training_time": "slow",
                "interpretability": "very_low",
                "data_requirements": "large",
                "ensemble_weight": 0.21
            },
            
            "Autoencoder": {
                "category": "Deep Learning",
                "strength": "Dimensionality reduction, anomaly detection",
                "weakness": "Not directly predictive",
                "best_for": ["Feature extraction", "Anomaly detection", "Data preprocessing"],
                "market_conditions": ["irregular", "anomaly_detection"],
                "training_time": "medium",
                "interpretability": "low",
                "data_requirements": "medium",
                "ensemble_weight": 0.10
            },
            
            "BiLSTM": {
                "category": "Deep Learning",
                "strength": "Bidirectional processing, captures future and past",
                "weakness": "Cannot be used for real-time prediction",
                "best_for": ["Historical analysis", "Pattern recognition", "Complete sequences"],
                "market_conditions": ["trending", "cyclical"],
                "training_time": "slow",
                "interpretability": "low",
                "data_requirements": "large",
                "ensemble_weight": 0.23
            },
            
            "TemporalFusionTransformer": {
                "category": "Deep Learning",
                "strength": "State-of-the-art time series, attention mechanism",
                "weakness": "Extremely complex, requires large datasets",
                "best_for": ["Multi-variate forecasting", "Complex temporal patterns", "Advanced predictions"],
                "market_conditions": ["complex", "multi_factor", "long_term"],
                "training_time": "very_slow",
                "interpretability": "low",
                "data_requirements": "very_large",
                "ensemble_weight": 0.28
            },
            
            "FeedforwardMLP": {
                "category": "Deep Learning",
                "strength": "Universal approximator, flexible architecture",
                "weakness": "No temporal awareness, prone to overfitting",
                "best_for": ["Non-sequential patterns", "Feature interactions", "General classification"],
                "market_conditions": ["cross_sectional", "feature_rich"],
                "training_time": "medium",
                "interpretability": "low",
                "data_requirements": "medium",
                "ensemble_weight": 0.14
            },
            
            # ===== TRADITIONAL ML MODELS =====
            "LinearRegression": {
                "category": "Traditional ML",
                "strength": "Simple, interpretable, fast",
                "weakness": "Assumes linear relationships",
                "best_for": ["Linear trends", "Baseline models", "Feature selection"],
                "market_conditions": ["stable", "trending"],
                "accuracy_range": [0.60, 0.70],
                "training_time": "very_fast",
                "interpretability": "very_high",
                "data_requirements": "small",
                "ensemble_weight": 0.05
            },
            
            "Ridge": {
                "category": "Traditional ML",
                "strength": "Handles multicollinearity, regularization",
                "weakness": "Still assumes linearity",
                "best_for": ["High-dimensional data", "Regularized linear models"],
                "market_conditions": ["stable"],
                "accuracy_range": [0.62, 0.72],
                "training_time": "fast",
                "interpretability": "high",
                "data_requirements": "medium",
                "ensemble_weight": 0.06
            },
            
            "Lasso": {
                "category": "Traditional ML",
                "strength": "Feature selection, sparsity",
                "weakness": "Can be unstable with correlated features",
                "best_for": ["Feature selection", "Sparse models", "Identifying key factors"],
                "market_conditions": ["stable"],
                "accuracy_range": [0.61, 0.71],
                "training_time": "fast",
                "interpretability": "high",
                "data_requirements": "medium",
                "ensemble_weight": 0.06
            },
            
            "SVR": {
                "category": "Traditional ML",
                "strength": "Non-linear relationships, kernel trick",
                "weakness": "Sensitive to feature scaling, slow on large datasets",
                "best_for": ["Non-linear patterns", "Medium datasets", "Robust predictions"],
                "market_conditions": ["volatile", "non_linear"],
                "accuracy_range": [0.72, 0.82],
                "training_time": "medium",
                "interpretability": "low",
                "data_requirements": "medium",
                "ensemble_weight": 0.13
            },
            
            "ElasticNet": {
                "category": "Traditional ML",
                "strength": "Combines Ridge and Lasso benefits",
                "weakness": "Still linear assumption",
                "best_for": ["Balanced regularization", "Feature selection with groups"],
                "market_conditions": ["stable"],
                "training_time": "fast",
                "interpretability": "high",
                "data_requirements": "medium",
                "ensemble_weight": 0.07
            },
            
            "KNearestNeighbors": {
                "category": "Traditional ML",
                "strength": "Non-parametric, captures local patterns",
                "weakness": "Sensitive to curse of dimensionality",
                "best_for": ["Pattern recognition", "Local similarity", "Anomaly detection"],
                "market_conditions": ["pattern_based", "local_trends"],
                "training_time": "fast",
                "interpretability": "medium",
                "data_requirements": "medium",
                "ensemble_weight": 0.09
            },
            
            "LogisticRegression": {
                "category": "Traditional ML",
                "strength": "Probabilistic outputs, interpretable coefficients",
                "weakness": "Assumes linear decision boundary",
                "best_for": ["Binary classification", "Probability estimation", "Feature importance"],
                "market_conditions": ["classification_tasks", "probability_needed"],
                "training_time": "very_fast",
                "interpretability": "very_high",
                "data_requirements": "small",
                "ensemble_weight": 0.08
            },
            
            # ===== TIME SERIES MODELS =====
            "ARIMA": {
                "category": "Time Series",
                "strength": "Classical time series, seasonal patterns",
                "weakness": "Assumes stationarity, linear relationships",
                "best_for": ["Seasonal forecasting", "Traditional time series", "Trend analysis"],
                "market_conditions": ["trending", "seasonal"],
                "training_time": "medium",
                "interpretability": "medium",
                "data_requirements": "medium",
                "ensemble_weight": 0.11
            },
            
            "SARIMA": {
                "category": "Time Series",
                "strength": "Seasonal ARIMA with advanced seasonal modeling",
                "weakness": "Complex parameter tuning, computationally intensive",
                "best_for": ["Complex seasonal patterns", "Multi-seasonal data", "Advanced forecasting"],
                "market_conditions": ["seasonal", "complex_patterns"],
                "training_time": "slow",
                "interpretability": "medium",
                "data_requirements": "large",
                "ensemble_weight": 0.13
            },
            
            "Prophet": {
                "category": "Time Series",
                "strength": "Handles holidays, trends, seasonality automatically",
                "weakness": "Less flexible than deep learning",
                "best_for": ["Business forecasting", "Holiday effects", "Multiple seasonalities"],
                "market_conditions": ["seasonal", "holiday_effects"],
                "training_time": "fast",
                "interpretability": "medium",
                "data_requirements": "medium",
                "ensemble_weight": 0.12
            },
            
            "ExponentialSmoothing": {
                "category": "Time Series",
                "strength": "Simple, robust, handles trends and seasonality",
                "weakness": "Limited to exponential patterns",
                "best_for": ["Short-term forecasting", "Simple trends", "Inventory management"],
                "market_conditions": ["stable", "trending"],
                "training_time": "very_fast",
                "interpretability": "high",
                "data_requirements": "small",
                "ensemble_weight": 0.08
            },
            
            "RollingMeanReversion": {
                "category": "Time Series",
                "strength": "Captures mean-reverting behavior",
                "weakness": "Assumes stationary mean",
                "best_for": ["Mean reversion strategies", "Range-bound markets", "Pair trading"],
                "market_conditions": ["range_bound", "mean_reverting"],
                "training_time": "fast",
                "interpretability": "high",
                "data_requirements": "medium",
                "ensemble_weight": 0.09
            },
            
            "KalmanFilter": {
                "category": "Time Series",
                "strength": "State space modeling, noise filtering",
                "weakness": "Requires careful state definition",
                "best_for": ["Noisy data", "State estimation", "Real-time filtering"],
                "market_conditions": ["noisy", "real_time"],
                "training_time": "medium",
                "interpretability": "medium",
                "data_requirements": "medium",
                "ensemble_weight": 0.11
            },
            
            "SeasonalDecomposition": {
                "category": "Time Series",
                "strength": "Separates trend, seasonal, residual components",
                "weakness": "Descriptive rather than predictive",
                "best_for": ["Data exploration", "Understanding components", "Preprocessing"],
                "market_conditions": ["seasonal", "analysis"],
                "training_time": "fast",
                "interpretability": "very_high",
                "data_requirements": "small",
                "ensemble_weight": 0.05
            },
            
            # ===== FINANCIAL MODELS =====
            "BlackScholes": {
                "category": "Financial Models",
                "strength": "Option pricing, theoretical foundation",
                "weakness": "Assumes constant volatility",
                "best_for": ["Option pricing", "Risk management", "Derivatives"],
                "market_conditions": ["stable_volatility"],
                "accuracy_range": [0.75, 0.85],
                "training_time": "very_fast",
                "interpretability": "very_high",
                "data_requirements": "small",
                "ensemble_weight": 0.12
            },
            
            "MonteCarloSimulation": {
                "category": "Financial Models", 
                "strength": "Handles complex scenarios, uncertainty quantification",
                "weakness": "Computationally intensive",
                "best_for": ["Risk assessment", "Scenario analysis", "Portfolio optimization"],
                "market_conditions": ["uncertain", "risk_analysis"],
                "accuracy_range": [0.70, 0.80],
                "training_time": "slow",
                "interpretability": "medium",
                "data_requirements": "medium",
                "ensemble_weight": 0.11
            },
            
            "VaR": {
                "category": "Financial Models",
                "strength": "Risk quantification, regulatory compliance",
                "weakness": "Tail risk underestimation",
                "best_for": ["Risk management", "Regulatory reporting", "Loss estimation"],
                "market_conditions": ["risk_management"],
                "accuracy_range": [0.72, 0.82],
                "training_time": "fast",
                "interpretability": "high",
                "data_requirements": "medium",
                "ensemble_weight": 0.09
            },
            
            "GARCH": {
                "category": "Financial Models",
                "strength": "Volatility modeling, clustering effects",
                "weakness": "Complex parameter estimation",
                "best_for": ["Volatility forecasting", "Risk modeling", "Heteroskedasticity"],
                "market_conditions": ["volatile", "clustering"],
                "accuracy_range": [0.74, 0.84],
                "training_time": "medium",
                "interpretability": "medium",
                "data_requirements": "medium",
                "ensemble_weight": 0.13
            },
            
            "MarkovSwitching": {
                "category": "Financial Models",
                "strength": "Regime changes, state transitions",
                "weakness": "Complex interpretation",
                "best_for": ["Regime detection", "Market transitions", "State-dependent modeling"],
                "market_conditions": ["regime_change", "transitions"],
                "accuracy_range": [0.76, 0.86],
                "training_time": "medium",
                "interpretability": "low",
                "data_requirements": "large",
                "ensemble_weight": 0.14
            },
            
            # ===== META-LEARNING MODELS =====
            "EnsembleVoting": {
                "category": "Meta-Learning",
                "strength": "Combines multiple models, reduces overfitting",
                "weakness": "Performance limited by weakest models",
                "best_for": ["Model combination", "Robust predictions", "Ensemble learning"],
                "market_conditions": ["all"],
                "training_time": "medium",
                "interpretability": "medium",
                "data_requirements": "medium",
                "ensemble_weight": 0.18
            },
            
            "VotingClassifier": {
                "category": "Meta-Learning",
                "strength": "Simple voting mechanism, interpretable",
                "weakness": "Equal weight assumption",
                "best_for": ["Classification tasks", "Simple ensembles", "Model consensus"],
                "market_conditions": ["classification", "consensus_needed"],
                "training_time": "fast",
                "interpretability": "high",
                "data_requirements": "medium",
                "ensemble_weight": 0.14
            },
            
            "Bagging": {
                "category": "Meta-Learning",
                "strength": "Bootstrap aggregation, reduces variance",
                "weakness": "May not improve bias",
                "best_for": ["Variance reduction", "Unstable models", "Bootstrap sampling"],
                "market_conditions": ["high_variance", "unstable"],
                "training_time": "medium",
                "interpretability": "medium",
                "data_requirements": "medium",
                "ensemble_weight": 0.16
            },
            
            "BoostedTreesEnsemble": {
                "category": "Meta-Learning",
                "strength": "Sequential learning, bias reduction",
                "weakness": "Prone to overfitting",
                "best_for": ["Sequential improvement", "Bias reduction", "Iterative learning"],
                "market_conditions": ["sequential", "bias_correction"],
                "training_time": "slow",
                "interpretability": "medium",
                "data_requirements": "large",
                "ensemble_weight": 0.19
            },
            
            "StackingEnsemble": {
                "category": "Meta-Learning",
                "strength": "Meta-learning, optimal combination",
                "weakness": "Risk of overfitting to meta-features",
                "best_for": ["Maximum accuracy", "Competition models", "Complex ensembles"],
                "market_conditions": ["all", "complex"],
                "training_time": "slow",
                "interpretability": "low",
                "data_requirements": "large",
                "ensemble_weight": 0.25
            },
            
            "MetaLearnerStrategySelector": {
                "category": "Meta-Learning",
                "strength": "Strategy selection, adaptive learning",
                "weakness": "Complex strategy definition",
                "best_for": ["Strategy selection", "Adaptive trading", "Meta-strategy"],
                "market_conditions": ["strategy_based", "adaptive"],
                "training_time": "slow",
                "interpretability": "low",
                "data_requirements": "large",
                "ensemble_weight": 0.21
            },
            
            "MetaLearner": {
                "category": "Meta-Learning",
                "strength": "Learns to learn, adaptive algorithms",
                "weakness": "Requires diverse tasks",
                "best_for": ["Few-shot learning", "Transfer learning", "Adaptive models"],
                "market_conditions": ["changing", "adaptive"],
                "training_time": "very_slow",
                "interpretability": "very_low",
                "data_requirements": "very_large",
                "ensemble_weight": 0.20
            },
            
            "MAML": {
                "category": "Meta-Learning",
                "strength": "Model-agnostic meta-learning",
                "weakness": "Very complex, requires many tasks",
                "best_for": ["Fast adaptation", "New market conditions", "Transfer learning"],
                "market_conditions": ["new_regimes", "adaptation"],
                "training_time": "very_slow",
                "interpretability": "very_low",
                "data_requirements": "very_large",
                "ensemble_weight": 0.22
            },
            
            # ===== REINFORCEMENT LEARNING =====
            "QLearning": {
                "category": "Reinforcement Learning",
                "strength": "Model-free learning, simple implementation",
                "weakness": "Requires discrete action space",
                "best_for": ["Simple trading rules", "Discrete actions", "Learning environments"],
                "market_conditions": ["structured", "rule_based"],
                "training_time": "medium",
                "interpretability": "medium",
                "data_requirements": "medium",
                "ensemble_weight": 0.12
            },
            
            "DoubleQLearning": {
                "category": "Reinforcement Learning", 
                "strength": "Reduces overestimation bias",
                "weakness": "More complex than Q-Learning",
                "best_for": ["Improved Q-Learning", "Bias reduction", "Stable learning"],
                "market_conditions": ["stable_learning", "bias_sensitive"],
                "training_time": "medium",
                "interpretability": "medium",
                "data_requirements": "medium",
                "ensemble_weight": 0.13
            },
            
            "DuelingDQN": {
                "category": "Reinforcement Learning",
                "strength": "Separates state value and advantage",
                "weakness": "More complex architecture",
                "best_for": ["Value-based learning", "State evaluation", "Action selection"],
                "market_conditions": ["value_focused", "state_dependent"],
                "training_time": "slow",
                "interpretability": "low",
                "data_requirements": "large",
                "ensemble_weight": 0.16
            },
            
            "DQN": {
                "category": "Reinforcement Learning",
                "strength": "Deep Q-Learning, experience replay",
                "weakness": "Requires reward engineering",
                "best_for": ["Trading decisions", "Portfolio optimization", "Action selection"],
                "market_conditions": ["dynamic", "decision_making"],
                "training_time": "very_slow",
                "interpretability": "low",
                "data_requirements": "very_large",
                "ensemble_weight": 0.15
            },
            
            "RegimeAwareDQN": {
                "category": "Reinforcement Learning",
                "strength": "Adapts to market regimes, context-aware",
                "weakness": "Very complex, hard to debug",
                "best_for": ["Regime-specific trading", "Adaptive strategies", "Complex markets"],
                "market_conditions": ["regime_dependent", "adaptive"],
                "training_time": "very_slow",
                "interpretability": "very_low",
                "data_requirements": "very_large",
                "ensemble_weight": 0.17
            },
            
            # ===== REGIME DETECTION & CLUSTERING MODELS =====
            "HiddenMarkovModel": {
                "category": "Regime Detection",
                "strength": "Hidden state modeling, regime transitions",
                "weakness": "Assumes Markovian transitions",
                "best_for": ["Regime detection", "State transitions", "Hidden patterns"],
                "market_conditions": ["regime_transitions", "hidden_states"],
                "training_time": "medium",
                "interpretability": "medium",
                "data_requirements": "medium",
                "ensemble_weight": 0.17
            },
            
            "KMeansClustering": {
                "category": "Regime Detection",
                "strength": "Simple clustering, interpretable centers",
                "weakness": "Assumes spherical clusters",
                "best_for": ["Market regime clustering", "Pattern grouping", "Unsupervised learning"],
                "market_conditions": ["clustering", "regime_identification"],
                "training_time": "fast",
                "interpretability": "high",
                "data_requirements": "medium",
                "ensemble_weight": 0.12
            },
            
            "BayesianChangePointDetection": {
                "category": "Regime Detection",
                "strength": "Probabilistic change detection, uncertainty quantification",
                "weakness": "Complex prior specification",
                "best_for": ["Change point detection", "Regime shifts", "Bayesian inference"],
                "market_conditions": ["change_points", "regime_shifts"],
                "training_time": "slow",
                "interpretability": "medium",
                "data_requirements": "large",
                "ensemble_weight": 0.18
            },
            
            "RollingZScoreRegimeScorer": {
                "category": "Regime Detection",
                "strength": "Simple statistical approach, real-time capable",
                "weakness": "Limited to statistical regimes",
                "best_for": ["Real-time regime scoring", "Statistical anomalies", "Quick detection"],
                "market_conditions": ["real_time", "statistical_regimes"],
                "training_time": "very_fast",
                "interpretability": "very_high",
                "data_requirements": "small",
                "ensemble_weight": 0.10
            },
            
            # ===== FORECASTING & OPTIMIZATION MODELS =====
            "BayesianRidgeForecast": {
                "category": "Forecasting & Optimization",
                "strength": "Bayesian uncertainty, regularization",
                "weakness": "Still assumes linearity",
                "best_for": ["Uncertain forecasting", "Bayesian predictions", "Risk quantification"],
                "market_conditions": ["uncertainty", "bayesian_needed"],
                "training_time": "fast",
                "interpretability": "high",
                "data_requirements": "medium",
                "ensemble_weight": 0.11
            },
            
            "MarkowitzMeanVarianceOptimizer": {
                "category": "Forecasting & Optimization",
                "strength": "Classical portfolio theory, risk-return optimization",
                "weakness": "Assumes normal distributions",
                "best_for": ["Portfolio optimization", "Risk-return balance", "Asset allocation"],
                "market_conditions": ["portfolio_optimization", "risk_return"],
                "training_time": "fast",
                "interpretability": "very_high",
                "data_requirements": "small",
                "ensemble_weight": 0.13
            },
            
            "DynamicRiskParityModel": {
                "category": "Forecasting & Optimization",
                "strength": "Risk-based allocation, dynamic rebalancing",
                "weakness": "Risk estimation challenges",
                "best_for": ["Risk parity", "Dynamic allocation", "Risk management"],
                "market_conditions": ["risk_parity", "dynamic_allocation"],
                "training_time": "medium",
                "interpretability": "medium",
                "data_requirements": "medium",
                "ensemble_weight": 0.15
            },
            
            "MaximumSharpeRatioOptimizer": {
                "category": "Forecasting & Optimization",
                "strength": "Sharpe ratio maximization, performance focus",
                "weakness": "Sensitive to estimation errors",
                "best_for": ["Performance optimization", "Sharpe maximization", "Efficient frontier"],
                "market_conditions": ["performance_focused", "sharpe_optimization"],
                "training_time": "fast",
                "interpretability": "high",
                "data_requirements": "medium",
                "ensemble_weight": 0.14
            },
            
            # ===== NLP & SENTIMENT MODELS =====
            "FinBERTSentimentClassifier": {
                "category": "NLP & Sentiment",
                "strength": "Financial domain expertise, pre-trained",
                "weakness": "Requires text preprocessing",
                "best_for": ["Financial sentiment", "News analysis", "Market sentiment"],
                "market_conditions": ["sentiment_analysis", "news_driven"],
                "training_time": "medium",
                "interpretability": "low",
                "data_requirements": "large",
                "ensemble_weight": 0.20
            },
            
            "BERTClassificationHead": {
                "category": "NLP & Sentiment",
                "strength": "General NLP capabilities, transferable",
                "weakness": "Not finance-specific",
                "best_for": ["General text classification", "Transfer learning", "Multi-domain"],
                "market_conditions": ["text_classification", "general_nlp"],
                "training_time": "slow",
                "interpretability": "low",
                "data_requirements": "very_large",
                "ensemble_weight": 0.18
            },
            
            "SentenceTransformerEmbedding": {
                "category": "NLP & Sentiment",
                "strength": "Semantic embeddings, clustering capability",
                "weakness": "Indirect prediction approach",
                "best_for": ["Semantic analysis", "Text clustering", "Similarity detection"],
                "market_conditions": ["semantic_analysis", "text_clustering"],
                "training_time": "medium",
                "interpretability": "low",
                "data_requirements": "large",
                "ensemble_weight": 0.16
            },
            
            # ===== MOMENTUM-SPECIFIC MODELS =====
            "RSIModel": {
                "category": "Momentum Models",
                "strength": "Momentum overbought/oversold detection, proven track record",
                "weakness": "Can give false signals in trending markets",
                "best_for": ["Momentum reversal", "Entry/exit timing", "Short-term signals"],
                "market_conditions": ["momentum_shifts", "reversal_points"],
                "training_time": "very_fast",
                "interpretability": "very_high",
                "data_requirements": "small",
                "ensemble_weight": 0.15
            },
            
            "MACDModel": {
                "category": "Momentum Models", 
                "strength": "Trend and momentum convergence/divergence, dual signal",
                "weakness": "Lagging indicator, false signals in sideways markets",
                "best_for": ["Trend confirmation", "Momentum shifts", "Signal quality"],
                "market_conditions": ["trending", "momentum_changes"],
                "training_time": "very_fast",
                "interpretability": "very_high", 
                "data_requirements": "small",
                "ensemble_weight": 0.18
            },
            
            "BollingerBreakoutModel": {
                "category": "Momentum Models",
                "strength": "Volatility-adjusted momentum breakouts, adaptive bands",
                "weakness": "False breakouts in choppy markets",
                "best_for": ["Momentum breakouts", "Volatility expansion", "Trend initiation"],
                "market_conditions": ["breakout", "momentum_acceleration"],
                "training_time": "fast",
                "interpretability": "high",
                "data_requirements": "medium",
                "ensemble_weight": 0.16
            },
            
            "VolumePriceTrendModel": {
                "category": "Momentum Models",
                "strength": "Volume-confirmed momentum, institutional flow detection",
                "weakness": "Requires reliable volume data",
                "best_for": ["Volume confirmation", "Institutional momentum", "Smart money flow"],
                "market_conditions": ["volume_surge", "institutional_activity"],
                "training_time": "fast",
                "interpretability": "high",
                "data_requirements": "medium",
                "ensemble_weight": 0.17
            },
            
            # ===== CUTTING-EDGE AI MODELS =====
            "VisionTransformerChart": {
                "category": "Cutting-Edge AI",
                "strength": "Chart pattern recognition, visual momentum patterns",
                "weakness": "Requires extensive training data, computationally intensive",
                "best_for": ["Chart patterns", "Visual momentum", "Pattern completion"],
                "market_conditions": ["pattern_formation", "visual_signals"],
                "training_time": "very_slow",
                "interpretability": "low",
                "data_requirements": "very_large",
                "ensemble_weight": 0.22
            },
            
            "GraphNeuralNetwork": {
                "category": "Cutting-Edge AI",
                "strength": "Market relationship modeling, interconnected momentum",
                "weakness": "Complex architecture, relationship dependency",
                "best_for": ["Market relationships", "Sector momentum", "Interconnected signals"],
                "market_conditions": ["sector_rotation", "relationship_driven"],
                "training_time": "slow",
                "interpretability": "low",
                "data_requirements": "large",
                "ensemble_weight": 0.20
            },
            
            "AdversarialMomentumNet": {
                "category": "Cutting-Edge AI",
                "strength": "Robust momentum predictions, adversarial training",
                "weakness": "Training complexity, potential overfitting",
                "best_for": ["Robust predictions", "Noise resistance", "Adversarial markets"],
                "market_conditions": ["noisy_markets", "adversarial_conditions"],
                "training_time": "very_slow",
                "interpretability": "very_low",
                "data_requirements": "very_large",
                "ensemble_weight": 0.24
            },
            
            # ===== ADDITIONAL MOMENTUM MODELS =====
            "WilliamsRModel": {
                "category": "Momentum Models",
                "strength": "Momentum extremes, reversal signals",
                "weakness": "Can be noisy in trending markets",
                "best_for": ["Reversal timing", "Extreme momentum", "Entry points"],
                "market_conditions": ["momentum_extremes", "reversal_setup"],
                "training_time": "very_fast",
                "interpretability": "very_high",
                "data_requirements": "small",
                "ensemble_weight": 0.14
            },
            
            "StochasticModel": {
                "category": "Momentum Models",
                "strength": "Smooth momentum oscillator, crossover signals",
                "weakness": "Lagging in fast moves",
                "best_for": ["Smooth momentum", "Crossover signals", "Timing"],
                "market_conditions": ["smooth_trends", "crossover_setups"],
                "training_time": "very_fast",
                "interpretability": "very_high",
                "data_requirements": "small",
                "ensemble_weight": 0.13
            },
            
            "CommodityChannelIndex": {
                "category": "Momentum Models",
                "strength": "Cyclical turning points, momentum divergence",
                "weakness": "Complex interpretation",
                "best_for": ["Cyclical analysis", "Divergence detection", "Turning points"],
                "market_conditions": ["cyclical_markets", "divergence_signals"],
                "training_time": "fast",
                "interpretability": "medium",
                "data_requirements": "medium",
                "ensemble_weight": 0.15
            },
            
            "AccumulationDistributionLine": {
                "category": "Momentum Models",
                "strength": "Volume-price relationship, accumulation detection",
                "weakness": "Requires accurate volume data",
                "best_for": ["Smart money tracking", "Accumulation phases", "Volume confirmation"],
                "market_conditions": ["accumulation_phase", "volume_analysis"],
                "training_time": "fast",
                "interpretability": "high",
                "data_requirements": "medium",
                "ensemble_weight": 0.16
            },
            
            # ===== ADVANCED TIME SERIES MODELS =====
            "FractalModel": {
                "category": "Cutting-Edge AI",
                "strength": "Multi-timeframe analysis, fractal patterns",
                "weakness": "Complex mathematical framework",
                "best_for": ["Multi-timeframe momentum", "Fractal patterns", "Scale analysis"],
                "market_conditions": ["multi_timeframe", "fractal_patterns"],
                "training_time": "slow",
                "interpretability": "low",
                "data_requirements": "large",
                "ensemble_weight": 0.19
            },
            
            "WaveletTransformModel": {
                "category": "Cutting-Edge AI",
                "strength": "Time-frequency analysis, momentum decomposition",
                "weakness": "Requires signal processing expertise",
                "best_for": ["Time-frequency analysis", "Momentum decomposition", "Multi-scale"],
                "market_conditions": ["frequency_analysis", "multi_scale"],
                "training_time": "medium",
                "interpretability": "low",
                "data_requirements": "large",
                "ensemble_weight": 0.18
            },
            
            "EmpiricalModeDecomposition": {
                "category": "Cutting-Edge AI",
                "strength": "Trend decomposition, intrinsic mode functions",
                "weakness": "Computational complexity",
                "best_for": ["Trend decomposition", "Mode analysis", "Signal separation"],
                "market_conditions": ["trend_analysis", "mode_separation"],
                "training_time": "medium",
                "interpretability": "medium",
                "data_requirements": "large",
                "ensemble_weight": 0.17
            },
            
            # ===== CRITICAL MISSING MODELS FROM INVENTORY =====
            "KellyCriterionBayesian": {
                "category": "Financial Engineering",
                "strength": "Optimal position sizing with Bayesian estimation",
                "weakness": "Requires accurate probability estimates",
                "best_for": ["Position sizing", "Risk management", "Capital allocation"],
                "market_conditions": ["position_optimization", "risk_sizing"],
                "training_time": "fast",
                "interpretability": "very_high",
                "data_requirements": "medium",
                "ensemble_weight": 0.19
            },
            
            "ThresholdAutoregressive": {
                "category": "Time Series Models",
                "strength": "Nonlinear time series modeling, regime thresholds",
                "weakness": "Complex threshold determination",
                "best_for": ["Nonlinear patterns", "Threshold detection", "Regime switching"],
                "market_conditions": ["nonlinear_markets", "threshold_breaks"],
                "training_time": "medium",
                "interpretability": "medium",
                "data_requirements": "large",
                "ensemble_weight": 0.16
            },
            
            "HurstExponentFractal": {
                "category": "Signal Processing",
                "strength": "Long memory detection, persistence analysis",
                "weakness": "Sensitive to data length",
                "best_for": ["Momentum persistence", "Long memory", "Fractal analysis"],
                "market_conditions": ["persistent_trends", "memory_effects"],
                "training_time": "fast",
                "interpretability": "high",
                "data_requirements": "large",
                "ensemble_weight": 0.14
            },
            
            "TransferEntropy": {
                "category": "Information Theory",
                "strength": "Information flow analysis, causality detection",
                "weakness": "Requires large datasets",
                "best_for": ["Lead-lag relationships", "Information flow", "Causality"],
                "market_conditions": ["information_flow", "causal_analysis"],
                "training_time": "medium",
                "interpretability": "low",
                "data_requirements": "very_large",
                "ensemble_weight": 0.18
            },
            
            "ProximalPolicyOptimization": {
                "category": "Reinforcement Learning",
                "strength": "Advanced RL for trading strategies, policy optimization",
                "weakness": "Complex training, exploration challenges",
                "best_for": ["Dynamic trading", "Strategy optimization", "Sequential decisions"],
                "market_conditions": ["dynamic_strategies", "adaptive_trading"],
                "training_time": "very_slow",
                "interpretability": "very_low",
                "data_requirements": "very_large",
                "ensemble_weight": 0.25
            },
            
            "EWMARiskMetrics": {
                "category": "Volatility Models",
                "strength": "Exponentially weighted volatility, industry standard",
                "weakness": "Parameter sensitivity",
                "best_for": ["Volatility forecasting", "Risk metrics", "Real-time updates"],
                "market_conditions": ["volatility_modeling", "risk_measurement"],
                "training_time": "very_fast",
                "interpretability": "very_high",
                "data_requirements": "medium",
                "ensemble_weight": 0.12
            },
            
            "ShannonEntropyMutualInfo": {
                "category": "Information Theory",
                "strength": "Information content analysis, entropy measurement",
                "weakness": "Interpretation complexity",
                "best_for": ["Information content", "Pattern complexity", "Entropy analysis"],
                "market_conditions": ["information_analysis", "pattern_detection"],
                "training_time": "fast",
                "interpretability": "medium",
                "data_requirements": "large",
                "ensemble_weight": 0.15
            },
            
            "NeuralODEFinancial": {
                "category": "Cutting-Edge AI",
                "strength": "Continuous dynamics modeling, differential equations",
                "weakness": "Very complex, requires ODE expertise",
                "best_for": ["Continuous dynamics", "Differential modeling", "Advanced AI"],
                "market_conditions": ["continuous_dynamics", "complex_systems"],
                "training_time": "very_slow",
                "interpretability": "very_low",
                "data_requirements": "very_large",
                "ensemble_weight": 0.28
            },
            
            # ===== ADDITIONAL CRITICAL MODELS FROM INVENTORY =====
            "GrangerCausalityTest": {
                "category": "Information Theory",
                "strength": "Lead-lag relationship detection, causality testing",
                "weakness": "Requires stationarity assumptions",
                "best_for": ["Lead-lag analysis", "Causality detection", "Temporal relationships"],
                "market_conditions": ["causality_analysis", "temporal_patterns"],
                "training_time": "fast",
                "interpretability": "high",
                "data_requirements": "large",
                "ensemble_weight": 0.16
            },
            
            "NetworkTopologyAnalysis": {
                "category": "Information Theory",
                "strength": "Graph-based market analysis, network effects",
                "weakness": "Complex network construction",
                "best_for": ["Market networks", "Correlation analysis", "System relationships"],
                "market_conditions": ["network_effects", "systemic_analysis"],
                "training_time": "medium",
                "interpretability": "medium",
                "data_requirements": "very_large",
                "ensemble_weight": 0.22
            },
            
            "RegimeSwitchingVolatility": {
                "category": "Volatility Models",
                "strength": "State-dependent volatility modeling",
                "weakness": "Complex regime identification",
                "best_for": ["Volatility regimes", "State switching", "Risk modeling"],
                "market_conditions": ["regime_volatility", "state_dependent"],
                "training_time": "medium",
                "interpretability": "medium",
                "data_requirements": "large",
                "ensemble_weight": 0.18
            },
            
            "OnlineLearningUpdates": {
                "category": "Meta-Learning",
                "strength": "Real-time model adaptation, continuous learning",
                "weakness": "Catastrophic forgetting risk",
                "best_for": ["Real-time adaptation", "Continuous learning", "Model updates"],
                "market_conditions": ["dynamic_adaptation", "continuous_updates"],
                "training_time": "fast",
                "interpretability": "low",
                "data_requirements": "medium",
                "ensemble_weight": 0.20
            },
            
            "ConceptDriftDetection": {
                "category": "Meta-Learning",
                "strength": "Distribution change detection, model degradation alerts",
                "weakness": "False positive sensitivity",
                "best_for": ["Model monitoring", "Distribution shifts", "Performance degradation"],
                "market_conditions": ["drift_detection", "model_monitoring"],
                "training_time": "fast",
                "interpretability": "high",
                "data_requirements": "medium",
                "ensemble_weight": 0.17
            },
            
            "MarketImpactModels": {
                "category": "Market Microstructure",
                "strength": "Price impact modeling, trade execution cost",
                "weakness": "Market-specific calibration",
                "best_for": ["Execution cost", "Price impact", "Trade sizing"],
                "market_conditions": ["execution_modeling", "impact_analysis"],
                "training_time": "medium",
                "interpretability": "high",
                "data_requirements": "large",
                "ensemble_weight": 0.19
            },
            
            "OrderFlowAnalysis": {
                "category": "Market Microstructure",
                "strength": "Trade flow pattern analysis, institutional detection",
                "weakness": "High-frequency data requirements",
                "best_for": ["Flow analysis", "Institutional tracking", "Volume patterns"],
                "market_conditions": ["order_flow", "institutional_activity"],
                "training_time": "medium",
                "interpretability": "medium",
                "data_requirements": "very_large",
                "ensemble_weight": 0.21
            },
            
            "YieldCurveAnalysis": {
                "category": "Macro Analysis",
                "strength": "Interest rate structure analysis, yield curve modeling",
                "weakness": "Bond market specific",
                "best_for": ["Interest rate analysis", "Yield modeling", "Macro trends"],
                "market_conditions": ["interest_rate_analysis", "macro_modeling"],
                "training_time": "medium",
                "interpretability": "high",
                "data_requirements": "large",
                "ensemble_weight": 0.16
            },
            
            "WalkForwardOptimization": {
                "category": "Optimization",
                "strength": "Out-of-sample parameter optimization, robustness testing",
                "weakness": "Computationally intensive",
                "best_for": ["Parameter optimization", "Robustness testing", "Model validation"],
                "market_conditions": ["parameter_optimization", "validation"],
                "training_time": "very_slow",
                "interpretability": "high",
                "data_requirements": "very_large",
                "ensemble_weight": 0.23
            },
            
            "BayesianParameterOptimization": {
                "category": "Optimization",
                "strength": "Efficient hyperparameter tuning, uncertainty quantification",
                "weakness": "Complex prior specification",
                "best_for": ["Hyperparameter tuning", "Efficient optimization", "Uncertainty"],
                "market_conditions": ["hyperparameter_optimization", "efficient_tuning"],
                "training_time": "medium",
                "interpretability": "medium",
                "data_requirements": "medium",
                "ensemble_weight": 0.20
            },
            
            "AnomalyPatternRecognition": {
                "category": "System Intelligence",
                "strength": "Outlier detection, anomaly identification",
                "weakness": "False positive management",
                "best_for": ["Anomaly detection", "Outlier identification", "Pattern breaks"],
                "market_conditions": ["anomaly_detection", "outlier_analysis"],
                "training_time": "medium",
                "interpretability": "high",
                "data_requirements": "large",
                "ensemble_weight": 0.18
            },
            
            "AutomatedFeatureEngineering": {
                "category": "System Intelligence",
                "strength": "Dynamic feature creation, automated selection",
                "weakness": "Feature explosion risk",
                "best_for": ["Feature creation", "Automated selection", "Dynamic features"],
                "market_conditions": ["feature_engineering", "automated_selection"],
                "training_time": "slow",
                "interpretability": "low",
                "data_requirements": "large",
                "ensemble_weight": 0.22
            },
            
            # ===== FINAL COMPREHENSIVE MODELS FROM INVENTORY =====
            "HyperparameterEvolution": {
                "category": "System Intelligence",
                "strength": "Genetic algorithm optimization, evolutionary tuning",
                "weakness": "Computationally expensive",
                "best_for": ["Parameter evolution", "Genetic optimization", "Hyperparameter search"],
                "market_conditions": ["parameter_evolution", "optimization"],
                "training_time": "very_slow",
                "interpretability": "low",
                "data_requirements": "large",
                "ensemble_weight": 0.24
            },
            
            "CausalInferenceDiscovery": {
                "category": "System Intelligence",
                "strength": "Causality detection methods, causal relationships",
                "weakness": "Complex causal assumptions",
                "best_for": ["Causal analysis", "Relationship discovery", "Inference"],
                "market_conditions": ["causal_discovery", "relationship_analysis"],
                "training_time": "slow",
                "interpretability": "medium",
                "data_requirements": "very_large",
                "ensemble_weight": 0.26
            },
            
            "OmegaRatio": {
                "category": "Risk Analytics",
                "strength": "Probability-weighted gains vs losses",
                "weakness": "Threshold sensitivity",
                "best_for": ["Risk-adjusted returns", "Probability weighting", "Performance"],
                "market_conditions": ["risk_analytics", "performance_measurement"],
                "training_time": "fast",
                "interpretability": "high",
                "data_requirements": "medium",
                "ensemble_weight": 0.13
            },
            
            "SterlingRatio": {
                "category": "Risk Analytics",
                "strength": "Modified Calmar ratio with average drawdown",
                "weakness": "Drawdown period dependency",
                "best_for": ["Risk-adjusted performance", "Drawdown analysis", "Sterling ratio"],
                "market_conditions": ["performance_analysis", "risk_measurement"],
                "training_time": "fast",
                "interpretability": "high",
                "data_requirements": "medium",
                "ensemble_weight": 0.12
            },
            
            "InformationRatio": {
                "category": "Risk Analytics",
                "strength": "Risk-adjusted excess return measurement",
                "weakness": "Benchmark dependency",
                "best_for": ["Excess return analysis", "Risk adjustment", "Performance attribution"],
                "market_conditions": ["performance_analysis", "benchmark_comparison"],
                "training_time": "fast",
                "interpretability": "high",
                "data_requirements": "medium",
                "ensemble_weight": 0.14
            },
            
            "BidAskSpreadAnalysis": {
                "category": "Market Microstructure",
                "strength": "Liquidity cost measurement, spread analysis",
                "weakness": "High-frequency data requirements",
                "best_for": ["Liquidity analysis", "Spread measurement", "Transaction costs"],
                "market_conditions": ["liquidity_analysis", "microstructure"],
                "training_time": "fast",
                "interpretability": "high",
                "data_requirements": "large",
                "ensemble_weight": 0.17
            },
            
            "LiquidityAssessment": {
                "category": "Market Microstructure",
                "strength": "Market liquidity measurement, depth analysis",
                "weakness": "Market-specific calibration",
                "best_for": ["Liquidity measurement", "Market depth", "Trading capacity"],
                "market_conditions": ["liquidity_assessment", "depth_analysis"],
                "training_time": "medium",
                "interpretability": "high",
                "data_requirements": "large",
                "ensemble_weight": 0.18
            },
            
            "MarketStressIndicators": {
                "category": "Macro Analysis",
                "strength": "Stress level calculation, crisis detection",
                "weakness": "Stress threshold calibration",
                "best_for": ["Stress detection", "Crisis identification", "Market tension"],
                "market_conditions": ["stress_analysis", "crisis_detection"],
                "training_time": "medium",
                "interpretability": "high",
                "data_requirements": "large",
                "ensemble_weight": 0.19
            },
            
            "SectorRotationAnalysis": {
                "category": "Macro Analysis",
                "strength": "Cross-sector momentum detection, rotation patterns",
                "weakness": "Sector classification dependency",
                "best_for": ["Sector analysis", "Rotation detection", "Cross-sector momentum"],
                "market_conditions": ["sector_rotation", "momentum_analysis"],
                "training_time": "medium",
                "interpretability": "high",
                "data_requirements": "large",
                "ensemble_weight": 0.20
            },
            
            "ModelArchitectureSearch": {
                "category": "System Intelligence",
                "strength": "Neural architecture search, automated design",
                "weakness": "Extremely computationally intensive",
                "best_for": ["Architecture optimization", "Neural design", "Automated ML"],
                "market_conditions": ["architecture_search", "automated_design"],
                "training_time": "extremely_slow",
                "interpretability": "very_low",
                "data_requirements": "very_large",
                "ensemble_weight": 0.30
            },
            
            "FederatedLearning": {
                "category": "Meta-Learning",
                "strength": "Distributed model training, privacy preservation",
                "weakness": "Complex coordination requirements",
                "best_for": ["Distributed learning", "Privacy preservation", "Collaborative training"],
                "market_conditions": ["distributed_learning", "collaborative_training"],
                "training_time": "slow",
                "interpretability": "low",
                "data_requirements": "very_large",
                "ensemble_weight": 0.25
            },
            
            "ADFKPSSTests": {
                "category": "Statistical Tests",
                "strength": "Stationarity testing, unit root detection",
                "weakness": "Test power limitations",
                "best_for": ["Stationarity testing", "Unit root detection", "Time series validation"],
                "market_conditions": ["stationarity_testing", "statistical_validation"],
                "training_time": "fast",
                "interpretability": "high",
                "data_requirements": "medium",
                "ensemble_weight": 0.11
            },
            
            "LempelZivComplexity": {
                "category": "Information Theory",
                "strength": "Algorithmic complexity measures, pattern complexity",
                "weakness": "Interpretation complexity",
                "best_for": ["Complexity analysis", "Pattern measurement", "Algorithmic entropy"],
                "market_conditions": ["complexity_analysis", "pattern_complexity"],
                "training_time": "medium",
                "interpretability": "medium",
                "data_requirements": "large",
                "ensemble_weight": 0.16
            },
            
            # ===== ADVANCED MODELS =====
            "MultiHeadAttention": {
                "category": "Advanced Models",
                "strength": "Multiple attention mechanisms, parallel processing",
                "weakness": "Very complex, requires large datasets",
                "best_for": ["Complex pattern recognition", "Multi-factor analysis", "State-of-the-art"],
                "market_conditions": ["complex", "multi_factor"],
                "training_time": "very_slow",
                "interpretability": "very_low",
                "data_requirements": "very_large",
                "ensemble_weight": 0.26
            }
        }
    
    def _initialize_combination_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Strategic model combinations for different objectives"""
        return {
            "maximum_accuracy": {
                "models": ["Transformer", "StackingEnsemble", "MultiHeadAttention", "LSTM", "XGBoost"],
                "weights": [0.25, 0.25, 0.20, 0.15, 0.15],
                "description": "Highest possible accuracy combination",
                "use_when": "Competition or critical predictions needed"
            },
            
            "robust_trading": {
                "models": ["XGBoost", "LightGBM", "RandomForest", "LSTM", "EnsembleVoting"],
                "weights": [0.25, 0.20, 0.20, 0.20, 0.15],
                "description": "Balanced accuracy and robustness",
                "use_when": "Real trading with moderate risk tolerance"
            },
            
            "fast_execution": {
                "models": ["LightGBM", "RandomForest", "LinearRegression", "Ridge"],
                "weights": [0.40, 0.30, 0.15, 0.15],
                "description": "Speed-optimized predictions",
                "use_when": "High-frequency trading or real-time decisions"
            },
            
            "interpretable_predictions": {
                "models": ["DecisionTree", "LinearRegression", "Ridge", "BlackScholes"],
                "weights": [0.30, 0.25, 0.25, 0.20],
                "description": "Explainable model decisions",
                "use_when": "Regulatory compliance or explanation needed"
            },
            
            "volatile_markets": {
                "models": ["GARCH", "SVR", "LSTM", "RegimeAwareDQN", "MarkovSwitching"],
                "weights": [0.25, 0.20, 0.20, 0.20, 0.15],
                "description": "Optimized for high volatility periods",
                "use_when": "Market stress or uncertainty"
            },
            
            "trend_following": {
                "models": ["LSTM", "GRU", "Prophet", "ARIMA", "XGBoost"],
                "weights": [0.30, 0.25, 0.20, 0.15, 0.10],
                "description": "Trend identification and following",
                "use_when": "Strong directional markets"
            },
            
            "regime_detection": {
                "models": ["MarkovSwitching", "RegimeAwareDQN", "MetaLearner", "GARCH"],
                "weights": [0.30, 0.25, 0.25, 0.20],
                "description": "Market regime identification",
                "use_when": "Market transition periods"
            },
            
            "risk_management": {
                "models": ["VaR", "MonteCarloSimulation", "GARCH", "BlackScholes", "SVR"],
                "weights": [0.25, 0.25, 0.20, 0.15, 0.15],
                "description": "Risk assessment and management",
                "use_when": "Portfolio risk evaluation"
            },
            
            # ===== MOMENTUM-SPECIFIC ENSEMBLE STRATEGIES =====
            "momentum_identification": {
                "models": ["RSIModel", "MACDModel", "BollingerBreakoutModel", "VolumePriceTrendModel", "LSTM"],
                "weights": [0.25, 0.25, 0.20, 0.20, 0.10],
                "description": "Pure momentum stock identification for 7-10 day, 3 month, 9 month targets",
                "use_when": "Primary objective: momentum identification with +7%, +25%, +75% targets"
            },
            
            "multi_timeframe_momentum": {
                "models": ["VisionTransformerChart", "FractalModel", "TemporalFusionTransformer", "MACDModel", "Prophet"],
                "weights": [0.30, 0.25, 0.20, 0.15, 0.10],
                "description": "Multi-timeframe momentum analysis across 7-10 days, 3 months, 9 months",
                "use_when": "Need comprehensive timeframe analysis for momentum targets"
            },
            
            "high_confidence_momentum": {
                "models": ["AdversarialMomentumNet", "StackingEnsemble", "GraphNeuralNetwork", "XGBoost", "RSIModel"],
                "weights": [0.30, 0.25, 0.20, 0.15, 0.10],
                "description": "Maximum confidence momentum predictions (85%+ threshold)",
                "use_when": "Need 85% confidence threshold compliance for momentum targets"
            },
            
            "breakout_momentum": {
                "models": ["BollingerBreakoutModel", "VisionTransformerChart", "VolumePriceTrendModel", "AccumulationDistributionLine", "CommodityChannelIndex"],
                "weights": [0.30, 0.25, 0.20, 0.15, 0.10],
                "description": "Momentum breakout identification with volume confirmation",
                "use_when": "Identifying momentum breakouts with institutional confirmation"
            },
            
            "advanced_signal_processing": {
                "models": ["WaveletTransformModel", "EmpiricalModeDecomposition", "FractalModel", "TemporalFusionTransformer", "MultiHeadAttention"],
                "weights": [0.25, 0.25, 0.20, 0.15, 0.15],
                "description": "Advanced signal processing for complex momentum patterns",
                "use_when": "Complex market conditions requiring advanced pattern recognition"
            }
        }
    
    def _initialize_market_mappings(self) -> Dict[str, List[str]]:
        """Technical specifications for model characteristics under different market conditions"""
        return {
            "stable_conditions": ["RandomForest", "XGBoost", "LinearRegression", "Ridge", "CatBoost"],
            "trending_patterns": ["LSTM", "GRU", "ARIMA", "Prophet", "LightGBM"],
            "volatile_regimes": ["GARCH", "SVR", "RegimeAwareDQN", "CNN_LSTM", "MonteCarloSimulation"],
            "seasonal_patterns": ["Prophet", "ARIMA", "SeasonalDecomposition"],
            "complex_structures": ["Transformer", "MultiHeadAttention", "StackingEnsemble", "MetaLearner"],
            "regime_transitions": ["MarkovSwitching", "RegimeAwareDQN", "MAML"],
            "high_frequency_data": ["LightGBM", "RandomForest", "LinearRegression"],
            "risk_measurement": ["VaR", "MonteCarloSimulation", "GARCH", "BlackScholes"]
        }
    
    def _initialize_ensemble_recipes(self) -> Dict[str, Dict[str, Any]]:
        """Pre-configured ensemble recipes for different accuracy targets"""
        return {
            "stable_market_ensemble": {
                "objective": "stable market prediction",
                "models": ["RandomForest", "XGBoost", "LightGBM", "GradientBoosting"],
                "ensemble_method": "weighted_voting",
                "weights": [0.30, 0.30, 0.25, 0.15],
                "training_strategy": "cross_validation",
                "data_requirements": "medium"
            },
            
            "volatile_market_ensemble": {
                "objective": "volatile market adaptation",
                "models": ["LSTM", "GRU", "CNN_LSTM", "Transformer"],
                "ensemble_method": "dynamic_weighting",
                "weights": [0.30, 0.25, 0.25, 0.20],
                "training_strategy": "rolling_window",
                "data_requirements": "large"
            },
            
            "fast_execution_ensemble": {
                "objective": "real-time trading decisions",
                "models": ["LightGBM", "RandomForest", "GRU"],
                "ensemble_method": "simple_averaging",
                "weights": [0.40, 0.35, 0.25],
                "training_strategy": "incremental_learning",
                "data_requirements": "medium"
            }
        }
    
    def get_model_recommendations(self, market_condition: str, accuracy_target: float = 0.85, 
                                speed_priority: str = "medium") -> Dict[str, Any]:
        """Get intelligent model recommendations based on conditions"""
        
        # Get base models for market condition
        base_models = self.market_condition_mappings.get(market_condition, [])
        
        # Filter by accuracy target
        recommended_models = []
        for model in base_models:
            model_info = self.model_knowledge.get(model, {})
            accuracy_range = model_info.get("accuracy_range", [0.5, 0.6])
            if accuracy_range[1] >= accuracy_target:
                recommended_models.append(model)
        
        # Add high-accuracy models if target is high
        if accuracy_target >= 0.90:
            high_accuracy_models = ["Transformer", "StackingEnsemble", "MultiHeadAttention", "BiLSTM"]
            for model in high_accuracy_models:
                if model not in recommended_models:
                    recommended_models.append(model)
        
        # Filter by speed if needed
        if speed_priority == "high":
            fast_models = []
            for model in recommended_models:
                model_info = self.model_knowledge.get(model, {})
                if model_info.get("training_time") in ["very_fast", "fast"]:
                    fast_models.append(model)
            if fast_models:
                recommended_models = fast_models
        
        # Get ensemble strategy
        ensemble_strategy = self._get_optimal_ensemble_strategy(accuracy_target, market_condition)
        
        return {
            "recommended_models": recommended_models[:5],  # Top 5 recommendations
            "market_condition": market_condition,
            "accuracy_target": accuracy_target,
            "ensemble_strategy": ensemble_strategy,
            "model_details": {model: self.model_knowledge.get(model, {}) for model in recommended_models[:5]},
            "reasoning": self._generate_recommendation_reasoning(market_condition, accuracy_target, recommended_models[:5])
        }
    
    def get_ensemble_strategy(self, objective: str) -> Dict[str, Any]:
        """Get specific ensemble strategy"""
        return self.combination_strategies.get(objective, self.combination_strategies["robust_trading"])
    
    def get_high_accuracy_recipe(self, target_accuracy: float = 0.65) -> Dict[str, Any]:
        """Get recipe for achieving specific accuracy targets"""
        if target_accuracy >= 0.70:
            return self.ensemble_recipes["70_percent_accuracy"]
        elif target_accuracy >= 0.65:
            return self.ensemble_recipes["65_percent_accuracy"]
        else:
            return self.ensemble_recipes["65_percent_accuracy"]
    
    def _get_optimal_ensemble_strategy(self, accuracy_target: float, market_condition: str) -> str:
        """Determine optimal ensemble strategy"""
        if accuracy_target >= 0.95:
            return "advanced_stacking"
        elif accuracy_target >= 0.90:
            return "weighted_voting"
        elif market_condition in ["volatile", "complex"]:
            return "robust_ensemble"
        else:
            return "simple_voting"
    
    def _generate_recommendation_reasoning(self, market_condition: str, accuracy_target: float, 
                                         models: List[str]) -> str:
        """Generate human-readable reasoning for recommendations"""
        reasoning = f"For {market_condition} market conditions with {accuracy_target:.0%} accuracy target:\n\n"
        
        for i, model in enumerate(models, 1):
            model_info = self.model_knowledge.get(model, {})
            strength = model_info.get("strength", "General purpose")
            best_for = model_info.get("best_for", ["Various tasks"])
            
            reasoning += f"{i}. {model}: {strength}\n"
            reasoning += f"   Best for: {', '.join(best_for[:2])}\n"
        
        reasoning += f"\nThis combination provides balanced accuracy, robustness, and suitability for {market_condition} conditions."
        
        return reasoning
    
    def get_all_models_summary(self) -> Dict[str, Any]:
        """Get complete summary of all 32 models with intelligent categorization"""
        summary = {
            "total_models": len(self.model_knowledge),
            "categories": {},
            "accuracy_tiers": {
                "high_accuracy": [],  # >85%
                "medium_accuracy": [],  # 70-85%
                "baseline_accuracy": []  # <70%
            },
            "speed_tiers": {
                "very_fast": [],
                "fast": [],
                "medium": [],
                "slow": [],
                "very_slow": []
            },
            "use_case_mapping": {},
            "ensemble_ready": [],
            "interpretable": [],
            "production_ready": []
        }
        
        # Process each model
        for model_name, model_info in self.model_knowledge.items():
            category = model_info.get("category", "Unknown")
            
            # Category counts
            if category not in summary["categories"]:
                summary["categories"][category] = []
            summary["categories"][category].append(model_name)
            
            # Accuracy tiers
            accuracy_range = model_info.get("accuracy_range", [0.5, 0.6])
            max_accuracy = accuracy_range[1]
            
            if max_accuracy >= 0.85:
                summary["accuracy_tiers"]["high_accuracy"].append(model_name)
            elif max_accuracy >= 0.70:
                summary["accuracy_tiers"]["medium_accuracy"].append(model_name)
            else:
                summary["accuracy_tiers"]["baseline_accuracy"].append(model_name)
            
            # Speed tiers
            training_time = model_info.get("training_time", "medium")
            summary["speed_tiers"][training_time].append(model_name)
            
            # Use case mapping
            best_for = model_info.get("best_for", [])
            for use_case in best_for:
                if use_case not in summary["use_case_mapping"]:
                    summary["use_case_mapping"][use_case] = []
                summary["use_case_mapping"][use_case].append(model_name)
            
            # Ensemble readiness (high accuracy models)
            if max_accuracy >= 0.80:
                summary["ensemble_ready"].append(model_name)
            
            # Interpretability
            interpretability = model_info.get("interpretability", "low")
            if interpretability in ["high", "very_high"]:
                summary["interpretable"].append(model_name)
            
            # Production readiness (fast + accurate)
            if training_time in ["very_fast", "fast", "medium"] and max_accuracy >= 0.75:
                summary["production_ready"].append(model_name)
        
        return summary

# Global instance for easy access
_model_intelligence = None

def get_model_intelligence() -> ModelIntelligence:
    """Get global ModelIntelligence instance"""
    global _model_intelligence
    if _model_intelligence is None:
        _model_intelligence = ModelIntelligence()
    return _model_intelligence