"""
mlTrainer - Centralized Model Registry
=====================================

SINGLE SOURCE OF TRUTH for all 105+ ML and mathematical models.
All other modules reference this registry to ensure consistency.

This is the authoritative definition of all available models in the system.
"""

from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Centralized registry - SINGLE SOURCE OF TRUTH for all models"""
    
    def __init__(self):
        self._models = self._initialize_complete_model_registry()
        logger.info(f"ModelRegistry initialized with {len(self._models)} unique models")
    
    def _initialize_complete_model_registry(self) -> Dict[str, Dict[str, Any]]:
        """COMPLETE 105+ MODEL REGISTRY - SINGLE SOURCE OF TRUTH"""
        return {
            # ===== TIME SERIES MODELS (8) =====
            "ARIMA": {
                "category": "Time Series",
                "type": "statistical",
                "implementation": "statsmodels",
                "training_time": "medium",
                "data_requirements": "medium",
                "interpretability": "high"
            },
            "SARIMA": {
                "category": "Time Series", 
                "type": "statistical",
                "implementation": "statsmodels",
                "training_time": "medium",
                "data_requirements": "medium",
                "interpretability": "high"
            },
            "Prophet": {
                "category": "Time Series",
                "type": "bayesian",
                "implementation": "prophet",
                "training_time": "fast",
                "data_requirements": "small",
                "interpretability": "high"
            },
            "ExponentialSmoothing": {
                "category": "Time Series",
                "type": "statistical",
                "implementation": "statsmodels",
                "training_time": "fast",
                "data_requirements": "small",
                "interpretability": "high"
            },
            "RollingMeanReversion": {
                "category": "Time Series",
                "type": "statistical",
                "implementation": "custom",
                "training_time": "fast",
                "data_requirements": "small",
                "interpretability": "high"
            },
            "GARCH": {
                "category": "Time Series",
                "type": "volatility",
                "implementation": "arch",
                "training_time": "medium",
                "data_requirements": "medium",
                "interpretability": "medium"
            },
            "KalmanFilter": {
                "category": "Time Series",
                "type": "state_space",
                "implementation": "pykalman",
                "training_time": "fast",
                "data_requirements": "medium",
                "interpretability": "medium"
            },
            "SeasonalDecomposition": {
                "category": "Time Series",
                "type": "decomposition",
                "implementation": "statsmodels",
                "training_time": "fast",
                "data_requirements": "medium",
                "interpretability": "high"
            },
            
            # ===== TRADITIONAL ML MODELS (11) =====
            "RandomForest": {
                "category": "Traditional ML",
                "type": "ensemble",
                "implementation": "sklearn",
                "training_time": "fast",
                "data_requirements": "medium",
                "interpretability": "high"
            },
            "XGBoost": {
                "category": "Traditional ML",
                "type": "boosting",
                "implementation": "xgboost",
                "training_time": "medium",
                "data_requirements": "medium",
                "interpretability": "medium"
            },
            "LightGBM": {
                "category": "Traditional ML",
                "type": "boosting",
                "implementation": "lightgbm",
                "training_time": "fast",
                "data_requirements": "large",
                "interpretability": "medium"
            },
            "CatBoost": {
                "category": "Traditional ML",
                "type": "boosting",
                "implementation": "catboost",
                "training_time": "slow",
                "data_requirements": "medium",
                "interpretability": "medium"
            },
            "LogisticRegression": {
                "category": "Traditional ML",
                "type": "linear",
                "implementation": "sklearn",
                "training_time": "fast",
                "data_requirements": "small",
                "interpretability": "high"
            },
            "KNearestNeighbors": {
                "category": "Traditional ML",
                "type": "instance_based",
                "implementation": "sklearn",
                "training_time": "fast",
                "data_requirements": "medium",
                "interpretability": "medium"
            },
            "SVR": {
                "category": "Traditional ML",
                "type": "kernel",
                "implementation": "sklearn",
                "training_time": "slow",
                "data_requirements": "medium",
                "interpretability": "low"
            },
            "LinearRegression": {
                "category": "Traditional ML",
                "type": "linear",
                "implementation": "sklearn",
                "training_time": "fast",
                "data_requirements": "small",
                "interpretability": "high"
            },
            "Ridge": {
                "category": "Traditional ML",
                "type": "linear",
                "implementation": "sklearn",
                "training_time": "fast",
                "data_requirements": "small",
                "interpretability": "high"
            },
            "Lasso": {
                "category": "Traditional ML",
                "type": "linear",
                "implementation": "sklearn",
                "training_time": "fast",
                "data_requirements": "small",
                "interpretability": "high"
            },
            "ElasticNet": {
                "category": "Traditional ML",
                "type": "linear",
                "implementation": "sklearn",
                "training_time": "fast",
                "data_requirements": "small",
                "interpretability": "high"
            },
            
            # ===== DEEP LEARNING MODELS (8) =====
            "LSTM": {
                "category": "Deep Learning",
                "type": "recurrent",
                "implementation": "tensorflow",
                "training_time": "slow",
                "data_requirements": "large",
                "interpretability": "low"
            },
            "GRU": {
                "category": "Deep Learning",
                "type": "recurrent",
                "implementation": "tensorflow",
                "training_time": "medium",
                "data_requirements": "medium",
                "interpretability": "low"
            },
            "BiLSTM": {
                "category": "Deep Learning",
                "type": "recurrent",
                "implementation": "tensorflow",
                "training_time": "slow",
                "data_requirements": "large",
                "interpretability": "low"
            },
            "CNN_LSTM": {
                "category": "Deep Learning",
                "type": "hybrid",
                "implementation": "tensorflow",
                "training_time": "slow",
                "data_requirements": "large",
                "interpretability": "very_low"
            },
            "Autoencoder": {
                "category": "Deep Learning",
                "type": "unsupervised",
                "implementation": "tensorflow",
                "training_time": "medium",
                "data_requirements": "large",
                "interpretability": "low"
            },
            "Transformer": {
                "category": "Deep Learning",
                "type": "attention",
                "implementation": "tensorflow",
                "training_time": "very_slow",
                "data_requirements": "very_large",
                "interpretability": "very_low"
            },
            "TemporalFusionTransformer": {
                "category": "Deep Learning",
                "type": "attention",
                "implementation": "pytorch_forecasting",
                "training_time": "very_slow",
                "data_requirements": "very_large",
                "interpretability": "low"
            },
            "FeedforwardMLP": {
                "category": "Deep Learning",
                "type": "feedforward",
                "implementation": "tensorflow",
                "training_time": "medium",
                "data_requirements": "medium",
                "interpretability": "low"
            },
            
            # ===== REINFORCEMENT LEARNING MODELS (5) =====
            "QLearning": {
                "category": "Reinforcement Learning",
                "type": "value_based",
                "implementation": "stable_baselines3",
                "training_time": "slow",
                "data_requirements": "large",
                "interpretability": "low"
            },
            "DoubleQLearning": {
                "category": "Reinforcement Learning", 
                "type": "value_based",
                "implementation": "stable_baselines3",
                "training_time": "slow",
                "data_requirements": "large",
                "interpretability": "low"
            },
            "DuelingDQN": {
                "category": "Reinforcement Learning",
                "type": "deep_q",
                "implementation": "stable_baselines3",
                "training_time": "very_slow",
                "data_requirements": "very_large",
                "interpretability": "very_low"
            },
            "DQN": {
                "category": "Reinforcement Learning",
                "type": "deep_q",
                "implementation": "stable_baselines3",
                "training_time": "slow",
                "data_requirements": "large",
                "interpretability": "very_low"
            },
            "RegimeAwareDQN": {
                "category": "Reinforcement Learning",
                "type": "deep_q",
                "implementation": "custom",
                "training_time": "very_slow",
                "data_requirements": "very_large",
                "interpretability": "very_low"
            },
            
            # ===== ENSEMBLE & META-LEARNING (8) =====
            "StackingEnsemble": {
                "category": "Ensemble & Meta-Learning",
                "type": "stacking",
                "implementation": "sklearn",
                "training_time": "slow",
                "data_requirements": "large",
                "interpretability": "low"
            },
            "VotingClassifier": {
                "category": "Ensemble & Meta-Learning",
                "type": "voting",
                "implementation": "sklearn",
                "training_time": "medium",
                "data_requirements": "medium",
                "interpretability": "medium"
            },
            "Bagging": {
                "category": "Ensemble & Meta-Learning",
                "type": "bagging",
                "implementation": "sklearn",
                "training_time": "medium",
                "data_requirements": "medium",
                "interpretability": "medium"
            },
            "BoostedTreesEnsemble": {
                "category": "Ensemble & Meta-Learning",
                "type": "boosting",
                "implementation": "sklearn",
                "training_time": "medium",
                "data_requirements": "medium",
                "interpretability": "medium"
            },
            "MetaLearnerStrategySelector": {
                "category": "Ensemble & Meta-Learning",
                "type": "meta_learning",
                "implementation": "custom",
                "training_time": "slow",
                "data_requirements": "large",
                "interpretability": "low"
            },
            "EnsembleVoting": {
                "category": "Ensemble & Meta-Learning",
                "type": "voting",
                "implementation": "custom",
                "training_time": "medium",
                "data_requirements": "medium",
                "interpretability": "medium"
            },
            "MetaLearner": {
                "category": "Ensemble & Meta-Learning",
                "type": "meta_learning",
                "implementation": "custom",
                "training_time": "slow",
                "data_requirements": "large",
                "interpretability": "low"
            },
            "MAML": {
                "category": "Ensemble & Meta-Learning",
                "type": "meta_learning",
                "implementation": "custom",
                "training_time": "very_slow",
                "data_requirements": "very_large",
                "interpretability": "very_low"
            },
            
            # ===== REGIME DETECTION & CLUSTERING (5) =====
            "HiddenMarkovModel": {
                "category": "Regime Detection & Clustering",
                "type": "probabilistic",
                "implementation": "hmmlearn",
                "training_time": "medium",
                "data_requirements": "medium",
                "interpretability": "medium"
            },
            "KMeansClustering": {
                "category": "Regime Detection & Clustering",
                "type": "clustering",
                "implementation": "sklearn",
                "training_time": "fast",
                "data_requirements": "medium",
                "interpretability": "high"
            },
            "BayesianChangePointDetection": {
                "category": "Regime Detection & Clustering",
                "type": "bayesian",
                "implementation": "ruptures",
                "training_time": "medium",
                "data_requirements": "medium",
                "interpretability": "medium"
            },
            "RollingZScoreRegimeScorer": {
                "category": "Regime Detection & Clustering",
                "type": "statistical",
                "implementation": "custom",
                "training_time": "fast",
                "data_requirements": "small",
                "interpretability": "high"
            },
            "MarkovSwitching": {
                "category": "Regime Detection & Clustering",
                "type": "probabilistic",
                "implementation": "statsmodels",
                "training_time": "slow",
                "data_requirements": "large",
                "interpretability": "medium"
            },
            
            # ===== FORECASTING & OPTIMIZATION (7) =====
            "BayesianRidgeForecast": {
                "category": "Forecasting & Optimization",
                "type": "bayesian",
                "implementation": "sklearn",
                "training_time": "fast",
                "data_requirements": "small",
                "interpretability": "high"
            },
            "MarkowitzMeanVarianceOptimizer": {
                "category": "Forecasting & Optimization",
                "type": "optimization",
                "implementation": "pypfopt",
                "training_time": "fast",
                "data_requirements": "medium",
                "interpretability": "high"
            },
            "DynamicRiskParityModel": {
                "category": "Forecasting & Optimization",
                "type": "risk_management",
                "implementation": "custom",
                "training_time": "medium",
                "data_requirements": "medium",
                "interpretability": "medium"
            },
            "MaximumSharpeRatioOptimizer": {
                "category": "Forecasting & Optimization",
                "type": "optimization",
                "implementation": "pypfopt",
                "training_time": "fast",
                "data_requirements": "medium",
                "interpretability": "high"
            },
            "BlackScholes": {
                "category": "Forecasting & Optimization",
                "type": "financial_model",
                "implementation": "custom",
                "training_time": "fast",
                "data_requirements": "small",
                "interpretability": "high"
            },
            "MonteCarloSimulation": {
                "category": "Forecasting & Optimization",
                "type": "simulation",
                "implementation": "numpy",
                "training_time": "medium",
                "data_requirements": "small",
                "interpretability": "medium"
            },
            "VaR": {
                "category": "Forecasting & Optimization",
                "type": "risk_management",
                "implementation": "custom",
                "training_time": "fast",
                "data_requirements": "medium",
                "interpretability": "high"
            },
            
            # ===== NLP & SENTIMENT MODELS (3) =====
            "FinBERTSentimentClassifier": {
                "category": "NLP & Sentiment",
                "type": "transformer",
                "implementation": "transformers",
                "training_time": "very_slow",
                "data_requirements": "very_large",
                "interpretability": "very_low"
            },
            "BERTClassificationHead": {
                "category": "NLP & Sentiment",
                "type": "transformer",
                "implementation": "transformers",
                "training_time": "very_slow",
                "data_requirements": "very_large",
                "interpretability": "very_low"
            },
            "SentenceTransformerEmbedding": {
                "category": "NLP & Sentiment",
                "type": "embedding",
                "implementation": "sentence_transformers",
                "training_time": "slow",
                "data_requirements": "large",
                "interpretability": "low"
            },
            
            # ===== MOMENTUM-SPECIFIC MODELS (8) =====
            "RSIModel": {
                "category": "Momentum-Specific",
                "type": "technical_indicator",
                "implementation": "ta",
                "training_time": "fast",
                "data_requirements": "small",
                "interpretability": "high"
            },
            "MACDModel": {
                "category": "Momentum-Specific",
                "type": "technical_indicator",
                "implementation": "ta",
                "training_time": "fast",
                "data_requirements": "small",
                "interpretability": "high"
            },
            "BollingerBreakoutModel": {
                "category": "Momentum-Specific",
                "type": "technical_indicator",
                "implementation": "ta",
                "training_time": "fast",
                "data_requirements": "small",
                "interpretability": "high"
            },
            "VolumePriceTrendModel": {
                "category": "Momentum-Specific",
                "type": "technical_indicator",
                "implementation": "ta",
                "training_time": "fast",
                "data_requirements": "small",
                "interpretability": "high"
            },
            "WilliamsRModel": {
                "category": "Momentum-Specific",
                "type": "technical_indicator",
                "implementation": "ta",
                "training_time": "fast",
                "data_requirements": "small",
                "interpretability": "high"
            },
            "StochasticModel": {
                "category": "Momentum-Specific",
                "type": "technical_indicator",
                "implementation": "ta",
                "training_time": "fast",
                "data_requirements": "small",
                "interpretability": "high"
            },
            "CommodityChannelIndex": {
                "category": "Momentum-Specific",
                "type": "technical_indicator",
                "implementation": "ta",
                "training_time": "fast",
                "data_requirements": "small",
                "interpretability": "high"
            },
            "AccumulationDistributionLine": {
                "category": "Momentum-Specific",
                "type": "volume_indicator",
                "implementation": "ta",
                "training_time": "fast",
                "data_requirements": "small",
                "interpretability": "high"
            },
            
            # ===== CUTTING-EDGE AI MODELS (10) =====
            "VisionTransformerChart": {
                "category": "Cutting-Edge AI",
                "type": "vision_transformer",
                "implementation": "transformers",
                "training_time": "very_slow",
                "data_requirements": "very_large",
                "interpretability": "very_low"
            },
            "GraphNeuralNetwork": {
                "category": "Cutting-Edge AI",
                "type": "graph_neural",
                "implementation": "pytorch_geometric",
                "training_time": "very_slow",
                "data_requirements": "very_large",
                "interpretability": "very_low"
            },
            "AdversarialMomentumNet": {
                "category": "Cutting-Edge AI",
                "type": "adversarial",
                "implementation": "custom",
                "training_time": "very_slow",
                "data_requirements": "very_large",
                "interpretability": "very_low"
            },
            "FractalModel": {
                "category": "Cutting-Edge AI",
                "type": "mathematical",
                "implementation": "custom",
                "training_time": "medium",
                "data_requirements": "medium",
                "interpretability": "medium"
            },
            "WaveletTransformModel": {
                "category": "Cutting-Edge AI",
                "type": "signal_processing",
                "implementation": "pywt",
                "training_time": "fast",
                "data_requirements": "medium",
                "interpretability": "medium"
            },
            "EmpiricalModeDecomposition": {
                "category": "Cutting-Edge AI",
                "type": "signal_processing",
                "implementation": "emd",
                "training_time": "medium",
                "data_requirements": "medium",
                "interpretability": "medium"
            },
            "NeuralODEFinancial": {
                "category": "Cutting-Edge AI",
                "type": "neural_ode",
                "implementation": "torchdiffeq",
                "training_time": "very_slow",
                "data_requirements": "very_large",
                "interpretability": "very_low"
            },
            "ModelArchitectureSearch": {
                "category": "Cutting-Edge AI",
                "type": "automl",
                "implementation": "custom",
                "training_time": "very_slow",
                "data_requirements": "very_large",
                "interpretability": "very_low"
            },
            "HurstExponentFractal": {
                "category": "Cutting-Edge AI",
                "type": "fractal_analysis",
                "implementation": "custom",
                "training_time": "fast",
                "data_requirements": "medium",
                "interpretability": "high"
            },
            "ThresholdAutoregressive": {
                "category": "Cutting-Edge AI",
                "type": "nonlinear_time_series",
                "implementation": "custom",
                "training_time": "medium",
                "data_requirements": "medium",
                "interpretability": "medium"
            },
            
            # ===== FINANCIAL ENGINEERING (4) =====
            "KellyCriterionBayesian": {
                "category": "Financial Engineering",
                "type": "position_sizing",
                "implementation": "custom",
                "training_time": "fast",
                "data_requirements": "small",
                "interpretability": "high"
            },
            "EWMARiskMetrics": {
                "category": "Financial Engineering",
                "type": "risk_metrics",
                "implementation": "custom",
                "training_time": "fast",
                "data_requirements": "medium",
                "interpretability": "high"
            },
            "RegimeSwitchingVolatility": {
                "category": "Financial Engineering",
                "type": "volatility_modeling",
                "implementation": "custom",
                "training_time": "slow",
                "data_requirements": "large",
                "interpretability": "medium"
            },
            "ProximalPolicyOptimization": {
                "category": "Financial Engineering",
                "type": "reinforcement_learning",
                "implementation": "stable_baselines3",
                "training_time": "very_slow",
                "data_requirements": "very_large",
                "interpretability": "very_low"
            },
            
            # ===== INFORMATION THEORY (5) =====
            "TransferEntropy": {
                "category": "Information Theory",
                "type": "entropy_measure",
                "implementation": "pyinform",
                "training_time": "medium",
                "data_requirements": "medium",
                "interpretability": "medium"
            },
            "ShannonEntropyMutualInfo": {
                "category": "Information Theory",
                "type": "entropy_measure",
                "implementation": "sklearn",
                "training_time": "fast",
                "data_requirements": "medium",
                "interpretability": "medium"
            },
            "GrangerCausalityTest": {
                "category": "Information Theory",
                "type": "causality",
                "implementation": "statsmodels",
                "training_time": "fast",
                "data_requirements": "medium",
                "interpretability": "high"
            },
            "NetworkTopologyAnalysis": {
                "category": "Information Theory",
                "type": "graph_analysis",
                "implementation": "networkx",
                "training_time": "medium",
                "data_requirements": "medium",
                "interpretability": "medium"
            },
            "LempelZivComplexity": {
                "category": "Information Theory",
                "type": "complexity_measure",
                "implementation": "custom",
                "training_time": "fast",
                "data_requirements": "small",
                "interpretability": "medium"
            },
            
            # ===== RISK ANALYTICS (4) =====
            "OmegaRatio": {
                "category": "Risk Analytics",
                "type": "risk_measure",
                "implementation": "custom",
                "training_time": "fast",
                "data_requirements": "small",
                "interpretability": "high"
            },
            "SterlingRatio": {
                "category": "Risk Analytics",
                "type": "risk_measure",
                "implementation": "custom",
                "training_time": "fast",
                "data_requirements": "small",
                "interpretability": "high"
            },
            "InformationRatio": {
                "category": "Risk Analytics",
                "type": "risk_measure",
                "implementation": "custom",
                "training_time": "fast",
                "data_requirements": "small",
                "interpretability": "high"
            },
            "MarketStressIndicators": {
                "category": "Risk Analytics",
                "type": "stress_testing",
                "implementation": "custom",
                "training_time": "medium",
                "data_requirements": "medium",
                "interpretability": "high"
            },
            
            # ===== MARKET MICROSTRUCTURE (4) =====
            "MarketImpactModels": {
                "category": "Market Microstructure",
                "type": "impact_modeling",
                "implementation": "custom",
                "training_time": "medium",
                "data_requirements": "large",
                "interpretability": "medium"
            },
            "OrderFlowAnalysis": {
                "category": "Market Microstructure",
                "type": "flow_analysis",
                "implementation": "custom",
                "training_time": "fast",
                "data_requirements": "large",
                "interpretability": "medium"
            },
            "BidAskSpreadAnalysis": {
                "category": "Market Microstructure",
                "type": "spread_analysis",
                "implementation": "custom",
                "training_time": "fast",
                "data_requirements": "medium",
                "interpretability": "high"
            },
            "LiquidityAssessment": {
                "category": "Market Microstructure",
                "type": "liquidity_measure",
                "implementation": "custom",
                "training_time": "fast",
                "data_requirements": "medium",
                "interpretability": "high"
            },
            
            # ===== SYSTEM INTELLIGENCE (5) =====
            "AutomatedFeatureEngineering": {
                "category": "System Intelligence",
                "type": "feature_engineering",
                "implementation": "featuretools",
                "training_time": "slow",
                "data_requirements": "large",
                "interpretability": "low"
            },
            "HyperparameterEvolution": {
                "category": "System Intelligence",
                "type": "optimization",
                "implementation": "optuna",
                "training_time": "very_slow",
                "data_requirements": "large",
                "interpretability": "medium"
            },
            "CausalInferenceDiscovery": {
                "category": "System Intelligence",
                "type": "causal_inference",
                "implementation": "dowhy",
                "training_time": "slow",
                "data_requirements": "large",
                "interpretability": "medium"
            },
            "OnlineLearningUpdates": {
                "category": "System Intelligence",
                "type": "online_learning",
                "implementation": "river",
                "training_time": "fast",
                "data_requirements": "streaming",
                "interpretability": "medium"
            },
            "ConceptDriftDetection": {
                "category": "System Intelligence",
                "type": "drift_detection",
                "implementation": "river",
                "training_time": "fast",
                "data_requirements": "streaming",
                "interpretability": "high"
            },
            
            # ===== OPTIMIZATION (3) =====
            "WalkForwardOptimization": {
                "category": "Optimization",
                "type": "time_series_optimization",
                "implementation": "custom",
                "training_time": "slow",
                "data_requirements": "large",
                "interpretability": "medium"
            },
            "BayesianParameterOptimization": {
                "category": "Optimization",
                "type": "bayesian_optimization",
                "implementation": "optuna",
                "training_time": "slow",
                "data_requirements": "medium",
                "interpretability": "medium"
            },
            "FederatedLearning": {
                "category": "Optimization",
                "type": "distributed_learning",
                "implementation": "flower",
                "training_time": "very_slow",
                "data_requirements": "very_large",
                "interpretability": "low"
            },
            
            # ===== MACRO ANALYSIS (3) =====
            "YieldCurveAnalysis": {
                "category": "Macro Analysis",
                "type": "curve_analysis",
                "implementation": "custom",
                "training_time": "fast",
                "data_requirements": "small",
                "interpretability": "high"
            },
            "SectorRotationAnalysis": {
                "category": "Macro Analysis",
                "type": "sector_analysis",
                "implementation": "custom",
                "training_time": "medium",
                "data_requirements": "medium",
                "interpretability": "high"
            },
            "ADFKPSSTests": {
                "category": "Macro Analysis",
                "type": "stationarity_test",
                "implementation": "statsmodels",
                "training_time": "fast",
                "data_requirements": "small",
                "interpretability": "high"
            },
            
            # ===== ADVANCED MODELS (3) =====
            "MultiHeadAttention": {
                "category": "Advanced Models",
                "type": "attention_mechanism",
                "implementation": "tensorflow",
                "training_time": "slow",
                "data_requirements": "large",
                "interpretability": "very_low"
            },
            "GradientBoosting": {
                "category": "Advanced Models",
                "type": "boosting",
                "implementation": "sklearn",
                "training_time": "medium",
                "data_requirements": "medium",
                "interpretability": "medium"
            },
            "DecisionTree": {
                "category": "Advanced Models",
                "type": "tree",
                "implementation": "sklearn",
                "training_time": "fast",
                "data_requirements": "small",
                "interpretability": "high"
            },
            "CCIEnsemble": {
                "category": "Momentum-Specific",
                "type": "ensemble",
                "implementation": "custom",
                "training_time": "fast",
                "data_requirements": "small",
                "interpretability": "high"
            },
            "EMAModel": {
                "category": "Momentum-Specific",
                "type": "technical_indicator",
                "implementation": "custom",
                "training_time": "fast",
                "data_requirements": "small",
                "interpretability": "high"
            },
            "ROCModel": {
                "category": "Momentum-Specific", 
                "type": "technical_indicator",
                "implementation": "custom",
                "training_time": "fast",
                "data_requirements": "small",
                "interpretability": "high"
            },
            "ParabolicSARModel": {
                "category": "Momentum-Specific",
                "type": "technical_indicator", 
                "implementation": "custom",
                "training_time": "fast",
                "data_requirements": "small",
                "interpretability": "high"
            },
            "OBVModel": {
                "category": "Volume Analysis",
                "type": "volume_indicator",
                "implementation": "custom", 
                "training_time": "fast",
                "data_requirements": "small",
                "interpretability": "high"
            },
            "VolumeSpikeModel": {
                "category": "Volume Analysis",
                "type": "volume_indicator",
                "implementation": "custom",
                "training_time": "fast",
                "data_requirements": "small", 
                "interpretability": "high"
            },
            "VolumePriceAnalysisModel": {
                "category": "Volume Analysis",
                "type": "volume_indicator",
                "implementation": "custom",
                "training_time": "fast",
                "data_requirements": "medium",
                "interpretability": "high"
            },
            "BreakoutDetectionModel": {
                "category": "Pattern Recognition",
                "type": "pattern_detector",
                "implementation": "custom",
                "training_time": "medium",
                "data_requirements": "medium",
                "interpretability": "high"
            },
            "SupportResistanceModel": {
                "category": "Pattern Recognition",
                "type": "pattern_detector", 
                "implementation": "custom",
                "training_time": "medium",
                "data_requirements": "medium",
                "interpretability": "high"
            },
            "CandlestickPatternModel": {
                "category": "Pattern Recognition",
                "type": "pattern_detector",
                "implementation": "custom",
                "training_time": "fast",
                "data_requirements": "small",
                "interpretability": "high"
            },
            "HighTightFlagModel": {
                "category": "Pattern Recognition",
                "type": "breakout_pattern",
                "implementation": "custom",
                "training_time": "medium",
                "data_requirements": "medium",
                "interpretability": "high"
            },
            "EMACrossoverModel": {
                "category": "Momentum-Specific",
                "type": "crossover_system",
                "implementation": "custom",
                "training_time": "fast",
                "data_requirements": "small",
                "interpretability": "high"
            },
            "MomentumBreakoutModel": {
                "category": "Momentum-Specific",
                "type": "momentum_system",
                "implementation": "custom",
                "training_time": "medium",
                "data_requirements": "medium",
                "interpretability": "high"
            },
            "TrendReversalModel": {
                "category": "Momentum-Specific",
                "type": "reversal_detector",
                "implementation": "custom",
                "training_time": "medium",
                "data_requirements": "medium",
                "interpretability": "high"
            },
            "VolumeConfirmedBreakoutModel": {
                "category": "Volume Analysis",
                "type": "volume_breakout",
                "implementation": "custom",
                "training_time": "medium",
                "data_requirements": "medium",
                "interpretability": "high"
            },
            "VPAModel": {
                "category": "Volume Analysis",
                "type": "volume_price_analysis",
                "implementation": "custom",
                "training_time": "medium",
                "data_requirements": "medium",
                "interpretability": "high"
            }
        }
    
    def get_all_models(self) -> List[str]:
        """Get complete list of all model names"""
        return list(self._models.keys())
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get complete information for a specific model"""
        return self._models.get(model_name, {})
    
    def get_models_by_category(self, category: str) -> List[str]:
        """Get all models in a specific category"""
        return [name for name, info in self._models.items() 
                if info.get("category") == category]
    
    def get_categories(self) -> List[str]:
        """Get all available categories"""
        categories = set(info.get("category", "Unknown") for info in self._models.values())
        return sorted(list(categories))
    
    def get_model_count(self) -> int:
        """Get total number of models"""
        return len(self._models)
    
    def get_category_counts(self) -> Dict[str, int]:
        """Get count of models per category"""
        counts = {}
        for info in self._models.values():
            category = info.get("category", "Unknown")
            counts[category] = counts.get(category, 0) + 1
        return counts


# Global singleton instance
_model_registry = None

def get_model_registry() -> ModelRegistry:
    """Get the global model registry instance"""
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry()
    return _model_registry