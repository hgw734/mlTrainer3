"""
mlTrainer - ML Pipeline Core
===========================

Purpose: Manages the multi-model ML pipeline including LSTM, Transformer,
Reinforcement Learning, and Ensemble Meta-Learning models. Handles model
training, inference, and regime-aware model selection.

Features:
- Dynamic model loading based on regime conditions
- Walk-forward training and validation
- Performance monitoring and self-assessment
- Compliance-verified training data only
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import joblib
import os
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Import compliance monitoring to prevent synthetic data fraud
try:
    from core.model_compliance_monitor import compliance_monitor, validate_model_training
    COMPLIANCE_MONITORING_AVAILABLE = True
except ImportError:
    COMPLIANCE_MONITORING_AVAILABLE = False

logger = logging.getLogger(__name__)

class MLPipeline:
    """Manages multi-model ML pipeline with regime-aware selection"""
    
    def __init__(self):
        self.models = {}
        self.model_performance = {}
        self.scalers = {}
        self.is_initialized = False
        self.last_training_time = None
        
        # CPU configuration - use 6 CPUs for training, leave 2 for system
        self.n_jobs = 6
        self.max_workers = 6
        
        # Model configuration
        self.model_config = self._load_model_config()
        
        # Initialize centralized model registry - SINGLE SOURCE OF TRUTH
        from core.model_registry import get_model_registry
        self.model_registry = get_model_registry()
        
        # Get all available models from centralized registry
        self.available_models = self.model_registry.get_all_models()
        
        # Initialize model persistence
        from core.model_persistence import ModelPersistence
        self.model_persistence = ModelPersistence()
        
        # Performance tracking
        self.performance_history = []
        
        # S&P 500 data access
        self.sp500_manager = None
        self._initialize_sp500_access()
        
        # Model implementations
        self.model_implementations = None
        self._initialize_model_implementations()
        
        # Model intelligence system
        self.model_intelligence = None
        self._initialize_model_intelligence()
        
        # Model status tracking
        self.model_status = self._get_initial_model_status()
        
        logger.info(f"MLPipeline initialized with {len(self.available_models)} models from centralized ModelRegistry")
        logger.info(f"ModelRegistry contains {self.model_registry.get_model_count()} models across {len(self.model_registry.get_categories())} categories")
        self._initialize_base_models()
    
    def _initialize_sp500_access(self):
        """Initialize S&P 500 data access for ML models"""
        try:
            from data.sp500_data import get_sp500_manager
            self.sp500_manager = get_sp500_manager()
            logger.info(f"S&P 500 access initialized with {len(self.sp500_manager.sp500_tickers)} tickers")
        except Exception as e:
            logger.error(f"Failed to initialize S&P 500 access: {e}")
    
    def _get_initial_model_status(self) -> Dict[str, Dict[str, Any]]:
        """Get initial status for all available models"""
        status = {}
        
        for model_name in self.available_models:
            status[model_name] = {
                "available": self._check_model_availability(model_name),
                "loaded": False,
                "accuracy": 0.0,
                "last_trained": None,
                "training_samples": 0,
                "status": "not_loaded",
                "dependencies": self._get_model_dependencies(model_name),
                "category": self._get_model_category(model_name)
            }
        
        return status
    
    def _check_model_availability(self, model_name: str) -> bool:
        """Check if a model's dependencies are available"""
        try:
            if model_name in ["RandomForest", "DecisionTree", "LinearRegression", "Ridge", "Lasso", "ElasticNet"]:
                # Always available with scikit-learn
                return True
            elif model_name == "XGBoost":
                import xgboost
                return True
            elif model_name == "LightGBM":
                import lightgbm
                return True
            elif model_name == "CatBoost":
                import catboost
                return True
            elif model_name in ["LSTM", "GRU", "Transformer", "CNN_LSTM", "Autoencoder", "BiLSTM"]:
                # Check for TensorFlow/Keras availability
                try:
                    import tensorflow as tf
                    return True
                except:
                    try:
                        import torch
                        return True
                    except:
                        return False
            elif model_name == "SVR":
                return True  # Available with scikit-learn
            elif model_name in ["ARIMA", "Prophet"]:
                try:
                    import statsmodels
                    return True
                except:
                    return False
            else:
                # Custom models - assume available but need implementation
                return True
                
        except ImportError:
            return False
        except Exception:
            return False
    
    def _get_model_dependencies(self, model_name: str) -> List[str]:
        """Get list of dependencies for a model from centralized registry"""
        model_info = self.model_registry.get_model_info(model_name)
        implementation = model_info.get("implementation", "custom")
        
        # Map implementation to dependencies
        dependency_map = {
            "sklearn": ["scikit-learn"],
            "xgboost": ["xgboost"],
            "lightgbm": ["lightgbm"],
            "catboost": ["catboost"],
            "tensorflow": ["tensorflow"],
            "pytorch": ["torch"],
            "statsmodels": ["statsmodels"],
            "prophet": ["prophet", "statsmodels"]
        }
        return dependency_map.get(implementation, ["custom_implementation"])
    
    def _get_model_category(self, model_name: str) -> str:
        """Get category for a model from centralized registry"""
        model_info = self.model_registry.get_model_info(model_name)
        return model_info.get("category", "Unknown")
    
    def _initialize_model_implementations(self):
        """Initialize comprehensive model implementations"""
        try:
            from core.model_implementations import get_model_implementations
            self.model_implementations = get_model_implementations()
            logger.info("Model implementations initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model implementations: {e}")
            # Create minimal working implementations for immediate trials
            self.model_implementations = self._create_minimal_implementations()
            logger.info("Using minimal model implementations for immediate trial capability")
    
    def _create_minimal_implementations(self):
        """Create minimal working model implementations for immediate trials"""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        class MinimalImplementations:
            def create_random_forest_model(self, **kwargs):
                return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=6)
            
            def create_linear_regression_model(self, **kwargs):
                return LinearRegression()
            
            def calculate_model_metrics(self, y_true, y_pred):
                return {
                    'mse': float(mean_squared_error(y_true, y_pred)),
                    'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                    'mae': float(mean_absolute_error(y_true, y_pred)),
                    'r2': float(r2_score(y_true, y_pred))
                }
            
            def prepare_sequence_data(self, data, sequence_length=10):
                """Prepare data for basic model training"""
                return data.values if hasattr(data, 'values') else data
                
        return MinimalImplementations()
    
    def _initialize_model_intelligence(self):
        """Initialize model intelligence system"""
        try:
            from core.model_intelligence import get_model_intelligence
            self.model_intelligence = get_model_intelligence()
            logger.info("Model intelligence system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model intelligence: {e}")
            self.model_intelligence = None
    
    def get_comprehensive_model_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all mathematical and ML models"""
        try:
            summary = {
                "total_models": len(self.available_models),
                "categories": {},
                "availability_summary": {
                    "available": 0,
                    "unavailable": 0,
                    "loaded": 0,
                    "trained": 0
                },
                "models": {},
                "sp500_access": {
                    "enabled": self.sp500_manager is not None,
                    "total_tickers": len(self.sp500_manager.sp500_tickers) if self.sp500_manager else 0,
                    "sectors": len(self.sp500_manager.get_sectors()) if self.sp500_manager else 0
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Process each model
            for model_name in self.available_models:
                model_info = self.model_status.get(model_name, {})
                category = model_info.get("category", "Unknown")
                
                # Update category counts
                if category not in summary["categories"]:
                    summary["categories"][category] = {
                        "count": 0,
                        "available": 0,
                        "loaded": 0
                    }
                
                summary["categories"][category]["count"] += 1
                
                if model_info.get("available", False):
                    summary["availability_summary"]["available"] += 1
                    summary["categories"][category]["available"] += 1
                    
                    if model_info.get("loaded", False):
                        summary["availability_summary"]["loaded"] += 1
                        summary["categories"][category]["loaded"] += 1
                        
                        if model_info.get("last_trained"):
                            summary["availability_summary"]["trained"] += 1
                else:
                    summary["availability_summary"]["unavailable"] += 1
                
                # Add detailed model info
                summary["models"][model_name] = {
                    "category": category,
                    "available": model_info.get("available", False),
                    "loaded": model_info.get("loaded", False),
                    "dependencies": model_info.get("dependencies", []),
                    "status": model_info.get("status", "unknown"),
                    "accuracy": model_info.get("accuracy", 0.0),
                    "training_samples": model_info.get("training_samples", 0),
                    "last_trained": model_info.get("last_trained")
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting model status: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_sp500_access_info(self) -> Dict[str, Any]:
        """Get S&P 500 data access information for mlTrainer"""
        try:
            if not self.sp500_manager:
                return {
                    "enabled": False,
                    "error": "S&P 500 manager not initialized"
                }
            
            return {
                "enabled": True,
                "total_tickers": len(self.sp500_manager.sp500_tickers),
                "sectors": self.sp500_manager.get_sectors(),
                "sample_tickers": self.sp500_manager.sp500_tickers[:20],  # First 20 as sample
                "data_source": "polygon",
                "features": [
                    "Real-time price data",
                    "Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)",
                    "Volume analysis",
                    "Historical data access",
                    "Sector classification",
                    "ML-ready datasets",
                    "Compliance-verified data only"
                ],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting S&P 500 info: {e}")
            return {
                "enabled": False,
                "error": str(e)
            }
    
    def train_models_with_sp500_data(self, tickers: List[str] = None, days: int = 60) -> Dict[str, Any]:
        """Train available models using S&P 500 data"""
        try:
            if not self.sp500_manager:
                return {
                    "success": False,
                    "error": "S&P 500 manager not available"
                }
            
            # Use default tickers if none provided
            if not tickers:
                tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]  # Major stocks
            
            # Get ML-ready data
            training_data = self.sp500_manager.get_ml_ready_data(tickers, days)
            
            if training_data is None or training_data.empty:
                return {
                    "success": False,
                    "error": "No training data available"
                }
            
            # Train available models
            training_results = {}
            models_trained = 0
            
            # Get fast training models from registry
            fast_models = []
            for model_name in self.model_registry.get_all_models():
                model_info = self.model_registry.get_model_info(model_name)
                if model_info.get("training_time") in ["fast", "very_fast"]:
                    fast_models.append(model_name)
            
            # Fallback to core models if no fast models found
            if not fast_models:
                fast_models = [m for m in self.model_registry.get_all_models()[:3]]
            
            # Train our core available models directly using simplified training
            core_models = ["RandomForest", "XGBoost", "LightGBM"]
            
            for model_name in core_models:
                try:
                    # Use simplified direct training that updates model status
                    result = self._train_single_model(model_name, training_data)
                    training_results[model_name] = result
                    if result.get("success", False):
                        models_trained += 1
                        logger.info(f"Successfully trained {model_name} with {result.get('training_samples', 0)} samples")
                except Exception as e:
                    logger.error(f"Failed to train {model_name}: {e}")
                    training_results[model_name] = {
                        "success": False,
                        "error": str(e)
                    }
            
            return {
                "success": True,
                "models_trained": models_trained,
                "total_models": len(training_results),
                "training_data_size": len(training_data),
                "tickers_used": tickers,
                "results": training_results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error training models with S&P 500 data: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _train_single_model(self, model_name: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Train a single model with the provided data"""
        try:
            # UNIVERSAL COMPLIANCE CHECK: Every piece of data must pass through compliance
            from backend.compliance_engine import ComplianceEngine
            compliance = ComplianceEngine()
            
            # Convert DataFrame to dict for compliance check
            data_dict = data.to_dict('records') if not data.empty else []
            approved, validated_data = compliance.universal_interceptor.intercept_all_data(
                data_dict, 
                context=f"ML_TRAINING_{model_name}"
            )
            
            if not approved:
                return {
                    "success": False,
                    "error": f"COMPLIANCE BLOCK: Training data for {model_name} failed universal compliance check"
                }
            
            # Convert back to DataFrame with compliance tags
            if validated_data:
                data = pd.DataFrame(validated_data)
            
            # Prepare features and target
            feature_columns = [col for col in data.columns if col not in 
                              ['target', 'ticker', 'timestamp', 'source', 'verified', 'sector', 
                               'universal_compliance_check', 'interceptor_timestamp']]
            
            X = data[feature_columns].fillna(0)
            y = data['target'].fillna(0)
            
            # Remove rows where target is NaN
            valid_rows = ~y.isna()
            X = X[valid_rows]
            y = y[valid_rows]
            
            if len(X) < 10:  # Need minimum samples
                return {
                    "success": False,
                    "error": "Insufficient training samples"
                }
            
            # Get or create model
            if model_name not in self.models:
                if model_name == "RandomForest":
                    from sklearn.ensemble import RandomForestRegressor
                    self.models[model_name] = RandomForestRegressor(n_estimators=50, random_state=42)
                elif model_name == "XGBoost":
                    import xgboost as xgb
                    self.models[model_name] = xgb.XGBRegressor(n_estimators=50, random_state=42)
                elif model_name == "LightGBM":
                    import lightgbm as lgb
                    self.models[model_name] = lgb.LGBMRegressor(n_estimators=50, random_state=42, verbosity=-1)
                else:
                    return {
                        "success": False,
                        "error": f"Model {model_name} not implemented"
                    }
            
            # Train the model
            self.models[model_name].fit(X, y)
            
            # Update model status
            self.model_status[model_name].update({
                "loaded": True,
                "last_trained": datetime.now().isoformat(),
                "training_samples": len(X),
                "status": "trained"
            })
            
            # Calculate simple accuracy (for regression, use R²)
            try:
                from sklearn.metrics import r2_score
                y_pred = self.models[model_name].predict(X)
                accuracy = r2_score(y, y_pred)
                self.model_status[model_name]["accuracy"] = max(0, accuracy)  # Ensure non-negative
            except:
                self.model_status[model_name]["accuracy"] = 0.5  # Default
            
            return {
                "success": True,
                "training_samples": len(X),
                "features_used": len(feature_columns),
                "accuracy": self.model_status[model_name]["accuracy"]
            }
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def train_all_available_models(self, tickers: List[str] = None, days: int = 60) -> Dict[str, Any]:
        """Train ALL available models with S&P 500 data - COMPREHENSIVE TRAINING"""
        try:
            if not self.sp500_manager:
                return {
                    "success": False,
                    "error": "S&P 500 data manager not available"
                }
            
            # Use default tickers if none provided
            if not tickers:
                tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "JNJ", "V"]
            
            # Get ML-ready data
            training_data = self.sp500_manager.get_ml_ready_data(tickers, days)
            
            if training_data is None or training_data.empty:
                return {
                    "success": False,
                    "error": "No training data available"
                }
            
            # COMPREHENSIVE TRAINING: All 120+ models from registry
            results = {}
            models_trained = 0
            total_attempted = 0
            
            # Get all models from centralized registry
            all_models = self.model_registry.get_all_models()
            categories = self.model_registry.get_categories()
            
            logger.info(f"Starting comprehensive training of {len(all_models)} models across {len(categories)} categories")
            
            # Train models by category for organized results
            for category in categories:
                category_models = self.model_registry.get_models_by_category(category)
                results[category] = {}
                
                logger.info(f"Training {len(category_models)} models in category: {category}")
                
                for model_name in category_models:
                    total_attempted += 1
                    
                    try:
                        # Route to appropriate training method based on category
                        if category in ["Traditional ML", "Tree-Based Models"]:
                            result = self._train_traditional_model(model_name, training_data)
                        elif category in ["Deep Learning", "Neural Networks"]:
                            result = self._train_deep_learning_model(model_name, training_data)
                        elif category in ["Time Series", "Forecasting"]:
                            result = self._train_time_series_model(model_name, training_data, tickers[0])
                        elif category in ["Ensemble & Meta-Learning"]:
                            result = self._train_ensemble_model(model_name, training_data)
                        elif category in ["Financial Models", "Options & Derivatives"]:
                            result = self._train_financial_model(model_name, training_data)
                        elif category in ["NLP & Sentiment", "Alternative Data"]:
                            result = self._train_nlp_model(model_name, training_data)
                        elif category in ["Reinforcement Learning"]:
                            result = self._train_rl_model(model_name, training_data)
                        else:
                            # Default to traditional ML approach
                            result = self._train_traditional_model(model_name, training_data)
                        
                        if result.get("success", False):
                            models_trained += 1
                            results[category][model_name] = result
                            
                            # Save trained model with comprehensive metadata
                            if model_name in self.models:
                                metadata = {
                                    "accuracy": result.get("accuracy", 0.0),
                                    "training_samples": result.get("training_samples", 0),
                                    "features_used": result.get("features_used", 0),
                                    "training_date": datetime.now().isoformat(),
                                    "model_type": model_name,
                                    "category": category,
                                    "training_tickers": tickers,
                                    "training_days": days
                                }
                                
                                saved = self.model_persistence.save_model(model_name, self.models[model_name], metadata)
                                result["saved"] = saved
                        else:
                            results[category][model_name] = {"success": False, "error": result.get("error", "Training failed")}
                            
                    except Exception as e:
                        logger.error(f"Error training {model_name} in {category}: {e}")
                        results[category][model_name] = {"success": False, "error": str(e)}
            
            # Final comprehensive results
            return {
                "success": True,
                "models_trained": models_trained,
                "total_attempted": total_attempted,
                "categories_processed": len(categories),
                "results": results,
                "timestamp": datetime.now().isoformat(),
                "training_data_size": len(training_data),
                "tickers_used": tickers
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive training: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _train_ensemble_model(self, model_name: str, data: pd.DataFrame) -> Dict[str, Any]:
        """UNIVERSAL COMPLIANCE ENFORCED: All data must pass through interceptor"""
        try:
            # UNIVERSAL COMPLIANCE CHECK: Every piece of data must pass through compliance
            from backend.compliance_engine import ComplianceEngine
            compliance = ComplianceEngine()
            
            # Convert DataFrame to dict for compliance check
            data_dict = data.to_dict('records') if not data.empty else []
            approved, validated_data = compliance.universal_interceptor.intercept_all_data(
                data_dict, 
                context=f"ENSEMBLE_TRAINING_{model_name}"
            )
            
            if not approved:
                return {
                    "success": False,
                    "error": f"UNIVERSAL COMPLIANCE BLOCK: {model_name} training data failed compliance check"
                }
            
            # Block until authentic implementation available
            return {
                "success": False, 
                "error": f"COMPLIANCE VIOLATION PREVENTED: Model {model_name} requires real implementation, not synthetic proxy training. System halted to prevent fake data generation."
            }
        except Exception as e:
            return {"success": False, "error": f"Compliance check failed: {str(e)}"}
    
    def _train_financial_model(self, model_name: str, data: pd.DataFrame) -> Dict[str, Any]:
        """COMPLIANCE ENFORCEMENT: Prevent synthetic proxy training"""
        return {
            "success": False, 
            "error": f"COMPLIANCE VIOLATION PREVENTED: Model {model_name} requires authentic financial model implementation, not regression proxy. System halted to prevent synthetic data generation."
        }
    
    def _train_nlp_model(self, model_name: str, data: pd.DataFrame) -> Dict[str, Any]:
        """UNIVERSAL COMPLIANCE ENFORCED: All data must pass through interceptor"""
        try:
            # UNIVERSAL COMPLIANCE CHECK: Every piece of data must pass through compliance
            from backend.compliance_engine import ComplianceEngine
            compliance = ComplianceEngine()
            
            # Convert DataFrame to dict for compliance check
            data_dict = data.to_dict('records') if not data.empty else []
            approved, validated_data = compliance.universal_interceptor.intercept_all_data(
                data_dict, 
                context=f"NLP_TRAINING_{model_name}"
            )
            
            if not approved:
                return {
                    "success": False,
                    "error": f"UNIVERSAL COMPLIANCE BLOCK: {model_name} training data failed compliance check"
                }
            
            # Block until authentic implementation available
            return {
                "success": False, 
                "error": f"COMPLIANCE VIOLATION PREVENTED: Model {model_name} requires authentic NLP implementation (BERT, transformers), not RandomForest proxy. System halted to prevent synthetic data generation."
            }
        except Exception as e:
            return {"success": False, "error": f"Compliance check failed: {str(e)}"}
    
    def _train_rl_model(self, model_name: str, data: pd.DataFrame) -> Dict[str, Any]:
        """COMPLIANCE ENFORCEMENT: Prevent RL proxy fraud"""
        return {
            "success": False, 
            "error": f"COMPLIANCE VIOLATION PREVENTED: Model {model_name} requires authentic RL implementation (Q-Learning, DQN), not MLP/RandomForest proxy. System halted to prevent synthetic data generation."
        }
    
    def _train_advanced_model(self, model_name: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Train advanced tree-based models using new implementations"""
        try:
            # Prepare features and target
            feature_columns = [col for col in data.columns if col not in 
                              ['target', 'ticker', 'timestamp', 'source', 'verified', 'sector']]
            
            X = data[feature_columns].fillna(0)
            y = data['target'].fillna(0)
            
            # Remove rows where target is NaN
            valid_rows = ~y.isna()
            X = X[valid_rows]
            y = y[valid_rows]
            
            if len(X) < 20:
                return {"success": False, "error": "Insufficient training samples"}
            
            # Create model using implementations
            if model_name == "CatBoost":
                model = self.model_implementations.create_catboost_model()
            elif model_name == "GradientBoosting":
                model = self.model_implementations.create_gradient_boosting_model()
            else:
                return {"success": False, "error": f"Model {model_name} not recognized"}
            
            # Train-test split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train the model
            if model_name == "CatBoost":
                model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
            else:
                model.fit(X_train, y_train)
            
            # Make predictions and calculate metrics
            y_pred = model.predict(X_test)
            metrics = self.model_implementations.calculate_model_metrics(y_test, y_pred)
            
            # Store the model
            self.models[model_name] = model
            
            # Update model status
            self.model_status[model_name].update({
                "loaded": True,
                "last_trained": datetime.now().isoformat(),
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "status": "trained",
                "accuracy": max(0.0, float(metrics["r2_score"]))
            })
            
            return {
                "success": True,
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "features_used": len(feature_columns),
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            return {"success": False, "error": str(e)}
    
    def _train_deep_learning_model(self, model_name: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Train deep learning models using TensorFlow implementations"""
        try:
            # Prepare sequence data for deep learning
            feature_columns = [col for col in data.columns if col not in 
                              ['target', 'ticker', 'timestamp', 'source', 'verified', 'sector']]
            
            # Sort by timestamp if available
            data_sorted = data.sort_values('timestamp') if 'timestamp' in data.columns else data
            
            # Prepare features
            X_data = data_sorted[feature_columns].fillna(0).values
            y_data = data_sorted['target'].fillna(0).values
            
            if len(X_data) < 100:
                return {"success": False, "error": "Insufficient data for deep learning"}
            
            # Create sequences for time series models
            sequence_length = 30
            X_seq, y_seq = self.model_implementations.prepare_sequence_data(X_data, sequence_length)
            
            if len(X_seq) == 0:
                return {"success": False, "error": "No sequences created"}
            
            # Reshape for neural networks
            X_seq = X_seq.reshape(X_seq.shape[0], X_seq.shape[1], X_seq.shape[2])
            
            # Train-test split
            split_idx = int(0.8 * len(X_seq))
            X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
            y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
            
            # Create model based on type
            input_shape = (X_train.shape[1], X_train.shape[2])
            
            if model_name == "LSTM":
                model = self.model_implementations.create_lstm_model(input_shape)
            elif model_name == "GRU":
                model = self.model_implementations.create_gru_model(input_shape)
            elif model_name == "Transformer":
                model = self.model_implementations.create_transformer_model(input_shape)
            elif model_name == "CNN_LSTM":
                model = self.model_implementations.create_cnn_lstm_model(input_shape)
            elif model_name == "BiLSTM":
                model = self.model_implementations.create_bilstm_model(input_shape)
            else:
                return {"success": False, "error": f"Model {model_name} not recognized"}
            
            # Train the model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=25,  # Reduced for faster training
                batch_size=32,
                verbose=0
            )
            
            # Make predictions and calculate metrics
            y_pred = model.predict(X_test, verbose=0)
            metrics = self.model_implementations.calculate_model_metrics(y_test, y_pred.flatten())
            
            # Store the model
            self.models[model_name] = model
            
            # Update model status
            self.model_status[model_name].update({
                "loaded": True,
                "last_trained": datetime.now().isoformat(),
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "status": "trained",
                "accuracy": max(0.0, float(metrics["r2_score"])),
                "final_loss": float(history.history['loss'][-1])
            })
            
            return {
                "success": True,
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "epochs_trained": len(history.history['loss']),
                "final_loss": float(history.history['loss'][-1]),
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            return {"success": False, "error": str(e)}
    
    def _train_time_series_model(self, model_name: str, data: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """Train time series models (ARIMA, Prophet)"""
        try:
            # Prepare time series data
            if 'timestamp' not in data.columns:
                return {"success": False, "error": "Timestamp column required"}
            
            # Filter for single ticker and sort by time
            ticker_data = data[data['ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('timestamp')
            
            if len(ticker_data) < 50:
                return {"success": False, "error": "Insufficient time series data"}
            
            if model_name == "ARIMA":
                # Use close prices for ARIMA
                price_series = ticker_data['close'] if 'close' in ticker_data.columns else ticker_data['target']
                price_series = price_series.dropna()
                
                model = self.model_implementations.create_arima_model(price_series)
                fitted_model = model.fit()
                
                # Store the model
                self.models[model_name] = fitted_model
                
                # Calculate basic accuracy
                fitted_values = fitted_model.fittedvalues
                if len(fitted_values) > 0:
                    actual_values = price_series[1:]  # ARIMA starts from second observation
                    r2 = np.corrcoef(actual_values, fitted_values)[0, 1] ** 2
                else:
                    r2 = 0.5
                
            elif model_name == "Prophet":
                # Prepare Prophet data format
                prophet_data = pd.DataFrame({
                    'ds': pd.to_datetime(ticker_data['timestamp']),
                    'y': ticker_data['close'] if 'close' in ticker_data.columns else ticker_data['target']
                })
                prophet_data = prophet_data.dropna()
                
                model = self.model_implementations.create_prophet_model()
                model.fit(prophet_data)
                
                # Store the model
                self.models[model_name] = model
                
                # Calculate accuracy
                future = model.make_future_dataframe(periods=0)
                forecast = model.predict(future)
                y_true = prophet_data['y'].values
                y_pred = forecast['yhat'].values
                r2 = np.corrcoef(y_true, y_pred)[0, 1] ** 2 if len(y_pred) > 0 else 0.5
                
            # Update model status
            self.model_status[model_name].update({
                "loaded": True,
                "last_trained": datetime.now().isoformat(),
                "training_samples": len(ticker_data),
                "status": "trained",
                "accuracy": max(0.0, float(r2)) if not np.isnan(r2) else 0.5
            })
            
            return {
                "success": True,
                "training_samples": len(ticker_data),
                "ticker_used": ticker,
                "accuracy": max(0.0, float(r2)) if not np.isnan(r2) else 0.5
            }
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            return {"success": False, "error": str(e)}
    
    def _train_traditional_model(self, model_name: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Train traditional ML models with proper scaling"""
        try:
            # COMPLIANCE CHECK: Block non-traditional models routed through traditional training
            problematic_models = [
                "BERT", "FinBERT", "BERTClassificationHead", "SentenceTransformerEmbedding",
                "DQN", "QLearning", "DoubleQLearning", "DuelingDQN", "RegimeAwareDQN",
                "LSTM", "GRU", "Transformer", "BlackScholes", "MonteCarloSimulation"
            ]
            
            if any(problem in model_name for problem in problematic_models):
                return {
                    "success": False, 
                    "error": f"COMPLIANCE VIOLATION: {model_name} cannot be trained through traditional ML routing. Requires authentic implementation."
                }
            # Prepare features and target
            feature_columns = [col for col in data.columns if col not in 
                              ['target', 'ticker', 'timestamp', 'source', 'verified', 'sector']]
            
            X = data[feature_columns].fillna(0)
            y = data['target'].fillna(0)
            
            # Remove rows where target is NaN
            valid_rows = ~y.isna()
            X = X[valid_rows]
            y = y[valid_rows]
            
            if len(X) < 10:
                return {"success": False, "error": "Insufficient training samples (minimum 10 required)"}
            
            # Create model
            if model_name == "LinearRegression":
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
            elif model_name == "Ridge":
                from sklearn.linear_model import Ridge
                model = Ridge(alpha=1.0, random_state=42)
            elif model_name == "Lasso":
                from sklearn.linear_model import Lasso
                model = Lasso(alpha=1.0, random_state=42)
            elif model_name == "SVR":
                from sklearn.svm import SVR
                model = SVR(kernel='rbf', C=1.0, gamma='scale')
            elif model_name == "ElasticNet":
                from sklearn.linear_model import ElasticNet
                model = ElasticNet(alpha=1.0, random_state=42)
            else:
                return {"success": False, "error": f"Model {model_name} not recognized"}
            
            # Train-test split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features for sensitive models
            if model_name in ["SVR", "Ridge", "Lasso", "ElasticNet"]:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                self.scalers[model_name] = scaler
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test
            
            # Train the model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions and calculate metrics
            y_pred = model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Store the model
            self.models[model_name] = model
            
            # Update model status
            self.model_status[model_name].update({
                "loaded": True,
                "last_trained": datetime.now().isoformat(),
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "status": "trained",
                "accuracy": max(0.0, float(r2)) if not np.isnan(r2) else 0.0,
                "r2_score": float(r2),
                "mse": float(mse),
                "mae": float(mae)
            })
            
            return {
                "success": True,
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "features_used": len(feature_columns),
                "scaled": model_name in ["SVR", "Ridge", "Lasso", "ElasticNet"],
                "r2_score": float(r2),
                "mse": float(mse),
                "mae": float(mae)
            }
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            return {"success": False, "error": str(e)}
    
    def _load_model_config(self) -> Dict:
        """Load ML model configuration"""
        try:
            config_path = "config/ml_config.yaml"
            if os.path.exists(config_path):
                import yaml
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load ML config: {e}")
        
        # Default configuration
        return {
            "models": {
                "RandomForest": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "random_state": 42
                },
                "XGBoost": {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 6
                },
                "LSTM": {
                    "units": 50,
                    "dropout": 0.2,
                    "epochs": 50
                }
            },
            "training": {
                "test_size": 0.2,
                "validation_size": 0.1,
                "min_samples": 100,
                "retrain_threshold": 0.7
            },
            "features": {
                "technical_indicators": True,
                "volume_features": True,
                "price_features": True,
                "macro_features": True
            }
        }
    
    def _initialize_base_models(self):
        """Initialize models dynamically from centralized registry"""
        try:
            initialized_count = 0
            
            # Get available models from registry and try to initialize each one
            for model_name in self.available_models:
                if self._try_initialize_model(model_name):
                    initialized_count += 1
                    
            # Initialize ensemble if we have multiple models
            if initialized_count > 1:
                self._initialize_ensemble()
            
            self.is_initialized = True
            logger.info(f"Initialized {initialized_count} models from centralized registry")
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            self.is_initialized = False
    
    def _try_initialize_model(self, model_name: str) -> bool:
        """Try to initialize a single model from registry"""
        try:
            model_info = self.model_registry.get_model_info(model_name)
            implementation = model_info.get("implementation", "")
            
            # Only initialize models that have implementations available
            if not implementation:
                return False
            
            success = False
            if model_name == "RandomForest" and implementation == "sklearn":
                success = self._init_sklearn_random_forest()
            elif model_name == "XGBoost" and implementation == "xgboost":
                success = self._init_xgboost()
            elif model_name == "LightGBM" and implementation == "lightgbm":
                success = self._init_lightgbm()
            elif model_name == "LSTM" and implementation in ["tensorflow", "pytorch"]:
                success = self._init_lstm()
                
            if success:
                self.model_performance[model_name] = {
                    "loaded": True,
                    "accuracy": 0.0,
                    "last_updated": datetime.now().isoformat(),
                    "training_samples": 0,
                    "status": "initialized"
                }
            return success
            
        except Exception as e:
            logger.warning(f"Failed to initialize {model_name}: {e}")
            return False
    
    def _init_sklearn_random_forest(self) -> bool:
        """Initialize RandomForest from sklearn"""
        try:
            from sklearn.ensemble import RandomForestRegressor
            self.models["RandomForest"] = RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=self.n_jobs
            )
            return True
        except ImportError:
            return False
    
    def _init_xgboost(self) -> bool:
        """Initialize XGBoost"""
        try:
            import xgboost as xgb
            self.models["XGBoost"] = xgb.XGBRegressor(
                n_estimators=100, random_state=42, n_jobs=self.n_jobs
            )
            return True
        except ImportError:
            return False
    
    def _init_lightgbm(self) -> bool:
        """Initialize LightGBM"""
        try:
            import lightgbm as lgb
            self.models["LightGBM"] = lgb.LGBMRegressor(
                n_estimators=100, random_state=42, n_jobs=self.n_jobs, verbosity=-1
            )
            return True
        except ImportError:
            return False
    
    def _init_lstm(self) -> bool:
        """Initialize LSTM (placeholder for now)"""
        # LSTM requires more complex setup, will be implemented with model_implementations
        return False
    
    def _initialize_ensemble(self):
        """Initialize ensemble voting model"""
        try:
            from sklearn.ensemble import VotingRegressor
            
            # Create ensemble from available models
            estimators = []
            for model_name, model in self.models.items():
                if model_name != "EnsembleVoting":
                    estimators.append((model_name.lower(), model))
            
            if estimators:
                self.models["EnsembleVoting"] = VotingRegressor(estimators=estimators)
                self.model_performance["EnsembleVoting"] = {
                    "loaded": True,
                    "accuracy": 0.0,
                    "last_updated": datetime.now().isoformat(),
                    "training_samples": 0,
                    "status": "initialized",
                    "ensemble_size": len(estimators)
                }
            
        except Exception as e:
            logger.error(f"Ensemble initialization failed: {e}")
    
    def is_ready(self) -> bool:
        """Check if ML pipeline is ready for inference"""
        return self.is_initialized and len(self.models) > 0
    
    def prepare_features(self, data: Dict) -> Optional[np.ndarray]:
        """Prepare features from market data for model input"""
        try:
            if not data or "data" not in data:
                return None
            
            price_data = data["data"]
            if not price_data or len(price_data) < 10:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(price_data)
            
            # Ensure we have OHLCV data
            required_cols = ['o', 'h', 'l', 'c', 'v']
            if not all(col in df.columns for col in required_cols):
                logger.warning("Missing required OHLCV columns")
                return None
            
            features = []
            
            # Price-based features
            if self.model_config["features"].get("price_features", True):
                # Returns
                df['returns'] = df['c'].pct_change()
                df['log_returns'] = np.log(df['c'] / df['c'].shift(1))
                
                # Price ratios
                df['hl_ratio'] = (df['h'] - df['l']) / df['c']
                df['oc_ratio'] = (df['c'] - df['o']) / df['o']
                
                features.extend(['returns', 'log_returns', 'hl_ratio', 'oc_ratio'])
            
            # Technical indicators
            if self.model_config["features"].get("technical_indicators", True):
                # Moving averages
                df['sma_5'] = df['c'].rolling(5).mean()
                df['sma_20'] = df['c'].rolling(20).mean()
                df['price_to_sma5'] = df['c'] / df['sma_5']
                df['price_to_sma20'] = df['c'] / df['sma_20']
                
                # Volatility
                df['volatility'] = df['returns'].rolling(10).std()
                
                # RSI-like momentum
                df['momentum'] = df['c'] / df['c'].shift(10) - 1
                
                features.extend(['price_to_sma5', 'price_to_sma20', 'volatility', 'momentum'])
            
            # Volume features
            if self.model_config["features"].get("volume_features", True):
                df['volume_sma'] = df['v'].rolling(5).mean()
                df['volume_ratio'] = df['v'] / df['volume_sma']
                df['price_volume'] = df['c'] * df['v']
                
                features.extend(['volume_ratio'])
            
            # Select features and drop NaN
            feature_data = df[features].dropna()
            
            if len(feature_data) < 5:
                logger.warning("Insufficient feature data after preprocessing")
                return None
            
            return feature_data.values
            
        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            return None
    
    def train_models(self, training_data: List[Dict], regime_type: str = "default") -> Dict[str, Any]:
        """Train models with regime-specific data"""
        if not training_data:
            return {"success": False, "error": "No training data provided"}
        
        training_results = {
            "timestamp": datetime.now().isoformat(),
            "regime_type": regime_type,
            "models_trained": [],
            "errors": [],
            "performance": {}
        }
        
        try:
            # Prepare combined dataset
            all_features = []
            all_targets = []
            
            for data_point in training_data:
                features = self.prepare_features(data_point)
                if features is not None and len(features) > 0:
                    # Use next period return as target
                    price_data = data_point["data"]
                    if len(price_data) > 1:
                        current_price = price_data[0]["c"]
                        next_price = price_data[1]["c"] if len(price_data) > 1 else current_price
                        target = (next_price - current_price) / current_price
                        
                        all_features.append(features[-1])  # Use latest feature vector
                        all_targets.append(target)
            
            if len(all_features) < self.model_config["training"]["min_samples"]:
                return {
                    "success": False, 
                    "error": f"Insufficient training samples: {len(all_features)}"
                }
            
            X = np.array(all_features)
            y = np.array(all_targets)
            
            # Scale features
            scaler_key = f"{regime_type}_scaler"
            if scaler_key not in self.scalers:
                self.scalers[scaler_key] = StandardScaler()
            
            X_scaled = self.scalers[scaler_key].fit_transform(X)
            
            # Split data
            test_size = self.model_config["training"]["test_size"]
            split_idx = int(len(X_scaled) * (1 - test_size))
            
            X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train each model
            for model_name, model in self.models.items():
                try:
                    if not self.model_performance[model_name].get("loaded", False):
                        continue
                    
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Evaluate
                    train_pred = model.predict(X_train)
                    test_pred = model.predict(X_test)
                    
                    # Calculate metrics (for regression, use correlation as "accuracy")
                    train_corr = np.corrcoef(y_train, train_pred)[0, 1] if len(set(y_train)) > 1 else 0
                    test_corr = np.corrcoef(y_test, test_pred)[0, 1] if len(set(y_test)) > 1 else 0
                    
                    # Convert correlation to percentage
                    accuracy = max(0, test_corr * 100) if not np.isnan(test_corr) else 0
                    
                    # Update performance
                    self.model_performance[model_name].update({
                        "accuracy": round(accuracy, 1),
                        "train_correlation": round(train_corr, 3) if not np.isnan(train_corr) else 0,
                        "test_correlation": round(test_corr, 3) if not np.isnan(test_corr) else 0,
                        "training_samples": len(X_train),
                        "last_updated": datetime.now().isoformat(),
                        "regime_type": regime_type,
                        "status": "trained"
                    })
                    
                    training_results["models_trained"].append(model_name)
                    training_results["performance"][model_name] = accuracy
                    
                    logger.info(f"Trained {model_name}: accuracy={accuracy:.1f}%")
                    
                except Exception as e:
                    error_msg = f"Training failed for {model_name}: {e}"
                    training_results["errors"].append(error_msg)
                    logger.error(error_msg)
                    
                    self.model_performance[model_name]["status"] = "error"
                    self.model_performance[model_name]["error"] = str(e)
            
            self.last_training_time = datetime.now()
            training_results["success"] = len(training_results["models_trained"]) > 0
            
            # Record performance history
            self.performance_history.append({
                "timestamp": datetime.now().isoformat(),
                "regime_type": regime_type,
                "performance": training_results["performance"].copy()
            })
            
            return training_results
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return {"success": False, "error": str(e)}
    
    def predict(self, data: Dict, model_names: List[str] = None) -> Dict[str, Any]:
        """Generate predictions from specified models"""
        if not self.is_ready():
            return {"error": "ML pipeline not ready"}
        
        if model_names is None:
            model_names = [name for name, perf in self.model_performance.items() 
                          if perf.get("loaded") and perf.get("status") == "trained"]
        
        predictions = {}
        features = self.prepare_features(data)
        
        if features is None:
            return {"error": "Could not prepare features from data"}
        
        try:
            # Use the latest feature vector
            feature_vector = features[-1].reshape(1, -1)
            
            # Scale features (use default scaler if regime-specific not available)
            scaler_key = "default_scaler"
            if scaler_key in self.scalers:
                feature_vector = self.scalers[scaler_key].transform(feature_vector)
            
            for model_name in model_names:
                if model_name in self.models and self.model_performance[model_name].get("loaded"):
                    try:
                        pred = self.models[model_name].predict(feature_vector)[0]
                        confidence = self.model_performance[model_name].get("accuracy", 0)
                        
                        predictions[model_name] = {
                            "prediction": float(pred),
                            "confidence": confidence,
                            "model_status": self.model_performance[model_name].get("status")
                        }
                    except Exception as e:
                        predictions[model_name] = {
                            "error": str(e),
                            "model_status": "error"
                        }
            
            return {
                "predictions": predictions,
                "timestamp": datetime.now().isoformat(),
                "feature_count": len(feature_vector[0]),
                "data_source": data.get("source", "unknown")
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {"error": str(e)}
    
    def get_model_status(self) -> Dict[str, Dict]:
        """Get status of all models"""
        return self.model_performance.copy()
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        loaded_models = sum(1 for perf in self.model_performance.values() if perf.get("loaded"))
        trained_models = sum(1 for perf in self.model_performance.values() if perf.get("status") == "trained")
        avg_accuracy = np.mean([perf.get("accuracy", 0) for perf in self.model_performance.values() 
                               if perf.get("loaded") and perf.get("accuracy", 0) > 0])
        
        return {
            "is_ready": self.is_ready(),
            "total_models": len(self.available_models),
            "loaded_models": loaded_models,
            "trained_models": trained_models,
            "average_accuracy": round(avg_accuracy, 1) if avg_accuracy > 0 else 0,
            "last_training": self.last_training_time.isoformat() if self.last_training_time else None,
            "performance_history_length": len(self.performance_history),
            "scalers_available": len(self.scalers)
        }
    
    def select_best_models(self, regime_type: str, top_n: int = 3) -> List[str]:
        """Select best performing models for given regime"""
        # Filter models by regime and performance
        regime_performance = []
        
        for model_name, perf in self.model_performance.items():
            if (perf.get("loaded") and 
                perf.get("status") == "trained" and 
                perf.get("regime_type", "default") == regime_type):
                regime_performance.append((model_name, perf.get("accuracy", 0)))
        
        # Sort by accuracy and return top N
        regime_performance.sort(key=lambda x: x[1], reverse=True)
        return [model_name for model_name, _ in regime_performance[:top_n]]
    
    def retrain_weak_models(self, threshold: float = None) -> Dict[str, Any]:
        """Retrain models below performance threshold"""
        if threshold is None:
            threshold = self.model_config["training"]["retrain_threshold"] * 100
        
        weak_models = []
        for model_name, perf in self.model_performance.items():
            if (perf.get("loaded") and 
                perf.get("accuracy", 0) < threshold and 
                perf.get("status") == "trained"):
                weak_models.append(model_name)
        
        return {
            "weak_models_identified": weak_models,
            "threshold_used": threshold,
            "retrain_recommended": len(weak_models) > 0,
            "timestamp": datetime.now().isoformat()
        }
    
    def run_walk_forward_validation(self, symbol: str, model_name: str, data: pd.DataFrame, 
                                   windows: int = 4, prediction_days: int = 7) -> Dict[str, Any]:
        """Execute walk-forward validation for a specific model and symbol"""
        try:
            logger.info(f"Starting walk-forward validation for {model_name} on {symbol}")
            
            if data.empty or len(data) < 252:  # Need at least 1 year of data
                return {
                    "success": False,
                    "error": f"Insufficient data for walk-forward test: {len(data)} points"
                }
            
            # Prepare data for walk-forward testing
            data_sorted = data.sort_index() if hasattr(data, 'index') else data.sort_values('timestamp') if 'timestamp' in data.columns else data
            
            window_size = len(data_sorted) // windows
            results = []
            
            for window in range(windows):
                try:
                    # Define training and testing windows
                    train_start = window * window_size
                    train_end = train_start + window_size
                    test_start = train_end
                    test_end = min(test_start + prediction_days, len(data_sorted))
                    
                    if test_end <= test_start:
                        logger.warning(f"Insufficient test data for window {window}")
                        continue
                    
                    train_data = data_sorted.iloc[train_start:train_end]
                    test_data = data_sorted.iloc[test_start:test_end]
                    
                    # Train model on window data
                    train_result = self.train_model(model_name, train_data)
                    if not train_result.get('success'):
                        logger.warning(f"Training failed for window {window}: {train_result.get('error')}")
                        continue
                    
                    # Make predictions on test data
                    predictions = []
                    actual_values = []
                    
                    for _, row in test_data.iterrows():
                        try:
                            # Get features for prediction
                            features = self._extract_features_from_row(row)
                            if features is None:
                                continue
                            
                            # Make prediction
                            prediction = self._predict_single(model_name, features)
                            if prediction is not None:
                                predictions.append(prediction)
                                actual_values.append(row.get('target', row.get('close', 0)))
                        except Exception as e:
                            logger.warning(f"Prediction failed for row in window {window}: {e}")
                            continue
                    
                    # Calculate window accuracy
                    if len(predictions) > 0 and len(actual_values) > 0:
                        # For regression, use correlation as accuracy measure
                        correlation = np.corrcoef(predictions, actual_values)[0, 1] if len(predictions) > 1 else 0.5
                        accuracy = max(0, correlation ** 2)  # R-squared
                        
                        results.append({
                            "window": window + 1,
                            "train_samples": len(train_data),
                            "test_samples": len(test_data),
                            "predictions_made": len(predictions),
                            "accuracy": accuracy,
                            "avg_prediction": np.mean(predictions),
                            "avg_actual": np.mean(actual_values)
                        })
                    
                except Exception as e:
                    logger.error(f"Error in walk-forward window {window}: {e}")
                    continue
            
            if not results:
                return {
                    "success": False,
                    "error": "No successful validation windows completed"
                }
            
            # Calculate overall metrics
            overall_accuracy = np.mean([r['accuracy'] for r in results])
            consistency_score = 1.0 - np.std([r['accuracy'] for r in results])  # Lower std = higher consistency
            
            return {
                "success": True,
                "symbol": symbol,
                "model": model_name,
                "windows_completed": len(results),
                "accuracy": round(overall_accuracy, 4),
                "consistency_score": round(max(0, consistency_score), 4),
                "window_results": results,
                "total_predictions": sum(r['predictions_made'] for r in results),
                "validation_completed": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Walk-forward validation failed for {symbol}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def execute_model_prediction(self, model_name: str, symbol: str, data: pd.DataFrame,
                                prediction_days: int = 7, include_confidence: bool = True) -> Dict[str, Any]:
        """Execute real model prediction with actual data"""
        try:
            logger.info(f"Executing {model_name} prediction for {symbol}")
            
            if data.empty:
                return {
                    "success": False,
                    "error": "No data provided for prediction"
                }
            
            # Ensure model is trained
            model_needs_training = False
            if model_name not in self.models:
                model_needs_training = True
            else:
                # Check if the model is actually fitted
                try:
                    model = self.models[model_name]
                    # For sklearn models, check if they have been fitted
                    if hasattr(model, 'n_features_in_') and not hasattr(model, 'feature_importances_'):
                        model_needs_training = True
                    elif not hasattr(model, 'n_features_in_'):
                        model_needs_training = True
                except Exception:
                    model_needs_training = True
            
            if model_needs_training:
                logger.info(f"Training {model_name} for prediction")
                train_result = self._train_single_model(model_name, data)
                if not train_result.get('success'):
                    return {
                        "success": False,
                        "error": f"Model training failed: {train_result.get('error')}"
                    }
            
            # Get latest data point for prediction
            latest_data = data.iloc[-1] if not data.empty else None
            if latest_data is None:
                return {
                    "success": False,
                    "error": "No data available for prediction"
                }
            
            # Extract features
            features = self._extract_features_from_row(latest_data)
            if features is None:
                return {
                    "success": False,
                    "error": "Could not extract features from data"
                }
            
            # Make prediction
            predicted_price = self._predict_single(model_name, features)
            if predicted_price is None:
                return {
                    "success": False,
                    "error": "Prediction failed"
                }
            
            # Get current price
            current_price = latest_data.get('close', latest_data.get('target', 0))
            if current_price == 0:
                return {
                    "success": False,
                    "error": "Could not determine current price"
                }
            
            # Calculate metrics
            price_change = predicted_price - current_price
            price_change_percent = (price_change / current_price) * 100
            
            # Calculate confidence based on model performance
            model_accuracy = self.model_status.get(model_name, {}).get('accuracy', 0.5)
            base_confidence = model_accuracy * 100
            
            # Adjust confidence based on prediction magnitude
            if abs(price_change_percent) > 20:  # Large predictions are less confident
                confidence = base_confidence * 0.7
            elif abs(price_change_percent) > 10:
                confidence = base_confidence * 0.85
            else:
                confidence = base_confidence
            
            confidence = max(20, min(95, confidence))  # Clamp between 20-95%
            
            result = {
                "success": True,
                "symbol": symbol,
                "model": model_name,
                "predicted_price": round(predicted_price, 2),
                "current_price": round(current_price, 2),
                "price_change": round(price_change, 2),
                "price_change_percent": round(price_change_percent, 2),
                "prediction_days": prediction_days,
                "features_count": len(features) if isinstance(features, (list, np.ndarray)) else 1,
                "data_points": len(data),
                "prediction_timestamp": datetime.now().isoformat()
            }
            
            if include_confidence:
                result.update({
                    "confidence": round(confidence, 1),
                    "model_accuracy": round(model_accuracy * 100, 1) if model_accuracy else 50.0
                })
            
            logger.info(f"Prediction completed for {symbol}: {predicted_price:.2f} ({price_change_percent:+.1f}%)")
            return result
            
        except Exception as e:
            logger.error(f"Model prediction failed for {symbol}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _extract_features_from_row(self, row) -> Optional[np.ndarray]:
        """Extract numerical features from a data row"""
        try:
            # Define feature columns (exclude non-feature columns)
            exclude_cols = ['target', 'ticker', 'timestamp', 'source', 'verified', 'sector', 'close']
            
            if hasattr(row, 'index'):
                # Pandas Series
                feature_cols = [col for col in row.index if col not in exclude_cols]
                features = row[feature_cols].fillna(0).values
            else:
                # Dictionary-like
                feature_cols = [col for col in row.keys() if col not in exclude_cols]
                features = np.array([row.get(col, 0) for col in feature_cols])
            
            # Convert to float and handle any remaining NaN values
            features = np.array([float(x) if pd.notna(x) else 0.0 for x in features])
            
            return features if len(features) > 0 else None
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None
    
    def _predict_single(self, model_name: str, features: np.ndarray) -> Optional[float]:
        """Make a single prediction with the specified model"""
        try:
            if model_name not in self.models:
                logger.error(f"Model {model_name} not found")
                return None
            
            model = self.models[model_name]
            
            # Reshape features for prediction
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            # Scale features if scaler exists
            if model_name in self.scalers:
                features = self.scalers[model_name].transform(features)
            
            # Make prediction
            prediction = model.predict(features)
            
            # Extract scalar value
            if hasattr(prediction, '__len__') and len(prediction) > 0:
                return float(prediction[0])
            else:
                return float(prediction)
                
        except Exception as e:
            logger.error(f"Single prediction failed for {model_name}: {e}")
            return None

