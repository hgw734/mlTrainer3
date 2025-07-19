"""
mlTrainer - Comprehensive Model Trainer
=======================================

Purpose: Complete training of ALL 120+ models with verified data only.
Monitors every model training operation and reports detailed progress.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple
import sys
import os

# Import centralized configuration - no more hard coding
from utils.system_config import get_system_config, get_training_params, get_authentic_models

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_registry import ModelRegistry
from backend.compliance_engine import ComplianceEngine
from backend.data_sources import DataSourceManager

logger = logging.getLogger(__name__)

class ComprehensiveTrainer:
    """Complete training system for ALL models with detailed monitoring"""
    
    def __init__(self):
        # Load centralized configuration - single source of truth
        self.config = get_system_config()
        self.training_params = get_training_params()
        self.authentic_models = get_authentic_models()
        
        # Initialize core components
        self.model_registry = ModelRegistry()
        self.compliance_engine = ComplianceEngine()
        self.data_source_manager = DataSourceManager()
        self.training_progress = {}
        self.failed_models = {}
        self.successful_models = {}
        
        # Get directories from centralized config
        self.directories = self.config.get_directories()
        self.model_save_dir = self.directories.get("models", "models")
        
        # Get model registry info from config
        model_info = self.config.get_model_registry_info()
        
        logger.info(f"🚀 ComprehensiveTrainer initialized with centralized configuration")
        logger.info(f"📊 Target: {model_info['total_count']} models across {model_info['categories']} categories")
        logger.info(f"🎯 Training limits: {self.training_params['min_samples']}-{self.training_params['max_samples']} samples")
    
    def train_all_models_comprehensive(self, tickers: List[str], days: int = 90) -> Dict[str, Any]:
        """Train ALL 120+ models with comprehensive monitoring and reporting"""
        
        logger.info(f"🎯 STARTING COMPREHENSIVE TRAINING: {len(tickers)} tickers, {days} days")
        
        # Get all models from registry
        all_models = self.model_registry.get_all_models()
        total_models = len(all_models)
        
        logger.info(f"📊 TOTAL MODELS TO TRAIN: {total_models}")
        
        # Initialize progress tracking
        self.training_progress = {
            "total_models": total_models,
            "completed": 0,
            "successful": 0,
            "failed": 0,
            "start_time": datetime.now().isoformat(),
            "current_model": None,
            "models_by_category": {}
        }
        
        # Get verified training data
        training_data = self._get_verified_training_data(tickers, days)
        if training_data.empty:
            logger.error("🚨 CRITICAL: No verified training data available")
            return {"success": False, "error": "No verified training data"}
        
        logger.info(f"✅ VERIFIED DATA ACQUIRED: {len(training_data)} samples")
        
        # Train models by category for organized progress tracking
        categories = self.model_registry.get_categories()
        
        for category in categories:
            category_models = self.model_registry.get_models_by_category(category)
            self.training_progress["models_by_category"][category] = {
                "total": len(category_models),
                "completed": 0,
                "successful": 0,
                "failed": 0,
                "models": {}
            }
            
            logger.info(f"🔄 TRAINING CATEGORY: {category} ({len(category_models)} models)")
            
            for model_name in category_models:
                self._train_single_model_monitored(model_name, category, training_data)
                
        # Generate final comprehensive report
        return self._generate_final_report()
    
    def _get_verified_training_data(self, tickers: List[str], days: int) -> pd.DataFrame:
        """Get verified training data from Polygon and FRED only"""
        
        logger.info(f"📥 ACQUIRING VERIFIED DATA: {tickers} for {days} days")
        
        try:
            # Use verified data sources from centralized configuration
            verified_sources = self.config.get_verified_sources()
            primary_market_source = self.config.get_primary_market_source()
            
            if primary_market_source not in verified_sources:
                logger.error(f"❌ Primary market source {primary_market_source} not in verified sources")
                return pd.DataFrame()
            
            # Get S&P 500 data and add technical indicators
            from data.sp500_data import SP500DataManager
            sp500_manager = SP500DataManager()
            
            all_data = []
            
            for ticker in tickers:
                # Get market data from verified source only
                ticker_data = sp500_manager.get_stock_data(ticker, days)
                
                if ticker_data is not None and not ticker_data.empty:
                    # Add technical indicators
                    ticker_data_with_indicators = self._add_technical_indicators(ticker_data)
                    
                    # Universal compliance check
                    data_dict = ticker_data_with_indicators.to_dict('records')
                    approved, validated_data = self.compliance_engine.universal_interceptor.intercept_all_data(
                        data_dict, 
                        context=f"TRAINING_DATA_{ticker}"
                    )
                    
                    if approved and validated_data:
                        validated_df = pd.DataFrame(validated_data)
                        all_data.append(validated_df)
                        logger.info(f"✅ {ticker}: {len(validated_df)} verified samples with {len(validated_df.columns)} features")
                    else:
                        logger.error(f"🚨 {ticker}: Data failed compliance check")
                else:
                    logger.warning(f"⚠️ {ticker}: No data available")
            
            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                logger.info(f"📊 TOTAL VERIFIED SAMPLES: {len(combined_data)} with {len(combined_data.columns)} features")
                return combined_data
            else:
                logger.error("🚨 NO VERIFIED DATA AVAILABLE")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"🚨 ERROR ACQUIRING DATA: {e}")
            return pd.DataFrame()
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators to market data"""
        try:
            df = data.copy()
            
            # Ensure we have the required columns
            if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                logger.error("Missing required OHLCV columns")
                return df
            
            # Sort by timestamp to ensure proper calculation
            if 'timestamp' in df.columns:
                df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Get indicator parameters from centralized configuration
            indicators_config = self.config.get_feature_config()
            
            # Simple Moving Averages using config parameters
            sma_periods = indicators_config.get('sma_periods', [5, 10, 20, 50])
            for period in sma_periods:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            
            # Exponential Moving Averages using config parameters
            ema_fast = indicators_config.get('macd_fast', 12)
            ema_slow = indicators_config.get('macd_slow', 26)
            df[f'ema_{ema_fast}'] = df['close'].ewm(span=ema_fast).mean()
            df[f'ema_{ema_slow}'] = df['close'].ewm(span=ema_slow).mean()
            
            # MACD using config parameters
            macd_signal_period = indicators_config.get('macd_signal', 9)
            df['macd'] = df[f'ema_{ema_fast}'] - df[f'ema_{ema_slow}']
            df['macd_signal'] = df['macd'].ewm(span=macd_signal_period).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # RSI using config parameters
            rsi_period = indicators_config.get('rsi_period', 14)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands using config parameters
            bb_period = indicators_config.get('bollinger_period', 20)
            bb_std_dev = indicators_config.get('bollinger_std_dev', 2)
            df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
            bb_std = df['close'].rolling(window=bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * bb_std_dev)
            df['bb_lower'] = df['bb_middle'] - (bb_std * bb_std_dev)
            df['bb_width'] = df['bb_upper'] - df['bb_lower']
            df['bb_position'] = (df['close'] - df['bb_lower']) / df['bb_width']
            
            # Volume indicators using config parameters
            volume_period = indicators_config.get('volume_period', 20)
            df['volume_sma'] = df['volume'].rolling(window=volume_period).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Price indicators
            df['high_low_pct'] = (df['high'] - df['low']) / df['close']
            df['close_open_pct'] = (df['close'] - df['open']) / df['open']
            
            # Volatility (ATR approximation) using config parameters
            atr_period = indicators_config.get('atr_period', 14)
            df['tr'] = np.maximum(df['high'] - df['low'], 
                                 np.maximum(abs(df['high'] - df['close'].shift(1)),
                                          abs(df['low'] - df['close'].shift(1))))
            df['atr'] = df['tr'].rolling(window=atr_period).mean()
            
            # Target variable (next day return)
            df['target'] = df['close'].shift(-1) / df['close'] - 1
            
            # Drop rows with NaN values (due to rolling calculations)
            df = df.dropna()
            
            logger.info(f"Added technical indicators using centralized config: {len(df.columns)} total features")
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return data
    
    def _train_single_model_monitored(self, model_name: str, category: str, data: pd.DataFrame):
        """Train a single model with comprehensive monitoring"""
        
        self.training_progress["current_model"] = model_name
        self.training_progress["completed"] += 1
        
        progress = self.training_progress["completed"]
        total = self.training_progress["total_models"]
        
        logger.info(f"🔄 TRAINING [{progress}/{total}]: {model_name} ({category})")
        
        try:
            # Route to appropriate training method based on category and model requirements
            result = self._route_model_training(model_name, category, data)
            
            if result.get("success", False):
                self.successful_models[model_name] = result
                self.training_progress["successful"] += 1
                self.training_progress["models_by_category"][category]["successful"] += 1
                self.training_progress["models_by_category"][category]["models"][model_name] = "SUCCESS"
                
                logger.info(f"✅ SUCCESS [{progress}/{total}]: {model_name} - {result.get('training_samples', 0)} samples")
                
            else:
                error_msg = result.get("error", "Unknown error")
                self.failed_models[model_name] = error_msg
                self.training_progress["failed"] += 1
                self.training_progress["models_by_category"][category]["failed"] += 1
                self.training_progress["models_by_category"][category]["models"][model_name] = f"FAILED: {error_msg}"
                
                logger.error(f"❌ FAILED [{progress}/{total}]: {model_name} - {error_msg}")
                
        except Exception as e:
            error_msg = str(e)
            self.failed_models[model_name] = error_msg
            self.training_progress["failed"] += 1
            self.training_progress["models_by_category"][category]["failed"] += 1
            self.training_progress["models_by_category"][category]["models"][model_name] = f"EXCEPTION: {error_msg}"
            
            logger.error(f"🚨 EXCEPTION [{progress}/{total}]: {model_name} - {error_msg}")
        
        finally:
            self.training_progress["models_by_category"][category]["completed"] += 1
    
    def _route_model_training(self, model_name: str, category: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Route model training to appropriate implementation"""
        
        # Universal compliance check for all training data
        data_dict = data.to_dict('records') if not data.empty else []
        approved, validated_data = self.compliance_engine.universal_interceptor.intercept_all_data(
            data_dict, 
            context=f"MODEL_TRAINING_{model_name}"
        )
        
        if not approved:
            return {
                "success": False,
                "error": f"Training data failed universal compliance check"
            }
        
        # Convert back to DataFrame
        if validated_data:
            data = pd.DataFrame(validated_data)
        
        # Route based on category and implementation requirements
        if category in ["Traditional ML"]:
            return self._train_traditional_ml_authentic(model_name, data)
        elif category in ["Time Series"]:
            return self._train_time_series_authentic(model_name, data)
        elif category in ["Ensemble & Meta-Learning"]:
            return self._train_ensemble_authentic(model_name, data)
        elif category in ["Deep Learning", "Neural Networks"]:
            return self._train_deep_learning_authentic(model_name, data)
        elif category in ["NLP & Sentiment"]:
            return self._train_nlp_authentic(model_name, data)
        elif category in ["Reinforcement Learning"]:
            return self._train_rl_authentic(model_name, data)
        else:
            return {
                "success": False,
                "error": f"Category {category} requires authentic implementation - placeholder blocking"
            }
    
    def _train_traditional_ml_authentic(self, model_name: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Train traditional ML models with authentic implementations only"""
        
        # Get authentic models from centralized configuration
        authentic_traditional_models = self.authentic_models.get("traditional_ml", [])
        if not authentic_traditional_models:
            # Fallback if config not available
            authentic_traditional_models = [
                "LinearRegression", "Ridge", "Lasso", "ElasticNet", "SVR",
                "KNearestNeighbors", "LogisticRegression", "RandomForest", 
                "XGBoost", "LightGBM", "CatBoost", "GradientBoosting",
                "ExtraTrees", "AdaBoost", "DecisionTree", "NaiveBayes"
            ]
        
        if model_name not in authentic_traditional_models:
            return {
                "success": False,
                "error": f"Model {model_name} requires authentic implementation verification"
            }
        
        try:
            # Prepare data for training
            if 'target' not in data.columns:
                return {
                    "success": False,
                    "error": f"Missing target column in training data"
                }
            
            # Select feature columns (exclude non-numeric and target)
            feature_columns = [col for col in data.columns 
                             if col not in ['target', 'timestamp', 'ticker', 'source', 'verified'] 
                             and data[col].dtype in ['int64', 'float64']]
            
            if len(feature_columns) == 0:
                return {
                    "success": False,
                    "error": f"No valid feature columns found"
                }
            
            # Remove rows with NaN values
            clean_data = data[feature_columns + ['target']].dropna()
            
            # Use minimum samples from centralized config
            min_samples = self.training_params['min_samples']
            if len(clean_data) < min_samples:
                return {
                    "success": False,
                    "error": f"Insufficient clean training samples: {len(clean_data)} < {min_samples}"
                }
            
            X = clean_data[feature_columns]
            y = clean_data['target']
            
            # Train using authentic scikit-learn implementations
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, r2_score
            import joblib
            import os
            
            # Split data using config parameters
            test_size = self.training_params['test_size']
            random_state = self.training_params['random_state']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            
            # Initialize model with authentic implementation
            model = self._get_authentic_model(model_name)
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Save model using configured directory and pattern
            sklearn_dir = os.path.join(self.model_save_dir, "sklearn")
            os.makedirs(sklearn_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Use file pattern from config if available
            file_patterns = self.config.get_file_patterns()
            if 'model_save' in file_patterns:
                filename = file_patterns['model_save'].format(model_name=model_name, timestamp=timestamp)
            else:
                filename = f"{model_name}_{timestamp}.joblib"
            
            model_path = os.path.join(sklearn_dir, filename)
            joblib.dump(model, model_path)
            
            logger.info(f"✅ Successfully trained {model_name}: MSE={mse:.4f}, R²={r2:.4f}")
            
            return {
                "success": True,
                "model_name": model_name,
                "training_samples": len(clean_data),
                "features_used": len(feature_columns),
                "mse": mse,
                "r2_score": r2,
                "model_path": model_path,
                "timestamp": timestamp
            }
            
        except Exception as e:
            return {"success": False, "error": f"Training failed: {str(e)}"}
    
    def _get_authentic_model(self, model_name: str):
        """Get authentic model implementation from scikit-learn"""
        from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
        from sklearn.svm import SVR
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.naive_bayes import GaussianNB
        
        try:
            import xgboost as xgb
            import lightgbm as lgb
            import catboost as cb
        except ImportError:
            # Fallback if advanced libraries not available
            pass
        
        # Get model parameters from centralized configuration
        n_estimators = self.training_params['n_estimators']
        random_state = self.training_params['random_state']
        
        model_map = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=1.0),
            "ElasticNet": ElasticNet(alpha=1.0),
            "SVR": SVR(kernel='rbf'),
            "KNearestNeighbors": KNeighborsRegressor(n_neighbors=5),
            "LogisticRegression": LogisticRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=n_estimators, random_state=random_state),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=n_estimators, random_state=random_state),
            "ExtraTrees": ExtraTreesRegressor(n_estimators=n_estimators, random_state=random_state),
            "AdaBoost": AdaBoostRegressor(n_estimators=n_estimators, random_state=random_state),
            "DecisionTree": DecisionTreeRegressor(random_state=random_state),
            "NaiveBayes": GaussianNB()
        }
        
        # Add advanced models if available, using centralized parameters
        try:
            model_map["XGBoost"] = xgb.XGBRegressor(n_estimators=n_estimators, random_state=random_state)
        except:
            pass
            
        try:
            model_map["LightGBM"] = lgb.LGBMRegressor(n_estimators=n_estimators, random_state=random_state, verbose=-1)
        except:
            pass
            
        try:
            model_map["CatBoost"] = cb.CatBoostRegressor(iterations=n_estimators, random_state=random_state, verbose=False)
        except:
            pass
        
        return model_map.get(model_name, LinearRegression())
    
    def _train_time_series_authentic(self, model_name: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Train time series models with authentic implementations"""
        return {
            "success": False,
            "error": f"Time series model {model_name} requires authentic statistical implementation"
        }
    
    def _train_ensemble_authentic(self, model_name: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Train ensemble models with authentic implementations"""
        return {
            "success": False,
            "error": f"Ensemble model {model_name} requires authentic ensemble implementation"
        }
    
    def _train_deep_learning_authentic(self, model_name: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Train deep learning models with authentic implementations"""
        return {
            "success": False,
            "error": f"Deep learning model {model_name} requires authentic neural network implementation"
        }
    
    def _train_nlp_authentic(self, model_name: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Train NLP models with authentic implementations"""
        return {
            "success": False,
            "error": f"NLP model {model_name} requires authentic transformer/BERT implementation"
        }
    
    def _train_rl_authentic(self, model_name: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Train RL models with authentic implementations"""
        return {
            "success": False,
            "error": f"RL model {model_name} requires authentic reinforcement learning implementation"
        }
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final training report"""
        
        self.training_progress["end_time"] = datetime.now().isoformat()
        
        logger.info("📋 GENERATING FINAL COMPREHENSIVE REPORT")
        logger.info(f"✅ SUCCESSFUL MODELS: {self.training_progress['successful']}")
        logger.info(f"❌ FAILED MODELS: {self.training_progress['failed']}")
        logger.info(f"📊 TOTAL PROCESSED: {self.training_progress['completed']}")
        
        # Log successful models
        if self.successful_models:
            logger.info("🎯 SUCCESSFULLY TRAINED MODELS:")
            for model_name, result in self.successful_models.items():
                samples = result.get('training_samples', 0)
                accuracy = result.get('accuracy', 0)
                logger.info(f"  ✅ {model_name}: {samples} samples, accuracy: {accuracy}")
        
        # Log failed models with reasons
        if self.failed_models:
            logger.info("⚠️ FAILED MODELS AND REASONS:")
            for model_name, error in self.failed_models.items():
                logger.info(f"  ❌ {model_name}: {error}")
        
        return {
            "success": True,
            "training_progress": self.training_progress,
            "successful_models": self.successful_models,
            "failed_models": self.failed_models,
            "summary": {
                "total_models": self.training_progress["total_models"],
                "successful": self.training_progress["successful"],
                "failed": self.training_progress["failed"],
                "success_rate": (self.training_progress["successful"] / self.training_progress["total_models"]) * 100
            }
        }
    
    def get_progress_status(self) -> Dict[str, Any]:
        """Get current training progress status"""
        return self.training_progress