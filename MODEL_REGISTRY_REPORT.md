# mlTrainer3 Model Registry Report

Generated: 2025-07-19T15:10:27.687761

## Summary

- **Total Models**: 180
- **Categories**: 10
- **Files Scanned**: 19

## Models by Category

- **Volatility Models**: 21 models
- **Machine Learning**: 49 models
- **Ensemble Methods**: 3 models
- **Market Regime Detection**: 16 models
- **Risk Management**: 16 models
- **Technical Analysis**: 10 models

## Model Details

### Ensemble Methods

#### CCIEnsemble
- **Type**: class
- **File**: `custom/indicators.py`
- **Parameters**: periods, threshold
- **Description**: Commodity Channel Index Ensemble

#### CCIEnsemble
- **Type**: class
- **File**: `custom/momentum.py`
- **Parameters**: window
- **Description**: Commodity Channel Index Ensemble

#### EnsembleMLModel
- **Type**: class
- **File**: `custom/machine_learning.py`
- **Parameters**: n_estimators, max_depth, learning_rate
- **Description**: Ensemble machine learning model

### Machine Learning

#### AnomalyDetector
- **Type**: class
- **File**: `custom/detectors.py`
- **Parameters**: threshold
- **Description**: Anomaly Detection for Trading Signals

#### AutoMLEnsemble
- **Type**: class
- **File**: `custom/automl.py`
- **Parameters**: n_models
- **Description**: Automatic Ensemble Model Selection

#### BaseMLModel
- **Type**: class
- **File**: `custom/machine_learning.py`
- **Description**: Base class for machine learning models

#### BaseVolumeModel
- **Type**: class
- **File**: `custom/volume.py`
- **Description**: Base class for volume analysis models

#### BaseVolumeModel
- **Type**: class
- **File**: `custom/volume_analysis.py`
- **Description**: Base class for volume analysis models

#### ComplexityModel
- **Type**: class
- **File**: `custom/complexity.py`
- **Parameters**: complexity_window
- **Description**: Complexity Model

#### DeepLearningModel
- **Type**: class
- **File**: `custom/machine_learning.py`
- **Parameters**: hidden_layers, dropout_rate
- **Description**: Deep learning model

#### FractalModel
- **Type**: class
- **File**: `custom/fractal.py`
- **Parameters**: fractal_window
- **Description**: Fractal Model

#### MachineLearningModel
- **Type**: class
- **File**: `custom/machine_learning.py`
- **Parameters**: ml_window
- **Description**: Comprehensive machine learning model for S&P 500 trading

#### ModelArchitectureSearch
- **Type**: class
- **File**: `custom/automl.py`
- **Parameters**: max_models, search_iterations
- **Description**: Model Architecture Search

#### NonlinearModel
- **Type**: class
- **File**: `custom/nonlinear.py`
- **Parameters**: nonlinear_window
- **Description**: Nonlinear Model

#### PatternsModel
- **Type**: class
- **File**: `custom/patterns.py`
- **Parameters**: pattern_window
- **Description**: Patterns Model

#### ReinforcementLearningModel
- **Type**: class
- **File**: `custom/machine_learning.py`
- **Parameters**: learning_rate, discount_factor
- **Description**: Reinforcement learning model

#### VolumeAnalysisModel
- **Type**: class
- **File**: `custom/volume_analysis.py`
- **Parameters**: analysis_window
- **Description**: Comprehensive volume analysis model for S&P 500 trading

#### VolumeClusteringModel
- **Type**: class
- **File**: `custom/volume_analysis.py`
- **Parameters**: n_clusters, window
- **Description**: Volume clustering analysis model

#### VolumeDivergenceModel
- **Type**: class
- **File**: `custom/volume_analysis.py`
- **Parameters**: window, divergence_threshold
- **Description**: Volume divergence detection model

#### VolumeMomentumModel
- **Type**: class
- **File**: `custom/volume_analysis.py`
- **Parameters**: short_window, long_window, momentum_threshold
- **Description**: Volume momentum analysis model

#### VolumePriceRelationshipModel
- **Type**: class
- **File**: `custom/volume_analysis.py`
- **Parameters**: window, relationship_threshold
- **Description**: Volume-price relationship analysis model

#### VolumeWeightedModel
- **Type**: class
- **File**: `custom/volume_analysis.py`
- **Parameters**: window, volume_threshold
- **Description**: Volume-weighted analysis model

#### _initialize_models
- **Type**: function
- **File**: `custom/machine_learning.py`
- **Parameters**: self
- **Description**: Initialize all machine learning models

#### _initialize_models
- **Type**: function
- **File**: `custom/volume_analysis.py`
- **Parameters**: self
- **Description**: Initialize all volume analysis models

#### fit
- **Type**: function
- **File**: `custom/fractal.py`
- **Parameters**: self, data
- **Description**: No description available

#### fit
- **Type**: function
- **File**: `custom/machine_learning.py`
- **Parameters**: self, data, target
- **Description**: Fit all machine learning models

#### fit
- **Type**: function
- **File**: `custom/patterns.py`
- **Parameters**: self, data
- **Description**: No description available

#### fit
- **Type**: function
- **File**: `custom/systems.py`
- **Parameters**: self, data
- **Description**: No description available

#### fit
- **Type**: function
- **File**: `custom/nonlinear.py`
- **Parameters**: self, data
- **Description**: No description available

#### fit
- **Type**: function
- **File**: `custom/volume_analysis.py`
- **Parameters**: self, data, volume_data
- **Description**: Fit all volume analysis models

#### fit
- **Type**: function
- **File**: `custom/momentum.py`
- **Parameters**: self, data
- **Description**: No description available

#### fit
- **Type**: function
- **File**: `custom/adversarial.py`
- **Parameters**: self, data
- **Description**: No description available

#### fit
- **Type**: function
- **File**: `custom/detectors.py`
- **Parameters**: self, data
- **Description**: No description available

#### fit
- **Type**: function
- **File**: `custom/complexity.py`
- **Parameters**: self, data
- **Description**: No description available

#### fit
- **Type**: function
- **File**: `custom/automl.py`
- **Parameters**: self, data
- **Description**: No description available

#### fit_predict
- **Type**: function
- **File**: `custom/automl.py`
- **Parameters**: self, data
- **Description**: Simplified ensemble - returns mean

#### get_available_models
- **Type**: function
- **File**: `custom/machine_learning.py`
- **Parameters**: self
- **Description**: Get list of available machine learning models

#### get_available_models
- **Type**: function
- **File**: `custom/volume_analysis.py`
- **Parameters**: self
- **Description**: Get list of available volume analysis models

#### get_model_parameters
- **Type**: function
- **File**: `custom/machine_learning.py`
- **Parameters**: self, model_name
- **Description**: Get parameters for a specific model

#### get_model_parameters
- **Type**: function
- **File**: `custom/volume_analysis.py`
- **Parameters**: self, model_name
- **Description**: Get parameters for a specific model

#### load_model
- **Type**: function
- **File**: `custom/machine_learning.py`
- **Parameters**: self, model_name, filepath
- **Description**: Load trained model

#### predict
- **Type**: function
- **File**: `custom/fractal.py`
- **Parameters**: self, data
- **Description**: No description available

#### predict
- **Type**: function
- **File**: `custom/machine_learning.py`
- **Parameters**: self, data, model_name
- **Description**: Make machine learning prediction

#### predict
- **Type**: function
- **File**: `custom/patterns.py`
- **Parameters**: self, data
- **Description**: No description available

#### predict
- **Type**: function
- **File**: `custom/systems.py`
- **Parameters**: self, data
- **Description**: No description available

#### predict
- **Type**: function
- **File**: `custom/nonlinear.py`
- **Parameters**: self, data
- **Description**: No description available

#### predict
- **Type**: function
- **File**: `custom/momentum.py`
- **Parameters**: self, data
- **Description**: No description available

#### predict
- **Type**: function
- **File**: `custom/adversarial.py`
- **Parameters**: self, data
- **Description**: No description available

#### predict
- **Type**: function
- **File**: `custom/detectors.py`
- **Parameters**: self, data
- **Description**: No description available

#### predict
- **Type**: function
- **File**: `custom/complexity.py`
- **Parameters**: self, data
- **Description**: No description available

#### predict
- **Type**: function
- **File**: `custom/automl.py`
- **Parameters**: self, data
- **Description**: No description available

#### save_model
- **Type**: function
- **File**: `custom/machine_learning.py`
- **Parameters**: self, model_name, filepath
- **Description**: Save trained model

### Market Regime Detection

#### BaseRegimeModel
- **Type**: class
- **File**: `custom/regime_detection.py`
- **Description**: Base class for regime detection models

#### ChangePointRegimeModel
- **Type**: class
- **File**: `custom/regime_detection.py`
- **Parameters**: window, change_threshold
- **Description**: Change point detection for regime identification

#### ClusteringRegimeModel
- **Type**: class
- **File**: `custom/regime_detection.py`
- **Parameters**: window, n_regimes, features
- **Description**: Clustering-based regime detection

#### EnsembleRegimeModel
- **Type**: class
- **File**: `custom/regime_detection.py`
- **Parameters**: models
- **Description**: Ensemble regime detection model

#### GaussianMixtureRegimeModel
- **Type**: class
- **File**: `custom/regime_detection.py`
- **Parameters**: window, n_regimes
- **Description**: Gaussian Mixture Model for regime detection

#### MarketConditionRegimeModel
- **Type**: class
- **File**: `custom/regime_detection.py`
- **Parameters**: window, condition_threshold
- **Description**: Market condition classifier

#### PerformanceBasedRegimeModel
- **Type**: class
- **File**: `custom/regime_detection.py`
- **Parameters**: window, performance_threshold
- **Description**: Performance-based regime detection with reweighting

#### RegimeDetectionModel
- **Type**: class
- **File**: `custom/regime_detection.py`
- **Parameters**: regime_window
- **Description**: Comprehensive regime detection model for S&P 500 trading

#### RegimeDetector
- **Type**: class
- **File**: `custom/detectors.py`
- **Parameters**: lookback
- **Description**: Market Regime Detection

#### RegimeSwitchingModel
- **Type**: class
- **File**: `custom/regime_detection.py`
- **Parameters**: window, n_regimes, transition_threshold
- **Description**: Regime switching model with transition probabilities

#### VolatilityRegimeModel
- **Type**: class
- **File**: `custom/regime_detection.py`
- **Parameters**: window, n_regimes, volatility_threshold
- **Description**: Volatility-based regime detection

#### _initialize_models
- **Type**: function
- **File**: `custom/regime_detection.py`
- **Parameters**: self
- **Description**: Initialize all regime detection models

#### fit
- **Type**: function
- **File**: `custom/regime_detection.py`
- **Parameters**: self, data
- **Description**: Fit all regime detection models

#### get_available_models
- **Type**: function
- **File**: `custom/regime_detection.py`
- **Parameters**: self
- **Description**: Get list of available regime detection models

#### get_model_parameters
- **Type**: function
- **File**: `custom/regime_detection.py`
- **Parameters**: self, model_name
- **Description**: Get parameters for a specific model

#### predict
- **Type**: function
- **File**: `custom/regime_detection.py`
- **Parameters**: self, data, model_name
- **Description**: Detect regimes

### Risk Management

#### BaseRiskModel
- **Type**: class
- **File**: `custom/risk.py`
- **Description**: Base class for risk management models

#### BaseRiskModel
- **Type**: class
- **File**: `custom/risk_management.py`
- **Description**: Base class for risk management models

#### DynamicRiskAdjustmentModel
- **Type**: class
- **File**: `custom/risk_management.py`
- **Parameters**: base_volatility, adjustment_window
- **Description**: Dynamic risk adjustment model

#### MaximumDrawdownProtectionModel
- **Type**: class
- **File**: `custom/risk_management.py`
- **Parameters**: max_drawdown_threshold, protection_window
- **Description**: Maximum drawdown protection model

#### PortfolioRiskModel
- **Type**: class
- **File**: `custom/risk_management.py`
- **Parameters**: risk_free_rate, target_volatility
- **Description**: Portfolio risk management model

#### RiskManagementModel
- **Type**: class
- **File**: `custom/risk_management.py`
- **Parameters**: risk_window
- **Description**: Comprehensive risk management model for S&P 500 trading

#### RiskParityModel
- **Type**: class
- **File**: `custom/risk_management.py`
- **Parameters**: target_volatility
- **Description**: Risk parity model

#### StressTestingModel
- **Type**: class
- **File**: `custom/risk_management.py`
- **Parameters**: stress_scenarios
- **Description**: Stress testing model

#### VaRModel
- **Type**: class
- **File**: `custom/risk_management.py`
- **Parameters**: confidence_level, window
- **Description**: Value at Risk (VaR) calculation model

#### VolatilityTargetingModel
- **Type**: class
- **File**: `custom/risk_management.py`
- **Parameters**: target_volatility, rebalance_window
- **Description**: Volatility targeting model

#### _initialize_models
- **Type**: function
- **File**: `custom/risk_management.py`
- **Parameters**: self
- **Description**: Initialize all risk management models

#### fit
- **Type**: function
- **File**: `custom/risk.py`
- **Parameters**: self, data
- **Description**: No description available

#### fit
- **Type**: function
- **File**: `custom/risk_management.py`
- **Parameters**: self, data
- **Description**: Fit all risk management models

#### get_available_models
- **Type**: function
- **File**: `custom/risk_management.py`
- **Parameters**: self
- **Description**: Get list of available risk management models

#### get_model_parameters
- **Type**: function
- **File**: `custom/risk_management.py`
- **Parameters**: self, model_name
- **Description**: Get parameters for a specific model

#### predict
- **Type**: function
- **File**: `custom/risk.py`
- **Parameters**: self, data
- **Description**: No description available

### Technical Analysis

#### BaseTechnicalModel
- **Type**: class
- **File**: `custom/technical_analysis.py`
- **Description**: Base class for technical analysis models

#### OscillatorModel
- **Type**: class
- **File**: `custom/technical_analysis.py`
- **Parameters**: rsi_period, macd_fast, macd_slow
- **Description**: Oscillator analysis model

#### PatternRecognitionModel
- **Type**: class
- **File**: `custom/technical_analysis.py`
- **Parameters**: pattern_window, pattern_threshold
- **Description**: Pattern recognition model

#### SupportResistanceModel
- **Type**: class
- **File**: `custom/technical_analysis.py`
- **Parameters**: window, level_threshold
- **Description**: Support and resistance level model

#### TechnicalAnalysisModel
- **Type**: class
- **File**: `custom/technical_analysis.py`
- **Parameters**: analysis_window
- **Description**: Comprehensive technical analysis model for S&P 500 trading

#### TrendAnalysisModel
- **Type**: class
- **File**: `custom/technical_analysis.py`
- **Parameters**: short_window, long_window, trend_threshold
- **Description**: Trend analysis model

#### _initialize_models
- **Type**: function
- **File**: `custom/technical_analysis.py`
- **Parameters**: self
- **Description**: Initialize all technical analysis models

#### fit
- **Type**: function
- **File**: `custom/technical_analysis.py`
- **Parameters**: self, data
- **Description**: Fit all technical analysis models

#### get_available_models
- **Type**: function
- **File**: `custom/technical_analysis.py`
- **Parameters**: self
- **Description**: Get list of available technical analysis models

#### get_model_parameters
- **Type**: function
- **File**: `custom/technical_analysis.py`
- **Parameters**: self, model_name
- **Description**: Get parameters for a specific model

### Volatility Models

#### BaseVolatilityModel
- **Type**: class
- **File**: `custom/volatility_models.py`
- **Description**: Base class for volatility models

#### BaseVolatilityModel
- **Type**: class
- **File**: `custom/volatility.py`
- **Description**: Base class for volatility models

#### GARCHVolatilityModel
- **Type**: class
- **File**: `custom/volatility_models.py`
- **Parameters**: p, q, window
- **Description**: GARCH volatility model

#### MockGARCHModel
- **Type**: class
- **File**: `custom/volatility_models.py`
- **Parameters**: conditional_volatility
- **Description**: No description available

#### RealizedVolatilityModel
- **Type**: class
- **File**: `custom/volatility_models.py`
- **Parameters**: window, frequency
- **Description**: Realized volatility model

#### SimpleVolatilityModel
- **Type**: class
- **File**: `custom/volatility_models.py`
- **Parameters**: window, annualization_factor
- **Description**: Simple volatility calculation model

#### VolatilityClusteringModel
- **Type**: class
- **File**: `custom/volatility_models.py`
- **Parameters**: cluster_window, n_clusters
- **Description**: Volatility clustering model

#### VolatilityForecastingModel
- **Type**: class
- **File**: `custom/volatility_models.py`
- **Parameters**: forecast_horizon, model_type
- **Description**: Volatility forecasting model

#### VolatilityModel
- **Type**: class
- **File**: `custom/volatility_models.py`
- **Parameters**: volatility_window
- **Description**: Comprehensive volatility model for S&P 500 trading

#### VolatilityRegimeDetectionModel
- **Type**: class
- **File**: `custom/volatility_models.py`
- **Parameters**: regime_window, n_regimes
- **Description**: Volatility regime detection model

#### _calculate_forecast_confidence
- **Type**: function
- **File**: `custom/volatility_models.py`
- **Parameters**: self, rolling_vol
- **Description**: Calculate forecast confidence

#### _fit_garch_model
- **Type**: function
- **File**: `custom/volatility_models.py`
- **Parameters**: self, returns
- **Description**: Fit GARCH model to returns

#### _forecast_regime_volatility
- **Type**: function
- **File**: `custom/volatility_models.py`
- **Parameters**: self, rolling_vol, regime
- **Description**: Forecast volatility based on regime

#### _forecast_volatility
- **Type**: function
- **File**: `custom/volatility_models.py`
- **Parameters**: self, rolling_vol
- **Description**: Forecast volatility

#### _initialize_models
- **Type**: function
- **File**: `custom/volatility_models.py`
- **Parameters**: self
- **Description**: Initialize all volatility models

#### fit
- **Type**: function
- **File**: `custom/volatility_models.py`
- **Parameters**: self, data
- **Description**: Fit all volatility models

#### fit
- **Type**: function
- **File**: `custom/volatility.py`
- **Parameters**: self, data
- **Description**: No description available

#### fit_predict
- **Type**: function
- **File**: `custom/volatility.py`
- **Parameters**: self, high, low, close
- **Description**: No description available

#### get_available_models
- **Type**: function
- **File**: `custom/volatility_models.py`
- **Parameters**: self
- **Description**: Get list of available volatility models

#### get_model_parameters
- **Type**: function
- **File**: `custom/volatility_models.py`
- **Parameters**: self, model_name
- **Description**: Get parameters for a specific model

#### predict
- **Type**: function
- **File**: `custom/volatility.py`
- **Parameters**: self, data
- **Description**: No description available

