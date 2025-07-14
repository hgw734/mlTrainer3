# mlTrainer Models Documentation

## Overview

The mlTrainer system includes a comprehensive suite of trading models, from traditional rule-based strategies to advanced machine learning algorithms. All models are designed to work with real financial data through the DataPipeline and include proper backtesting capabilities.

## Implemented Models

### 1. Rule-Based Trading Models

#### MomentumBreakoutEnhanced
- **Type**: Trend-following strategy
- **Signals**: Identifies strong momentum moves with volume confirmation
- **Key Features**:
  - Price breakout detection above rolling highs
  - Volume surge confirmation (1.5x average)
  - ADX trend strength filter
  - Volatility-adjusted position sizing

#### MeanReversionEnhanced
- **Type**: Counter-trend strategy
- **Signals**: Trades reversals at statistical extremes
- **Key Features**:
  - Bollinger Bands (2 standard deviations)
  - RSI oversold/overbought conditions
  - Z-score based entry/exit
  - Multiple timeframe analysis

#### VolatilityRegimeEnhanced
- **Type**: Adaptive strategy
- **Signals**: Adjusts strategy based on market volatility
- **Key Features**:
  - Real-time volatility calculation
  - Low/Medium/High volatility regimes
  - Regime-specific trading rules
  - Dynamic stop-loss adjustment

### 2. Machine Learning Models

#### RandomForestEnhanced
- **Algorithm**: Ensemble of decision trees
- **Features**: 40+ technical and market microstructure features
- **Capabilities**:
  - Classification (buy/hold/sell) or regression
  - Feature importance ranking
  - Cross-validation support
  - Handles non-linear relationships

#### XGBoostEnhanced
- **Algorithm**: Gradient boosting
- **Features**: 60+ features including seasonal patterns
- **Capabilities**:
  - Superior performance on tabular data
  - Built-in regularization
  - Early stopping to prevent overfitting
  - Feature importance with multiple metrics

#### LSTMEnhanced
- **Algorithm**: Long Short-Term Memory neural network
- **Architecture**: 3 LSTM layers + dense layers
- **Capabilities**:
  - Sequence learning (60-day windows)
  - Captures temporal dependencies
  - Dropout for regularization
  - Both classification and regression modes

### 3. Advanced Trading Strategies

#### PairsTradingEnhanced
- **Type**: Statistical arbitrage
- **Method**: Cointegration-based spread trading
- **Features**:
  - Engle-Granger cointegration test
  - Dynamic hedge ratio calculation
  - Ornstein-Uhlenbeck mean reversion
  - Z-score based entry/exit signals

#### MarketRegimeDetector
- **Algorithm**: Hidden Markov Model
- **Regimes**: Bull, Bear, Sideways
- **Features**:
  - Probabilistic regime identification
  - Regime-specific trading strategies
  - Transition probability matrix
  - Real-time regime updates

#### PortfolioOptimizer
- **Methods**: Multiple optimization techniques
- **Supported Optimizations**:
  - Maximum Sharpe Ratio
  - Minimum Volatility
  - Hierarchical Risk Parity
  - Risk Parity
  - Black-Litterman
  - CVaR Optimization
- **Features**:
  - Position constraints
  - Discrete allocation
  - Rebalancing signals
  - Efficient frontier visualization

## Usage Examples

### Training a Random Forest Model

```python
from models.random_forest_enhanced import RandomForestEnhanced
from core.data_pipeline import DataPipeline

# Get data
pipeline = DataPipeline()
data = pipeline.fetch_stock_data('AAPL', start_date='2020-01-01')
data = pipeline.add_all_features(data)

# Train model
model = RandomForestEnhanced(n_estimators=100, max_depth=10)
model.fit(data)

# Generate signals
signals = model.predict(data)

# Get feature importance
importance = model.feature_importance_
```

### Running a Backtest

```python
from core.backtesting_engine import BacktestingEngine, BacktestConfig

# Configure backtest
config = BacktestConfig(
    initial_capital=100000,
    commission=0.001,
    slippage=0.0005,
    stop_loss=0.02
)

# Run backtest
engine = BacktestingEngine(config)
results = engine.run(data, signals)

# View metrics
print(f"Total Return: {results['metrics']['total_return']:.2%}")
print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['metrics']['max_drawdown']:.2%}")
```

### Portfolio Optimization

```python
from models.portfolio_optimizer import PortfolioOptimizer

# Prepare multi-asset data
assets_data = {
    'AAPL': pipeline.fetch_stock_data('AAPL'),
    'GOOGL': pipeline.fetch_stock_data('GOOGL'),
    'MSFT': pipeline.fetch_stock_data('MSFT'),
    'AMZN': pipeline.fetch_stock_data('AMZN')
}

# Optimize portfolio
optimizer = PortfolioOptimizer(optimization_method='max_sharpe')
optimizer.fit(assets_data)

# Get optimal weights
weights = optimizer.weights
print("Optimal Portfolio Weights:")
for asset, weight in weights.items():
    print(f"{asset}: {weight:.2%}")
```

## Model Performance Metrics

All models are evaluated using comprehensive metrics:

### Return Metrics
- Total Return
- Annual Return
- Risk-Adjusted Return (Sharpe Ratio)

### Risk Metrics
- Volatility
- Maximum Drawdown
- Value at Risk (VaR)
- Conditional VaR (CVaR)

### Trading Metrics
- Win Rate
- Profit Factor
- Average Win/Loss
- Trade Duration

## Best Practices

1. **Data Quality**: Always use clean, adjusted price data
2. **Validation**: Use walk-forward analysis for time series
3. **Risk Management**: Implement proper position sizing and stop-losses
4. **Diversification**: Combine multiple models for robustness
5. **Monitoring**: Track model performance and retrain regularly

## Model Selection Guide

| Market Condition | Recommended Models |
|-----------------|-------------------|
| Trending | MomentumBreakout, XGBoost |
| Ranging | MeanReversion, PairsTrading |
| High Volatility | VolatilityRegime, MinVolPortfolio |
| Low Volatility | LSTM, RandomForest |
| Multi-Asset | PortfolioOptimizer, MarketRegime |

## Compliance Integration

All models integrate with the mlTrainer compliance system:
- Real-time validation of predictions
- Audit trail for all decisions
- Risk limit enforcement
- Regulatory reporting support

## Future Enhancements

1. **Reinforcement Learning**: Deep Q-Networks for dynamic strategy selection
2. **Alternative Data**: Sentiment analysis, satellite imagery
3. **High-Frequency Models**: Microsecond execution strategies
4. **Quantum Computing**: Quantum annealing for portfolio optimization
5. **Explainable AI**: SHAP/LIME integration for model interpretability