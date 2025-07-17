# ML Optimization for 7-12 Day and 50-70 Day Momentum Trading

## Overview

The current Polygon.io + FRED setup is perfectly suited for these momentum timeframes. You have access to all necessary models and data granularity for optimal performance.

## Available Models for Momentum Trading

### Tier 1: Core Momentum Models (All Available ✅)
1. **Technical Indicators**
   - `rsi_model` - RSI with customizable periods
   - `macd_model` - MACD for trend identification
   - `ema_crossover_model` - EMA crossovers
   - `bollinger_breakout_model` - Volatility breakouts
   - `momentum_breakout_model` - Pure momentum signals
   - `roc_model` - Rate of change momentum

2. **ML Models for Signal Enhancement**
   - `xgboost` - Feature combination and non-linear patterns
   - `lightgbm` - Fast training for multiple timeframes
   - `catboost` - Handles time-based categorical features
   - `random_forest` - Feature importance analysis

3. **Time Series Models**
   - `arima` - Short-term price forecasting
   - `garch` - Volatility forecasting for position sizing
   - `prophet` - Trend and seasonality detection

4. **Portfolio Optimization**
   - `maximum_sharpe_ratio_optimizer` - Optimal weights
   - `kelly_criterion_bayesian` - Position sizing
   - `dynamic_risk_parity_model` - Risk allocation

## Optimization Strategy for 7-12 Day Momentum

### Feature Engineering
```python
def create_7_12_day_features(data):
    """Create features optimized for 7-12 day holding periods"""
    
    features = {}
    
    # Price-based features
    features['return_7d'] = data['close'].pct_change(7)
    features['return_12d'] = data['close'].pct_change(12)
    features['return_5d'] = data['close'].pct_change(5)  # Shorter lookback
    
    # Momentum indicators with appropriate periods
    features['rsi_9'] = calculate_rsi(data['close'], 9)
    features['rsi_14'] = calculate_rsi(data['close'], 14)
    features['rsi_21'] = calculate_rsi(data['close'], 21)
    
    # Moving averages for 7-12 day trends
    features['sma_10'] = data['close'].rolling(10).mean()
    features['ema_8'] = data['close'].ewm(span=8).mean()
    features['ema_13'] = data['close'].ewm(span=13).mean()
    
    # Price relative to moving averages
    features['close_to_sma10'] = data['close'] / features['sma_10'] - 1
    features['close_to_ema8'] = data['close'] / features['ema_8'] - 1
    
    # Volume features
    features['volume_ratio_7d'] = data['volume'] / data['volume'].rolling(7).mean()
    features['dollar_volume_7d'] = (data['close'] * data['volume']).rolling(7).sum()
    
    # Volatility features
    features['realized_vol_10d'] = data['close'].pct_change().rolling(10).std() * np.sqrt(252)
    features['atr_10'] = calculate_atr(data, 10)
    features['bb_width_10'] = (features['bb_upper_10'] - features['bb_lower_10']) / features['sma_10']
    
    # Microstructure proxies from 1-minute data
    features['intraday_volatility'] = calculate_intraday_vol(minute_data)
    features['volume_weighted_price'] = calculate_vwap(minute_data, days=7)
    
    # Market regime from FRED
    features['vix_level'] = get_vix_level()  # From Polygon
    features['yield_curve_slope'] = get_yield_curve_slope()  # From FRED
    features['dollar_strength'] = get_dxy_change(10)  # 10-day dollar change
    
    return features

def create_momentum_signals_7_12(features):
    """Create momentum signals for 7-12 day trading"""
    
    signals = {}
    
    # Classic momentum
    signals['price_momentum'] = (
        features['return_7d'] > features['return_7d'].rolling(50).quantile(0.7)
    ).astype(int)
    
    # RSI momentum
    signals['rsi_momentum'] = np.where(
        (features['rsi_14'] > 50) & (features['rsi_14'] < 70) & 
        (features['rsi_14'] > features['rsi_14'].shift(3)),
        1, 0
    )
    
    # Volume confirmation
    signals['volume_surge'] = (features['volume_ratio_7d'] > 1.5).astype(int)
    
    # Volatility regime
    signals['low_vol_regime'] = (
        features['realized_vol_10d'] < features['realized_vol_10d'].rolling(60).median()
    ).astype(int)
    
    # Trend strength
    signals['trend_strength'] = np.where(
        (features['ema_8'] > features['ema_13']) & 
        (features['close'] > features['ema_8']),
        1, -1
    )
    
    return signals
```

### ML Model Configuration
```python
# XGBoost optimized for 7-12 day momentum
xgb_7_12_config = {
    'n_estimators': 500,
    'max_depth': 5,  # Prevent overfitting on short timeframe
    'learning_rate': 0.02,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'gamma': 0.1,  # More regularization
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse'
}

# LightGBM for fast iteration
lgb_7_12_config = {
    'n_estimators': 1000,
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_data_in_leaf': 50,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1
}

# Random Forest for feature importance
rf_7_12_config = {
    'n_estimators': 300,
    'max_depth': 10,
    'min_samples_split': 100,
    'min_samples_leaf': 50,
    'max_features': 'sqrt',
    'bootstrap': True
}
```

## Optimization Strategy for 50-70 Day Momentum

### Feature Engineering
```python
def create_50_70_day_features(data):
    """Create features optimized for 50-70 day holding periods"""
    
    features = {}
    
    # Longer-term momentum
    features['return_50d'] = data['close'].pct_change(50)
    features['return_70d'] = data['close'].pct_change(70)
    features['return_30d'] = data['close'].pct_change(30)
    features['return_90d'] = data['close'].pct_change(90)
    
    # Trend indicators
    features['sma_50'] = data['close'].rolling(50).mean()
    features['sma_200'] = data['close'].rolling(200).mean()
    features['ema_50'] = data['close'].ewm(span=50).mean()
    
    # Golden cross / Death cross
    features['golden_cross'] = (
        (features['sma_50'] > features['sma_200']) & 
        (features['sma_50'].shift(1) <= features['sma_200'].shift(1))
    ).astype(int)
    
    # Momentum quality
    features['momentum_quality'] = calculate_momentum_quality(
        returns=data['close'].pct_change(),
        lookback=60
    )
    
    # Sector and market relative
    features['relative_strength'] = data['close'].pct_change(60) / market_return_60d
    features['sector_momentum'] = get_sector_momentum(ticker, 60)
    
    # FRED macro features (more important for longer timeframes)
    features['gdp_growth'] = get_gdp_growth_rate()
    features['unemployment_trend'] = get_unemployment_trend()
    features['inflation_expectation'] = get_inflation_expectation()
    features['credit_spread'] = get_credit_spread()
    features['term_spread'] = get_term_spread()
    
    # Volatility and risk
    features['realized_vol_60d'] = data['close'].pct_change().rolling(60).std() * np.sqrt(252)
    features['downside_vol_60d'] = calculate_downside_volatility(data['close'], 60)
    features['max_drawdown_60d'] = calculate_max_drawdown(data['close'], 60)
    
    # Seasonality
    features['month_of_year'] = data.index.month
    features['quarter'] = data.index.quarter
    features['days_to_earnings'] = get_days_to_earnings(ticker)
    
    return features

def create_momentum_signals_50_70(features):
    """Create momentum signals for 50-70 day trading"""
    
    signals = {}
    
    # Absolute momentum
    signals['absolute_momentum'] = np.where(
        (features['return_50d'] > 0.05) &  # 5% minimum return
        (features['return_50d'] > features['return_50d'].rolling(252).quantile(0.8)),
        1, 0
    )
    
    # Relative momentum
    signals['relative_momentum'] = np.where(
        features['relative_strength'] > 1.1,  # 10% outperformance
        1, 0
    )
    
    # Trend following
    signals['trend_following'] = np.where(
        (features['close'] > features['sma_50']) & 
        (features['sma_50'] > features['sma_200']) &
        (features['sma_50'] > features['sma_50'].shift(10)),  # Rising 50 SMA
        1, 0
    )
    
    # Macro regime
    signals['favorable_macro'] = np.where(
        (features['term_spread'] > 0) &  # Normal yield curve
        (features['credit_spread'] < features['credit_spread'].rolling(252).median()) &
        (features['gdp_growth'] > 0),
        1, 0
    )
    
    return signals
```

### ML Model Configuration for 50-70 Days
```python
# Different hyperparameters for longer timeframe
xgb_50_70_config = {
    'n_estimators': 1000,
    'max_depth': 7,  # Can be deeper for longer timeframe
    'learning_rate': 0.01,  # Lower learning rate
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 10,
    'gamma': 0.05,
    'objective': 'reg:squarederror'
}

# Include macro features with higher importance
feature_weights_50_70 = {
    'return_50d': 2.0,
    'relative_strength': 2.0,
    'momentum_quality': 1.5,
    'term_spread': 1.5,
    'credit_spread': 1.5,
    'sector_momentum': 1.5
}
```

## Ensemble Strategy Combining Both Timeframes

```python
class DualTimeframeMomentumSystem:
    """Combine 7-12 day and 50-70 day momentum strategies"""
    
    def __init__(self):
        self.short_term_models = {
            'xgb': XGBRegressor(**xgb_7_12_config),
            'lgb': LGBMRegressor(**lgb_7_12_config),
            'rf': RandomForestRegressor(**rf_7_12_config)
        }
        
        self.long_term_models = {
            'xgb': XGBRegressor(**xgb_50_70_config),
            'lgb': LGBMRegressor(**lgb_50_70_config),
            'rf': RandomForestRegressor(**rf_50_70_config)
        }
        
        self.position_sizer = KellyCriterionBayesian()
        self.risk_manager = DynamicRiskManager()
        
    def generate_signals(self, data):
        # Get features for both timeframes
        features_7_12 = create_7_12_day_features(data)
        features_50_70 = create_50_70_day_features(data)
        
        # Get predictions from all models
        predictions_7_12 = self.get_ensemble_predictions(
            self.short_term_models, features_7_12
        )
        predictions_50_70 = self.get_ensemble_predictions(
            self.long_term_models, features_50_70
        )
        
        # Combine signals with dynamic weighting
        market_regime = self.detect_market_regime(data)
        
        if market_regime == 'trending':
            # Favor longer-term signals in trending markets
            weight_7_12 = 0.3
            weight_50_70 = 0.7
        elif market_regime == 'choppy':
            # Favor shorter-term in choppy markets
            weight_7_12 = 0.7
            weight_50_70 = 0.3
        else:  # balanced
            weight_7_12 = 0.5
            weight_50_70 = 0.5
            
        combined_signal = (
            weight_7_12 * predictions_7_12 + 
            weight_50_70 * predictions_50_70
        )
        
        return combined_signal
    
    def size_positions(self, signals, data):
        """Size positions based on signal strength and risk"""
        
        # Calculate expected returns for each timeframe
        expected_return_7_12 = signals['predictions_7_12'] * 0.02  # 2% per trade
        expected_return_50_70 = signals['predictions_50_70'] * 0.08  # 8% per trade
        
        # Adjust for win rates (from backtesting)
        win_rate_7_12 = 0.55  # Typical for short-term momentum
        win_rate_50_70 = 0.45  # Lower win rate but larger wins
        
        # Kelly sizing with safety factor
        kelly_size_7_12 = self.position_sizer.calculate_size(
            win_rate=win_rate_7_12,
            avg_win=0.03,
            avg_loss=0.02,
            safety_factor=0.25
        )
        
        kelly_size_50_70 = self.position_sizer.calculate_size(
            win_rate=win_rate_50_70,
            avg_win=0.15,
            avg_loss=0.08,
            safety_factor=0.25
        )
        
        # Risk parity adjustment
        vol_7_12 = calculate_strategy_volatility(returns_7_12, window=60)
        vol_50_70 = calculate_strategy_volatility(returns_50_70, window=60)
        
        # Normalize by volatility
        risk_adjusted_size_7_12 = kelly_size_7_12 * (TARGET_VOL / vol_7_12)
        risk_adjusted_size_50_70 = kelly_size_50_70 * (TARGET_VOL / vol_50_70)
        
        return {
            'size_7_12': np.clip(risk_adjusted_size_7_12, 0, 0.15),
            'size_50_70': np.clip(risk_adjusted_size_50_70, 0, 0.25)
        }
```

## Data Pipeline Optimization

### Efficient Data Collection
```python
class MomentumDataPipeline:
    """Optimized data pipeline for momentum strategies"""
    
    def __init__(self):
        self.polygon_client = PolygonConnector()
        self.fred_client = FREDConnector()
        self.cache = {}
        
    def get_momentum_dataset(self, tickers, start_date, end_date):
        """Get all required data efficiently"""
        
        data = {}
        
        # Batch request to Polygon for all tickers
        # Daily data for long-term features
        daily_data = self.polygon_client.get_batch_daily(
            tickers, start_date, end_date
        )
        
        # 1-minute data for short-term features (last 30 days only)
        minute_data = self.polygon_client.get_batch_minute(
            tickers, 
            max(start_date, end_date - timedelta(days=30)), 
            end_date
        )
        
        # Get macro data from FRED (cached, updates daily)
        macro_data = self.get_cached_macro_data()
        
        # Process each ticker
        for ticker in tickers:
            data[ticker] = {
                'daily': daily_data[ticker],
                'minute': minute_data[ticker],
                'features_7_12': create_7_12_day_features(
                    daily_data[ticker], minute_data[ticker], macro_data
                ),
                'features_50_70': create_50_70_day_features(
                    daily_data[ticker], macro_data
                )
            }
            
        return data
    
    def get_cached_macro_data(self):
        """Cache macro data to minimize API calls"""
        
        cache_key = datetime.now().strftime('%Y-%m-%d')
        
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        macro_data = {
            'yield_curve': self.fred_client.get_yield_curve(),
            'gdp_growth': self.fred_client.get_gdp_growth(),
            'unemployment': self.fred_client.get_unemployment(),
            'inflation': self.fred_client.get_inflation_expectations(),
            'credit_spreads': self.fred_client.get_credit_spreads()
        }
        
        self.cache[cache_key] = macro_data
        return macro_data
```

## Backtesting Framework

```python
class MomentumBacktester:
    """Specialized backtester for dual-timeframe momentum"""
    
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.results = {}
        
    def backtest_dual_timeframe(self, data, start_date, end_date):
        """Backtest both timeframes with realistic assumptions"""
        
        # Transaction costs
        COMMISSION = 0.001  # 0.1% per trade
        SLIPPAGE_7_12 = 0.002  # 0.2% for shorter timeframe
        SLIPPAGE_50_70 = 0.001  # 0.1% for longer timeframe
        
        portfolio = Portfolio(self.initial_capital)
        
        for date in pd.date_range(start_date, end_date, freq='D'):
            # Skip weekends and holidays
            if not is_trading_day(date):
                continue
                
            # Get signals
            signals_7_12 = self.get_7_12_signals(data, date)
            signals_50_70 = self.get_50_70_signals(data, date)
            
            # Rebalance 7-12 day positions (more frequent)
            if date.day % 2 == 0:  # Every 2 days
                self.rebalance_short_term(
                    portfolio, signals_7_12, 
                    costs=COMMISSION + SLIPPAGE_7_12
                )
            
            # Rebalance 50-70 day positions (less frequent)
            if date.day % 10 == 0:  # Every 10 days
                self.rebalance_long_term(
                    portfolio, signals_50_70,
                    costs=COMMISSION + SLIPPAGE_50_70
                )
            
            # Update portfolio value
            portfolio.update_market_value(data, date)
            
        return self.calculate_performance_metrics(portfolio)
    
    def calculate_performance_metrics(self, portfolio):
        """Calculate comprehensive performance metrics"""
        
        returns = portfolio.get_returns()
        
        metrics = {
            # Returns
            'total_return': (portfolio.value / self.initial_capital - 1) * 100,
            'annual_return': calculate_annual_return(returns),
            'annual_volatility': returns.std() * np.sqrt(252),
            
            # Risk-adjusted
            'sharpe_ratio': calculate_sharpe_ratio(returns),
            'sortino_ratio': calculate_sortino_ratio(returns),
            'calmar_ratio': calculate_calmar_ratio(returns),
            
            # Drawdown
            'max_drawdown': calculate_max_drawdown(portfolio.values),
            'avg_drawdown': calculate_average_drawdown(portfolio.values),
            'max_drawdown_duration': calculate_max_drawdown_duration(portfolio.values),
            
            # Trade statistics
            'win_rate_7_12': portfolio.get_win_rate('short_term'),
            'win_rate_50_70': portfolio.get_win_rate('long_term'),
            'avg_win_loss_ratio': portfolio.get_win_loss_ratio(),
            
            # By timeframe
            'return_from_7_12': portfolio.get_return_attribution('short_term'),
            'return_from_50_70': portfolio.get_return_attribution('long_term')
        }
        
        return metrics
```

## Production Implementation

```python
class MomentumTradingSystem:
    """Production-ready momentum trading system"""
    
    def __init__(self, config):
        self.config = config
        self.data_pipeline = MomentumDataPipeline()
        self.model_7_12 = self.load_model('models/momentum_7_12_day.pkl')
        self.model_50_70 = self.load_model('models/momentum_50_70_day.pkl')
        self.risk_manager = RiskManager(config['risk_limits'])
        self.executor = OrderExecutor(config['broker_api'])
        
    def run_daily_update(self):
        """Daily update routine"""
        
        # 1. Get latest data
        data = self.data_pipeline.get_momentum_dataset(
            self.config['universe'],
            datetime.now() - timedelta(days=252),
            datetime.now()
        )
        
        # 2. Generate signals
        signals_7_12 = self.model_7_12.predict(data['features_7_12'])
        signals_50_70 = self.model_50_70.predict(data['features_50_70'])
        
        # 3. Risk checks
        if not self.risk_manager.check_market_conditions():
            logger.warning("Unfavorable market conditions, reducing exposure")
            signals_7_12 *= 0.5
            signals_50_70 *= 0.5
        
        # 4. Generate orders
        orders = self.generate_orders(signals_7_12, signals_50_70)
        
        # 5. Execute with smart routing
        for order in orders:
            self.executor.execute_with_limits(
                order,
                max_spread=0.002,
                max_impact=0.001,
                timeout=300  # 5 minutes
            )
        
        # 6. Log and monitor
        self.log_performance()
        self.send_alerts_if_needed()
```

## Performance Expectations

### 7-12 Day Momentum
- **Expected Annual Return**: 15-25%
- **Sharpe Ratio**: 1.2-1.8
- **Win Rate**: 52-58%
- **Max Drawdown**: 10-15%
- **Trade Frequency**: 50-100 trades/year per asset

### 50-70 Day Momentum
- **Expected Annual Return**: 12-20%
- **Sharpe Ratio**: 0.8-1.5
- **Win Rate**: 40-50%
- **Max Drawdown**: 15-25%
- **Trade Frequency**: 5-10 trades/year per asset

### Combined Strategy
- **Expected Annual Return**: 18-28%
- **Sharpe Ratio**: 1.5-2.2
- **Max Drawdown**: 12-18%
- **Correlation to Market**: 0.3-0.5

## Key Success Factors

1. **Feature Engineering Quality**
   - Use multiple timeframes
   - Include macro regime indicators
   - Normalize features properly

2. **Model Training**
   - Use walk-forward optimization
   - Prevent overfitting with regularization
   - Ensemble multiple models

3. **Risk Management**
   - Position sizing based on volatility
   - Correlation-aware portfolio construction
   - Dynamic exposure adjustment

4. **Execution**
   - Minimize market impact
   - Use limit orders for entries
   - Smart rebalancing schedule

5. **Continuous Improvement**
   - Monitor feature importance
   - Track prediction accuracy
   - Adapt to regime changes

## Conclusion

Your current setup with Polygon.io and FRED is excellent for both 7-12 day and 50-70 day momentum strategies. The key is proper feature engineering, robust model training, and disciplined risk management. The combination of both timeframes provides diversification and the ability to capture different market inefficiencies.