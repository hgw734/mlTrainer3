#!/usr/bin/env python3
"""
Practical Implementation Example: Dual-Timeframe Momentum Strategy
Using Polygon.io and FRED data for 7-12 day and 50-70 day momentum
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import connectors (assuming they exist in your system)
from polygon_connector import PolygonConnector
from fred_connector import FREDConnector

class DualMomentumStrategy:
    """Complete implementation of dual-timeframe momentum strategy"""
    
    def __init__(self, universe=['SPY', 'QQQ', 'IWM', 'DIA', 'EFA', 'EEM']):
        self.universe = universe
        self.polygon = PolygonConnector()
        self.fred = FREDConnector()
        
        # Initialize models
        self.models_7_12 = self._initialize_short_term_models()
        self.models_50_70 = self._initialize_long_term_models()
        
        # Scalers for feature normalization
        self.scaler_7_12 = StandardScaler()
        self.scaler_50_70 = StandardScaler()
        
        # Performance tracking
        self.performance_history = []
        
    def _initialize_short_term_models(self):
        """Initialize models for 7-12 day momentum"""
        return {
            'xgb': xgb.XGBRegressor(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.02,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                gamma=0.1,
                random_state=42
            ),
            'lgb': lgb.LGBMRegressor(
                n_estimators=1000,
                num_leaves=31,
                learning_rate=0.05,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                min_data_in_leaf=50,
                lambda_l1=0.1,
                lambda_l2=0.1,
                random_state=42
            ),
            'rf': RandomForestRegressor(
                n_estimators=300,
                max_depth=10,
                min_samples_split=100,
                min_samples_leaf=50,
                max_features='sqrt',
                random_state=42
            )
        }
    
    def _initialize_long_term_models(self):
        """Initialize models for 50-70 day momentum"""
        return {
            'xgb': xgb.XGBRegressor(
                n_estimators=1000,
                max_depth=7,
                learning_rate=0.01,
                subsample=0.7,
                colsample_bytree=0.7,
                min_child_weight=10,
                gamma=0.05,
                random_state=42
            ),
            'lgb': lgb.LGBMRegressor(
                n_estimators=1500,
                num_leaves=63,
                learning_rate=0.02,
                feature_fraction=0.7,
                bagging_fraction=0.7,
                bagging_freq=7,
                min_data_in_leaf=100,
                lambda_l1=0.05,
                lambda_l2=0.05,
                random_state=42
            ),
            'rf': RandomForestRegressor(
                n_estimators=500,
                max_depth=15,
                min_samples_split=50,
                min_samples_leaf=25,
                max_features='sqrt',
                random_state=42
            )
        }
    
    def fetch_data(self, start_date, end_date):
        """Fetch all required data from Polygon and FRED"""
        data = {}
        
        # Get daily data for all tickers
        for ticker in self.universe:
            daily_data = self.polygon.get_ohlcv_data(ticker, start_date, end_date)
            
            # Get minute data for last 30 days (for short-term features)
            minute_start = max(start_date, end_date - timedelta(days=30))
            minute_data = self.polygon.get_minute_data(ticker, minute_start, end_date)
            
            data[ticker] = {
                'daily': daily_data,
                'minute': minute_data
            }
        
        # Get macro data from FRED
        macro_data = {
            'yield_2y': self.fred.get_series_data('DGS2', start_date, end_date),
            'yield_10y': self.fred.get_series_data('DGS10', start_date, end_date),
            'vix': self.polygon.get_ohlcv_data('VIX', start_date, end_date),
            'unemployment': self.fred.get_series_data('UNRATE', start_date, end_date),
            'gdp_growth': self.fred.get_series_data('A191RL1Q225SBEA', start_date, end_date),
            'credit_spread': self._calculate_credit_spread(start_date, end_date)
        }
        
        return data, macro_data
    
    def _calculate_credit_spread(self, start_date, end_date):
        """Calculate credit spread from FRED data"""
        baa = self.fred.get_series_data('BAAFFM', start_date, end_date)
        aaa = self.fred.get_series_data('AAAFFM', start_date, end_date)
        return baa - aaa
    
    def create_features_7_12(self, ticker_data, macro_data):
        """Create features for 7-12 day momentum"""
        df = ticker_data['daily'].copy()
        
        # Price momentum features
        df['return_5d'] = df['close'].pct_change(5)
        df['return_7d'] = df['close'].pct_change(7)
        df['return_10d'] = df['close'].pct_change(10)
        df['return_12d'] = df['close'].pct_change(12)
        
        # Technical indicators
        df['rsi_9'] = self._calculate_rsi(df['close'], 9)
        df['rsi_14'] = self._calculate_rsi(df['close'], 14)
        
        # Moving averages
        df['sma_10'] = df['close'].rolling(10).mean()
        df['ema_8'] = df['close'].ewm(span=8).mean()
        df['ema_13'] = df['close'].ewm(span=13).mean()
        
        # Price position
        df['close_to_sma10'] = (df['close'] / df['sma_10']) - 1
        df['ema_8_13_spread'] = (df['ema_8'] / df['ema_13']) - 1
        
        # Volume features
        df['volume_ratio_7d'] = df['volume'] / df['volume'].rolling(7).mean()
        df['dollar_volume_7d'] = (df['close'] * df['volume']).rolling(7).mean()
        
        # Volatility
        df['realized_vol_10d'] = df['close'].pct_change().rolling(10).std() * np.sqrt(252)
        df['atr_10'] = self._calculate_atr(df, 10)
        
        # Intraday features from minute data
        if ticker_data['minute'] is not None and len(ticker_data['minute']) > 0:
            df['intraday_vol'] = self._calculate_intraday_volatility(ticker_data['minute'])
            df['vwap_ratio'] = self._calculate_vwap_ratio(ticker_data['minute'], df)
        
        # Macro features
        df['vix_level'] = macro_data['vix']['close'].reindex(df.index, method='ffill')
        df['yield_curve'] = (
            macro_data['yield_10y'].reindex(df.index, method='ffill') - 
            macro_data['yield_2y'].reindex(df.index, method='ffill')
        )
        
        # Market regime
        df['vol_regime'] = df['realized_vol_10d'] / df['realized_vol_10d'].rolling(60).mean()
        
        return df
    
    def create_features_50_70(self, ticker_data, macro_data):
        """Create features for 50-70 day momentum"""
        df = ticker_data['daily'].copy()
        
        # Long-term momentum
        df['return_30d'] = df['close'].pct_change(30)
        df['return_50d'] = df['close'].pct_change(50)
        df['return_60d'] = df['close'].pct_change(60)
        df['return_70d'] = df['close'].pct_change(70)
        df['return_90d'] = df['close'].pct_change(90)
        
        # Trend indicators
        df['sma_50'] = df['close'].rolling(50).mean()
        df['sma_200'] = df['close'].rolling(200).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        
        # Trend strength
        df['close_to_sma50'] = (df['close'] / df['sma_50']) - 1
        df['close_to_sma200'] = (df['close'] / df['sma_200']) - 1
        df['sma50_to_sma200'] = (df['sma_50'] / df['sma_200']) - 1
        
        # Momentum quality (smoothness of returns)
        df['momentum_quality'] = self._calculate_momentum_quality(df['close'], 60)
        
        # Volatility and risk
        df['realized_vol_60d'] = df['close'].pct_change().rolling(60).std() * np.sqrt(252)
        df['downside_vol_60d'] = self._calculate_downside_volatility(df['close'], 60)
        df['max_dd_60d'] = self._calculate_rolling_max_drawdown(df['close'], 60)
        
        # Relative strength
        spy_returns = self.polygon.get_ohlcv_data('SPY', df.index[0], df.index[-1])
        spy_returns = spy_returns['close'].pct_change(60).reindex(df.index, method='ffill')
        df['relative_strength'] = df['return_60d'] / spy_returns
        
        # Macro features (more important for longer timeframes)
        df['gdp_growth'] = macro_data['gdp_growth'].reindex(df.index, method='ffill')
        df['unemployment_trend'] = macro_data['unemployment'].pct_change(3).reindex(df.index, method='ffill')
        df['credit_spread'] = macro_data['credit_spread'].reindex(df.index, method='ffill')
        
        # Seasonality
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['day_of_year'] = df.index.dayofyear
        
        return df
    
    def _calculate_rsi(self, prices, period):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, df, period):
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean()
    
    def _calculate_intraday_volatility(self, minute_data):
        """Calculate intraday volatility from minute data"""
        # Group by day and calculate daily volatility from minute returns
        minute_returns = minute_data['close'].pct_change()
        daily_vol = minute_returns.groupby(minute_data.index.date).std() * np.sqrt(390)
        return daily_vol.mean()
    
    def _calculate_vwap_ratio(self, minute_data, daily_data):
        """Calculate VWAP ratio"""
        # Calculate daily VWAP from minute data
        minute_data['dollar_volume'] = minute_data['close'] * minute_data['volume']
        daily_vwap = (
            minute_data.groupby(minute_data.index.date)['dollar_volume'].sum() /
            minute_data.groupby(minute_data.index.date)['volume'].sum()
        )
        
        # Calculate ratio of close to VWAP
        vwap_ratio = daily_data['close'] / daily_vwap.reindex(daily_data.index)
        return vwap_ratio.fillna(1.0)
    
    def _calculate_momentum_quality(self, prices, lookback):
        """Calculate momentum quality (consistency of returns)"""
        returns = prices.pct_change()
        rolling_returns = returns.rolling(lookback)
        
        # Ratio of positive days
        positive_days = (rolling_returns > 0).sum()
        total_days = rolling_returns.count()
        hit_rate = positive_days / total_days
        
        # Smoothness (lower volatility of returns)
        return_vol = rolling_returns.std()
        avg_return = rolling_returns.mean()
        smoothness = avg_return / (return_vol + 1e-6)
        
        # Combined quality score
        quality = hit_rate * smoothness
        return quality
    
    def _calculate_downside_volatility(self, prices, window):
        """Calculate downside volatility"""
        returns = prices.pct_change()
        downside_returns = returns.where(returns < 0, 0)
        return downside_returns.rolling(window).std() * np.sqrt(252)
    
    def _calculate_rolling_max_drawdown(self, prices, window):
        """Calculate rolling maximum drawdown"""
        rolling_max = prices.rolling(window).max()
        drawdown = (prices - rolling_max) / rolling_max
        return drawdown.rolling(window).min()
    
    def train_models(self, train_data, train_start, train_end):
        """Train all models on historical data"""
        
        # Prepare training features
        X_train_7_12 = []
        X_train_50_70 = []
        y_train_7_12 = []
        y_train_50_70 = []
        
        for ticker in self.universe:
            # Create features
            features_7_12 = self.create_features_7_12(train_data[ticker], train_data['macro'])
            features_50_70 = self.create_features_50_70(train_data[ticker], train_data['macro'])
            
            # Define feature columns
            feature_cols_7_12 = [
                'return_5d', 'return_7d', 'return_10d', 'rsi_9', 'rsi_14',
                'close_to_sma10', 'ema_8_13_spread', 'volume_ratio_7d',
                'realized_vol_10d', 'vol_regime', 'vix_level', 'yield_curve'
            ]
            
            feature_cols_50_70 = [
                'return_30d', 'return_50d', 'return_60d', 'close_to_sma50',
                'close_to_sma200', 'sma50_to_sma200', 'momentum_quality',
                'realized_vol_60d', 'downside_vol_60d', 'relative_strength',
                'gdp_growth', 'credit_spread', 'month', 'quarter'
            ]
            
            # Create targets (future returns)
            features_7_12['target'] = features_7_12['close'].pct_change(10).shift(-10)
            features_50_70['target'] = features_50_70['close'].pct_change(60).shift(-60)
            
            # Remove NaN values
            features_7_12 = features_7_12.dropna()
            features_50_70 = features_50_70.dropna()
            
            # Add to training sets
            if len(features_7_12) > 0:
                X_train_7_12.append(features_7_12[feature_cols_7_12])
                y_train_7_12.append(features_7_12['target'])
            
            if len(features_50_70) > 0:
                X_train_50_70.append(features_50_70[feature_cols_50_70])
                y_train_50_70.append(features_50_70['target'])
        
        # Combine all training data
        X_train_7_12 = pd.concat(X_train_7_12)
        y_train_7_12 = pd.concat(y_train_7_12)
        X_train_50_70 = pd.concat(X_train_50_70)
        y_train_50_70 = pd.concat(y_train_50_70)
        
        # Scale features
        X_train_7_12_scaled = self.scaler_7_12.fit_transform(X_train_7_12)
        X_train_50_70_scaled = self.scaler_50_70.fit_transform(X_train_50_70)
        
        # Train models
        print("Training 7-12 day models...")
        for name, model in self.models_7_12.items():
            print(f"  Training {name}...")
            model.fit(X_train_7_12_scaled, y_train_7_12)
        
        print("Training 50-70 day models...")
        for name, model in self.models_50_70.items():
            print(f"  Training {name}...")
            model.fit(X_train_50_70_scaled, y_train_50_70)
        
        print("Model training complete!")
    
    def generate_signals(self, current_data):
        """Generate trading signals for current date"""
        signals = {}
        
        for ticker in self.universe:
            # Create features
            features_7_12 = self.create_features_7_12(current_data[ticker], current_data['macro'])
            features_50_70 = self.create_features_50_70(current_data[ticker], current_data['macro'])
            
            # Get latest features
            latest_7_12 = features_7_12.iloc[-1]
            latest_50_70 = features_50_70.iloc[-1]
            
            # Make predictions
            pred_7_12 = self._ensemble_predict(self.models_7_12, latest_7_12, self.scaler_7_12)
            pred_50_70 = self._ensemble_predict(self.models_50_70, latest_50_70, self.scaler_50_70)
            
            # Combine signals based on market regime
            market_regime = self._detect_market_regime(features_50_70)
            
            if market_regime == 'trending':
                weight_7_12 = 0.3
                weight_50_70 = 0.7
            elif market_regime == 'choppy':
                weight_7_12 = 0.7
                weight_50_70 = 0.3
            else:
                weight_7_12 = 0.5
                weight_50_70 = 0.5
            
            combined_signal = weight_7_12 * pred_7_12 + weight_50_70 * pred_50_70
            
            # Position sizing
            position_size = self._calculate_position_size(
                combined_signal,
                features_7_12['realized_vol_10d'].iloc[-1],
                features_50_70['max_dd_60d'].iloc[-1]
            )
            
            signals[ticker] = {
                'signal': combined_signal,
                'position_size': position_size,
                'pred_7_12': pred_7_12,
                'pred_50_70': pred_50_70,
                'regime': market_regime
            }
        
        return signals
    
    def _ensemble_predict(self, models, features, scaler):
        """Make ensemble prediction from multiple models"""
        predictions = []
        
        feature_array = features.values.reshape(1, -1)
        feature_scaled = scaler.transform(feature_array)
        
        for name, model in models.items():
            pred = model.predict(feature_scaled)[0]
            predictions.append(pred)
        
        # Simple average ensemble
        return np.mean(predictions)
    
    def _detect_market_regime(self, features):
        """Detect current market regime"""
        # Simple regime detection based on trend and volatility
        trend_strength = features['sma50_to_sma200'].iloc[-1]
        volatility = features['realized_vol_60d'].iloc[-1]
        vol_percentile = (features['realized_vol_60d'] < volatility).mean()
        
        if trend_strength > 0.02 and vol_percentile < 0.5:
            return 'trending'
        elif vol_percentile > 0.7:
            return 'choppy'
        else:
            return 'neutral'
    
    def _calculate_position_size(self, signal, short_vol, max_dd):
        """Calculate position size based on Kelly criterion and risk"""
        # Expected return based on signal strength
        expected_return = signal * 0.05  # 5% base return
        
        # Win rate estimation (from backtesting)
        win_rate = 0.5 + min(abs(signal) * 0.2, 0.2)  # 50-70% win rate
        
        # Kelly fraction
        avg_win = 0.08
        avg_loss = 0.05
        kelly_f = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        
        # Risk adjustment
        vol_scalar = 0.15 / max(short_vol, 0.1)  # Target 15% volatility
        dd_scalar = 0.15 / max(abs(max_dd), 0.1)  # Target 15% max drawdown
        
        # Final position size
        position_size = kelly_f * 0.25 * vol_scalar * dd_scalar  # 25% Kelly
        
        return np.clip(position_size, -0.2, 0.2)  # Max 20% position

# Example usage
if __name__ == "__main__":
    # Initialize strategy
    strategy = DualMomentumStrategy(
        universe=['SPY', 'QQQ', 'IWM', 'DIA', 'EFA', 'EEM', 'TLT', 'GLD']
    )
    
    # Set dates
    train_start = datetime(2020, 1, 1)
    train_end = datetime(2023, 1, 1)
    test_start = datetime(2023, 1, 1)
    test_end = datetime.now()
    
    print("Fetching training data...")
    train_data, macro_data = strategy.fetch_data(train_start, train_end)
    train_data['macro'] = macro_data
    
    print("Training models...")
    strategy.train_models(train_data, train_start, train_end)
    
    print("Generating current signals...")
    current_data, current_macro = strategy.fetch_data(
        test_end - timedelta(days=252),
        test_end
    )
    current_data['macro'] = current_macro
    
    signals = strategy.generate_signals(current_data)
    
    # Display signals
    print("\nCurrent Trading Signals:")
    print("-" * 60)
    for ticker, signal_data in signals.items():
        print(f"{ticker}:")
        print(f"  Combined Signal: {signal_data['signal']:.4f}")
        print(f"  Position Size: {signal_data['position_size']:.2%}")
        print(f"  7-12 Day Pred: {signal_data['pred_7_12']:.4f}")
        print(f"  50-70 Day Pred: {signal_data['pred_50_70']:.4f}")
        print(f"  Market Regime: {signal_data['regime']}")
        print()