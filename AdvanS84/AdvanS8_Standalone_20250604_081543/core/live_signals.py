"""
Live Signal Generator Module
Real-time signal generation using 15-minute delayed Polygon data
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
from .data_loader import get_stock_data, add_technical_indicators
from .regime_logic import get_market_regime, get_regime_adaptive_rsi_threshold
from .exit_model import hybrid_exit_strategy, predict_exit_strategy

logger = logging.getLogger(__name__)

class LiveSignalGenerator:
    """
    Live signal generator using authentic 15-minute delayed market data
    """
    
    def __init__(self, polygon_api_key, fred_api_key):
        """Initialize live signal generator with API keys"""
        self.polygon_api_key = polygon_api_key
        self.fred_api_key = fred_api_key
        self.signal_log = []
        self.active_positions = {}
        
    def get_latest_market_data(self, symbol, lookback_days=5):
        """
        Get latest market data using Polygon's latest data endpoint
        
        Args:
            symbol: Stock symbol
            lookback_days: Days of historical data to fetch
        
        Returns:
            DataFrame: Latest market data with technical indicators
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        # Get 15-minute data for real-time analysis
        data = get_stock_data(symbol, start_date, end_date, self.polygon_api_key, '15min')
        
        if data is not None and len(data) > 50:
            # Add technical indicators to authentic price data
            data = add_technical_indicators(data)
            logger.debug(f"Retrieved {len(data)} 15-minute bars for {symbol}")
            return data
        
        logger.warning(f"Insufficient data retrieved for {symbol}")
        return None
    
    def scan_for_signals(self, universe, optimized_params=None):
        """
        Scan universe for live trading signals using authentic data
        
        Args:
            universe: List of stock symbols to scan
            optimized_params: Optimized trading parameters
        
        Returns:
            list: List of current trading signals
        """
        signals = []
        
        # Default parameters if none provided
        params = optimized_params or {
            'momentum_threshold': 0.025,
            'volume_multiplier': 2.0,
            'risk_tolerance': 0.03,
            'hold_period': 12
        }
        
        logger.info(f"Scanning {len(universe)} symbols for live signals...")
        
        for symbol in universe:
            try:
                signal = self._analyze_symbol_for_signal(symbol, params)
                if signal:
                    signals.append(signal)
                    logger.info(f"Signal detected: {symbol} - {signal['signal_type']}")
            
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
        
        # Log signals
        if signals:
            self._log_signals(signals)
        
        return signals
    
    def _analyze_symbol_for_signal(self, symbol, params):
        """
        Analyze individual symbol for trading signals using authentic data
        
        Args:
            symbol: Stock symbol
            params: Trading parameters
        
        Returns:
            dict or None: Signal information if signal detected
        """
        # Get latest market data
        data = self.get_latest_market_data(symbol)
        if data is None or len(data) < 50:
            return None
        
        current_bar = data.iloc[-1]
        current_time = data.index[-1]
        
        # Skip if essential indicators missing
        if pd.isna(current_bar['momentum_5']) or pd.isna(current_bar['rsi_14']):
            return None
        
        # Get current market regime from VIX data
        market_regime = self._get_current_market_regime()
        
        # Check for entry signal
        entry_signal = self._check_entry_signal(current_bar, params, market_regime)
        
        if entry_signal:
            signal_strength = self._calculate_signal_strength(current_bar, market_regime)
            
            return {
                'symbol': symbol,
                'signal_type': 'BUY',
                'timestamp': current_time,
                'price': current_bar['close'],
                'signal_strength': signal_strength,
                'market_regime': market_regime,
                'technical_data': {
                    'momentum_5': current_bar['momentum_5'],
                    'rsi_14': current_bar['rsi_14'],
                    'volume_ratio': current_bar['volume_ratio'],
                    'atr_pct_14': current_bar['atr_pct_14']
                },
                'recommended_action': 'ENTER_POSITION',
                'confidence': 'HIGH' if signal_strength > 0.7 else 'MEDIUM'
            }
        
        # Check exit signals for existing positions
        if symbol in self.active_positions:
            exit_signal = self._check_exit_signal(symbol, data, params, market_regime)
            if exit_signal:
                return exit_signal
        
        return None
    
    def _check_entry_signal(self, current_bar, params, market_regime):
        """Check if current conditions generate entry signal"""
        
        # Core entry criteria
        momentum_ok = current_bar['momentum_5'] >= params['momentum_threshold']
        volume_ok = current_bar['volume_ratio'] >= params['volume_multiplier']
        trend_ok = current_bar['close'] > current_bar['sma_20']
        
        # Regime-adaptive RSI threshold
        rsi_threshold = get_regime_adaptive_rsi_threshold(market_regime)
        rsi_ok = current_bar['rsi_14'] > rsi_threshold
        
        # Risk management check
        risk_ok = True
        if not pd.isna(current_bar['atr_pct_14']):
            risk_tolerance = params['risk_tolerance']
            if 'crisis' in market_regime:
                risk_tolerance *= 0.7  # Tighter risk in crisis
            risk_ok = current_bar['atr_pct_14'] < risk_tolerance
        
        return momentum_ok and volume_ok and trend_ok and rsi_ok and risk_ok
    
    def _check_exit_signal(self, symbol, data, params, market_regime):
        """Check exit signal for existing position"""
        
        position = self.active_positions.get(symbol)
        if not position:
            return None
        
        current_bar = data.iloc[-1]
        current_time = data.index[-1]
        
        # Calculate position metrics
        current_return = (current_bar['close'] - position['entry_price']) / position['entry_price']
        days_held = (current_time - position['entry_time']).days
        
        # Exit conditions
        exit_reason = None
        
        # Stop loss
        if current_return < -0.08:
            exit_reason = 'STOP_LOSS'
        
        # Take profit (regime adaptive)
        elif current_return > (0.15 if 'low_volatility' in market_regime else 0.25):
            exit_reason = 'TAKE_PROFIT'
        
        # Time-based exit
        elif days_held >= params.get('hold_period', 12):
            exit_reason = 'TIME_EXIT'
        
        # Momentum reversal
        elif current_bar['momentum_5'] < params['momentum_threshold'] * 0.5:
            exit_reason = 'MOMENTUM_REVERSAL'
        
        if exit_reason:
            return {
                'symbol': symbol,
                'signal_type': 'SELL',
                'timestamp': current_time,
                'price': current_bar['close'],
                'exit_reason': exit_reason,
                'position_return': current_return,
                'recommended_action': 'EXIT_POSITION',
                'confidence': 'HIGH'
            }
        
        return None
    
    def _calculate_signal_strength(self, current_bar, market_regime):
        """Calculate signal strength score (0-1)"""
        score = 0
        
        # Momentum component (0-0.3)
        momentum_score = min(current_bar['momentum_5'] / 0.05, 1.0) * 0.3
        score += momentum_score
        
        # Volume component (0-0.2)
        volume_score = min(current_bar['volume_ratio'] / 3.0, 1.0) * 0.2
        score += volume_score
        
        # RSI component (0-0.2)
        rsi_norm = (current_bar['rsi_14'] - 50) / 50
        rsi_score = max(0, rsi_norm) * 0.2
        score += rsi_score
        
        # Trend strength (0-0.2)
        trend_strength = (current_bar['close'] / current_bar['sma_20'] - 1)
        trend_score = min(abs(trend_strength) / 0.1, 1.0) * 0.2
        score += trend_score
        
        # Regime bonus/penalty (0-0.1)
        if 'low_volatility' in market_regime:
            score += 0.1  # Favorable conditions
        elif 'crisis' in market_regime:
            score *= 0.8  # Penalty for crisis
        
        return min(score, 1.0)
    
    def _get_current_market_regime(self):
        """Get current market regime using latest VIX data"""
        try:
            from .data_loader import get_vix_data
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5)
            
            vix_data = get_vix_data(start_date, end_date, self.fred_api_key)
            
            if vix_data is not None and len(vix_data) > 0:
                return get_market_regime(vix_data.index[-1], vix_data)
            
        except Exception as e:
            logger.warning(f"Error getting market regime: {e}")
        
        return 'moderate_volatility_normal'
    
    def _log_signals(self, signals):
        """Log signals to file for analysis"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Add to signal log
        for signal in signals:
            signal['logged_at'] = timestamp
            self.signal_log.append(signal)
        
        # Save to file
        signals_dir = 'data/signals'
        os.makedirs(signals_dir, exist_ok=True)
        
        filename = f"{signals_dir}/live_signals_{timestamp}.csv"
        
        if signals:
            signals_df = pd.DataFrame(signals)
            signals_df.to_csv(filename, index=False)
            logger.info(f"Signals logged to {filename}")
    
    def update_position(self, symbol, action, price, timestamp=None):
        """
        Update position tracking for signal monitoring
        
        Args:
            symbol: Stock symbol
            action: 'ENTER' or 'EXIT'
            price: Transaction price
            timestamp: Transaction timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if action == 'ENTER':
            self.active_positions[symbol] = {
                'entry_time': timestamp,
                'entry_price': price,
                'status': 'ACTIVE'
            }
            logger.info(f"Position opened: {symbol} @ ${price:.2f}")
        
        elif action == 'EXIT' and symbol in self.active_positions:
            position = self.active_positions[symbol]
            trade_return = (price - position['entry_price']) / position['entry_price']
            
            logger.info(f"Position closed: {symbol} @ ${price:.2f} - Return: {trade_return:.2%}")
            del self.active_positions[symbol]
    
    def get_signal_summary(self, hours=24):
        """Get summary of signals generated in last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_signals = [
            signal for signal in self.signal_log
            if datetime.strptime(signal['logged_at'], '%Y%m%d_%H%M%S') > cutoff_time
        ]
        
        return {
            'total_signals': len(recent_signals),
            'buy_signals': len([s for s in recent_signals if s['signal_type'] == 'BUY']),
            'sell_signals': len([s for s in recent_signals if s['signal_type'] == 'SELL']),
            'high_confidence': len([s for s in recent_signals if s.get('confidence') == 'HIGH']),
            'symbols_active': list(self.active_positions.keys()),
            'last_scan_time': datetime.now().isoformat()
        }

def create_live_signal_generator(polygon_api_key, fred_api_key):
    """Factory function to create live signal generator"""
    return LiveSignalGenerator(polygon_api_key, fred_api_key)