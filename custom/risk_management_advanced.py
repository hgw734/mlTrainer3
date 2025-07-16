"""
Advanced Risk Management Models
================================
Implements sophisticated risk management strategies for position sizing,
stop loss management, and portfolio risk control.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')


@dataclass
class RiskMetrics:
    """Risk metrics container"""
    position_size: float
    stop_loss: float
    risk_amount: float
    confidence: float
    method: str
    timestamp: datetime
    additional_info: Dict[str, Any] = None


class BaseRiskModel(ABC):
    """Base class for risk management models"""
    
    def __init__(self, lookback_period: int = 252):
        self.lookback_period = lookback_period
        self.last_calculation = None
        
    @abstractmethod
    def calculate_risk_metrics(self, data: pd.DataFrame, **kwargs) -> RiskMetrics:
        """Calculate risk metrics"""
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data"""
        if data is None or data.empty:
            return False
        if len(data) < self.lookback_period:
            return False
        return True


class KellyCriterionModel(BaseRiskModel):
    """
    Optimal position sizing based on:
    - Historical win rate
    - Average win/loss ratio
    - Confidence intervals
    - Half-Kelly for safety
    """
    
    def __init__(self, lookback_period: int = 252, kelly_fraction: float = 0.5):
        super().__init__(lookback_period)
        self.kelly_fraction = kelly_fraction  # Half-Kelly for safety
        
    def calculate_risk_metrics(self, data: pd.DataFrame, capital: float = 100000, 
                             current_price: float = None, **kwargs) -> RiskMetrics:
        """Calculate optimal position size using Kelly Criterion"""
        if not self.validate_data(data):
            return self._default_metrics()
        
        # Calculate returns
        returns = data['close'].pct_change().dropna()
        returns = returns.tail(self.lookback_period)
        
        # Calculate win/loss statistics
        winning_returns = returns[returns > 0]
        losing_returns = returns[returns < 0]
        
        if len(winning_returns) == 0 or len(losing_returns) == 0:
            return self._default_metrics()
        
        win_rate = len(winning_returns) / len(returns)
        avg_win = winning_returns.mean()
        avg_loss = abs(losing_returns.mean())
        
        # Kelly formula: f = (p*b - q) / b
        # where p = win rate, q = loss rate, b = win/loss ratio
        b = avg_win / avg_loss if avg_loss > 0 else 0
        q = 1 - win_rate
        
        if b > 0:
            kelly = (win_rate * b - q) / b
            # Apply safety fraction
            kelly = kelly * self.kelly_fraction
            # Cap at maximum 25% of capital
            kelly = min(max(kelly, 0), 0.25)
        else:
            kelly = 0
        
        # Calculate position size
        position_value = capital * kelly
        current_price = current_price or data['close'].iloc[-1]
        position_size = int(position_value / current_price)
        
        # Calculate stop loss based on average loss
        stop_loss = current_price * (1 - avg_loss * 2)
        
        return RiskMetrics(
            position_size=position_size,
            stop_loss=stop_loss,
            risk_amount=position_value,
            confidence=win_rate,
            method='kelly_criterion',
            timestamp=datetime.now(),
            additional_info={
                'kelly_percentage': kelly,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'win_loss_ratio': b
            }
        )
    
    def _default_metrics(self) -> RiskMetrics:
        """Return default metrics when calculation fails"""
        return RiskMetrics(
            position_size=0,
            stop_loss=0,
            risk_amount=0,
            confidence=0,
            method='kelly_criterion',
            timestamp=datetime.now()
        )


class DynamicStopLossModel(BaseRiskModel):
    """
    Adaptive stops using:
    - ATR-based levels
    - Volatility clusters
    - Support/resistance
    - Trailing mechanisms
    """
    
    def __init__(self, atr_period: int = 14, atr_multiplier: float = 2.0,
                 lookback_period: int = 252):
        super().__init__(lookback_period)
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        
    def calculate_risk_metrics(self, data: pd.DataFrame, entry_price: float = None,
                             position_type: str = 'long', **kwargs) -> RiskMetrics:
        """Calculate dynamic stop loss levels"""
        if not self.validate_data(data):
            return self._default_metrics()
        
        # Calculate ATR
        high = data['high'].tail(self.lookback_period)
        low = data['low'].tail(self.lookback_period)
        close = data['close'].tail(self.lookback_period)
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=self.atr_period).mean().iloc[-1]
        
        current_price = close.iloc[-1]
        entry_price = entry_price or current_price
        
        # Calculate stop loss based on ATR
        if position_type == 'long':
            stop_loss = entry_price - (atr * self.atr_multiplier)
            
            # Find support level
            support = self._find_support_resistance(data, 'support')
            if support and support > stop_loss:
                stop_loss = support * 0.99  # Just below support
        else:
            stop_loss = entry_price + (atr * self.atr_multiplier)
            
            # Find resistance level
            resistance = self._find_support_resistance(data, 'resistance')
            if resistance and resistance < stop_loss:
                stop_loss = resistance * 1.01  # Just above resistance
        
        # Calculate risk amount
        risk_per_share = abs(entry_price - stop_loss)
        
        # Volatility clustering adjustment
        volatility_regime = self._detect_volatility_regime(data)
        if volatility_regime == 'high':
            stop_loss = entry_price - (atr * self.atr_multiplier * 1.5) if position_type == 'long' else entry_price + (atr * self.atr_multiplier * 1.5)
        
        return RiskMetrics(
            position_size=0,  # Not calculated by this model
            stop_loss=stop_loss,
            risk_amount=risk_per_share,
            confidence=0.8,  # Based on ATR reliability
            method='dynamic_stop_loss',
            timestamp=datetime.now(),
            additional_info={
                'atr': atr,
                'volatility_regime': volatility_regime,
                'entry_price': entry_price,
                'position_type': position_type
            }
        )
    
    def _find_support_resistance(self, data: pd.DataFrame, level_type: str) -> Optional[float]:
        """Find support or resistance levels"""
        close = data['close'].tail(50)
        
        if level_type == 'support':
            # Find recent lows
            lows = close.rolling(window=5).min()
            support_levels = lows[lows == lows.rolling(window=10).min()].dropna()
            if not support_levels.empty:
                return support_levels.iloc[-1]
        else:
            # Find recent highs
            highs = close.rolling(window=5).max()
            resistance_levels = highs[highs == highs.rolling(window=10).max()].dropna()
            if not resistance_levels.empty:
                return resistance_levels.iloc[-1]
        
        return None
    
    def _detect_volatility_regime(self, data: pd.DataFrame) -> str:
        """Detect volatility regime"""
        returns = data['close'].pct_change().dropna().tail(20)
        recent_vol = returns.std()
        historical_vol = data['close'].pct_change().dropna().tail(252).std()
        
        if recent_vol > historical_vol * 1.5:
            return 'high'
        elif recent_vol < historical_vol * 0.5:
            return 'low'
        else:
            return 'normal'
    
    def _default_metrics(self) -> RiskMetrics:
        """Return default metrics when calculation fails"""
        return RiskMetrics(
            position_size=0,
            stop_loss=0,
            risk_amount=0,
            confidence=0,
            method='dynamic_stop_loss',
            timestamp=datetime.now()
        )


class RiskParityModel(BaseRiskModel):
    """
    Equal risk contribution:
    - Volatility weighting
    - Correlation adjustment
    - Dynamic rebalancing
    - Leverage constraints
    """
    
    def __init__(self, lookback_period: int = 252, target_volatility: float = 0.15):
        super().__init__(lookback_period)
        self.target_volatility = target_volatility
        
    def calculate_risk_metrics(self, data: Dict[str, pd.DataFrame], 
                             capital: float = 100000, **kwargs) -> Dict[str, RiskMetrics]:
        """Calculate risk parity weights for portfolio"""
        if not data or len(data) < 2:
            return {}
        
        # Calculate returns for all assets
        returns_dict = {}
        for symbol, df in data.items():
            if self.validate_data(df):
                returns = df['close'].pct_change().dropna().tail(self.lookback_period)
                returns_dict[symbol] = returns
        
        if not returns_dict:
            return {}
        
        # Create returns matrix
        returns_df = pd.DataFrame(returns_dict)
        
        # Calculate covariance matrix
        cov_matrix = returns_df.cov() * 252  # Annualized
        
        # Calculate initial equal weights
        n_assets = len(returns_dict)
        weights = np.ones(n_assets) / n_assets
        
        # Iterative risk parity optimization
        for _ in range(100):
            # Calculate marginal risk contributions
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            marginal_contrib = (cov_matrix @ weights) / portfolio_vol
            
            # Update weights to equalize risk contributions
            risk_contrib = weights * marginal_contrib
            target_contrib = portfolio_vol / n_assets
            
            weights = weights * (target_contrib / risk_contrib)
            weights = weights / weights.sum()
        
        # Apply volatility targeting
        current_vol = np.sqrt(weights @ cov_matrix @ weights)
        vol_scalar = self.target_volatility / current_vol
        weights = weights * vol_scalar
        
        # Create risk metrics for each asset
        results = {}
        symbols = list(returns_dict.keys())
        
        for i, symbol in enumerate(symbols):
            weight = weights[i]
            position_value = capital * weight
            current_price = data[symbol]['close'].iloc[-1]
            position_size = int(position_value / current_price)
            
            # Calculate individual asset volatility
            asset_vol = np.sqrt(cov_matrix.iloc[i, i])
            
            results[symbol] = RiskMetrics(
                position_size=position_size,
                stop_loss=current_price * (1 - 2 * asset_vol / np.sqrt(252)),
                risk_amount=position_value,
                confidence=0.85,
                method='risk_parity',
                timestamp=datetime.now(),
                additional_info={
                    'weight': weight,
                    'volatility': asset_vol,
                    'risk_contribution': risk_contrib[i] / portfolio_vol
                }
            )
        
        return results


class DrawdownControlModel(BaseRiskModel):
    """
    Position scaling by:
    - Current drawdown level
    - Historical max drawdown
    - Recovery periods
    - Risk budgets
    """
    
    def __init__(self, max_drawdown_limit: float = 0.20, 
                 recovery_factor: float = 0.5, lookback_period: int = 252):
        super().__init__(lookback_period)
        self.max_drawdown_limit = max_drawdown_limit
        self.recovery_factor = recovery_factor
        
    def calculate_risk_metrics(self, data: pd.DataFrame, capital: float = 100000,
                             base_position_size: float = 1.0, **kwargs) -> RiskMetrics:
        """Calculate position size based on drawdown"""
        if not self.validate_data(data):
            return self._default_metrics()
        
        # Calculate drawdown series
        close = data['close'].tail(self.lookback_period)
        rolling_max = close.expanding().max()
        drawdown = (close - rolling_max) / rolling_max
        current_drawdown = drawdown.iloc[-1]
        max_historical_drawdown = drawdown.min()
        
        # Calculate recovery metrics
        in_drawdown = drawdown < 0
        drawdown_periods = self._identify_drawdown_periods(drawdown)
        avg_recovery_time = self._calculate_avg_recovery_time(drawdown_periods)
        
        # Position scaling based on drawdown
        if current_drawdown < -self.max_drawdown_limit:
            # Exceeded limit, minimal position
            scale_factor = 0.1
        elif current_drawdown < 0:
            # In drawdown, reduce position proportionally
            drawdown_ratio = abs(current_drawdown) / self.max_drawdown_limit
            scale_factor = 1 - (drawdown_ratio * (1 - self.recovery_factor))
        else:
            # At new highs, full position
            scale_factor = 1.0
        
        # Adjust for historical drawdown patterns
        if abs(max_historical_drawdown) > self.max_drawdown_limit * 1.5:
            # Asset has shown severe drawdowns, be more conservative
            scale_factor *= 0.8
        
        # Calculate final position
        adjusted_position_size = base_position_size * scale_factor
        current_price = close.iloc[-1]
        position_value = capital * adjusted_position_size * 0.02  # 2% base risk
        position_size = int(position_value / current_price)
        
        # Dynamic stop based on drawdown state
        if current_drawdown < 0:
            # Tighter stop when in drawdown
            stop_loss = current_price * (1 + current_drawdown * 0.5)
        else:
            # Normal stop at new highs
            stop_loss = current_price * 0.95
        
        return RiskMetrics(
            position_size=position_size,
            stop_loss=stop_loss,
            risk_amount=position_value,
            confidence=scale_factor,
            method='drawdown_control',
            timestamp=datetime.now(),
            additional_info={
                'current_drawdown': current_drawdown,
                'max_historical_drawdown': max_historical_drawdown,
                'scale_factor': scale_factor,
                'avg_recovery_days': avg_recovery_time,
                'in_drawdown': bool(current_drawdown < 0)
            }
        )
    
    def _identify_drawdown_periods(self, drawdown: pd.Series) -> List[Dict]:
        """Identify individual drawdown periods"""
        periods = []
        in_drawdown = False
        start_idx = None
        
        for i, dd in enumerate(drawdown):
            if dd < 0 and not in_drawdown:
                # Start of drawdown
                in_drawdown = True
                start_idx = i
            elif dd >= 0 and in_drawdown:
                # End of drawdown
                in_drawdown = False
                periods.append({
                    'start': start_idx,
                    'end': i,
                    'duration': i - start_idx,
                    'max_drawdown': drawdown.iloc[start_idx:i].min()
                })
        
        return periods
    
    def _calculate_avg_recovery_time(self, periods: List[Dict]) -> float:
        """Calculate average recovery time from drawdowns"""
        if not periods:
            return 0
        
        recovery_times = [p['duration'] for p in periods]
        return np.mean(recovery_times)
    
    def _default_metrics(self) -> RiskMetrics:
        """Return default metrics when calculation fails"""
        return RiskMetrics(
            position_size=0,
            stop_loss=0,
            risk_amount=0,
            confidence=0,
            method='drawdown_control',
            timestamp=datetime.now()
        )


class VolatilityTargetingModel(BaseRiskModel):
    """
    Maintain target vol:
    - GARCH forecasting
    - Dynamic leverage
    - Risk on/off signals
    - Vol regime detection
    """
    
    def __init__(self, target_volatility: float = 0.15, 
                 vol_lookback: int = 20, lookback_period: int = 252):
        super().__init__(lookback_period)
        self.target_volatility = target_volatility
        self.vol_lookback = vol_lookback
        
    def calculate_risk_metrics(self, data: pd.DataFrame, capital: float = 100000,
                             use_garch: bool = True, **kwargs) -> RiskMetrics:
        """Calculate position size to target specific volatility"""
        if not self.validate_data(data):
            return self._default_metrics()
        
        # Calculate returns
        returns = data['close'].pct_change().dropna().tail(self.lookback_period)
        
        # Estimate current volatility
        if use_garch:
            forecast_vol = self._garch_forecast(returns)
        else:
            # Simple rolling volatility
            forecast_vol = returns.tail(self.vol_lookback).std() * np.sqrt(252)
        
        # Detect volatility regime
        vol_regime = self._detect_vol_regime(returns)
        
        # Calculate position scaling
        if forecast_vol > 0:
            vol_scalar = self.target_volatility / forecast_vol
            # Apply regime-based adjustments
            if vol_regime == 'high_vol':
                vol_scalar *= 0.7  # Reduce in high vol
            elif vol_regime == 'vol_spike':
                vol_scalar *= 0.5  # Significant reduction in spikes
            
            # Cap leverage
            vol_scalar = min(vol_scalar, 2.0)  # Max 2x leverage
            vol_scalar = max(vol_scalar, 0.1)  # Min 10% position
        else:
            vol_scalar = 0.5
        
        # Risk on/off signal
        risk_on = self._calculate_risk_on_signal(data, returns)
        if not risk_on:
            vol_scalar *= 0.3  # Reduce position in risk-off environment
        
        # Calculate position
        base_position_value = capital * 0.1  # 10% base allocation
        adjusted_position_value = base_position_value * vol_scalar
        current_price = data['close'].iloc[-1]
        position_size = int(adjusted_position_value / current_price)
        
        # Dynamic stop based on volatility
        stop_distance = current_price * forecast_vol / np.sqrt(252) * 2
        stop_loss = current_price - stop_distance
        
        return RiskMetrics(
            position_size=position_size,
            stop_loss=stop_loss,
            risk_amount=adjusted_position_value,
            confidence=0.8 if risk_on else 0.3,
            method='volatility_targeting',
            timestamp=datetime.now(),
            additional_info={
                'forecast_volatility': forecast_vol,
                'volatility_scalar': vol_scalar,
                'vol_regime': vol_regime,
                'risk_on': risk_on,
                'target_volatility': self.target_volatility
            }
        )
    
    def _garch_forecast(self, returns: pd.Series) -> float:
        """Simple GARCH(1,1) volatility forecast"""
        # Note: This is a simplified implementation
        # In production, use arch library for proper GARCH
        
        # Calculate squared returns
        returns2 = returns ** 2
        
        # Simple EWMA as proxy for GARCH
        # Weight recent observations more heavily
        alpha = 0.94
        weights = np.array([(1-alpha) * alpha**i for i in range(len(returns2))])
        weights = weights[::-1]  # Reverse so recent is first
        weights = weights / weights.sum()
        
        # Weighted variance
        variance = np.sum(weights * returns2)
        volatility = np.sqrt(variance * 252)  # Annualized
        
        return volatility
    
    def _detect_vol_regime(self, returns: pd.Series) -> str:
        """Detect volatility regime"""
        # Short-term vs long-term volatility
        short_vol = returns.tail(20).std() * np.sqrt(252)
        medium_vol = returns.tail(60).std() * np.sqrt(252)
        long_vol = returns.std() * np.sqrt(252)
        
        # Check for volatility spike
        if short_vol > long_vol * 2:
            return 'vol_spike'
        elif short_vol > medium_vol * 1.5:
            return 'high_vol'
        elif short_vol < long_vol * 0.5:
            return 'low_vol'
        else:
            return 'normal'
    
    def _calculate_risk_on_signal(self, data: pd.DataFrame, returns: pd.Series) -> bool:
        """Calculate risk on/off signal"""
        # Multiple factors for risk assessment
        
        # 1. Trend - price above 50-day MA
        sma50 = data['close'].tail(50).mean()
        current_price = data['close'].iloc[-1]
        trend_positive = current_price > sma50
        
        # 2. Volatility not spiking
        vol_regime = self._detect_vol_regime(returns)
        vol_normal = vol_regime not in ['vol_spike', 'high_vol']
        
        # 3. Recent returns positive
        recent_return = returns.tail(5).mean()
        returns_positive = recent_return > 0
        
        # Risk on if at least 2 of 3 conditions met
        risk_score = sum([trend_positive, vol_normal, returns_positive])
        
        return risk_score >= 2
    
    def _default_metrics(self) -> RiskMetrics:
        """Return default metrics when calculation fails"""
        return RiskMetrics(
            position_size=0,
            stop_loss=0,
            risk_amount=0,
            confidence=0,
            method='volatility_targeting',
            timestamp=datetime.now()
        )