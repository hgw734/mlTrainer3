"""
Utility functions for market regime detection and helper calculations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def calculate_market_regime(vix_value: Optional[float] = None, 
                          spy_data: Optional[pd.DataFrame] = None) -> str:
    """
    Determine current market regime based on VIX and market conditions
    
    Args:
        vix_value: Current VIX level
        spy_data: SPY price data for trend analysis
        
    Returns:
        Market regime string ('bull_market', 'bear_market', 'volatile_market', 'neutral_market')
    """
    try:
        # Default regime
        regime = 'neutral_market'
        
        # VIX-based regime detection
        if vix_value:
            if vix_value > 30:
                regime = 'volatile_market'
            elif vix_value > 25:
                regime = 'bear_market'
            elif vix_value < 15:
                regime = 'bull_market'
            else:
                regime = 'neutral_market'
        
        # SPY trend confirmation
        if spy_data is not None and len(spy_data) >= 20:
            # Calculate 20-day trend
            sma_20 = spy_data['close'].rolling(20).mean()
            current_price = spy_data['close'].iloc[-1]
            
            if len(sma_20) > 0:
                trend_strength = (current_price / sma_20.iloc[-1] - 1) * 100
                
                # Adjust regime based on trend
                if trend_strength > 5 and regime != 'volatile_market':
                    regime = 'bull_market'
                elif trend_strength < -5:
                    regime = 'bear_market'
        
        return regime
        
    except Exception as e:
        logger.error(f"Market regime calculation failed: {e}")
        return 'neutral_market'

def format_percentage(value: float, decimals: int = 2) -> str:
    """Format value as percentage"""
    try:
        return f"{value:.{decimals}f}%"
    except:
        return "0.00%"

def format_currency(value: float, decimals: int = 2) -> str:
    """Format value as currency"""
    try:
        return f"${value:,.{decimals}f}"
    except:
        return "$0.00"

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except:
        return default

def normalize_score(score: float, min_val: float = 0, max_val: float = 100) -> float:
    """Normalize score to specified range"""
    try:
        return max(min_val, min(max_val, score))
    except:
        return (min_val + max_val) / 2

def calculate_volatility(prices: pd.Series, window: int = 20) -> float:
    """Calculate rolling volatility"""
    try:
        if len(prices) < window:
            return 0.0
        
        returns = prices.pct_change().dropna()
        volatility = returns.rolling(window).std().iloc[-1] * np.sqrt(252)
        return volatility if not np.isnan(volatility) else 0.0
        
    except Exception as e:
        logger.error(f"Volatility calculation failed: {e}")
        return 0.0

def get_trading_session() -> str:
    """Get current trading session"""
    try:
        from datetime import datetime
        import pytz
        
        # Get current time in EST
        est = pytz.timezone('US/Eastern')
        current_time = datetime.now(est).time()
        
        # Define trading hours
        market_open = datetime.strptime('09:30', '%H:%M').time()
        market_close = datetime.strptime('16:00', '%H:%M').time()
        
        if market_open <= current_time <= market_close:
            return 'MARKET_HOURS'
        elif current_time < market_open:
            return 'PRE_MARKET'
        else:
            return 'AFTER_MARKET'
            
    except Exception as e:
        logger.error(f"Trading session detection failed: {e}")
        return 'UNKNOWN'

def is_market_day() -> bool:
    """Check if today is a market trading day"""
    try:
        from datetime import datetime
        
        today = datetime.now().weekday()
        # Monday = 0, Sunday = 6
        return today < 5  # Monday through Friday
        
    except Exception as e:
        logger.error(f"Market day check failed: {e}")
        return True  # Default to True