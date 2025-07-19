"""
Market Regime Logic Module
Advanced market regime classification using VIX and multi-factor analysis
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def get_market_regime(date, vix_data):
    """
    Enhanced market regime identification with multi-factor analysis
    
    Args:
        date: Current date for regime classification
        vix_data: VIX DataFrame with historical data
    
    Returns:
        str: Market regime classification
    """
    if vix_data is None or date not in vix_data.index:
        return default_regime_classification(date)
    
    vix_value = vix_data.loc[date, 'vix']
    vix_sma = vix_data.loc[date, 'vix_sma_20'] if 'vix_sma_20' in vix_data.columns else vix_value
    
    # Multi-factor regime classification
    regime_factors = {
        'vix_level': classify_vix_level(vix_value),
        'vix_trend': classify_vix_trend(vix_value, vix_sma),
        'market_phase': classify_market_phase(date, vix_data)
    }
    
    # Combine factors for final regime
    return combine_regime_factors(regime_factors)

def classify_vix_level(vix_value):
    """
    Classify current VIX level
    
    Args:
        vix_value: Current VIX value
    
    Returns:
        str: VIX level classification
    """
    if vix_value > 30:
        return 'extreme_volatility'
    elif vix_value > 25:
        return 'high_volatility'
    elif vix_value > 15:
        return 'moderate_volatility'
    else:
        return 'low_volatility'

def classify_vix_trend(current_vix, vix_sma):
    """
    Classify VIX trend direction
    
    Args:
        current_vix: Current VIX value
        vix_sma: VIX simple moving average
    
    Returns:
        str: VIX trend classification
    """
    trend_ratio = current_vix / vix_sma if vix_sma > 0 else 1.0
    
    if trend_ratio > 1.15:
        return 'rising_volatility'
    elif trend_ratio < 0.85:
        return 'falling_volatility'
    else:
        return 'stable_volatility'

def classify_market_phase(date, vix_data):
    """
    Classify broader market phase using 30-day VIX analysis
    
    Args:
        date: Current date
        vix_data: VIX DataFrame
    
    Returns:
        str: Market phase classification
    """
    try:
        end_date = date
        start_date = date - pd.Timedelta(days=30)
        recent_vix = vix_data[start_date:end_date]['vix']
        
        if len(recent_vix) > 10:
            avg_vix = recent_vix.mean()
            if avg_vix > 22:
                return 'stress_phase'
            elif avg_vix < 12:
                return 'complacency_phase'
            else:
                return 'normal_phase'
    except Exception as e:
        logger.warning(f"Error in market phase classification: {e}")
    
    return 'normal_phase'

def combine_regime_factors(factors):
    """
    Combine multiple factors into final regime classification
    
    Args:
        factors: Dictionary with vix_level, vix_trend, market_phase
    
    Returns:
        str: Combined regime classification
    """
    vix_level = factors['vix_level']
    vix_trend = factors['vix_trend']
    market_phase = factors['market_phase']
    
    # Priority-based combination
    if vix_level == 'extreme_volatility' or market_phase == 'stress_phase':
        return 'crisis_regime'
    elif vix_level == 'high_volatility' and vix_trend == 'rising_volatility':
        return 'high_volatility_rising'
    elif vix_level == 'high_volatility':
        return 'high_volatility_stable'
    elif vix_level == 'low_volatility' and market_phase == 'complacency_phase':
        return 'low_volatility_complacent'
    elif vix_level == 'low_volatility':
        return 'low_volatility_normal'
    else:
        return 'moderate_volatility_normal'

def default_regime_classification(date):
    """
    Default regime when VIX data unavailable
    
    Args:
        date: Current date
    
    Returns:
        str: Default regime classification
    """
    return 'moderate_volatility_normal'

def classify_regime(market_data, symbol, date):
    """
    General regime classification function for compatibility
    
    Args:
        market_data: Dictionary containing market data
        symbol: Stock symbol
        date: Date for classification
    
    Returns:
        str: Market regime classification
    """
    vix_data = market_data.get('VIX')
    return get_market_regime(date, vix_data)

def get_exit_configuration(market_regime, adaptation_strength):
    """
    Get exit configuration based on detailed market regime
    
    Args:
        market_regime: Current market regime
        adaptation_strength: Adaptation parameter
    
    Returns:
        dict: Exit configuration parameters
    """
    base_config = {
        'momentum_threshold': 0.005,
        'max_hold_days': 15,
        'time_multiplier': 1.0,
        'volatility_sensitivity': 1.0
    }
    
    # Regime-specific adjustments
    regime_adjustments = {
        'crisis_regime': {
            'momentum_threshold': 0.008,
            'max_hold_days': 8,
            'time_multiplier': 0.6,
            'volatility_sensitivity': 2.0
        },
        'high_volatility_rising': {
            'momentum_threshold': 0.007,
            'max_hold_days': 10,
            'time_multiplier': 0.7,
            'volatility_sensitivity': 1.5
        },
        'high_volatility_stable': {
            'momentum_threshold': 0.006,
            'max_hold_days': 12,
            'time_multiplier': 0.8,
            'volatility_sensitivity': 1.3
        },
        'low_volatility_complacent': {
            'momentum_threshold': 0.003,
            'max_hold_days': 20,
            'time_multiplier': 1.3,
            'volatility_sensitivity': 0.5
        },
        'low_volatility_normal': {
            'momentum_threshold': 0.004,
            'max_hold_days': 18,
            'time_multiplier': 1.1,
            'volatility_sensitivity': 0.7
        }
    }
    
    # Apply regime adjustments
    if market_regime in regime_adjustments:
        for key, value in regime_adjustments[market_regime].items():
            base_config[key] = value
    
    # Apply adaptation strength
    base_config['momentum_threshold'] *= (1 + adaptation_strength * 0.4)
    base_config['time_multiplier'] *= (1 + adaptation_strength * 0.2)
    
    return base_config

def get_regime_adaptive_rsi_threshold(market_regime):
    """
    Get RSI threshold based on market regime
    
    Args:
        market_regime: Current market regime
    
    Returns:
        float: RSI threshold for entry criteria
    """
    regime_rsi_thresholds = {
        'crisis_regime': 70,
        'high_volatility_rising': 65,
        'high_volatility_stable': 60,
        'moderate_volatility_normal': 55,
        'low_volatility_normal': 50,
        'low_volatility_complacent': 45
    }
    
    return regime_rsi_thresholds.get(market_regime, 55)