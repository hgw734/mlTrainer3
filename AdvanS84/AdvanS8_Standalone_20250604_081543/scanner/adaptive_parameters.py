"""
Adaptive parameter system that adjusts scanning parameters dynamically
based on market conditions, volatility, and trading session timing.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime, time
import logging

logger = logging.getLogger(__name__)

class AdaptiveParameterEngine:
    """
    Dynamically adjusts scanning parameters based on:
    - Market regime (bull/bear/volatile/neutral)
    - Intraday timing (pre-market, first hour, mid-day, power hour, after-hours)
    - Volatility environment (VIX levels)
    - Sector rotation patterns
    """
    
    def __init__(self):
        """Initialize adaptive parameter engine"""
        self.last_update = None
        self.market_state = {}
        self.cached_parameters = None
        
        # Initialize continuous optimizer
        from scanner.continuous_optimizer import ContinuousOptimizer
        self.continuous_optimizer = ContinuousOptimizer()
        
        # Base parameter configurations for different market regimes
        self.regime_configs = {
            'bull_market': {
                'momentum_weight': 0.45,                # High momentum focus for trend following
                'technical_weight': 0.30,               # Strong technical signals
                'fundamental_weight': 0.15,             # Moderate fundamentals
                'sentiment_weight': 0.10,               # Sentiment support
                'min_momentum_threshold': 4.0,          # Moderate threshold for more signals
                'volume_multiplier_required': 2.5,      # 2.5x+ volume expansion
                'rsi_overbought_threshold': 75,         # Allow higher RSI in bull markets
                'trend_strength_minimum': 2.0,          # Moderate trend strength
                'score_threshold_multiplier': 0.70,     # Lower threshold for more opportunities
                'min_score_threshold': 25,              # Lower threshold for bull markets
                'regime_multiplier': 0.8,               # Lower multiplier for more opportunities
                'exit_strategy_multipliers': {          # Exit strategy parameter multipliers
                    'stop_loss': 1.2,                   # Wider stops in bull markets
                    'trailing_stop': 1.1,               # Allow more room for momentum
                    'profit_protection': 0.8,           # Take profits earlier in strong trends
                    'momentum_exit': 0.9                # Quicker momentum exits
                },
                'avoid_defensive_sectors': True,        # Exclude utilities, staples (-0.62)
                'min_earnings_growth': 10.0,            # Require >10% earnings growth (-0.59)
                'max_beta': 2.0                         # Allow high beta for momentum
            },
            'bear_market': {
                'momentum_weight': 0.25,                # Reduced for oversold bounce focus
                'technical_weight': 0.40,               # Strong technical for oversold signals (+0.69)
                'fundamental_weight': 0.30,             # High dividend/defensive focus (+0.59)
                'sentiment_weight': 0.05,               # Minimal in fearful conditions
                'min_momentum_threshold': 0.5,          # Minimal momentum - maximize oversold RSI bounces (+0.79)
                'volume_multiplier_required': 2.5,      # Volume spike confirmation for bounces
                'rsi_overbought_threshold': 25,         # Strict oversold focus (RSI <25 = +0.79 correlation)
                'trend_strength_minimum': 1.2,          # Lower for bounce opportunities
                'score_threshold_multiplier': 1.0,      # Standard threshold for defensive plays
                'avoid_growth_sectors': True,           # Exclude high P/E growth stocks (-0.76)
                'avoid_speculative_sectors': True,      # Exclude crypto, meme stocks (-0.65)
                'max_beta': 1.3,                        # Limit high beta exposure (-0.69)
                'require_dividend_yield': 2.0          # Prefer dividend stocks in bear markets
            },
            'volatile_market': {
                'momentum_weight': 0.35,                # Balanced momentum for volatility
                'technical_weight': 0.35,               # Strong technical signals
                'fundamental_weight': 0.20,             # Moderate fundamentals for stability
                'sentiment_weight': 0.10,               # Higher sentiment weight for volatility
                'min_momentum_threshold': 5.0,          # High threshold for quality
                'volume_multiplier_required': 4.0,      # High volume confirmation
                'rsi_overbought_threshold': 70,         # Standard overbought level
                'trend_strength_minimum': 2.5,          # Strong trend needed
                'score_threshold_multiplier': 1.10,     # Higher standards for volatile conditions
                'avoid_earnings_proximity': True,       # Exclude earnings proximity
                'min_market_cap': 1000000000,           # Avoid small caps
                'require_strong_technicals': True       # Clear technical levels
            },
            'neutral_market': {
                'momentum_weight': 0.30,                # Reduced for catalyst focus
                'technical_weight': 0.40,               # Strong technical for pattern completion (+0.62)
                'fundamental_weight': 0.25,             # High for earnings catalysts (+0.66)
                'sentiment_weight': 0.05,               # Minimal in sideways markets
                'min_momentum_threshold': 2.0,          # Lower to allow fundamental catalyst plays
                'volume_multiplier_required': 2.0,      # Pattern completion volume
                'rsi_overbought_threshold': 50,         # Neutral range - focus on relative strength leaders (+0.75)
                'trend_strength_minimum': 1.5,          # Allow relative strength plays
                'score_threshold_multiplier': 1.4,      # Highest standards for neutral markets (70+ scores)
                'min_score_threshold': 35,              # Base minimum score threshold
                'regime_multiplier': 1.0,               # Base regime multiplier
                'exit_strategy_multipliers': {          # Exit strategy parameter multipliers
                    'stop_loss': 1.0,
                    'trailing_stop': 1.0,
                    'profit_protection': 1.0,
                    'momentum_exit': 1.0
                },
                'avoid_momentum_only': True,            # Exclude pure momentum strategies (-0.75)
                'require_catalyst': True,               # Require fundamental catalysts (-0.61)
                'avoid_weak_fundamentals': True,       # Exclude declining earnings (-0.72)
                'require_relative_strength': True      # Focus on sector leaders (-0.68)
            },
            'recovery_market': {
                'momentum_weight': 0.35,
                'technical_weight': 0.30,
                'fundamental_weight': 0.25,
                'sentiment_weight': 0.10,
                'min_momentum_threshold': 1.5,
                'volume_multiplier_required': 1.2,
                'rsi_overbought_threshold': 72,
                'trend_strength_minimum': 1.2,
                'score_threshold_multiplier': 0.6
            }
        }
        
        # Intraday adjustments based on trading session
        self.session_adjustments = {
            'pre_market': {
                'momentum_boost': 1.1,
                'volume_threshold_reduction': 0.8,
                'news_sentiment_weight': 1.3
            },
            'market_open': {
                'momentum_boost': 1.2,
                'volume_threshold_reduction': 0.6,
                'volatility_tolerance': 1.4
            },
            'mid_day': {
                'momentum_boost': 1.0,
                'volume_threshold_reduction': 1.0,
                'technical_emphasis': 1.1
            },
            'power_hour': {
                'momentum_boost': 1.15,
                'volume_threshold_reduction': 0.7,
                'institutional_weight': 1.2
            },
            'after_hours': {
                'momentum_boost': 0.9,
                'volume_threshold_reduction': 1.2,
                'news_sentiment_weight': 1.1
            }
        }
        
        # Initialize with default parameters to prevent None errors
        self.current_parameters = self._get_default_parameters()
    
    def update_parameters(self, market_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Update parameters 30 minutes before market open and when market conditions change
        
        Args:
            market_data: Current market indicators (VIX, SPY trend, sector rotation)
            
        Returns:
            Updated parameter dictionary
        """
        try:
            current_time = datetime.now()
            
            # Check if we need to update (30 min before open or significant market change)
            if self._should_update_parameters(current_time, market_data):
                logger.info("Updating adaptive parameters based on market conditions")
                
                # Determine current market regime
                market_regime = self._analyze_market_regime(market_data)
                
                # Get trading session
                trading_session = self._get_trading_session(current_time)
                
                # Calculate volatility environment
                volatility_regime = self._analyze_volatility_environment(market_data)
                
                # Get base parameters for market regime
                base_params = self.regime_configs.get(market_regime, self.regime_configs['neutral_market'])
                
                # Apply session-based adjustments
                session_adjustments = self.session_adjustments.get(trading_session, {})
                
                # Apply volatility adjustments
                volatility_adjustments = self._get_volatility_adjustments(volatility_regime)
                
                # Combine all adjustments
                self.current_parameters = self._combine_adjustments(
                    base_params, session_adjustments, volatility_adjustments
                )
                
                # Add market state information
                self.market_state = {
                    'regime': market_regime,
                    'session': trading_session,
                    'volatility': volatility_regime,
                    'last_update': current_time,
                    'vix_level': market_data.get('vix', 20) if market_data else 20,
                    'spy_trend': market_data.get('spy_trend', 0) if market_data else 0
                }
                
                self.last_update = current_time
                
                logger.info(f"Parameters updated for {market_regime} regime, {trading_session} session")
                
            return self.current_parameters
            
        except Exception as e:
            logger.error(f"Parameter update failed: {e}")
            return self._get_default_parameters()
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """Get current parameters"""
        if hasattr(self, 'current_parameters') and self.current_parameters:
            return self.current_parameters
        else:
            return self._get_default_parameters()
    
    def _should_update_parameters(self, current_time: datetime, market_data: Optional[Dict]) -> bool:
        """Determine if parameters should be updated based on actual market condition changes"""
        try:
            # Always update if no previous update
            if self.last_update is None:
                return True
            
            # Check if markets are open
            is_market_open = (current_time.weekday() < 5 and 
                            time(9, 30) <= current_time.time() <= time(16, 0))
            
            # During market hours: update only if significant market change detected
            if is_market_open:
                if market_data and self._detect_significant_market_change(market_data):
                    logger.info("Significant market condition change detected")
                    return True
                
                # Check if it's a new trading day
                if current_time.date() > self.last_update.date():
                    logger.info("New trading day - updating parameters")
                    return True
            
            # Outside market hours: only update if we haven't set parameters yet today
            else:
                if current_time.date() > self.last_update.date():
                    logger.info("New day - setting baseline parameters")
                    return True
            
            # No update needed - market conditions unchanged
            return False
            
        except Exception as e:
            logger.error(f"Update check failed: {e}")
            return True
    
    def _analyze_market_regime(self, market_data: Optional[Dict]) -> str:
        """Analyze current market regime"""
        try:
            if not market_data:
                # Fallback: analyze current date to determine historical regime
                current_date = datetime.now()
                return self._get_historical_regime_by_date(current_date)
            
            vix = market_data.get('vix', 20)
            spy_trend = market_data.get('spy_trend', 0)  # 20-day trend %
            
            # VIX-based regime classification
            if vix > 30:
                return 'volatile_market'
            elif vix > 25:
                if spy_trend > 2:
                    return 'volatile_market'
                else:
                    return 'bear_market'
            elif vix < 15:
                if spy_trend > 0:
                    return 'bull_market'
                else:
                    return 'neutral_market'
            else:
                if spy_trend > 5:
                    return 'bull_market'
                elif spy_trend < -5:
                    return 'bear_market'
                else:
                    return 'neutral_market'
                    
        except Exception as e:
            logger.error(f"Market regime analysis failed: {e}")
            return 'neutral_market'
    
    def _get_historical_regime_by_date(self, date: datetime) -> str:
        """Determine market regime based on historical periods for backtesting"""
        year = date.year
        
        if year == 2021:
            return 'bull_market'
        elif year == 2022:
            return 'bear_market'
        elif year == 2023:
            return 'recovery_market'
        elif year >= 2024:
            return 'neutral_market'
        else:
            return 'neutral_market'
    
    def _get_trading_session(self, current_time: datetime) -> str:
        """Determine current trading session"""
        try:
            current_time_only = current_time.time()
            
            if time(4, 0) <= current_time_only < time(9, 30):
                return 'pre_market'
            elif time(9, 30) <= current_time_only < time(10, 30):
                return 'market_open'
            elif time(10, 30) <= current_time_only < time(15, 0):
                return 'mid_day'
            elif time(15, 0) <= current_time_only < time(16, 0):
                return 'power_hour'
            else:
                return 'after_hours'
                
        except Exception as e:
            logger.error(f"Trading session detection failed: {e}")
            return 'mid_day'
    
    def _analyze_volatility_environment(self, market_data: Optional[Dict]) -> str:
        """Analyze volatility environment"""
        try:
            if not market_data:
                return 'normal'
            
            vix = market_data.get('vix', 20)
            
            if vix > 35:
                return 'extreme'
            elif vix > 25:
                return 'high'
            elif vix > 15:
                return 'normal'
            else:
                return 'low'
                
        except Exception as e:
            logger.error(f"Volatility analysis failed: {e}")
            return 'normal'
    
    def _get_volatility_adjustments(self, volatility_regime: str) -> Dict[str, float]:
        """Get parameter adjustments based on volatility"""
        adjustments = {
            'extreme': {
                'momentum_weight_multiplier': 1.3,
                'volume_threshold_multiplier': 2.5,
                'risk_penalty_multiplier': 1.4
            },
            'high': {
                'momentum_weight_multiplier': 1.15,
                'volume_threshold_multiplier': 1.8,
                'risk_penalty_multiplier': 1.2
            },
            'normal': {
                'momentum_weight_multiplier': 1.0,
                'volume_threshold_multiplier': 1.0,
                'risk_penalty_multiplier': 1.0
            },
            'low': {
                'momentum_weight_multiplier': 0.9,
                'volume_threshold_multiplier': 0.8,
                'risk_penalty_multiplier': 0.9
            }
        }
        
        return adjustments.get(volatility_regime, adjustments['normal'])
    
    def _detect_significant_market_change(self, market_data: Dict) -> bool:
        """Detect if market conditions have changed significantly"""
        try:
            if not hasattr(self, 'previous_market_data'):
                self.previous_market_data = market_data
                return False
            
            # Check VIX change
            vix_change = abs(market_data.get('vix', 20) - self.previous_market_data.get('vix', 20))
            if vix_change > 5:
                return True
            
            # Check SPY trend change
            trend_change = abs(market_data.get('spy_trend', 0) - self.previous_market_data.get('spy_trend', 0))
            if trend_change > 3:
                return True
            
            self.previous_market_data = market_data
            return False
            
        except Exception as e:
            logger.error(f"Market change detection failed: {e}")
            return False
    
    def _combine_adjustments(self, base_params: Dict, session_adj: Dict, vol_adj: Dict) -> Dict[str, Any]:
        """Combine all parameter adjustments"""
        try:
            combined = base_params.copy()
            
            # Apply session adjustments
            if 'momentum_boost' in session_adj:
                combined['momentum_weight'] *= session_adj['momentum_boost']
            
            if 'volume_threshold_reduction' in session_adj:
                combined['volume_multiplier_required'] *= session_adj['volume_threshold_reduction']
            
            # Apply volatility adjustments
            if 'momentum_weight_multiplier' in vol_adj:
                combined['momentum_weight'] *= vol_adj['momentum_weight_multiplier']
            
            if 'volume_threshold_multiplier' in vol_adj:
                combined['volume_multiplier_required'] *= vol_adj['volume_threshold_multiplier']
            
            # Normalize weights to sum to 1.0
            total_weight = (combined['momentum_weight'] + combined['technical_weight'] + 
                          combined['fundamental_weight'] + combined['sentiment_weight'])
            
            if total_weight > 0:
                combined['momentum_weight'] /= total_weight
                combined['technical_weight'] /= total_weight
                combined['fundamental_weight'] /= total_weight
                combined['sentiment_weight'] /= total_weight
            
            return combined
            
        except Exception as e:
            logger.error(f"Parameter combination failed: {e}")
            return self._get_default_parameters()
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters as fallback"""
        defaults = self.regime_configs['neutral_market'].copy()
        # Ensure all required weights exist
        if 'momentum_weight' not in defaults:
            defaults['momentum_weight'] = 0.35
        if 'technical_weight' not in defaults:
            defaults['technical_weight'] = 0.30
        if 'fundamental_weight' not in defaults:
            defaults['fundamental_weight'] = 0.25
        if 'sentiment_weight' not in defaults:
            defaults['sentiment_weight'] = 0.10
        return defaults
    
    def get_current_market_state(self) -> Dict[str, Any]:
        """Get current market state information"""
        return self.market_state.copy()
    
    def get_cached_parameters(self) -> Dict[str, Any]:
        """Get cached parameters for deterministic behavior when markets are closed"""
        if self.cached_parameters is None:
            # Initialize with neutral market parameters for consistency
            self.cached_parameters = self._get_default_parameters()
            logger.info("Initialized cached parameters for deterministic mode")
        return self.cached_parameters.copy()
    
    def get_parameter_explanation(self) -> Dict[str, str]:
        """Get explanation of current parameter settings"""
        return {
            'momentum_weight': 'Weight given to price momentum across multiple timeframes',
            'technical_weight': 'Weight given to technical indicators (RSI, MACD, etc.)',
            'fundamental_weight': 'Weight given to earnings, analyst ratings, financial health',
            'sentiment_weight': 'Weight given to news and social media sentiment',
            'min_momentum_threshold': 'Minimum momentum % required for consideration',
            'volume_multiplier_required': 'Volume must be X times average to qualify',
            'rsi_overbought_threshold': 'RSI level considered overbought',
            'trend_strength_minimum': 'Minimum trend strength required'
        }