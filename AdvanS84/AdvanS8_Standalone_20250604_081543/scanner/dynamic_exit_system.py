"""
Dynamic exit system that monitors momentum and exits only when conditions deteriorate,
rather than using fixed time-based exits.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class DynamicExitSystem:
    """
    Dynamic exit system that continuously monitors momentum and technical conditions
    to determine optimal exit timing based on deteriorating signals rather than fixed time.
    """
    
    def __init__(self):
        """Initialize dynamic exit system"""
        self.positions = {}
        
    def should_exit_position(self, 
                           symbol: str, 
                           current_data: Dict, 
                           position_info: Dict,
                           market_regime: str = 'neutral_market') -> Dict[str, Any]:
        """
        Determine if position should be exited based on momentum deterioration
        
        Args:
            symbol: Stock symbol
            current_data: Current market data and indicators
            position_info: Information about the position (entry date, price, score, etc.)
            market_regime: Current market regime
            
        Returns:
            Dictionary with exit decision and reasoning
        """
        
        # Market regime-specific exit parameters
        regime_params = self._get_regime_exit_parameters(market_regime)
        
        exit_signals = {
            'should_exit': False,
            'exit_reason': None,
            'confidence': 0.0,
            'momentum_score': 0.0,
            'technical_deterioration': False,
            'regime_specific_exit': False,
            'regime_params': regime_params
        }
        
        # Get current momentum metrics
        momentum_metrics = self._calculate_momentum_metrics(current_data)
        exit_signals['momentum_score'] = momentum_metrics.get('composite_momentum', 0)
        
        # Check for momentum deterioration
        momentum_deterioration = self._check_momentum_deterioration(
            momentum_metrics, position_info, market_regime
        )
        
        # Check for technical deterioration
        technical_deterioration = self._check_technical_deterioration(
            current_data, position_info, market_regime
        )
        
        # Check regime-specific exit conditions
        regime_exit = self._check_regime_specific_exits(
            current_data, position_info, market_regime
        )
        
        # Combine exit signals with weighted importance
        exit_factors = []
        
        if momentum_deterioration['severe_deterioration']:
            exit_factors.append(('momentum_collapse', 0.4))
            exit_signals['exit_reason'] = 'Severe momentum deterioration detected'
            
        if technical_deterioration['breakdown']:
            exit_factors.append(('technical_breakdown', 0.3))
            exit_signals['technical_deterioration'] = True
            
        if regime_exit['should_exit']:
            exit_factors.append(('regime_change', 0.2))
            exit_signals['regime_specific_exit'] = True
            
        # Risk management exits (always priority)
        risk_exit = self._check_risk_management_exits(current_data, position_info)
        if risk_exit['stop_loss_triggered']:
            exit_signals['should_exit'] = True
            exit_signals['exit_reason'] = 'Stop loss triggered'
            exit_signals['confidence'] = 1.0
            return exit_signals
            
        # Exit logic depends heavily on profit/loss status - CUT LOSERS FAST, LET WINNERS RUN
        current_return = ((current_data.get('current_price', 0) / position_info.get('entry_price', 1)) - 1) * 100
        
        if exit_factors:
            total_weight = sum(weight for _, weight in exit_factors)
            exit_signals['confidence'] = min(1.0, total_weight)
            
            if current_return < 0:
                # LOSING TRADE: Much more aggressive exits - any sign of trouble = EXIT
                if exit_signals['confidence'] >= 0.25:  # 25% confidence threshold for losers
                    exit_signals['should_exit'] = True
                    if not exit_signals['exit_reason']:
                        exit_signals['exit_reason'] = f"Losing trade quick exit (confidence: {exit_signals['confidence']:.1%})"
            else:
                # WINNING TRADE: Much more patient - let winners run longer
                if exit_signals['confidence'] >= 0.75:  # 75% confidence threshold for winners
                    exit_signals['should_exit'] = True
                    if not exit_signals['exit_reason']:
                        exit_signals['exit_reason'] = f"Winning trade degradation (confidence: {exit_signals['confidence']:.1%})"
        
        return exit_signals
    
    def _get_regime_exit_parameters(self, market_regime: str) -> Dict[str, float]:
        """Get market regime-specific exit parameters"""
        regime_params = {
            'bull_market': {
                'stop_loss_pct': -2.5,
                'momentum_exit_pct': -6.0,
                'profit_protection_pct': 20.0,
                'trailing_stop_pct': 10.0,
                'max_hold_days': 8,
                'rsi_exit_threshold': 80
            },
            'bear_market': {
                'stop_loss_pct': -1.5,
                'momentum_exit_pct': -3.0,
                'profit_protection_pct': 8.0,
                'trailing_stop_pct': 5.0,
                'max_hold_days': 3,
                'rsi_exit_threshold': 65
            },
            'volatile_market': {
                'stop_loss_pct': -2.0,
                'momentum_exit_pct': -4.0,
                'profit_protection_pct': 12.0,
                'trailing_stop_pct': 6.0,
                'max_hold_days': 5,
                'rsi_exit_threshold': 75
            },
            'neutral_market': {
                'stop_loss_pct': -2.0,
                'momentum_exit_pct': -4.5,
                'profit_protection_pct': 10.0,
                'trailing_stop_pct': 7.0,
                'max_hold_days': 6,
                'rsi_exit_threshold': 70
            }
        }
        
        return regime_params.get(market_regime, regime_params['neutral_market'])
    
    def _calculate_momentum_metrics(self, data: Dict) -> Dict[str, float]:
        """Calculate current momentum metrics"""
        
        try:
            prices = data.get('prices', [])
            if len(prices) < 20:
                return {'composite_momentum': 0}
            
            # Short-term momentum (5-day)
            short_momentum = (prices[-1] / prices[-6] - 1) * 100 if len(prices) >= 6 else 0
            
            # Medium-term momentum (10-day)
            medium_momentum = (prices[-1] / prices[-11] - 1) * 100 if len(prices) >= 11 else 0
            
            # Volume momentum
            volumes = data.get('volumes', [])
            if len(volumes) >= 10:
                recent_avg_vol = np.mean(volumes[-5:])
                historical_avg_vol = np.mean(volumes[-20:-5])
                volume_momentum = (recent_avg_vol / historical_avg_vol - 1) * 100 if historical_avg_vol > 0 else 0
            else:
                volume_momentum = 0
            
            # Relative strength vs market
            spy_prices = data.get('spy_prices', [])
            if len(spy_prices) >= 10 and len(prices) >= 10:
                stock_return = (prices[-1] / prices[-11] - 1) * 100
                market_return = (spy_prices[-1] / spy_prices[-11] - 1) * 100
                relative_strength = stock_return - market_return
            else:
                relative_strength = 0
            
            # Composite momentum score
            composite_momentum = (
                short_momentum * 0.4 +
                medium_momentum * 0.3 +
                volume_momentum * 0.2 +
                relative_strength * 0.1
            )
            
            return {
                'short_momentum': short_momentum,
                'medium_momentum': medium_momentum,
                'volume_momentum': volume_momentum,
                'relative_strength': relative_strength,
                'composite_momentum': composite_momentum
            }
            
        except Exception as e:
            logger.error(f"Error calculating momentum metrics: {e}")
            return {'composite_momentum': 0}
    
    def _check_momentum_deterioration(self, 
                                    momentum_metrics: Dict, 
                                    position_info: Dict,
                                    market_regime: str) -> Dict[str, Any]:
        """Check for momentum deterioration patterns"""
        
        current_momentum = momentum_metrics.get('composite_momentum', 0)
        entry_momentum = position_info.get('entry_momentum', current_momentum)
        
        # Calculate momentum decay
        momentum_decay = entry_momentum - current_momentum
        decay_percentage = (momentum_decay / abs(entry_momentum)) * 100 if entry_momentum != 0 else 0
        
        # Regime-specific deterioration thresholds
        deterioration_thresholds = {
            'bull_market': {
                'severe': 30,    # 30% momentum decay
                'moderate': 20,  # 20% momentum decay
                'negative_momentum': -2  # Negative momentum in bull market
            },
            'bear_market': {
                'severe': 15,    # 15% momentum decay (tighter in bear markets)
                'moderate': 10,  # 10% momentum decay
                'negative_momentum': -5  # More tolerance for negative momentum
            },
            'volatile_market': {
                'severe': 25,    # 25% momentum decay
                'moderate': 15,  # 15% momentum decay
                'negative_momentum': -3  # Moderate tolerance
            },
            'neutral_market': {
                'severe': 20,    # 20% momentum decay
                'moderate': 15,  # 15% momentum decay
                'negative_momentum': -2  # Low tolerance
            }
        }
        
        thresholds = deterioration_thresholds.get(market_regime, deterioration_thresholds['neutral_market'])
        
        return {
            'momentum_decay': momentum_decay,
            'decay_percentage': decay_percentage,
            'severe_deterioration': decay_percentage >= thresholds['severe'],
            'moderate_deterioration': decay_percentage >= thresholds['moderate'],
            'negative_momentum': current_momentum <= thresholds['negative_momentum']
        }
    
    def _check_technical_deterioration(self, 
                                     data: Dict, 
                                     position_info: Dict,
                                     market_regime: str) -> Dict[str, Any]:
        """Check for technical indicator deterioration"""
        
        try:
            rsi = data.get('rsi', 50)
            volume_ratio = data.get('volume_ratio', 1.0)
            price = data.get('current_price', 0)
            entry_price = position_info.get('entry_price', price)
            
            # Technical breakdown signals
            breakdown_signals = []
            
            # RSI extremes (regime-specific)
            if market_regime == 'bull_market':
                if rsi > 80:  # Severely overbought in bull market
                    breakdown_signals.append('severe_overbought')
                elif rsi > 75:
                    breakdown_signals.append('overbought')
            else:
                if rsi > 75:  # Overbought in other regimes
                    breakdown_signals.append('overbought')
            
            # Volume deterioration
            if volume_ratio < 0.5:  # Volume dropped to less than 50% of average
                breakdown_signals.append('volume_exhaustion')
            
            # Price action breakdown
            if price < entry_price * 0.92:  # 8% drawdown
                breakdown_signals.append('significant_drawdown')
            
            return {
                'breakdown': len(breakdown_signals) >= 2,  # At least 2 breakdown signals
                'breakdown_signals': breakdown_signals,
                'technical_score': max(0, 100 - len(breakdown_signals) * 25)
            }
            
        except Exception as e:
            logger.error(f"Error checking technical deterioration: {e}")
            return {'breakdown': False, 'breakdown_signals': [], 'technical_score': 50}
    
    def _check_regime_specific_exits(self, 
                                   data: Dict, 
                                   position_info: Dict,
                                   market_regime: str) -> Dict[str, Any]:
        """Check for regime-specific exit conditions"""
        
        days_held = (datetime.now() - position_info.get('entry_date', datetime.now())).days
        current_return = ((data.get('current_price', 0) / position_info.get('entry_price', 1)) - 1) * 100
        
        # Regime-specific exit rules
        regime_exits = {
            'bull_market': {
                'max_hold_days': 20,      # Longer holds in bull markets
                'profit_target': 25,      # 25% profit target
                'trailing_stop': 15       # 15% trailing stop from peak
            },
            'bear_market': {
                'max_hold_days': 7,       # Quick exits in bear markets
                'profit_target': 8,       # 8% profit target
                'trailing_stop': 5        # 5% trailing stop from peak
            },
            'volatile_market': {
                'max_hold_days': 10,      # Moderate holds
                'profit_target': 15,      # 15% profit target
                'trailing_stop': 10       # 10% trailing stop from peak
            },
            'neutral_market': {
                'max_hold_days': 15,      # Standard holds
                'profit_target': 20,      # 20% profit target
                'trailing_stop': 12       # 12% trailing stop from peak
            }
        }
        
        regime_rules = regime_exits.get(market_regime, regime_exits['neutral_market'])
        
        should_exit = False
        exit_reason = None
        
        # Maximum hold period reached
        if days_held >= regime_rules['max_hold_days']:
            should_exit = True
            exit_reason = f"Maximum hold period reached ({regime_rules['max_hold_days']} days)"
        
        # Profit target reached
        elif current_return >= regime_rules['profit_target']:
            should_exit = True
            exit_reason = f"Profit target reached ({current_return:.1f}% >= {regime_rules['profit_target']}%)"
        
        return {
            'should_exit': should_exit,
            'exit_reason': exit_reason,
            'days_held': days_held,
            'current_return': current_return
        }
    
    def _check_risk_management_exits(self, data: Dict, position_info: Dict) -> Dict[str, Any]:
        """Check for risk management exit triggers - AGGRESSIVE on losers"""
        
        current_price = data.get('current_price', 0)
        entry_price = position_info.get('entry_price', current_price)
        
        # Calculate current return
        position_return = ((current_price / entry_price) - 1) * 100 if entry_price > 0 else 0
        entry_date = position_info.get('entry_date')
        days_held = (datetime.now() - entry_date).days if entry_date else 0
        
        # AGGRESSIVE loss cutting - much faster exits for losing trades
        quick_exit_triggers = []
        
        # 1. Fast stop loss for immediate losses (3% in first 2 days)
        if days_held <= 2 and position_return <= -3:
            quick_exit_triggers.append('early_loss_3pct')
        
        # 2. Progressive stop loss gets tighter over time
        if days_held <= 1 and position_return <= -2:
            quick_exit_triggers.append('day1_loss_2pct')
        elif days_held <= 3 and position_return <= -4:
            quick_exit_triggers.append('day3_loss_4pct')
        elif days_held <= 5 and position_return <= -5:
            quick_exit_triggers.append('day5_loss_5pct')
        
        # 3. Any loss after 3 days is suspicious - tighten stops
        if days_held >= 3 and position_return <= -1.5:
            quick_exit_triggers.append('delayed_loss_1.5pct')
        
        # 4. Hard stop loss at 6% (reduced from 8%)
        stop_loss_level = entry_price * 0.94  # 6% stop loss
        stop_loss_triggered = current_price <= stop_loss_level
        if stop_loss_triggered:
            quick_exit_triggers.append('hard_stop_6pct')
        
        # 5. Momentum reversal detection for losing trades
        momentum_metrics = self._calculate_momentum_metrics(data)
        current_momentum = momentum_metrics.get('composite_momentum', 0)
        
        if position_return < 0 and current_momentum < -5:  # Negative momentum on losing trade
            quick_exit_triggers.append('momentum_reversal_on_loss')
        
        # 6. Volume exhaustion on losing trades (very bad sign)
        volume_ratio = data.get('volume_ratio', 1.0)
        if position_return < -2 and volume_ratio < 0.7:
            quick_exit_triggers.append('volume_exhaustion_on_loss')
        
        # Decision logic: Exit if ANY quick trigger is hit
        should_exit_quickly = len(quick_exit_triggers) > 0
        
        return {
            'stop_loss_triggered': should_exit_quickly,
            'stop_loss_level': stop_loss_level,
            'position_return': position_return,
            'days_held': days_held,
            'quick_exit_triggers': quick_exit_triggers,
            'exit_reason': f"Quick loss exit: {', '.join(quick_exit_triggers)}" if should_exit_quickly else None,
            'risk_level': 'critical' if position_return <= -3 else 'high' if position_return <= -1.5 else 'moderate' if position_return <= 0 else 'low'
        }
    
    def get_position_monitoring_summary(self, positions: List[Dict]) -> Dict[str, Any]:
        """Get summary of all positions and their exit signals"""
        
        summary = {
            'total_positions': len(positions),
            'exit_candidates': [],
            'strong_holds': [],
            'at_risk': [],
            'momentum_trends': {
                'improving': 0,
                'stable': 0,
                'deteriorating': 0
            }
        }
        
        for position in positions:
            # This would integrate with live data to get current metrics
            # For now, return structure for implementation
            pass
        
        return summary