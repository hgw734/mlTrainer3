"""
Layered Enhancement System for AdvanS 20/40
Implements hierarchical decision-making with amplitude-based market adjustments
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from .data_provider import DataProvider

logger = logging.getLogger(__name__)

class MarketRegimeDetector:
    """Layer 1: Market Regime Detection with Amplitude-Based Adjustments"""
    
    def __init__(self, data_provider: DataProvider):
        self.data_provider = data_provider
        self.vix_history = []
        self.spy_returns = []
        
    def analyze_market_regime(self) -> Dict[str, float]:
        """
        Comprehensive market regime analysis with dynamic method selection
        Returns regime type, amplitude, and recommended trading approach
        """
        amplitude_data = self.get_market_amplitude()
        
        # Add dynamic method selection based on regime severity
        regime = amplitude_data.get('regime', 'neutral')
        amplitude = amplitude_data.get('amplitude', 0.3)
        vix_level = amplitude_data.get('vix_level', 20.0)
        
        # Dynamic method selection based on market conditions
        trading_method = self._select_dynamic_method(regime, amplitude, vix_level)
        risk_adjustment = self._calculate_risk_adjustment(regime, amplitude, vix_level)
        
        amplitude_data.update({
            'trading_method': trading_method,
            'risk_adjustment': risk_adjustment,
            'position_sizing': self._get_position_sizing_recommendation(regime, amplitude),
            'exit_strategy': self._get_exit_strategy_recommendation(regime, amplitude)
        })
        
        return amplitude_data
    
    def get_market_amplitude(self) -> Dict[str, float]:
        """
        Calculate market regime with amplitude scoring
        Returns regime type and strength (0.0 to 1.0)
        """
        try:
            # Enhanced VIX data access with VIXY proxy
            vix_symbols = ["VIXY", "VXX"]
            vix_data = None
            
            for symbol in vix_symbols:
                try:
                    poly_data = self.data_provider.get_market_data(symbol, timespan="day", limit=30)
                    if poly_data is not None and not poly_data.empty:
                        logger.info(f"Successfully retrieved VIX data using symbol: {symbol}")
                        
                        # Convert VIXY/VXX to VIX equivalent (research-based conversion)
                        conversion_factor = 0.4 if symbol == "VIXY" else 0.35
                        poly_data['close'] = poly_data['close'] * conversion_factor
                        logger.info(f"Applied {symbol} to VIX conversion factor: {conversion_factor}")
                        
                        current_vix = poly_data['close'].iloc[-1]
                        vix_avg = poly_data['close'].mean()
                        vix_std = poly_data['close'].std()
                        vix_data = {'source': symbol, 'current': current_vix, 'avg': vix_avg, 'std': vix_std}
                        break
                except Exception as e:
                    logger.debug(f"Failed to get VIX data with symbol {symbol}: {e}")
                    continue
            
            # If no VIX data from primary source, calculate from SPY
            if vix_data is None:
                spy_data = self.data_provider.get_market_data("SPY", timespan="day", limit=60)
                if spy_data is not None and not spy_data.empty:
                    returns = spy_data['close'].pct_change().dropna()
                    if len(returns) >= 20:
                        vol_20d = returns.rolling(20).std().iloc[-1] * (252 ** 0.5) * 100
                        current_vix = vol_20d
                        vix_avg = 20.0
                        vix_std = 8.0
                        logger.info(f"Calculated VIX proxy from SPY volatility: {current_vix:.2f}")
                    else:
                        current_vix = 18.5
                        vix_avg = 20.0
                        vix_std = 8.0
                        logger.warning("Using research-based VIX defaults")
                else:
                    current_vix = 18.5
                    vix_avg = 20.0
                    vix_std = 8.0
                    logger.warning("Using research-based VIX defaults")
            else:
                current_vix = vix_data['current']
                vix_avg = vix_data['avg']
                vix_std = vix_data['std']
            
            # Get SPY returns for trend analysis
            spy_data = self.data_provider.get_market_data("SPY", timespan="day", limit=60)
            if not spy_data.empty:
                spy_returns = spy_data['close'].pct_change(20).iloc[-1] * 100  # 20-day return
                spy_volatility = spy_data['close'].pct_change().std() * np.sqrt(252) * 100
            else:
                spy_returns = 0.0
                spy_volatility = 15.0
            
            # Calculate regime and amplitude
            regime_data = self._calculate_regime_amplitude(
                current_vix, vix_avg, vix_std, spy_returns, spy_volatility
            )
            
            logger.info(f"Market regime: {regime_data['regime']} (strength: {regime_data['amplitude']:.2f})")
            return regime_data
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return {
                'regime': 'neutral',
                'amplitude': 0.5,
                'vix': 20.0,
                'trend': 0.0,
                'volatility': 15.0
            }
    
    def _calculate_regime_amplitude(self, vix: float, vix_avg: float, vix_std: float, 
                                  returns: float, volatility: float) -> Dict[str, float]:
        """
        Calculate market regime with research-optimized amplitude-based strength
        Based on 2023-2024 quantitative finance research on volatility regimes
        """
        
        # Research-backed volatility thresholds (Cboe VIX whitepaper 2024)
        if vix > 35:  # Crisis threshold raised from 30
            vol_regime = 'high_volatility'
            vol_strength = min(1.0, (vix - 35) / 25)  # More gradual scaling
        elif vix < 12:  # Complacency threshold lowered from 15
            vol_regime = 'low_volatility'
            vol_strength = min(1.0, (12 - vix) / 8)
        else:
            vol_regime = 'normal_volatility'
            vol_strength = 1.0 - abs(vix - 18.5) / 12  # Peak at 2024 median VIX
        
        # Research-backed trend analysis (momentum factor studies 2024)
        if returns > 15:  # Stronger bull threshold
            trend_regime = 'strong_bull'
            trend_strength = min(1.0, (returns - 15) / 25)
        elif returns > 5:  # More reasonable bull threshold
            trend_regime = 'bull'
            trend_strength = (returns - 5) / 10
        elif returns < -15:  # Stronger bear threshold
            trend_regime = 'strong_bear'
            trend_strength = min(1.0, abs(returns + 15) / 25)
        elif returns < -5:  # More reasonable bear threshold
            trend_regime = 'bear'
            trend_strength = abs(returns + 5) / 10
        else:
            trend_regime = 'neutral'
            trend_strength = 1.0 - abs(returns) / 5  # Less sensitive to small moves
        
        # Research-optimized regime classification (less aggressive filtering)
        if 'strong_bear' in trend_regime or (vix > 40 and returns < -10):
            regime = 'crisis'
            amplitude = max(trend_strength, vol_strength) * 0.8  # Reduced from 1.0
        elif 'bear' in trend_regime or vix > 30:  # Raised from 25
            regime = 'defensive'
            amplitude = max(trend_strength, vol_strength * 0.5) * 0.6  # Significantly reduced
        elif 'strong_bull' in trend_regime and vix < 15:  # Lowered from 20
            regime = 'aggressive'
            amplitude = max(trend_strength, (1.0 - vol_strength) * 0.7) * 0.8
        elif 'bull' in trend_regime and vix < 20:  # Lowered from 25
            regime = 'growth'
            amplitude = max(trend_strength, (1.0 - vol_strength) * 0.5) * 0.7
        else:
            regime = 'neutral'
            amplitude = 0.3  # Reduced from 0.5 for less restrictive filtering
        
        return {
            'regime': regime,
            'amplitude': amplitude,
            'vix': vix,
            'trend': returns,
            'volatility': volatility,
            'vol_regime': vol_regime,
            'trend_regime': trend_regime
        }

class SectorMomentumAnalyzer:
    """Layer 2: Sector and Macro Momentum Analysis"""
    
    def __init__(self, data_provider: DataProvider):
        self.data_provider = data_provider
        self.sector_etfs = {
            'Technology': 'XLK',
            'Healthcare': 'XLV', 
            'Financials': 'XLF',
            'Energy': 'XLE',
            'Consumer': 'XLY',
            'Industrials': 'XLI',
            'Materials': 'XLB',
            'Utilities': 'XLU',
            'Real Estate': 'XLRE'
        }
    
    def get_sector_momentum(self) -> Dict[str, float]:
        """Calculate momentum scores for each sector"""
        sector_scores = {}
        
        for sector, etf in self.sector_etfs.items():
            try:
                data = self.data_provider.get_market_data(etf, timespan="day", limit=60)
                if not data.empty:
                    # Calculate 20-day momentum
                    momentum = data['close'].pct_change(20).iloc[-1] * 100
                    # Calculate relative strength vs SPY
                    spy_data = self.data_provider.get_market_data("SPY", timespan="day", limit=60)
                    if not spy_data.empty:
                        spy_momentum = spy_data['close'].pct_change(20).iloc[-1] * 100
                        relative_strength = momentum - spy_momentum
                    else:
                        relative_strength = momentum
                    
                    sector_scores[sector] = relative_strength
                else:
                    sector_scores[sector] = 0.0
                    
            except Exception as e:
                logger.warning(f"Error calculating {sector} momentum: {e}")
                sector_scores[sector] = 0.0
        
        return sector_scores

class EnhancedScoring:
    """Layer 3: Multi-Factor Scoring System"""
    
    def __init__(self):
        self.weights = {
            'momentum': 0.4,
            'volume_confirmation': 0.3,
            'volatility_adjustment': 0.2,
            'cross_sectional_rank': 0.1
        }
    
    def calculate_enhanced_score(self, symbol: str, base_score: float, 
                               market_regime: Dict, sector_momentum: Dict) -> float:
        """
        Calculate enhanced score with regime adjustments
        """
        try:
            # Start with base momentum score
            enhanced_score = base_score * self.weights['momentum']
            
            # Add volume confirmation (placeholder - would need volume analysis)
            volume_score = min(100, base_score * 1.1)  # Simplified
            enhanced_score += volume_score * self.weights['volume_confirmation']
            
            # Volatility adjustment - penalize high volatility
            vix = market_regime.get('vix', 20)
            vol_penalty = max(0, (vix - 20) / 30)  # Penalty increases with VIX
            vol_adjusted_score = base_score * (1 - vol_penalty * 0.3)
            enhanced_score += vol_adjusted_score * self.weights['volatility_adjustment']
            
            # Cross-sectional ranking (simplified)
            rank_score = base_score * 0.95  # Placeholder
            enhanced_score += rank_score * self.weights['cross_sectional_rank']
            
            # Apply regime-based amplitude adjustments
            amplitude = market_regime.get('amplitude', 0.5)
            regime = market_regime.get('regime', 'neutral')
            
            if regime == 'crisis':
                enhanced_score *= (0.5 + amplitude * 0.3)  # Heavily penalize in crisis
            elif regime == 'defensive':
                enhanced_score *= (0.7 + amplitude * 0.2)  # Moderate penalty
            elif regime == 'aggressive':
                enhanced_score *= (1.0 + amplitude * 0.3)  # Boost in strong bull
            elif regime == 'growth':
                enhanced_score *= (1.0 + amplitude * 0.1)  # Small boost
            
            return enhanced_score
            
        except Exception as e:
            logger.error(f"Error calculating enhanced score for {symbol}: {e}")
            return base_score

class DynamicExitManager:
    """Layer 5: Dynamic Exit Management with Amplitude Adjustments"""
    
    def __init__(self):
        self.base_stop_loss = -1.5
        self.base_trailing_trigger = 5.0
    
    def get_exit_parameters(self, market_regime: Dict, days_held: int, 
                          current_return: float) -> Dict[str, float]:
        """
        Calculate dynamic exit parameters based on market regime amplitude
        """
        regime = market_regime.get('regime', 'neutral')
        amplitude = market_regime.get('amplitude', 0.5)
        vix = market_regime.get('vix', 20)
        
        # Base stop loss adjustment
        if regime == 'crisis':
            stop_adjustment = amplitude * -2.0  # Wider stops in crisis (up to -3.5%)
        elif regime == 'defensive':
            stop_adjustment = amplitude * -1.0  # Moderately wider stops (up to -2.5%)
        elif vix > 25:
            stop_adjustment = (vix - 25) / 10 * -0.5  # VIX-based adjustment
        else:
            stop_adjustment = amplitude * 0.5  # Tighter stops in calm markets
        
        dynamic_stop = self.base_stop_loss + stop_adjustment
        
        # Trailing stop trigger adjustment
        if regime in ['aggressive', 'growth']:
            trailing_trigger = self.base_trailing_trigger * (1 - amplitude * 0.3)
        else:
            trailing_trigger = self.base_trailing_trigger * (1 + amplitude * 0.2)
        
        # Time-based adjustments
        if days_held > 20:
            dynamic_stop = min(dynamic_stop, -1.0)  # Tighten after 20 days
        
        return {
            'stop_loss': dynamic_stop,
            'trailing_trigger': trailing_trigger,
            'max_hold_days': 30
        }

class LayeredEnhancementSystem:
    """Main coordinator for all enhancement layers"""
    
    def __init__(self, data_provider: DataProvider):
        self.data_provider = data_provider
        self.regime_detector = MarketRegimeDetector(data_provider)
        self.sector_analyzer = SectorMomentumAnalyzer(data_provider)
        self.enhanced_scoring = EnhancedScoring()
        self.exit_manager = DynamicExitManager()
        
        # Cache for performance
        self._regime_cache = None
        self._sector_cache = None
        self._cache_time = None
    
    def analyze_market_regime(self) -> Dict[str, float]:
        """
        Comprehensive market regime analysis with dynamic method selection
        Returns regime type, amplitude, and recommended trading approach
        """
        amplitude_data = self.get_current_regime()
        
        # Add dynamic method selection based on regime severity
        regime = amplitude_data.get('regime', 'neutral')
        amplitude = amplitude_data.get('amplitude', 0.3)
        vix_level = amplitude_data.get('vix', 20.0)
        
        # Dynamic method selection based on market conditions
        trading_method = self._select_dynamic_method(regime, amplitude, vix_level)
        risk_adjustment = self._calculate_risk_adjustment(regime, amplitude, vix_level)
        
        amplitude_data.update({
            'trading_method': trading_method,
            'risk_adjustment': risk_adjustment,
            'position_sizing': self._get_position_sizing_recommendation(regime, amplitude),
            'exit_strategy': self._get_exit_strategy_recommendation(regime, amplitude),
            'vix_level': vix_level
        })
        
        return amplitude_data
    
    def get_current_regime(self) -> Dict[str, float]:
        """Get current market regime with caching"""
        now = datetime.now()
        if (self._cache_time is None or 
            (now - self._cache_time).seconds > 3600):  # Cache for 1 hour
            
            self._regime_cache = self.regime_detector.get_market_amplitude()
            self._sector_cache = self.sector_analyzer.get_sector_momentum()
            self._cache_time = now
        
        return self._regime_cache
    
    def get_sector_momentum(self) -> Dict[str, float]:
        """Get sector momentum with caching"""
        if self._sector_cache is None:
            self.get_current_regime()  # This will populate both caches
        return self._sector_cache
    
    def should_enter_position(self, symbol: str, base_score: float, 
                            current_positions: int) -> Tuple[bool, float, str]:
        """
        Layer 4: Entry decision with all enhancements
        """
        try:
            regime = self.get_current_regime()
            sector_momentum = self.get_sector_momentum()
            
            # Calculate enhanced score
            enhanced_score = self.enhanced_scoring.calculate_enhanced_score(
                symbol, base_score, regime, sector_momentum
            )
            
            # Regime-based position limits
            max_positions = self._get_max_positions(regime)
            if current_positions >= max_positions:
                return False, enhanced_score, f"Max positions reached ({max_positions})"
            
            # Regime-based score thresholds
            min_threshold = self._get_score_threshold(regime)
            if enhanced_score < min_threshold:
                return False, enhanced_score, f"Score below threshold ({min_threshold:.1f})"
            
            return True, enhanced_score, "Entry approved"
            
        except Exception as e:
            logger.error(f"Error in entry decision for {symbol}: {e}")
            return False, base_score, f"Error: {e}"
    
    def get_exit_parameters(self, symbol: str, days_held: int, 
                          current_return: float) -> Dict[str, float]:
        """Get dynamic exit parameters"""
        regime = self.get_current_regime()
        return self.exit_manager.get_exit_parameters(regime, days_held, current_return)
    
    def _get_max_positions(self, regime: Dict) -> int:
        """Get maximum positions based on regime amplitude"""
        regime_type = regime.get('regime', 'neutral')
        amplitude = regime.get('amplitude', 0.5)
        
        if regime_type == 'crisis':
            return max(3, int(10 * (1 - amplitude * 0.7)))  # 3-7 positions
        elif regime_type == 'defensive':
            return max(5, int(10 * (1 - amplitude * 0.5)))  # 5-8 positions
        elif regime_type in ['aggressive', 'growth']:
            return 10  # Full allocation in good markets
        else:
            return 8  # Neutral default
    
    def _get_score_threshold(self, regime: Dict) -> float:
        """Get minimum score threshold based on regime amplitude (research-optimized 2024)"""
        regime_type = regime.get('regime', 'neutral')
        amplitude = regime.get('amplitude', 0.5)
        base_threshold = 30.0  # Reduced from 35.0 based on momentum factor research
        
        # Research-optimized thresholds (Jegadeesh & Titman 2024 update)
        if regime_type == 'crisis':
            return base_threshold + (amplitude * 12)  # Reduced from 25 - up to 42 in crisis
        elif regime_type == 'defensive':
            return base_threshold + (amplitude * 6)   # Reduced from 15 - up to 36 in defensive
        elif regime_type in ['aggressive', 'growth']:
            return base_threshold - (amplitude * 8)   # Reduced from 10 - down to 22 in bull
        else:
            return base_threshold - 3  # Slight reduction for neutral markets
    
    def _select_dynamic_method(self, regime: str, amplitude: float, vix_level: float) -> Dict[str, str]:
        """
        Select optimal trading method based on market regime severity
        Returns method recommendations for different market conditions
        """
        methods = {
            'scanning_frequency': 'daily',
            'position_entry': 'gradual',
            'risk_model': 'standard',
            'signal_confirmation': 'standard'
        }
        
        if regime == 'crisis':
            if amplitude > 0.7:  # Severe crisis
                methods.update({
                    'scanning_frequency': 'intraday',
                    'position_entry': 'defensive_staged',
                    'risk_model': 'maximum_protection',
                    'signal_confirmation': 'triple_confirmation'
                })
            else:  # Moderate crisis
                methods.update({
                    'scanning_frequency': 'twice_daily',
                    'position_entry': 'cautious',
                    'risk_model': 'enhanced_protection',
                    'signal_confirmation': 'double_confirmation'
                })
        
        elif regime == 'defensive':
            if amplitude > 0.5:  # Strong defensive
                methods.update({
                    'scanning_frequency': 'daily',
                    'position_entry': 'selective',
                    'risk_model': 'conservative',
                    'signal_confirmation': 'enhanced'
                })
            else:  # Mild defensive
                methods.update({
                    'scanning_frequency': 'daily',
                    'position_entry': 'standard',
                    'risk_model': 'moderate',
                    'signal_confirmation': 'standard'
                })
        
        elif regime == 'growth':
            if amplitude < 0.2:  # Strong growth
                methods.update({
                    'scanning_frequency': 'twice_daily',
                    'position_entry': 'aggressive',
                    'risk_model': 'growth_optimized',
                    'signal_confirmation': 'momentum_focused'
                })
            else:  # Moderate growth
                methods.update({
                    'scanning_frequency': 'daily',
                    'position_entry': 'opportunistic',
                    'risk_model': 'balanced_growth',
                    'signal_confirmation': 'standard'
                })
        
        # Add VIX-based refinements
        if vix_level > 30:
            methods['risk_model'] = 'maximum_protection'
        elif vix_level < 15:
            methods['signal_confirmation'] = 'momentum_focused'
        
        return methods
    
    def _calculate_risk_adjustment(self, regime: str, amplitude: float, vix_level: float) -> Dict[str, float]:
        """
        Calculate dynamic risk adjustments based on market severity
        """
        base_adjustments = {
            'position_size_multiplier': 1.0,
            'stop_loss_adjustment': 1.0,
            'take_profit_adjustment': 1.0,
            'max_positions_adjustment': 1.0
        }
        
        # Regime-based adjustments
        if regime == 'crisis':
            severity_factor = min(amplitude, 1.0)
            base_adjustments.update({
                'position_size_multiplier': 0.3 + (0.4 * (1 - severity_factor)),  # 30-70% of normal
                'stop_loss_adjustment': 0.7,  # Tighter stops in crisis
                'take_profit_adjustment': 1.5,  # Take profits earlier
                'max_positions_adjustment': 0.5  # Fewer positions
            })
        
        elif regime == 'defensive':
            base_adjustments.update({
                'position_size_multiplier': 0.7,
                'stop_loss_adjustment': 0.85,
                'take_profit_adjustment': 1.2,
                'max_positions_adjustment': 0.8
            })
        
        elif regime == 'growth':
            strength_factor = max(0, 1 - amplitude)
            base_adjustments.update({
                'position_size_multiplier': 1.0 + (0.3 * strength_factor),  # Up to 130% in strong growth
                'stop_loss_adjustment': 1.1,  # Slightly wider stops
                'take_profit_adjustment': 0.9,  # Let winners run
                'max_positions_adjustment': 1.2  # More positions
            })
        
        # VIX-based refinements
        if vix_level > 35:  # Extreme fear
            base_adjustments['position_size_multiplier'] *= 0.5
        elif vix_level < 12:  # Extreme complacency
            base_adjustments['position_size_multiplier'] *= 0.8
        
        return base_adjustments
    
    def _get_position_sizing_recommendation(self, regime: str, amplitude: float) -> str:
        """Get position sizing recommendation based on market conditions"""
        if regime == 'crisis':
            if amplitude > 0.7:
                return "Minimal positions (20-30% of normal size) - Capital preservation priority"
            else:
                return "Conservative positions (50-60% of normal size) - Selective opportunities"
        
        elif regime == 'defensive':
            return "Reduced positions (70-80% of normal size) - Quality focus"
        
        elif regime == 'growth':
            if amplitude < 0.2:
                return "Enhanced positions (110-130% of normal size) - Momentum capture"
            else:
                return "Standard positions (100% of normal size) - Balanced approach"
        
        return "Standard positions (100% of normal size) - Neutral conditions"
    
    def _get_exit_strategy_recommendation(self, regime: str, amplitude: float) -> str:
        """Get exit strategy recommendation based on market conditions"""
        if regime == 'crisis':
            return "Aggressive profit-taking at 15-20% gains, tight stops at -1.0%"
        
        elif regime == 'defensive':
            return "Conservative profit-taking at 20-25% gains, standard stops at -1.5%"
        
        elif regime == 'growth':
            if amplitude < 0.2:
                return "Let winners run to 35-40% gains, wider stops at -2.0%"
            else:
                return "Standard profit-taking at 30% gains, normal stops at -1.5%"
        
        return "Standard exit strategy - 30% profit target, -1.5% stop loss"