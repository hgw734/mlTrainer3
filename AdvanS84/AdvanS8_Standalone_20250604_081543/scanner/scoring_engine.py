"""
Institutional-grade scoring engine for momentum trading.
Combines technical, fundamental, and sentiment analysis with adaptive weights.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ScoringEngine:
    """
    Advanced scoring engine that combines multiple analysis components
    with market regime-adaptive weighting for institutional-grade results.
    """
    
    def __init__(self):
        """Initialize scoring engine with adaptive parameters"""
        # Base weights for different market regimes
        self.regime_weights = {
            'bull_market': {
                'technical': 0.4,
                'momentum': 0.3,
                'fundamental': 0.2,
                'sentiment': 0.1
            },
            'bear_market': {
                'technical': 0.3,
                'momentum': 0.2,
                'fundamental': 0.4,
                'sentiment': 0.1
            },
            'volatile_market': {
                'technical': 0.5,
                'momentum': 0.2,
                'fundamental': 0.2,
                'sentiment': 0.1
            },
            'neutral_market': {
                'technical': 0.35,
                'momentum': 0.25,
                'fundamental': 0.25,
                'sentiment': 0.15
            }
        }
        
        # Quality multipliers for different score ranges
        self.quality_multipliers = {
            'excellent': {'min': 85, 'multiplier': 1.1},
            'good': {'min': 70, 'multiplier': 1.05},
            'average': {'min': 50, 'multiplier': 1.0},
            'poor': {'min': 30, 'multiplier': 0.9},
            'very_poor': {'min': 0, 'multiplier': 0.8}
        }
    
    def calculate_composite_score(self, 
                                technical_results: Dict[str, Any],
                                fundamental_results: Optional[Dict[str, Any]] = None,
                                sentiment_results: Optional[Dict[str, Any]] = None,
                                market_regime: str = 'neutral_market',
                                timeframe: str = 'all',
                                money_flow_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Calculate institutional-grade composite score
        
        Args:
            technical_results: Technical analysis results
            fundamental_results: Fundamental analysis results
            sentiment_results: Sentiment analysis results
            market_regime: Current market regime
            timeframe: Analysis timeframe
            money_flow_results: Money flow analysis results (MarketStructureEdge)
            
        Returns:
            Dictionary with composite scores and analysis
        """
        try:
            # Get regime-specific weights
            weights = self.regime_weights.get(market_regime, self.regime_weights['neutral_market'])
            
            # Extract component scores
            technical_score = technical_results.get('composite_score', 50.0)
            momentum_score = self._calculate_momentum_score(technical_results)
            fundamental_score = fundamental_results.get('fundamental_score', 50.0) if fundamental_results else 50.0
            sentiment_score = sentiment_results.get('sentiment_score', 50.0) if sentiment_results else 50.0
            
            # Calculate money flow score (MarketStructureEdge methodology)
            money_flow_score = self._calculate_money_flow_score(money_flow_results) if money_flow_results else 50.0
            
            # Adjust weights to include money flow (reduce others proportionally)
            total_original_weight = weights['technical'] + weights['momentum'] + weights['fundamental'] + weights['sentiment']
            money_flow_weight = 0.15  # 15% weight for money flow analysis
            scaling_factor = (1.0 - money_flow_weight) / total_original_weight
            
            adjusted_weights = {
                'technical': weights['technical'] * scaling_factor,
                'momentum': weights['momentum'] * scaling_factor,
                'fundamental': weights['fundamental'] * scaling_factor,
                'sentiment': weights['sentiment'] * scaling_factor,
                'money_flow': money_flow_weight
            }
            
            # Calculate weighted composite score with money flow
            composite_score = (
                technical_score * adjusted_weights['technical'] +
                momentum_score * adjusted_weights['momentum'] +
                fundamental_score * adjusted_weights['fundamental'] +
                sentiment_score * adjusted_weights['sentiment'] +
                money_flow_score * adjusted_weights['money_flow']
            )
            
            # Apply quality multipliers
            composite_score = self._apply_quality_multiplier(composite_score)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence(
                technical_results, fundamental_results, sentiment_results
            )
            
            # Determine institutional grade
            grade = self._assign_institutional_grade(composite_score, confidence_score)
            
            # Generate recommendation
            recommendation = self._generate_recommendation(composite_score, confidence_score, market_regime)
            
            return {
                'composite_score': round(composite_score, 2),
                'technical_score': round(technical_score, 2),
                'momentum_score': round(momentum_score, 2),
                'fundamental_score': round(fundamental_score, 2),
                'sentiment_score': round(sentiment_score, 2),
                'confidence_score': round(confidence_score, 2),
                'institutional_grade': grade,
                'recommendation': recommendation,
                'market_regime': market_regime,
                'timeframe': timeframe,
                'component_weights': weights
            }
            
        except Exception as e:
            logger.error(f"Composite score calculation failed: {e}")
            return self._empty_scoring_result()
    
    def _calculate_momentum_score(self, technical_results: Dict[str, Any]) -> float:
        """Calculate specialized momentum score from technical analysis"""
        try:
            momentum_components = []
            
            # Multi-timeframe momentum
            for period in ['3d', '5d', '10d', '20d', '50d']:
                momentum_key = f'momentum_{period}'
                if momentum_key in technical_results:
                    momentum_val = technical_results[momentum_key]
                    # Convert momentum percentage to score (0-100)
                    if momentum_val > 20:
                        score = 90
                    elif momentum_val > 15:
                        score = 85
                    elif momentum_val > 10:
                        score = 75
                    elif momentum_val > 5:
                        score = 65
                    elif momentum_val > 0:
                        score = 55
                    elif momentum_val > -5:
                        score = 45
                    elif momentum_val > -10:
                        score = 35
                    else:
                        score = 25
                    
                    momentum_components.append(score)
            
            # Volume momentum
            volume_signal = technical_results.get('volume_signal', 'NORMAL')
            if volume_signal == 'HIGH_VOLUME_BREAKOUT':
                momentum_components.append(90)
            elif volume_signal == 'ABOVE_AVERAGE':
                momentum_components.append(70)
            elif volume_signal == 'NORMAL':
                momentum_components.append(50)
            else:
                momentum_components.append(30)
            
            # Price acceleration
            acceleration_signal = technical_results.get('acceleration_signal', 'STABLE')
            if acceleration_signal == 'ACCELERATING_UP':
                momentum_components.append(85)
            elif acceleration_signal == 'STABLE':
                momentum_components.append(50)
            else:
                momentum_components.append(25)
            
            return np.mean(momentum_components) if momentum_components else 50.0
            
        except Exception as e:
            logger.error(f"Momentum score calculation failed: {e}")
            return 50.0
    
    def _apply_quality_multiplier(self, score: float) -> float:
        """Apply quality-based score multipliers"""
        try:
            for quality, params in self.quality_multipliers.items():
                if score >= params['min']:
                    return score * params['multiplier']
            
            return score * 0.8  # Default poor quality multiplier
            
        except Exception as e:
            logger.error(f"Quality multiplier application failed: {e}")
            return score
    
    def _calculate_confidence(self, 
                            technical_results: Dict[str, Any],
                            fundamental_results: Optional[Dict[str, Any]],
                            sentiment_results: Optional[Dict[str, Any]]) -> float:
        """Calculate confidence score based on data availability and consistency"""
        try:
            confidence_factors = []
            
            # Technical analysis confidence
            tech_indicators = ['rsi_14', 'macd', 'bb_position', 'adx']
            tech_coverage = sum(1 for indicator in tech_indicators if indicator in technical_results)
            confidence_factors.append((tech_coverage / len(tech_indicators)) * 100)
            
            # Fundamental analysis confidence
            if fundamental_results:
                fund_indicators = ['earnings_score', 'analyst_score', 'financial_score', 'valuation_score']
                fund_coverage = sum(1 for indicator in fund_indicators if indicator in fundamental_results)
                confidence_factors.append((fund_coverage / len(fund_indicators)) * 100)
            else:
                confidence_factors.append(30)  # Penalty for missing fundamental data
            
            # Sentiment analysis confidence
            if sentiment_results:
                sentiment_indicators = ['news_sentiment', 'social_sentiment']
                sentiment_coverage = sum(1 for indicator in sentiment_indicators if indicator in sentiment_results)
                confidence_factors.append((sentiment_coverage / len(sentiment_indicators)) * 100)
                
                # Bonus for high mention volume
                mention_volume = sentiment_results.get('mention_volume', 0)
                if mention_volume > 50:
                    confidence_factors.append(90)
                elif mention_volume > 20:
                    confidence_factors.append(70)
                else:
                    confidence_factors.append(50)
            else:
                confidence_factors.append(40)  # Penalty for missing sentiment data
            
            # Score consistency check
            scores = [
                technical_results.get('composite_score', 50),
                fundamental_results.get('fundamental_score', 50) if fundamental_results else 50,
                sentiment_results.get('sentiment_score', 50) if sentiment_results else 50
            ]
            
            score_std = np.std(scores)
            if score_std < 10:  # Consistent scores
                confidence_factors.append(85)
            elif score_std < 20:
                confidence_factors.append(70)
            else:
                confidence_factors.append(50)
            
            return np.mean(confidence_factors)
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 50.0
    
    def _calculate_money_flow_score(self, money_flow_results: Dict[str, Any]) -> float:
        """
        Calculate money flow score based on MarketStructureEdge methodology
        Focuses on supply/demand imbalances and institutional flow patterns
        """
        try:
            # Extract key money flow metrics
            demand_pressure = money_flow_results.get('demand_pressure', 0.5)
            supply_pressure = money_flow_results.get('supply_pressure', 0.5)
            imbalance_score = money_flow_results.get('imbalance_score', 0.0)
            flow_strength = money_flow_results.get('flow_strength', 50.0)
            money_flow_index = money_flow_results.get('money_flow_index', 50.0)
            
            # Get institutional flow indicators
            institutional_flow = money_flow_results.get('institutional_flow', {})
            institutional_participation = institutional_flow.get('institutional_participation', 0.0)
            institutional_signal = institutional_flow.get('institutional_signal', False)
            
            # Calculate base score from supply/demand imbalance
            # Following MarketStructureEdge: Buy rising demand + falling supply
            supply_demand_score = 50 + (imbalance_score * 50)  # Scale -1 to +1 into 0-100
            
            # Apply flow strength multiplier
            flow_adjusted_score = supply_demand_score * (flow_strength / 100)
            
            # Money Flow Index component
            mfi_component = money_flow_index
            
            # Institutional participation bonus
            institutional_bonus = 0
            if institutional_signal:
                institutional_bonus = 15
            elif institutional_participation > 0.3:
                institutional_bonus = 10
            elif institutional_participation > 0.15:
                institutional_bonus = 5
            
            # Combine components
            money_flow_score = (
                flow_adjusted_score * 0.4 +  # Supply/demand balance (40%)
                mfi_component * 0.4 +         # Money flow index (40%)
                institutional_bonus           # Institutional flow bonus (up to 15 points)
            )
            
            # Volume quality adjustment
            volume_profile = money_flow_results.get('volume_profile', {})
            if volume_profile.get('volume_surge', False):
                money_flow_score += 5  # Bonus for volume surge
            
            # Ensure score stays within bounds
            money_flow_score = max(0, min(100, money_flow_score))
            
            return float(money_flow_score)
            
        except Exception as e:
            logger.error(f"Money flow score calculation failed: {e}")
            return 50.0
    
    def _assign_institutional_grade(self, composite_score: float, confidence_score: float) -> str:
        """Assign institutional letter grade based on score and confidence"""
        try:
            # Adjust score based on confidence
            adjusted_score = composite_score * (confidence_score / 100)
            
            if adjusted_score >= 90:
                return 'A+'
            elif adjusted_score >= 85:
                return 'A'
            elif adjusted_score >= 80:
                return 'A-'
            elif adjusted_score >= 75:
                return 'B+'
            elif adjusted_score >= 70:
                return 'B'
            elif adjusted_score >= 65:
                return 'B-'
            elif adjusted_score >= 60:
                return 'C+'
            elif adjusted_score >= 55:
                return 'C'
            elif adjusted_score >= 50:
                return 'C-'
            elif adjusted_score >= 40:
                return 'D'
            else:
                return 'F'
                
        except Exception as e:
            logger.error(f"Grade assignment failed: {e}")
            return 'C'
    
    def _generate_recommendation(self, composite_score: float, confidence_score: float, market_regime: str) -> str:
        """Generate institutional trading recommendation"""
        try:
            # Base recommendation from score
            if composite_score >= 85 and confidence_score >= 70:
                base_rec = 'STRONG_BUY'
            elif composite_score >= 75 and confidence_score >= 60:
                base_rec = 'BUY'
            elif composite_score >= 65:
                base_rec = 'ACCUMULATE'
            elif composite_score >= 55:
                base_rec = 'HOLD'
            elif composite_score >= 45:
                base_rec = 'WEAK_HOLD'
            elif composite_score >= 35:
                base_rec = 'REDUCE'
            else:
                base_rec = 'SELL'
            
            # Adjust for market regime
            regime_adjustments = {
                'bear_market': {
                    'STRONG_BUY': 'BUY',
                    'BUY': 'ACCUMULATE',
                    'ACCUMULATE': 'HOLD',
                    'REDUCE': 'SELL'
                },
                'volatile_market': {
                    'STRONG_BUY': 'BUY',
                    'WEAK_HOLD': 'REDUCE'
                }
            }
            
            if market_regime in regime_adjustments:
                adjustments = regime_adjustments[market_regime]
                base_rec = adjustments.get(base_rec, base_rec)
            
            return base_rec
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return 'HOLD'
    
    def get_recommendation_explanation(self, results: Dict[str, Any]) -> str:
        """Get detailed explanation for the recommendation"""
        try:
            score = results.get('composite_score', 50)
            confidence = results.get('confidence_score', 50)
            grade = results.get('institutional_grade', 'C')
            recommendation = results.get('recommendation', 'HOLD')
            
            explanations = {
                'STRONG_BUY': f"Exceptional opportunity with {grade} grade (Score: {score:.1f}, Confidence: {confidence:.1f}%)",
                'BUY': f"Strong momentum signal with {grade} grade (Score: {score:.1f}, Confidence: {confidence:.1f}%)",
                'ACCUMULATE': f"Positive momentum trend with {grade} grade (Score: {score:.1f}, Confidence: {confidence:.1f}%)",
                'HOLD': f"Neutral signals with {grade} grade (Score: {score:.1f}, Confidence: {confidence:.1f}%)",
                'WEAK_HOLD': f"Mixed signals with {grade} grade (Score: {score:.1f}, Confidence: {confidence:.1f}%)",
                'REDUCE': f"Weakening momentum with {grade} grade (Score: {score:.1f}, Confidence: {confidence:.1f}%)",
                'SELL': f"Poor momentum signals with {grade} grade (Score: {score:.1f}, Confidence: {confidence:.1f}%)"
            }
            
            return explanations.get(recommendation, f"Standard analysis with {grade} grade")
            
        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            return "Analysis completed with standard methodology"
    
    def _empty_scoring_result(self) -> Dict[str, Any]:
        """Return empty scoring result structure"""
        return {
            'composite_score': 50.0,
            'technical_score': 50.0,
            'momentum_score': 50.0,
            'fundamental_score': 50.0,
            'sentiment_score': 50.0,
            'confidence_score': 50.0,
            'institutional_grade': 'C',
            'recommendation': 'HOLD',
            'market_regime': 'neutral_market',
            'timeframe': 'all',
            'component_weights': self.regime_weights['neutral_market']
        }