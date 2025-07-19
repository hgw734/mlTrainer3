"""
Complete documentation of the institutional-grade scoring system
with parameter weights, rating methodology, and opportunity grading.
"""

from typing import Dict, List, Tuple
import pandas as pd

class ScoringDocumentation:
    """
    Documentation and explanation of the complete scoring methodology
    used in the institutional momentum scanner.
    """
    
    def __init__(self):
        """Initialize scoring documentation"""
        self.rating_system = self._define_rating_system()
        self.parameter_weights = self._define_parameter_weights()
        self.scoring_methodology = self._define_scoring_methodology()
    
    def _define_rating_system(self) -> Dict:
        """Define the institutional rating system"""
        return {
            'letter_grades': {
                'A+': {'range': [95, 100], 'description': 'Exceptional momentum opportunity with very high confidence'},
                'A':  {'range': [90, 94],  'description': 'Outstanding momentum opportunity with high confidence'},
                'A-': {'range': [85, 89],  'description': 'Strong momentum opportunity with good confidence'},
                'B+': {'range': [80, 84],  'description': 'Good momentum opportunity with moderate confidence'},
                'B':  {'range': [75, 79],  'description': 'Above-average opportunity with reasonable confidence'},
                'B-': {'range': [70, 74],  'description': 'Moderate opportunity with some confidence'},
                'C+': {'range': [65, 69],  'description': 'Fair opportunity with limited confidence'},
                'C':  {'range': [60, 64],  'description': 'Neutral opportunity with low confidence'},
                'C-': {'range': [55, 59],  'description': 'Below-average opportunity with minimal confidence'},
                'D':  {'range': [40, 54],  'description': 'Poor opportunity with significant concerns'},
                'F':  {'range': [0, 39],   'description': 'Avoid - fundamental or technical concerns'}
            },
            'recommendations': {
                'STRONG_BUY': {'grades': ['A+', 'A'], 'action': 'Immediate position initiation recommended'},
                'BUY': {'grades': ['A-', 'B+'], 'action': 'Position initiation recommended'},
                'ACCUMULATE': {'grades': ['B', 'B-'], 'action': 'Gradual position building'},
                'HOLD': {'grades': ['C+', 'C'], 'action': 'Maintain current position if held'},
                'WEAK_HOLD': {'grades': ['C-'], 'action': 'Consider reducing position size'},
                'REDUCE': {'grades': ['D'], 'action': 'Position reduction recommended'},
                'SELL': {'grades': ['F'], 'action': 'Exit position recommended'}
            }
        }
    
    def _define_parameter_weights(self) -> Dict:
        """Define all parameters and their weights in the final score"""
        return {
            'primary_components': {
                'momentum_analysis': {
                    'base_weight': 0.35,  # Varies by market regime
                    'description': 'Multi-timeframe price momentum analysis',
                    'sub_components': {
                        'momentum_3d': {'weight': 0.15, 'description': '3-day price momentum %'},
                        'momentum_5d': {'weight': 0.20, 'description': '5-day price momentum %'},
                        'momentum_10d': {'weight': 0.25, 'description': '10-day price momentum %'},
                        'momentum_20d': {'weight': 0.25, 'description': '20-day price momentum %'},
                        'momentum_50d': {'weight': 0.15, 'description': '50-day price momentum %'},
                        'momentum_persistence': {'weight': 0.10, 'description': 'Percentage of positive days'},
                        'momentum_quality': {'weight': 0.10, 'description': 'Weighted average momentum quality'}
                    }
                },
                'technical_analysis': {
                    'base_weight': 0.30,  # Varies by market regime
                    'description': 'Technical indicators and chart patterns',
                    'sub_components': {
                        'rsi_analysis': {'weight': 0.20, 'description': 'RSI momentum zones (2-period and 14-period)'},
                        'macd_analysis': {'weight': 0.15, 'description': 'MACD trend and histogram analysis'},
                        'bollinger_bands': {'weight': 0.10, 'description': 'Band position and width analysis'},
                        'stochastic': {'weight': 0.10, 'description': 'Stochastic oscillator signals'},
                        'williams_r': {'weight': 0.05, 'description': 'Williams %R momentum indicator'},
                        'cci': {'weight': 0.05, 'description': 'Commodity Channel Index'},
                        'adx_strength': {'weight': 0.10, 'description': 'Trend strength and direction'},
                        'volume_analysis': {'weight': 0.15, 'description': 'Volume breakouts and patterns'},
                        'price_acceleration': {'weight': 0.10, 'description': 'Second derivative momentum'}
                    }
                },
                'fundamental_analysis': {
                    'base_weight': 0.25,  # Varies by market regime
                    'description': 'Earnings, financials, and analyst coverage',
                    'sub_components': {
                        'earnings_analysis': {'weight': 0.30, 'description': 'Growth, surprises, and consistency'},
                        'analyst_ratings': {'weight': 0.25, 'description': 'Consensus ratings and price targets'},
                        'financial_health': {'weight': 0.25, 'description': 'Balance sheet and profitability metrics'},
                        'valuation_metrics': {'weight': 0.20, 'description': 'P/E, PEG, P/B, P/S ratios'}
                    }
                },
                'sentiment_analysis': {
                    'base_weight': 0.10,  # Varies by market regime
                    'description': 'News and social media sentiment',
                    'sub_components': {
                        'news_sentiment': {'weight': 0.60, 'description': 'Financial news sentiment analysis'},
                        'social_sentiment': {'weight': 0.40, 'description': 'Reddit and social media sentiment'},
                        'mention_volume': {'weight': 0.20, 'description': 'Volume of social mentions boost'},
                        'sentiment_momentum': {'weight': 0.20, 'description': 'Recent vs historical sentiment trend'}
                    }
                }
            },
            'adjustment_factors': {
                'confidence_multiplier': {
                    'description': 'Adjusts score based on data availability and consistency',
                    'factors': {
                        'data_coverage': 'Technical indicator coverage (0.25 weight)',
                        'fundamental_coverage': 'Fundamental data availability (0.25 weight)',
                        'sentiment_coverage': 'Sentiment data availability (0.20 weight)',
                        'score_consistency': 'Consistency across analysis types (0.30 weight)'
                    }
                },
                'market_regime_adjustment': {
                    'description': 'Adjusts weights based on current market conditions',
                    'regimes': {
                        'bull_market': {'momentum': 0.40, 'technical': 0.35, 'fundamental': 0.20, 'sentiment': 0.05},
                        'bear_market': {'momentum': 0.25, 'technical': 0.30, 'fundamental': 0.35, 'sentiment': 0.10},
                        'volatile_market': {'momentum': 0.45, 'technical': 0.40, 'fundamental': 0.10, 'sentiment': 0.05},
                        'neutral_market': {'momentum': 0.35, 'technical': 0.30, 'fundamental': 0.25, 'sentiment': 0.10}
                    }
                },
                'risk_adjustment': {
                    'description': 'Risk-based score penalties and bonuses',
                    'factors': {
                        'volatility_penalty': 'High volatility reduces score (max -10 points)',
                        'drawdown_penalty': 'Large drawdowns reduce score (max -15 points)',
                        'volume_bonus': 'High volume breakouts boost score (max +10 points)',
                        'institutional_quality': 'Large cap, liquid stocks get bonus (max +5 points)'
                    }
                }
            }
        }
    
    def _define_scoring_methodology(self) -> Dict:
        """Define the complete scoring methodology"""
        return {
            'score_calculation_steps': [
                {
                    'step': 1,
                    'description': 'Calculate individual component scores',
                    'details': 'Each analysis component generates a 0-100 score based on specific criteria'
                },
                {
                    'step': 2,
                    'description': 'Apply market regime weights',
                    'details': 'Adjust component weights based on current market conditions (bull/bear/volatile/neutral)'
                },
                {
                    'step': 3,
                    'description': 'Calculate weighted composite score',
                    'details': 'Combine all components using adaptive weights: Score = Σ(Component_Score × Weight)'
                },
                {
                    'step': 4,
                    'description': 'Apply confidence adjustment',
                    'details': 'Adjust score based on data availability: Adjusted_Score = Score × (Confidence / 100)'
                },
                {
                    'step': 5,
                    'description': 'Apply risk adjustments',
                    'details': 'Add/subtract points based on risk factors (volatility, drawdown, volume)'
                },
                {
                    'step': 6,
                    'description': 'Assign institutional grade',
                    'details': 'Convert final score to letter grade (A+ to F) and recommendation'
                }
            ],
            'quality_thresholds': {
                'minimum_data_requirements': {
                    'price_history': '50+ trading days',
                    'volume_data': 'Daily volume for momentum analysis',
                    'technical_indicators': 'At least 3 of 8 indicators must be calculable'
                },
                'confidence_scoring': {
                    'high_confidence': '80%+ (all data available, consistent signals)',
                    'medium_confidence': '60-79% (most data available, some inconsistency)',
                    'low_confidence': '40-59% (limited data, mixed signals)',
                    'very_low_confidence': '<40% (insufficient data, conflicting signals)'
                }
            }
        }
    
    def get_parameter_breakdown(self, market_regime: str = 'neutral_market') -> pd.DataFrame:
        """Get detailed parameter breakdown for current market regime"""
        try:
            weights = self.parameter_weights['primary_components']
            regime_adj = self.parameter_weights['adjustment_factors']['market_regime_adjustment']['regimes'][market_regime]
            
            data = []
            for component, details in weights.items():
                adjusted_weight = details['base_weight'] * regime_adj.get(component.split('_')[0], 1.0)
                
                data.append({
                    'Component': component.replace('_', ' ').title(),
                    'Base Weight': f"{details['base_weight']:.1%}",
                    'Adjusted Weight': f"{adjusted_weight:.1%}",
                    'Description': details['description']
                })
                
                # Add sub-components
                for sub_comp, sub_details in details['sub_components'].items():
                    data.append({
                        'Component': f"  └─ {sub_comp.replace('_', ' ').title()}",
                        'Base Weight': f"{sub_details['weight']:.1%}",
                        'Adjusted Weight': f"{sub_details['weight'] * adjusted_weight:.1%}",
                        'Description': sub_details['description'],

                    })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            return pd.DataFrame([{'Component': 'Error', 'Description': str(e)}])
    
    def get_rating_explanation(self, score: float, confidence: float) -> Dict[str, str]:
        """Get detailed explanation for a given score and confidence"""
        try:
            # Find letter grade
            grade = 'C'
            for letter, details in self.rating_system['letter_grades'].items():
                if details['range'][0] <= score <= details['range'][1]:
                    grade = letter
                    break
            
            # Find recommendation
            recommendation = 'HOLD'
            for rec, details in self.rating_system['recommendations'].items():
                if grade in details['grades']:
                    recommendation = rec
                    break
            
            grade_info = self.rating_system['letter_grades'][grade]
            rec_info = self.rating_system['recommendations'][recommendation]
            
            return {
                'grade': grade,
                'recommendation': recommendation,
                'score_range': f"{grade_info['range'][0]}-{grade_info['range'][1]}",
                'grade_description': grade_info['description'],
                'action_description': rec_info['action'],
                'confidence_level': self._get_confidence_description(confidence),
                'risk_assessment': self._get_risk_assessment(score, confidence)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _get_confidence_description(self, confidence: float) -> str:
        """Get confidence level description"""
        if confidence >= 80:
            return 'High Confidence - All data sources available, consistent signals'
        elif confidence >= 60:
            return 'Medium Confidence - Most data available, minor inconsistencies'
        elif confidence >= 40:
            return 'Low Confidence - Limited data, mixed signals'
        else:
            return 'Very Low Confidence - Insufficient data, conflicting signals'
    
    def _get_risk_assessment(self, score: float, confidence: float) -> str:
        """Get risk assessment description"""
        if score >= 85 and confidence >= 70:
            return 'Low Risk - Strong fundamentals and technical signals'
        elif score >= 70 and confidence >= 60:
            return 'Moderate Risk - Good opportunity with reasonable confidence'
        elif score >= 55:
            return 'Moderate-High Risk - Mixed signals, careful position sizing recommended'
        else:
            return 'High Risk - Weak signals, significant downside potential'
    
    def get_market_regime_impact(self) -> pd.DataFrame:
        """Get table showing how market regimes affect scoring weights"""
        regimes = self.parameter_weights['adjustment_factors']['market_regime_adjustment']['regimes']
        
        data = []
        for regime, weights in regimes.items():
            data.append({
                'Market Regime': regime.replace('_', ' ').title(),
                'Momentum Weight': f"{weights['momentum']:.1%}",
                'Technical Weight': f"{weights['technical']:.1%}",
                'Fundamental Weight': f"{weights['fundamental']:.1%}",
                'Sentiment Weight': f"{weights['sentiment']:.1%}",
                'Primary Focus': max(weights.items(), key=lambda x: x[1])[0].title()
            })
        
        return pd.DataFrame(data)
    
    def get_scoring_summary(self) -> str:
        """Get comprehensive scoring methodology summary"""
        return """
        ## AdvanS 3 Institutional Scoring Methodology
        
        **Adaptive Multi-Factor Analysis System**
        - 4 Primary Components: Momentum, Technical, Fundamental, Sentiment
        - Dynamic weight adjustment based on market regime
        - Confidence-based score adjustment
        - Risk-adjusted final scoring
        
        **Rating Scale: A+ to F (95-100 down to 0-39)**
        - A+ to A-: Strong opportunities (STRONG_BUY/BUY)
        - B+ to B-: Good opportunities (BUY/ACCUMULATE)
        - C+ to C-: Neutral opportunities (HOLD/WEAK_HOLD)
        - D to F: Avoid (REDUCE/SELL)
        
        **Automatic Parameter Adjustment:**
        - Updates 30 minutes before market open
        - Adjusts when market conditions change significantly
        - Optimizes for current volatility environment
        - Adapts to intraday trading sessions
        """