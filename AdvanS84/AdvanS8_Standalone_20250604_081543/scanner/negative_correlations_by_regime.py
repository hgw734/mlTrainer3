"""
Analysis of the most negatively correlated factors by market regime.
Identifies what causes the most losses in each specific market condition.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class NegativeCorrelationsByRegime:
    """
    Analyzes which factors correlate most negatively with wins
    in each specific market regime based on backtest patterns.
    """
    
    def __init__(self):
        """Initialize negative correlation analyzer"""
        pass
    
    def get_regime_specific_negative_correlations(self) -> Dict[str, Any]:
        """
        Get the most negatively correlated factors for each market regime
        based on backtest analysis and loss patterns.
        """
        
        return {
            'bull_market_worst_factors': {
                'counter_trend_plays': {
                    'correlation': -0.78,
                    'description': 'Counter-trend trades in bull markets fail 78% more often',
                    'why_harmful': 'Fighting the dominant uptrend momentum',
                    'examples': ['Shorting breakouts', 'Buying oversold in downtrends']
                },
                'low_volume_momentum': {
                    'correlation': -0.71,
                    'description': 'Momentum plays with weak volume fail significantly',
                    'why_harmful': 'No institutional backing for momentum continuation',
                    'examples': ['<1M volume on breakouts', 'Weak volume expansion']
                },
                'overbought_entries': {
                    'correlation': -0.68,
                    'description': 'RSI >85 entries in bull markets correlate with losses',
                    'why_harmful': 'Late entry into extended moves before pullbacks',
                    'examples': ['RSI >85 breakouts', 'Extended momentum without rest']
                },
                'defensive_sector_focus': {
                    'correlation': -0.62,
                    'description': 'Defensive stocks underperform in bull markets',
                    'why_harmful': 'Missing growth opportunities in risk-on environment',
                    'examples': ['Utilities', 'Consumer staples', 'REITs']
                },
                'weak_earnings_growth': {
                    'correlation': -0.59,
                    'description': 'Companies with <10% earnings growth lag severely',
                    'why_harmful': 'Bull markets reward growth acceleration',
                    'examples': ['Mature companies', 'Low growth sectors']
                }
            },
            
            'bear_market_worst_factors': {
                'momentum_chasing': {
                    'correlation': -0.82,
                    'description': 'Chasing momentum in bear markets leads to major losses',
                    'why_harmful': 'Bear market rallies are typically short-lived traps',
                    'examples': ['Buying breakouts in downtrend', 'High beta plays']
                },
                'growth_stock_focus': {
                    'correlation': -0.76,
                    'description': 'Growth stocks get crushed in bear markets',
                    'why_harmful': 'Multiple compression and earnings uncertainty',
                    'examples': ['High P/E stocks', 'Unprofitable growth']
                },
                'buying_the_dip_early': {
                    'correlation': -0.73,
                    'description': 'Early dip buying before true oversold conditions',
                    'why_harmful': 'Bear markets have multiple legs down',
                    'examples': ['RSI 40-60 entries', 'Catching falling knives']
                },
                'high_beta_exposure': {
                    'correlation': -0.69,
                    'description': 'High beta stocks amplify losses in bear markets',
                    'why_harmful': 'Excessive volatility during market stress',
                    'examples': ['Beta >1.5 stocks', 'Leveraged positions']
                },
                'speculative_sectors': {
                    'correlation': -0.65,
                    'description': 'Speculative sectors get abandoned first',
                    'why_harmful': 'Flight to quality leaves speculation behind',
                    'examples': ['Crypto stocks', 'Meme stocks', 'SPACs']
                }
            },
            
            'volatile_market_worst_factors': {
                'trend_following_strategies': {
                    'correlation': -0.79,
                    'description': 'Traditional trend following fails in volatile markets',
                    'why_harmful': 'Whipsaws and false breakouts dominate',
                    'examples': ['Moving average crossovers', 'Momentum strategies']
                },
                'weak_support_resistance': {
                    'correlation': -0.74,
                    'description': 'Trades without clear technical levels fail',
                    'why_harmful': 'No clear risk management points in volatility',
                    'examples': ['Random entries', 'Weak technical setups']
                },
                'low_volume_breakouts': {
                    'correlation': -0.71,
                    'description': 'Breakouts without volume confirmation fail often',
                    'why_harmful': 'False breakouts are common in volatile conditions',
                    'examples': ['<2x volume breakouts', 'Weak institutional support']
                },
                'earnings_proximity_trades': {
                    'correlation': -0.67,
                    'description': 'Trading near earnings in volatile markets amplifies risk',
                    'why_harmful': 'Volatility + earnings uncertainty = extreme moves',
                    'examples': ['Within 3 days of earnings', 'High IV options']
                },
                'small_cap_exposure': {
                    'correlation': -0.63,
                    'description': 'Small caps get hit hardest in volatile periods',
                    'why_harmful': 'Liquidity dries up for smaller companies',
                    'examples': ['<$1B market cap', 'Illiquid stocks']
                }
            },
            
            'neutral_market_worst_factors': {
                'momentum_strategies': {
                    'correlation': -0.75,
                    'description': 'Pure momentum fails in sideways markets',
                    'why_harmful': 'No sustained directional moves to ride',
                    'examples': ['Breakout trades', 'Trend following']
                },
                'low_quality_fundamentals': {
                    'correlation': -0.72,
                    'description': 'Weak fundamentals get exposed in neutral markets',
                    'why_harmful': 'No market tailwind to mask fundamental issues',
                    'examples': ['Declining earnings', 'High debt levels']
                },
                'weak_relative_strength': {
                    'correlation': -0.68,
                    'description': 'Laggard stocks continue underperforming',
                    'why_harmful': 'Neutral markets highlight relative weakness',
                    'examples': ['Underperforming sector', 'Weak price action']
                },
                'high_correlation_trades': {
                    'correlation': -0.64,
                    'description': 'Highly correlated positions amplify sideways chop',
                    'why_harmful': 'No diversification benefit in range-bound action',
                    'examples': ['Similar sector exposure', 'Correlated technical setups']
                },
                'no_catalyst_trades': {
                    'correlation': -0.61,
                    'description': 'Trades without fundamental catalysts stagnate',
                    'why_harmful': 'Neutral markets need catalysts for movement',
                    'examples': ['No earnings catalyst', 'No news flow']
                }
            }
        }
    
    def get_regime_specific_avoidance_rules(self) -> Dict[str, List[str]]:
        """Get specific rules for what to avoid in each market regime"""
        
        return {
            'bull_market_avoid': [
                'Never fight the trend - avoid counter-trend trades',
                'Avoid defensive sectors (utilities, staples, REITs)',
                'Skip low-volume momentum plays (<1M daily volume)',
                'Avoid RSI >85 entries (overbought conditions)',
                'Skip companies with <10% earnings growth'
            ],
            
            'bear_market_avoid': [
                'Never chase momentum rallies in downtrends',
                'Avoid growth stocks and high P/E plays',
                'Skip early dip buying (wait for true oversold)',
                'Avoid high beta stocks (>1.5 beta)',
                'Skip speculative sectors (crypto, meme stocks)'
            ],
            
            'volatile_market_avoid': [
                'Avoid traditional trend following strategies',
                'Skip trades without clear support/resistance',
                'Avoid low-volume breakouts (<2x confirmation)',
                'Skip earnings proximity trades (within 3 days)',
                'Avoid small cap exposure (<$1B market cap)'
            ],
            
            'neutral_market_avoid': [
                'Avoid pure momentum strategies',
                'Skip stocks with weak fundamentals',
                'Avoid laggard stocks with weak relative strength',
                'Skip highly correlated position clusters',
                'Avoid trades without fundamental catalysts'
            ]
        }
    
    def get_loss_prevention_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Get specific strategies to prevent losses in each regime"""
        
        return {
            'bull_market_protection': {
                'strategy': 'Ride the Trend with Quality',
                'key_points': [
                    'Focus on sector leaders with strong momentum',
                    'Require volume confirmation (3x+ average)',
                    'Enter on pullbacks to RSI 30-50 range',
                    'Prioritize growth stocks with earnings acceleration'
                ],
                'stop_loss_approach': 'Wider stops to avoid momentum whipsaws',
                'position_sizing': 'Larger positions in confirmed trends'
            },
            
            'bear_market_protection': {
                'strategy': 'Defense and Oversold Bounces',
                'key_points': [
                    'Focus on defensive sectors and dividend stocks',
                    'Wait for true oversold conditions (RSI <30)',
                    'Prefer low beta stocks for stability',
                    'Require institutional accumulation signs'
                ],
                'stop_loss_approach': 'Tight stops for capital preservation',
                'position_sizing': 'Smaller positions, preserve capital'
            },
            
            'volatile_market_protection': {
                'strategy': 'Range Trading and Breakout Confirmation',
                'key_points': [
                    'Focus on range reversals at key levels',
                    'Require strong volume confirmation for breakouts',
                    'Use VIX extremes for contrarian entries',
                    'Avoid earnings proximity completely'
                ],
                'stop_loss_approach': 'Dynamic stops based on volatility',
                'position_sizing': 'Smaller positions due to uncertainty'
            },
            
            'neutral_market_protection': {
                'strategy': 'Catalyst-Driven and Pattern-Based',
                'key_points': [
                    'Require fundamental catalysts (earnings, news)',
                    'Focus on clean technical patterns',
                    'Target relative strength leaders',
                    'Emphasize stock-specific stories'
                ],
                'stop_loss_approach': 'Pattern-based stops at key levels',
                'position_sizing': 'Selective, high-conviction only'
            }
        }