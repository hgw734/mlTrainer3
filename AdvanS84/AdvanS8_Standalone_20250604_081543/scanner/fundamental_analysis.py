"""
Fundamental analysis module for institutional-grade stock evaluation.
Includes earnings analysis, analyst ratings, financial metrics, and valuation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class FundamentalAnalyzer:
    """
    Institutional-grade fundamental analysis combining earnings,
    analyst ratings, financial health, and valuation metrics.
    """
    
    def __init__(self):
        """Initialize fundamental analyzer"""
        self.earnings_weight = 0.3
        self.analyst_weight = 0.25
        self.financial_weight = 0.25
        self.valuation_weight = 0.2
        
    def analyze(self, symbol: str, company_data: Optional[Dict] = None,
                financial_data: Optional[Dict] = None,
                analyst_data: Optional[Dict] = None,
                earnings_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perform comprehensive fundamental analysis
        
        Args:
            symbol: Stock symbol
            company_data: Company information
            financial_data: Financial metrics
            analyst_data: Analyst ratings and recommendations
            earnings_data: Earnings history and surprises
            
        Returns:
            Dictionary of fundamental scores and metrics
        """
        try:
            results = {
                'symbol': symbol,
                'fundamental_score': 50.0,
                'earnings_score': 50.0,
                'analyst_score': 50.0,
                'financial_score': 50.0,
                'valuation_score': 50.0
            }
            
            # Earnings analysis
            if earnings_data:
                earnings_results = self._analyze_earnings(earnings_data)
                results.update(earnings_results)
            
            # Analyst ratings analysis
            if analyst_data:
                analyst_results = self._analyze_analyst_ratings(analyst_data)
                results.update(analyst_results)
            
            # Financial health analysis
            if financial_data:
                financial_results = self._analyze_financial_health(financial_data)
                results.update(financial_results)
            
            # Valuation analysis
            if company_data and financial_data:
                valuation_results = self._analyze_valuation(company_data, financial_data)
                results.update(valuation_results)
            
            # Calculate composite fundamental score
            results['fundamental_score'] = self._calculate_composite_score(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Fundamental analysis failed for {symbol}: {e}")
            return self._empty_result(symbol)
    
    def _analyze_earnings(self, earnings_data: Dict) -> Dict[str, Any]:
        """Analyze earnings trends and surprises"""
        try:
            results = {
                'earnings_growth': 0.0,
                'earnings_surprise': 0.0,
                'earnings_consistency': 50.0,
                'earnings_score': 50.0
            }
            
            # Extract earnings metrics
            if 'quarterly_earnings' in earnings_data:
                quarterly = earnings_data['quarterly_earnings']
                if len(quarterly) >= 4:
                    # Calculate earnings growth
                    recent_eps = [q.get('eps', 0) for q in quarterly[:4]]
                    if len(recent_eps) >= 4 and recent_eps[3] != 0:
                        growth = ((recent_eps[0] - recent_eps[3]) / abs(recent_eps[3])) * 100
                        results['earnings_growth'] = growth
                    
                    # Calculate earnings surprise
                    if 'eps_surprise_percent' in quarterly[0]:
                        results['earnings_surprise'] = quarterly[0]['eps_surprise_percent']
                    
                    # Calculate earnings consistency (less volatility = higher score)
                    if len(recent_eps) >= 4:
                        volatility = np.std(recent_eps) / (np.mean(recent_eps) + 0.01)
                        consistency = max(0, 100 - (volatility * 100))
                        results['earnings_consistency'] = consistency
            
            # Score earnings performance
            score = 50.0
            
            # Growth scoring (40 points)
            growth = results['earnings_growth']
            if growth > 20:
                score += 20
            elif growth > 10:
                score += 15
            elif growth > 5:
                score += 10
            elif growth > 0:
                score += 5
            else:
                score -= 10
            
            # Surprise scoring (30 points)
            surprise = results['earnings_surprise']
            if surprise > 10:
                score += 15
            elif surprise > 5:
                score += 10
            elif surprise > 0:
                score += 5
            else:
                score -= 5
            
            # Consistency scoring (30 points)
            consistency = results['earnings_consistency']
            score += (consistency - 50) * 0.3
            
            results['earnings_score'] = max(0, min(100, score))
            
            return results
            
        except Exception as e:
            logger.error(f"Earnings analysis failed: {e}")
            return {'earnings_score': 50.0}
    
    def _analyze_analyst_ratings(self, analyst_data: Dict) -> Dict[str, Any]:
        """Analyze analyst ratings and recommendations"""
        try:
            results = {
                'analyst_rating': 'HOLD',
                'analyst_score_avg': 3.0,
                'analyst_count': 0,
                'price_target_upside': 0.0,
                'analyst_score': 50.0
            }
            
            # Extract analyst metrics
            if 'consensus_rating' in analyst_data:
                results['analyst_rating'] = analyst_data['consensus_rating']
            
            if 'rating_score' in analyst_data:
                results['analyst_score_avg'] = analyst_data['rating_score']
            
            if 'analyst_count' in analyst_data:
                results['analyst_count'] = analyst_data['analyst_count']
            
            if 'price_target' in analyst_data and 'current_price' in analyst_data:
                target = analyst_data['price_target']
                current = analyst_data['current_price']
                if current > 0:
                    upside = ((target - current) / current) * 100
                    results['price_target_upside'] = upside
            
            # Score analyst sentiment
            score = 50.0
            
            # Rating scoring (50 points)
            rating = results['analyst_rating'].upper()
            if rating in ['STRONG_BUY', 'BUY']:
                score += 25
            elif rating == 'OUTPERFORM':
                score += 15
            elif rating == 'HOLD':
                score += 0
            elif rating == 'UNDERPERFORM':
                score -= 15
            else:  # SELL, STRONG_SELL
                score -= 25
            
            # Price target upside scoring (30 points)
            upside = results['price_target_upside']
            if upside > 20:
                score += 15
            elif upside > 10:
                score += 10
            elif upside > 5:
                score += 5
            elif upside > 0:
                score += 2
            else:
                score -= 10
            
            # Analyst count bonus (20 points max)
            count = results['analyst_count']
            if count >= 10:
                score += 10
            elif count >= 5:
                score += 5
            elif count >= 3:
                score += 2
            
            results['analyst_score'] = max(0, min(100, score))
            
            return results
            
        except Exception as e:
            logger.error(f"Analyst ratings analysis failed: {e}")
            return {'analyst_score': 50.0}
    
    def _analyze_financial_health(self, financial_data: Dict) -> Dict[str, Any]:
        """Analyze financial health and stability"""
        try:
            results = {
                'debt_to_equity': 0.5,
                'current_ratio': 1.5,
                'roa': 5.0,
                'roe': 10.0,
                'profit_margin': 10.0,
                'financial_score': 50.0
            }
            
            # Extract financial metrics
            if 'debt_to_equity' in financial_data:
                results['debt_to_equity'] = financial_data['debt_to_equity']
            
            if 'current_ratio' in financial_data:
                results['current_ratio'] = financial_data['current_ratio']
            
            if 'return_on_assets' in financial_data:
                results['roa'] = financial_data['return_on_assets']
            
            if 'return_on_equity' in financial_data:
                results['roe'] = financial_data['return_on_equity']
            
            if 'profit_margin' in financial_data:
                results['profit_margin'] = financial_data['profit_margin']
            
            # Score financial health
            score = 50.0
            
            # Debt management (25 points)
            debt_ratio = results['debt_to_equity']
            if debt_ratio < 0.3:
                score += 12
            elif debt_ratio < 0.5:
                score += 8
            elif debt_ratio < 1.0:
                score += 4
            elif debt_ratio < 2.0:
                score += 0
            else:
                score -= 10
            
            # Liquidity (25 points)
            current_ratio = results['current_ratio']
            if current_ratio > 2.0:
                score += 12
            elif current_ratio > 1.5:
                score += 10
            elif current_ratio > 1.2:
                score += 6
            elif current_ratio > 1.0:
                score += 3
            else:
                score -= 10
            
            # Profitability (50 points)
            roe = results['roe']
            roa = results['roa']
            margin = results['profit_margin']
            
            # ROE scoring (20 points)
            if roe > 20:
                score += 10
            elif roe > 15:
                score += 8
            elif roe > 10:
                score += 6
            elif roe > 5:
                score += 3
            else:
                score -= 5
            
            # ROA scoring (15 points)
            if roa > 10:
                score += 8
            elif roa > 5:
                score += 6
            elif roa > 3:
                score += 4
            elif roa > 1:
                score += 2
            else:
                score -= 3
            
            # Profit margin scoring (15 points)
            if margin > 20:
                score += 8
            elif margin > 15:
                score += 6
            elif margin > 10:
                score += 4
            elif margin > 5:
                score += 2
            else:
                score -= 3
            
            results['financial_score'] = max(0, min(100, score))
            
            return results
            
        except Exception as e:
            logger.error(f"Financial health analysis failed: {e}")
            return {'financial_score': 50.0}
    
    def _analyze_valuation(self, company_data: Dict, financial_data: Dict) -> Dict[str, Any]:
        """Analyze valuation metrics"""
        try:
            results = {
                'pe_ratio': 20.0,
                'peg_ratio': 1.0,
                'price_to_book': 3.0,
                'price_to_sales': 5.0,
                'valuation_score': 50.0
            }
            
            # Extract valuation metrics
            if 'pe_ratio' in financial_data:
                results['pe_ratio'] = financial_data['pe_ratio']
            
            if 'peg_ratio' in financial_data:
                results['peg_ratio'] = financial_data['peg_ratio']
            
            if 'price_to_book' in financial_data:
                results['price_to_book'] = financial_data['price_to_book']
            
            if 'price_to_sales' in financial_data:
                results['price_to_sales'] = financial_data['price_to_sales']
            
            # Score valuation attractiveness
            score = 50.0
            
            # P/E ratio scoring (30 points)
            pe = results['pe_ratio']
            if 10 <= pe <= 20:  # Sweet spot
                score += 15
            elif 5 <= pe <= 25:
                score += 10
            elif pe <= 30:
                score += 5
            elif pe > 50:
                score -= 15
            
            # PEG ratio scoring (25 points)
            peg = results['peg_ratio']
            if peg < 1.0:
                score += 12
            elif peg < 1.5:
                score += 8
            elif peg < 2.0:
                score += 4
            else:
                score -= 8
            
            # Price-to-Book scoring (25 points)
            pb = results['price_to_book']
            if pb < 1.5:
                score += 12
            elif pb < 3.0:
                score += 8
            elif pb < 5.0:
                score += 4
            else:
                score -= 5
            
            # Price-to-Sales scoring (20 points)
            ps = results['price_to_sales']
            if ps < 2.0:
                score += 10
            elif ps < 5.0:
                score += 6
            elif ps < 10.0:
                score += 3
            else:
                score -= 5
            
            results['valuation_score'] = max(0, min(100, score))
            
            return results
            
        except Exception as e:
            logger.error(f"Valuation analysis failed: {e}")
            return {'valuation_score': 50.0}
    
    def _calculate_composite_score(self, results: Dict[str, Any]) -> float:
        """Calculate weighted composite fundamental score"""
        try:
            earnings_score = results.get('earnings_score', 50.0)
            analyst_score = results.get('analyst_score', 50.0)
            financial_score = results.get('financial_score', 50.0)
            valuation_score = results.get('valuation_score', 50.0)
            
            composite = (
                earnings_score * self.earnings_weight +
                analyst_score * self.analyst_weight +
                financial_score * self.financial_weight +
                valuation_score * self.valuation_weight
            )
            
            return max(0, min(100, composite))
            
        except Exception as e:
            logger.error(f"Composite score calculation failed: {e}")
            return 50.0
    
    def _empty_result(self, symbol: str) -> Dict[str, Any]:
        """Return empty result structure"""
        return {
            'symbol': symbol,
            'fundamental_score': 50.0,
            'earnings_score': 50.0,
            'analyst_score': 50.0,
            'financial_score': 50.0,
            'valuation_score': 50.0,
            'earnings_growth': 0.0,
            'earnings_surprise': 0.0,
            'analyst_rating': 'HOLD',
            'price_target_upside': 0.0,
            'debt_to_equity': 0.5,
            'current_ratio': 1.5,
            'pe_ratio': 20.0,
            'peg_ratio': 1.0
        }