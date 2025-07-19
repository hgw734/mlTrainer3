"""
Enhanced market intelligence system for institutional-grade analysis.
Integrates regime detection, cross-asset momentum, earnings calendar, 
volume profile analysis, and options flow indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from polygon import RESTClient
import os

logger = logging.getLogger(__name__)

class MarketIntelligenceEngine:
    """
    Advanced market intelligence combining multiple data sources
    for enhanced signal accuracy and risk management.
    """
    
    def __init__(self):
        """Initialize market intelligence engine"""
        self.polygon_client = None
        self.market_data_cache = {}
        self.last_update = {}
        
        # Initialize Polygon client
        api_key = os.environ.get('POLYGON_API_KEY')
        if api_key:
            self.polygon_client = RESTClient(api_key)
        
        # Cross-asset symbols for regime detection
        self.market_etfs = {
            'SPY': 'S&P 500',
            'QQQ': 'NASDAQ 100', 
            'IWM': 'Russell 2000',
            'VIX': 'Volatility Index'
        }
        
        # Sector ETFs for rotation analysis
        self.sector_etfs = {
            'XLK': 'Technology',
            'XLF': 'Financials',
            'XLE': 'Energy',
            'XLV': 'Healthcare',
            'XLI': 'Industrials',
            'XLC': 'Communication',
            'XLY': 'Consumer Discretionary',
            'XLP': 'Consumer Staples',
            'XLRE': 'Real Estate',
            'XLU': 'Utilities',
            'XLB': 'Materials'
        }
    
    def get_market_regime(self) -> Dict[str, Any]:
        """
        Determine current market regime using multiple indicators
        
        Returns:
            Market regime analysis with confidence scores
        """
        try:
            if not self.polygon_client:
                logger.warning("Polygon client not available for regime detection")
                return self._get_fallback_regime()
            
            # Get market data for regime analysis
            market_data = self._fetch_market_indicators()
            
            # Analyze regime components
            volatility_regime = self._analyze_volatility(market_data)
            trend_regime = self._analyze_trend(market_data)
            volume_regime = self._analyze_volume_profile(market_data)
            
            # Combine indicators for final regime
            regime = self._classify_market_regime(volatility_regime, trend_regime, volume_regime)
            
            return {
                'regime': regime['classification'],
                'confidence': regime['confidence'],
                'volatility': volatility_regime,
                'trend': trend_regime,
                'volume': volume_regime,
                'timestamp': datetime.now(),
                'components': market_data
            }
            
        except Exception as e:
            logger.error(f"Market regime analysis failed: {e}")
            return self._get_fallback_regime()
    
    def get_cross_asset_momentum(self, symbol: str) -> Dict[str, float]:
        """
        Analyze momentum relative to market and sector ETFs
        
        Args:
            symbol: Stock symbol to analyze
            
        Returns:
            Cross-asset momentum scores
        """
        try:
            if not self.polygon_client:
                return {}
            
            # Get symbol's sector
            sector_etf = self._get_symbol_sector_etf(symbol)
            
            # Calculate momentum vs market and sector
            market_momentum = self._calculate_relative_momentum(symbol, 'SPY')
            sector_momentum = self._calculate_relative_momentum(symbol, sector_etf)
            tech_momentum = self._calculate_relative_momentum(symbol, 'QQQ')
            
            return {
                'vs_market': market_momentum,
                'vs_sector': sector_momentum,
                'vs_tech': tech_momentum,
                'momentum_score': (market_momentum + sector_momentum + tech_momentum) / 3
            }
            
        except Exception as e:
            logger.error(f"Cross-asset momentum analysis failed for {symbol}: {e}")
            return {}
    
    def check_earnings_calendar(self, symbol: str, days_ahead: int = 5) -> Dict[str, Any]:
        """
        Check if earnings announcement is upcoming for risk adjustment
        
        Args:
            symbol: Stock symbol
            days_ahead: Number of days to look ahead
            
        Returns:
            Earnings calendar information
        """
        try:
            if not self.polygon_client:
                return {'has_earnings': False}
            
            # Get company financials to check earnings date
            # Note: This would require the financials endpoint
            # For now, return structure for implementation
            
            return {
                'has_earnings': False,  # Would be determined from API
                'days_until_earnings': None,
                'risk_adjustment': 1.0,
                'recommended_action': 'proceed'
            }
            
        except Exception as e:
            logger.error(f"Earnings calendar check failed for {symbol}: {e}")
            return {'has_earnings': False}
    
    def analyze_volume_profile(self, symbol: str, bars_data: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze volume profile for institutional activity detection
        
        Args:
            symbol: Stock symbol
            bars_data: OHLCV data
            
        Returns:
            Volume profile analysis
        """
        try:
            if bars_data.empty:
                return {}
            
            # Calculate volume metrics
            current_volume = bars_data['volume'].iloc[-1]
            avg_volume_20d = bars_data['volume'].rolling(20).mean().iloc[-1]
            volume_ratio = current_volume / avg_volume_20d if avg_volume_20d > 0 else 1.0
            
            # Volume trend analysis
            volume_trend = self._calculate_volume_trend(bars_data)
            
            # Price-volume correlation
            price_volume_corr = self._calculate_price_volume_correlation(bars_data)
            
            # Institutional volume detection
            institutional_score = self._detect_institutional_volume(bars_data)
            
            return {
                'volume_ratio': volume_ratio,
                'volume_trend': volume_trend,
                'price_volume_correlation': price_volume_corr,
                'institutional_score': institutional_score,
                'volume_quality': min(100, max(0, institutional_score * 100))
            }
            
        except Exception as e:
            logger.error(f"Volume profile analysis failed for {symbol}: {e}")
            return {}
    
    def get_options_flow_signals(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze options flow for institutional positioning signals
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Options flow analysis
        """
        try:
            # Note: This would require options data from Polygon premium
            # For now, return structure for future implementation
            
            return {
                'unusual_activity': False,
                'call_put_ratio': 1.0,
                'volume_oi_ratio': 1.0,
                'institutional_flow': 'neutral',
                'options_sentiment': 0.0
            }
            
        except Exception as e:
            logger.error(f"Options flow analysis failed for {symbol}: {e}")
            return {}
    
    def _fetch_market_indicators(self) -> Dict[str, Any]:
        """Fetch key market indicators for regime analysis"""
        indicators = {}
        
        try:
            # Get SPY data for market trend
            spy_bars = list(self.polygon_client.get_aggs(
                ticker="SPY",
                multiplier=1,
                timespan="day",
                from_=datetime.now() - timedelta(days=30),
                to=datetime.now()
            ))
            
            if spy_bars:
                spy_data = pd.DataFrame([{
                    'close': bar.close,
                    'volume': bar.volume,
                    'date': pd.to_datetime(bar.timestamp, unit='ms')
                } for bar in spy_bars])
                
                # Calculate trend metrics
                indicators['spy_price'] = spy_data['close'].iloc[-1]
                indicators['spy_20d_change'] = ((spy_data['close'].iloc[-1] / spy_data['close'].iloc[-20]) - 1) * 100
                indicators['spy_volume_ratio'] = spy_data['volume'].iloc[-1] / spy_data['volume'].rolling(20).mean().iloc[-1]
                
                # Simple trend calculation
                indicators['spy_trend'] = np.polyfit(range(len(spy_data)), spy_data['close'], 1)[0]
            
            # Add VIX level (would need separate API call)
            indicators['vix'] = 20  # Placeholder - would fetch from data source
            
        except Exception as e:
            logger.error(f"Failed to fetch market indicators: {e}")
            indicators = {'spy_trend': 0, 'vix': 20, 'spy_20d_change': 0, 'spy_volume_ratio': 1.0}
        
        return indicators
    
    def _analyze_volatility(self, market_data: Dict) -> str:
        """Analyze volatility regime"""
        vix_level = market_data.get('vix', 20)
        
        if vix_level > 30:
            return 'high_volatility'
        elif vix_level > 20:
            return 'moderate_volatility'
        else:
            return 'low_volatility'
    
    def _analyze_trend(self, market_data: Dict) -> str:
        """Analyze trend regime"""
        spy_trend = market_data.get('spy_trend', 0)
        spy_20d_change = market_data.get('spy_20d_change', 0)
        
        if spy_trend > 0.5 and spy_20d_change > 3:
            return 'strong_uptrend'
        elif spy_trend > 0 and spy_20d_change > 0:
            return 'uptrend'
        elif spy_trend < -0.5 or spy_20d_change < -3:
            return 'downtrend'
        else:
            return 'sideways'
    
    def _analyze_volume_profile(self, market_data: Dict) -> str:
        """Analyze volume regime"""
        volume_ratio = market_data.get('spy_volume_ratio', 1.0)
        
        if volume_ratio > 1.5:
            return 'high_volume'
        elif volume_ratio > 1.2:
            return 'elevated_volume'
        else:
            return 'normal_volume'
    
    def _classify_market_regime(self, volatility: str, trend: str, volume: str) -> Dict[str, Any]:
        """Classify overall market regime"""
        
        # Define regime combinations
        if trend == 'strong_uptrend' and volatility == 'low_volatility' and volume in ['elevated_volume', 'high_volume']:
            return {'classification': 'bull_market', 'confidence': 0.9}
        elif trend == 'downtrend' and volatility == 'high_volatility':
            return {'classification': 'bear_market', 'confidence': 0.8}
        elif volatility == 'high_volatility':
            return {'classification': 'volatile_market', 'confidence': 0.85}
        elif trend in ['uptrend', 'strong_uptrend']:
            return {'classification': 'bullish_neutral', 'confidence': 0.7}
        elif trend == 'downtrend':
            return {'classification': 'bearish_neutral', 'confidence': 0.7}
        else:
            return {'classification': 'neutral_market', 'confidence': 0.6}
    
    def _get_fallback_regime(self) -> Dict[str, Any]:
        """Fallback regime when data is unavailable"""
        return {
            'regime': 'neutral_market',
            'confidence': 0.5,
            'volatility': 'moderate_volatility',
            'trend': 'sideways',
            'volume': 'normal_volume',
            'timestamp': datetime.now(),
            'components': {}
        }
    
    def _get_symbol_sector_etf(self, symbol: str) -> str:
        """Map symbol to appropriate sector ETF"""
        # Simplified mapping - would use sector classification API
        tech_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
        if symbol in tech_symbols:
            return 'XLK'
        return 'SPY'  # Default to market
    
    def _calculate_relative_momentum(self, symbol: str, benchmark: str) -> float:
        """Calculate momentum relative to benchmark"""
        try:
            # Would implement relative momentum calculation
            # For now return placeholder
            return 0.0
        except:
            return 0.0
    
    def _calculate_volume_trend(self, bars_data: pd.DataFrame) -> float:
        """Calculate volume trend over recent periods"""
        if len(bars_data) < 10:
            return 0.0
        
        volumes = bars_data['volume'].tail(10)
        trend = np.polyfit(range(len(volumes)), volumes, 1)[0]
        return trend / volumes.mean()  # Normalized trend
    
    def _calculate_price_volume_correlation(self, bars_data: pd.DataFrame) -> float:
        """Calculate price-volume correlation"""
        if len(bars_data) < 20:
            return 0.0
        
        price_changes = bars_data['close'].pct_change()
        volume_changes = bars_data['volume'].pct_change()
        
        correlation = price_changes.corr(volume_changes)
        return correlation if not pd.isna(correlation) else 0.0
    
    def _detect_institutional_volume(self, bars_data: pd.DataFrame) -> float:
        """Detect institutional volume patterns"""
        if len(bars_data) < 20:
            return 0.5
        
        # Volume spikes with price confirmation
        volume_ratio = bars_data['volume'].iloc[-1] / bars_data['volume'].rolling(20).mean().iloc[-1]
        price_momentum = (bars_data['close'].iloc[-1] / bars_data['close'].iloc[-5] - 1) * 100
        
        # Institutional activity often shows as volume spikes with sustained price moves
        if volume_ratio > 2.0 and abs(price_momentum) > 2:
            return min(1.0, volume_ratio / 3.0)
        elif volume_ratio > 1.5:
            return 0.7
        else:
            return 0.5