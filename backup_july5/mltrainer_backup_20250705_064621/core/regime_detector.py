"""
mlTrainer - Regime Detector
===========================

Purpose: Analyzes market data to detect regime changes and classify current
market conditions. Uses multi-dimensional analysis including volatility,
macro indicators, and sentiment to generate regime scores from 0-100.

Features:
- Real-time regime classification
- Multi-source regime indicators
- Spectrum-based scoring (0-100)
- Regime transition detection
- Compliance-verified data only
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import json

logger = logging.getLogger(__name__)

class RegimeDetector:
    """Detects and classifies market regimes using multi-dimensional analysis"""
    
    def __init__(self):
        self.is_initialized = False
        self.regime_history = []
        self.current_regime = None
        self.regime_thresholds = {
            "low_volatility": 30,
            "medium_volatility": 70,
            "high_volatility": 100
        }
        
        # Regime classification weights
        self.feature_weights = {
            "volatility": 0.4,
            "macro_indicators": 0.3,
            "market_momentum": 0.2,
            "sentiment": 0.1
        }
        
        # Historical regime patterns
        self.regime_patterns = {}
        
        logger.info("RegimeDetector initialized")
        self._initialize_regime_patterns()
    
    def _initialize_regime_patterns(self):
        """Initialize regime classification patterns"""
        self.regime_patterns = {
            "stable": {
                "volatility_range": (0, 30),
                "momentum_range": (-0.02, 0.02),
                "macro_score_range": (0, 40),
                "description": "Low volatility, stable macro environment"
            },
            "trending": {
                "volatility_range": (20, 60),
                "momentum_range": (0.02, 0.1),
                "macro_score_range": (20, 60),
                "description": "Moderate volatility with clear directional movement"
            },
            "volatile": {
                "volatility_range": (50, 80),
                "momentum_range": (-0.05, 0.05),
                "macro_score_range": (40, 80),
                "description": "High volatility, uncertain direction"
            },
            "crisis": {
                "volatility_range": (70, 100),
                "momentum_range": (-0.2, 0.2),
                "macro_score_range": (60, 100),
                "description": "Crisis conditions, extreme volatility"
            }
        }
        
        self.is_initialized = True
    
    def is_initialized(self) -> bool:
        """Check if regime detector is properly initialized"""
        return self.is_initialized
    
    def analyze_current_regime(self, market_data: Dict) -> Dict[str, Any]:
        """Analyze current market regime from comprehensive market data"""
        if not market_data:
            return self._create_error_response("No market data provided")
        
        try:
            regime_analysis = {
                "timestamp": datetime.now().isoformat(),
                "regime_type": "unknown",
                "regime_score": 50.0,
                "confidence": 0.0,
                "components": {},
                "sources": market_data.get("sources", []),
                "verified": market_data.get("verified", False)
            }
            
            # Analyze volatility component
            volatility_score = self._analyze_volatility(market_data)
            regime_analysis["components"]["volatility"] = volatility_score
            
            # Analyze macro indicators
            macro_score = self._analyze_macro_indicators(market_data)
            regime_analysis["components"]["macro_indicators"] = macro_score
            
            # Analyze market momentum
            momentum_score = self._analyze_market_momentum(market_data)
            regime_analysis["components"]["market_momentum"] = momentum_score
            
            # Analyze sentiment (if available)
            sentiment_score = self._analyze_sentiment(market_data)
            regime_analysis["components"]["sentiment"] = sentiment_score
            
            # Calculate composite regime score
            regime_score = self._calculate_composite_score(
                volatility_score, macro_score, momentum_score, sentiment_score
            )
            
            # Classify regime type
            regime_type = self._classify_regime_type(regime_score, regime_analysis["components"])
            
            # Calculate confidence
            confidence = self._calculate_confidence(regime_analysis["components"])
            
            # Update analysis
            regime_analysis.update({
                "regime_score": round(regime_score, 1),
                "regime_type": regime_type,
                "confidence": round(confidence, 1),
                "volatility": self._get_volatility_level(volatility_score),
                "macro_signal": self._get_macro_signal(macro_score)
            })
            
            # Store in history
            self._update_regime_history(regime_analysis)
            
            # Check for regime change
            regime_change = self._detect_regime_change(regime_analysis)
            if regime_change:
                regime_analysis["regime_change_detected"] = True
                regime_analysis["previous_regime"] = self.current_regime
            
            self.current_regime = regime_analysis
            
            return regime_analysis
            
        except Exception as e:
            logger.error(f"Regime analysis failed: {e}")
            return self._create_error_response(f"Analysis failed: {e}")
    
    def _analyze_volatility(self, market_data: Dict) -> float:
        """Analyze volatility from market data"""
        try:
            volatility_score = 50.0  # Default
            
            # Check for VIX in macro indicators
            macro_indicators = market_data.get("macro_indicators", {})
            if "VIXCLS" in macro_indicators:
                vix_value = macro_indicators["VIXCLS"]["value"]
                # VIX interpretation: <15=low, 15-25=normal, 25-35=elevated, >35=high
                if vix_value < 15:
                    volatility_score = 20
                elif vix_value < 25:
                    volatility_score = 40
                elif vix_value < 35:
                    volatility_score = 70
                else:
                    volatility_score = 90
            
            # Analyze market data volatility
            market_data_dict = market_data.get("market_data", {})
            if "SPY" in market_data_dict:
                spy_data = market_data_dict["SPY"]["data"]
                if len(spy_data) > 10:
                    prices = [item["c"] for item in spy_data[:10]]
                    returns = np.diff(prices) / prices[:-1]
                    realized_vol = np.std(returns) * np.sqrt(252) * 100  # Annualized %
                    
                    # Scale realized volatility to 0-100
                    vol_score = min(100, max(0, realized_vol * 5))
                    volatility_score = (volatility_score + vol_score) / 2
            
            return min(100, max(0, volatility_score))
            
        except Exception as e:
            logger.error(f"Volatility analysis failed: {e}")
            return 50.0
    
    def _analyze_macro_indicators(self, market_data: Dict) -> float:
        """Analyze macroeconomic indicators"""
        try:
            macro_score = 50.0
            macro_indicators = market_data.get("macro_indicators", {})
            
            if not macro_indicators:
                return macro_score
            
            indicator_scores = []
            
            # Interest rate environment (10-year treasury)
            if "DGS10" in macro_indicators:
                rate = macro_indicators["DGS10"]["value"]
                # Higher rates generally indicate tighter conditions
                rate_score = min(100, max(0, rate * 20))
                indicator_scores.append(rate_score)
            
            # Unemployment rate
            if "UNRATE" in macro_indicators:
                unemployment = macro_indicators["UNRATE"]["value"]
                # Higher unemployment = worse conditions
                unemployment_score = min(100, max(0, unemployment * 15))
                indicator_scores.append(unemployment_score)
            
            # CPI (inflation pressure)
            if "CPIAUCSL" in macro_indicators:
                # This would need year-over-year calculation in production
                # For now, use a simplified approach
                indicator_scores.append(40)  # Neutral inflation assumption
            
            if indicator_scores:
                macro_score = np.mean(indicator_scores)
            
            return min(100, max(0, macro_score))
            
        except Exception as e:
            logger.error(f"Macro analysis failed: {e}")
            return 50.0
    
    def _analyze_market_momentum(self, market_data: Dict) -> float:
        """Analyze market momentum and trend strength"""
        try:
            momentum_score = 50.0
            market_data_dict = market_data.get("market_data", {})
            
            momentum_values = []
            
            for symbol in ["SPY", "QQQ", "VIX"]:
                if symbol in market_data_dict:
                    data = market_data_dict[symbol]["data"]
                    if len(data) > 5:
                        prices = [item["c"] for item in data[:5]]
                        # Calculate short-term momentum
                        momentum = (prices[0] - prices[-1]) / prices[-1]
                        
                        # Convert to 0-100 score (higher = more volatile/uncertain)
                        abs_momentum = abs(momentum)
                        momentum_score_item = min(100, abs_momentum * 500)  # Scale factor
                        momentum_values.append(momentum_score_item)
            
            if momentum_values:
                momentum_score = np.mean(momentum_values)
            
            return min(100, max(0, momentum_score))
            
        except Exception as e:
            logger.error(f"Momentum analysis failed: {e}")
            return 50.0
    
    def _analyze_sentiment(self, market_data: Dict) -> float:
        """Analyze sentiment indicators"""
        try:
            sentiment_score = 50.0  # Neutral default
            sentiment_data = market_data.get("sentiment", {})
            
            if not sentiment_data:
                return sentiment_score
            
            sentiment_scores = []
            
            # Analyze Reddit WSB sentiment
            if "reddit_wsb" in sentiment_data:
                wsb_data = sentiment_data["reddit_wsb"]
                # Only use verified sentiment data - no synthetic fallbacks
                if wsb_data and "sentiment_score" in wsb_data:
                    sentiment_scores.append(wsb_data["sentiment_score"])
            
            # Analyze insider trading sentiment
            insider_keys = [key for key in sentiment_data.keys() if key.startswith("insider_")]
            if insider_keys:
                # Insider selling typically indicates caution
                sentiment_scores.append(60)  # Elevated concern
            
            if sentiment_scores:
                sentiment_score = np.mean(sentiment_scores)
            
            return min(100, max(0, sentiment_score))
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return 50.0
    
    def _calculate_composite_score(self, volatility: float, macro: float, 
                                 momentum: float, sentiment: float) -> float:
        """Calculate weighted composite regime score"""
        weights = self.feature_weights
        
        composite = (
            volatility * weights["volatility"] +
            macro * weights["macro_indicators"] +
            momentum * weights["market_momentum"] +
            sentiment * weights["sentiment"]
        )
        
        return min(100, max(0, composite))
    
    def _classify_regime_type(self, regime_score: float, components: Dict) -> str:
        """Classify regime type based on score and components"""
        volatility_score = components.get("volatility", 50)
        momentum_score = components.get("market_momentum", 50)
        macro_score = components.get("macro_indicators", 50)
        
        # Classify based on patterns
        for regime_name, pattern in self.regime_patterns.items():
            vol_range = pattern["volatility_range"]
            macro_range = pattern["macro_score_range"]
            
            if (vol_range[0] <= volatility_score <= vol_range[1] and
                macro_range[0] <= macro_score <= macro_range[1]):
                return regime_name
        
        # Fallback classification based on regime score
        if regime_score < 30:
            return "stable"
        elif regime_score < 60:
            return "trending"
        elif regime_score < 80:
            return "volatile"
        else:
            return "crisis"
    
    def _calculate_confidence(self, components: Dict) -> float:
        """Calculate confidence in regime classification"""
        # Base confidence on data availability and consistency
        available_components = len([v for v in components.values() if v is not None])
        data_coverage = available_components / 4.0 * 100  # 4 components max
        
        # Check for consistency in components
        scores = [v for v in components.values() if v is not None]
        if len(scores) > 1:
            score_std = np.std(scores)
            consistency = max(0, 100 - score_std)  # Lower std = higher consistency
        else:
            consistency = 50
        
        # Combine data coverage and consistency
        confidence = (data_coverage * 0.6 + consistency * 0.4)
        
        return min(100, max(0, confidence))
    
    def _get_volatility_level(self, volatility_score: float) -> str:
        """Convert volatility score to descriptive level"""
        if volatility_score < 30:
            return "low"
        elif volatility_score < 70:
            return "medium"
        else:
            return "high"
    
    def _get_macro_signal(self, macro_score: float) -> str:
        """Convert macro score to signal description"""
        if macro_score < 25:
            return "neutral"
        elif macro_score < 50:
            return "trending"
        elif macro_score < 75:
            return "macro_shift"
        else:
            return "shock"
    
    def _update_regime_history(self, regime_analysis: Dict):
        """Update regime history for trend analysis"""
        self.regime_history.append({
            "timestamp": regime_analysis["timestamp"],
            "regime_type": regime_analysis["regime_type"],
            "regime_score": regime_analysis["regime_score"],
            "confidence": regime_analysis["confidence"]
        })
        
        # Keep only last 100 regime analyses
        if len(self.regime_history) > 100:
            self.regime_history = self.regime_history[-100:]
    
    def _detect_regime_change(self, current_analysis: Dict) -> bool:
        """Detect if regime has changed significantly"""
        if not self.current_regime:
            return False
        
        # Check for regime type change
        if current_analysis["regime_type"] != self.current_regime["regime_type"]:
            return True
        
        # Check for significant score change
        score_change = abs(current_analysis["regime_score"] - self.current_regime["regime_score"])
        if score_change > 20:  # 20-point threshold
            return True
        
        return False
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "timestamp": datetime.now().isoformat(),
            "regime_type": "unknown",
            "regime_score": 50.0,
            "confidence": 0.0,
            "error": error_message,
            "verified": False
        }
    
    def get_regime_history(self, hours: int = 24) -> List[Dict]:
        """Get regime history for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_history = []
        for entry in self.regime_history:
            try:
                entry_time = datetime.fromisoformat(entry["timestamp"])
                if entry_time > cutoff_time:
                    recent_history.append(entry)
            except:
                continue
        
        return recent_history
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get comprehensive regime detection statistics"""
        if not self.regime_history:
            return {"error": "No regime history available"}
        
        recent_regimes = [entry["regime_type"] for entry in self.regime_history[-24:]]
        regime_counts = {}
        for regime in recent_regimes:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        recent_scores = [entry["regime_score"] for entry in self.regime_history[-24:]]
        
        return {
            "current_regime": self.current_regime["regime_type"] if self.current_regime else "unknown",
            "current_score": self.current_regime["regime_score"] if self.current_regime else 50,
            "regime_distribution": regime_counts,
            "score_statistics": {
                "mean": round(np.mean(recent_scores), 1) if recent_scores else 50,
                "std": round(np.std(recent_scores), 1) if recent_scores else 0,
                "min": round(min(recent_scores), 1) if recent_scores else 50,
                "max": round(max(recent_scores), 1) if recent_scores else 50
            },
            "history_length": len(self.regime_history),
            "last_update": self.current_regime["timestamp"] if self.current_regime else None
        }

