"""
Money Flow Analysis Engine inspired by MarketStructureEdge approach.
Focuses on Supply/Demand dynamics to improve signal accuracy from 47% to 70%+.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MoneyFlowAnalyzer:
    """
    Advanced money flow analysis focusing on supply/demand imbalances
    to identify high-probability momentum opportunities.
    """
    
    def __init__(self):
        """Initialize money flow analyzer with MarketStructureEdge methodology"""
        
        # Supply/Demand thresholds
        self.flow_thresholds = {
            'strong_demand': 0.7,      # 70%+ buying pressure
            'weak_supply': 0.3,        # 30%- selling pressure
            'volume_surge': 2.0,       # 2x average volume
            'accumulation': 0.6,       # 60%+ money flow positive
            'distribution': -0.6       # 60%+ money flow negative
        }
        
        # Sector rotation tracking
        self.sector_flow_weights = {
            'technology': 0.25,
            'healthcare': 0.15,
            'financials': 0.15,
            'consumer_discretionary': 0.12,
            'industrials': 0.10,
            'energy': 0.08,
            'materials': 0.08,
            'utilities': 0.04,
            'real_estate': 0.03
        }
        
    def analyze_money_flow(self, price_data: pd.DataFrame, volume_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Comprehensive money flow analysis following MarketStructureEdge principles
        
        Args:
            price_data: OHLCV data
            volume_data: Additional volume breakdown if available
            
        Returns:
            Money flow analysis results
        """
        
        try:
            if price_data.empty:
                return self._get_empty_analysis()
            
            # Core MarketStructureEdge metrics
            demand_pressure = self._calculate_demand_pressure(price_data)
            supply_pressure = self._calculate_supply_pressure(price_data)
            money_flow_index = self._calculate_money_flow_index(price_data)
            accumulation_distribution = self._calculate_accumulation_distribution(price_data)
            
            # Volume analysis
            volume_profile = self._analyze_volume_profile(price_data)
            institutional_flow = self._detect_institutional_flow(price_data)
            
            # Supply/Demand imbalance detection
            imbalance_score = self._calculate_imbalance_score(demand_pressure, supply_pressure)
            flow_strength = self._assess_flow_strength(money_flow_index, volume_profile)
            
            # Generate money flow signals
            flow_signals = self._generate_flow_signals(
                demand_pressure, supply_pressure, imbalance_score, flow_strength
            )
            
            return {
                'demand_pressure': demand_pressure,
                'supply_pressure': supply_pressure,
                'money_flow_index': money_flow_index,
                'accumulation_distribution': accumulation_distribution,
                'volume_profile': volume_profile,
                'institutional_flow': institutional_flow,
                'imbalance_score': imbalance_score,
                'flow_strength': flow_strength,
                'flow_signals': flow_signals,
                'flow_grade': self._assign_flow_grade(imbalance_score, flow_strength),
                'confidence': min(100, max(0, (abs(imbalance_score) * 100)))
            }
            
        except Exception as e:
            logger.error(f"Money flow analysis error: {e}")
            return self._get_empty_analysis()
    
    def _calculate_demand_pressure(self, data: pd.DataFrame) -> float:
        """Calculate demand pressure (buying interest) following MSE methodology"""
        
        try:
            # Price-volume relationship for demand
            closes = data['close'].values
            volumes = data['volume'].values
            highs = data['high'].values
            lows = data['low'].values
            
            # Demand indicators
            up_closes = np.sum(np.diff(closes) > 0)
            total_periods = len(closes) - 1
            
            # Volume-weighted demand
            price_ranges = highs - lows
            close_positions = (closes - lows) / np.where(price_ranges > 0, price_ranges, 1)
            
            # Weight by volume and recent periods
            weights = np.linspace(0.5, 1.0, len(close_positions))
            weighted_demand = np.average(close_positions, weights=volumes * weights)
            
            # Normalize to 0-1 scale
            return max(0, min(1, weighted_demand))
            
        except Exception as e:
            logger.warning(f"Demand pressure calculation error: {e}")
            return 0.5
    
    def _calculate_supply_pressure(self, data: pd.DataFrame) -> float:
        """Calculate supply pressure (selling pressure) following MSE methodology"""
        
        try:
            closes = data['close'].values
            volumes = data['volume'].values
            highs = data['high'].values
            lows = data['low'].values
            
            # Supply indicators (inverse of demand)
            down_closes = np.sum(np.diff(closes) < 0)
            total_periods = len(closes) - 1
            
            # Volume-weighted supply pressure
            price_ranges = highs - lows
            close_positions = (highs - closes) / np.where(price_ranges > 0, price_ranges, 1)
            
            # Weight by volume and recent periods
            weights = np.linspace(0.5, 1.0, len(close_positions))
            weighted_supply = np.average(close_positions, weights=volumes * weights)
            
            # Normalize to 0-1 scale
            return max(0, min(1, weighted_supply))
            
        except Exception as e:
            logger.warning(f"Supply pressure calculation error: {e}")
            return 0.5
    
    def _calculate_money_flow_index(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Money Flow Index (MFI) for institutional flow detection"""
        
        try:
            typical_prices = (data['high'] + data['low'] + data['close']) / 3
            money_flow = typical_prices * data['volume']
            
            # Positive and negative money flow
            positive_flow = money_flow.where(typical_prices.diff() > 0, 0).rolling(period).sum()
            negative_flow = money_flow.where(typical_prices.diff() < 0, 0).rolling(period).sum()
            
            # Money Flow Index
            money_ratio = positive_flow / (negative_flow + 1e-10)  # Avoid division by zero
            mfi = 100 - (100 / (1 + money_ratio))
            
            return float(mfi.iloc[-1]) if not mfi.empty else 50.0
            
        except Exception as e:
            logger.warning(f"MFI calculation error: {e}")
            return 50.0
    
    def _calculate_accumulation_distribution(self, data: pd.DataFrame) -> float:
        """Calculate Accumulation/Distribution indicator"""
        
        try:
            # A/D Line calculation
            close_location_value = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
            close_location_value = close_location_value.fillna(0)
            
            ad_line = (close_location_value * data['volume']).cumsum()
            
            # Return trend of A/D line (positive = accumulation, negative = distribution)
            if len(ad_line) >= 2:
                recent_trend = (ad_line.iloc[-1] - ad_line.iloc[-5]) / 5 if len(ad_line) >= 5 else ad_line.iloc[-1] - ad_line.iloc[-2]
                return float(np.tanh(recent_trend / data['volume'].mean()))  # Normalize
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"A/D calculation error: {e}")
            return 0.0
    
    def _analyze_volume_profile(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze volume profile for institutional activity detection"""
        
        try:
            volumes = data['volume'].values
            avg_volume = np.mean(volumes)
            recent_volume = np.mean(volumes[-5:]) if len(volumes) >= 5 else volumes[-1]
            
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            volume_trend = np.polyfit(range(len(volumes)), volumes, 1)[0] if len(volumes) > 1 else 0
            
            return {
                'volume_ratio': volume_ratio,
                'volume_trend': float(volume_trend),
                'avg_volume': float(avg_volume),
                'recent_volume': float(recent_volume),
                'volume_surge': volume_ratio > self.flow_thresholds['volume_surge']
            }
            
        except Exception as e:
            logger.warning(f"Volume profile analysis error: {e}")
            return {'volume_ratio': 1.0, 'volume_trend': 0.0, 'avg_volume': 0.0, 'recent_volume': 0.0, 'volume_surge': False}
    
    def _detect_institutional_flow(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect institutional money flow patterns"""
        
        try:
            # Large block detection (proxy for institutional activity)
            volumes = data['volume'].values
            volume_threshold = np.percentile(volumes, 90)  # Top 10% volume days
            
            institutional_days = np.sum(volumes > volume_threshold)
            total_days = len(volumes)
            
            # Price action on high volume days
            high_vol_mask = volumes > volume_threshold
            if np.any(high_vol_mask):
                closes = data['close'].values
                high_vol_returns = np.diff(closes[high_vol_mask])
                institutional_bias = np.mean(high_vol_returns) if len(high_vol_returns) > 0 else 0
            else:
                institutional_bias = 0
            
            return {
                'institutional_participation': institutional_days / total_days,
                'institutional_bias': float(institutional_bias),
                'large_volume_frequency': float(institutional_days),
                'institutional_signal': institutional_bias > 0 and institutional_days >= 2
            }
            
        except Exception as e:
            logger.warning(f"Institutional flow detection error: {e}")
            return {'institutional_participation': 0.0, 'institutional_bias': 0.0, 'large_volume_frequency': 0.0, 'institutional_signal': False}
    
    def _calculate_imbalance_score(self, demand: float, supply: float) -> float:
        """Calculate supply/demand imbalance score (-1 to +1)"""
        
        # MarketStructureEdge methodology: Buy rising demand + falling supply
        imbalance = demand - supply
        
        # Apply sigmoid function for smoother scaling
        return np.tanh(imbalance * 2)  # Scale and bound between -1 and 1
    
    def _assess_flow_strength(self, mfi: float, volume_profile: Dict) -> float:
        """Assess overall money flow strength (0-100)"""
        
        try:
            # Combine MFI with volume analysis
            mfi_component = (mfi - 50) / 50  # Convert MFI to -1 to +1 scale
            volume_component = min(1.0, volume_profile['volume_ratio'] - 1.0)  # Volume surge component
            
            # Weight components
            flow_strength = (mfi_component * 0.7) + (volume_component * 0.3)
            
            # Convert to 0-100 scale
            return max(0, min(100, (flow_strength + 1) * 50))
            
        except Exception as e:
            logger.warning(f"Flow strength assessment error: {e}")
            return 50.0
    
    def _generate_flow_signals(self, demand: float, supply: float, imbalance: float, strength: float) -> List[Dict[str, Any]]:
        """Generate actionable money flow signals"""
        
        signals = []
        
        # Strong demand + weak supply (MSE buy signal)
        if demand > self.flow_thresholds['strong_demand'] and supply < self.flow_thresholds['weak_supply']:
            signals.append({
                'signal': 'STRONG_BUY',
                'reason': 'Rising demand + falling supply',
                'confidence': min(100, strength * 1.2),
                'priority': 'HIGH'
            })
        
        # Moderate imbalance signals
        elif imbalance > 0.3 and strength > 60:
            signals.append({
                'signal': 'BUY',
                'reason': 'Positive demand/supply imbalance',
                'confidence': strength,
                'priority': 'MEDIUM'
            })
        
        elif imbalance < -0.3 and strength > 60:
            signals.append({
                'signal': 'SELL',
                'reason': 'Negative demand/supply imbalance',
                'confidence': strength,
                'priority': 'MEDIUM'
            })
        
        # Neutral signals
        else:
            signals.append({
                'signal': 'NEUTRAL',
                'reason': 'Balanced supply/demand',
                'confidence': 100 - strength,
                'priority': 'LOW'
            })
        
        return signals
    
    def _assign_flow_grade(self, imbalance_score: float, flow_strength: float) -> str:
        """Assign grade based on money flow analysis"""
        
        combined_score = (abs(imbalance_score) * 50) + (flow_strength * 0.5)
        
        if combined_score >= 80:
            return 'A+'
        elif combined_score >= 70:
            return 'A'
        elif combined_score >= 60:
            return 'B+'
        elif combined_score >= 50:
            return 'B'
        elif combined_score >= 40:
            return 'C'
        else:
            return 'D'
    
    def _get_empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis structure"""
        
        return {
            'demand_pressure': 0.5,
            'supply_pressure': 0.5,
            'money_flow_index': 50.0,
            'accumulation_distribution': 0.0,
            'volume_profile': {'volume_ratio': 1.0, 'volume_trend': 0.0, 'avg_volume': 0.0, 'recent_volume': 0.0, 'volume_surge': False},
            'institutional_flow': {'institutional_participation': 0.0, 'institutional_bias': 0.0, 'large_volume_frequency': 0.0, 'institutional_signal': False},
            'imbalance_score': 0.0,
            'flow_strength': 50.0,
            'flow_signals': [{'signal': 'NEUTRAL', 'reason': 'Insufficient data', 'confidence': 0, 'priority': 'LOW'}],
            'flow_grade': 'D',
            'confidence': 0
        }
    
    def get_sector_money_flow_analysis(self) -> Dict[str, Any]:
        """
        Framework for sector-wide money flow analysis
        (Would require sector ETF data integration)
        """
        
        return {
            'description': 'Sector money flow tracking ready for ETF data integration',
            'methodology': 'Track where institutional money is flowing sector by sector',
            'benefits': 'Identify sector rotation patterns and concentration of institutional capital',
            'implementation_status': 'Framework ready - requires sector ETF data feeds'
        }