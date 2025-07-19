"""
Recommendation Tracker
=====================

Generates trading recommendations from ML models and tracks their performance.
Integrates with the virtual portfolio manager for paper trading.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
import logging
from dataclasses import dataclass, asdict
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import model managers
from mltrainer_models import get_ml_model_manager
from mltrainer_financial_models import get_financial_model_manager

# Import virtual portfolio
from virtual_portfolio_manager import get_virtual_portfolio_manager, VirtualPosition

# Import data connectors
from polygon_connector import PolygonConnector
from fred_connector import FREDConnector

logger = logging.getLogger(__name__)


@dataclass
class TradingRecommendation:
    """Represents a trading recommendation"""
    timestamp: datetime
    symbol: str
    signal_strength: float  # 0-1
    profit_probability: float  # 0-1
    confidence: float  # 0-1
    entry_price: float
    target_price: float
    stop_loss: float
    timeframe: str  # '7-12 days' or '50-70 days'
    model_used: str
    features_json: str  # JSON string of key features
    market_conditions: str  # JSON string of market state

    def to_dict(self) -> Dict:
        """Convert to dictionary for database storage"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @property
    def risk_reward_ratio(self) -> float:
        """Calculate risk/reward ratio"""
        potential_gain = self.target_price - self.entry_price
        potential_loss = self.entry_price - self.stop_loss
        if potential_loss > 0:
            return potential_gain / potential_loss
        return 0.0

    @property
    def expected_return(self) -> float:
        """Calculate expected return based on probability"""
        gain_pct = ((self.target_price - self.entry_price) /
                    self.entry_price) * 100
        loss_pct = ((self.stop_loss - self.entry_price) /
                    self.entry_price) * 100
        return (gain_pct * self.profit_probability) + \
            (loss_pct * (1 - self.profit_probability))


class RecommendationTracker:
    """Tracks and manages trading recommendations"""

    def __init__(self):
        self.ml_manager = get_ml_model_manager()
        self.financial_manager = get_financial_model_manager()
        self.portfolio_manager = get_virtual_portfolio_manager()
        self.polygon_client = PolygonConnector()
        self.fred_client = FREDConnector()

        # Configuration
        self.min_signal_strength = 0.7
        self.min_confidence = 0.6
        self.min_profit_probability = 0.55
        self.min_risk_reward = 2.0

        # Tracking
        self.active_recommendations: List[TradingRecommendation] = []
        self.recommendation_history: List[TradingRecommendation] = []

        # Models to use for signal generation
        self.momentum_models = [
            'momentum_breakout_model',
            'rsi_model',
            'macd_model',
            'ema_crossover_model',
            'xgboost',
            'lightgbm',
            'random_forest'
        ]

        logger.info("Recommendation Tracker initialized")

    async def scan_for_opportunities(
            self, symbols: List[str]) -> List[TradingRecommendation]:
        """Scan symbols for trading opportunities"""
        recommendations = []

        # Get market conditions
        market_conditions = await self._get_market_conditions()

        # Process symbols in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for symbol in symbols:
                future = executor.submit(
                    self._analyze_symbol, symbol, market_conditions)
                futures.append((symbol, future))

            for symbol, future in futures:
                try:
                    result = future.result(timeout=30)
                    if result:
                        recommendations.extend(result)
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")

        # Filter and rank recommendations
        filtered = self._filter_recommendations(recommendations)
        ranked = self._rank_recommendations(filtered)

        # Store recommendations
        self.active_recommendations = ranked
        self.recommendation_history.extend(ranked)

        # Automatically paper trade top recommendations
        self._auto_paper_trade(ranked[:5])  # Top 5 recommendations

        return ranked

    def _analyze_symbol(
            self,
            symbol: str,
            market_conditions: Dict) -> List[TradingRecommendation]:
        """Analyze a single symbol for trading opportunities"""
        recommendations = []

        try:
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)

            historical_data = self.polygon_client.get_historical_data(
                symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

            if historical_data is None or len(historical_data) < 100:
                return recommendations

            # Convert to pandas Series
            price_series = pd.Series(
                historical_data['close'].values,
                index=pd.to_datetime(historical_data['timestamp'])
            )

            # Get current quote
            quote = self.polygon_client.get_quote(symbol)
            if not quote:
                return recommendations

            current_price = quote.price

            # Run models
            for model_name in self.momentum_models:
                try:
                    # Get model prediction
                    if model_name in ['xgboost', 'lightgbm', 'random_forest']:
                        # ML models
                        result = self.ml_manager.run_model(
                            model_name,
                            data=price_series,
                            symbol=symbol
                        )
                    else:
                        # Technical models
                        result = self.financial_manager.run_model(
                            model_name,
                            data=price_series,
                            symbol=symbol
                        )

                    if result and hasattr(result, 'signal_strength'):
                        # Check for 7-12 day momentum
                        recommendation_7_12 = self._create_recommendation(
                            symbol, current_price, result, model_name,
                            "7-12 days", market_conditions
                        )
                        if recommendation_7_12:
                            recommendations.append(recommendation_7_12)

                        # Check for 50-70 day momentum
                        recommendation_50_70 = self._create_recommendation(
                            symbol, current_price, result, model_name,
                            "50-70 days", market_conditions
                        )
                        if recommendation_50_70:
                            recommendations.append(recommendation_50_70)

                except Exception as e:
                    logger.error(
                        f"Error running {model_name} on {symbol}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")

        return recommendations

    def _create_recommendation(
            self,
            symbol: str,
            current_price: float,
            model_result: Any,
            model_name: str,
            timeframe: str,
            market_conditions: Dict) -> Optional[TradingRecommendation]:
        """Create a recommendation from model results"""
        try:
            # Extract signals
            signal_strength = getattr(model_result, 'signal_strength', 0)
            confidence = getattr(model_result, 'confidence', 0)

            # Skip if signals too weak
            if signal_strength < self.min_signal_strength or confidence < self.min_confidence:
                return None

            # Calculate targets based on timeframe
            if timeframe == "7-12 days":
                target_pct = 0.03  # 3% target
                stop_pct = 0.015   # 1.5% stop
            else:  # 50-70 days
                target_pct = 0.08  # 8% target
                stop_pct = 0.04    # 4% stop

            target_price = current_price * (1 + target_pct)
            stop_loss = current_price * (1 - stop_pct)

            # Calculate profit probability (simplified)
            # In reality, this would come from the model
            profit_probability = 0.5 + \
                (signal_strength * 0.3) + (confidence * 0.2)
            profit_probability = min(profit_probability, 0.85)  # Cap at 85%

            # Extract features
            features = {
                'signal_strength': signal_strength,
                'confidence': confidence,
                'model': model_name,
                'current_price': current_price
            }

            recommendation = TradingRecommendation(
                timestamp=datetime.now(),
                symbol=symbol,
                signal_strength=signal_strength,
                profit_probability=profit_probability,
                confidence=confidence,
                entry_price=current_price,
                target_price=target_price,
                stop_loss=stop_loss,
                timeframe=timeframe,
                model_used=model_name,
                features_json=json.dumps(features),
                market_conditions=json.dumps(market_conditions)
            )

            # Check risk/reward
            if recommendation.risk_reward_ratio < self.min_risk_reward:
                return None

            # Check expected return
            if recommendation.expected_return < 1.0:  # At least 1% expected return
                return None

            return recommendation

        except Exception as e:
            logger.error(f"Error creating recommendation: {e}")
            return None

    def _filter_recommendations(
            self,
            recommendations: List[TradingRecommendation]) -> List[TradingRecommendation]:
        """Filter recommendations based on criteria"""
        filtered = []

        for rec in recommendations:
            # Apply filters
            if rec.signal_strength >= self.min_signal_strength and \
               rec.confidence >= self.min_confidence and \
               rec.profit_probability >= self.min_profit_probability and \
               rec.risk_reward_ratio >= self.min_risk_reward:
                filtered.append(rec)

        return filtered

    def _rank_recommendations(
            self,
            recommendations: List[TradingRecommendation]) -> List[TradingRecommendation]:
        """Rank recommendations by composite score"""
        # Calculate composite score for each recommendation
        for rec in recommendations:
            # Weighted scoring
            score = (
                rec.signal_strength * 0.3 +
                rec.profit_probability * 0.3 +
                rec.confidence * 0.2 +
                min(rec.risk_reward_ratio / 5, 1.0) *
                0.2  # Normalize RR to 0-1
            )
            rec.composite_score = score

        # Sort by composite score
        ranked = sorted(
            recommendations,
            key=lambda x: x.composite_score,
            reverse=True)

        return ranked

    def _auto_paper_trade(self, recommendations: List[TradingRecommendation]):
        """Automatically paper trade top recommendations"""
        for rec in recommendations:
            try:
                # Convert to format expected by portfolio manager
                position_data = {
                    # Generate ID
                    'id': hash(f"{rec.symbol}{rec.timestamp}") % 1000000,
                    'symbol': rec.symbol,
                    'entry_price': rec.entry_price,
                    'target_price': rec.target_price,
                    'stop_loss': rec.stop_loss,
                    'timeframe': rec.timeframe
                }

                # Open virtual position
                position = self.portfolio_manager.open_position(position_data)
                if position:
                    logger.info(
                        f"Opened paper trade for {rec.symbol} based on {rec.model_used}")

            except Exception as e:
                logger.error(
                    f"Error opening paper trade for {rec.symbol}: {e}")

    async def _get_market_conditions(self) -> Dict:
        """Get current market conditions"""
        conditions = {}

        try:
            # Get VIX level
            vix_data = self.polygon_client.get_quote("VIX")
            if vix_data:
                conditions['vix_level'] = vix_data.price
                conditions['vix_status'] = 'high' if vix_data.price > 20 else 'normal'

            # Get SPY trend
            spy_data = self.polygon_client.get_historical_data(
                "SPY",
                (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                datetime.now().strftime('%Y-%m-%d')
            )
            if spy_data and len(spy_data) > 0:
                spy_returns = spy_data['close'].pct_change().dropna()
                conditions['spy_trend'] = 'uptrend' if spy_returns.mean(
                ) > 0 else 'downtrend'
                conditions['spy_volatility'] = spy_returns.std() * np.sqrt(252)

            # Get economic indicators from FRED
            # Yield curve (10Y - 2Y)
            ten_year = self.fred_client.get_series_data("DGS10")
            two_year = self.fred_client.get_series_data("DGS2")
            if ten_year and two_year:
                yield_curve = ten_year.data.iloc[-1] - two_year.data.iloc[-1]
                conditions['yield_curve'] = yield_curve.values[0]
                conditions['yield_curve_status'] = 'inverted' if yield_curve.values[0] < 0 else 'normal'

        except Exception as e:
            logger.error(f"Error getting market conditions: {e}")

        return conditions

    def get_active_recommendations(self) -> List[Dict]:
        """Get current active recommendations"""
        return [rec.to_dict() for rec in self.active_recommendations]

    def get_recommendation_performance(self) -> Dict:
        """Get performance metrics for recommendations"""
        if not self.recommendation_history:
            return {}

        # Group by model
        model_performance = {}
        for rec in self.recommendation_history:
            if rec.model_used not in model_performance:
                model_performance[rec.model_used] = {
                    'count': 0,
                    'avg_signal_strength': 0,
                    'avg_confidence': 0,
                    'avg_expected_return': 0
                }

            stats = model_performance[rec.model_used]
            stats['count'] += 1
            stats['avg_signal_strength'] += rec.signal_strength
            stats['avg_confidence'] += rec.confidence
            stats['avg_expected_return'] += rec.expected_return

        # Calculate averages
        for model, stats in model_performance.items():
            count = stats['count']
            stats['avg_signal_strength'] /= count
            stats['avg_confidence'] /= count
            stats['avg_expected_return'] /= count

        return model_performance


# Singleton instance
_recommendation_tracker = None


def get_recommendation_tracker() -> RecommendationTracker:
    """Get or create the recommendation tracker instance"""
    global _recommendation_tracker
    if _recommendation_tracker is None:
        _recommendation_tracker = RecommendationTracker()
    return _recommendation_tracker
