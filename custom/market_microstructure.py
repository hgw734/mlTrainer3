"""
Market Microstructure Models Implementation

Analyzes market mechanics, order flow, and trading dynamics.
All models require real Level 2/tick data - no synthetic generation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd


@dataclass
class MicrostructureSignal:
    """Base signal for microstructure analysis."""
    timestamp: datetime
    signal_type: str  # 'bullish', 'bearish', 'neutral'
    strength: float  # 0.0 to 1.0
    metrics: Dict[str, float]
    metadata: Dict[str, any]


@dataclass
class OrderFlowSignal(MicrostructureSignal):
    """Order flow specific signals."""
    buy_volume: float
    sell_volume: float
    imbalance_ratio: float
    large_order_side: Optional[str]
    sweep_detected: bool
    liquidity_levels: List[float]


@dataclass
class DepthSignal(MicrostructureSignal):
    """Market depth analysis signal."""
    bid_depth: float
    ask_depth: float
    bid_walls: List[Tuple[float, float]]  # (price, size)
    ask_walls: List[Tuple[float, float]]
    depth_imbalance: float
    spoofing_probability: float


@dataclass
class VPINMetrics(MicrostructureSignal):
    """VPIN (Volume-synchronized Probability of Informed Trading) metrics."""
    vpin_value: float
    flow_toxicity: float
    volume_bucket_size: int
    order_imbalance: float
    crash_probability: float


@dataclass
class TradeClassification(MicrostructureSignal):
    """Trade classification results."""
    uptick_volume: float
    downtick_volume: float
    neutral_volume: float
    aggressor_ratio: float  # buy initiated / total
    volume_clusters: List[Dict[str, float]]
    tick_momentum: float


@dataclass
class SpreadAnalysis(MicrostructureSignal):
    """Bid-ask spread analysis."""
    current_spread: float
    average_spread: float
    effective_spread: float
    spread_volatility: float
    liquidity_score: float
    optimal_execution_side: str


class BaseMicrostructureModel(ABC):
    """Base class for market microstructure models."""
    
    def __init__(self, min_data_points: int = 100):
        self.min_data_points = min_data_points
    
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> MicrostructureSignal:
        """Analyze market microstructure data."""
        pass
    
    def validate_data(self, data: pd.DataFrame, required_columns: List[str]) -> bool:
        """Validate input data has required columns and sufficient rows."""
        if data is None or len(data) < self.min_data_points:
            return False
        return all(col in data.columns for col in required_columns)


class OrderFlowModel(BaseMicrostructureModel):
    """
    Order flow analysis model.
    
    Analyzes:
    - Buy/sell volume imbalance
    - Large order detection
    - Sweep order identification
    - Liquidity level mapping
    
    Requires Level 2 or trade data with size and side information.
    """
    
    def __init__(self, large_order_threshold: float = 10000,
                 imbalance_threshold: float = 0.65,
                 sweep_window: int = 10):
        super().__init__()
        self.large_order_threshold = large_order_threshold
        self.imbalance_threshold = imbalance_threshold
        self.sweep_window = sweep_window
    
    def analyze(self, data: pd.DataFrame) -> OrderFlowSignal:
        """
        Analyze order flow from trade data.
        
        Expected columns: timestamp, price, size, side (buy/sell)
        """
        required_cols = ['timestamp', 'price', 'size', 'side']
        if not self.validate_data(data, required_cols):
            return self._default_signal()
        
        try:
            # Calculate buy/sell volumes
            buy_data = data[data['side'] == 'buy']
            sell_data = data[data['side'] == 'sell']
            
            buy_volume = buy_data['size'].sum()
            sell_volume = sell_data['size'].sum()
            total_volume = buy_volume + sell_volume
            
            # Calculate imbalance ratio
            imbalance_ratio = buy_volume / total_volume if total_volume > 0 else 0.5
            
            # Detect large orders
            large_orders = data[data['size'] >= self.large_order_threshold]
            large_order_side = None
            if len(large_orders) > 0:
                buy_large = large_orders[large_orders['side'] == 'buy']['size'].sum()
                sell_large = large_orders[large_orders['side'] == 'sell']['size'].sum()
                if buy_large > sell_large * 1.5:
                    large_order_side = 'buy'
                elif sell_large > buy_large * 1.5:
                    large_order_side = 'sell'
            
            # Detect sweep orders (multiple executions at different prices)
            sweep_detected = self._detect_sweeps(data)
            
            # Map liquidity levels (price levels with high volume)
            liquidity_levels = self._find_liquidity_levels(data)
            
            # Determine signal
            if imbalance_ratio > self.imbalance_threshold:
                signal_type = 'bullish'
                strength = min((imbalance_ratio - 0.5) * 2, 1.0)
            elif imbalance_ratio < (1 - self.imbalance_threshold):
                signal_type = 'bearish'
                strength = min((0.5 - imbalance_ratio) * 2, 1.0)
            else:
                signal_type = 'neutral'
                strength = 0.5
            
            # Adjust strength based on large orders
            if large_order_side == 'buy' and signal_type == 'bullish':
                strength = min(strength * 1.2, 1.0)
            elif large_order_side == 'sell' and signal_type == 'bearish':
                strength = min(strength * 1.2, 1.0)
            
            return OrderFlowSignal(
                timestamp=data['timestamp'].iloc[-1],
                signal_type=signal_type,
                strength=strength,
                buy_volume=buy_volume,
                sell_volume=sell_volume,
                imbalance_ratio=imbalance_ratio,
                large_order_side=large_order_side,
                sweep_detected=sweep_detected,
                liquidity_levels=liquidity_levels,
                metrics={
                    'total_volume': total_volume,
                    'large_order_count': len(large_orders),
                    'avg_trade_size': data['size'].mean()
                },
                metadata={
                    'data_points': len(data),
                    'time_range': (data['timestamp'].min(), data['timestamp'].max())
                }
            )
            
        except Exception as e:
            return self._default_signal()
    
    def _detect_sweeps(self, data: pd.DataFrame) -> bool:
        """Detect sweep orders (rapid executions across price levels)."""
        # Group trades by small time windows
        data['time_group'] = pd.to_datetime(data['timestamp']).dt.floor('1s')
        
        for _, group in data.groupby('time_group'):
            if len(group) >= self.sweep_window:
                # Check if trades hit multiple price levels quickly
                unique_prices = group['price'].nunique()
                if unique_prices >= 3:  # Hit at least 3 price levels
                    # Check if mostly same direction
                    side_counts = group['side'].value_counts()
                    if len(side_counts) > 0:
                        dominant_side_pct = side_counts.iloc[0] / len(group)
                        if dominant_side_pct > 0.8:  # 80% same direction
                            return True
        return False
    
    def _find_liquidity_levels(self, data: pd.DataFrame) -> List[float]:
        """Find price levels with high trading volume."""
        # Group by price levels (round to nearest tick)
        data['price_level'] = data['price'].round(2)
        volume_by_price = data.groupby('price_level')['size'].sum().sort_values(ascending=False)
        
        # Return top liquidity levels
        top_levels = volume_by_price.head(5).index.tolist()
        return sorted(top_levels)
    
    def _default_signal(self) -> OrderFlowSignal:
        """Return default neutral signal when analysis fails."""
        return OrderFlowSignal(
            timestamp=datetime.now(),
            signal_type='neutral',
            strength=0.5,
            buy_volume=0,
            sell_volume=0,
            imbalance_ratio=0.5,
            large_order_side=None,
            sweep_detected=False,
            liquidity_levels=[],
            metrics={},
            metadata={'error': 'Insufficient or invalid data'}
        )


class MarketDepthModel(BaseMicrostructureModel):
    """
    Market depth (Level 2) analysis model.
    
    Analyzes:
    - Bid/ask wall detection
    - Depth imbalance ratio
    - Spoofing detection
    - Support/resistance from order book
    
    Requires Level 2 order book data.
    """
    
    def __init__(self, wall_threshold_multiplier: float = 3.0,
                 depth_levels: int = 10,
                 spoofing_threshold: float = 0.7):
        super().__init__(min_data_points=20)
        self.wall_threshold_multiplier = wall_threshold_multiplier
        self.depth_levels = depth_levels
        self.spoofing_threshold = spoofing_threshold
    
    def analyze(self, data: pd.DataFrame) -> DepthSignal:
        """
        Analyze market depth from Level 2 data.
        
        Expected columns: timestamp, bid_price, bid_size, ask_price, ask_size, level
        """
        required_cols = ['timestamp', 'bid_price', 'bid_size', 'ask_price', 'ask_size']
        if not self.validate_data(data, required_cols):
            return self._default_signal()
        
        try:
            # Get latest snapshot
            latest_data = data.groupby('timestamp').last().iloc[-1]
            
            # Calculate total bid/ask depth
            bid_depth = data.groupby('timestamp')['bid_size'].sum().iloc[-1]
            ask_depth = data.groupby('timestamp')['ask_size'].sum().iloc[-1]
            total_depth = bid_depth + ask_depth
            
            # Calculate depth imbalance
            depth_imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0
            
            # Detect bid/ask walls
            bid_walls = self._detect_walls(data, 'bid')
            ask_walls = self._detect_walls(data, 'ask')
            
            # Spoofing detection (orders that appear/disappear quickly)
            spoofing_probability = self._detect_spoofing(data)
            
            # Determine signal
            if depth_imbalance > 0.2:
                signal_type = 'bullish'
                strength = min(0.5 + depth_imbalance, 1.0)
            elif depth_imbalance < -0.2:
                signal_type = 'bearish'
                strength = min(0.5 - depth_imbalance, 1.0)
            else:
                signal_type = 'neutral'
                strength = 0.5
            
            # Adjust for walls
            if bid_walls and not ask_walls:
                signal_type = 'bullish'
                strength = min(strength * 1.2, 1.0)
            elif ask_walls and not bid_walls:
                signal_type = 'bearish'
                strength = min(strength * 1.2, 1.0)
            
            # Reduce confidence if spoofing detected
            if spoofing_probability > self.spoofing_threshold:
                strength *= 0.7
            
            return DepthSignal(
                timestamp=latest_data.name if hasattr(latest_data, 'name') else datetime.now(),
                signal_type=signal_type,
                strength=strength,
                bid_depth=bid_depth,
                ask_depth=ask_depth,
                bid_walls=bid_walls,
                ask_walls=ask_walls,
                depth_imbalance=depth_imbalance,
                spoofing_probability=spoofing_probability,
                metrics={
                    'total_depth': total_depth,
                    'bid_ask_ratio': bid_depth / ask_depth if ask_depth > 0 else float('inf'),
                    'wall_count': len(bid_walls) + len(ask_walls)
                },
                metadata={
                    'depth_levels': self.depth_levels,
                    'snapshot_count': len(data.groupby('timestamp'))
                }
            )
            
        except Exception as e:
            return self._default_signal()
    
    def _detect_walls(self, data: pd.DataFrame, side: str) -> List[Tuple[float, float]]:
        """Detect large order walls in the book."""
        price_col = f'{side}_price'
        size_col = f'{side}_size'
        
        # Get latest book snapshot
        latest_book = data.groupby('timestamp').last()
        if len(latest_book) == 0:
            return []
        
        latest = latest_book.iloc[-1]
        
        # Calculate average size per level
        avg_size = latest[size_col] if isinstance(latest[size_col], (int, float)) else latest[size_col].mean()
        
        # Find levels with size > threshold * average
        walls = []
        if isinstance(latest[price_col], pd.Series):
            for i in range(len(latest[price_col])):
                if latest[size_col].iloc[i] > avg_size * self.wall_threshold_multiplier:
                    walls.append((latest[price_col].iloc[i], latest[size_col].iloc[i]))
        elif latest[size_col] > avg_size * self.wall_threshold_multiplier:
            walls.append((latest[price_col], latest[size_col]))
        
        return walls
    
    def _detect_spoofing(self, data: pd.DataFrame) -> float:
        """Estimate probability of spoofing based on order changes."""
        if len(data.groupby('timestamp')) < 3:
            return 0.0
        
        # Look for large orders that appear and disappear quickly
        # This is simplified - real spoofing detection would track individual orders
        snapshots = data.groupby('timestamp').agg({
            'bid_size': 'sum',
            'ask_size': 'sum'
        })
        
        if len(snapshots) < 3:
            return 0.0
        
        # Calculate volatility of depth
        bid_volatility = snapshots['bid_size'].std() / snapshots['bid_size'].mean() if snapshots['bid_size'].mean() > 0 else 0
        ask_volatility = snapshots['ask_size'].std() / snapshots['ask_size'].mean() if snapshots['ask_size'].mean() > 0 else 0
        
        # High volatility in depth might indicate spoofing
        avg_volatility = (bid_volatility + ask_volatility) / 2
        return min(avg_volatility, 1.0)
    
    def _default_signal(self) -> DepthSignal:
        """Return default neutral signal when analysis fails."""
        return DepthSignal(
            timestamp=datetime.now(),
            signal_type='neutral',
            strength=0.5,
            bid_depth=0,
            ask_depth=0,
            bid_walls=[],
            ask_walls=[],
            depth_imbalance=0,
            spoofing_probability=0,
            metrics={},
            metadata={'error': 'Insufficient or invalid data'}
        )


class VPINModel(BaseMicrostructureModel):
    """
    Volume-synchronized Probability of Informed Trading (VPIN) model.
    
    Measures flow toxicity and probability of adverse selection.
    High VPIN values indicate potential market stress.
    
    Based on Easley, Lopez de Prado, and O'Hara (2012).
    """
    
    def __init__(self, volume_bucket_size: int = 50,
                 n_buckets: int = 50,
                 toxicity_threshold: float = 0.4):
        super().__init__(min_data_points=500)
        self.volume_bucket_size = volume_bucket_size
        self.n_buckets = n_buckets
        self.toxicity_threshold = toxicity_threshold
    
    def analyze(self, data: pd.DataFrame) -> VPINMetrics:
        """
        Calculate VPIN from trade data.
        
        Expected columns: timestamp, price, size, [side]
        """
        required_cols = ['timestamp', 'price', 'size']
        if not self.validate_data(data, required_cols):
            return self._default_signal()
        
        try:
            # Create volume buckets
            buckets = self._create_volume_buckets(data)
            
            if len(buckets) < self.n_buckets:
                return self._default_signal()
            
            # Classify trades if side not provided
            if 'side' not in data.columns:
                data = self._classify_trades(data)
            
            # Calculate order imbalance for each bucket
            imbalances = []
            for bucket in buckets[-self.n_buckets:]:
                bucket_data = data.iloc[bucket['start_idx']:bucket['end_idx']+1]
                buy_volume = bucket_data[bucket_data['side'] == 'buy']['size'].sum()
                sell_volume = bucket_data[bucket_data['side'] == 'sell']['size'].sum()
                total_volume = buy_volume + sell_volume
                
                if total_volume > 0:
                    imbalance = abs(buy_volume - sell_volume) / total_volume
                    imbalances.append(imbalance)
            
            # Calculate VPIN
            vpin_value = np.mean(imbalances) if imbalances else 0.5
            
            # Calculate flow toxicity (rate of VPIN change)
            if len(buckets) > self.n_buckets * 2:
                # Calculate VPIN for previous period
                prev_imbalances = []
                for bucket in buckets[-self.n_buckets*2:-self.n_buckets]:
                    bucket_data = data.iloc[bucket['start_idx']:bucket['end_idx']+1]
                    buy_volume = bucket_data[bucket_data['side'] == 'buy']['size'].sum()
                    sell_volume = bucket_data[bucket_data['side'] == 'sell']['size'].sum()
                    total_volume = buy_volume + sell_volume
                    
                    if total_volume > 0:
                        imbalance = abs(buy_volume - sell_volume) / total_volume
                        prev_imbalances.append(imbalance)
                
                prev_vpin = np.mean(prev_imbalances) if prev_imbalances else 0.5
                flow_toxicity = vpin_value - prev_vpin
            else:
                flow_toxicity = 0.0
            
            # Estimate crash probability based on VPIN level
            if vpin_value > 0.7:
                crash_probability = min((vpin_value - 0.7) * 3.33, 1.0)
            else:
                crash_probability = 0.0
            
            # Determine signal
            if vpin_value > self.toxicity_threshold:
                signal_type = 'bearish'  # High toxicity suggests risk
                strength = min(vpin_value, 1.0)
            else:
                signal_type = 'neutral'
                strength = 0.5
            
            # Calculate average order imbalance
            order_imbalance = np.mean(imbalances) if imbalances else 0.5
            
            return VPINMetrics(
                timestamp=data['timestamp'].iloc[-1],
                signal_type=signal_type,
                strength=strength,
                vpin_value=vpin_value,
                flow_toxicity=flow_toxicity,
                volume_bucket_size=self.volume_bucket_size,
                order_imbalance=order_imbalance,
                crash_probability=crash_probability,
                metrics={
                    'buckets_analyzed': len(imbalances),
                    'avg_bucket_volume': np.mean([b['volume'] for b in buckets[-self.n_buckets:]]),
                    'imbalance_std': np.std(imbalances) if len(imbalances) > 1 else 0
                },
                metadata={
                    'total_buckets': len(buckets),
                    'data_points': len(data)
                }
            )
            
        except Exception as e:
            return self._default_signal()
    
    def _create_volume_buckets(self, data: pd.DataFrame) -> List[Dict]:
        """Create equal-volume buckets from trade data."""
        buckets = []
        current_volume = 0
        start_idx = 0
        
        for i, row in data.iterrows():
            current_volume += row['size']
            
            if current_volume >= self.volume_bucket_size:
                buckets.append({
                    'start_idx': start_idx,
                    'end_idx': i,
                    'volume': current_volume,
                    'trades': i - start_idx + 1
                })
                current_volume = 0
                start_idx = i + 1
        
        return buckets
    
    def _classify_trades(self, data: pd.DataFrame) -> pd.DataFrame:
        """Classify trades using tick rule if side not provided."""
        data = data.copy()
        data['price_change'] = data['price'].diff()
        
        # Tick rule: uptick = buy, downtick = sell
        data['side'] = 'buy'
        data.loc[data['price_change'] < 0, 'side'] = 'sell'
        data.loc[data['price_change'] == 0, 'side'] = 'neutral'
        
        # Handle first row
        data.loc[data.index[0], 'side'] = 'neutral'
        
        return data
    
    def _default_signal(self) -> VPINMetrics:
        """Return default neutral signal when analysis fails."""
        return VPINMetrics(
            timestamp=datetime.now(),
            signal_type='neutral',
            strength=0.5,
            vpin_value=0.5,
            flow_toxicity=0.0,
            volume_bucket_size=self.volume_bucket_size,
            order_imbalance=0.5,
            crash_probability=0.0,
            metrics={},
            metadata={'error': 'Insufficient or invalid data'}
        )


class TickRuleModel(BaseMicrostructureModel):
    """
    Trade classification using tick rule and derivatives.
    
    Classifies trades as buyer or seller initiated and
    analyzes volume patterns and momentum.
    """
    
    def __init__(self, momentum_window: int = 20,
                 cluster_threshold: float = 2.0):
        super().__init__(min_data_points=100)
        self.momentum_window = momentum_window
        self.cluster_threshold = cluster_threshold
    
    def analyze(self, data: pd.DataFrame) -> TradeClassification:
        """
        Classify trades and analyze patterns.
        
        Expected columns: timestamp, price, size
        """
        required_cols = ['timestamp', 'price', 'size']
        if not self.validate_data(data, required_cols):
            return self._default_signal()
        
        try:
            # Apply tick rule classification
            data = self._apply_tick_rule(data)
            
            # Calculate volumes by classification
            uptick_volume = data[data['tick_direction'] == 'uptick']['size'].sum()
            downtick_volume = data[data['tick_direction'] == 'downtick']['size'].sum()
            neutral_volume = data[data['tick_direction'] == 'neutral']['size'].sum()
            total_volume = uptick_volume + downtick_volume + neutral_volume
            
            # Calculate aggressor ratio (uptick = buy initiated)
            aggressor_ratio = uptick_volume / total_volume if total_volume > 0 else 0.5
            
            # Detect volume clusters
            volume_clusters = self._detect_volume_clusters(data)
            
            # Calculate tick momentum
            tick_momentum = self._calculate_tick_momentum(data)
            
            # Determine signal
            if aggressor_ratio > 0.6:
                signal_type = 'bullish'
                strength = min((aggressor_ratio - 0.5) * 2, 1.0)
            elif aggressor_ratio < 0.4:
                signal_type = 'bearish'
                strength = min((0.5 - aggressor_ratio) * 2, 1.0)
            else:
                signal_type = 'neutral'
                strength = 0.5
            
            # Adjust for momentum
            if tick_momentum > 0.2 and signal_type == 'bullish':
                strength = min(strength * 1.1, 1.0)
            elif tick_momentum < -0.2 and signal_type == 'bearish':
                strength = min(strength * 1.1, 1.0)
            
            return TradeClassification(
                timestamp=data['timestamp'].iloc[-1],
                signal_type=signal_type,
                strength=strength,
                uptick_volume=uptick_volume,
                downtick_volume=downtick_volume,
                neutral_volume=neutral_volume,
                aggressor_ratio=aggressor_ratio,
                volume_clusters=volume_clusters,
                tick_momentum=tick_momentum,
                metrics={
                    'total_trades': len(data),
                    'avg_trade_size': data['size'].mean(),
                    'volume_volatility': data['size'].std() / data['size'].mean() if data['size'].mean() > 0 else 0
                },
                metadata={
                    'time_range': (data['timestamp'].min(), data['timestamp'].max())
                }
            )
            
        except Exception as e:
            return self._default_signal()
    
    def _apply_tick_rule(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply tick rule to classify trades."""
        data = data.copy()
        
        # Calculate price changes
        data['price_change'] = data['price'].diff()
        
        # Classify based on price movement
        data['tick_direction'] = 'neutral'
        data.loc[data['price_change'] > 0, 'tick_direction'] = 'uptick'
        data.loc[data['price_change'] < 0, 'tick_direction'] = 'downtick'
        
        # For zero change, use previous non-zero change
        for i in range(1, len(data)):
            if data.iloc[i]['price_change'] == 0:
                # Look back for last non-zero change
                for j in range(i-1, -1, -1):
                    if data.iloc[j]['price_change'] != 0:
                        data.iloc[i, data.columns.get_loc('tick_direction')] = data.iloc[j]['tick_direction']
                        break
        
        return data
    
    def _detect_volume_clusters(self, data: pd.DataFrame) -> List[Dict[str, float]]:
        """Detect clusters of high volume trading."""
        clusters = []
        avg_volume = data['size'].mean()
        std_volume = data['size'].std()
        
        if std_volume == 0:
            return clusters
        
        # Find periods of high volume
        high_volume_threshold = avg_volume + self.cluster_threshold * std_volume
        data['is_high_volume'] = data['size'] > high_volume_threshold
        
        # Group consecutive high volume trades
        cluster_start = None
        for i in range(len(data)):
            if data.iloc[i]['is_high_volume']:
                if cluster_start is None:
                    cluster_start = i
            else:
                if cluster_start is not None:
                    cluster_data = data.iloc[cluster_start:i]
                    clusters.append({
                        'start_time': cluster_data['timestamp'].iloc[0],
                        'end_time': cluster_data['timestamp'].iloc[-1],
                        'total_volume': cluster_data['size'].sum(),
                        'avg_price': cluster_data['price'].mean(),
                        'direction': cluster_data['tick_direction'].mode()[0] if len(cluster_data['tick_direction'].mode()) > 0 else 'neutral'
                    })
                    cluster_start = None
        
        return clusters
    
    def _calculate_tick_momentum(self, data: pd.DataFrame) -> float:
        """Calculate momentum based on tick direction."""
        if len(data) < self.momentum_window:
            return 0.0
        
        recent_data = data.tail(self.momentum_window)
        
        # Calculate directional volume ratio
        uptick_vol = recent_data[recent_data['tick_direction'] == 'uptick']['size'].sum()
        downtick_vol = recent_data[recent_data['tick_direction'] == 'downtick']['size'].sum()
        total_vol = uptick_vol + downtick_vol
        
        if total_vol == 0:
            return 0.0
        
        momentum = (uptick_vol - downtick_vol) / total_vol
        return momentum
    
    def _default_signal(self) -> TradeClassification:
        """Return default neutral signal when analysis fails."""
        return TradeClassification(
            timestamp=datetime.now(),
            signal_type='neutral',
            strength=0.5,
            uptick_volume=0,
            downtick_volume=0,
            neutral_volume=0,
            aggressor_ratio=0.5,
            volume_clusters=[],
            tick_momentum=0.0,
            metrics={},
            metadata={'error': 'Insufficient or invalid data'}
        )


class BidAskSpreadModel(BaseMicrostructureModel):
    """
    Bid-ask spread analysis for liquidity and volatility signals.
    
    Analyzes spread dynamics to predict volatility and
    determine optimal execution timing.
    """
    
    def __init__(self, spread_window: int = 50,
                 volatility_lookback: int = 100):
        super().__init__(min_data_points=50)
        self.spread_window = spread_window
        self.volatility_lookback = volatility_lookback
    
    def analyze(self, data: pd.DataFrame) -> SpreadAnalysis:
        """
        Analyze bid-ask spread dynamics.
        
        Expected columns: timestamp, bid, ask, [bid_size, ask_size]
        """
        required_cols = ['timestamp', 'bid', 'ask']
        if not self.validate_data(data, required_cols):
            return self._default_signal()
        
        try:
            # Calculate spreads
            data['spread'] = data['ask'] - data['bid']
            data['spread_pct'] = data['spread'] / ((data['bid'] + data['ask']) / 2) * 100
            
            # Current and average spreads
            current_spread = data['spread'].iloc[-1]
            average_spread = data['spread'].tail(self.spread_window).mean()
            
            # Calculate effective spread (considering sizes if available)
            if 'bid_size' in data.columns and 'ask_size' in data.columns:
                # Weight by size
                data['effective_spread'] = data['spread'] * (
                    1 + abs(data['bid_size'] - data['ask_size']) / 
                    (data['bid_size'] + data['ask_size'])
                )
                effective_spread = data['effective_spread'].iloc[-1]
            else:
                effective_spread = current_spread
            
            # Calculate spread volatility
            spread_volatility = data['spread_pct'].tail(self.volatility_lookback).std()
            
            # Calculate liquidity score (inverse of spread)
            avg_spread_pct = data['spread_pct'].tail(self.spread_window).mean()
            liquidity_score = max(0, min(1, 1 - avg_spread_pct / 0.5))  # 0.5% spread = 0 score
            
            # Determine optimal execution side
            # If spread is widening, execute market orders on the passive side
            spread_trend = data['spread'].tail(20).diff().mean()
            if spread_trend > 0:  # Spread widening
                optimal_execution_side = 'limit'  # Use limit orders
            else:
                optimal_execution_side = 'market'  # Market orders acceptable
            
            # Determine signal based on spread dynamics
            if current_spread > average_spread * 1.5:
                # Wide spread indicates uncertainty/volatility
                signal_type = 'neutral'  # Caution
                strength = 0.3
            elif spread_volatility > data['spread_pct'].std() * 1.5:
                # High spread volatility
                signal_type = 'neutral'
                strength = 0.4
            else:
                # Normal spread conditions
                if liquidity_score > 0.7:
                    signal_type = 'bullish'  # Good liquidity
                    strength = 0.6 + liquidity_score * 0.2
                else:
                    signal_type = 'neutral'
                    strength = 0.5
            
            return SpreadAnalysis(
                timestamp=data['timestamp'].iloc[-1],
                signal_type=signal_type,
                strength=strength,
                current_spread=current_spread,
                average_spread=average_spread,
                effective_spread=effective_spread,
                spread_volatility=spread_volatility,
                liquidity_score=liquidity_score,
                optimal_execution_side=optimal_execution_side,
                metrics={
                    'spread_pct': data['spread_pct'].iloc[-1],
                    'avg_spread_pct': avg_spread_pct,
                    'spread_trend': spread_trend,
                    'min_spread': data['spread'].tail(self.spread_window).min(),
                    'max_spread': data['spread'].tail(self.spread_window).max()
                },
                metadata={
                    'data_points': len(data),
                    'has_size_data': 'bid_size' in data.columns
                }
            )
            
        except Exception as e:
            return self._default_signal()
    
    def _default_signal(self) -> SpreadAnalysis:
        """Return default neutral signal when analysis fails."""
        return SpreadAnalysis(
            timestamp=datetime.now(),
            signal_type='neutral',
            strength=0.5,
            current_spread=0,
            average_spread=0,
            effective_spread=0,
            spread_volatility=0,
            liquidity_score=0.5,
            optimal_execution_side='limit',
            metrics={},
            metadata={'error': 'Insufficient or invalid data'}
        )


# Model factory for easy instantiation
def create_microstructure_model(model_type: str, **kwargs) -> BaseMicrostructureModel:
    """Factory function to create microstructure models."""
    models = {
        'order_flow': OrderFlowModel,
        'market_depth': MarketDepthModel,
        'vpin': VPINModel,
        'tick_rule': TickRuleModel,
        'spread': BidAskSpreadModel
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return models[model_type](**kwargs)