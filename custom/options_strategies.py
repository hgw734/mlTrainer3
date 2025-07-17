"""
Options Strategies Models Implementation

Implements delta-neutral hedging, volatility arbitrage, spread strategies,
gamma scalping, and iron condor strategies using real options data.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


@dataclass
class OptionsSignal:
    """Base signal for options strategies."""
    timestamp: datetime
    strategy_type: str
    action: str  # 'open', 'close', 'adjust', 'hold'
    contracts: List[Dict[str, any]]  # List of option contracts
    net_premium: float
    max_profit: float
    max_loss: float
    probability_profit: float
    confidence: float
    metrics: Dict[str, float]
    metadata: Dict[str, any]


@dataclass
class HedgeOrders(OptionsSignal):
    """Delta neutral hedging orders."""
    portfolio_delta: float
    target_delta: float
    hedge_quantity: float
    hedge_instrument: str  # 'stock', 'options', 'futures'
    rebalance_urgency: str  # 'immediate', 'normal', 'low'
    transaction_cost: float


@dataclass
class VolArbTrades(OptionsSignal):
    """Volatility arbitrage trades."""
    implied_volatility: float
    realized_volatility: float
    volatility_spread: float
    hedge_ratio: float
    vega_exposure: float
    theta_decay: float


@dataclass
class SpreadOpportunities(OptionsSignal):
    """Options spread opportunities."""
    spread_type: str  # 'vertical', 'calendar', 'diagonal', 'butterfly'
    legs: List[Dict[str, any]]
    debit_credit: str  # 'debit', 'credit'
    breakeven_points: List[float]
    risk_reward_ratio: float
    days_to_expiry: int


@dataclass
class ScalpingSignals(OptionsSignal):
    """Gamma scalping signals."""
    gamma_exposure: float
    delta_band: Tuple[float, float]
    scalp_threshold: float
    current_pnl: float
    scalps_today: int
    rehedge_required: bool


@dataclass
class IronCondorTrade(OptionsSignal):
    """Iron condor trade setup."""
    short_call_strike: float
    long_call_strike: float
    short_put_strike: float
    long_put_strike: float
    credit_received: float
    margin_required: float
    profit_range: Tuple[float, float]
    adjustment_points: List[float]


class BaseOptionsModel(ABC):
    """Base class for options trading models."""
    
    def __init__(self, risk_free_rate: float = 0.05,
                 min_volume: int = 100,
                 min_open_interest: int = 100):
        self.risk_free_rate = risk_free_rate
        self.min_volume = min_volume
        self.min_open_interest = min_open_interest
    
    @abstractmethod
    def analyze(self, options_data: pd.DataFrame, spot_price: float) -> OptionsSignal:
        """Analyze options data for trading opportunities."""
        pass
    
    def validate_data(self, options_data: pd.DataFrame) -> bool:
        """Validate options chain data."""
        if options_data is None or len(options_data) == 0:
            return False
        
        required_cols = ['strike', 'expiry', 'type', 'bid', 'ask', 'volume', 'open_interest']
        return all(col in options_data.columns for col in required_cols)
    
    def calculate_greeks(self, spot: float, strike: float, time_to_expiry: float,
                        volatility: float, option_type: str) -> Dict[str, float]:
        """Calculate option Greeks using Black-Scholes."""
        # Convert time to years
        T = time_to_expiry / 365.0
        
        if T <= 0:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        
        # Calculate d1 and d2
        d1 = (np.log(spot / strike) + (self.risk_free_rate + 0.5 * volatility**2) * T) / (volatility * np.sqrt(T))
        d2 = d1 - volatility * np.sqrt(T)
        
        # Greeks calculation
        if option_type.lower() == 'call':
            delta = norm.cdf(d1)
            theta = (-spot * norm.pdf(d1) * volatility / (2 * np.sqrt(T)) 
                    - self.risk_free_rate * strike * np.exp(-self.risk_free_rate * T) * norm.cdf(d2)) / 365
        else:  # put
            delta = norm.cdf(d1) - 1
            theta = (-spot * norm.pdf(d1) * volatility / (2 * np.sqrt(T)) 
                    + self.risk_free_rate * strike * np.exp(-self.risk_free_rate * T) * norm.cdf(-d2)) / 365
        
        gamma = norm.pdf(d1) / (spot * volatility * np.sqrt(T))
        vega = spot * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% change in volatility
        rho = strike * T * np.exp(-self.risk_free_rate * T) * (norm.cdf(d2) if option_type.lower() == 'call' else -norm.cdf(-d2)) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    def calculate_implied_volatility(self, option_price: float, spot: float, strike: float,
                                   time_to_expiry: float, option_type: str) -> float:
        """Calculate implied volatility using Newton-Raphson method."""
        # Initial guess
        iv = 0.3
        
        for _ in range(100):  # Max iterations
            # Calculate option price with current IV
            bs_price = self.black_scholes_price(spot, strike, time_to_expiry, iv, option_type)
            
            # Calculate vega
            greeks = self.calculate_greeks(spot, strike, time_to_expiry, iv, option_type)
            vega = greeks['vega']
            
            # Newton-Raphson update
            if abs(vega) < 1e-10:
                break
                
            price_diff = option_price - bs_price
            iv = iv + price_diff / (vega * 100)  # Vega is per 1% change
            
            # Bound IV
            iv = max(0.01, min(5.0, iv))
            
            if abs(price_diff) < 0.01:
                break
        
        return iv
    
    def black_scholes_price(self, spot: float, strike: float, time_to_expiry: float,
                           volatility: float, option_type: str) -> float:
        """Calculate Black-Scholes option price."""
        T = time_to_expiry / 365.0
        
        if T <= 0:
            if option_type.lower() == 'call':
                return max(0, spot - strike)
            else:
                return max(0, strike - spot)
        
        d1 = (np.log(spot / strike) + (self.risk_free_rate + 0.5 * volatility**2) * T) / (volatility * np.sqrt(T))
        d2 = d1 - volatility * np.sqrt(T)
        
        if option_type.lower() == 'call':
            price = spot * norm.cdf(d1) - strike * np.exp(-self.risk_free_rate * T) * norm.cdf(d2)
        else:
            price = strike * np.exp(-self.risk_free_rate * T) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        
        return price


class DeltaNeutralModel(BaseOptionsModel):
    """
    Delta neutral portfolio management.
    
    Maintains zero delta exposure through dynamic hedging with
    underlying assets or other options.
    """
    
    def __init__(self, target_delta: float = 0.0,
                 rebalance_threshold: float = 0.05,
                 transaction_cost: float = 0.001):
        super().__init__()
        self.target_delta = target_delta
        self.rebalance_threshold = rebalance_threshold
        self.transaction_cost = transaction_cost
    
    def analyze(self, portfolio: Dict, spot_price: float) -> HedgeOrders:
        """
        Analyze portfolio and generate hedging orders.
        
        Args:
            portfolio: Dict containing positions with keys:
                      'options': List of option positions
                      'stock': Stock position
            spot_price: Current underlying price
        """
        # Calculate portfolio Greeks
        portfolio_greeks = self._calculate_portfolio_greeks(portfolio, spot_price)
        portfolio_delta = portfolio_greeks['delta']
        
        # Calculate hedge requirement
        delta_difference = portfolio_delta - self.target_delta
        
        # Determine if rebalancing needed
        if abs(delta_difference) < self.rebalance_threshold:
            return self._no_action_signal(portfolio_delta)
        
        # Calculate hedge orders
        hedge_orders = self._calculate_hedge_orders(
            delta_difference, spot_price, portfolio
        )
        
        # Determine urgency
        urgency = self._determine_urgency(delta_difference, portfolio_greeks)
        
        # Calculate transaction costs
        trans_cost = self._estimate_transaction_cost(hedge_orders)
        
        # Calculate metrics
        max_profit, max_loss = self._calculate_pnl_range(portfolio, spot_price)
        probability_profit = self._calculate_profit_probability(portfolio, spot_price)
        
        confidence = self._calculate_confidence(portfolio_greeks, delta_difference)
        
        return HedgeOrders(
            timestamp=datetime.now(),
            strategy_type='delta_neutral',
            action='adjust',
            contracts=hedge_orders['contracts'],
            net_premium=hedge_orders['net_cost'],
            max_profit=max_profit,
            max_loss=max_loss,
            probability_profit=probability_profit,
            confidence=confidence,
            portfolio_delta=portfolio_delta,
            target_delta=self.target_delta,
            hedge_quantity=hedge_orders['quantity'],
            hedge_instrument=hedge_orders['instrument'],
            rebalance_urgency=urgency,
            transaction_cost=trans_cost,
            metrics={
                'gamma': portfolio_greeks['gamma'],
                'theta': portfolio_greeks['theta'],
                'vega': portfolio_greeks['vega'],
                'delta_drift': self._calculate_delta_drift(portfolio_greeks)
            },
            metadata={
                'positions': len(portfolio.get('options', [])),
                'rebalance_threshold': self.rebalance_threshold
            }
        )
    
    def _calculate_portfolio_greeks(self, portfolio: Dict, spot_price: float) -> Dict[str, float]:
        """Calculate aggregate Greeks for portfolio."""
        total_greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        
        # Add stock delta
        if 'stock' in portfolio:
            total_greeks['delta'] += portfolio['stock']
        
        # Add options Greeks
        for option in portfolio.get('options', []):
            # Calculate time to expiry
            expiry = pd.to_datetime(option['expiry'])
            time_to_expiry = (expiry - datetime.now()).days
            
            # Calculate Greeks
            greeks = self.calculate_greeks(
                spot_price,
                option['strike'],
                time_to_expiry,
                option.get('volatility', 0.3),
                option['type']
            )
            
            # Add to total (multiply by position size)
            position_size = option.get('quantity', 0)
            for greek, value in greeks.items():
                total_greeks[greek] += value * position_size * 100  # Options are per 100 shares
        
        return total_greeks
    
    def _calculate_hedge_orders(self, delta_difference: float, 
                              spot_price: float, portfolio: Dict) -> Dict:
        """Calculate orders needed to hedge delta."""
        hedge_orders = {
            'contracts': [],
            'quantity': 0,
            'instrument': 'stock',
            'net_cost': 0
        }
        
        # Simple hedge with stock
        stock_hedge = -delta_difference
        
        hedge_orders['quantity'] = round(stock_hedge)
        hedge_orders['net_cost'] = abs(stock_hedge * spot_price * self.transaction_cost)
        
        if stock_hedge != 0:
            hedge_orders['contracts'].append({
                'instrument': 'stock',
                'action': 'buy' if stock_hedge > 0 else 'sell',
                'quantity': abs(round(stock_hedge)),
                'price': spot_price
            })
        
        # Could also hedge with options for more complex strategies
        # This would involve finding appropriate strikes and quantities
        
        return hedge_orders
    
    def _determine_urgency(self, delta_difference: float, 
                          portfolio_greeks: Dict[str, float]) -> str:
        """Determine rebalancing urgency."""
        # High gamma means delta changes quickly
        gamma_urgency = abs(portfolio_greeks['gamma']) > 0.02
        
        # Large delta difference
        delta_urgency = abs(delta_difference) > self.rebalance_threshold * 5
        
        if gamma_urgency and delta_urgency:
            return 'immediate'
        elif gamma_urgency or delta_urgency:
            return 'normal'
        else:
            return 'low'
    
    def _estimate_transaction_cost(self, hedge_orders: Dict) -> float:
        """Estimate total transaction costs."""
        cost = hedge_orders['net_cost']
        
        # Add spread costs
        for contract in hedge_orders['contracts']:
            if contract['instrument'] == 'stock':
                # Assume 0.01% spread
                cost += contract['quantity'] * contract['price'] * 0.0001
            else:
                # Options have wider spreads
                cost += contract['quantity'] * contract['price'] * 0.02
        
        return cost
    
    def _calculate_pnl_range(self, portfolio: Dict, spot_price: float) -> Tuple[float, float]:
        """Calculate max profit and loss for portfolio."""
        # Simplified - would need full payoff diagram calculation
        strikes = [opt['strike'] for opt in portfolio.get('options', [])]
        
        if not strikes:
            return 0, 0
        
        # Test range of prices
        test_prices = np.linspace(
            min(strikes) * 0.8,
            max(strikes) * 1.2,
            100
        )
        
        pnls = []
        for price in test_prices:
            pnl = self._calculate_portfolio_value(portfolio, price, spot_price)
            pnls.append(pnl)
        
        return max(pnls), min(pnls)
    
    def _calculate_portfolio_value(self, portfolio: Dict, 
                                  price_at_expiry: float, 
                                  current_price: float) -> float:
        """Calculate portfolio value at given price."""
        value = 0
        
        # Stock value change
        if 'stock' in portfolio:
            value += portfolio['stock'] * (price_at_expiry - current_price)
        
        # Options value at expiry
        for option in portfolio.get('options', []):
            intrinsic = 0
            if option['type'].lower() == 'call':
                intrinsic = max(0, price_at_expiry - option['strike'])
            else:
                intrinsic = max(0, option['strike'] - price_at_expiry)
            
            # Subtract premium paid/received
            option_value = intrinsic - option.get('premium', 0)
            value += option_value * option.get('quantity', 0) * 100
        
        return value
    
    def _calculate_profit_probability(self, portfolio: Dict, spot_price: float) -> float:
        """Estimate probability of profit."""
        # Simplified - assumes normal distribution
        # Get breakeven points
        max_profit, max_loss = self._calculate_pnl_range(portfolio, spot_price)
        
        if max_loss >= 0:
            return 1.0  # Always profitable
        if max_profit <= 0:
            return 0.0  # Always loss
        
        # Find breakeven price (simplified)
        # Would need proper root finding in practice
        return 0.5  # Placeholder
    
    def _calculate_delta_drift(self, portfolio_greeks: Dict[str, float]) -> float:
        """Calculate expected delta change from gamma."""
        # Delta drift = gamma * expected price move
        # Assume 1% daily move
        return portfolio_greeks['gamma'] * 0.01
    
    def _calculate_confidence(self, portfolio_greeks: Dict[str, float], 
                            delta_difference: float) -> float:
        """Calculate confidence in hedge effectiveness."""
        # Low gamma = more stable hedge
        gamma_confidence = 1 - min(abs(portfolio_greeks['gamma']) * 10, 1.0)
        
        # Small delta difference = closer to target
        delta_confidence = 1 - min(abs(delta_difference) / 100, 1.0)
        
        return (gamma_confidence + delta_confidence) / 2
    
    def _no_action_signal(self, portfolio_delta: float) -> HedgeOrders:
        """Return signal when no hedging needed."""
        return HedgeOrders(
            timestamp=datetime.now(),
            strategy_type='delta_neutral',
            action='hold',
            contracts=[],
            net_premium=0,
            max_profit=0,
            max_loss=0,
            probability_profit=0,
            confidence=1.0,
            portfolio_delta=portfolio_delta,
            target_delta=self.target_delta,
            hedge_quantity=0,
            hedge_instrument='none',
            rebalance_urgency='none',
            transaction_cost=0,
            metrics={},
            metadata={'status': 'delta within threshold'}
        )


class VolatilityArbModel(BaseOptionsModel):
    """
    Volatility arbitrage between implied and realized volatility.
    
    Identifies mispricing between option implied volatility and
    expected realized volatility.
    """
    
    def __init__(self, lookback_period: int = 30,
                 vol_threshold: float = 0.05,
                 min_edge: float = 0.02):
        super().__init__()
        self.lookback_period = lookback_period
        self.vol_threshold = vol_threshold
        self.min_edge = min_edge
    
    def analyze(self, options_chain: pd.DataFrame, price_history: pd.DataFrame) -> VolArbTrades:
        """
        Find volatility arbitrage opportunities.
        
        Args:
            options_chain: Current options data
            price_history: Historical price data for volatility calculation
        """
        if not self.validate_data(options_chain) or len(price_history) < self.lookback_period:
            return self._default_signal()
        
        try:
            # Calculate realized volatility
            realized_vol = self._calculate_realized_volatility(price_history)
            
            # Calculate volatility forecast
            vol_forecast = self._forecast_volatility(price_history)
            
            # Find best volatility arbitrage opportunity
            best_trade = self._find_best_vol_arb(
                options_chain, realized_vol, vol_forecast, price_history.iloc[-1]['close']
            )
            
            if best_trade is None:
                return self._default_signal()
            
            # Calculate hedge ratio
            hedge_ratio = self._calculate_hedge_ratio(best_trade, price_history.iloc[-1]['close'])
            
            # Calculate risk metrics
            vega_exposure = best_trade['vega'] * best_trade['quantity'] * 100
            theta_decay = best_trade['theta'] * best_trade['quantity'] * 100
            
            # Estimate P&L
            max_profit, max_loss = self._estimate_vol_arb_pnl(
                best_trade, realized_vol, vol_forecast
            )
            
            # Probability of profit
            prob_profit = self._calculate_vol_arb_probability(
                best_trade['iv'], realized_vol, vol_forecast
            )
            
            confidence = self._calculate_confidence(
                best_trade['iv'], realized_vol, vol_forecast
            )
            
            return VolArbTrades(
                timestamp=datetime.now(),
                strategy_type='volatility_arbitrage',
                action='open',
                contracts=[best_trade],
                net_premium=best_trade['premium'] * best_trade['quantity'] * 100,
                max_profit=max_profit,
                max_loss=max_loss,
                probability_profit=prob_profit,
                confidence=confidence,
                implied_volatility=best_trade['iv'],
                realized_volatility=realized_vol,
                volatility_spread=best_trade['iv'] - vol_forecast,
                hedge_ratio=hedge_ratio,
                vega_exposure=vega_exposure,
                theta_decay=theta_decay,
                metrics={
                    'historical_vol': realized_vol,
                    'forecast_vol': vol_forecast,
                    'vol_premium': best_trade['iv'] - realized_vol,
                    'sharpe_ratio': self._calculate_vol_arb_sharpe(best_trade, realized_vol)
                },
                metadata={
                    'lookback_period': self.lookback_period,
                    'option_type': best_trade['type'],
                    'days_to_expiry': best_trade['days_to_expiry']
                }
            )
            
        except Exception as e:
            return self._default_signal()
    
    def _calculate_realized_volatility(self, price_history: pd.DataFrame) -> float:
        """Calculate historical realized volatility."""
        returns = price_history['close'].pct_change().dropna()
        
        # Annualized volatility
        vol = returns.tail(self.lookback_period).std() * np.sqrt(252)
        
        return vol
    
    def _forecast_volatility(self, price_history: pd.DataFrame) -> float:
        """Forecast future volatility using GARCH or similar."""
        # Simplified - use weighted average of recent volatilities
        returns = price_history['close'].pct_change().dropna()
        
        # Calculate volatilities over different windows
        vol_5 = returns.tail(5).std() * np.sqrt(252)
        vol_10 = returns.tail(10).std() * np.sqrt(252)
        vol_20 = returns.tail(20).std() * np.sqrt(252)
        vol_30 = returns.tail(30).std() * np.sqrt(252)
        
        # Weighted average (more weight on recent)
        forecast = (vol_5 * 0.4 + vol_10 * 0.3 + vol_20 * 0.2 + vol_30 * 0.1)
        
        return forecast
    
    def _find_best_vol_arb(self, options_chain: pd.DataFrame, 
                          realized_vol: float, vol_forecast: float,
                          spot_price: float) -> Optional[Dict]:
        """Find best volatility arbitrage trade."""
        best_trade = None
        best_edge = 0
        
        for _, option in options_chain.iterrows():
            # Filter liquid options
            if option['volume'] < self.min_volume or option['open_interest'] < self.min_open_interest:
                continue
            
            # Calculate mid price
            mid_price = (option['bid'] + option['ask']) / 2
            
            # Calculate implied volatility
            days_to_expiry = (pd.to_datetime(option['expiry']) - datetime.now()).days
            
            iv = self.calculate_implied_volatility(
                mid_price, spot_price, option['strike'], 
                days_to_expiry, option['type']
            )
            
            # Calculate edge
            if iv > vol_forecast + self.vol_threshold:
                # Sell volatility (sell options)
                edge = iv - vol_forecast
                action = 'sell'
            elif iv < vol_forecast - self.vol_threshold:
                # Buy volatility (buy options)
                edge = vol_forecast - iv
                action = 'buy'
            else:
                continue
            
            # Check minimum edge
            if edge < self.min_edge:
                continue
            
            # Calculate Greeks
            greeks = self.calculate_greeks(
                spot_price, option['strike'], days_to_expiry, iv, option['type']
            )
            
            if edge > best_edge:
                best_edge = edge
                best_trade = {
                    'strike': option['strike'],
                    'expiry': option['expiry'],
                    'type': option['type'],
                    'action': action,
                    'quantity': -1 if action == 'sell' else 1,
                    'premium': mid_price,
                    'iv': iv,
                    'days_to_expiry': days_to_expiry,
                    **greeks
                }
        
        return best_trade
    
    def _calculate_hedge_ratio(self, trade: Dict, spot_price: float) -> float:
        """Calculate delta hedge ratio."""
        return -trade['delta'] * trade['quantity']
    
    def _estimate_vol_arb_pnl(self, trade: Dict, realized_vol: float, 
                            vol_forecast: float) -> Tuple[float, float]:
        """Estimate P&L from volatility arbitrage."""
        # Vega P&L from volatility change
        vol_change = vol_forecast - trade['iv']
        vega_pnl = trade['vega'] * vol_change * 100 * trade['quantity'] * 100
        
        # Theta P&L
        theta_pnl = trade['theta'] * trade['days_to_expiry'] * trade['quantity'] * 100
        
        # Max profit (if vol moves our way)
        if trade['action'] == 'sell':
            max_profit = trade['premium'] * abs(trade['quantity']) * 100
        else:
            # Buying vol - profit if vol increases significantly
            max_vol_increase = 0.3  # 30% vol increase
            max_profit = trade['vega'] * max_vol_increase * 100 * trade['quantity'] * 100
        
        # Max loss
        if trade['action'] == 'sell':
            # Selling options - potentially unlimited loss
            max_loss = -trade['premium'] * abs(trade['quantity']) * 100 * 3  # 3x premium
        else:
            # Buying options - max loss is premium paid
            max_loss = -trade['premium'] * abs(trade['quantity']) * 100
        
        return max_profit, max_loss
    
    def _calculate_vol_arb_probability(self, iv: float, realized_vol: float, 
                                     vol_forecast: float) -> float:
        """Calculate probability of profit for vol arb."""
        # Simplified - based on historical accuracy of forecast
        vol_diff = abs(iv - vol_forecast)
        
        # Higher difference = higher confidence
        if vol_diff > 0.1:  # 10% vol difference
            return 0.7
        elif vol_diff > 0.05:
            return 0.6
        else:
            return 0.5
    
    def _calculate_vol_arb_sharpe(self, trade: Dict, realized_vol: float) -> float:
        """Calculate Sharpe ratio for volatility arbitrage."""
        # Expected return from vol premium
        vol_premium = trade['iv'] - realized_vol
        expected_return = vol_premium * trade['vega'] * trade['quantity']
        
        # Risk (volatility of volatility)
        vol_of_vol = 0.3  # Assumed
        risk = abs(trade['vega'] * vol_of_vol * trade['quantity'])
        
        if risk > 0:
            return expected_return / risk
        return 0
    
    def _calculate_confidence(self, iv: float, realized_vol: float, 
                            vol_forecast: float) -> float:
        """Calculate confidence in volatility arbitrage."""
        # Edge size
        edge = abs(iv - vol_forecast)
        edge_confidence = min(edge / 0.1, 1.0)  # 10% edge = full confidence
        
        # Historical vs forecast alignment
        forecast_accuracy = 1 - min(abs(vol_forecast - realized_vol) / realized_vol, 1.0)
        
        return (edge_confidence + forecast_accuracy) / 2
    
    def _default_signal(self) -> VolArbTrades:
        """Return default signal when no opportunities found."""
        return VolArbTrades(
            timestamp=datetime.now(),
            strategy_type='volatility_arbitrage',
            action='hold',
            contracts=[],
            net_premium=0,
            max_profit=0,
            max_loss=0,
            probability_profit=0,
            confidence=0,
            implied_volatility=0,
            realized_volatility=0,
            volatility_spread=0,
            hedge_ratio=0,
            vega_exposure=0,
            theta_decay=0,
            metrics={},
            metadata={'status': 'no opportunities found'}
        )


class OptionsSpreadsModel(BaseOptionsModel):
    """
    Options spread strategies.
    
    Identifies and trades vertical, calendar, diagonal spreads,
    and complex strategies like butterflies and condors.
    """
    
    def __init__(self, min_credit: float = 0.1,
                 max_risk_reward: float = 3.0,
                 target_probability: float = 0.6):
        super().__init__()
        self.min_credit = min_credit
        self.max_risk_reward = max_risk_reward
        self.target_probability = target_probability
    
    def analyze(self, options_chain: pd.DataFrame, spot_price: float) -> SpreadOpportunities:
        """Find optimal spread trading opportunities."""
        if not self.validate_data(options_chain):
            return self._default_signal()
        
        try:
            # Find different spread opportunities
            spreads = []
            
            # Vertical spreads
            vertical_spreads = self._find_vertical_spreads(options_chain, spot_price)
            spreads.extend(vertical_spreads)
            
            # Calendar spreads
            calendar_spreads = self._find_calendar_spreads(options_chain, spot_price)
            spreads.extend(calendar_spreads)
            
            # Butterfly spreads
            butterfly_spreads = self._find_butterfly_spreads(options_chain, spot_price)
            spreads.extend(butterfly_spreads)
            
            # Select best spread
            best_spread = self._select_best_spread(spreads)
            
            if best_spread is None:
                return self._default_signal()
            
            # Calculate detailed metrics
            breakeven_points = self._calculate_breakevens(best_spread)
            risk_reward = self._calculate_risk_reward(best_spread)
            prob_profit = self._calculate_spread_probability(best_spread, spot_price)
            
            confidence = self._calculate_confidence(best_spread, risk_reward, prob_profit)
            
            return SpreadOpportunities(
                timestamp=datetime.now(),
                strategy_type='options_spread',
                action='open',
                contracts=best_spread['legs'],
                net_premium=best_spread['net_premium'],
                max_profit=best_spread['max_profit'],
                max_loss=best_spread['max_loss'],
                probability_profit=prob_profit,
                confidence=confidence,
                spread_type=best_spread['type'],
                legs=best_spread['legs'],
                debit_credit=best_spread['debit_credit'],
                breakeven_points=breakeven_points,
                risk_reward_ratio=risk_reward,
                days_to_expiry=best_spread['days_to_expiry'],
                metrics={
                    'theta': best_spread['theta'],
                    'delta': best_spread['delta'],
                    'gamma': best_spread['gamma'],
                    'vega': best_spread['vega'],
                    'margin_required': self._calculate_margin(best_spread)
                },
                metadata={
                    'underlying_price': spot_price,
                    'spread_width': best_spread.get('width', 0)
                }
            )
            
        except Exception as e:
            return self._default_signal()
    
    def _find_vertical_spreads(self, options_chain: pd.DataFrame, 
                              spot_price: float) -> List[Dict]:
        """Find vertical spread opportunities."""
        spreads = []
        
        # Group by expiry and type
        for expiry in options_chain['expiry'].unique():
            for option_type in ['call', 'put']:
                expiry_options = options_chain[
                    (options_chain['expiry'] == expiry) & 
                    (options_chain['type'] == option_type)
                ].sort_values('strike')
                
                if len(expiry_options) < 2:
                    continue
                
                # Check adjacent strikes
                for i in range(len(expiry_options) - 1):
                    lower = expiry_options.iloc[i]
                    upper = expiry_options.iloc[i + 1]
                    
                    # Skip illiquid options
                    if (lower['volume'] < self.min_volume or 
                        upper['volume'] < self.min_volume):
                        continue
                    
                    # Calculate spread metrics
                    spread = self._create_vertical_spread(
                        lower, upper, option_type, spot_price
                    )
                    
                    if spread and self._is_valid_spread(spread):
                        spreads.append(spread)
        
        return spreads
    
    def _find_calendar_spreads(self, options_chain: pd.DataFrame,
                              spot_price: float) -> List[Dict]:
        """Find calendar spread opportunities."""
        spreads = []
        
        # Group by strike and type
        for strike in options_chain['strike'].unique():
            for option_type in ['call', 'put']:
                strike_options = options_chain[
                    (options_chain['strike'] == strike) & 
                    (options_chain['type'] == option_type)
                ].sort_values('expiry')
                
                if len(strike_options) < 2:
                    continue
                
                # Compare different expiries
                for i in range(len(strike_options) - 1):
                    near = strike_options.iloc[i]
                    far = strike_options.iloc[i + 1]
                    
                    # Skip illiquid options
                    if (near['volume'] < self.min_volume or 
                        far['volume'] < self.min_volume):
                        continue
                    
                    # Create calendar spread
                    spread = self._create_calendar_spread(
                        near, far, option_type, spot_price
                    )
                    
                    if spread and self._is_valid_spread(spread):
                        spreads.append(spread)
        
        return spreads
    
    def _find_butterfly_spreads(self, options_chain: pd.DataFrame,
                               spot_price: float) -> List[Dict]:
        """Find butterfly spread opportunities."""
        spreads = []
        
        # Group by expiry and type
        for expiry in options_chain['expiry'].unique():
            for option_type in ['call', 'put']:
                expiry_options = options_chain[
                    (options_chain['expiry'] == expiry) & 
                    (options_chain['type'] == option_type)
                ].sort_values('strike')
                
                if len(expiry_options) < 3:
                    continue
                
                # Look for equally spaced strikes
                for i in range(len(expiry_options) - 2):
                    lower = expiry_options.iloc[i]
                    middle = expiry_options.iloc[i + 1]
                    upper = expiry_options.iloc[i + 2]
                    
                    # Check if strikes are equally spaced
                    if (upper['strike'] - middle['strike'] != 
                        middle['strike'] - lower['strike']):
                        continue
                    
                    # Skip illiquid options
                    if (lower['volume'] < self.min_volume or 
                        middle['volume'] < self.min_volume or
                        upper['volume'] < self.min_volume):
                        continue
                    
                    # Create butterfly spread
                    spread = self._create_butterfly_spread(
                        lower, middle, upper, option_type, spot_price
                    )
                    
                    if spread and self._is_valid_spread(spread):
                        spreads.append(spread)
        
        return spreads
    
    def _create_vertical_spread(self, lower_option: pd.Series, upper_option: pd.Series,
                               option_type: str, spot_price: float) -> Dict:
        """Create vertical spread structure."""
        # Calculate days to expiry
        days_to_expiry = (pd.to_datetime(lower_option['expiry']) - datetime.now()).days
        
        # Bull spread (calls) or bear spread (puts)
        if option_type == 'call':
            # Bull call spread - buy lower, sell upper
            legs = [
                {
                    'action': 'buy',
                    'strike': lower_option['strike'],
                    'type': 'call',
                    'quantity': 1,
                    'premium': (lower_option['bid'] + lower_option['ask']) / 2
                },
                {
                    'action': 'sell',
                    'strike': upper_option['strike'],
                    'type': 'call',
                    'quantity': 1,
                    'premium': (upper_option['bid'] + upper_option['ask']) / 2
                }
            ]
            net_debit = legs[0]['premium'] - legs[1]['premium']
            max_profit = (upper_option['strike'] - lower_option['strike'] - net_debit) * 100
            max_loss = net_debit * 100
        else:
            # Bear put spread - buy upper, sell lower
            legs = [
                {
                    'action': 'buy',
                    'strike': upper_option['strike'],
                    'type': 'put',
                    'quantity': 1,
                    'premium': (upper_option['bid'] + upper_option['ask']) / 2
                },
                {
                    'action': 'sell',
                    'strike': lower_option['strike'],
                    'type': 'put',
                    'quantity': 1,
                    'premium': (lower_option['bid'] + lower_option['ask']) / 2
                }
            ]
            net_debit = legs[0]['premium'] - legs[1]['premium']
            max_profit = (upper_option['strike'] - lower_option['strike'] - net_debit) * 100
            max_loss = net_debit * 100
        
        # Calculate Greeks
        total_greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
        
        for i, leg in enumerate(legs):
            option = lower_option if i == 0 else upper_option
            greeks = self.calculate_greeks(
                spot_price, option['strike'], days_to_expiry,
                0.3, option['type']  # Simplified IV
            )
            
            multiplier = 1 if leg['action'] == 'buy' else -1
            for greek, value in greeks.items():
                total_greeks[greek] += value * multiplier
        
        return {
            'type': 'vertical',
            'legs': legs,
            'net_premium': net_debit,
            'debit_credit': 'debit' if net_debit > 0 else 'credit',
            'max_profit': max_profit,
            'max_loss': abs(max_loss),
            'days_to_expiry': days_to_expiry,
            'width': upper_option['strike'] - lower_option['strike'],
            **total_greeks
        }
    
    def _create_calendar_spread(self, near_option: pd.Series, far_option: pd.Series,
                               option_type: str, spot_price: float) -> Dict:
        """Create calendar spread structure."""
        # Sell near, buy far
        legs = [
            {
                'action': 'sell',
                'strike': near_option['strike'],
                'type': option_type,
                'expiry': near_option['expiry'],
                'quantity': 1,
                'premium': (near_option['bid'] + near_option['ask']) / 2
            },
            {
                'action': 'buy',
                'strike': far_option['strike'],
                'type': option_type,
                'expiry': far_option['expiry'],
                'quantity': 1,
                'premium': (far_option['bid'] + far_option['ask']) / 2
            }
        ]
        
        net_debit = legs[1]['premium'] - legs[0]['premium']
        
        # Calendar spreads have complex P&L
        # Max profit occurs when near expires at strike
        max_profit = legs[0]['premium'] * 100  # Simplified
        max_loss = net_debit * 100
        
        days_to_expiry = (pd.to_datetime(near_option['expiry']) - datetime.now()).days
        
        return {
            'type': 'calendar',
            'legs': legs,
            'net_premium': net_debit,
            'debit_credit': 'debit' if net_debit > 0 else 'credit',
            'max_profit': max_profit,
            'max_loss': abs(max_loss),
            'days_to_expiry': days_to_expiry,
            'delta': 0,  # Near neutral
            'gamma': 0,
            'theta': 0.01,  # Positive theta
            'vega': 0.02   # Positive vega
        }
    
    def _create_butterfly_spread(self, lower: pd.Series, middle: pd.Series,
                                upper: pd.Series, option_type: str,
                                spot_price: float) -> Dict:
        """Create butterfly spread structure."""
        # Buy 1 lower, sell 2 middle, buy 1 upper
        legs = [
            {
                'action': 'buy',
                'strike': lower['strike'],
                'type': option_type,
                'quantity': 1,
                'premium': (lower['bid'] + lower['ask']) / 2
            },
            {
                'action': 'sell',
                'strike': middle['strike'],
                'type': option_type,
                'quantity': 2,
                'premium': (middle['bid'] + middle['ask']) / 2
            },
            {
                'action': 'buy',
                'strike': upper['strike'],
                'type': option_type,
                'quantity': 1,
                'premium': (upper['bid'] + upper['ask']) / 2
            }
        ]
        
        net_debit = (legs[0]['premium'] + legs[2]['premium'] - 
                    2 * legs[1]['premium'])
        
        # Max profit at middle strike
        max_profit = (middle['strike'] - lower['strike'] - net_debit) * 100
        max_loss = net_debit * 100
        
        days_to_expiry = (pd.to_datetime(lower['expiry']) - datetime.now()).days
        
        return {
            'type': 'butterfly',
            'legs': legs,
            'net_premium': net_debit,
            'debit_credit': 'debit',
            'max_profit': max_profit,
            'max_loss': abs(max_loss),
            'days_to_expiry': days_to_expiry,
            'delta': 0,  # Near neutral
            'gamma': -0.01,  # Negative gamma
            'theta': 0.02,   # Positive theta
            'vega': -0.01    # Negative vega
        }
    
    def _is_valid_spread(self, spread: Dict) -> bool:
        """Check if spread meets criteria."""
        # Check minimum credit for credit spreads
        if spread['debit_credit'] == 'credit' and spread['net_premium'] < self.min_credit:
            return False
        
        # Check risk/reward ratio
        if spread['max_profit'] > 0 and spread['max_loss'] > 0:
            risk_reward = spread['max_loss'] / spread['max_profit']
            if risk_reward > self.max_risk_reward:
                return False
        
        return True
    
    def _select_best_spread(self, spreads: List[Dict]) -> Optional[Dict]:
        """Select best spread based on criteria."""
        if not spreads:
            return None
        
        # Score each spread
        scored_spreads = []
        for spread in spreads:
            score = self._score_spread(spread)
            scored_spreads.append((score, spread))
        
        # Sort by score
        scored_spreads.sort(key=lambda x: x[0], reverse=True)
        
        return scored_spreads[0][1] if scored_spreads else None
    
    def _score_spread(self, spread: Dict) -> float:
        """Score spread based on multiple criteria."""
        score = 0
        
        # Risk/reward ratio (lower is better)
        if spread['max_profit'] > 0:
            risk_reward = spread['max_loss'] / spread['max_profit']
            score += (1 / (1 + risk_reward)) * 30
        
        # Theta (positive is better for credit spreads)
        if spread['debit_credit'] == 'credit':
            score += max(0, spread['theta']) * 20
        
        # Days to expiry (prefer 20-45 days)
        if 20 <= spread['days_to_expiry'] <= 45:
            score += 20
        
        # Net credit (for credit spreads)
        if spread['debit_credit'] == 'credit':
            score += min(spread['net_premium'] * 10, 30)
        
        return score
    
    def _calculate_breakevens(self, spread: Dict) -> List[float]:
        """Calculate breakeven points for spread."""
        breakevens = []
        
        if spread['type'] == 'vertical':
            # Single breakeven
            if spread['legs'][0]['type'] == 'call':
                # Bull call spread
                breakeven = spread['legs'][0]['strike'] + spread['net_premium']
            else:
                # Bear put spread
                breakeven = spread['legs'][0]['strike'] - spread['net_premium']
            breakevens = [breakeven]
            
        elif spread['type'] == 'butterfly':
            # Two breakevens
            lower_strike = spread['legs'][0]['strike']
            middle_strike = spread['legs'][1]['strike']
            
            breakeven1 = lower_strike + spread['net_premium']
            breakeven2 = 2 * middle_strike - lower_strike - spread['net_premium']
            breakevens = [breakeven1, breakeven2]
        
        return breakevens
    
    def _calculate_risk_reward(self, spread: Dict) -> float:
        """Calculate risk/reward ratio."""
        if spread['max_profit'] > 0:
            return spread['max_loss'] / spread['max_profit']
        return float('inf')
    
    def _calculate_spread_probability(self, spread: Dict, spot_price: float) -> float:
        """Calculate probability of profit for spread."""
        # Simplified - based on moneyness and days to expiry
        if spread['type'] == 'vertical':
            strike = spread['legs'][0]['strike']
            
            # Distance from current price
            distance = abs(strike - spot_price) / spot_price
            
            # Time decay factor
            time_factor = spread['days_to_expiry'] / 365
            
            # Probability decreases with distance and time
            prob = 0.5 + (0.3 if distance < 0.05 else -0.1)
            prob *= (1 - time_factor * 0.2)
            
            return max(0, min(1, prob))
        
        return 0.5  # Default
    
    def _calculate_margin(self, spread: Dict) -> float:
        """Calculate margin requirement for spread."""
        if spread['debit_credit'] == 'debit':
            return spread['net_premium'] * 100
        else:
            # Credit spread - difference in strikes minus credit
            if spread['type'] == 'vertical':
                strike_diff = abs(spread['legs'][0]['strike'] - spread['legs'][1]['strike'])
                return (strike_diff - spread['net_premium']) * 100
        
        return 0
    
    def _calculate_confidence(self, spread: Dict, risk_reward: float, 
                            prob_profit: float) -> float:
        """Calculate confidence in spread trade."""
        # Risk/reward component
        rr_confidence = 1 / (1 + risk_reward) if risk_reward < float('inf') else 0
        
        # Probability component
        prob_confidence = prob_profit
        
        # Theta component (for credit spreads)
        theta_confidence = 0.5
        if spread['debit_credit'] == 'credit' and spread['theta'] > 0:
            theta_confidence = min(spread['theta'] * 50, 1.0)
        
        return (rr_confidence + prob_confidence + theta_confidence) / 3
    
    def _default_signal(self) -> SpreadOpportunities:
        """Return default signal when no opportunities found."""
        return SpreadOpportunities(
            timestamp=datetime.now(),
            strategy_type='options_spread',
            action='hold',
            contracts=[],
            net_premium=0,
            max_profit=0,
            max_loss=0,
            probability_profit=0,
            confidence=0,
            spread_type='none',
            legs=[],
            debit_credit='none',
            breakeven_points=[],
            risk_reward_ratio=0,
            days_to_expiry=0,
            metrics={},
            metadata={'status': 'no opportunities found'}
        )


class GammaScalpingModel(BaseOptionsModel):
    """
    Gamma scalping strategy for dynamic hedging profits.
    
    Profits from realized volatility through continuous rehedging
    of delta-neutral positions with positive gamma.
    """
    
    def __init__(self, gamma_threshold: float = 0.01,
                 delta_band: float = 0.1,
                 min_scalp_profit: float = 10):
        super().__init__()
        self.gamma_threshold = gamma_threshold
        self.delta_band = delta_band
        self.min_scalp_profit = min_scalp_profit
        self.scalp_history = []
    
    def analyze(self, position: Dict, spot_price: float, 
               price_history: pd.DataFrame) -> ScalpingSignals:
        """
        Generate gamma scalping signals.
        
        Args:
            position: Current options position with Greeks
            spot_price: Current underlying price
            price_history: Recent price history for volatility
        """
        # Calculate current portfolio Greeks
        portfolio_greeks = self._calculate_portfolio_greeks(position, spot_price)
        
        # Check if position has sufficient gamma
        if abs(portfolio_greeks['gamma']) < self.gamma_threshold:
            return self._no_scalp_signal(portfolio_greeks)
        
        # Calculate delta bands
        delta_bands = self._calculate_delta_bands(portfolio_greeks['delta'])
        
        # Check if rehedge needed
        rehedge_signal = self._check_rehedge_needed(
            portfolio_greeks['delta'], delta_bands
        )
        
        # Calculate scalping P&L
        current_pnl = self._calculate_scalping_pnl(position, spot_price)
        
        # Estimate profit from scalp
        scalp_profit = self._estimate_scalp_profit(
            portfolio_greeks, spot_price, price_history
        )
        
        # Count today's scalps
        scalps_today = self._count_todays_scalps()
        
        # Determine action
        if rehedge_signal and scalp_profit > self.min_scalp_profit:
            action = 'adjust'
            contracts = self._generate_hedge_orders(portfolio_greeks['delta'], spot_price)
            confidence = self._calculate_confidence(portfolio_greeks, scalp_profit)
        else:
            action = 'hold'
            contracts = []
            confidence = 0.5
        
        # Calculate max P&L estimates
        max_profit, max_loss = self._estimate_pnl_range(
            position, spot_price, price_history
        )
        
        # Probability of profit
        prob_profit = self._calculate_scalping_probability(
            portfolio_greeks, price_history
        )
        
        return ScalpingSignals(
            timestamp=datetime.now(),
            strategy_type='gamma_scalping',
            action=action,
            contracts=contracts,
            net_premium=0,  # Scalping doesn't involve premium
            max_profit=max_profit,
            max_loss=max_loss,
            probability_profit=prob_profit,
            confidence=confidence,
            gamma_exposure=portfolio_greeks['gamma'],
            delta_band=delta_bands,
            scalp_threshold=self.delta_band,
            current_pnl=current_pnl,
            scalps_today=scalps_today,
            rehedge_required=rehedge_signal,
            metrics={
                'portfolio_delta': portfolio_greeks['delta'],
                'portfolio_theta': portfolio_greeks['theta'],
                'portfolio_vega': portfolio_greeks['vega'],
                'realized_vol': self._calculate_realized_vol(price_history),
                'gamma_pnl': self._calculate_gamma_pnl(position, price_history)
            },
            metadata={
                'position_count': len(position.get('options', [])),
                'hedge_frequency': self._calculate_hedge_frequency()
            }
        )
    
    def _calculate_portfolio_greeks(self, position: Dict, spot_price: float) -> Dict[str, float]:
        """Calculate aggregate portfolio Greeks."""
        total_greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
        
        # Stock delta
        if 'stock' in position:
            total_greeks['delta'] += position['stock']
        
        # Options Greeks
        for option in position.get('options', []):
            days_to_expiry = (pd.to_datetime(option['expiry']) - datetime.now()).days
            
            greeks = self.calculate_greeks(
                spot_price, option['strike'], days_to_expiry,
                option.get('volatility', 0.3), option['type']
            )
            
            # Add weighted by position
            for greek, value in greeks.items():
                total_greeks[greek] += value * option['quantity'] * 100
        
        return total_greeks
    
    def _calculate_delta_bands(self, current_delta: float) -> Tuple[float, float]:
        """Calculate delta rehedge bands."""
        return (
            current_delta - self.delta_band,
            current_delta + self.delta_band
        )
    
    def _check_rehedge_needed(self, current_delta: float, 
                             delta_bands: Tuple[float, float]) -> bool:
        """Check if delta has moved outside bands."""
        return current_delta < delta_bands[0] or current_delta > delta_bands[1]
    
    def _calculate_scalping_pnl(self, position: Dict, spot_price: float) -> float:
        """Calculate current P&L from gamma scalping."""
        pnl = 0
        
        # Sum up all scalp trades
        for scalp in self.scalp_history:
            if scalp['date'].date() == datetime.now().date():
                pnl += scalp['pnl']
        
        return pnl
    
    def _estimate_scalp_profit(self, portfolio_greeks: Dict[str, float],
                             spot_price: float, price_history: pd.DataFrame) -> float:
        """Estimate profit from rehedging."""
        # Gamma P&L = 0.5 * Gamma * (price move)^2
        recent_vol = price_history['close'].pct_change().tail(20).std()
        expected_move = spot_price * recent_vol * np.sqrt(1/252)  # Daily move
        
        gamma_pnl = 0.5 * portfolio_greeks['gamma'] * (expected_move ** 2)
        
        # Subtract transaction costs
        hedge_size = abs(portfolio_greeks['delta'])
        trans_cost = hedge_size * spot_price * 0.0001  # 1 bps
        
        return gamma_pnl - trans_cost
    
    def _count_todays_scalps(self) -> int:
        """Count number of scalps done today."""
        today = datetime.now().date()
        return sum(1 for scalp in self.scalp_history if scalp['date'].date() == today)
    
    def _generate_hedge_orders(self, current_delta: float, spot_price: float) -> List[Dict]:
        """Generate orders to flatten delta."""
        hedge_quantity = -current_delta  # Opposite of current delta
        
        return [{
            'instrument': 'stock',
            'action': 'buy' if hedge_quantity > 0 else 'sell',
            'quantity': abs(round(hedge_quantity)),
            'price': spot_price
        }]
    
    def _estimate_pnl_range(self, position: Dict, spot_price: float,
                           price_history: pd.DataFrame) -> Tuple[float, float]:
        """Estimate P&L range from gamma scalping."""
        # Based on realized volatility
        realized_vol = self._calculate_realized_vol(price_history)
        
        # Daily gamma P&L estimate
        daily_move = spot_price * realized_vol * np.sqrt(1/252)
        gamma = self._calculate_portfolio_greeks(position, spot_price)['gamma']
        
        daily_gamma_pnl = 0.5 * gamma * (daily_move ** 2)
        
        # Theta decay
        theta = self._calculate_portfolio_greeks(position, spot_price)['theta']
        
        # Over remaining life of options
        days_remaining = min(option.get('days_to_expiry', 30) 
                           for option in position.get('options', [{'days_to_expiry': 30}]))
        
        max_profit = daily_gamma_pnl * days_remaining * 2  # Optimistic
        max_loss = theta * days_remaining  # Theta decay
        
        return max_profit, abs(max_loss)
    
    def _calculate_scalping_probability(self, portfolio_greeks: Dict[str, float],
                                      price_history: pd.DataFrame) -> float:
        """Calculate probability of profitable gamma scalping."""
        # Based on gamma/theta ratio and realized vs implied vol
        gamma_theta_ratio = abs(portfolio_greeks['gamma'] / (portfolio_greeks['theta'] + 1e-6))
        
        # Higher ratio = better chance
        if gamma_theta_ratio > 0.1:
            base_prob = 0.6
        elif gamma_theta_ratio > 0.05:
            base_prob = 0.5
        else:
            base_prob = 0.4
        
        # Adjust for volatility environment
        realized_vol = self._calculate_realized_vol(price_history)
        if realized_vol > 0.3:  # High vol environment
            base_prob += 0.1
        
        return min(0.8, base_prob)
    
    def _calculate_realized_vol(self, price_history: pd.DataFrame) -> float:
        """Calculate realized volatility."""
        returns = price_history['close'].pct_change().dropna()
        return returns.tail(20).std() * np.sqrt(252)
    
    def _calculate_gamma_pnl(self, position: Dict, price_history: pd.DataFrame) -> float:
        """Calculate historical gamma P&L."""
        # Simplified - would need position history in practice
        returns = price_history['close'].pct_change().dropna().tail(20)
        price_moves = price_history['close'].diff().dropna().tail(20)
        
        avg_gamma = 0.01  # Placeholder
        gamma_pnl = 0.5 * avg_gamma * (price_moves ** 2).sum()
        
        return gamma_pnl
    
    def _calculate_hedge_frequency(self) -> float:
        """Calculate average hedging frequency."""
        if not self.scalp_history:
            return 0
        
        # Scalps per day
        days = (self.scalp_history[-1]['date'] - self.scalp_history[0]['date']).days + 1
        return len(self.scalp_history) / max(days, 1)
    
    def _calculate_confidence(self, portfolio_greeks: Dict[str, float],
                            scalp_profit: float) -> float:
        """Calculate confidence in gamma scalping."""
        # Gamma strength
        gamma_confidence = min(abs(portfolio_greeks['gamma']) * 10, 1.0)
        
        # Profit potential
        profit_confidence = min(scalp_profit / 50, 1.0)  # $50 as baseline
        
        # Theta drag
        theta_impact = abs(portfolio_greeks['theta'] / scalp_profit) if scalp_profit > 0 else 1
        theta_confidence = 1 / (1 + theta_impact)
        
        return (gamma_confidence + profit_confidence + theta_confidence) / 3
    
    def _no_scalp_signal(self, portfolio_greeks: Dict[str, float]) -> ScalpingSignals:
        """Return signal when no scalping action needed."""
        return ScalpingSignals(
            timestamp=datetime.now(),
            strategy_type='gamma_scalping',
            action='hold',
            contracts=[],
            net_premium=0,
            max_profit=0,
            max_loss=0,
            probability_profit=0.5,
            confidence=0.5,
            gamma_exposure=portfolio_greeks['gamma'],
            delta_band=(portfolio_greeks['delta'] - self.delta_band,
                       portfolio_greeks['delta'] + self.delta_band),
            scalp_threshold=self.delta_band,
            current_pnl=self._calculate_scalping_pnl({}, 0),
            scalps_today=self._count_todays_scalps(),
            rehedge_required=False,
            metrics=portfolio_greeks,
            metadata={'status': 'within delta bands'}
        )


class IronCondorModel(BaseOptionsModel):
    """
    Iron condor strategy for range-bound markets.
    
    Sells OTM call and put spreads to collect premium when
    expecting low volatility and range-bound price action.
    """
    
    def __init__(self, min_credit: float = 0.3,
                 min_width: float = 5,
                 target_delta: float = 0.15,
                 target_days: int = 45):
        super().__init__()
        self.min_credit = min_credit
        self.min_width = min_width
        self.target_delta = target_delta
        self.target_days = target_days
    
    def analyze(self, options_chain: pd.DataFrame, spot_price: float,
               volatility_forecast: Optional[float] = None) -> IronCondorTrade:
        """
        Find optimal iron condor setup.
        
        Args:
            options_chain: Available options
            spot_price: Current underlying price
            volatility_forecast: Expected volatility (optional)
        """
        if not self.validate_data(options_chain):
            return self._default_signal()
        
        try:
            # Find optimal expiry
            optimal_expiry = self._find_optimal_expiry(options_chain)
            
            if optimal_expiry is None:
                return self._default_signal()
            
            # Filter to optimal expiry
            expiry_chain = options_chain[options_chain['expiry'] == optimal_expiry]
            
            # Find optimal strikes
            strikes = self._find_optimal_strikes(
                expiry_chain, spot_price, volatility_forecast
            )
            
            if not self._validate_strikes(strikes):
                return self._default_signal()
            
            # Calculate condor metrics
            condor_metrics = self._calculate_condor_metrics(
                strikes, expiry_chain, spot_price
            )
            
            # Calculate profit range
            profit_range = (strikes['short_put'], strikes['short_call'])
            
            # Calculate adjustment points
            adjustment_points = self._calculate_adjustment_points(
                strikes, condor_metrics['credit']
            )
            
            # Probability of profit
            prob_profit = self._calculate_condor_probability(
                strikes, spot_price, expiry_chain, volatility_forecast
            )
            
            confidence = self._calculate_confidence(
                condor_metrics, prob_profit, strikes
            )
            
            # Build contracts list
            contracts = self._build_condor_contracts(strikes, expiry_chain)
            
            return IronCondorTrade(
                timestamp=datetime.now(),
                strategy_type='iron_condor',
                action='open',
                contracts=contracts,
                net_premium=condor_metrics['credit'],
                max_profit=condor_metrics['max_profit'],
                max_loss=condor_metrics['max_loss'],
                probability_profit=prob_profit,
                confidence=confidence,
                short_call_strike=strikes['short_call'],
                long_call_strike=strikes['long_call'],
                short_put_strike=strikes['short_put'],
                long_put_strike=strikes['long_put'],
                credit_received=condor_metrics['credit'],
                margin_required=condor_metrics['margin'],
                profit_range=profit_range,
                adjustment_points=adjustment_points,
                metrics={
                    'delta': condor_metrics['delta'],
                    'theta': condor_metrics['theta'],
                    'vega': condor_metrics['vega'],
                    'profit_range_width': profit_range[1] - profit_range[0],
                    'days_to_expiry': condor_metrics['days_to_expiry']
                },
                metadata={
                    'underlying_price': spot_price,
                    'expiry': optimal_expiry,
                    'wing_width': strikes['long_call'] - strikes['short_call']
                }
            )
            
        except Exception as e:
            return self._default_signal()
    
    def _find_optimal_expiry(self, options_chain: pd.DataFrame) -> Optional[str]:
        """Find expiry closest to target days."""
        expiries = options_chain['expiry'].unique()
        
        best_expiry = None
        best_diff = float('inf')
        
        for expiry in expiries:
            days = (pd.to_datetime(expiry) - datetime.now()).days
            
            if abs(days - self.target_days) < best_diff:
                best_diff = abs(days - self.target_days)
                best_expiry = expiry
        
        return best_expiry
    
    def _find_optimal_strikes(self, expiry_chain: pd.DataFrame, spot_price: float,
                             volatility_forecast: Optional[float]) -> Dict[str, float]:
        """Find optimal strikes for iron condor."""
        # Calculate expected move
        days_to_expiry = (pd.to_datetime(expiry_chain.iloc[0]['expiry']) - datetime.now()).days
        
        if volatility_forecast:
            expected_move = spot_price * volatility_forecast * np.sqrt(days_to_expiry / 365)
        else:
            # Use ATM implied volatility
            atm_iv = self._get_atm_iv(expiry_chain, spot_price)
            expected_move = spot_price * atm_iv * np.sqrt(days_to_expiry / 365)
        
        # Find strikes around target delta
        call_strikes = expiry_chain[expiry_chain['type'] == 'call'].sort_values('strike')
        put_strikes = expiry_chain[expiry_chain['type'] == 'put'].sort_values('strike', ascending=False)
        
        # Short call strike (OTM)
        short_call = None
        for _, call in call_strikes.iterrows():
            if call['strike'] > spot_price:
                # Calculate delta
                greeks = self.calculate_greeks(
                    spot_price, call['strike'], days_to_expiry,
                    volatility_forecast or 0.3, 'call'
                )
                
                if abs(greeks['delta']) <= self.target_delta:
                    short_call = call['strike']
                    break
        
        # Short put strike (OTM)
        short_put = None
        for _, put in put_strikes.iterrows():
            if put['strike'] < spot_price:
                # Calculate delta
                greeks = self.calculate_greeks(
                    spot_price, put['strike'], days_to_expiry,
                    volatility_forecast or 0.3, 'put'
                )
                
                if abs(greeks['delta']) <= self.target_delta:
                    short_put = put['strike']
                    break
        
        if short_call is None or short_put is None:
            return {}
        
        # Find long strikes (further OTM)
        # Typically use same width wings
        wing_width = self._find_optimal_wing_width(expiry_chain, short_call, short_put)
        
        return {
            'short_call': short_call,
            'long_call': short_call + wing_width,
            'short_put': short_put,
            'long_put': short_put - wing_width
        }
    
    def _find_optimal_wing_width(self, expiry_chain: pd.DataFrame,
                                short_call: float, short_put: float) -> float:
        """Find optimal wing width for condor."""
        # Get available strikes
        strikes = sorted(expiry_chain['strike'].unique())
        
        # Find strike intervals
        intervals = []
        for i in range(len(strikes) - 1):
            intervals.append(strikes[i + 1] - strikes[i])
        
        # Use most common interval or minimum width
        if intervals:
            common_interval = max(set(intervals), key=intervals.count)
            return max(common_interval, self.min_width)
        
        return self.min_width
    
    def _validate_strikes(self, strikes: Dict[str, float]) -> bool:
        """Validate iron condor strikes."""
        if not strikes:
            return False
        
        # Check all strikes present
        required = ['short_call', 'long_call', 'short_put', 'long_put']
        if not all(k in strikes for k in required):
            return False
        
        # Check strike order
        if not (strikes['long_put'] < strikes['short_put'] < 
                strikes['short_call'] < strikes['long_call']):
            return False
        
        # Check minimum width
        if (strikes['long_call'] - strikes['short_call'] < self.min_width or
            strikes['short_put'] - strikes['long_put'] < self.min_width):
            return False
        
        return True
    
    def _calculate_condor_metrics(self, strikes: Dict[str, float],
                                 expiry_chain: pd.DataFrame,
                                 spot_price: float) -> Dict[str, float]:
        """Calculate iron condor metrics."""
        # Get option prices
        call_spread_credit = self._get_spread_credit(
            expiry_chain, strikes['short_call'], strikes['long_call'], 'call'
        )
        
        put_spread_credit = self._get_spread_credit(
            expiry_chain, strikes['short_put'], strikes['long_put'], 'put'
        )
        
        total_credit = call_spread_credit + put_spread_credit
        
        # Max profit = total credit
        max_profit = total_credit * 100
        
        # Max loss = wing width - credit
        wing_width = strikes['long_call'] - strikes['short_call']
        max_loss = (wing_width - total_credit) * 100
        
        # Margin = max loss
        margin = max_loss
        
        # Calculate Greeks
        days_to_expiry = (pd.to_datetime(expiry_chain.iloc[0]['expiry']) - datetime.now()).days
        
        # Net Greeks (simplified)
        greeks = {
            'delta': 0,  # Should be near zero
            'theta': total_credit * 0.02,  # Positive theta
            'vega': -total_credit * 0.5,   # Negative vega
            'gamma': -0.01  # Negative gamma
        }
        
        return {
            'credit': total_credit,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'margin': margin,
            'days_to_expiry': days_to_expiry,
            **greeks
        }
    
    def _get_spread_credit(self, expiry_chain: pd.DataFrame,
                          short_strike: float, long_strike: float,
                          option_type: str) -> float:
        """Calculate credit for a spread."""
        # Get short option
        short_option = expiry_chain[
            (expiry_chain['strike'] == short_strike) & 
            (expiry_chain['type'] == option_type)
        ]
        
        # Get long option
        long_option = expiry_chain[
            (expiry_chain['strike'] == long_strike) & 
            (expiry_chain['type'] == option_type)
        ]
        
        if short_option.empty or long_option.empty:
            return 0
        
        # Credit = short premium - long premium
        short_mid = (short_option.iloc[0]['bid'] + short_option.iloc[0]['ask']) / 2
        long_mid = (long_option.iloc[0]['bid'] + long_option.iloc[0]['ask']) / 2
        
        return short_mid - long_mid
    
    def _calculate_adjustment_points(self, strikes: Dict[str, float],
                                   credit: float) -> List[float]:
        """Calculate price points for adjustments."""
        # Common adjustment at 50% credit loss
        # This occurs when price moves to short strikes
        return [
            strikes['short_put'],
            strikes['short_call']
        ]
    
    def _calculate_condor_probability(self, strikes: Dict[str, float],
                                    spot_price: float,
                                    expiry_chain: pd.DataFrame,
                                    volatility_forecast: Optional[float]) -> float:
        """Calculate probability of profit for iron condor."""
        days_to_expiry = (pd.to_datetime(expiry_chain.iloc[0]['expiry']) - datetime.now()).days
        
        # Use forecast or ATM IV
        if volatility_forecast:
            vol = volatility_forecast
        else:
            vol = self._get_atm_iv(expiry_chain, spot_price)
        
        # Probability of staying between short strikes
        # Using normal distribution approximation
        time_factor = np.sqrt(days_to_expiry / 365)
        
        # Upper probability
        upper_move = (strikes['short_call'] - spot_price) / spot_price
        upper_z = upper_move / (vol * time_factor)
        prob_below_call = norm.cdf(upper_z)
        
        # Lower probability
        lower_move = (strikes['short_put'] - spot_price) / spot_price
        lower_z = lower_move / (vol * time_factor)
        prob_above_put = 1 - norm.cdf(lower_z)
        
        # Combined probability
        prob_profit = prob_below_call - (1 - prob_above_put)
        
        return max(0, min(1, prob_profit))
    
    def _get_atm_iv(self, expiry_chain: pd.DataFrame, spot_price: float) -> float:
        """Get at-the-money implied volatility."""
        # Find ATM strike
        strikes = expiry_chain['strike'].unique()
        atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
        
        # Get ATM call
        atm_option = expiry_chain[
            (expiry_chain['strike'] == atm_strike) & 
            (expiry_chain['type'] == 'call')
        ]
        
        if atm_option.empty:
            return 0.3  # Default
        
        # Calculate IV
        mid_price = (atm_option.iloc[0]['bid'] + atm_option.iloc[0]['ask']) / 2
        days_to_expiry = (pd.to_datetime(atm_option.iloc[0]['expiry']) - datetime.now()).days
        
        iv = self.calculate_implied_volatility(
            mid_price, spot_price, atm_strike, days_to_expiry, 'call'
        )
        
        return iv
    
    def _build_condor_contracts(self, strikes: Dict[str, float],
                               expiry_chain: pd.DataFrame) -> List[Dict]:
        """Build contract list for iron condor."""
        contracts = []
        
        # Sell call spread
        contracts.append({
            'action': 'sell',
            'strike': strikes['short_call'],
            'type': 'call',
            'quantity': 1
        })
        contracts.append({
            'action': 'buy',
            'strike': strikes['long_call'],
            'type': 'call',
            'quantity': 1
        })
        
        # Sell put spread
        contracts.append({
            'action': 'sell',
            'strike': strikes['short_put'],
            'type': 'put',
            'quantity': 1
        })
        contracts.append({
            'action': 'buy',
            'strike': strikes['long_put'],
            'type': 'put',
            'quantity': 1
        })
        
        return contracts
    
    def _calculate_confidence(self, condor_metrics: Dict[str, float],
                            prob_profit: float,
                            strikes: Dict[str, float]) -> float:
        """Calculate confidence in iron condor trade."""
        # Probability component
        prob_confidence = prob_profit
        
        # Credit component (higher credit = better)
        credit_confidence = min(condor_metrics['credit'] / 1.0, 1.0)
        
        # Risk/reward component
        rr_ratio = condor_metrics['max_loss'] / condor_metrics['max_profit']
        rr_confidence = 1 / (1 + rr_ratio)
        
        # Width component (wider profit range = better)
        range_width = strikes['short_call'] - strikes['short_put']
        width_confidence = min(range_width / (strikes['short_call'] * 0.2), 1.0)
        
        return (prob_confidence + credit_confidence + rr_confidence + width_confidence) / 4
    
    def _default_signal(self) -> IronCondorTrade:
        """Return default signal when no opportunities found."""
        return IronCondorTrade(
            timestamp=datetime.now(),
            strategy_type='iron_condor',
            action='hold',
            contracts=[],
            net_premium=0,
            max_profit=0,
            max_loss=0,
            probability_profit=0,
            confidence=0,
            short_call_strike=0,
            long_call_strike=0,
            short_put_strike=0,
            long_put_strike=0,
            credit_received=0,
            margin_required=0,
            profit_range=(0, 0),
            adjustment_points=[],
            metrics={},
            metadata={'status': 'no opportunities found'}
        )


class VolatilityArbitrageModel(BaseOptionsModel):
    """
    Volatility Arbitrage Model - Trades discrepancies between implied and realized volatility
    
    Features:
    - Implied volatility surface analysis
    - Historical volatility calculation and forecasting
    - Statistical arbitrage opportunities
    - Delta-neutral portfolio construction
    - Risk management with vega limits
    """
    
    def __init__(self, iv_lookback: int = 252, realized_lookback: int = 30,
                 vol_forecast_window: int = 20, confidence_threshold: float = 0.7,
                 min_edge: float = 0.05):
        """
        Initialize Volatility Arbitrage Model
        
        Args:
            iv_lookback: Days for IV percentile calculation
            realized_lookback: Days for realized vol calculation
            vol_forecast_window: Forward window for vol forecast
            confidence_threshold: Minimum confidence for trades
            min_edge: Minimum vol edge to trade (5% default)
        """
        super().__init__(confidence_threshold=confidence_threshold)
        self.iv_lookback = iv_lookback
        self.realized_lookback = realized_lookback
        self.vol_forecast_window = vol_forecast_window
        self.min_edge = min_edge
        
    def calculate_realized_volatility(self, prices: pd.Series, 
                                    lookback: Optional[int] = None) -> float:
        """Calculate annualized realized volatility"""
        if lookback is None:
            lookback = self.realized_lookback
            
        if len(prices) < lookback:
            return 0.0
            
        returns = prices.pct_change().dropna()
        if len(returns) < lookback:
            return 0.0
            
        # Calculate realized vol
        realized_vol = returns.tail(lookback).std() * np.sqrt(252)
        return realized_vol
        
    def forecast_volatility(self, prices: pd.Series) -> Tuple[float, float]:
        """
        Forecast future volatility using GARCH-like approach
        
        Returns:
            (forecast_vol, confidence)
        """
        if len(prices) < self.realized_lookback * 2:
            return 0.0, 0.0
            
        returns = prices.pct_change().dropna()
        
        # Simple EWMA forecast
        vol_series = returns.rolling(self.realized_lookback).std() * np.sqrt(252)
        vol_series = vol_series.dropna()
        
        if len(vol_series) < 10:
            return 0.0, 0.0
            
        # Exponential weighting for forecast
        weights = np.exp(np.linspace(-2, 0, len(vol_series)))
        weights /= weights.sum()
        
        forecast_vol = np.sum(vol_series.values * weights)
        
        # Confidence based on volatility stability
        vol_std = vol_series.std()
        vol_mean = vol_series.mean()
        confidence = 1.0 / (1.0 + vol_std / vol_mean) if vol_mean > 0 else 0.0
        
        return forecast_vol, confidence
        
    def analyze_iv_surface(self, options_chain: pd.DataFrame) -> Dict[str, Any]:
        """Analyze implied volatility surface for anomalies"""
        if options_chain.empty:
            return {'atm_iv': 0.0, 'iv_skew': 0.0, 'term_structure': []}
            
        analysis = {}
        
        # ATM IV
        spot_price = options_chain['underlying_price'].iloc[0]
        atm_options = options_chain[
            (np.abs(options_chain['strike'] - spot_price) / spot_price < 0.02)
        ]
        
        if not atm_options.empty:
            analysis['atm_iv'] = atm_options['implied_volatility'].mean()
        else:
            analysis['atm_iv'] = 0.0
            
        # IV Skew (25-delta put vs call)
        # Simplified: use 5% OTM options
        otm_puts = options_chain[
            (options_chain['type'] == 'put') &
            (options_chain['strike'] < spot_price * 0.95)
        ]
        otm_calls = options_chain[
            (options_chain['type'] == 'call') &
            (options_chain['strike'] > spot_price * 1.05)
        ]
        
        if not otm_puts.empty and not otm_calls.empty:
            analysis['iv_skew'] = (
                otm_puts['implied_volatility'].mean() - 
                otm_calls['implied_volatility'].mean()
            )
        else:
            analysis['iv_skew'] = 0.0
            
        # Term structure
        term_structure = []
        for expiry in options_chain['expiration'].unique():
            exp_options = options_chain[options_chain['expiration'] == expiry]
            if not exp_options.empty:
                term_structure.append({
                    'expiry': expiry,
                    'iv': exp_options['implied_volatility'].mean(),
                    'dte': exp_options['days_to_expiry'].iloc[0]
                })
                
        analysis['term_structure'] = sorted(term_structure, key=lambda x: x['dte'])
        
        return analysis
        
    def identify_arbitrage_opportunity(self, spot_prices: pd.Series,
                                     options_chain: pd.DataFrame) -> OptionsSignal:
        """Identify volatility arbitrage opportunities"""
        # Default signal
        default_signal = OptionsSignal(
            signal_type='vol_arb',
            direction=0,
            entry_price=0.0,
            strikes={},
            expiration=None,
            strategy_details={},
            greeks={},
            confidence=0.0,
            metadata={'reason': 'insufficient_data'}
        )
        
        if len(spot_prices) < self.realized_lookback * 2 or options_chain.empty:
            return default_signal
            
        # Calculate realized vol
        realized_vol = self.calculate_realized_volatility(spot_prices)
        
        # Forecast future vol
        forecast_vol, forecast_confidence = self.forecast_volatility(spot_prices)
        
        # Analyze IV surface
        iv_analysis = self.analyze_iv_surface(options_chain)
        atm_iv = iv_analysis['atm_iv']
        
        if atm_iv == 0 or realized_vol == 0:
            return default_signal
            
        # Calculate vol edge
        iv_premium = atm_iv - forecast_vol
        vol_ratio = atm_iv / realized_vol if realized_vol > 0 else 0
        
        # Determine trade direction
        # Positive edge: IV > forecast vol (sell vol)
        # Negative edge: IV < forecast vol (buy vol)
        if abs(iv_premium) < self.min_edge:
            return OptionsSignal(
                signal_type='vol_arb',
                direction=0,
                entry_price=spot_prices.iloc[-1],
                strikes={},
                expiration=None,
                strategy_details={'edge_too_small': iv_premium},
                greeks={},
                confidence=0.0,
                metadata={'reason': 'insufficient_edge'}
            )
            
        # Find optimal strikes for delta-neutral position
        spot_price = spot_prices.iloc[-1]
        
        # Select near-ATM options with 30-45 DTE
        suitable_options = options_chain[
            (options_chain['days_to_expiry'] >= 30) &
            (options_chain['days_to_expiry'] <= 45) &
            (np.abs(options_chain['strike'] - spot_price) / spot_price < 0.05)
        ]
        
        if suitable_options.empty:
            return default_signal
            
        # Select best expiration
        best_expiry = suitable_options.groupby('expiration')['volume'].sum().idxmax()
        exp_options = suitable_options[suitable_options['expiration'] == best_expiry]
        
        # Build delta-neutral straddle/strangle
        direction = -1 if iv_premium > 0 else 1  # Sell if IV high, buy if IV low
        
        # Find ATM strike
        atm_strike = exp_options.iloc[
            (exp_options['strike'] - spot_price).abs().argsort()[:1]
        ]['strike'].iloc[0]
        
        # Calculate position Greeks
        atm_call = exp_options[
            (exp_options['strike'] == atm_strike) & 
            (exp_options['type'] == 'call')
        ]
        atm_put = exp_options[
            (exp_options['strike'] == atm_strike) & 
            (exp_options['type'] == 'put')
        ]
        
        if atm_call.empty or atm_put.empty:
            return default_signal
            
        # Position details
        strategy_details = {
            'strategy': 'straddle' if direction == 1 else 'short_straddle',
            'realized_vol': realized_vol,
            'forecast_vol': forecast_vol,
            'implied_vol': atm_iv,
            'vol_premium': iv_premium,
            'vol_ratio': vol_ratio,
            'iv_skew': iv_analysis['iv_skew'],
            'forecast_confidence': forecast_confidence,
            'expected_edge': abs(iv_premium),
            'max_profit': None,  # Unlimited for long vol
            'max_loss': None if direction == 1 else (
                atm_call['ask'].iloc[0] + atm_put['ask'].iloc[0]
            ) * 100,
            'breakeven_upper': atm_strike + (
                atm_call['ask'].iloc[0] + atm_put['ask'].iloc[0]
            ),
            'breakeven_lower': atm_strike - (
                atm_call['ask'].iloc[0] + atm_put['ask'].iloc[0]
            )
        }
        
        # Aggregate Greeks (delta-neutral by construction)
        greeks = {
            'delta': 0.0,  # Delta neutral
            'gamma': (atm_call['gamma'].iloc[0] + atm_put['gamma'].iloc[0]) * direction,
            'vega': (atm_call['vega'].iloc[0] + atm_put['vega'].iloc[0]) * direction,
            'theta': (atm_call['theta'].iloc[0] + atm_put['theta'].iloc[0]) * direction,
            'rho': (atm_call['rho'].iloc[0] + atm_put['rho'].iloc[0]) * direction
        }
        
        # Calculate confidence
        confidence = min(
            forecast_confidence * 0.4 +  # Forecast quality
            min(abs(iv_premium) / 0.10, 1.0) * 0.3 +  # Edge size
            min(vol_ratio / 1.5, 1.0) * 0.3,  # Historical relationship
            self.confidence_threshold * 1.2  # Cap at threshold * 1.2
        )
        
        return OptionsSignal(
            signal_type='vol_arb',
            direction=direction,
            entry_price=spot_prices.iloc[-1],
            strikes={
                'straddle': atm_strike,
                'call': atm_strike,
                'put': atm_strike
            },
            expiration=best_expiry,
            strategy_details=strategy_details,
            greeks=greeks,
            confidence=confidence,
            metadata={
                'iv_percentile': self._calculate_iv_percentile(
                    options_chain, atm_iv
                ),
                'term_structure': iv_analysis['term_structure'],
                'entry_iv': atm_iv,
                'target_vol': forecast_vol,
                'vol_regime': 'high' if vol_ratio > 1.2 else 'normal',
                'days_to_expiry': exp_options['days_to_expiry'].iloc[0]
            }
        )
        
    def _calculate_iv_percentile(self, options_chain: pd.DataFrame, 
                                current_iv: float) -> float:
        """Calculate IV percentile rank"""
        # This would need historical IV data
        # For now, use current chain distribution
        all_ivs = options_chain['implied_volatility'].dropna()
        if all_ivs.empty:
            return 0.5
            
        return (all_ivs < current_iv).sum() / len(all_ivs)
        
    def calculate_signal(self, data: pd.DataFrame, 
                        options_chain: Optional[pd.DataFrame] = None) -> OptionsSignal:
        """Main signal calculation"""
        if options_chain is None:
            return OptionsSignal(
                signal_type='vol_arb',
                direction=0,
                entry_price=0.0,
                strikes={},
                expiration=None,
                strategy_details={},
                greeks={},
                confidence=0.0,
                metadata={'reason': 'no_options_data'}
            )
            
        # Validate data
        data = self.validate_data(data)
        if data is None or len(data) < self.realized_lookback * 2:
            return OptionsSignal(
                signal_type='vol_arb',
                direction=0,
                entry_price=0.0,
                strikes={},
                expiration=None,
                strategy_details={},
                greeks={},
                confidence=0.0,
                metadata={'reason': 'insufficient_data'}
            )
            
        # Get close prices
        close_prices = data['close'] if 'close' in data.columns else data.iloc[:, 0]
        
        return self.identify_arbitrage_opportunity(close_prices, options_chain)


# Model factory for easy instantiation
def create_options_model(model_type: str, **kwargs) -> BaseOptionsModel:
    """Factory function to create options models."""
    models = {
        'delta_neutral': DeltaNeutralModel,
        'volatility_arbitrage': VolatilityArbitrageModel,
        'spreads': OptionsSpreadsModel,
        'gamma_scalping': GammaScalpingModel,
        'iron_condor': IronCondorModel
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return models[model_type](**kwargs)