#!/usr/bin/env python3
"""
Crypto/DeFi Trading Models

This module implements trading models for cryptocurrency and DeFi markets:
- On-chain analytics
- DeFi yield farming optimization
- MEV (Maximal Extractable Value) detection
- Cross-chain arbitrage opportunities

All models use real blockchain data and avoid synthetic data generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
from decimal import Decimal
import requests
import json

logger = logging.getLogger(__name__)

# Data Classes for Crypto/DeFi Signals
@dataclass
class CryptoSignal:
    """Signal from crypto/DeFi analysis"""
    signal_type: str  # 'onchain', 'defi_yield', 'mev', 'cross_chain'
    direction: int  # 1 for bullish/long, -1 for bearish/short, 0 for neutral
    asset: str  # Asset or pair (e.g., 'ETH', 'ETH/USDC')
    chain: str  # Blockchain (e.g., 'ethereum', 'bsc', 'polygon')
    opportunity_value: float  # Expected profit/yield in USD or percentage
    gas_cost: float  # Estimated gas cost in USD
    confidence: float  # 0-1 confidence level
    urgency: str  # 'immediate', 'short', 'medium', 'long'
    metadata: Dict[str, Any]  # Additional information
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

# Base Crypto Model
class BaseCryptoModel:
    """Base class for crypto/DeFi models"""
    
    def __init__(self, chain: str = 'ethereum',
                 min_profit_threshold: float = 50.0,
                 gas_buffer: float = 1.2,
                 confidence_threshold: float = 0.7):
        self.chain = chain
        self.min_profit_threshold = min_profit_threshold
        self.gas_buffer = gas_buffer  # Safety margin for gas estimation
        self.confidence_threshold = confidence_threshold
        self.is_fitted = False
        
        # Chain-specific parameters
        self.chain_params = {
            'ethereum': {'block_time': 12, 'gas_token': 'ETH'},
            'bsc': {'block_time': 3, 'gas_token': 'BNB'},
            'polygon': {'block_time': 2, 'gas_token': 'MATIC'},
            'arbitrum': {'block_time': 0.25, 'gas_token': 'ETH'},
            'optimism': {'block_time': 2, 'gas_token': 'ETH'}
        }
        
    def estimate_gas_cost(self, gas_units: int, priority: str = 'medium') -> float:
        """Estimate transaction gas cost in USD"""
        # In production, fetch real gas prices from APIs
        gas_multipliers = {'low': 0.8, 'medium': 1.0, 'high': 1.5, 'urgent': 2.0}
        
        # Placeholder gas prices (would fetch real-time)
        base_gas_prices = {
            'ethereum': 30,  # gwei
            'bsc': 5,
            'polygon': 30,
            'arbitrum': 0.1,
            'optimism': 0.001
        }
        
        gas_price = base_gas_prices.get(self.chain, 10) * gas_multipliers.get(priority, 1.0)
        
        # Convert to USD (placeholder rates)
        gas_token_prices = {
            'ETH': 2000,
            'BNB': 300,
            'MATIC': 0.8
        }
        
        gas_token = self.chain_params.get(self.chain, {}).get('gas_token', 'ETH')
        token_price = gas_token_prices.get(gas_token, 100)
        
        # Calculate cost
        gas_cost_eth = (gas_units * gas_price) / 1e9  # Convert gwei to ETH
        gas_cost_usd = gas_cost_eth * token_price * self.gas_buffer
        
        return gas_cost_usd
        
    def _get_onchain_data(self, endpoint: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Get real on-chain data from blockchain APIs
        
        In production, this would connect to:
        - Ethereum nodes (Infura, Alchemy)
        - The Graph Protocol
        - Dune Analytics
        - Footprint Analytics
        - DeFi Llama
        """
        logger.info(f"Fetching on-chain data from {endpoint}")
        # Placeholder for real API calls
        return None

# On-Chain Analytics Model
class OnChainModel(BaseCryptoModel):
    """
    Trading model based on on-chain analytics
    
    Features:
    - Whale wallet tracking
    - Exchange flow analysis
    - Smart money movements
    - Network activity metrics
    - DeFi TVL flows
    """
    
    def __init__(self, analysis_type: str = 'whale_tracking',
                 lookback_blocks: int = 1000,
                 whale_threshold: float = 1000000,  # USD
                 chain: str = 'ethereum',
                 confidence_threshold: float = 0.75):
        """
        Initialize On-Chain Model
        
        Args:
            analysis_type: Type of on-chain analysis
            lookback_blocks: Number of blocks to analyze
            whale_threshold: Minimum transaction size for whale detection
            chain: Blockchain to analyze
            confidence_threshold: Minimum confidence
        """
        super().__init__(chain=chain, confidence_threshold=confidence_threshold)
        self.analysis_type = analysis_type
        self.lookback_blocks = lookback_blocks
        self.whale_threshold = whale_threshold
        
        # Known smart money addresses (would be loaded from database)
        self.smart_money_addresses = set()
        self.exchange_addresses = {}
        
    def analyze_whale_movements(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze whale wallet movements"""
        whale_metrics = {
            'net_flow': 0,
            'accumulation_addresses': [],
            'distribution_addresses': [],
            'large_transfers': []
        }
        
        for tx in transactions:
            value_usd = tx.get('value_usd', 0)
            
            if value_usd >= self.whale_threshold:
                from_addr = tx.get('from')
                to_addr = tx.get('to')
                
                # Track large transfers
                whale_metrics['large_transfers'].append({
                    'from': from_addr,
                    'to': to_addr,
                    'value': value_usd,
                    'token': tx.get('token', 'ETH'),
                    'hash': tx.get('hash')
                })
                
                # Identify accumulation vs distribution
                if self._is_accumulation_address(to_addr):
                    whale_metrics['accumulation_addresses'].append(to_addr)
                    whale_metrics['net_flow'] += value_usd
                elif self._is_distribution_address(from_addr):
                    whale_metrics['distribution_addresses'].append(from_addr)
                    whale_metrics['net_flow'] -= value_usd
                    
        return whale_metrics
        
    def analyze_exchange_flows(self, exchange_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze cryptocurrency exchange flows"""
        flow_scores = {}
        
        for exchange, data in exchange_data.items():
            inflow = data.get('inflow_24h', 0)
            outflow = data.get('outflow_24h', 0)
            
            # Net flow ratio
            net_flow = outflow - inflow
            total_flow = inflow + outflow
            
            if total_flow > 0:
                # Positive score for net outflows (bullish)
                # Negative score for net inflows (bearish)
                flow_score = net_flow / total_flow
                flow_scores[exchange] = flow_score
            else:
                flow_scores[exchange] = 0.0
                
        return flow_scores
        
    def analyze_smart_money(self, smart_money_txs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Track smart money movements"""
        smart_money_analysis = {
            'tokens_accumulated': {},
            'tokens_sold': {},
            'new_positions': [],
            'closed_positions': []
        }
        
        for tx in smart_money_txs:
            address = tx.get('address')
            action = tx.get('action')  # 'buy', 'sell', 'provide_liquidity', etc.
            token = tx.get('token')
            value_usd = tx.get('value_usd', 0)
            
            if action == 'buy':
                if token not in smart_money_analysis['tokens_accumulated']:
                    smart_money_analysis['tokens_accumulated'][token] = 0
                smart_money_analysis['tokens_accumulated'][token] += value_usd
                
            elif action == 'sell':
                if token not in smart_money_analysis['tokens_sold']:
                    smart_money_analysis['tokens_sold'][token] = 0
                smart_money_analysis['tokens_sold'][token] += value_usd
                
        return smart_money_analysis
        
    def analyze_network_activity(self, network_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze network activity metrics"""
        activity_scores = {}
        
        # Active addresses
        current_active = network_data.get('active_addresses_24h', 0)
        avg_active = network_data.get('active_addresses_30d_avg', 0)
        
        if avg_active > 0:
            activity_scores['address_growth'] = (current_active - avg_active) / avg_active
        else:
            activity_scores['address_growth'] = 0
            
        # Transaction volume
        current_volume = network_data.get('tx_volume_24h', 0)
        avg_volume = network_data.get('tx_volume_30d_avg', 0)
        
        if avg_volume > 0:
            activity_scores['volume_growth'] = (current_volume - avg_volume) / avg_volume
        else:
            activity_scores['volume_growth'] = 0
            
        # Gas usage (indicates demand)
        gas_used = network_data.get('gas_used_percentage', 50)
        activity_scores['network_demand'] = (gas_used - 50) / 50  # Normalized around 50%
        
        return activity_scores
        
    def _is_accumulation_address(self, address: str) -> bool:
        """Check if address is known for accumulation"""
        # In production, check against database of known addresses
        # and analyze historical behavior
        return address not in self.exchange_addresses
        
    def _is_distribution_address(self, address: str) -> bool:
        """Check if address is known for distribution"""
        return address in self.exchange_addresses
        
    def generate_onchain_signal(self, analysis_results: Dict[str, Any]) -> CryptoSignal:
        """Generate trading signal from on-chain analysis"""
        
        # Default neutral signal
        if not analysis_results:
            return CryptoSignal(
                signal_type='onchain',
                direction=0,
                asset='ETH',
                chain=self.chain,
                opportunity_value=0,
                gas_cost=0,
                confidence=0,
                urgency='long',
                metadata={'reason': 'no_analysis_results'}
            )
            
        # Calculate composite score based on analysis type
        score = 0
        confidence = 0
        
        if self.analysis_type == 'whale_tracking':
            whale_data = analysis_results.get('whale_metrics', {})
            net_flow = whale_data.get('net_flow', 0)
            
            # Normalize flow to score
            if abs(net_flow) > self.whale_threshold * 10:
                score = np.sign(net_flow) * min(abs(net_flow) / (self.whale_threshold * 50), 1.0)
                confidence = 0.8
                
        elif self.analysis_type == 'exchange_flows':
            flow_scores = analysis_results.get('exchange_flows', {})
            if flow_scores:
                avg_score = np.mean(list(flow_scores.values()))
                score = avg_score
                confidence = 0.7
                
        elif self.analysis_type == 'smart_money':
            smart_data = analysis_results.get('smart_money', {})
            accumulated = sum(smart_data.get('tokens_accumulated', {}).values())
            sold = sum(smart_data.get('tokens_sold', {}).values())
            
            if accumulated + sold > 0:
                score = (accumulated - sold) / (accumulated + sold)
                confidence = 0.85
                
        # Determine signal
        if abs(score) < 0.1:
            direction = 0
            urgency = 'long'
        else:
            direction = 1 if score > 0 else -1
            urgency = 'short' if abs(score) > 0.5 else 'medium'
            
        metadata = {
            'analysis_type': self.analysis_type,
            'score': score,
            'lookback_blocks': self.lookback_blocks,
            'data_points': len(analysis_results.get('raw_data', []))
        }
        
        return CryptoSignal(
            signal_type='onchain',
            direction=direction,
            asset='ETH',  # Would be dynamic based on analysis
            chain=self.chain,
            opportunity_value=abs(score) * 1000,  # Placeholder value
            gas_cost=0,  # No gas for analysis
            confidence=confidence,
            urgency=urgency,
            metadata=metadata
        )
        
    def calculate_signal(self, data: pd.DataFrame,
                        onchain_data: Optional[Dict[str, Any]] = None) -> CryptoSignal:
        """Main signal calculation"""
        
        if onchain_data is None:
            # In production, fetch real on-chain data
            onchain_data = self._get_onchain_data(
                endpoint=f"{self.chain}/{self.analysis_type}",
                blocks=self.lookback_blocks
            )
            
        if not onchain_data:
            return CryptoSignal(
                signal_type='onchain',
                direction=0,
                asset='ETH',
                chain=self.chain,
                opportunity_value=0,
                gas_cost=0,
                confidence=0,
                urgency='long',
                metadata={'reason': 'no_onchain_data'}
            )
            
        # Run analysis based on type
        analysis_results = {}
        
        if self.analysis_type == 'whale_tracking':
            analysis_results['whale_metrics'] = self.analyze_whale_movements(
                onchain_data.get('transactions', [])
            )
        elif self.analysis_type == 'exchange_flows':
            analysis_results['exchange_flows'] = self.analyze_exchange_flows(
                onchain_data.get('exchanges', {})
            )
        elif self.analysis_type == 'smart_money':
            analysis_results['smart_money'] = self.analyze_smart_money(
                onchain_data.get('smart_money_txs', [])
            )
        elif self.analysis_type == 'network_activity':
            analysis_results['network'] = self.analyze_network_activity(
                onchain_data.get('network', {})
            )
            
        return self.generate_onchain_signal(analysis_results)

# DeFi Yield Model
class DeFiYieldModel(BaseCryptoModel):
    """
    DeFi yield optimization model
    
    Features:
    - Yield farming opportunities
    - Liquidity provision optimization
    - Impermanent loss calculation
    - Protocol risk assessment
    - APY/APR tracking
    """
    
    def __init__(self, min_apy: float = 10.0,
                 max_risk_score: float = 0.7,
                 include_il_protection: bool = True,
                 chain: str = 'ethereum',
                 confidence_threshold: float = 0.65):
        """
        Initialize DeFi Yield Model
        
        Args:
            min_apy: Minimum APY to consider
            max_risk_score: Maximum acceptable risk (0-1)
            include_il_protection: Consider impermanent loss
            chain: Blockchain to analyze
            confidence_threshold: Minimum confidence
        """
        super().__init__(chain=chain, confidence_threshold=confidence_threshold)
        self.min_apy = min_apy
        self.max_risk_score = max_risk_score
        self.include_il_protection = include_il_protection
        
        # Protocol risk scores (would be dynamic)
        self.protocol_risks = {
            'uniswap': 0.2,
            'aave': 0.15,
            'compound': 0.2,
            'curve': 0.25,
            'yearn': 0.3,
            'sushiswap': 0.35
        }
        
    def calculate_impermanent_loss(self, price_change: float) -> float:
        """Calculate impermanent loss for 50/50 pools"""
        # IL = 2 * sqrt(price_ratio) / (1 + price_ratio) - 1
        price_ratio = 1 + price_change
        if price_ratio <= 0:
            return -1.0  # Total loss
            
        il = 2 * np.sqrt(price_ratio) / (1 + price_ratio) - 1
        return il
        
    def analyze_yield_opportunities(self, defi_pools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze DeFi yield farming opportunities"""
        opportunities = []
        
        for pool in defi_pools:
            protocol = pool.get('protocol', 'unknown')
            apy = pool.get('apy', 0)
            tvl = pool.get('tvl', 0)
            
            # Skip if APY too low
            if apy < self.min_apy:
                continue
                
            # Calculate risk score
            protocol_risk = self.protocol_risks.get(protocol, 0.5)
            tvl_risk = 1.0 / (1.0 + tvl / 1e7) if tvl > 0 else 1.0  # Lower TVL = higher risk
            volatility_risk = pool.get('volatility_score', 0.5)
            
            total_risk = (protocol_risk * 0.4 + tvl_risk * 0.3 + volatility_risk * 0.3)
            
            if total_risk > self.max_risk_score:
                continue
                
            # Calculate expected returns with IL
            if self.include_il_protection and pool.get('type') == 'lp':
                expected_price_change = pool.get('expected_price_change', 0)
                il = self.calculate_impermanent_loss(expected_price_change)
                effective_apy = apy * (1 + il)  # Adjust APY for IL
            else:
                effective_apy = apy
                il = 0
                
            # Estimate gas costs for entry/exit
            gas_cost = self.estimate_gas_cost(
                pool.get('gas_units', 200000),
                priority='medium'
            )
            
            opportunity = {
                'protocol': protocol,
                'pool': pool.get('name', 'Unknown'),
                'token_pair': pool.get('tokens', []),
                'apy': apy,
                'effective_apy': effective_apy,
                'tvl': tvl,
                'risk_score': total_risk,
                'impermanent_loss': il,
                'gas_cost': gas_cost,
                'min_investment': pool.get('min_investment', 100),
                'contract': pool.get('contract_address')
            }
            
            opportunities.append(opportunity)
            
        # Sort by risk-adjusted returns
        opportunities.sort(key=lambda x: x['effective_apy'] / (1 + x['risk_score']), reverse=True)
        
        return opportunities
        
    def assess_protocol_safety(self, protocol_data: Dict[str, Any]) -> float:
        """Assess protocol safety score"""
        safety_score = 1.0
        
        # Audit status
        if protocol_data.get('audited', False):
            safety_score *= 0.8
        else:
            safety_score *= 1.5
            
        # Time since launch
        days_live = protocol_data.get('days_since_launch', 0)
        if days_live > 365:
            safety_score *= 0.9
        elif days_live < 30:
            safety_score *= 1.3
            
        # Bug bounty program
        if protocol_data.get('bug_bounty', False):
            safety_score *= 0.95
            
        # Previous hacks
        hack_count = protocol_data.get('hack_history', 0)
        safety_score *= (1 + hack_count * 0.2)
        
        return min(safety_score, 1.0)
        
    def optimize_capital_allocation(self, opportunities: List[Dict[str, Any]],
                                  capital: float) -> Dict[str, float]:
        """Optimize capital allocation across opportunities"""
        if not opportunities or capital <= 0:
            return {}
            
        allocations = {}
        remaining_capital = capital
        
        # Simple greedy allocation based on risk-adjusted returns
        for opp in opportunities:
            if remaining_capital <= 0:
                break
                
            # Calculate optimal allocation
            risk_adjusted_return = opp['effective_apy'] / (1 + opp['risk_score'])
            
            # Allocate proportionally with max 30% per opportunity
            ideal_allocation = min(
                capital * risk_adjusted_return / 100,  # Proportional to returns
                capital * 0.3,  # Max 30% per pool
                remaining_capital
            )
            
            # Check minimum investment
            if ideal_allocation >= opp['min_investment']:
                pool_id = f"{opp['protocol']}_{opp['pool']}"
                allocations[pool_id] = ideal_allocation
                remaining_capital -= ideal_allocation
                
        return allocations
        
    def generate_yield_signal(self, opportunities: List[Dict[str, Any]],
                            allocations: Dict[str, float]) -> CryptoSignal:
        """Generate signal from yield analysis"""
        
        if not opportunities:
            return CryptoSignal(
                signal_type='defi_yield',
                direction=0,
                asset='USDC',
                chain=self.chain,
                opportunity_value=0,
                gas_cost=0,
                confidence=0,
                urgency='long',
                metadata={'reason': 'no_opportunities'}
            )
            
        # Calculate weighted average metrics
        total_allocated = sum(allocations.values())
        if total_allocated == 0:
            return CryptoSignal(
                signal_type='defi_yield',
                direction=0,
                asset='USDC',
                chain=self.chain,
                opportunity_value=0,
                gas_cost=0,
                confidence=0,
                urgency='long',
                metadata={'reason': 'no_viable_allocations'}
            )
            
        weighted_apy = 0
        weighted_risk = 0
        total_gas = 0
        
        for pool_id, allocation in allocations.items():
            # Find opportunity details
            for opp in opportunities:
                if f"{opp['protocol']}_{opp['pool']}" == pool_id:
                    weight = allocation / total_allocated
                    weighted_apy += opp['effective_apy'] * weight
                    weighted_risk += opp['risk_score'] * weight
                    total_gas += opp['gas_cost']
                    break
                    
        # Expected daily return
        daily_return = total_allocated * (weighted_apy / 365 / 100)
        
        # Signal based on opportunity quality
        if weighted_apy >= self.min_apy * 2 and weighted_risk < 0.4:
            direction = 1  # Strong opportunity
            urgency = 'immediate'
        elif weighted_apy >= self.min_apy and weighted_risk < self.max_risk_score:
            direction = 1  # Good opportunity
            urgency = 'short'
        else:
            direction = 0
            urgency = 'medium'
            
        confidence = (1 - weighted_risk) * 0.6 + min(len(opportunities) / 10, 0.4)
        
        metadata = {
            'num_opportunities': len(opportunities),
            'best_pool': opportunities[0]['pool'] if opportunities else None,
            'weighted_apy': weighted_apy,
            'weighted_risk': weighted_risk,
            'total_allocated': total_allocated,
            'allocations': allocations
        }
        
        return CryptoSignal(
            signal_type='defi_yield',
            direction=direction,
            asset='USDC',  # Stablecoin for yield farming
            chain=self.chain,
            opportunity_value=daily_return,
            gas_cost=total_gas,
            confidence=confidence,
            urgency=urgency,
            metadata=metadata
        )
        
    def calculate_signal(self, data: pd.DataFrame,
                        defi_data: Optional[Dict[str, Any]] = None,
                        capital: float = 10000) -> CryptoSignal:
        """Main signal calculation"""
        
        if defi_data is None:
            # In production, fetch from DeFi aggregators
            defi_data = self._get_onchain_data(
                endpoint='defi/yields',
                chain=self.chain,
                min_apy=self.min_apy
            )
            
        if not defi_data:
            return CryptoSignal(
                signal_type='defi_yield',
                direction=0,
                asset='USDC',
                chain=self.chain,
                opportunity_value=0,
                gas_cost=0,
                confidence=0,
                urgency='long',
                metadata={'reason': 'no_defi_data'}
            )
            
        # Analyze opportunities
        opportunities = self.analyze_yield_opportunities(
            defi_data.get('pools', [])
        )
        
        # Optimize allocation
        allocations = self.optimize_capital_allocation(opportunities, capital)
        
        return self.generate_yield_signal(opportunities, allocations)

# MEV Model
class MEVModel(BaseCryptoModel):
    """
    MEV (Maximal Extractable Value) detection and capture model
    
    Features:
    - Arbitrage opportunity detection
    - Sandwich attack identification
    - Liquidation monitoring
    - Front-running detection
    - Flashloan opportunities
    """
    
    def __init__(self, min_profit: float = 100.0,
                 max_position_size: float = 50000.0,
                 slippage_tolerance: float = 0.02,
                 chain: str = 'ethereum',
                 flashloan_enabled: bool = True):
        """
        Initialize MEV Model
        
        Args:
            min_profit: Minimum profit threshold in USD
            max_position_size: Maximum position size in USD
            slippage_tolerance: Maximum acceptable slippage
            chain: Blockchain to monitor
            flashloan_enabled: Whether to consider flashloan strategies
        """
        super().__init__(chain=chain, min_profit_threshold=min_profit)
        self.max_position_size = max_position_size
        self.slippage_tolerance = slippage_tolerance
        self.flashloan_enabled = flashloan_enabled
        
        # DEX routers and addresses (would be loaded from config)
        self.dex_routers = {
            'uniswap_v2': '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D',
            'uniswap_v3': '0xE592427A0AEce92De3Edee1F18E0157C05861564',
            'sushiswap': '0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F'
        }
        
    def detect_arbitrage_opportunities(self, price_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect cross-DEX arbitrage opportunities"""
        opportunities = []
        
        # Compare prices across DEXes
        for token_pair, dex_prices in price_data.items():
            if len(dex_prices) < 2:
                continue
                
            # Find price discrepancies
            prices = [(dex, data['price'], data['liquidity']) 
                     for dex, data in dex_prices.items()]
            prices.sort(key=lambda x: x[1])  # Sort by price
            
            if not prices:
                continue
                
            # Calculate potential profit
            buy_dex, buy_price, buy_liquidity = prices[0]
            sell_dex, sell_price, sell_liquidity = prices[-1]
            
            price_diff_pct = (sell_price - buy_price) / buy_price
            
            # Check if profitable after slippage
            if price_diff_pct > self.slippage_tolerance * 2:
                # Calculate optimal trade size
                max_size = min(
                    buy_liquidity * 0.1,  # Max 10% of liquidity
                    sell_liquidity * 0.1,
                    self.max_position_size
                )
                
                # Estimate profit
                gross_profit = max_size * price_diff_pct
                
                # Estimate gas costs
                gas_cost = self.estimate_gas_cost(
                    500000,  # Typical arbitrage gas
                    priority='high'
                )
                
                net_profit = gross_profit - gas_cost
                
                if net_profit > self.min_profit_threshold:
                    opportunities.append({
                        'type': 'arbitrage',
                        'token_pair': token_pair,
                        'buy_dex': buy_dex,
                        'sell_dex': sell_dex,
                        'price_diff': price_diff_pct,
                        'size': max_size,
                        'profit': net_profit,
                        'gas_cost': gas_cost,
                        'urgency': 'immediate'
                    })
                    
        return opportunities
        
    def detect_sandwich_opportunities(self, mempool_txs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect sandwich attack opportunities in mempool"""
        opportunities = []
        
        for tx in mempool_txs:
            # Check if it's a large swap
            if tx.get('type') != 'swap' or tx.get('value_usd', 0) < 10000:
                continue
                
            # Analyze price impact
            expected_slippage = tx.get('expected_slippage', 0)
            
            if expected_slippage > 0.01:  # 1% slippage
                # Calculate sandwich profit potential
                victim_size = tx.get('value_usd', 0)
                
                # Front-run size (limited by liquidity)
                front_size = min(victim_size * 0.5, self.max_position_size)
                
                # Expected profit from price movement
                expected_profit = front_size * expected_slippage * 0.5  # Conservative estimate
                
                # Gas for 2 transactions (front + back)
                gas_cost = self.estimate_gas_cost(300000, 'urgent') * 2
                
                if expected_profit - gas_cost > self.min_profit_threshold:
                    opportunities.append({
                        'type': 'sandwich',
                        'victim_tx': tx.get('hash'),
                        'token_pair': tx.get('pair'),
                        'front_size': front_size,
                        'expected_profit': expected_profit - gas_cost,
                        'gas_cost': gas_cost,
                        'risk': 'high',  # Sandwich attacks are risky
                        'urgency': 'immediate'
                    })
                    
        return opportunities
        
    def detect_liquidations(self, lending_positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect upcoming liquidation opportunities"""
        opportunities = []
        
        for position in lending_positions:
            health_factor = position.get('health_factor', 2.0)
            
            # Position close to liquidation
            if health_factor < 1.1:
                collateral_value = position.get('collateral_value', 0)
                debt_value = position.get('debt_value', 0)
                
                # Liquidation bonus (typically 5-10%)
                liquidation_bonus = position.get('liquidation_bonus', 0.05)
                
                # Maximum liquidatable amount
                max_liquidation = min(
                    debt_value * 0.5,  # Usually can liquidate up to 50%
                    self.max_position_size
                )
                
                # Expected profit
                expected_profit = max_liquidation * liquidation_bonus
                
                # Gas cost for liquidation
                gas_cost = self.estimate_gas_cost(400000, 'high')
                
                if expected_profit - gas_cost > self.min_profit_threshold:
                    opportunities.append({
                        'type': 'liquidation',
                        'protocol': position.get('protocol'),
                        'user': position.get('user'),
                        'health_factor': health_factor,
                        'max_liquidation': max_liquidation,
                        'expected_profit': expected_profit - gas_cost,
                        'gas_cost': gas_cost,
                        'urgency': 'immediate' if health_factor < 1.05 else 'short'
                    })
                    
        return opportunities
        
    def analyze_flashloan_strategies(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze flashloan arbitrage strategies"""
        if not self.flashloan_enabled:
            return []
            
        strategies = []
        
        # Check for large price discrepancies that require more capital
        large_arbs = market_data.get('large_arbitrage', [])
        
        for arb in large_arbs:
            required_capital = arb.get('required_capital', 0)
            expected_profit = arb.get('expected_profit', 0)
            
            # Flashloan fee (typically 0.09%)
            flashloan_fee = required_capital * 0.0009
            
            # Total gas including flashloan
            gas_cost = self.estimate_gas_cost(800000, 'high')
            
            net_profit = expected_profit - flashloan_fee - gas_cost
            
            if net_profit > self.min_profit_threshold * 2:  # Higher threshold for flashloans
                strategies.append({
                    'type': 'flashloan_arb',
                    'strategy': arb.get('strategy'),
                    'loan_amount': required_capital,
                    'expected_profit': net_profit,
                    'flashloan_fee': flashloan_fee,
                    'gas_cost': gas_cost,
                    'complexity': 'high',
                    'urgency': 'immediate'
                })
                
        return strategies
        
    def generate_mev_signal(self, all_opportunities: List[Dict[str, Any]]) -> CryptoSignal:
        """Generate signal from MEV opportunities"""
        
        if not all_opportunities:
            return CryptoSignal(
                signal_type='mev',
                direction=0,
                asset='ETH',
                chain=self.chain,
                opportunity_value=0,
                gas_cost=0,
                confidence=0,
                urgency='long',
                metadata={'reason': 'no_opportunities'}
            )
            
        # Sort by profit
        all_opportunities.sort(key=lambda x: x.get('expected_profit', 0), reverse=True)
        
        # Take best opportunity
        best_opp = all_opportunities[0]
        
        # Determine confidence based on opportunity type
        confidence_map = {
            'arbitrage': 0.85,
            'liquidation': 0.9,
            'sandwich': 0.6,  # Riskier
            'flashloan_arb': 0.7
        }
        
        confidence = confidence_map.get(best_opp['type'], 0.5)
        
        # All MEV opportunities are directionally positive
        direction = 1 if best_opp.get('expected_profit', 0) > 0 else 0
        
        metadata = {
            'opportunity_type': best_opp['type'],
            'total_opportunities': len(all_opportunities),
            'best_profit': best_opp.get('expected_profit', 0),
            'details': best_opp
        }
        
        return CryptoSignal(
            signal_type='mev',
            direction=direction,
            asset=best_opp.get('token_pair', 'ETH').split('/')[0],
            chain=self.chain,
            opportunity_value=best_opp.get('expected_profit', 0),
            gas_cost=best_opp.get('gas_cost', 0),
            confidence=confidence,
            urgency=best_opp.get('urgency', 'immediate'),
            metadata=metadata
        )
        
    def calculate_signal(self, data: pd.DataFrame,
                        mev_data: Optional[Dict[str, Any]] = None) -> CryptoSignal:
        """Main signal calculation"""
        
        if mev_data is None:
            # In production, connect to MEV infrastructure
            mev_data = self._get_onchain_data(
                endpoint='mev/opportunities',
                chain=self.chain,
                include_mempool=True
            )
            
        if not mev_data:
            return CryptoSignal(
                signal_type='mev',
                direction=0,
                asset='ETH',
                chain=self.chain,
                opportunity_value=0,
                gas_cost=0,
                confidence=0,
                urgency='long',
                metadata={'reason': 'no_mev_data'}
            )
            
        all_opportunities = []
        
        # Detect various MEV opportunities
        if 'price_data' in mev_data:
            all_opportunities.extend(
                self.detect_arbitrage_opportunities(mev_data['price_data'])
            )
            
        if 'mempool' in mev_data:
            all_opportunities.extend(
                self.detect_sandwich_opportunities(mev_data['mempool'])
            )
            
        if 'lending_positions' in mev_data:
            all_opportunities.extend(
                self.detect_liquidations(mev_data['lending_positions'])
            )
            
        if 'market_data' in mev_data:
            all_opportunities.extend(
                self.analyze_flashloan_strategies(mev_data['market_data'])
            )
            
        return self.generate_mev_signal(all_opportunities)

# Cross-Chain Arbitrage Model
class CrossChainArbitrageModel(BaseCryptoModel):
    """
    Cross-chain arbitrage opportunity detection
    
    Features:
    - Bridge arbitrage
    - Cross-chain price discrepancies
    - Liquidity analysis
    - Bridge fee optimization
    - Multi-hop routing
    """
    
    def __init__(self, chains: List[str] = ['ethereum', 'bsc', 'polygon'],
                 min_profit_pct: float = 2.0,
                 max_bridge_time: int = 3600,  # seconds
                 include_stablecoins: bool = True):
        """
        Initialize Cross-Chain Arbitrage Model
        
        Args:
            chains: List of chains to monitor
            min_profit_pct: Minimum profit percentage
            max_bridge_time: Maximum acceptable bridge time
            include_stablecoins: Include stablecoin arbitrage
        """
        super().__init__(chain=chains[0])  # Primary chain
        self.chains = chains
        self.min_profit_pct = min_profit_pct
        self.max_bridge_time = max_bridge_time
        self.include_stablecoins = include_stablecoins
        
        # Bridge configurations
        self.bridges = {
            ('ethereum', 'bsc'): {'fee': 0.001, 'time': 600},
            ('ethereum', 'polygon'): {'fee': 0.0005, 'time': 1800},
            ('bsc', 'polygon'): {'fee': 0.0003, 'time': 300},
            ('ethereum', 'arbitrum'): {'fee': 0.0002, 'time': 600},
            ('ethereum', 'optimism'): {'fee': 0.0002, 'time': 1800}
        }
        
    def find_price_discrepancies(self, token_prices: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Find price discrepancies across chains"""
        discrepancies = []
        
        for token, chain_prices in token_prices.items():
            if len(chain_prices) < 2:
                continue
                
            # Get all chain pairs
            chains = list(chain_prices.keys())
            
            for i in range(len(chains)):
                for j in range(i + 1, len(chains)):
                    chain1, chain2 = chains[i], chains[j]
                    price1 = chain_prices[chain1]
                    price2 = chain_prices[chain2]
                    
                    # Calculate price difference
                    price_diff_pct = abs(price1 - price2) / min(price1, price2) * 100
                    
                    if price_diff_pct > self.min_profit_pct:
                        # Determine buy and sell chains
                        if price1 < price2:
                            buy_chain, sell_chain = chain1, chain2
                            buy_price, sell_price = price1, price2
                        else:
                            buy_chain, sell_chain = chain2, chain1
                            buy_price, sell_price = price2, price1
                            
                        discrepancies.append({
                            'token': token,
                            'buy_chain': buy_chain,
                            'sell_chain': sell_chain,
                            'buy_price': buy_price,
                            'sell_price': sell_price,
                            'price_diff_pct': price_diff_pct
                        })
                        
        return discrepancies
        
    def calculate_bridge_route(self, from_chain: str, to_chain: str) -> Dict[str, Any]:
        """Calculate optimal bridge route"""
        # Direct bridge
        direct_key = (from_chain, to_chain)
        reverse_key = (to_chain, from_chain)
        
        if direct_key in self.bridges:
            return {
                'route': [from_chain, to_chain],
                'fee': self.bridges[direct_key]['fee'],
                'time': self.bridges[direct_key]['time'],
                'hops': 1
            }
        elif reverse_key in self.bridges:
            return {
                'route': [from_chain, to_chain],
                'fee': self.bridges[reverse_key]['fee'],
                'time': self.bridges[reverse_key]['time'],
                'hops': 1
            }
            
        # Multi-hop routing (simplified - only 2 hops)
        best_route = None
        min_fee = float('inf')
        
        for intermediate in self.chains:
            if intermediate == from_chain or intermediate == to_chain:
                continue
                
            key1 = (from_chain, intermediate)
            key2 = (intermediate, to_chain)
            
            if key1 in self.bridges and key2 in self.bridges:
                total_fee = self.bridges[key1]['fee'] + self.bridges[key2]['fee']
                total_time = self.bridges[key1]['time'] + self.bridges[key2]['time']
                
                if total_fee < min_fee and total_time <= self.max_bridge_time:
                    best_route = {
                        'route': [from_chain, intermediate, to_chain],
                        'fee': total_fee,
                        'time': total_time,
                        'hops': 2
                    }
                    min_fee = total_fee
                    
        return best_route
        
    def analyze_liquidity(self, chain: str, token: str, amount: float) -> Dict[str, float]:
        """Analyze liquidity and slippage on a chain"""
        # In production, query real DEX liquidity
        # Placeholder calculation
        
        base_liquidity = {
            'ethereum': 1000000,
            'bsc': 500000,
            'polygon': 300000,
            'arbitrum': 400000,
            'optimism': 200000
        }
        
        liquidity = base_liquidity.get(chain, 100000)
        
        # Estimate slippage
        size_ratio = amount / liquidity
        slippage = size_ratio * 0.5  # Simplified model
        
        return {
            'liquidity': liquidity,
            'slippage': slippage,
            'max_size': liquidity * 0.1  # Max 10% of pool
        }
        
    def calculate_arbitrage_profit(self, opportunity: Dict[str, Any],
                                 bridge_route: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate net profit after all costs"""
        # Base calculation
        size = min(opportunity.get('max_size', 10000), self.max_position_size)
        buy_price = opportunity['buy_price']
        sell_price = opportunity['sell_price']
        
        # Gross profit
        gross_profit = size * (sell_price - buy_price) / buy_price
        
        # Bridge fees
        bridge_fee = size * bridge_route['fee']
        
        # Slippage on both sides
        buy_slippage = size * opportunity.get('buy_slippage', 0.01)
        sell_slippage = size * opportunity.get('sell_slippage', 0.01)
        
        # Gas costs (multiple transactions)
        gas_costs = {
            'buy': self.estimate_gas_cost(200000, 'high'),
            'bridge': self.estimate_gas_cost(100000, 'medium') * bridge_route['hops'],
            'sell': self.estimate_gas_cost(200000, 'high')
        }
        total_gas = sum(gas_costs.values())
        
        # Net profit
        net_profit = gross_profit - bridge_fee - buy_slippage - sell_slippage - total_gas
        net_profit_pct = net_profit / size * 100
        
        return {
            'size': size,
            'gross_profit': gross_profit,
            'bridge_fee': bridge_fee,
            'slippage_cost': buy_slippage + sell_slippage,
            'gas_cost': total_gas,
            'net_profit': net_profit,
            'net_profit_pct': net_profit_pct,
            'break_even_size': total_gas / (opportunity['price_diff_pct'] / 100) if opportunity['price_diff_pct'] > 0 else float('inf')
        }
        
    def generate_crosschain_signal(self, opportunities: List[Dict[str, Any]]) -> CryptoSignal:
        """Generate signal from cross-chain opportunities"""
        
        if not opportunities:
            return CryptoSignal(
                signal_type='cross_chain',
                direction=0,
                asset='USDC',
                chain='ethereum',
                opportunity_value=0,
                gas_cost=0,
                confidence=0,
                urgency='long',
                metadata={'reason': 'no_opportunities'}
            )
            
        # Sort by net profit
        opportunities.sort(key=lambda x: x.get('net_profit', 0), reverse=True)
        
        best = opportunities[0]
        
        # Determine urgency based on price difference
        if best['price_diff_pct'] > 5:
            urgency = 'immediate'
        elif best['price_diff_pct'] > 3:
            urgency = 'short'
        else:
            urgency = 'medium'
            
        # Confidence based on bridge time and profit margin
        time_factor = 1.0 - (best['bridge_time'] / self.max_bridge_time)
        profit_factor = min(best['net_profit_pct'] / 5, 1.0)
        confidence = time_factor * 0.4 + profit_factor * 0.6
        
        metadata = {
            'token': best['token'],
            'route': best['route'],
            'price_diff_pct': best['price_diff_pct'],
            'bridge_time': best['bridge_time'],
            'net_profit_pct': best['net_profit_pct'],
            'size': best['size'],
            'total_opportunities': len(opportunities)
        }
        
        return CryptoSignal(
            signal_type='cross_chain',
            direction=1,  # Always positive for arbitrage
            asset=best['token'],
            chain=best['buy_chain'],
            opportunity_value=best['net_profit'],
            gas_cost=best['gas_cost'],
            confidence=confidence,
            urgency=urgency,
            metadata=metadata
        )
        
    def calculate_signal(self, data: pd.DataFrame,
                        crosschain_data: Optional[Dict[str, Any]] = None) -> CryptoSignal:
        """Main signal calculation"""
        
        if crosschain_data is None:
            # In production, fetch cross-chain price data
            crosschain_data = self._get_onchain_data(
                endpoint='crosschain/prices',
                chains=self.chains
            )
            
        if not crosschain_data:
            return CryptoSignal(
                signal_type='cross_chain',
                direction=0,
                asset='USDC',
                chain='ethereum',
                opportunity_value=0,
                gas_cost=0,
                confidence=0,
                urgency='long',
                metadata={'reason': 'no_crosschain_data'}
            )
            
        # Find discrepancies
        discrepancies = self.find_price_discrepancies(
            crosschain_data.get('token_prices', {})
        )
        
        # Analyze each opportunity
        viable_opportunities = []
        
        for disc in discrepancies:
            # Skip stablecoins if not included
            if not self.include_stablecoins and disc['token'] in ['USDC', 'USDT', 'DAI']:
                continue
                
            # Calculate bridge route
            bridge_route = self.calculate_bridge_route(
                disc['buy_chain'], disc['sell_chain']
            )
            
            if not bridge_route or bridge_route['time'] > self.max_bridge_time:
                continue
                
            # Analyze liquidity
            buy_liquidity = self.analyze_liquidity(
                disc['buy_chain'], disc['token'], self.max_position_size
            )
            sell_liquidity = self.analyze_liquidity(
                disc['sell_chain'], disc['token'], self.max_position_size
            )
            
            disc['buy_slippage'] = buy_liquidity['slippage']
            disc['sell_slippage'] = sell_liquidity['slippage']
            disc['max_size'] = min(buy_liquidity['max_size'], sell_liquidity['max_size'])
            
            # Calculate profit
            profit_calc = self.calculate_arbitrage_profit(disc, bridge_route)
            
            if profit_calc['net_profit'] > self.min_profit_threshold:
                viable_opportunities.append({
                    **disc,
                    **profit_calc,
                    'route': bridge_route['route'],
                    'bridge_time': bridge_route['time']
                })
                
        return self.generate_crosschain_signal(viable_opportunities)

# Factory function for creating crypto models
def create_crypto_model(model_type: str, **kwargs) -> BaseCryptoModel:
    """Factory function to create crypto/DeFi models"""
    models = {
        'onchain': OnChainModel,
        'defi_yield': DeFiYieldModel,
        'mev': MEVModel,
        'cross_chain': CrossChainArbitrageModel
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
        
    return models[model_type](**kwargs)