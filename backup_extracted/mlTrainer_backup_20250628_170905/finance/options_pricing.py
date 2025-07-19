
import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, Union
import logging

logger = logging.getLogger(__name__)

class OptionsPricing:
    def __init__(self):
        pass
    
    def black_scholes(self, S: float, K: float, T: float, r: float, sigma: float, 
                     option_type: str = "call") -> Dict[str, float]:
        """
        Black-Scholes option pricing model
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        sigma: Volatility
        """
        logger.info(f"📈 Calculating Black-Scholes {option_type} option price")
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type.lower() == "call":
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        # Calculate Greeks
        greeks = self._calculate_greeks(S, K, T, r, sigma, d1, d2, option_type)
        
        return {
            "price": price,
            "delta": greeks["delta"],
            "gamma": greeks["gamma"],
            "theta": greeks["theta"],
            "vega": greeks["vega"],
            "rho": greeks["rho"],
            "d1": d1,
            "d2": d2
        }
    
    def _calculate_greeks(self, S: float, K: float, T: float, r: float, sigma: float,
                         d1: float, d2: float, option_type: str) -> Dict[str, float]:
        """Calculate option Greeks"""
        
        if option_type.lower() == "call":
            delta = norm.cdf(d1)
            rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            delta = -norm.cdf(-d1)
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta_call = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                     - r * K * np.exp(-r * T) * norm.cdf(d2))
        theta_put = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    + r * K * np.exp(-r * T) * norm.cdf(-d2))
        theta = theta_call / 365 if option_type.lower() == "call" else theta_put / 365
        
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        
        return {
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
            "rho": rho / 100
        }
    
    def monte_carlo_option(self, S: float, K: float, T: float, r: float, sigma: float,
                          option_type: str = "call", n_simulations: int = 100000) -> Dict[str, float]:
        """
        Monte Carlo option pricing
        """
        logger.info(f"🎲 Running Monte Carlo simulation for {option_type} option ({n_simulations:,} paths)")
        
        # Generate random paths
        dt = T / 252  # Daily steps
        n_steps = int(T * 252)
        
        # Random walk simulation
        Z = np.random.standard_normal((n_simulations, n_steps))
        
        # Stock price paths
        S_paths = np.zeros((n_simulations, n_steps + 1))
        S_paths[:, 0] = S
        
        for t in range(1, n_steps + 1):
            S_paths[:, t] = S_paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t-1])
        
        # Calculate payoffs
        S_T = S_paths[:, -1]
        if option_type.lower() == "call":
            payoffs = np.maximum(S_T - K, 0)
        else:  # put
            payoffs = np.maximum(K - S_T, 0)
        
        # Discount to present value
        option_price = np.exp(-r * T) * np.mean(payoffs)
        standard_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_simulations)
        
        return {
            "price": option_price,
            "standard_error": standard_error,
            "confidence_interval": (option_price - 1.96 * standard_error, 
                                   option_price + 1.96 * standard_error),
            "n_simulations": n_simulations
        }

options_pricing = OptionsPricing()
