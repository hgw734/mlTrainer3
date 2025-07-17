"""
Statistical Arbitrage Models Implementation

Implements pairs trading, cointegration, and factor-based strategies.
All models require real market data - no synthetic price generation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


@dataclass
class StatArbSignal:
    """Base signal for statistical arbitrage strategies."""
    timestamp: datetime
    signal_type: str  # 'long_spread', 'short_spread', 'neutral'
    strength: float  # 0.0 to 1.0
    z_score: float
    half_life: Optional[float]
    confidence: float
    metrics: Dict[str, float]
    metadata: Dict[str, any]


@dataclass
class OUParameters(StatArbSignal):
    """Ornstein-Uhlenbeck process parameters and signals."""
    mean_reversion_speed: float
    long_term_mean: float
    volatility: float
    entry_threshold: float
    exit_threshold: float
    current_spread: float
    predicted_spread: float


@dataclass
class KalmanSignal(StatArbSignal):
    """Kalman filter pairs trading signal."""
    hedge_ratio: float
    hedge_ratio_variance: float
    spread_mean: float
    spread_variance: float
    innovation: float
    pairs: Tuple[str, str]


@dataclass
class PCAFactors(StatArbSignal):
    """PCA-based factor trading signal."""
    eigenportfolios: pd.DataFrame
    explained_variance: np.ndarray
    factor_loadings: pd.DataFrame
    factor_returns: pd.DataFrame
    selected_factors: List[int]
    portfolio_weights: Dict[str, float]


@dataclass
class CopulaParameters(StatArbSignal):
    """Copula model parameters and dependency signals."""
    copula_type: str
    tail_dependence: Dict[str, float]
    correlation_matrix: pd.DataFrame
    regime: str  # 'normal', 'stressed', 'transitioning'
    pair_signals: Dict[Tuple[str, str], float]


@dataclass
class VECMResults(StatArbSignal):
    """Vector Error Correction Model results."""
    cointegration_rank: int
    error_correction_terms: pd.DataFrame
    adjustment_speeds: np.ndarray
    long_run_matrix: np.ndarray
    impulse_responses: Dict[str, pd.DataFrame]
    portfolio_weights: np.ndarray


class BaseStatArbModel(ABC):
    """Base class for statistical arbitrage models."""
    
    def __init__(self, lookback_period: int = 252, min_data_points: int = 100):
        self.lookback_period = lookback_period
        self.min_data_points = min_data_points
    
    @abstractmethod
    def analyze(self, data: Union[pd.DataFrame, pd.Series]) -> StatArbSignal:
        """Analyze data for statistical arbitrage opportunities."""
        pass
    
    def validate_data(self, data: Union[pd.DataFrame, pd.Series]) -> bool:
        """Validate input data."""
        if data is None:
            return False
        if isinstance(data, pd.Series):
            return len(data) >= self.min_data_points
        else:
            return len(data) >= self.min_data_points and len(data.columns) >= 2


class OrnsteinUhlenbeckModel(BaseStatArbModel):
    """
    Ornstein-Uhlenbeck mean reversion model.
    
    Fits OU process to spread/price series:
    dX_t = θ(μ - X_t)dt + σdW_t
    
    Where:
    - θ: mean reversion speed
    - μ: long-term mean
    - σ: volatility
    - W_t: Brownian motion
    """
    
    def __init__(self, lookback_period: int = 252,
                 entry_z_score: float = 2.0,
                 exit_z_score: float = 0.5):
        super().__init__(lookback_period)
        self.entry_z_score = entry_z_score
        self.exit_z_score = exit_z_score
    
    def analyze(self, spread_data: pd.Series) -> OUParameters:
        """
        Fit OU process to spread data and generate signals.
        
        Args:
            spread_data: Time series of spread values
        """
        if not self.validate_data(spread_data):
            return self._default_signal()
        
        try:
            # Fit OU parameters
            params = self._fit_ou_process(spread_data)
            theta, mu, sigma = params['theta'], params['mu'], params['sigma']
            
            # Calculate half-life
            half_life = np.log(2) / theta if theta > 0 else None
            
            # Current spread value
            current_spread = spread_data.iloc[-1]
            
            # Calculate z-score
            spread_mean = spread_data.tail(self.lookback_period).mean()
            spread_std = spread_data.tail(self.lookback_period).std()
            z_score = (current_spread - spread_mean) / spread_std if spread_std > 0 else 0
            
            # Entry/exit thresholds
            entry_threshold = mu + self.entry_z_score * sigma
            exit_threshold = mu + self.exit_z_score * sigma
            
            # Predict future spread (mean reversion)
            dt = 1  # 1 period ahead
            predicted_spread = current_spread + theta * (mu - current_spread) * dt
            
            # Generate signal
            if z_score > self.entry_z_score:
                signal_type = 'short_spread'  # Spread too high, will revert down
                strength = min(abs(z_score) / 3, 1.0)
            elif z_score < -self.entry_z_score:
                signal_type = 'long_spread'  # Spread too low, will revert up
                strength = min(abs(z_score) / 3, 1.0)
            elif abs(z_score) < self.exit_z_score:
                signal_type = 'neutral'  # Close to mean, exit positions
                strength = 0.2
            else:
                signal_type = 'neutral'
                strength = 0.5
            
            # Calculate confidence based on fit quality and half-life
            confidence = self._calculate_confidence(params, half_life, spread_data)
            
            return OUParameters(
                timestamp=spread_data.index[-1] if hasattr(spread_data.index, '__getitem__') else datetime.now(),
                signal_type=signal_type,
                strength=strength,
                z_score=z_score,
                half_life=half_life,
                confidence=confidence,
                mean_reversion_speed=theta,
                long_term_mean=mu,
                volatility=sigma,
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
                current_spread=current_spread,
                predicted_spread=predicted_spread,
                metrics={
                    'spread_mean': spread_mean,
                    'spread_std': spread_std,
                    'sharpe_ratio': self._calculate_sharpe(spread_data),
                    'max_deviation': abs(spread_data.max() - mu)
                },
                metadata={
                    'fit_method': 'maximum_likelihood',
                    'data_points': len(spread_data)
                }
            )
            
        except Exception as e:
            return self._default_signal()
    
    def _fit_ou_process(self, spread_data: pd.Series) -> Dict[str, float]:
        """Fit OU process parameters using maximum likelihood."""
        y = spread_data.values
        n = len(y)
        
        # Calculate differences
        dy = np.diff(y)
        y_lag = y[:-1]
        
        # OLS regression: dy = a + b*y_lag + error
        # Where: a = theta*mu*dt, b = -theta*dt
        X = np.column_stack([np.ones(n-1), y_lag])
        coeffs = np.linalg.lstsq(X, dy, rcond=None)[0]
        
        a, b = coeffs
        
        # Extract parameters (assuming dt = 1)
        theta = -b
        mu = a / theta if theta != 0 else np.mean(y)
        
        # Calculate residuals and volatility
        residuals = dy - (a + b * y_lag)
        sigma = np.std(residuals) * np.sqrt(1 / (1 - np.exp(-2 * theta))) if theta > 0 else np.std(residuals)
        
        # Ensure positive theta (mean reversion)
        if theta < 0:
            theta = 0.001  # Small positive value
        
        return {
            'theta': theta,
            'mu': mu,
            'sigma': sigma,
            'log_likelihood': self._calculate_log_likelihood(y, theta, mu, sigma)
        }
    
    def _calculate_log_likelihood(self, y: np.ndarray, theta: float, mu: float, sigma: float) -> float:
        """Calculate log-likelihood of OU process."""
        n = len(y)
        dt = 1
        
        # Calculate expected values and variances
        y_mean = mu + (y[:-1] - mu) * np.exp(-theta * dt)
        y_var = (sigma**2 / (2 * theta)) * (1 - np.exp(-2 * theta * dt)) if theta > 0 else sigma**2 * dt
        
        # Log-likelihood
        ll = -0.5 * n * np.log(2 * np.pi * y_var) - 0.5 * np.sum((y[1:] - y_mean)**2 / y_var)
        
        return ll
    
    def _calculate_sharpe(self, spread_data: pd.Series) -> float:
        """Calculate Sharpe ratio of mean reversion strategy."""
        returns = spread_data.pct_change().dropna()
        if len(returns) == 0:
            return 0.0
        
        # Simple strategy: long when below mean, short when above
        mean = spread_data.mean()
        positions = np.where(spread_data < mean, 1, -1)
        strategy_returns = returns * positions[:-1]  # Align with returns
        
        sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0
        return sharpe
    
    def _calculate_confidence(self, params: Dict, half_life: Optional[float], data: pd.Series) -> float:
        """Calculate confidence in OU model fit."""
        # Base confidence on:
        # 1. Reasonable half-life (not too fast, not too slow)
        hl_confidence = 0.0
        if half_life:
            if 1 < half_life < 50:  # Between 1 and 50 periods
                hl_confidence = 0.4
            elif 50 <= half_life < 100:
                hl_confidence = 0.2
        
        # 2. Mean reversion speed significance
        theta_confidence = min(params['theta'] / 0.1, 1.0) * 0.3  # 0.1 as baseline
        
        # 3. Data sufficiency
        data_confidence = min(len(data) / (self.lookback_period * 2), 1.0) * 0.3
        
        return hl_confidence + theta_confidence + data_confidence
    
    def _default_signal(self) -> OUParameters:
        """Return default neutral signal when analysis fails."""
        return OUParameters(
            timestamp=datetime.now(),
            signal_type='neutral',
            strength=0.5,
            z_score=0.0,
            half_life=None,
            confidence=0.0,
            mean_reversion_speed=0.0,
            long_term_mean=0.0,
            volatility=0.0,
            entry_threshold=0.0,
            exit_threshold=0.0,
            current_spread=0.0,
            predicted_spread=0.0,
            metrics={},
            metadata={'error': 'Insufficient or invalid data'}
        )


class KalmanPairsModel(BaseStatArbModel):
    """
    Dynamic pairs trading using Kalman filter.
    
    Estimates time-varying hedge ratio and tracks spread evolution.
    Adapts to changing market relationships dynamically.
    """
    
    def __init__(self, lookback_period: int = 252,
                 delta: float = 1e-4,
                 entry_threshold: float = 2.0):
        super().__init__(lookback_period)
        self.delta = delta  # Covariance of random walk for beta
        self.entry_threshold = entry_threshold
        
        # Kalman filter state
        self.beta = None  # Hedge ratio
        self.P = None     # Error covariance
        self.e = None     # Prediction error
        self.Q = None     # Spread variance
        self.R = None     # Measurement noise
    
    def analyze(self, pair_data: pd.DataFrame) -> KalmanSignal:
        """
        Apply Kalman filter to estimate dynamic hedge ratio.
        
        Args:
            pair_data: DataFrame with two columns (asset prices)
        """
        if not self.validate_data(pair_data) or len(pair_data.columns) < 2:
            return self._default_signal()
        
        try:
            # Extract price series
            y = pair_data.iloc[:, 0].values  # Dependent variable
            x = pair_data.iloc[:, 1].values  # Independent variable
            pair_names = (pair_data.columns[0], pair_data.columns[1])
            
            # Initialize Kalman filter
            self._initialize_kalman(y, x)
            
            # Run Kalman filter
            betas, spreads, innovations = self._run_kalman_filter(y, x)
            
            # Current values
            current_beta = betas[-1]
            current_spread = spreads[-1]
            current_innovation = innovations[-1]
            
            # Calculate spread statistics
            spread_mean = np.mean(spreads[-self.lookback_period:])
            spread_std = np.std(spreads[-self.lookback_period:])
            z_score = (current_spread - spread_mean) / spread_std if spread_std > 0 else 0
            
            # Generate signal
            if z_score > self.entry_threshold:
                signal_type = 'short_spread'
                strength = min(abs(z_score) / 3, 1.0)
            elif z_score < -self.entry_threshold:
                signal_type = 'long_spread'
                strength = min(abs(z_score) / 3, 1.0)
            else:
                signal_type = 'neutral'
                strength = 0.5
            
            # Calculate half-life
            half_life = self._calculate_half_life(spreads)
            
            # Calculate confidence
            confidence = self._calculate_confidence(innovations, spread_std)
            
            return KalmanSignal(
                timestamp=pair_data.index[-1] if hasattr(pair_data.index, '__getitem__') else datetime.now(),
                signal_type=signal_type,
                strength=strength,
                z_score=z_score,
                half_life=half_life,
                confidence=confidence,
                hedge_ratio=current_beta,
                hedge_ratio_variance=self.P,
                spread_mean=spread_mean,
                spread_variance=spread_std**2,
                innovation=current_innovation,
                pairs=pair_names,
                metrics={
                    'beta_stability': self._calculate_beta_stability(betas),
                    'spread_sharpe': self._calculate_spread_sharpe(spreads),
                    'innovation_ratio': abs(current_innovation) / spread_std if spread_std > 0 else 0,
                    'max_drawdown': self._calculate_max_drawdown(spreads)
                },
                metadata={
                    'filter_type': 'kalman',
                    'delta': self.delta,
                    'data_points': len(pair_data)
                }
            )
            
        except Exception as e:
            return self._default_signal()
    
    def _initialize_kalman(self, y: np.ndarray, x: np.ndarray):
        """Initialize Kalman filter parameters."""
        # Initial hedge ratio estimate (OLS)
        self.beta = np.cov(y, x)[0, 1] / np.var(x)
        
        # Initial error covariance
        self.P = 1.0
        
        # Process noise (random walk variance for beta)
        self.Q = self.delta
        
        # Initial measurement noise
        spread = y - self.beta * x
        self.R = np.var(spread)
    
    def _run_kalman_filter(self, y: np.ndarray, x: np.ndarray) -> Tuple[List, List, List]:
        """Run Kalman filter to estimate time-varying beta."""
        n = len(y)
        betas = []
        spreads = []
        innovations = []
        
        for i in range(n):
            # Prediction step
            beta_pred = self.beta
            P_pred = self.P + self.Q
            
            # Update step
            y_pred = beta_pred * x[i]
            e = y[i] - y_pred  # Innovation
            
            # Kalman gain
            K = P_pred * x[i] / (x[i]**2 * P_pred + self.R)
            
            # Update estimates
            self.beta = beta_pred + K * e
            self.P = (1 - K * x[i]) * P_pred
            
            # Update measurement noise (adaptive)
            self.R = 0.95 * self.R + 0.05 * e**2
            
            # Store results
            betas.append(self.beta)
            spreads.append(y[i] - self.beta * x[i])
            innovations.append(e)
        
        return betas, spreads, innovations
    
    def _calculate_half_life(self, spreads: List[float]) -> float:
        """Calculate half-life of mean reversion."""
        spreads_array = np.array(spreads)
        
        # Fit AR(1) model
        y = spreads_array[1:]
        x = spreads_array[:-1]
        
        # OLS: y = a + b*x
        b = np.cov(y, x)[0, 1] / np.var(x) if np.var(x) > 0 else 0
        
        # Half-life = -log(2) / log(b)
        if 0 < b < 1:
            half_life = -np.log(2) / np.log(b)
            return min(half_life, 252)  # Cap at 1 year
        else:
            return None
    
    def _calculate_beta_stability(self, betas: List[float]) -> float:
        """Calculate stability of hedge ratio over time."""
        if len(betas) < 2:
            return 0.0
        
        beta_changes = np.diff(betas)
        stability = 1 - np.std(beta_changes) / (np.mean(np.abs(betas)) + 1e-6)
        return max(0, min(1, stability))
    
    def _calculate_spread_sharpe(self, spreads: List[float]) -> float:
        """Calculate Sharpe ratio of spread trading strategy."""
        if len(spreads) < 2:
            return 0.0
        
        spread_returns = np.diff(spreads) / np.abs(spreads[:-1] + 1e-6)
        if len(spread_returns) == 0 or np.std(spread_returns) == 0:
            return 0.0
        
        sharpe = np.sqrt(252) * np.mean(spread_returns) / np.std(spread_returns)
        return sharpe
    
    def _calculate_max_drawdown(self, spreads: List[float]) -> float:
        """Calculate maximum drawdown of spread."""
        spreads_array = np.array(spreads)
        cummax = np.maximum.accumulate(spreads_array)
        drawdown = (spreads_array - cummax) / (cummax + 1e-6)
        return abs(np.min(drawdown))
    
    def _calculate_confidence(self, innovations: List[float], spread_std: float) -> float:
        """Calculate confidence in Kalman filter results."""
        # Innovation consistency
        innovation_std = np.std(innovations)
        innovation_confidence = min(1 - innovation_std / (spread_std + 1e-6), 1.0) * 0.5
        
        # Data sufficiency
        data_confidence = min(len(innovations) / self.lookback_period, 1.0) * 0.5
        
        return innovation_confidence + data_confidence
    
    def _default_signal(self) -> KalmanSignal:
        """Return default neutral signal when analysis fails."""
        return KalmanSignal(
            timestamp=datetime.now(),
            signal_type='neutral',
            strength=0.5,
            z_score=0.0,
            half_life=None,
            confidence=0.0,
            hedge_ratio=1.0,
            hedge_ratio_variance=1.0,
            spread_mean=0.0,
            spread_variance=1.0,
            innovation=0.0,
            pairs=('', ''),
            metrics={},
            metadata={'error': 'Insufficient or invalid data'}
        )


class PCAStrategyModel(BaseStatArbModel):
    """
    Principal Component Analysis for statistical arbitrage.
    
    Creates eigenportfolios and trades factor rotations.
    Identifies statistical arbitrage opportunities in factor space.
    """
    
    def __init__(self, lookback_period: int = 252,
                 n_factors: int = 3,
                 min_explained_variance: float = 0.8):
        super().__init__(lookback_period)
        self.n_factors = n_factors
        self.min_explained_variance = min_explained_variance
    
    def analyze(self, returns_matrix: pd.DataFrame) -> PCAFactors:
        """
        Perform PCA analysis and generate factor-based signals.
        
        Args:
            returns_matrix: DataFrame of asset returns (assets as columns)
        """
        if not self.validate_data(returns_matrix):
            return self._default_signal()
        
        try:
            # Standardize returns
            returns_std = (returns_matrix - returns_matrix.mean()) / returns_matrix.std()
            
            # Perform PCA
            eigenvalues, eigenvectors = self._perform_pca(returns_std)
            
            # Calculate explained variance
            explained_variance = eigenvalues / eigenvalues.sum()
            cumsum_variance = explained_variance.cumsum()
            
            # Select number of factors
            n_factors_selected = self._select_n_factors(cumsum_variance)
            
            # Create eigenportfolios
            eigenportfolios = pd.DataFrame(
                eigenvectors[:, :n_factors_selected],
                index=returns_matrix.columns,
                columns=[f'PC{i+1}' for i in range(n_factors_selected)]
            )
            
            # Calculate factor returns
            factor_returns = returns_std @ eigenportfolios
            
            # Identify arbitrage opportunities
            arb_signal, selected_factors = self._identify_arbitrage(factor_returns)
            
            # Create portfolio weights
            portfolio_weights = self._create_portfolio_weights(
                eigenportfolios, selected_factors, arb_signal
            )
            
            # Calculate z-score of first factor (market factor)
            z_score = self._calculate_factor_zscore(factor_returns.iloc[:, 0])
            
            # Determine signal
            if arb_signal == 'mean_reversion':
                signal_type = 'long_spread' if z_score < -2 else 'short_spread'
                strength = min(abs(z_score) / 3, 1.0)
            elif arb_signal == 'momentum':
                signal_type = 'long_spread' if z_score > 0 else 'short_spread'
                strength = 0.6
            else:
                signal_type = 'neutral'
                strength = 0.5
            
            # Calculate confidence
            confidence = self._calculate_confidence(explained_variance, n_factors_selected)
            
            return PCAFactors(
                timestamp=returns_matrix.index[-1] if hasattr(returns_matrix.index, '__getitem__') else datetime.now(),
                signal_type=signal_type,
                strength=strength,
                z_score=z_score,
                half_life=self._calculate_factor_halflife(factor_returns.iloc[:, 0]),
                confidence=confidence,
                eigenportfolios=eigenportfolios,
                explained_variance=explained_variance[:n_factors_selected],
                factor_loadings=eigenportfolios,
                factor_returns=factor_returns,
                selected_factors=selected_factors,
                portfolio_weights=portfolio_weights,
                metrics={
                    'total_variance_explained': cumsum_variance[n_factors_selected-1],
                    'factor_sharpe': self._calculate_factor_sharpe(factor_returns),
                    'condition_number': self._calculate_condition_number(returns_std.corr()),
                    'factor_stability': self._calculate_factor_stability(returns_std, eigenportfolios)
                },
                metadata={
                    'n_assets': len(returns_matrix.columns),
                    'n_factors': n_factors_selected,
                    'data_points': len(returns_matrix)
                }
            )
            
        except Exception as e:
            return self._default_signal()
    
    def _perform_pca(self, returns_std: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Perform PCA decomposition."""
        # Calculate correlation matrix
        corr_matrix = returns_std.corr()
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
        
        # Sort by eigenvalue (descending)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        return eigenvalues, eigenvectors
    
    def _select_n_factors(self, cumsum_variance: np.ndarray) -> int:
        """Select number of factors based on explained variance."""
        # Find minimum factors to explain target variance
        n_factors = np.argmax(cumsum_variance >= self.min_explained_variance) + 1
        
        # Bound by maximum factors
        n_factors = min(n_factors, self.n_factors, len(cumsum_variance))
        
        return max(1, n_factors)
    
    def _identify_arbitrage(self, factor_returns: pd.DataFrame) -> Tuple[str, List[int]]:
        """Identify statistical arbitrage opportunities in factor space."""
        arb_opportunities = []
        
        # Check each factor for mean reversion
        for i in range(len(factor_returns.columns)):
            factor = factor_returns.iloc[:, i]
            
            # Test for stationarity (simplified)
            if self._is_stationary(factor):
                z_score = self._calculate_factor_zscore(factor)
                if abs(z_score) > 2:
                    arb_opportunities.append((i, 'mean_reversion', abs(z_score)))
        
        # Check factor momentum
        factor_momentum = factor_returns.tail(20).mean()
        for i, momentum in enumerate(factor_momentum):
            if abs(momentum) > factor_returns.iloc[:, i].std() * 0.5:
                arb_opportunities.append((i, 'momentum', abs(momentum)))
        
        if arb_opportunities:
            # Sort by strength
            arb_opportunities.sort(key=lambda x: x[2], reverse=True)
            best_opp = arb_opportunities[0]
            return best_opp[1], [best_opp[0]]
        else:
            return 'neutral', []
    
    def _create_portfolio_weights(self, eigenportfolios: pd.DataFrame, 
                                 selected_factors: List[int], 
                                 signal_type: str) -> Dict[str, float]:
        """Create portfolio weights based on selected factors."""
        weights = {}
        
        if not selected_factors or signal_type == 'neutral':
            # Equal weight portfolio
            n_assets = len(eigenportfolios.index)
            for asset in eigenportfolios.index:
                weights[asset] = 1.0 / n_assets
        else:
            # Use eigenportfolio weights
            factor_weights = eigenportfolios.iloc[:, selected_factors[0]]
            
            # Normalize to sum to 1
            weight_sum = factor_weights.abs().sum()
            for asset, weight in factor_weights.items():
                weights[asset] = weight / weight_sum if weight_sum > 0 else 0
        
        return weights
    
    def _calculate_factor_zscore(self, factor_returns: pd.Series) -> float:
        """Calculate z-score of factor returns."""
        mean = factor_returns.mean()
        std = factor_returns.std()
        current = factor_returns.iloc[-1]
        
        return (current - mean) / std if std > 0 else 0
    
    def _calculate_factor_halflife(self, factor_returns: pd.Series) -> float:
        """Calculate half-life of factor mean reversion."""
        # Fit AR(1) model
        y = factor_returns.values[1:]
        x = factor_returns.values[:-1]
        
        if len(y) < 2 or np.var(x) == 0:
            return None
        
        # OLS coefficient
        b = np.cov(y, x)[0, 1] / np.var(x)
        
        # Half-life
        if 0 < b < 1:
            return -np.log(2) / np.log(b)
        else:
            return None
    
    def _is_stationary(self, series: pd.Series) -> bool:
        """Simple stationarity test (ADF test simplified)."""
        # Check if mean reverting using rolling statistics
        rolling_mean = series.rolling(window=20).mean()
        rolling_std = series.rolling(window=20).std()
        
        # If rolling stats are relatively stable, assume stationary
        mean_stability = rolling_mean.std() / (series.std() + 1e-6)
        std_stability = rolling_std.std() / (rolling_std.mean() + 1e-6)
        
        return mean_stability < 0.5 and std_stability < 0.5
    
    def _calculate_factor_sharpe(self, factor_returns: pd.DataFrame) -> float:
        """Calculate average Sharpe ratio of factors."""
        sharpes = []
        
        for col in factor_returns.columns:
            returns = factor_returns[col]
            if returns.std() > 0:
                sharpe = np.sqrt(252) * returns.mean() / returns.std()
                sharpes.append(sharpe)
        
        return np.mean(sharpes) if sharpes else 0.0
    
    def _calculate_condition_number(self, corr_matrix: pd.DataFrame) -> float:
        """Calculate condition number of correlation matrix."""
        eigenvalues = np.linalg.eigvalsh(corr_matrix)
        return max(eigenvalues) / min(eigenvalues) if min(eigenvalues) > 0 else float('inf')
    
    def _calculate_factor_stability(self, returns: pd.DataFrame, eigenportfolios: pd.DataFrame) -> float:
        """Calculate stability of factor loadings over time."""
        # Split data in half and compare PCAs
        mid_point = len(returns) // 2
        
        # First half PCA
        returns1 = returns.iloc[:mid_point]
        _, eigenvec1 = self._perform_pca((returns1 - returns1.mean()) / returns1.std())
        
        # Second half PCA
        returns2 = returns.iloc[mid_point:]
        _, eigenvec2 = self._perform_pca((returns2 - returns2.mean()) / returns2.std())
        
        # Compare first few eigenvectors (alignment)
        n_compare = min(3, eigenvec1.shape[1], eigenvec2.shape[1])
        stability = 0
        
        for i in range(n_compare):
            # Check alignment (absolute value of dot product)
            alignment = abs(np.dot(eigenvec1[:, i], eigenvec2[:, i]))
            stability += alignment
        
        return stability / n_compare if n_compare > 0 else 0.0
    
    def _calculate_confidence(self, explained_variance: np.ndarray, n_factors: int) -> float:
        """Calculate confidence in PCA strategy."""
        # Variance explained confidence
        var_confidence = min(explained_variance[:n_factors].sum(), 1.0) * 0.5
        
        # Factor concentration (not too concentrated in first factor)
        concentration = explained_variance[0] / explained_variance[:n_factors].sum()
        concentration_confidence = (1 - concentration) * 0.5 if concentration < 0.9 else 0.1
        
        return var_confidence + concentration_confidence
    
    def _default_signal(self) -> PCAFactors:
        """Return default neutral signal when analysis fails."""
        return PCAFactors(
            timestamp=datetime.now(),
            signal_type='neutral',
            strength=0.5,
            z_score=0.0,
            half_life=None,
            confidence=0.0,
            eigenportfolios=pd.DataFrame(),
            explained_variance=np.array([]),
            factor_loadings=pd.DataFrame(),
            factor_returns=pd.DataFrame(),
            selected_factors=[],
            portfolio_weights={},
            metrics={},
            metadata={'error': 'Insufficient or invalid data'}
        )


class CopulaModel(BaseStatArbModel):
    """
    Copula-based dependency modeling for pairs trading.
    
    Models non-linear dependencies and tail correlations between assets.
    Detects regime changes and correlation breaks.
    """
    
    def __init__(self, lookback_period: int = 252,
                 copula_type: str = 'gaussian',
                 tail_threshold: float = 0.1):
        super().__init__(lookback_period)
        self.copula_type = copula_type
        self.tail_threshold = tail_threshold
    
    def analyze(self, returns_data: pd.DataFrame) -> CopulaParameters:
        """
        Fit copula model and identify trading opportunities.
        
        Args:
            returns_data: DataFrame of asset returns
        """
        if not self.validate_data(returns_data):
            return self._default_signal()
        
        try:
            # Convert returns to uniform marginals
            uniform_data = self._transform_to_uniform(returns_data)
            
            # Fit copula
            copula_params = self._fit_copula(uniform_data)
            
            # Calculate dependency measures
            tail_dependence = self._calculate_tail_dependence(uniform_data)
            
            # Detect regime
            regime = self._detect_regime(uniform_data, tail_dependence)
            
            # Generate pair signals
            pair_signals = self._generate_pair_signals(returns_data, copula_params)
            
            # Overall signal based on regime and dependencies
            if regime == 'stressed':
                signal_type = 'short_spread'  # Risk-off
                strength = 0.8
            elif regime == 'normal' and pair_signals:
                # Find strongest pair signal
                best_pair = max(pair_signals.items(), key=lambda x: abs(x[1]))
                signal_type = 'long_spread' if best_pair[1] > 0 else 'short_spread'
                strength = min(abs(best_pair[1]), 1.0)
            else:
                signal_type = 'neutral'
                strength = 0.5
            
            # Calculate z-score (simplified - based on correlation breaks)
            z_score = self._calculate_correlation_zscore(returns_data)
            
            # Calculate confidence
            confidence = self._calculate_confidence(regime, tail_dependence)
            
            return CopulaParameters(
                timestamp=returns_data.index[-1] if hasattr(returns_data.index, '__getitem__') else datetime.now(),
                signal_type=signal_type,
                strength=strength,
                z_score=z_score,
                half_life=None,  # Not applicable for copula
                confidence=confidence,
                copula_type=self.copula_type,
                tail_dependence=tail_dependence,
                correlation_matrix=returns_data.corr(),
                regime=regime,
                pair_signals=pair_signals,
                metrics={
                    'avg_correlation': returns_data.corr().values[np.triu_indices_from(returns_data.corr().values, k=1)].mean(),
                    'max_tail_dependence': max(tail_dependence.values()) if tail_dependence else 0,
                    'correlation_stability': self._calculate_correlation_stability(returns_data),
                    'regime_probability': self._calculate_regime_probability(regime, uniform_data)
                },
                metadata={
                    'n_assets': len(returns_data.columns),
                    'copula_type': self.copula_type,
                    'data_points': len(returns_data)
                }
            )
            
        except Exception as e:
            return self._default_signal()
    
    def _transform_to_uniform(self, returns_data: pd.DataFrame) -> pd.DataFrame:
        """Transform returns to uniform marginals using empirical CDF."""
        uniform_data = pd.DataFrame(index=returns_data.index, columns=returns_data.columns)
        
        for col in returns_data.columns:
            # Empirical CDF transformation
            sorted_returns = returns_data[col].sort_values()
            ranks = returns_data[col].rank() / (len(returns_data) + 1)
            uniform_data[col] = ranks
        
        return uniform_data
    
    def _fit_copula(self, uniform_data: pd.DataFrame) -> Dict:
        """Fit copula model to uniform data."""
        if self.copula_type == 'gaussian':
            # Gaussian copula - use correlation of normal quantiles
            normal_data = pd.DataFrame()
            for col in uniform_data.columns:
                normal_data[col] = stats.norm.ppf(uniform_data[col])
            
            # Correlation matrix is the copula parameter
            rho = normal_data.corr()
            
            return {'type': 'gaussian', 'rho': rho}
        
        elif self.copula_type == 't':
            # Student-t copula (simplified)
            # Would need to estimate degrees of freedom
            normal_data = pd.DataFrame()
            for col in uniform_data.columns:
                normal_data[col] = stats.norm.ppf(uniform_data[col])
            
            rho = normal_data.corr()
            df = 5  # Simplified - would estimate from data
            
            return {'type': 't', 'rho': rho, 'df': df}
        
        else:
            # Default to Gaussian
            return self._fit_copula_gaussian(uniform_data)
    
    def _calculate_tail_dependence(self, uniform_data: pd.DataFrame) -> Dict[Tuple[str, str], float]:
        """Calculate empirical tail dependence coefficients."""
        tail_dep = {}
        columns = list(uniform_data.columns)
        
        for i in range(len(columns)):
            for j in range(i+1, len(columns)):
                col1, col2 = columns[i], columns[j]
                
                # Upper tail dependence
                upper_tail = self._empirical_tail_dependence(
                    uniform_data[col1], uniform_data[col2], 
                    self.tail_threshold, upper=True
                )
                
                # Lower tail dependence
                lower_tail = self._empirical_tail_dependence(
                    uniform_data[col1], uniform_data[col2], 
                    self.tail_threshold, upper=False
                )
                
                # Average tail dependence
                tail_dep[(col1, col2)] = (upper_tail + lower_tail) / 2
        
        return tail_dep
    
    def _empirical_tail_dependence(self, u1: pd.Series, u2: pd.Series, 
                                  threshold: float, upper: bool = True) -> float:
        """Calculate empirical tail dependence coefficient."""
        if upper:
            # Upper tail: P(U2 > 1-threshold | U1 > 1-threshold)
            condition = u1 > (1 - threshold)
            if condition.sum() == 0:
                return 0.0
            prob = ((u1 > (1 - threshold)) & (u2 > (1 - threshold))).sum() / condition.sum()
        else:
            # Lower tail: P(U2 < threshold | U1 < threshold)
            condition = u1 < threshold
            if condition.sum() == 0:
                return 0.0
            prob = ((u1 < threshold) & (u2 < threshold)).sum() / condition.sum()
        
        # Normalize to get tail dependence coefficient
        return prob / threshold if threshold > 0 else 0.0
    
    def _detect_regime(self, uniform_data: pd.DataFrame, tail_dependence: Dict) -> str:
        """Detect market regime based on dependencies."""
        # High tail dependence indicates stress
        avg_tail_dep = np.mean(list(tail_dependence.values()))
        
        # Rolling correlation breaks
        window = min(60, len(uniform_data) // 4)
        recent_corr = uniform_data.tail(window).corr()
        full_corr = uniform_data.corr()
        
        # Correlation difference
        corr_diff = np.abs(recent_corr - full_corr).values[np.triu_indices_from(recent_corr.values, k=1)].mean()
        
        if avg_tail_dep > 0.7 or corr_diff > 0.3:
            return 'stressed'
        elif avg_tail_dep < 0.3 and corr_diff < 0.1:
            return 'normal'
        else:
            return 'transitioning'
    
    def _generate_pair_signals(self, returns_data: pd.DataFrame, copula_params: Dict) -> Dict[Tuple[str, str], float]:
        """Generate trading signals for each pair."""
        signals = {}
        columns = list(returns_data.columns)
        
        for i in range(len(columns)):
            for j in range(i+1, len(columns)):
                pair = (columns[i], columns[j])
                
                # Calculate dependency-adjusted spread
                spread = self._calculate_copula_spread(
                    returns_data[columns[i]], 
                    returns_data[columns[j]],
                    copula_params
                )
                
                # Generate signal based on spread deviation
                spread_mean = spread.mean()
                spread_std = spread.std()
                current_spread = spread.iloc[-1]
                
                if spread_std > 0:
                    z_score = (current_spread - spread_mean) / spread_std
                    if abs(z_score) > 2:
                        signals[pair] = -z_score  # Negative for mean reversion
        
        return signals
    
    def _calculate_copula_spread(self, returns1: pd.Series, returns2: pd.Series, copula_params: Dict) -> pd.Series:
        """Calculate spread adjusted for non-linear dependencies."""
        # Simple linear spread
        beta = returns1.cov(returns2) / returns2.var() if returns2.var() > 0 else 0
        spread = returns1 - beta * returns2
        
        # Adjust for copula (simplified)
        if copula_params['type'] == 'gaussian':
            # No adjustment for Gaussian copula
            pass
        elif copula_params['type'] == 't':
            # Adjust for heavier tails
            df = copula_params.get('df', 5)
            adjustment = np.sqrt((df - 2) / df) if df > 2 else 1
            spread = spread * adjustment
        
        return spread
    
    def _calculate_correlation_zscore(self, returns_data: pd.DataFrame) -> float:
        """Calculate z-score based on correlation breaks."""
        # Recent vs historical correlation
        window = min(60, len(returns_data) // 4)
        recent_corr = returns_data.tail(window).corr()
        hist_corr = returns_data.corr()
        
        # Average correlation difference
        corr_diff = (recent_corr - hist_corr).values[np.triu_indices_from(recent_corr.values, k=1)]
        
        return np.mean(corr_diff) / (np.std(corr_diff) + 1e-6)
    
    def _calculate_correlation_stability(self, returns_data: pd.DataFrame) -> float:
        """Calculate stability of correlations over time."""
        window = min(60, len(returns_data) // 4)
        rolling_corr = []
        
        for i in range(window, len(returns_data), 20):
            corr = returns_data.iloc[i-window:i].corr()
            rolling_corr.append(corr.values[np.triu_indices_from(corr.values, k=1)])
        
        if len(rolling_corr) < 2:
            return 1.0
        
        # Calculate variance of correlations
        corr_array = np.array(rolling_corr)
        stability = 1 - np.mean(np.std(corr_array, axis=0))
        
        return max(0, min(1, stability))
    
    def _calculate_regime_probability(self, regime: str, uniform_data: pd.DataFrame) -> float:
        """Calculate probability of current regime."""
        # Simplified regime probability based on tail behavior
        recent_data = uniform_data.tail(20)
        
        # Count extreme co-movements
        extreme_threshold = 0.1
        upper_extreme = ((recent_data > (1 - extreme_threshold)).sum(axis=1) >= 2).sum()
        lower_extreme = ((recent_data < extreme_threshold).sum(axis=1) >= 2).sum()
        
        extreme_ratio = (upper_extreme + lower_extreme) / len(recent_data)
        
        if regime == 'stressed':
            return min(extreme_ratio * 2, 1.0)
        elif regime == 'normal':
            return max(1 - extreme_ratio * 2, 0.0)
        else:
            return 0.5
    
    def _calculate_confidence(self, regime: str, tail_dependence: Dict) -> float:
        """Calculate confidence in copula model."""
        # Regime clarity
        if regime in ['stressed', 'normal']:
            regime_confidence = 0.6
        else:
            regime_confidence = 0.3
        
        # Tail dependence significance
        avg_tail = np.mean(list(tail_dependence.values()))
        tail_confidence = min(avg_tail, 0.4)
        
        return regime_confidence + tail_confidence
    
    def _default_signal(self) -> CopulaParameters:
        """Return default neutral signal when analysis fails."""
        return CopulaParameters(
            timestamp=datetime.now(),
            signal_type='neutral',
            strength=0.5,
            z_score=0.0,
            half_life=None,
            confidence=0.0,
            copula_type=self.copula_type,
            tail_dependence={},
            correlation_matrix=pd.DataFrame(),
            regime='normal',
            pair_signals={},
            metrics={},
            metadata={'error': 'Insufficient or invalid data'}
        )


class VECMModel(BaseStatArbModel):
    """
    Vector Error Correction Model for cointegrated systems.
    
    Models long-run equilibrium relationships and short-run dynamics.
    Handles multiple cointegrated assets simultaneously.
    """
    
    def __init__(self, lookback_period: int = 252,
                 max_lag: int = 5,
                 significance_level: float = 0.05):
        super().__init__(lookback_period)
        self.max_lag = max_lag
        self.significance_level = significance_level
    
    def analyze(self, price_data: pd.DataFrame) -> VECMResults:
        """
        Fit VECM model and generate trading signals.
        
        Args:
            price_data: DataFrame of asset prices (in levels)
        """
        if not self.validate_data(price_data):
            return self._default_signal()
        
        try:
            # Test for cointegration
            coint_rank, coint_vectors = self._test_cointegration(price_data)
            
            if coint_rank == 0:
                # No cointegration found
                return self._default_signal()
            
            # Fit VECM model
            vecm_params = self._fit_vecm(price_data, coint_rank, coint_vectors)
            
            # Extract components
            error_correction = vecm_params['error_correction']
            adjustment_speeds = vecm_params['adjustment_speeds']
            long_run_matrix = vecm_params['long_run_matrix']
            
            # Calculate impulse responses
            impulse_responses = self._calculate_impulse_response(vecm_params)
            
            # Generate trading signal
            current_errors = self._calculate_current_errors(price_data, coint_vectors)
            z_scores = current_errors / current_errors.std()
            
            # Portfolio weights from cointegrating vector
            portfolio_weights = coint_vectors[:, 0] / np.sum(np.abs(coint_vectors[:, 0]))
            
            # Signal based on error correction
            max_zscore = np.max(np.abs(z_scores))
            if max_zscore > 2:
                idx = np.argmax(np.abs(z_scores))
                if z_scores.iloc[idx] > 0:
                    signal_type = 'short_spread'
                else:
                    signal_type = 'long_spread'
                strength = min(max_zscore / 3, 1.0)
            else:
                signal_type = 'neutral'
                strength = 0.5
            
            # Calculate half-life from adjustment speeds
            half_life = self._calculate_halflife_from_adjustment(adjustment_speeds)
            
            # Calculate confidence
            confidence = self._calculate_confidence(coint_rank, adjustment_speeds)
            
            return VECMResults(
                timestamp=price_data.index[-1] if hasattr(price_data.index, '__getitem__') else datetime.now(),
                signal_type=signal_type,
                strength=strength,
                z_score=float(max_zscore),
                half_life=half_life,
                confidence=confidence,
                cointegration_rank=coint_rank,
                error_correction_terms=error_correction,
                adjustment_speeds=adjustment_speeds,
                long_run_matrix=long_run_matrix,
                impulse_responses=impulse_responses,
                portfolio_weights=portfolio_weights,
                metrics={
                    'max_eigenvalue': self._calculate_max_eigenvalue(coint_vectors),
                    'trace_statistic': self._calculate_trace_statistic(price_data),
                    'model_stability': self._check_model_stability(vecm_params),
                    'forecast_error': self._calculate_forecast_error(price_data, vecm_params)
                },
                metadata={
                    'n_assets': len(price_data.columns),
                    'lag_order': self._select_lag_order(price_data),
                    'data_points': len(price_data)
                }
            )
            
        except Exception as e:
            return self._default_signal()
    
    def _test_cointegration(self, price_data: pd.DataFrame) -> Tuple[int, np.ndarray]:
        """Test for cointegration using Johansen test (simplified)."""
        # Calculate differences and lagged values
        diff_data = price_data.diff().dropna()
        lagged_data = price_data.shift(1).dropna()
        
        # Align data
        diff_data = diff_data.iloc[1:]
        lagged_data = lagged_data.iloc[1:]
        
        # Simplified cointegration test using eigenvalues
        # In practice, would use proper Johansen test
        X = lagged_data.values
        Y = diff_data.values
        
        # OLS regression
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        residuals = Y - X @ beta
        
        # Check residual stationarity (simplified)
        coint_rank = 0
        coint_vectors = []
        
        # Use correlation matrix eigenvalues as proxy
        corr_matrix = price_data.corr()
        eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
        
        # Count small eigenvalues (potential cointegrating relationships)
        for i, eigenval in enumerate(eigenvalues):
            if eigenval < 0.1:  # Simplified threshold
                coint_rank += 1
                coint_vectors.append(eigenvectors[:, i])
        
        if coint_rank > 0:
            coint_vectors = np.column_stack(coint_vectors)
        else:
            coint_vectors = np.array([[]])
        
        return min(coint_rank, len(price_data.columns) - 1), coint_vectors
    
    def _fit_vecm(self, price_data: pd.DataFrame, coint_rank: int, coint_vectors: np.ndarray) -> Dict:
        """Fit VECM model parameters."""
        # Calculate error correction terms
        if coint_vectors.size == 0:
            error_correction = pd.DataFrame()
            adjustment_speeds = np.array([])
            long_run_matrix = np.array([[]])
        else:
            # Error correction terms = price_data @ cointegrating vectors
            error_correction = pd.DataFrame(
                price_data.values @ coint_vectors,
                index=price_data.index,
                columns=[f'ECT_{i+1}' for i in range(coint_rank)]
            )
            
            # Estimate adjustment speeds (simplified)
            diff_data = price_data.diff().dropna()
            ect_lagged = error_correction.shift(1).dropna()
            
            # Align data
            diff_data = diff_data.iloc[1:]
            ect_lagged = ect_lagged.iloc[1:]
            
            # OLS for adjustment speeds
            adjustment_speeds = np.zeros((len(price_data.columns), coint_rank))
            for i, col in enumerate(price_data.columns):
                y = diff_data[col].values
                X = ect_lagged.values
                if X.shape[0] > 0:
                    adjustment_speeds[i, :] = np.linalg.lstsq(X, y, rcond=None)[0]
            
            # Long-run matrix = adjustment_speeds @ cointegrating_vectors.T
            long_run_matrix = adjustment_speeds @ coint_vectors.T
        
        return {
            'error_correction': error_correction,
            'adjustment_speeds': adjustment_speeds,
            'long_run_matrix': long_run_matrix,
            'cointegrating_vectors': coint_vectors
        }
    
    def _calculate_current_errors(self, price_data: pd.DataFrame, coint_vectors: np.ndarray) -> pd.Series:
        """Calculate current error correction terms."""
        if coint_vectors.size == 0:
            return pd.Series([0], index=['ECT_1'])
        
        current_prices = price_data.iloc[-1].values
        errors = current_prices @ coint_vectors
        
        return pd.Series(errors, index=[f'ECT_{i+1}' for i in range(len(errors))])
    
    def _calculate_impulse_response(self, vecm_params: Dict, horizon: int = 20) -> Dict[str, pd.DataFrame]:
        """Calculate impulse response functions."""
        impulse_responses = {}
        
        # Simplified impulse response based on adjustment speeds
        adjustment_speeds = vecm_params['adjustment_speeds']
        
        if adjustment_speeds.size > 0:
            n_vars = adjustment_speeds.shape[0]
            
            for i in range(n_vars):
                responses = np.zeros((horizon, n_vars))
                
                # Initial shock
                shock = np.zeros(n_vars)
                shock[i] = 1.0
                
                # Propagate shock (simplified)
                for h in range(horizon):
                    if h == 0:
                        responses[h, :] = shock
                    else:
                        # Decay based on adjustment speeds
                        responses[h, :] = responses[h-1, :] * (1 - np.abs(adjustment_speeds[:, 0]) * 0.1)
                
                impulse_responses[f'shock_var_{i+1}'] = pd.DataFrame(
                    responses,
                    columns=[f'response_var_{j+1}' for j in range(n_vars)]
                )
        
        return impulse_responses
    
    def _calculate_halflife_from_adjustment(self, adjustment_speeds: np.ndarray) -> Optional[float]:
        """Calculate half-life from adjustment speeds."""
        if adjustment_speeds.size == 0:
            return None
        
        # Use maximum adjustment speed
        max_speed = np.max(np.abs(adjustment_speeds))
        
        if max_speed > 0:
            # Half-life = -log(2) / log(1 - speed)
            if max_speed < 1:
                half_life = -np.log(2) / np.log(1 - max_speed)
                return min(half_life, 252)
        
        return None
    
    def _select_lag_order(self, price_data: pd.DataFrame) -> int:
        """Select optimal lag order (simplified using AIC)."""
        # Would use proper information criteria in practice
        return min(self.max_lag, len(price_data) // 50)
    
    def _calculate_max_eigenvalue(self, coint_vectors: np.ndarray) -> float:
        """Calculate maximum eigenvalue statistic."""
        if coint_vectors.size == 0:
            return 0.0
        
        # Simplified - return norm of first cointegrating vector
        return np.linalg.norm(coint_vectors[:, 0]) if coint_vectors.ndim > 1 else np.linalg.norm(coint_vectors)
    
    def _calculate_trace_statistic(self, price_data: pd.DataFrame) -> float:
        """Calculate trace statistic (simplified)."""
        # Would implement proper trace test in practice
        eigenvalues = np.linalg.eigvalsh(price_data.corr())
        return -len(price_data) * np.sum(np.log(1 - eigenvalues[eigenvalues < 0.99]))
    
    def _check_model_stability(self, vecm_params: Dict) -> float:
        """Check VECM model stability."""
        adjustment_speeds = vecm_params['adjustment_speeds']
        
        if adjustment_speeds.size == 0:
            return 0.0
        
        # Check if all adjustment speeds are negative (stable)
        stability_ratio = np.sum(adjustment_speeds < 0) / adjustment_speeds.size
        
        return stability_ratio
    
    def _calculate_forecast_error(self, price_data: pd.DataFrame, vecm_params: Dict) -> float:
        """Calculate one-step-ahead forecast error."""
        if len(price_data) < 2:
            return 0.0
        
        # Simplified forecast error
        last_change = price_data.iloc[-1] - price_data.iloc[-2]
        forecast_error = np.mean(np.abs(last_change))
        
        return float(forecast_error)
    
    def _calculate_confidence(self, coint_rank: int, adjustment_speeds: np.ndarray) -> float:
        """Calculate confidence in VECM model."""
        # Cointegration rank confidence
        rank_confidence = min(coint_rank / 3, 1.0) * 0.5
        
        # Adjustment speed significance
        if adjustment_speeds.size > 0:
            significant_speeds = np.sum(np.abs(adjustment_speeds) > 0.05) / adjustment_speeds.size
            speed_confidence = significant_speeds * 0.5
        else:
            speed_confidence = 0.0
        
        return rank_confidence + speed_confidence
    
    def _default_signal(self) -> VECMResults:
        """Return default neutral signal when analysis fails."""
        return VECMResults(
            timestamp=datetime.now(),
            signal_type='neutral',
            strength=0.5,
            z_score=0.0,
            half_life=None,
            confidence=0.0,
            cointegration_rank=0,
            error_correction_terms=pd.DataFrame(),
            adjustment_speeds=np.array([]),
            long_run_matrix=np.array([[]]),
            impulse_responses={},
            portfolio_weights=np.array([]),
            metrics={},
            metadata={'error': 'Insufficient or invalid data'}
        )


# Model factory for easy instantiation
def create_stat_arb_model(model_type: str, **kwargs) -> BaseStatArbModel:
    """Factory function to create statistical arbitrage models."""
    models = {
        'ornstein_uhlenbeck': OrnsteinUhlenbeckModel,
        'kalman_pairs': KalmanPairsModel,
        'pca_strategy': PCAStrategyModel,
        'copula': CopulaModel,
        'vecm': VECMModel
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return models[model_type](**kwargs)