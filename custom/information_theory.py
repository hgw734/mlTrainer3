"""
Custom Information Theory Implementation
=====================================

Replaces PyInform library for Apple Silicon compatibility.
Implements transfer entropy and other information theory measures.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TransferEntropy:
    """Transfer entropy calculation for time series"""

    def __init__(self, history: int = 1):
        self.history = history

        def calculate_transfer_entropy(self, source: np.ndarray, target: np.ndarray) -> float:
            """Calculate transfer entropy from source to target"""
            try:
                if len(source) != len(target):
                    raise ValueError("Source and target must have same length")

                    if len(source) < self.history + 1:
                        raise ValueError(f"Insufficient data: {len(source)} < {self.history + 1}")

                        # Discretize the data for entropy calculation
                        source_disc = self._discretize_series(source)
                        target_disc = self._discretize_series(target)

                        # Calculate joint and conditional entropies
                        te = self._compute_transfer_entropy(source_disc, target_disc)

                        return te

                        except Exception as e:
                            logger.error(f"Error calculating transfer entropy: {e}")
                            return 0.0

                            def _discretize_series(self, series: np.ndarray, bins: int = 10) -> np.ndarray:
                                """Discretize continuous series into bins"""
                                # Use quantile-based discretization
                                percentiles = np.linspace(0, 100, bins + 1)
                                thresholds = np.percentile(series, percentiles)

                                discretized = np.zeros_like(series, dtype=int)
                                for i, val in enumerate(series):
                                    for j, threshold in enumerate(thresholds[:-1]):
                                        if val <= threshold:
                                            discretized[i] = j
                                            break
                                        else:
                                            discretized[i] = bins - 1

                                            return discretized

                                            def _compute_transfer_entropy(self, source: np.ndarray, target: np.ndarray) -> float:
                                                """Compute transfer entropy using entropy calculations"""
                                                n = len(source)

                                                # Create history vectors
                                                target_history = np.zeros((n - self.history, self.history))
                                                for i in range(self.history):
                                                    target_history[:, i] = target[self.history - i - 1 : n - i - 1]

                                                    source_history = np.zeros((n - self.history, self.history))
                                                    for i in range(self.history):
                                                        source_history[:, i] = source[self.history - i - 1 : n - i - 1]

                                                        current_target = target[self.history :]

                                                        # Calculate entropies
                                                        h_current = self._entropy(current_target)
                                                        h_joint = self._joint_entropy(current_target, target_history)
                                                        h_conditional = self._conditional_entropy(current_target, target_history)
                                                        h_joint_with_source = self._joint_entropy(current_target, np.column_stack([target_history, source_history]))
                                                        h_conditional_with_source = self._conditional_entropy(
                                                        current_target, np.column_stack([target_history, source_history])
                                                        )

                                                        # Transfer entropy = H(Y_t|Y_{t-1}) - H(Y_t|Y_{t-1}, X_{t-1})
                                                        te = h_conditional - h_conditional_with_source

                                                        return max(0, te)  # Transfer entropy should be non-negative

                                                        def _entropy(self, data: np.ndarray) -> float:
                                                            """Calculate Shannon entropy"""
                                                            if len(data) == 0:
                                                                return 0.0

                                                                # Count unique values
                                                                unique, counts = np.unique(data, return_counts=True)
                                                                probabilities = counts / len(data)

                                                                # Calculate entropy
                                                                entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
                                                                return entropy

                                                                def _joint_entropy(self, data1: np.ndarray, data2: np.ndarray) -> float:
                                                                    """Calculate joint entropy of two variables"""
                                                                    if len(data1) != len(data2):
                                                                        raise ValueError("Data arrays must have same length")

                                                                        # Create joint states
                                                                        joint_states = list(zip(data1, data2.flatten() if data2.ndim > 1 else data2))

                                                                        # Count unique joint states
                                                                        unique_states, counts = np.unique(joint_states, axis=0, return_counts=True)
                                                                        probabilities = counts / len(joint_states)

                                                                        # Calculate joint entropy
                                                                        joint_entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
                                                                        return joint_entropy

                                                                        def _conditional_entropy(self, data1: np.ndarray, data2: np.ndarray) -> float:
                                                                            """Calculate conditional entropy H(X|Y)"""
                                                                            joint_entropy = self._joint_entropy(data1, data2)
                                                                            marginal_entropy = self._entropy(data2.flatten() if data2.ndim > 1 else data2)

                                                                            conditional_entropy = joint_entropy - marginal_entropy
                                                                            return max(0, conditional_entropy)


                                                                            @dataclass
                                                                            class MutualInformation:
                                                                                """Mutual information calculation"""

                                                                                def __init__(self, bins: int = 10):
                                                                                    self.bins = bins

                                                                                    def calculate_mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
                                                                                        """Calculate mutual information between two variables"""
                                                                                        try:
                                                                                            if len(x) != len(y):
                                                                                                raise ValueError("Arrays must have same length")

                                                                                                # Discretize if continuous
                                                                                                x_disc = self._discretize_series(x, self.bins)
                                                                                                y_disc = self._discretize_series(y, self.bins)

                                                                                                # Calculate mutual information
                                                                                                mi = self._compute_mutual_information(x_disc, y_disc)

                                                                                                return mi

                                                                                                except Exception as e:
                                                                                                    logger.error(f"Error calculating mutual information: {e}")
                                                                                                    return 0.0

                                                                                                    def _discretize_series(self, series: np.ndarray, bins: int) -> np.ndarray:
                                                                                                        """Discretize continuous series"""
                                                                                                        percentiles = np.linspace(0, 100, bins + 1)
                                                                                                        thresholds = np.percentile(series, percentiles)

                                                                                                        discretized = np.zeros_like(series, dtype=int)
                                                                                                        for i, val in enumerate(series):
                                                                                                            for j, threshold in enumerate(thresholds[:-1]):
                                                                                                                if val <= threshold:
                                                                                                                    discretized[i] = j
                                                                                                                    break
                                                                                                                else:
                                                                                                                    discretized[i] = bins - 1

                                                                                                                    return discretized

                                                                                                                    def _compute_mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
                                                                                                                        """Compute mutual information I(X;Y) = H(X) + H(Y) - H(X,Y)"""
                                                                                                                        h_x = self._entropy(x)
                                                                                                                        h_y = self._entropy(y)
                                                                                                                        h_xy = self._joint_entropy(x, y)

                                                                                                                        mi = h_x + h_y - h_xy
                                                                                                                        return max(0, mi)

                                                                                                                        def _entropy(self, data: np.ndarray) -> float:
                                                                                                                            """Calculate Shannon entropy"""
                                                                                                                            if len(data) == 0:
                                                                                                                                return 0.0

                                                                                                                                unique, counts = np.unique(data, return_counts=True)
                                                                                                                                probabilities = counts / len(data)

                                                                                                                                entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
                                                                                                                                return entropy

                                                                                                                                def _joint_entropy(self, data1: np.ndarray, data2: np.ndarray) -> float:
                                                                                                                                    """Calculate joint entropy"""
                                                                                                                                    joint_states = list(zip(data1, data2))
                                                                                                                                    unique_states, counts = np.unique(joint_states, axis=0, return_counts=True)
                                                                                                                                    probabilities = counts / len(joint_states)

                                                                                                                                    joint_entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
                                                                                                                                    return joint_entropy


                                                                                                                                    @dataclass
                                                                                                                                    class GrangerCausality:
                                                                                                                                        """Granger causality analysis implementation"""

                                                                                                                                        def __init__(self, max_lag: int = 4):
                                                                                                                                            self.max_lag = max_lag

                                                                                                                                            def granger_causality_analysis(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
                                                                                                                                                """Perform Granger causality analysis"""
                                                                                                                                                try:
                                                                                                                                                    if len(x) != len(y):
                                                                                                                                                        raise ValueError("Arrays must have same length")

                                                                                                                                                        # Perform Granger causality analysis
                                                                                                                                                        f_stat, p_value = self._compute_granger_causality(x, y)

                                                                                                                                                        return {
                                                                                                                                                        "f_statistic": f_stat,
                                                                                                                                                        "p_value": p_value,
                                                                                                                                                        "significant": p_value < 0.05,
                                                                                                                                                        "causality_direction": "x->y" if p_value < 0.05 else "no_causality",
                                                                                                                                                        }

                                                                                                                                                        except Exception as e:
                                                                                                                                                            logger.error(f"Error in Granger causality analysis: {e}")
                                                                                                                                                            return {"f_statistic": 0.0, "p_value": 1.0, "significant": False, "causality_direction": "error"}

                                                                                                                                                            def _compute_granger_causality(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
                                                                                                                                                                """Compute Granger causality F-statistic and p-value"""
                                                                                                                                                                n = len(x)

                                                                                                                                                                # Create lagged variables
                                                                                                                                                                y_lagged = np.zeros((n - self.max_lag, self.max_lag))
                                                                                                                                                                for i in range(self.max_lag):
                                                                                                                                                                    y_lagged[:, i] = y[self.max_lag - i - 1 : n - i - 1]

                                                                                                                                                                    x_lagged = np.zeros((n - self.max_lag, self.max_lag))
                                                                                                                                                                    for i in range(self.max_lag):
                                                                                                                                                                        x_lagged[:, i] = x[self.max_lag - i - 1 : n - i - 1]

                                                                                                                                                                        y_current = y[self.max_lag :]

                                                                                                                                                                        # Restricted model (without x)
                                                                                                                                                                        ssr_restricted = self._compute_ssr(y_current, y_lagged)

                                                                                                                                                                        # Unrestricted model (with x)
                                                                                                                                                                        ssr_unrestricted = self._compute_ssr(y_current, np.column_stack([y_lagged, x_lagged]))

                                                                                                                                                                        # F-statistic
                                                                                                                                                                        df1 = float(self.max_lag)
                                                                                                                                                                        df2 = float(n - 2 * self.max_lag - 1)

                                                                                                                                                                        f_stat = float(((ssr_restricted - ssr_unrestricted) / df1) / (ssr_unrestricted / df2))

                                                                                                                                                                        # Approximate p-value using F-distribution
                                                                                                                                                                        from scipy.stats import f

                                                                                                                                                                        p_value = float(1.0 - f.cdf(f_stat, df1, df2))

                                                                                                                                                                        return float(f_stat), float(p_value)

                                                                                                                                                                        def _compute_ssr(self, y: np.ndarray, X: np.ndarray) -> float:
                                                                                                                                                                            """Compute sum of squared residuals"""
                                                                                                                                                                            try:
                                                                                                                                                                                # OLS regression
                                                                                                                                                                                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                                                                                                                                                                                y_pred = X @ beta
                                                                                                                                                                                residuals = y - y_pred
                                                                                                                                                                                ssr = float(np.sum(residuals**2))
                                                                                                                                                                                return ssr
                                                                                                                                                                                except:
                                                                                                                                                                                    return np.inf


                                                                                                                                                                                    # Factory functions for compatibility with PyInform API
                                                                                                                                                                                    def transfer_entropy(source: np.ndarray, target: np.ndarray, history: int = 1) -> float:
                                                                                                                                                                                        """Calculate transfer entropy (PyInform compatible)"""
                                                                                                                                                                                        te_calculator = TransferEntropy(history=history)
                                                                                                                                                                                        return te_calculator.calculate_transfer_entropy(source, target)


                                                                                                                                                                                        def mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
                                                                                                                                                                                            """Calculate mutual information (PyInform compatible)"""
                                                                                                                                                                                            mi_calculator = MutualInformation(bins=bins)
                                                                                                                                                                                            return mi_calculator.calculate_mutual_information(x, y)


                                                                                                                                                                                            def granger_causality_analysis(x: np.ndarray, y: np.ndarray, max_lag: int = 4) -> Dict[str, Any]:
                                                                                                                                                                                                """Perform Granger causality analysis (statsmodels compatible)"""
                                                                                                                                                                                                gc_calculator = GrangerCausality(max_lag=max_lag)
                                                                                                                                                                                                return gc_calculator.granger_causality_analysis(x, y)
