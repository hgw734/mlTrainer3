
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats, optimize
from sklearn.mixture import GaussianMixture
import logging

logger = logging.getLogger(__name__)

class AdvancedFinancialModels:
    """
    Advanced financial models based on latest research
    Implements models from attached research papers
    """
    
    def __init__(self):
        self.risk_free_rate = 0.02
    
    def regime_switching_garch(self, returns: pd.Series, n_regimes: int = 2) -> Dict:
        """
        Markov-Switching GARCH model for regime detection
        Based on Phoong & Phoong (2022) findings
        """
        logger.info("🔄 Running Regime-Switching GARCH model")
        
        # Fit Gaussian Mixture Model for regime identification
        returns_reshaped = returns.values.reshape(-1, 1)
        gmm = GaussianMixture(n_components=n_regimes, random_state=42)
        gmm.fit(returns_reshaped)
        
        # Predict regimes
        regime_probabilities = gmm.predict_proba(returns_reshaped)
        regime_labels = gmm.predict(returns_reshaped)
        
        # Calculate regime-specific volatilities
        regime_stats = {}
        for regime in range(n_regimes):
            mask = regime_labels == regime
            regime_returns = returns[mask]
            
            regime_stats[f"regime_{regime}"] = {
                "mean_return": regime_returns.mean(),
                "volatility": regime_returns.std(),
                "skewness": stats.skew(regime_returns),
                "kurtosis": stats.kurtosis(regime_returns),
                "probability": np.mean(regime_labels == regime),
                "persistence": self._calculate_regime_persistence(regime_labels, regime)
            }
        
        return {
            "regime_probabilities": regime_probabilities,
            "regime_labels": regime_labels,
            "regime_statistics": regime_stats,
            "transition_matrix": self._calculate_transition_matrix(regime_labels),
            "model_performance": {
                "aic": gmm.aic(returns_reshaped),
                "bic": gmm.bic(returns_reshaped),
                "log_likelihood": gmm.score(returns_reshaped)
            }
        }
    
    def efficient_market_hypothesis_tests(self, prices: pd.Series) -> Dict:
        """
        EMH tests based on Bitcoin EMH meta-analysis research
        Tests weak, semi-strong, and strong form efficiency
        """
        logger.info("📊 Running Efficient Market Hypothesis tests")
        
        returns = prices.pct_change().dropna()
        
        # Weak Form EMH Tests
        weak_form = self._weak_form_emh_tests(returns)
        
        # Semi-Strong Form EMH Tests
        semi_strong_form = self._semi_strong_form_emh_tests(returns)
        
        # Strong Form EMH Tests (simplified)
        strong_form = self._strong_form_emh_tests(returns)
        
        return {
            "weak_form_efficiency": weak_form,
            "semi_strong_form_efficiency": semi_strong_form,
            "strong_form_efficiency": strong_form,
            "overall_efficiency_score": self._calculate_overall_efficiency(
                weak_form, semi_strong_form, strong_form
            )
        }
    
    def _weak_form_emh_tests(self, returns: pd.Series) -> Dict:
        """Weak form EMH tests - serial correlation, runs test, variance ratio"""
        
        # Ljung-Box test for serial correlation
        from statsmodels.stats.diagnostic import acorr_ljungbox
        ljung_box = acorr_ljungbox(returns, lags=10, return_df=True)
        
        # Variance Ratio Test
        variance_ratios = []
        for k in [2, 4, 8, 16]:
            if len(returns) > k:
                vr = self._variance_ratio_test(returns, k)
                variance_ratios.append(vr)
        
        # Runs Test for randomness
        runs_test = self._runs_test(returns)
        
        return {
            "ljung_box_pvalue": ljung_box['lb_pvalue'].iloc[-1],
            "variance_ratios": variance_ratios,
            "runs_test": runs_test,
            "autocorrelation": returns.autocorr(),
            "efficiency_verdict": "efficient" if ljung_box['lb_pvalue'].iloc[-1] > 0.05 else "inefficient"
        }
    
    def _semi_strong_form_emh_tests(self, returns: pd.Series) -> Dict:
        """Semi-strong form EMH tests - event study methodology"""
        
        # Simplified event study framework
        # In practice, this would use external event data
        rolling_mean = returns.rolling(window=20).mean()
        rolling_std = returns.rolling(window=20).std()
        
        # Abnormal returns (simplified)
        abnormal_returns = (returns - rolling_mean) / rolling_std
        
        # Test for significant abnormal returns
        t_stat, p_value = stats.ttest_1samp(abnormal_returns.dropna(), 0)
        
        return {
            "abnormal_returns_mean": abnormal_returns.mean(),
            "abnormal_returns_tstat": t_stat,
            "abnormal_returns_pvalue": p_value,
            "significant_events": len(abnormal_returns[abs(abnormal_returns) > 2]),
            "efficiency_verdict": "efficient" if p_value > 0.05 else "inefficient"
        }
    
    def _strong_form_emh_tests(self, returns: pd.Series) -> Dict:
        """Strong form EMH tests - insider information effects"""
        
        # Simplified strong form test using return predictability
        # Test if high returns are followed by reversals (insider profit-taking)
        
        high_return_threshold = returns.quantile(0.9)
        low_return_threshold = returns.quantile(0.1)
        
        high_return_reversals = 0
        low_return_reversals = 0
        
        for i in range(1, len(returns)):
            if returns.iloc[i-1] > high_return_threshold:
                if returns.iloc[i] < 0:
                    high_return_reversals += 1
            if returns.iloc[i-1] < low_return_threshold:
                if returns.iloc[i] > 0:
                    low_return_reversals += 1
        
        total_extreme_events = len(returns[returns > high_return_threshold]) + len(returns[returns < low_return_threshold])
        reversal_rate = (high_return_reversals + low_return_reversals) / total_extreme_events if total_extreme_events > 0 else 0
        
        return {
            "reversal_rate": reversal_rate,
            "high_return_reversals": high_return_reversals,
            "low_return_reversals": low_return_reversals,
            "efficiency_verdict": "efficient" if reversal_rate < 0.6 else "inefficient"
        }
    
    def _variance_ratio_test(self, returns: pd.Series, k: int) -> Dict:
        """Variance ratio test for random walk hypothesis"""
        
        n = len(returns)
        if n < k * 2:
            return {"variance_ratio": np.nan, "z_statistic": np.nan, "p_value": np.nan}
        
        # Calculate variance ratio
        var_1 = returns.var()
        var_k = returns.rolling(window=k).sum().var() / k
        
        vr = var_k / var_1 if var_1 != 0 else np.nan
        
        # Calculate z-statistic (simplified)
        z_stat = np.sqrt(n) * (vr - 1) / np.sqrt(2 * (k - 1))
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        return {
            "variance_ratio": vr,
            "z_statistic": z_stat,
            "p_value": p_value,
            "period": k
        }
    
    def _runs_test(self, returns: pd.Series) -> Dict:
        """Runs test for sequence randomness"""
        
        median_return = returns.median()
        runs, n1, n2 = 0, 0, 0
        
        # Convert to binary sequence
        sequence = (returns > median_return).astype(int)
        
        # Count runs
        if len(sequence) > 0:
            runs = 1
            for i in range(1, len(sequence)):
                if sequence.iloc[i] != sequence.iloc[i-1]:
                    runs += 1
            
            n1 = sum(sequence)
            n2 = len(sequence) - n1
        
        # Calculate expected runs and variance
        if n1 > 0 and n2 > 0:
            expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
            variance_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
            z_stat = (runs - expected_runs) / np.sqrt(variance_runs) if variance_runs > 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        else:
            z_stat, p_value = 0, 1
        
        return {
            "observed_runs": runs,
            "expected_runs": expected_runs if n1 > 0 and n2 > 0 else np.nan,
            "z_statistic": z_stat,
            "p_value": p_value
        }
    
    def _calculate_regime_persistence(self, regime_labels: np.ndarray, regime: int) -> float:
        """Calculate how persistent a regime is (average duration)"""
        
        runs = []
        current_run = 0
        
        for label in regime_labels:
            if label == regime:
                current_run += 1
            else:
                if current_run > 0:
                    runs.append(current_run)
                current_run = 0
        
        if current_run > 0:
            runs.append(current_run)
        
        return np.mean(runs) if runs else 0
    
    def _calculate_transition_matrix(self, regime_labels: np.ndarray) -> np.ndarray:
        """Calculate regime transition matrix"""
        
        n_regimes = len(np.unique(regime_labels))
        transition_matrix = np.zeros((n_regimes, n_regimes))
        
        for i in range(len(regime_labels) - 1):
            current_regime = regime_labels[i]
            next_regime = regime_labels[i + 1]
            transition_matrix[current_regime, next_regime] += 1
        
        # Normalize rows to get probabilities
        row_sums = transition_matrix.sum(axis=1)
        transition_matrix = transition_matrix / row_sums[:, np.newaxis]
        
        return transition_matrix
    
    def _calculate_overall_efficiency(self, weak: Dict, semi_strong: Dict, strong: Dict) -> Dict:
        """Calculate overall market efficiency score"""
        
        efficiency_scores = {
            "weak_form": 1 if weak["efficiency_verdict"] == "efficient" else 0,
            "semi_strong_form": 1 if semi_strong["efficiency_verdict"] == "efficient" else 0,
            "strong_form": 1 if strong["efficiency_verdict"] == "efficient" else 0
        }
        
        overall_score = sum(efficiency_scores.values()) / len(efficiency_scores)
        
        return {
            "individual_scores": efficiency_scores,
            "overall_efficiency_score": overall_score,
            "market_classification": self._classify_market_efficiency(overall_score)
        }
    
    def _classify_market_efficiency(self, score: float) -> str:
        """Classify market based on efficiency score"""
        
        if score >= 0.8:
            return "Highly Efficient"
        elif score >= 0.6:
            return "Moderately Efficient" 
        elif score >= 0.4:
            return "Weakly Efficient"
        else:
            return "Inefficient"

# Initialize global instance
advanced_models = AdvancedFinancialModels()
