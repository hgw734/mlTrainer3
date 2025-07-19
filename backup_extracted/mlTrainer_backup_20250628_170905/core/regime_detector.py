
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class MultidimensionalRegimeDetector:
    """Advanced multidimensional market regime detection"""
    
    def __init__(self, lookback_period: int = 252):
        self.lookback_period = lookback_period
        self.current_regime = None
        self.regime_history = []
        self.scaler = MinMaxScaler()
        self.pca = PCA(n_components=3)
        self.regime_clusters = KMeans(n_clusters=8, random_state=42)
        self.is_fitted = False
        
    def extract_regime_features(self, prices: pd.Series) -> pd.DataFrame:
        """Extract comprehensive regime features across multiple dimensions"""
        try:
            returns = prices.pct_change().dropna()
            
            # Volatility dimension (multiple timeframes)
            vol_features = {
                'vol_1d': returns.rolling(1).std() * np.sqrt(252),
                'vol_5d': returns.rolling(5).std() * np.sqrt(252), 
                'vol_20d': returns.rolling(20).std() * np.sqrt(252),
                'vol_60d': returns.rolling(60).std() * np.sqrt(252),
                'vol_regime_score': self._calculate_vol_regime_score(returns)
            }
            
            # Trend dimension (multiple timeframes and methods)
            trend_features = {
                'trend_5d': self._calculate_trend_strength(prices, 5),
                'trend_20d': self._calculate_trend_strength(prices, 20),
                'trend_60d': self._calculate_trend_strength(prices, 60),
                'momentum_rsi': self._calculate_rsi(prices, 14),
                'momentum_macd': self._calculate_macd_signal(prices),
                'trend_regime_score': self._calculate_trend_regime_score(prices)
            }
            
            # Distribution dimension
            dist_features = {
                'skewness_20d': returns.rolling(20).skew(),
                'kurtosis_20d': returns.rolling(20).kurt(),
                'skewness_60d': returns.rolling(60).skew(), 
                'kurtosis_60d': returns.rolling(60).kurt(),
                'tail_risk': self._calculate_tail_risk(returns),
                'distribution_regime_score': self._calculate_distribution_regime_score(returns)
            }
            
            # Market structure dimension
            structure_features = {
                'autocorr_lag1': returns.rolling(60).apply(lambda x: x.autocorr(lag=1)),
                'autocorr_lag5': returns.rolling(60).apply(lambda x: x.autocorr(lag=5)),
                'hurst_exponent': returns.rolling(60).apply(self._calculate_hurst),
                'fractal_dimension': returns.rolling(60).apply(self._calculate_fractal_dim),
                'structure_regime_score': self._calculate_structure_regime_score(returns)
            }
            
            # Combine all features
            all_features = {**vol_features, **trend_features, **dist_features, **structure_features}
            feature_df = pd.DataFrame(all_features, index=prices.index)
            
            return feature_df.fillna(method='ffill').dropna()
            
        except Exception as e:
            logger.error(f"❌ Feature extraction failed: {e}")
            return pd.DataFrame()
    
    def _calculate_vol_regime_score(self, returns: pd.Series, window: int = 60) -> pd.Series:
        """Calculate continuous volatility regime score (0-100)"""
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        vol_percentile = rolling_vol.rolling(252).rank(pct=True) * 100
        return vol_percentile.fillna(50)
    
    def _calculate_trend_regime_score(self, prices: pd.Series, window: int = 60) -> pd.Series:
        """Calculate continuous trend regime score (0-100)"""
        # Multiple moving averages
        sma_20 = prices.rolling(20).mean()
        sma_50 = prices.rolling(50).mean()
        sma_200 = prices.rolling(200).mean()
        
        # Trend strength components
        price_vs_sma20 = (prices / sma_20 - 1) * 100
        sma20_vs_sma50 = (sma_20 / sma_50 - 1) * 100
        sma50_vs_sma200 = (sma_50 / sma_200 - 1) * 100
        
        # Weighted trend score
        trend_score = (price_vs_sma20 * 0.5 + sma20_vs_sma50 * 0.3 + sma50_vs_sma200 * 0.2)
        
        # Normalize to 0-100 (50 = neutral)
        trend_percentile = trend_score.rolling(252).rank(pct=True) * 100
        return trend_percentile.fillna(50)
    
    def _calculate_distribution_regime_score(self, returns: pd.Series, window: int = 60) -> pd.Series:
        """Calculate distribution regime score based on skewness and kurtosis"""
        skew = returns.rolling(window).skew()
        kurt = returns.rolling(window).kurt()
        
        # Normalize skewness and kurtosis
        skew_norm = (skew + 2) / 4 * 100  # Assume range [-2, 2]
        kurt_norm = (kurt.clip(0, 10)) / 10 * 100  # Clip extreme kurtosis
        
        # Combined distribution score
        dist_score = (skew_norm * 0.6 + kurt_norm * 0.4)
        return dist_score.fillna(50)
    
    def _calculate_structure_regime_score(self, returns: pd.Series, window: int = 60) -> pd.Series:
        """Calculate market structure regime score"""
        autocorr = returns.rolling(window).apply(lambda x: x.autocorr(lag=1))
        autocorr_norm = (autocorr + 1) / 2 * 100  # Normalize [-1,1] to [0,100]
        return autocorr_norm.fillna(50)
    
    def _calculate_trend_strength(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate trend strength using linear regression slope"""
        def trend_slope(x):
            if len(x) < window:
                return 0
            return np.polyfit(range(len(x)), x, 1)[0]
        
        return prices.rolling(window).apply(trend_slope)
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI momentum indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd_signal(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD signal line"""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9).mean()
        return macd_line - signal_line
    
    def _calculate_tail_risk(self, returns: pd.Series, window: int = 60) -> pd.Series:
        """Calculate tail risk using VaR"""
        return returns.rolling(window).quantile(0.05)
    
    def _calculate_hurst(self, x):
        """Calculate Hurst exponent for mean reversion/momentum detection"""
        try:
            if len(x) < 10:
                return 0.5
            lags = range(2, min(20, len(x)//2))
            tau = [np.std(np.diff(x, n)) for n in lags]
            return np.polyfit(np.log(lags), np.log(tau), 1)[0]
        except:
            return 0.5
    
    def _calculate_fractal_dim(self, x):
        """Calculate fractal dimension"""
        try:
            if len(x) < 10:
                return 1.5
            n = len(x)
            # Simplified box-counting method
            return 2 - self._calculate_hurst(x)
        except:
            return 1.5
    
    def detect_multidimensional_regime(self, prices: pd.Series) -> Dict[str, Any]:
        """Detect regime using multidimensional continuous approach"""
        try:
            # Extract all regime features
            features_df = self.extract_regime_features(prices)
            
            if features_df.empty:
                return {"error": "Failed to extract features"}
            
            # Get latest feature vector
            latest_features = features_df.iloc[-1]
            
            # Calculate composite regime scores for each dimension
            regime_dimensions = {
                'volatility_score': latest_features['vol_regime_score'],
                'trend_score': latest_features['trend_regime_score'], 
                'distribution_score': latest_features['distribution_regime_score'],
                'structure_score': latest_features['structure_regime_score']
            }
            
            # Calculate market stress indicator
            stress_components = [
                latest_features['vol_60d'],
                abs(latest_features['skewness_60d']),
                latest_features['kurtosis_60d'],
                abs(latest_features['tail_risk'])
            ]
            market_stress = np.mean([self._normalize_to_percentile(comp, features_df.iloc[-252:]) 
                                   for comp in stress_components if not np.isnan(comp)])
            
            # Calculate regime stability
            recent_vol = features_df['vol_regime_score'].iloc[-20:].std()
            recent_trend = features_df['trend_regime_score'].iloc[-20:].std()
            regime_stability = 100 - min(100, (recent_vol + recent_trend) * 2)
            
            # Create comprehensive regime profile
            regime_profile = {
                **regime_dimensions,
                'market_stress': float(market_stress),
                'regime_stability': float(regime_stability),
                'composite_regime_vector': [
                    regime_dimensions['volatility_score'],
                    regime_dimensions['trend_score'],
                    regime_dimensions['distribution_score'], 
                    regime_dimensions['structure_score']
                ],
                'timestamp': datetime.now().isoformat(),
                'confidence': self._calculate_multidim_confidence(regime_dimensions)
            }
            
            # Store regime
            self.current_regime = regime_profile
            self.regime_history.append(regime_profile)
            
            logger.info(f"🎯 Multidimensional regime: Vol={regime_dimensions['volatility_score']:.1f}, "
                       f"Trend={regime_dimensions['trend_score']:.1f}, "
                       f"Stress={market_stress:.1f}, Stability={regime_stability:.1f}")
            
            return regime_profile
            
        except Exception as e:
            logger.error(f"❌ Multidimensional regime detection failed: {e}")
            return {"error": str(e)}
    
    def _normalize_to_percentile(self, value: float, reference_series: pd.Series) -> float:
        """Normalize a value to percentile within reference series"""
        if pd.isna(value) or len(reference_series) == 0:
            return 50
        return (reference_series <= value).mean() * 100
    
    def _calculate_multidim_confidence(self, regime_dims: Dict[str, float]) -> float:
        """Calculate confidence based on regime dimension consistency"""
        scores = list(regime_dims.values())
        consistency = 100 - np.std(scores)  # Higher consistency = higher confidence
        return max(0.1, min(1.0, consistency / 100))
    
    def get_adaptive_strategy_config(self, regime_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Get strategy configuration using multidimensional model activation"""
        try:
            from model_activation import get_active_models, get_regime_specific_parameters
            
            # Get multidimensional model activation
            model_config = get_active_models(regime_profile)
            regime_params = get_regime_specific_parameters(regime_profile)
            
            vol_score = regime_profile['volatility_score']
            trend_score = regime_profile['trend_score']
            stress_score = regime_profile['market_stress']
            stability_score = regime_profile['regime_stability']
            
            # Position sizing based on regime stability and stress
            base_position = 0.5
            stability_factor = stability_score / 100
            stress_factor = max(0.2, (100 - stress_score) / 100)
            position_size = base_position * stability_factor * stress_factor
            
            # Risk profile based on regime classification
            regime_type = model_config['regime_classification']
            if regime_type in ['CRISIS', 'HIGH_VOL_STRESS']:
                risk_profile = "defensive"
            elif regime_type in ['LOW_VOL_STABLE', 'MEDIUM_VOL_STABLE']:
                risk_profile = "balanced"
            elif regime_type in ['HIGH_VOL_MOMENTUM', 'STRONG_TREND']:
                risk_profile = "aggressive"
            else:
                risk_profile = "adaptive"
            
            return {
                "active_models": model_config['active_models'],
                "model_weights": model_config['model_weights'],
                "regime_classification": model_config['regime_classification'],
                "position_size": round(position_size, 3),
                "risk_profile": risk_profile,
                "stop_loss": max(0.01, regime_params['stop_loss_factor'] * 0.02),
                "take_profit": min(0.25, 0.05 + trend_score/500),
                "rebalance_frequency": regime_params['rebalance_frequency'],
                "regime_vector": regime_profile['composite_regime_vector'],
                "hyperparameters": regime_params,
                "activation_reasoning": model_config['activation_reasoning']
            }
            
        except ImportError:
            logger.warning("⚠️ model_activation.py not found, using fallback strategy")
            # Fallback to simplified logic
            return {
                "active_models": ["XGBoost", "LSTM", "EnsembleVoting"],
                "model_weights": {"XGBoost": 0.4, "LSTM": 0.4, "EnsembleVoting": 0.2},
                "regime_classification": "MIXED_CONDITIONS",
                "position_size": 0.5,
                "risk_profile": "balanced",
                "regime_vector": regime_profile['composite_regime_vector']
            }

# Global instance
regime_detector = MultidimensionalRegimeDetector()
