
import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class ExplainableFinancialModel:
    """
    Explainable AI framework for financial predictions
    Maintains transparency across different market regimes
    Based on research recommendations for XAI in financial engineering
    """
    
    def __init__(self, base_model=None):
        self.base_model = base_model or RandomForestRegressor(n_estimators=100, random_state=42)
        self.feature_names = None
        self.explainer = None
        self.lime_explainer = None
        self.regime_explanations = {}
        
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
        """Fit the model and initialize explainers"""
        logger.info("🔍 Training Explainable Financial Model")
        
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        # Train base model
        self.base_model.fit(X, y)
        
        # Initialize SHAP explainer
        if hasattr(self.base_model, 'predict'):
            self.explainer = shap.Explainer(self.base_model)
            
        # Initialize LIME explainer
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X, feature_names=self.feature_names, mode='regression'
        )
        
        logger.info("✅ Explainable model training complete")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.base_model.predict(X)
    
    def explain_prediction(self, X_sample: np.ndarray, method: str = 'shap') -> Dict[str, Any]:
        """
        Explain a single prediction using SHAP or LIME
        
        Args:
            X_sample: Single sample to explain (1D array)
            method: 'shap' or 'lime'
        """
        if method == 'shap' and self.explainer is not None:
            return self._explain_with_shap(X_sample)
        elif method == 'lime' and self.lime_explainer is not None:
            return self._explain_with_lime(X_sample)
        else:
            logger.error(f"❌ Explanation method '{method}' not available")
            return {}
    
    def _explain_with_shap(self, X_sample: np.ndarray) -> Dict[str, Any]:
        """Generate SHAP explanations"""
        try:
            shap_values = self.explainer(X_sample.reshape(1, -1))
            
            explanation = {
                'method': 'shap',
                'prediction': self.predict(X_sample.reshape(1, -1))[0],
                'base_value': shap_values.base_values[0],
                'shap_values': shap_values.values[0],
                'feature_importance': dict(zip(self.feature_names, shap_values.values[0])),
                'most_important_features': self._get_top_features(shap_values.values[0], 5)
            }
            
            return explanation
        except Exception as e:
            logger.error(f"❌ SHAP explanation failed: {e}")
            return {}
    
    def _explain_with_lime(self, X_sample: np.ndarray) -> Dict[str, Any]:
        """Generate LIME explanations"""
        try:
            explanation = self.lime_explainer.explain_instance(
                X_sample, self.base_model.predict, num_features=len(self.feature_names)
            )
            
            # Extract feature importance
            feature_importance = dict(explanation.as_list())
            
            return {
                'method': 'lime',
                'prediction': self.predict(X_sample.reshape(1, -1))[0],
                'feature_importance': feature_importance,
                'most_important_features': explanation.as_list()[:5],
                'score': explanation.score
            }
        except Exception as e:
            logger.error(f"❌ LIME explanation failed: {e}")
            return {}
    
    def _get_top_features(self, shap_values: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Get top K most important features"""
        feature_importance = list(zip(self.feature_names, shap_values))
        return sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)[:top_k]
    
    def explain_regime_differences(self, X_data: np.ndarray, regime_labels: np.ndarray) -> Dict[str, Any]:
        """
        Explain how model behavior changes across different regimes
        
        Args:
            X_data: Full dataset
            regime_labels: Regime labels for each sample
        """
        logger.info("🔄 Analyzing regime-specific explanations")
        
        regime_explanations = {}
        unique_regimes = np.unique(regime_labels)
        
        for regime in unique_regimes:
            regime_mask = regime_labels == regime
            regime_data = X_data[regime_mask]
            
            if len(regime_data) == 0:
                continue
                
            # Sample a few examples from this regime
            n_samples = min(50, len(regime_data))
            sample_indices = np.random.choice(len(regime_data), n_samples, replace=False)
            
            regime_shap_values = []
            
            for idx in sample_indices:
                try:
                    if self.explainer is not None:
                        shap_vals = self.explainer(regime_data[idx:idx+1])
                        regime_shap_values.append(shap_vals.values[0])
                except:
                    continue
            
            if regime_shap_values:
                # Average SHAP values for this regime
                avg_shap = np.mean(regime_shap_values, axis=0)
                
                regime_explanations[f"regime_{regime}"] = {
                    'avg_feature_importance': dict(zip(self.feature_names, avg_shap)),
                    'top_features': self._get_top_features(avg_shap, 5),
                    'sample_count': len(regime_data),
                    'regime_characteristics': self._characterize_regime(avg_shap)
                }
        
        # Compare regimes
        regime_comparison = self._compare_regimes(regime_explanations)
        
        self.regime_explanations = regime_explanations
        
        return {
            'regime_explanations': regime_explanations,
            'regime_comparison': regime_comparison,
            'summary': self._generate_regime_summary(regime_explanations)
        }
    
    def _characterize_regime(self, avg_shap: np.ndarray) -> Dict[str, str]:
        """Characterize a regime based on feature importance"""
        top_features = self._get_top_features(avg_shap, 3)
        
        characteristics = {}
        for feature_name, importance in top_features:
            if 'volatility' in feature_name.lower():
                characteristics['volatility_driven'] = importance > 0
            elif 'trend' in feature_name.lower():
                characteristics['trend_driven'] = importance > 0
            elif 'momentum' in feature_name.lower():
                characteristics['momentum_driven'] = importance > 0
        
        return characteristics
    
    def _compare_regimes(self, regime_explanations: Dict) -> Dict[str, Any]:
        """Compare feature importance across regimes"""
        if len(regime_explanations) < 2:
            return {}
        
        # Create feature importance matrix
        regimes = list(regime_explanations.keys())
        feature_matrix = []
        
        for regime in regimes:
            importances = [regime_explanations[regime]['avg_feature_importance'].get(feat, 0) 
                          for feat in self.feature_names]
            feature_matrix.append(importances)
        
        feature_matrix = np.array(feature_matrix)
        
        # Calculate regime differences
        regime_distances = {}
        for i, regime1 in enumerate(regimes):
            for j, regime2 in enumerate(regimes[i+1:], i+1):
                distance = np.linalg.norm(feature_matrix[i] - feature_matrix[j])
                regime_distances[f"{regime1}_vs_{regime2}"] = distance
        
        return {
            'feature_importance_matrix': feature_matrix.tolist(),
            'regime_distances': regime_distances,
            'most_different_regimes': max(regime_distances.items(), key=lambda x: x[1])
        }
    
    def _generate_regime_summary(self, regime_explanations: Dict) -> str:
        """Generate human-readable summary of regime differences"""
        if not regime_explanations:
            return "No regime explanations available"
        
        summary = "Regime Analysis Summary:\n"
        
        for regime, explanation in regime_explanations.items():
            top_feature = explanation['top_features'][0] if explanation['top_features'] else ('unknown', 0)
            summary += f"- {regime}: Driven primarily by {top_feature[0]} (importance: {top_feature[1]:.3f})\n"
        
        return summary
    
    def generate_explanation_report(self, output_path: str = None) -> str:
        """Generate a comprehensive explanation report"""
        if not self.regime_explanations:
            return "No regime explanations available. Run explain_regime_differences() first."
        
        report = "EXPLAINABLE AI FINANCIAL MODEL REPORT\n"
        report += "="*50 + "\n\n"
        
        report += "REGIME-SPECIFIC EXPLANATIONS:\n"
        report += "-"*30 + "\n"
        
        for regime, explanation in self.regime_explanations.items():
            report += f"\n{regime.upper()}:\n"
            report += f"Sample Count: {explanation['sample_count']}\n"
            report += "Top Features:\n"
            
            for feature, importance in explanation['top_features']:
                report += f"  - {feature}: {importance:.4f}\n"
            
            report += f"Characteristics: {explanation['regime_characteristics']}\n"
        
        report += f"\nSUMMARY:\n{self._generate_regime_summary(self.regime_explanations)}"
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"📄 Explanation report saved to {output_path}")
        
        return report

def create_explainable_pipeline(price_data: pd.Series, features: pd.DataFrame, 
                               regime_data: pd.DataFrame) -> ExplainableFinancialModel:
    """
    Create an explainable AI pipeline for financial predictions
    
    Args:
        price_data: Price time series
        features: Feature matrix
        regime_data: Regime classification data
    """
    logger.info("🔬 Creating Explainable AI Pipeline")
    
    # Prepare target (next period returns)
    returns = price_data.pct_change().dropna()
    targets = returns.shift(-1).dropna()
    
    # Align data
    min_length = min(len(features), len(targets), len(regime_data))
    X = features.iloc[:min_length].values
    y = targets.iloc[:min_length].values
    regime_labels = regime_data.iloc[:min_length]['regime_score'].apply(
        lambda x: 'bull' if x > 70 else ('bear' if x < 30 else 'neutral')
    ).values
    
    # Create and train explainable model
    model = ExplainableFinancialModel()
    model.fit(X, y, feature_names=features.columns.tolist())
    
    # Generate regime explanations
    model.explain_regime_differences(X, regime_labels)
    
    logger.info("✅ Explainable AI pipeline ready")
    return model
