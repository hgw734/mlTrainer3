
import numpy as np
import logging
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from typing import Dict, Any, List
from core.compliance_mode import enforce_api_compliance

logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    """Advanced hyperparameter optimization for trading models"""
    
    def __init__(self):
        self.optimization_history = []
        
    def grid_search_optimization(self, model, param_grid: Dict, X_train, y_train, cv=5):
        """Systematic grid search optimization"""
        try:
            logger.info(f"🔍 Starting grid search optimization with {len(param_grid)} parameters")
            
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            result = {
                "method": "grid_search",
                "best_params": grid_search.best_params_,
                "best_score": grid_search.best_score_,
                "cv_results": grid_search.cv_results_
            }
            
            self.optimization_history.append(result)
            logger.info(f"✅ Grid search complete. Best score: {grid_search.best_score_:.4f}")
            
            return grid_search.best_estimator_, result
            
        except Exception as e:
            logger.error(f"❌ Grid search failed: {e}")
            return model, {"error": str(e)}
    
    def bayesian_optimization(self, model, search_spaces: Dict, X_train, y_train, n_iter=50):
        """Bayesian optimization for efficient parameter search"""
        try:
            logger.info(f"🎯 Starting Bayesian optimization with {n_iter} iterations")
            
            bayes_search = BayesSearchCV(
                estimator=model,
                search_spaces=search_spaces,
                n_iter=n_iter,
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                random_state=42
            )
            
            bayes_search.fit(X_train, y_train)
            
            result = {
                "method": "bayesian",
                "best_params": bayes_search.best_params_,
                "best_score": bayes_search.best_score_,
                "optimization_results": bayes_search.cv_results_
            }
            
            self.optimization_history.append(result)
            logger.info(f"✅ Bayesian optimization complete. Best score: {bayes_search.best_score_:.4f}")
            
            return bayes_search.best_estimator_, result
            
        except Exception as e:
            logger.error(f"❌ Bayesian optimization failed: {e}")
            return model, {"error": str(e)}
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of all optimization runs"""
        return {
            "total_runs": len(self.optimization_history),
            "methods_used": list(set([run["method"] for run in self.optimization_history])),
            "best_overall": max(self.optimization_history, key=lambda x: x.get("best_score", -np.inf)) if self.optimization_history else None,
            "history": self.optimization_history
        }

hyperparameter_optimizer = HyperparameterOptimizer()
