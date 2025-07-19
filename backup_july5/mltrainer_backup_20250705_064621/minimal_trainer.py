#!/usr/bin/env python3
"""
Minimal Model Trainer - Pure Python approach bypassing pandas/numpy dependency issues
"""

import os
import sys
import json
import random
import math
from datetime import datetime

def create_minimal_data():
    """Create training data without pandas/numpy"""
    random.seed(42)
    n_samples = 200
    
    # Create synthetic stock-like data
    data = []
    for i in range(n_samples):
        open_price = random.uniform(100, 200)
        high_price = random.uniform(open_price, 220)
        low_price = random.uniform(80, open_price)
        close_price = random.uniform(low_price, high_price)
        volume = random.uniform(1000000, 10000000)
        
        # Calculate basic indicators
        sma_5 = close_price + random.uniform(-5, 5)
        sma_10 = close_price + random.uniform(-10, 10)
        rsi = random.uniform(20, 80)
        macd = random.uniform(-2, 2)
        volatility = random.uniform(0.01, 0.05)
        
        # Target (next day return)
        target = random.uniform(-0.05, 0.05)
        
        data.append([
            open_price, high_price, low_price, close_price, volume,
            sma_5, sma_10, rsi, macd, volatility, target
        ])
    
    # Split features and target
    X = [row[:-1] for row in data]  # All features except target
    y = [row[-1] for row in data]   # Target only
    
    return X, y

def calculate_accuracy(y_true, y_pred):
    """Calculate simple accuracy metric"""
    if len(y_true) != len(y_pred):
        return 0.0
    
    # For regression, use 1 - normalized RMSE
    mse = sum((true - pred) ** 2 for true, pred in zip(y_true, y_pred)) / len(y_true)
    rmse = math.sqrt(mse)
    
    # Normalize by target variance
    target_var = sum((y - sum(y_true)/len(y_true)) ** 2 for y in y_true) / len(y_true)
    target_std = math.sqrt(target_var)
    
    if target_std == 0:
        return 0.0
    
    normalized_rmse = rmse / target_std
    accuracy = max(0, min(1, 1 - normalized_rmse))
    
    return accuracy

def train_minimal_models():
    """Train models using minimal approach"""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Starting minimal model training")
    
    # Create directories
    os.makedirs('models/sklearn', exist_ok=True)
    
    # Create training data
    X, y = create_minimal_data()
    print(f"Training data: {len(X)} samples, {len(X[0])} features")
    
    trained_models = {}
    failed_models = {}
    
    # Try to import and train models one by one
    model_configs = [
        ('LinearRegression', 'from sklearn.linear_model import LinearRegression; model = LinearRegression()'),
        ('Ridge', 'from sklearn.linear_model import Ridge; model = Ridge(alpha=1.0)'),
        ('Lasso', 'from sklearn.linear_model import Lasso; model = Lasso(alpha=1.0)'),
        ('ElasticNet', 'from sklearn.linear_model import ElasticNet; model = ElasticNet(alpha=1.0)'),
        ('BayesianRidge', 'from sklearn.linear_model import BayesianRidge; model = BayesianRidge()'),
        ('SGDRegressor', 'from sklearn.linear_model import SGDRegressor; model = SGDRegressor()'),
        ('RandomForest', 'from sklearn.ensemble import RandomForestRegressor; model = RandomForestRegressor(n_estimators=10, random_state=42)'),
        ('DecisionTree', 'from sklearn.tree import DecisionTreeRegressor; model = DecisionTreeRegressor(random_state=42)'),
        ('SVR', 'from sklearn.svm import SVR; model = SVR(kernel="linear")'),
        ('KNeighbors', 'from sklearn.neighbors import KNeighborsRegressor; model = KNeighborsRegressor(n_neighbors=5)'),
    ]
    
    for model_name, model_code in model_configs:
        try:
            print(f"Training {model_name}...")
            
            # Import and create model
            local_vars = {}
            exec(model_code, globals(), local_vars)
            model = local_vars['model']
            
            # Train model
            model.fit(X, y)
            
            # Make predictions
            y_pred = model.predict(X)
            
            # Convert predictions to list if necessary
            if hasattr(y_pred, 'tolist'):
                y_pred = y_pred.tolist()
            
            # Calculate accuracy
            accuracy = calculate_accuracy(y, y_pred)
            
            # Save model
            try:
                import joblib
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = f"models/sklearn/{model_name}_{timestamp}.joblib"
                joblib.dump(model, model_path)
                
                trained_models[model_name] = {
                    'accuracy': accuracy,
                    'path': model_path,
                    'features': len(X[0]),
                    'samples': len(X),
                    'timestamp': timestamp
                }
                
                print(f"✅ {model_name}: {accuracy:.4f} accuracy - saved to {model_path}")
                
            except Exception as save_error:
                print(f"⚠️ {model_name}: {accuracy:.4f} accuracy - save failed: {save_error}")
                trained_models[model_name] = {
                    'accuracy': accuracy,
                    'path': None,
                    'features': len(X[0]),
                    'samples': len(X),
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
                }
                
        except Exception as e:
            failed_models[model_name] = str(e)
            print(f"❌ {model_name} failed: {e}")
    
    # Try additional models that might work
    additional_models = [
        ('ExtraTrees', 'from sklearn.ensemble import ExtraTreesRegressor; model = ExtraTreesRegressor(n_estimators=10, random_state=42)'),
        ('GradientBoosting', 'from sklearn.ensemble import GradientBoostingRegressor; model = GradientBoostingRegressor(n_estimators=10, random_state=42)'),
        ('AdaBoost', 'from sklearn.ensemble import AdaBoostRegressor; model = AdaBoostRegressor(n_estimators=10, random_state=42)'),
        ('MLPRegressor', 'from sklearn.neural_network import MLPRegressor; model = MLPRegressor(hidden_layer_sizes=(10,), max_iter=100)'),
    ]
    
    for model_name, model_code in additional_models:
        try:
            print(f"Training {model_name}...")
            
            local_vars = {}
            exec(model_code, globals(), local_vars)
            model = local_vars['model']
            
            model.fit(X, y)
            y_pred = model.predict(X)
            
            if hasattr(y_pred, 'tolist'):
                y_pred = y_pred.tolist()
            
            accuracy = calculate_accuracy(y, y_pred)
            
            try:
                import joblib
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = f"models/sklearn/{model_name}_{timestamp}.joblib"
                joblib.dump(model, model_path)
                
                trained_models[model_name] = {
                    'accuracy': accuracy,
                    'path': model_path,
                    'features': len(X[0]),
                    'samples': len(X),
                    'timestamp': timestamp
                }
                
                print(f"✅ {model_name}: {accuracy:.4f} accuracy - saved to {model_path}")
                
            except Exception as save_error:
                print(f"⚠️ {model_name}: {accuracy:.4f} accuracy - save failed: {save_error}")
                trained_models[model_name] = {
                    'accuracy': accuracy,
                    'path': None,
                    'features': len(X[0]),
                    'samples': len(X),
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
                }
                
        except Exception as e:
            failed_models[model_name] = str(e)
            print(f"❌ {model_name} failed: {e}")
    
    # Try external libraries if available
    external_models = [
        ('XGBoost', 'import xgboost as xgb; model = xgb.XGBRegressor(n_estimators=10, random_state=42)'),
        ('LightGBM', 'import lightgbm as lgb; model = lgb.LGBMRegressor(n_estimators=10, random_state=42, verbose=-1)'),
        ('CatBoost', 'import catboost as cb; model = cb.CatBoostRegressor(iterations=10, random_state=42, verbose=False)'),
    ]
    
    for model_name, model_code in external_models:
        try:
            print(f"Training {model_name}...")
            
            local_vars = {}
            exec(model_code, globals(), local_vars)
            model = local_vars['model']
            
            model.fit(X, y)
            y_pred = model.predict(X)
            
            if hasattr(y_pred, 'tolist'):
                y_pred = y_pred.tolist()
            
            accuracy = calculate_accuracy(y, y_pred)
            
            try:
                import joblib
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = f"models/sklearn/{model_name}_{timestamp}.joblib"
                joblib.dump(model, model_path)
                
                trained_models[model_name] = {
                    'accuracy': accuracy,
                    'path': model_path,
                    'features': len(X[0]),
                    'samples': len(X),
                    'timestamp': timestamp
                }
                
                print(f"✅ {model_name}: {accuracy:.4f} accuracy - saved to {model_path}")
                
            except Exception as save_error:
                print(f"⚠️ {model_name}: {accuracy:.4f} accuracy - save failed: {save_error}")
                trained_models[model_name] = {
                    'accuracy': accuracy,
                    'path': None,
                    'features': len(X[0]),
                    'samples': len(X),
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
                }
                
        except Exception as e:
            failed_models[model_name] = str(e)
            print(f"❌ {model_name} failed: {e}")
    
    # Final report
    total_trained = len(trained_models)
    total_target = 105  # Target number of models
    success_rate = (total_trained / total_target) * 100 if total_target > 0 else 0
    
    print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Minimal training completed")
    print(f"Successfully trained: {total_trained} models")
    print(f"Failed models: {len(failed_models)}")
    print(f"Success rate: {success_rate:.1f}%")
    
    # Save results
    results = {
        'total_trained': total_trained,
        'target_count': total_target,
        'success_rate': success_rate,
        'trained_models': trained_models,
        'failed_models': failed_models,
        'completion_time': datetime.now().isoformat()
    }
    
    with open('minimal_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to minimal_training_results.json")
    
    # List successful models
    if trained_models:
        print("\nSuccessfully trained models:")
        for model_name, metrics in trained_models.items():
            print(f"   - {model_name}: {metrics['accuracy']:.4f} accuracy")
    
    return results

if __name__ == "__main__":
    train_minimal_models()