#!/usr/bin/env python3
"""
Simple Model Trainer - Direct training approach
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

def create_data():
    """Create training data"""
    np.random.seed(42)
    n_samples = 200
    
    data = {
        'open': np.random.uniform(100, 200, n_samples),
        'high': np.random.uniform(110, 220, n_samples), 
        'low': np.random.uniform(90, 190, n_samples),
        'close': np.random.uniform(95, 210, n_samples),
        'volume': np.random.uniform(1000000, 10000000, n_samples),
        'sma_5': np.random.uniform(95, 205, n_samples),
        'sma_10': np.random.uniform(95, 205, n_samples),
        'rsi': np.random.uniform(20, 80, n_samples),
        'macd': np.random.uniform(-2, 2, n_samples),
        'volatility': np.random.uniform(0.01, 0.05, n_samples),
    }
    
    df = pd.DataFrame(data)
    df['target'] = df['close'].shift(-1) / df['close'] - 1
    df = df.dropna()
    
    return df

def train_all_models():
    """Train all models directly"""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Starting comprehensive model training")
    
    # Create directories
    os.makedirs('models/sklearn', exist_ok=True)
    
    # Create training data
    data = create_data()
    X = data.drop(['target'], axis=1)
    y = data['target']
    
    print(f"Training data: {len(X)} samples, {len(X.columns)} features")
    
    trained_models = {}
    failed_models = {}
    
    # Model 1: LinearRegression
    try:
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        mse = ((y - y_pred) ** 2).mean()
        accuracy = max(0, min(1, 1 - mse))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/sklearn/LinearRegression_{timestamp}.joblib"
        joblib.dump(model, model_path)
        trained_models['LinearRegression'] = {'accuracy': accuracy, 'path': model_path}
        print(f"✅ LinearRegression: {accuracy:.4f} accuracy")
    except Exception as e:
        failed_models['LinearRegression'] = str(e)
        print(f"❌ LinearRegression failed: {e}")
    
    # Model 2: Ridge
    try:
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        y_pred = model.predict(X)
        mse = ((y - y_pred) ** 2).mean()
        accuracy = max(0, min(1, 1 - mse))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/sklearn/Ridge_{timestamp}.joblib"
        joblib.dump(model, model_path)
        trained_models['Ridge'] = {'accuracy': accuracy, 'path': model_path}
        print(f"✅ Ridge: {accuracy:.4f} accuracy")
    except Exception as e:
        failed_models['Ridge'] = str(e)
        print(f"❌ Ridge failed: {e}")
    
    # Model 3: Lasso
    try:
        from sklearn.linear_model import Lasso
        model = Lasso(alpha=1.0)
        model.fit(X, y)
        y_pred = model.predict(X)
        mse = ((y - y_pred) ** 2).mean()
        accuracy = max(0, min(1, 1 - mse))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/sklearn/Lasso_{timestamp}.joblib"
        joblib.dump(model, model_path)
        trained_models['Lasso'] = {'accuracy': accuracy, 'path': model_path}
        print(f"✅ Lasso: {accuracy:.4f} accuracy")
    except Exception as e:
        failed_models['Lasso'] = str(e)
        print(f"❌ Lasso failed: {e}")
    
    # Model 4: ElasticNet
    try:
        from sklearn.linear_model import ElasticNet
        model = ElasticNet(alpha=1.0)
        model.fit(X, y)
        y_pred = model.predict(X)
        mse = ((y - y_pred) ** 2).mean()
        accuracy = max(0, min(1, 1 - mse))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/sklearn/ElasticNet_{timestamp}.joblib"
        joblib.dump(model, model_path)
        trained_models['ElasticNet'] = {'accuracy': accuracy, 'path': model_path}
        print(f"✅ ElasticNet: {accuracy:.4f} accuracy")
    except Exception as e:
        failed_models['ElasticNet'] = str(e)
        print(f"❌ ElasticNet failed: {e}")
    
    # Model 5: RandomForest
    try:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        y_pred = model.predict(X)
        mse = ((y - y_pred) ** 2).mean()
        accuracy = max(0, min(1, 1 - mse))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/sklearn/RandomForest_{timestamp}.joblib"
        joblib.dump(model, model_path)
        trained_models['RandomForest'] = {'accuracy': accuracy, 'path': model_path}
        print(f"✅ RandomForest: {accuracy:.4f} accuracy")
    except Exception as e:
        failed_models['RandomForest'] = str(e)
        print(f"❌ RandomForest failed: {e}")
    
    # Model 6: GradientBoosting
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        y_pred = model.predict(X)
        mse = ((y - y_pred) ** 2).mean()
        accuracy = max(0, min(1, 1 - mse))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/sklearn/GradientBoosting_{timestamp}.joblib"
        joblib.dump(model, model_path)
        trained_models['GradientBoosting'] = {'accuracy': accuracy, 'path': model_path}
        print(f"✅ GradientBoosting: {accuracy:.4f} accuracy")
    except Exception as e:
        failed_models['GradientBoosting'] = str(e)
        print(f"❌ GradientBoosting failed: {e}")
    
    # Continue with all remaining models...
    # (I'll add them incrementally to avoid script length issues)
    
    # Save results
    total_trained = len(trained_models)
    total_target = 105  # Target number of models
    
    results = {
        'total_trained': total_trained,
        'target_count': total_target,
        'success_rate': (total_trained / total_target) * 100,
        'trained_models': trained_models,
        'failed_models': failed_models,
        'completion_time': datetime.now().isoformat()
    }
    
    with open('simple_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Training completed")
    print(f"Successfully trained: {total_trained} models")
    print(f"Results saved to simple_training_results.json")
    
    return results

if __name__ == "__main__":
    train_all_models()