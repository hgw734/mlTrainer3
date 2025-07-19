"""
Exit Model Module
ML-based exit strategy prediction with XGBoost and LSTM models
"""

import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import logging

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)

def train_exit_predictor(market_data, universe):
    """
    Train ML exit predictor with enhanced forward walk validation
    
    Args:
        market_data: Dictionary with authentic market data
        universe: List of stock symbols
    
    Returns:
        tuple: (trained_model, label_encoder, is_trained)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/exit_model_{timestamp}.pkl"
    metrics_path = f"models/exit_model_metrics_{timestamp}.txt"
    csv_log_path = "models/model_performance_log.csv"
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    logger.info("Training ML exit predictor with enhanced forward walk validation...")
    
    data = []
    labels = []
    
    # Extract features and labels from authentic market data
    for symbol in universe:
        if symbol == 'VIX' or symbol not in market_data:
            continue
            
        df = market_data[symbol]
        if len(df) < 200:
            continue
        
        # Add trend_strength if missing
        if 'trend_strength' not in df.columns:
            df['trend_strength'] = (df['close'] / df['sma_20'] - 1).rolling(window=5).mean()
        
        # Extract training samples from authentic data
        for i in range(30, len(df) - 15):
            row = df.iloc[i]
            
            # Skip rows with missing data
            if pd.isna(row['rsi_14']) or pd.isna(row['momentum_5']) or pd.isna(row['trend_strength']):
                continue
            
            # Calculate actual exit performance using authentic prices
            entry_price = row['close']
            exit_price = df.iloc[i+10]['close']
            ret = (exit_price - entry_price) / entry_price
            
            # Label based on actual performance
            if ret > 0.02:
                label = 'momentum_reversal'
            elif ret < -0.01:
                label = 'volatility_exit'
            else:
                label = 'time_exit'
            
            # Feature vector from authentic market indicators
            feature = [
                row['rsi_14'], 
                row['momentum_5'], 
                row['volume_ratio'], 
                row['atr_pct_14'], 
                row['trend_strength']
            ]
            
            data.append(feature)
            labels.append(label)
    
    if len(data) < 100:
        logger.warning("Insufficient authentic data for ML training")
        return None, None, False
    
    X = np.array(data)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    
    # Enhanced forward walk validation
    window_size = 1000
    step_size = 250
    scores = []
    
    logger.info("Performing forward walk validation...")
    for start in range(0, len(X) - window_size, step_size):
        end = start + window_size
        X_train, y_train = X[:start+step_size], y[:start+step_size]
        X_test, y_test = X[start+step_size:end], y[start+step_size:end]
        
        if len(X_train) < 500 or len(X_test) < 100:
            continue
        
        # Train model on this window
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = f1_score(y_test, y_pred, average='weighted')
        scores.append(score)
        
        logger.info(f"Walk-forward window {len(scores)}: F1 = {score:.3f}")
    
    # Train final model on all authentic data
    if XGBOOST_AVAILABLE:
        exit_predictor = XGBClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            eval_metric='logloss'
        )
        model_type = "XGBoost"
    else:
        exit_predictor = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        model_type = "RandomForest"
    
    exit_predictor.fit(X, y)
    
    # Save model and performance metrics
    joblib.dump(exit_predictor, model_path)
    joblib.dump(label_encoder, f"models/label_encoder_{timestamp}.pkl")
    logger.info(f"Model saved to {model_path}")
    
    avg_score = np.mean(scores) if scores else 0.0
    report = f"Forward walk F1 scores: {scores}\nAverage F1: {avg_score:.4f}\nModel type: {model_type}"
    
    # Save detailed metrics
    with open(metrics_path, "w") as f:
        f.write("Exit strategy predictor performance (walk-forward):\n")
        f.write(report)
    
    # Log to CSV for model comparison
    metrics_row = {
        'timestamp': timestamp,
        'model_path': model_path,
        'metrics_path': metrics_path,
        'f1_weighted': avg_score,
        'model_type': model_type,
        'training_samples': len(data)
    }
    
    if not os.path.exists(csv_log_path):
        pd.DataFrame([metrics_row]).to_csv(csv_log_path, index=False)
    else:
        pd.DataFrame([metrics_row]).to_csv(csv_log_path, mode='a', header=False, index=False)
    
    logger.info(f"Metrics saved to {metrics_path} and logged in CSV")
    logger.info(f"{model_type} exit predictor trained - Walk-forward F1: {avg_score:.3f}")
    
    return exit_predictor, label_encoder, True

def perform_forward_walk_validation(training_df):
    """
    Perform time-based walk-forward validation
    
    Args:
        training_df: DataFrame with time-sorted training data
    
    Returns:
        list: List of validation results for each time window
    """
    results = []
    feature_cols = ['rsi_14', 'momentum_5', 'momentum_10', 'volume_ratio', 'macd', 'volatility_20', 'atr_pct_14']
    
    # Define time windows for walk-forward (6-month training, 2-month testing)
    start_date = training_df['timestamp'].min()
    end_date = training_df['timestamp'].max()
    
    current_date = start_date + pd.DateOffset(months=8)  # First test period
    
    while current_date < end_date:
        # Training window: 6 months before current_date
        train_start = current_date - pd.DateOffset(months=6)
        train_end = current_date
        
        # Test window: 2 months after current_date
        test_start = current_date
        test_end = current_date + pd.DateOffset(months=2)
        
        # Split data
        train_mask = (training_df['timestamp'] >= train_start) & (training_df['timestamp'] < train_end)
        test_mask = (training_df['timestamp'] >= test_start) & (training_df['timestamp'] < test_end)
        
        train_data = training_df[train_mask]
        test_data = training_df[test_mask]
        
        if len(train_data) < 30 or len(test_data) < 10:
            current_date += pd.DateOffset(months=2)
            continue
        
        # Prepare training and test sets
        X_train = train_data[feature_cols].fillna(0)
        X_test = test_data[feature_cols].fillna(0)
        
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(train_data['exit_strategy'])
        
        # Handle unseen labels in test set
        try:
            y_test = label_encoder.transform(test_data['exit_strategy'])
        except ValueError:
            # Skip if test set has unseen labels
            current_date += pd.DateOffset(months=2)
            continue
        
        # Train model for this window
        if XGBOOST_AVAILABLE:
            model = XGBClassifier(n_estimators=100, max_depth=6, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = model.predict(X_test)
        
        from sklearn.metrics import f1_score, accuracy_score
        f1 = f1_score(y_test, y_pred, average='weighted')
        accuracy = accuracy_score(y_test, y_pred)
        
        result = {
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'f1_score': f1,
            'accuracy': accuracy,
            'train_samples': len(train_data),
            'test_samples': len(test_data)
        }
        
        results.append(result)
        logger.info(f"Walk-forward period {test_start.strftime('%Y-%m')} - F1: {f1:.3f}, Accuracy: {accuracy:.3f}")
        
        current_date += pd.DateOffset(months=2)
    
    return results

def save_walkforward_log(results):
    """Save walk-forward validation results to CSV"""
    if not results:
        return
    
    log_dir = 'data/logs'
    os.makedirs(log_dir, exist_ok=True)
    
    log_df = pd.DataFrame(results)
    log_file = f"{log_dir}/walkforward_log_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    log_df.to_csv(log_file, index=False)
    
    logger.info(f"Walk-forward validation log saved to {log_file}")
    
    # Summary statistics
    avg_f1 = log_df['f1_score'].mean()
    std_f1 = log_df['f1_score'].std()
    logger.info(f"Walk-forward summary - Mean F1: {avg_f1:.3f} ± {std_f1:.3f}")

def determine_optimal_exit_strategy(data, entry_idx):
    """
    Determine optimal exit strategy using authentic price movements
    
    Args:
        data: Authentic price data DataFrame
        entry_idx: Entry point index
    
    Returns:
        str: Optimal exit strategy name
    """
    entry_price = data.iloc[entry_idx]['close']
    
    # Test different exit strategies on authentic data
    strategies = {
        'momentum_exit': test_momentum_exit(data, entry_idx, entry_price),
        'volatility_exit': test_volatility_exit(data, entry_idx, entry_price),
        'time_exit': test_time_exit(data, entry_idx, entry_price, 12)
    }
    
    # Return strategy with best risk-adjusted return on authentic data
    valid_strategies = {k: v for k, v in strategies.items() if v is not None}
    if not valid_strategies:
        return None
    
    best_strategy = max(valid_strategies.items(), key=lambda x: x[1])
    return best_strategy[0]

def test_momentum_exit(data, entry_idx, entry_price):
    """Test momentum exit strategy on authentic data"""
    for i in range(3, min(15, len(data) - entry_idx)):
        if data.iloc[entry_idx + i]['momentum_5'] < 0.005:
            return (data.iloc[entry_idx + i]['close'] - entry_price) / entry_price
    return None

def test_volatility_exit(data, entry_idx, entry_price):
    """Test volatility exit strategy on authentic data"""
    entry_vol = data.iloc[entry_idx]['volatility_20']
    for i in range(5, min(18, len(data) - entry_idx)):
        if data.iloc[entry_idx + i]['volatility_20'] > entry_vol * 1.5:
            return (data.iloc[entry_idx + i]['close'] - entry_price) / entry_price
    return None

def test_time_exit(data, entry_idx, entry_price, days):
    """Test time-based exit strategy on authentic data"""
    if entry_idx + days >= len(data):
        return None
    return (data.iloc[entry_idx + days]['close'] - entry_price) / entry_price

def hybrid_exit_strategy(df, entry_idx, params, symbol, market_regime, exit_predictor, label_encoder):
    """
    Hybrid exit strategy combining ML prediction with regime-based logic
    
    Args:
        df: Authentic market data
        entry_idx: Entry index
        params: Strategy parameters
        symbol: Stock symbol
        market_regime: Current market regime
        exit_predictor: Trained ML model
        label_encoder: Label encoder for exit strategies
    
    Returns:
        float: Trade return based on authentic data
    """
    from .regime_logic import get_exit_configuration
    
    entry_price = df.iloc[entry_idx]['close']
    adaptation_strength = params['adaptation_strength']
    
    # Get regime-based exit configuration
    exit_config = get_exit_configuration(market_regime, adaptation_strength)
    
    # ML prediction for exit strategy
    current = df.iloc[entry_idx]
    ml_features = {
        'rsi_14': current['rsi_14'],
        'momentum_5': current['momentum_5'],
        'momentum_10': current['momentum_10'],
        'volume_ratio': current['volume_ratio'],
        'macd': current['macd'],
        'volatility_20': current['volatility_20'],
        'atr_pct_14': current['atr_pct_14']
    }
    
    predicted_exit = predict_exit_strategy(ml_features, exit_predictor, label_encoder)
    
    # Execute predicted strategy with regime adaptation
    if predicted_exit == 'momentum_exit':
        return calculate_momentum_exit(df, entry_idx, entry_price, exit_config)
    elif predicted_exit == 'volatility_exit':
        return calculate_volatility_exit(df, entry_idx, entry_price, exit_config)
    else:  # time_exit
        return calculate_time_exit(df, entry_idx, entry_price, params, exit_config)

def predict_exit_strategy(features, exit_predictor, label_encoder):
    """
    Predict exit strategy using trained ML model
    
    Args:
        features: Current market features
        exit_predictor: Trained ML model
        label_encoder: Label encoder
    
    Returns:
        str: Predicted exit strategy
    """
    if exit_predictor is None or label_encoder is None:
        return 'time_exit'
    
    feature_cols = ['rsi_14', 'momentum_5', 'momentum_10', 'volume_ratio', 'macd', 'volatility_20', 'atr_pct_14']
    feature_vector = [features.get(col, 0) for col in feature_cols]
    
    try:
        prediction = exit_predictor.predict([feature_vector])[0]
        return label_encoder.inverse_transform([prediction])[0]
    except Exception as e:
        logger.warning(f"Error in exit prediction: {e}")
        return 'time_exit'

def calculate_momentum_exit(df, entry_idx, entry_price, config):
    """Calculate momentum reversal exit using authentic data"""
    momentum_threshold = config['momentum_threshold']
    max_hold_days = config['max_hold_days']
    
    for i in range(3, min(max_hold_days, len(df) - entry_idx)):
        current_momentum = df.iloc[entry_idx + i]['momentum_5']
        if current_momentum < momentum_threshold:
            exit_price = df.iloc[entry_idx + i]['close']
            return (exit_price - entry_price) / entry_price
    
    return None

def calculate_volatility_exit(df, entry_idx, entry_price, config):
    """Calculate volatility-based exit using authentic data"""
    if 'volatility_20' not in df.columns:
        return None
    
    entry_vol = df.iloc[entry_idx]['volatility_20']
    sensitivity = config['volatility_sensitivity']
    
    for i in range(2, min(10, len(df) - entry_idx)):
        current_vol = df.iloc[entry_idx + i]['volatility_20']
        if current_vol > entry_vol * sensitivity:
            exit_price = df.iloc[entry_idx + i]['close']
            return (exit_price - entry_price) / entry_price
    
    return None

def calculate_time_exit(df, entry_idx, entry_price, params, config):
    """Calculate regime-adaptive time exit using authentic data"""
    base_hold = int(params['hold_period'])
    time_multiplier = config['time_multiplier']
    
    hold_days = max(5, min(25, int(base_hold * time_multiplier)))
    
    if entry_idx + hold_days >= len(df):
        return None
    
    exit_price = df.iloc[entry_idx + hold_days]['close']
    return (exit_price - entry_price) / entry_price

def build_lstm_model(input_shape, n_classes=3):
    """
    Build LSTM model for exit strategy prediction
    
    Args:
        input_shape: Input shape tuple
        n_classes: Number of exit strategy classes
    
    Returns:
        Sequential: Compiled LSTM model
    """
    if not TENSORFLOW_AVAILABLE:
        logger.warning("TensorFlow not available, using RandomForest instead")
        return None
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info("LSTM model built successfully")
    return model

def compare_saved_models(model_dir="models/"):
    """
    Compare performance of saved exit prediction models
    
    Args:
        model_dir: Directory containing saved models
    
    Returns:
        dict: Model comparison results
    """
    # Implementation for model comparison
    # This would load and compare different saved models
    logger.info("Model comparison functionality - to be implemented")
    return {"status": "comparison_pending"}