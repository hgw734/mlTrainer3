import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def build_lstm_model(input_shape: tuple) -> tf.keras.Model:
    """Builds a simple LSTM model for time series prediction."""
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Output layer for regression

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def prepare_sequences(X: np.ndarray, y: np.ndarray, window_size: int = 10):
    """Convert timeseries into LSTM input/output sequences"""
    X_seq, y_seq = [], []
    for i in range(window_size, len(X)):
        X_seq.append(X[i-window_size:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

def train_lstm(X_train_raw, y_train_raw, window_size: int = 10):
    """
    Train LSTM model on processed timeseries input.
    X_train_raw, y_train_raw: 2D arrays (features, target)
    Returns trained model.
    """
    logger.info("🧠 Preprocessing input data for LSTM...")

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X_train_raw)
    y_scaled = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).flatten()

    X_seq, y_seq = prepare_sequences(X_scaled, y_scaled, window_size)

    input_shape = (X_seq.shape[1], X_seq.shape[2]) if X_seq.ndim == 3 else (X_seq.shape[1], 1)
    if X_seq.ndim == 2:
        X_seq = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], 1))

    model = build_lstm_model(input_shape)

    logger.info("🚀 Training LSTM model...")
    es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

    model.fit(X_seq, y_seq, epochs=50, batch_size=32, verbose=1, callbacks=[es])

    logger.info("✅ LSTM training complete.")
    return model
