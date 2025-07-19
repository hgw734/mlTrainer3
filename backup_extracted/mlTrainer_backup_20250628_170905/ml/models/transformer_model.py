import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout
from tensorflow.keras.layers import MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def build_transformer_model(input_shape: tuple) -> tf.keras.Model:
    """
    Builds a simple Transformer model for time series regression.
    """
    inputs = Input(shape=input_shape)

    # Multi-head self-attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
    x = Dropout(0.1)(x)
    x = LayerNormalization(epsilon=1e-6)(x)

    # Global average pooling and dense output
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def prepare_sequences(X: np.ndarray, y: np.ndarray, window_size: int = 10):
    """Convert timeseries into Transformer input/output sequences."""
    X_seq, y_seq = [], []
    for i in range(window_size, len(X)):
        X_seq.append(X[i-window_size:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

def train_transformer(X_raw, y_raw, window_size: int = 10):
    """
    Train a Transformer model on time series input.
    X_raw, y_raw: 2D arrays (features, target)
    """
    logger.info("🧠 Preprocessing input data for Transformer...")

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X_raw)
    y_scaled = scaler_y.fit_transform(y_raw.reshape(-1, 1)).flatten()

    X_seq, y_seq = prepare_sequences(X_scaled, y_scaled, window_size)

    if X_seq.ndim == 2:
        X_seq = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], 1))

    input_shape = (X_seq.shape[1], X_seq.shape[2])

    model = build_transformer_model(input_shape)

    X_train, X_val, y_train, y_val = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42
    )

    logger.info("🚀 Training Transformer model...")
    model.fit(X_train, y_train, epochs=40, batch_size=32, validation_data=(X_val, y_val), verbose=1)

    logger.info("✅ Transformer training complete.")
    return model
