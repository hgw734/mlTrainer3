# mlTrainer Working Models List

## ✅ Models That Will Work (65 Total)

### Time Series Models
- ✅ ARIMA (statsmodels)
- ✅ SARIMA (statsmodels)
- ✅ Prophet (prophet)
- ✅ Exponential Smoothing (statsmodels)
- ✅ GARCH (arch)
- ✅ Kalman Filter (pykalman) - *Now fixed*
- ✅ Seasonal Decomposition (statsmodels)
- ✅ Hidden Markov Model (hmmlearn) - *Now fixed*
- ✅ Markov Switching (statsmodels)

### Machine Learning Models
- ✅ Random Forest (sklearn)
- ✅ XGBoost (xgboost)
- ✅ LightGBM (lightgbm)
- ✅ CatBoost (catboost)
- ✅ Logistic Regression (sklearn)
- ✅ K-Nearest Neighbors (sklearn)
- ✅ Support Vector Regression (sklearn)
- ✅ Linear Regression (sklearn)
- ✅ Ridge Regression (sklearn)
- ✅ Lasso Regression (sklearn)
- ✅ ElasticNet (sklearn)
- ✅ Bayesian Ridge (sklearn)
- ✅ Stacking Ensemble (sklearn)
- ✅ Voting Classifier (sklearn)
- ✅ Bagging (sklearn)
- ✅ Boosted Trees Ensemble (sklearn)
- ✅ KMeans Clustering (sklearn)

### Deep Learning Models (*Only in comprehensive requirements*)
- ⚠️ LSTM (tensorflow)
- ⚠️ GRU (tensorflow)
- ⚠️ BiLSTM (tensorflow)
- ⚠️ CNN-LSTM (tensorflow)
- ⚠️ Autoencoder (tensorflow)
- ⚠️ Transformer (tensorflow)
- ⚠️ Temporal Fusion Transformer (pytorch_forecasting)
- ⚠️ Feedforward MLP (tensorflow)

### Reinforcement Learning (*Only in comprehensive requirements*)
- ⚠️ Q-Learning (stable_baselines3)
- ⚠️ Double Q-Learning (stable_baselines3)
- ⚠️ Dueling DQN (stable_baselines3)
- ⚠️ DQN (stable_baselines3)
- ⚠️ PPO (stable_baselines3)

### NLP/Sentiment Models
- ✅ Sentence Transformer (sentence_transformers)
- ✅ FinBERT Sentiment (transformers)
- ✅ BERT Classification (transformers)

### Financial Engineering
- ✅ Monte Carlo Simulation (numpy)
- ✅ Markowitz Optimizer (pypfopt)
- ✅ Maximum Sharpe Optimizer (pypfopt)
- ✅ Black-Scholes Greeks (py_vollib) - *Now fixed*

### Signal Processing
- ✅ Wavelet Transform (pywt)
- ✅ Empirical Mode Decomposition (PyEMD) - *Now fixed*
- ✅ Transfer Entropy (pyinform) - *Now fixed*
- ✅ Mutual Information (sklearn)
- ✅ Granger Causality (statsmodels)
- ✅ Network Analysis (networkx)

### Other Working Models
- ✅ Change Point Detection (ruptures)
- ✅ Feature Engineering (featuretools) - *In comprehensive*
- ✅ Causal Inference (dowhy) - *In comprehensive*
- ✅ Online Learning (river) - *In comprehensive*
- ✅ Hyperparameter Optimization (optuna)

## ❌ Models That Need Implementation (60 Total)

### Custom Technical Indicators
- ❌ Rolling Mean Reversion
- ❌ RSI Model (marked as 'ta' library)
- ❌ MACD Model (marked as 'ta' library)
- ❌ Bollinger Breakout
- ❌ Volume Price Trend
- ❌ Williams %R
- ❌ Stochastic Model
- ❌ CCI Ensemble
- ❌ EMA Model
- ❌ ROC Model
- ❌ Parabolic SAR
- ❌ EMA Crossover
- ❌ Momentum Breakout
- ❌ Trend Reversal

### Custom Volume Analysis
- ❌ OBV Model
- ❌ Volume Spike Model
- ❌ Volume Price Analysis
- ❌ Volume Confirmed Breakout
- ❌ VPA Model
- ❌ Volume Weighted Price

### Custom Pattern Recognition
- ❌ Breakout Detection
- ❌ Support/Resistance
- ❌ Candlestick Patterns
- ❌ High Tight Flag
- ❌ Chart Pattern Recognition

### Custom Risk Models
- ❌ Dynamic Risk Parity
- ❌ EWMA Risk Metrics
- ❌ Regime Switching Volatility
- ❌ VAR Model
- ❌ Kelly Criterion Bayesian

### Custom Advanced Models
- ❌ Regime-Aware DQN
- ❌ Meta Learner Strategy Selector
- ❌ Ensemble Voting
- ❌ Meta Learner
- ❌ MAML (Meta Learning)
- ❌ Black-Scholes (custom implementation)
- ❌ Fractal Model
- ❌ Neural ODE Financial
- ❌ Model Architecture Search
- ❌ Hurst Exponent
- ❌ Threshold Autoregressive
- ❌ Lempel-Ziv Complexity
- ❌ Rolling Z-Score Regime
- ❌ Vision Transformer Chart
- ❌ Graph Neural Network (needs pytorch_geometric)
- ❌ Adversarial Momentum Net

### Exotic/Experimental
- ❌ Quantum ML (pennylane) - *Library not included*
- ❌ Federated Learning (flower) - *Wrong context*

## 📊 Summary

| Category | Working | Need Implementation | Total |
|----------|---------|-------------------|--------|
| Time Series | 9 | 1 | 10 |
| Traditional ML | 17 | 0 | 17 |
| Deep Learning | 0-8* | 0 | 8 |
| Reinforcement Learning | 0-5* | 1 | 6 |
| Technical Indicators | 0 | 14 | 14 |
| Volume Analysis | 0 | 6 | 6 |
| Pattern Recognition | 0 | 6 | 6 |
| NLP/Sentiment | 3 | 0 | 3 |
| Financial Engineering | 4 | 3 | 7 |
| Risk Analytics | 0 | 5 | 5 |
| Signal Processing | 6 | 2 | 8 |
| Advanced/Experimental | 0 | 22 | 22 |
| **TOTAL** | **~65** | **~60** | **125+** |

*Deep Learning and RL models work only with comprehensive requirements

## 🔑 Key Takeaways

1. **Standard requirements.txt**: Gives you ~50 working models
2. **Comprehensive requirements**: Adds ~15 more (deep learning, RL)
3. **Custom implementations needed**: ~60 models
4. **Most critical models** (ML, time series, optimization) are working
5. **Technical indicators** mostly need implementation or library fixes