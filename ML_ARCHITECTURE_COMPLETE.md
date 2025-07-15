# mlTrainer Complete ML Architecture

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          mlTrainer Trading Platform                          │
│                         92+ Working Models (27 New)                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    │                                       │
            ┌───────▼───────┐                       ┌──────▼──────┐
            │ Data Pipeline │                       │  Compliance │
            │               │                       │   System    │
            │ • Polygon API │                       │             │
            │ • FRED API    │                       │ • Immutable │
            │ • yfinance    │                       │   Rules     │
            │ • 20+ Features│                       │ • Runtime   │
            └───────┬───────┘                       │   Hooks     │
                    │                               └──────┬──────┘
                    │                                      │
                    └──────────────┬───────────────────────┘
                                   │
        ┌──────────────────────────┴────────────────────────────┐
        │                    Model Categories                    │
        └────────────────────────────────────────────────────────┘
                                   │
    ┌──────────────────────────────┴────────────────────────────────┐
    │                                                                │
    ▼                                                                ▼
┌───────────────────────┐  ┌────────────────────┐  ┌────────────────────┐
│  RULE-BASED MODELS    │  │ MACHINE LEARNING   │  │  ADVANCED MODELS   │
│      (3 New)          │  │    (3 New)         │  │     (3 New)        │
├───────────────────────┤  ├────────────────────┤  ├────────────────────┤
│ • Momentum Breakout   │  │ • Random Forest    │  │ • Pairs Trading    │
│ • Mean Reversion      │  │   (40+ features)   │  │   (Cointegration)  │
│ • Volatility Regime   │  │ • XGBoost          │  │ • Market Regime    │
│                       │  │   (60+ features)   │  │   (HMM)            │
│                       │  │ • LSTM Enhanced    │  │ • Portfolio Opt    │
│                       │  │   (3-layer)        │  │   (6 methods)      │
└───────────────────────┘  └────────────────────┘  └────────────────────┘
                                   │
    ┌──────────────────────────────┴────────────────────────────────┐
    │                                                                │
    ▼                                                                ▼
┌───────────────────────┐  ┌────────────────────┐  ┌────────────────────┐
│ TECHNICAL INDICATORS  │  │  VOLUME ANALYSIS   │  │ PATTERN RECOGNITION│
│     (8 New)           │  │     (5 New)        │  │     (5 New)        │
├───────────────────────┤  ├────────────────────┤  ├────────────────────┤
│ • RSI + Divergence    │  │ • OBV + Divergence │  │ • Candlesticks     │
│ • MACD + Histogram    │  │ • Volume Spike     │  │   (Hammer, Doji)   │
│ • Bollinger Breakout  │  │ • VPA (Wyckoff)    │  │ • Support/Resist   │
│ • Stochastic          │  │ • Volume Breakout  │  │ • Chart Patterns   │
│ • Williams %R         │  │ • VWAP + Bands     │  │   (Triangle,Flag)  │
│ • CCI Ensemble        │  │                    │  │ • Breakout Detect  │
│ • Parabolic SAR       │  │                    │  │ • High Tight Flag  │
│ • EMA Crossover       │  │                    │  │                    │
└───────────────────────┘  └────────────────────┘  └────────────────────┘
                                   │
    ┌──────────────────────────────┴────────────────────────────────┐
    │                     PRE-EXISTING MODELS (65+)                  │
    └────────────────────────────────────────────────────────────────┘
                                   │
    ┌──────────────────────────────┼────────────────────────────────┐
    │                              │                                 │
    ▼                              ▼                                 ▼
┌────────────────┐  ┌──────────────────────┐  ┌─────────────────────┐
│  TIME SERIES   │  │   TRADITIONAL ML     │  │   DEEP LEARNING*    │
│    (9 Models)  │  │    (17 Models)       │  │    (8 Models)       │
├────────────────┤  ├──────────────────────┤  ├─────────────────────┤
│ • ARIMA/SARIMA │  │ • Random Forest      │  │ • LSTM/GRU/BiLSTM   │
│ • Prophet      │  │ • XGBoost/LightGBM   │  │ • CNN-LSTM          │
│ • Exp Smooth   │  │ • CatBoost           │  │ • Autoencoder       │
│ • GARCH        │  │ • SVM/KNN            │  │ • Transformer       │
│ • Kalman       │  │ • Linear/Ridge/Lasso │  │ • Temporal Fusion   │
│ • HMM          │  │ • Ensemble Methods   │  │ • MLP               │
└────────────────┘  └──────────────────────┘  └─────────────────────┘
                                   │
    ┌──────────────────────────────┼────────────────────────────────┐
    │                              │                                 │
    ▼                              ▼                                 ▼
┌────────────────┐  ┌──────────────────────┐  ┌─────────────────────┐
│ NLP/SENTIMENT  │  │ FINANCIAL ENGINEER   │  │ SIGNAL PROCESSING   │
│  (3 Models)    │  │    (4 Models)        │  │    (6 Models)       │
├────────────────┤  ├──────────────────────┤  ├─────────────────────┤
│ • Sentence     │  │ • Monte Carlo        │  │ • Wavelet Transform │
│   Transformer  │  │ • Markowitz          │  │ • EMD               │
│ • FinBERT      │  │ • Maximum Sharpe     │  │ • Transfer Entropy  │
│ • BERT Class   │  │ • Black-Scholes      │  │ • Mutual Info       │
│                │  │                      │  │ • Granger Causality │
│                │  │                      │  │ • Network Analysis  │
└────────────────┘  └──────────────────────┘  └─────────────────────┘


## Model Ensemble Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ENSEMBLE MODELS                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐     ┌─────────────────┐              │
│  │ Tech Indicators │     │ Volume Analysis │              │
│  │    Ensemble     │     │    Ensemble     │              │
│  │                 │     │                 │              │
│  │ Weighted voting │     │ Weighted voting │              │
│  │ of 8 indicators │     │ of 5 models    │              │
│  └────────┬────────┘     └────────┬────────┘              │
│           │                       │                        │
│           └───────────┬───────────┘                        │
│                       ▼                                    │
│              ┌─────────────────┐                          │
│              │ Pattern Recog.  │                          │
│              │    Ensemble     │                          │
│              │                 │                          │
│              │ Weighted voting │                          │
│              │ of 5 patterns   │                          │
│              └─────────────────┘                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Infrastructure Components

```
┌─────────────────────────────────────────────────────────────┐
│                    INFRASTRUCTURE                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐      │
│  │ Backtesting │  │   Model     │  │  Prediction  │      │
│  │   Engine    │  │  Trainer    │  │   Service    │      │
│  ├─────────────┤  ├─────────────┤  ├──────────────┤      │
│  │ • Costs     │  │ • Training  │  │ • Real-time  │      │
│  │ • Slippage  │  │ • Persist   │  │ • Batch      │      │
│  │ • Metrics   │  │ • Version   │  │ • API        │      │
│  └─────────────┘  └─────────────┘  └──────────────┘      │
│                                                             │
│  ┌─────────────────────────────────────────────────┐      │
│  │              DEPLOYMENT                          │      │
│  ├─────────────────────────────────────────────────┤      │
│  │ • Docker containers                              │      │
│  │ • Docker Compose orchestration                   │      │
│  │ • GitHub Actions CI/CD                          │      │
│  │ • Requirements management                        │      │
│  └─────────────────────────────────────────────────┘      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Key Statistics

- **Total Working Models**: 92+ (65 pre-existing + 27 new)
- **Model Categories**: 9 major categories
- **Data Sources**: 3 (Polygon, FRED, yfinance) 
- **Feature Engineering**: 20+ technical indicators
- **Backtesting Metrics**: 15+ (Sharpe, Sortino, Calmar, etc.)
- **Compliance**: Immutable rules with runtime enforcement
- **Code Quality**: Production-grade with comprehensive error handling

*Deep Learning models require TensorFlow (see requirements_comprehensive.txt)