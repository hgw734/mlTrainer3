# Advanced Models Data Requirements Guide

## Overview

This document details the specific data requirements for the advanced models that have been sandboxed. These models require specialized data that goes beyond standard OHLCV price data available from Polygon.io.

## Advanced Model Categories

### 1. Neural ODE Models

#### Model: `neural_ode_financial`
- **Data Requirement**: `continuous_time_series`
- **What This Means**:
  - Requires **irregularly sampled** time series data
  - Needs timestamps with microsecond or nanosecond precision
  - Data points don't need to be evenly spaced
  - Examples: High-frequency tick data, asynchronous trade data

**Why Polygon.io/FRED Can't Provide This**:
- Polygon.io provides regular interval data (1-minute bars minimum)
- FRED provides daily/weekly/monthly economic data
- Neither provides true continuous-time tick-by-tick data

**Data Sources Needed**:
- **Tick Data Providers**: Refinitiv, Bloomberg Terminal, ICE Data Services
- **Exchange Direct Feeds**: NYSE TAQ, NASDAQ TotalView
- **Specialized Vendors**: TickData.com, Kibot, FirstRate Data

### 2. Transformer Models

#### Model: `transformer`
- **Data Requirements**: `sequential_data`, `tokenized_data`
- **What This Means**:
  - Requires text data that can be tokenized
  - Needs sequence data with semantic meaning
  - Examples: News articles, earnings transcripts, SEC filings

#### Model: `temporal_fusion_transformer`
- **Data Requirements**: `time_series`, `categorical_features`, `static_features`
- **What This Means**:
  - Requires multi-modal data (numerical + categorical)
  - Needs static metadata (company sector, market cap category)
  - Requires known future inputs (holiday calendars, scheduled events)

**Why Polygon.io/FRED Can't Provide This**:
- No text/NLP data available
- Limited categorical metadata
- No forward-looking event calendars

**Data Sources Needed**:
- **Financial Text**: Refinitiv News, Bloomberg News API, Benzinga Pro
- **Earnings/Filings**: SEC EDGAR API, AlphaSense, S&P Capital IQ
- **Event Calendars**: Trading Economics, Econoday, DailyFX

### 3. Fractal Analysis Models

#### Model: `fractal_model`
- **Data Requirement**: `price_data` (but with specific characteristics)
- **What This Means**:
  - Requires **high-frequency** price data for fractal analysis
  - Needs at least tick-level or sub-second data
  - Must capture microstructure patterns

#### Model: `hurst_exponent_fractal`
- **Data Requirement**: `time_series` (high-frequency)
- **What This Means**:
  - Requires long time series with fine granularity
  - Needs consistent sampling rate
  - Must have sufficient data points (10,000+) for reliable Hurst calculation

**Why Standard Data Isn't Sufficient**:
- Fractal patterns emerge at high frequencies
- 1-minute bars from Polygon.io too coarse for fractal analysis
- Need to capture self-similarity at multiple timescales

**Data Sources Needed**:
- **High-Frequency Data**: Refinitiv Tick History, Bloomberg B-PIPE
- **Microstructure Data**: LOBSTER, Nasdaq ITCH
- **Crypto (24/7 fractals)**: Kaiko, CryptoCompare, Tardis

### 4. Signal Processing Models

#### Model: `wavelet_transform_model`
- **Data Requirement**: `time_series`
- **What This Means**:
  - Requires evenly sampled time series
  - Needs sufficient length for multi-scale analysis
  - Benefits from high-frequency data

#### Model: `empirical_mode_decomposition`
- **Data Requirement**: `time_series`
- **What This Means**:
  - Requires non-stationary time series
  - Needs data with multiple intrinsic modes
  - Benefits from long, continuous series

**Special Considerations**:
- While these models list "time_series" which Polygon.io can provide, they perform optimally with:
  - Higher frequency data than 1-minute bars
  - Longer continuous histories
  - Multiple correlated series for cross-decomposition

**Enhanced Data Sources**:
- **Intraday Bars**: IEX Cloud (millisecond data), Twelve Data
- **Historical Depth**: Tiingo, Quandl (for longer histories)
- **Multi-Asset**: ICE Data Services, Morningstar Direct

### 5. Nonlinear Time Series Models

#### Model: `threshold_autoregressive`
- **Data Requirement**: `time_series`
- **What This Means**:
  - Requires time series with regime changes
  - Needs sufficient data to identify thresholds
  - Benefits from economic regime indicators

**Enhanced Requirements**:
- While basic time series from Polygon.io could work, optimal performance requires:
  - Regime indicators (VIX levels, yield curve data)
  - Macro regime data (expansion/recession indicators)
  - Cross-asset correlations

**Additional Data Sources**:
- **Regime Indicators**: CBOE (VIX data), Federal Reserve (yield curves)
- **Macro Regimes**: NBER recession dates, Conference Board LEI
- **Cross-Asset**: CME Group, ICE, various futures exchanges

## Summary of Data Gaps

### What Polygon.io Provides:
- ✅ 1-minute+ bar data for stocks
- ✅ Daily historical data
- ✅ Basic OHLCV information
- ✅ Some options data

### What Advanced Models Need:
- ❌ Tick-level/microsecond data
- ❌ Text/NLP data sources
- ❌ Categorical metadata
- ❌ Forward-looking calendars
- ❌ True continuous-time data
- ❌ Multi-modal data combinations
- ❌ Regime indicators
- ❌ Cross-asset correlations

## Recommended Data Integration Priority

### Phase 1: Enable Core Advanced Models
1. **High-Frequency Data Provider** (Refinitiv or Bloomberg)
   - Enables: fractal models, neural ODEs, signal processing
   - Cost: $$$$ (Enterprise pricing)

2. **Financial Text API** (Benzinga or Refinitiv News)
   - Enables: transformer models, NLP
   - Cost: $$$ (Mid-tier pricing)

### Phase 2: Enhanced Capabilities
3. **Event Calendar API** (Trading Economics)
   - Enables: temporal fusion transformer
   - Cost: $$ (Affordable)

4. **Microstructure Data** (LOBSTER or exchange feeds)
   - Enables: advanced fractal analysis
   - Cost: $$$$ (Expensive)

### Phase 3: Specialized Sources
5. **Alternative Data** (As needed for specific strategies)
   - Satellite, weather, shipping, etc.
   - Cost: Varies widely

## Implementation Notes

⚠️ **Important**: All data sources must be:
1. Integrated with the data lineage system
2. Tagged at the source and maintained indefinitely
3. Properly authenticated and rate-limited
4. Compliant with vendor terms of service
5. Cached appropriately to minimize API calls

## Cost Considerations

Enabling these advanced models requires significant investment:
- **Minimum Budget**: $5,000-10,000/month for basic coverage
- **Comprehensive Coverage**: $20,000-50,000/month
- **Enterprise Level**: $100,000+/month

Consider starting with a subset of models that provide the most value for your specific use case.