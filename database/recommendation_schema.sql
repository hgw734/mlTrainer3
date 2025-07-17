-- mlTrainer Recommendation System Database Schema
-- For tracking recommendations, virtual positions, and user selections

-- Recommendations table: All system-generated trading recommendations
CREATE TABLE IF NOT EXISTS recommendations (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    symbol VARCHAR(10) NOT NULL,
    signal_strength FLOAT NOT NULL CHECK (signal_strength >= 0 AND signal_strength <= 1),
    profit_probability FLOAT NOT NULL CHECK (profit_probability >= 0 AND profit_probability <= 1),
    confidence FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    entry_price FLOAT NOT NULL,
    target_price FLOAT NOT NULL,
    stop_loss FLOAT NOT NULL,
    timeframe VARCHAR(20) NOT NULL, -- '7-12 days' or '50-70 days'
    model_used VARCHAR(50) NOT NULL,
    features_json TEXT, -- JSON of key features that triggered the signal
    market_conditions TEXT, -- JSON of market conditions at time of recommendation
    INDEX idx_timestamp (timestamp),
    INDEX idx_symbol (symbol),
    INDEX idx_signal_strength (signal_strength DESC)
);

-- Virtual Portfolio: System's paper trading positions
CREATE TABLE IF NOT EXISTS virtual_positions (
    id SERIAL PRIMARY KEY,
    recommendation_id INT NOT NULL,
    entry_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    entry_price FLOAT NOT NULL,
    shares INT NOT NULL DEFAULT 100, -- Default position size
    current_price FLOAT,
    last_update TIMESTAMP,
    exit_time TIMESTAMP,
    exit_price FLOAT,
    profit_loss FLOAT,
    profit_loss_pct FLOAT,
    status VARCHAR(20) NOT NULL DEFAULT 'OPEN', -- OPEN, CLOSED, STOPPED_OUT
    exit_reason VARCHAR(50), -- TARGET_HIT, STOP_LOSS, TIMEOUT, SIGNAL_REVERSAL
    max_profit FLOAT DEFAULT 0,
    max_loss FLOAT DEFAULT 0,
    hold_days INT,
    FOREIGN KEY (recommendation_id) REFERENCES recommendations(id),
    INDEX idx_status (status),
    INDEX idx_entry_time (entry_time)
);

-- User Portfolio: Manually selected positions by user
CREATE TABLE IF NOT EXISTS user_portfolio (
    id SERIAL PRIMARY KEY,
    recommendation_id INT NOT NULL,
    selected_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    actual_buy_price FLOAT,
    actual_shares INT,
    actual_sell_price FLOAT,
    actual_sell_time TIMESTAMP,
    actual_profit_loss FLOAT,
    notes TEXT,
    status VARCHAR(20) NOT NULL DEFAULT 'WATCHING', -- WATCHING, BOUGHT, SOLD
    FOREIGN KEY (recommendation_id) REFERENCES recommendations(id),
    INDEX idx_selected_time (selected_time)
);

-- Performance Metrics: Track model performance over time
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    total_recommendations INT DEFAULT 0,
    winning_trades INT DEFAULT 0,
    losing_trades INT DEFAULT 0,
    win_rate FLOAT,
    avg_profit_pct FLOAT,
    avg_loss_pct FLOAT,
    sharpe_ratio FLOAT,
    max_drawdown FLOAT,
    total_return_pct FLOAT,
    UNIQUE KEY unique_date_model (date, model_name),
    INDEX idx_date (date)
);

-- Recommendation Feedback: Learn from outcomes
CREATE TABLE IF NOT EXISTS recommendation_feedback (
    id SERIAL PRIMARY KEY,
    recommendation_id INT NOT NULL,
    virtual_outcome VARCHAR(20), -- WIN, LOSS, BREAKEVEN
    virtual_return_pct FLOAT,
    user_selected BOOLEAN DEFAULT FALSE,
    user_outcome VARCHAR(20), -- WIN, LOSS, BREAKEVEN, NULL if not traded
    user_return_pct FLOAT,
    feedback_processed BOOLEAN DEFAULT FALSE,
    processed_time TIMESTAMP,
    FOREIGN KEY (recommendation_id) REFERENCES recommendations(id),
    INDEX idx_processed (feedback_processed)
);

-- Market Conditions: Track market state for context
CREATE TABLE IF NOT EXISTS market_conditions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    vix_level FLOAT,
    spy_trend VARCHAR(20), -- UPTREND, DOWNTREND, SIDEWAYS
    market_regime VARCHAR(20), -- BULL, BEAR, NEUTRAL
    volume_ratio FLOAT, -- Current vs average volume
    breadth_ratio FLOAT, -- Advancing vs declining stocks
    sector_rotation TEXT, -- JSON of sector performance
    INDEX idx_timestamp (timestamp)
);

-- Create views for easy querying
CREATE VIEW active_recommendations AS
SELECT 
    r.*,
    CASE 
        WHEN vp.id IS NOT NULL THEN 'VIRTUAL_POSITION'
        WHEN up.id IS NOT NULL THEN 'USER_WATCHING'
        ELSE 'PENDING'
    END as status
FROM recommendations r
LEFT JOIN virtual_positions vp ON r.id = vp.recommendation_id AND vp.status = 'OPEN'
LEFT JOIN user_portfolio up ON r.id = up.recommendation_id AND up.status = 'WATCHING'
WHERE r.timestamp > NOW() - INTERVAL '30 days';

CREATE VIEW model_performance_summary AS
SELECT 
    model_used,
    COUNT(*) as total_recommendations,
    SUM(CASE WHEN rf.virtual_outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
    SUM(CASE WHEN rf.virtual_outcome = 'LOSS' THEN 1 ELSE 0 END) as losses,
    AVG(rf.virtual_return_pct) as avg_return,
    SUM(CASE WHEN rf.user_selected THEN 1 ELSE 0 END) as user_selections,
    AVG(CASE WHEN rf.user_selected THEN rf.user_return_pct ELSE NULL END) as avg_user_return
FROM recommendations r
JOIN recommendation_feedback rf ON r.id = rf.recommendation_id
WHERE rf.feedback_processed = TRUE
GROUP BY model_used;