import os
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

# Scanner Configuration
SCANNER_CONFIG: Dict[str, Any] = {
    # Performance settings
    'batch_size': 25,                    # Symbols to process in parallel
    'max_workers': 4,                    # Maximum thread workers
    'rate_limit_delay': 1.0,             # Seconds between API batches
    'analysis_timeout': 30,              # Timeout per symbol analysis (seconds)
    
    # Data filtering
    'min_price': 5.0,                    # Minimum stock price ($)
    'max_price': 1000.0,                 # Maximum stock price ($)
    'min_volume': 500000,                # Minimum average volume
    'min_market_cap': 100e6,             # Minimum market cap ($100M)
    
    # Technical analysis parameters
    'rsi_periods': [2, 14],              # RSI calculation periods
    'macd_params': {
        'fast': 12,
        'slow': 26,
        'signal': 9
    },
    'bb_period': 20,                     # Bollinger Bands period
    'bb_std': 2,                         # Bollinger Bands standard deviations
    'atr_period': 14,                    # ATR calculation period
    'adx_period': 14,                    # ADX calculation period
    
    # Momentum analysis
    'momentum_periods': [3, 5, 10, 20, 50],  # Momentum calculation periods
    'momentum_weights': {
        'short_term': [0.4, 0.6, 0.0, 0.0, 0.0],    # 3d, 5d focus
        'medium_term': [0.1, 0.2, 0.4, 0.3, 0.0],   # 10d, 20d focus
        'long_term': [0.0, 0.1, 0.2, 0.3, 0.4],     # 20d, 50d focus
        'balanced': [0.15, 0.25, 0.25, 0.20, 0.15]  # All timeframes
    },
    
    # Risk management
    'max_volatility': 0.60,              # Maximum annual volatility (60%)
    'min_sharpe_ratio': -2.0,            # Minimum Sharpe ratio threshold
    'max_drawdown': 0.75,                # Maximum drawdown threshold (75%)
    'risk_free_rate': 0.02,              # Annual risk-free rate (2%)
    'var_confidence_levels': [0.95, 0.99], # VaR confidence levels
    
    # Scoring weights
    'scoring_weights': {
        'technical': 0.40,
        'fundamental': 0.35,
        'sentiment': 0.25
    },
    
    # Market regime adaptive weights
    'regime_adjustments': {
        'BULLISH': {
            'technical': 1.1,
            'fundamental': 0.9,
            'sentiment': 1.0
        },
        'BEARISH': {
            'technical': 0.9,
            'fundamental': 1.2,
            'sentiment': 0.8
        },
        'VOLATILE': {
            'technical': 1.3,
            'fundamental': 0.8,
            'sentiment': 0.9
        },
        'NEUTRAL': {
            'technical': 1.0,
            'fundamental': 1.0,
            'sentiment': 1.0
        }
    },
    
    # Institutional scoring bonuses
    'bonus_points': {
        'strong_buy_rating': 15,
        'earnings_beat': 15,
        'guidance_raise': 10,
        'elite_quant_score': 10,
        'institutional_backing': 5,
        'volume_breakout': 5,
        'momentum_acceleration': 10
    },
    
    # Volume analysis
    'volume_breakout_threshold': 2.0,    # 2x average volume for breakout
    'volume_confirmation_period': 5,     # Days to confirm volume trend
    
    # Pattern recognition
    'pattern_confidence_threshold': 0.6,  # Minimum pattern confidence
    'pattern_lookback_days': 30,          # Days to look for patterns
    
    # Cache settings
    'cache_ttl': {
        'market_data': 300,              # 5 minutes for market data
        'company_info': 86400,           # 24 hours for company info
        'analyst_data': 3600,            # 1 hour for analyst data
        'earnings_data': 21600,          # 6 hours for earnings data
        'news_sentiment': 1800,          # 30 minutes for news
        'social_sentiment': 900          # 15 minutes for social data
    }
}

# API Configuration
API_CONFIG: Dict[str, Any] = {
    'polygon': {
        'base_url': 'https://api.polygon.io',
        'rate_limit': 5,                 # Requests per minute for free tier
        'timeout': 30,                   # Request timeout (seconds)
        'retry_attempts': 3,             # Number of retry attempts
        'retry_delay': 1.0              # Delay between retries
    },
    
    'alpha_vantage': {
        'base_url': 'https://www.alphavantage.co/query',
        'rate_limit': 5,                 # Requests per minute
        'timeout': 30,
        'retry_attempts': 3,
        'retry_delay': 1.0
    },
    
    'news_api': {
        'base_url': 'https://newsapi.org/v2',
        'rate_limit': 1000,              # Requests per day
        'timeout': 30,
        'retry_attempts': 2,
        'retry_delay': 1.0
    },
    
    # Reddit API (for social sentiment)
    'reddit': {
        'user_agent': 'MomentumScanner/1.0',
        'rate_limit': 60,                # Requests per minute
        'timeout': 30,
        'subreddits': [
            'wallstreetbets',
            'stocks',
            'investing', 
            'SecurityAnalysis',
            'ValueInvesting',
            'StockMarket'
        ]
    },
    
    # Twitter API (for social sentiment)
    'twitter': {
        'api_version': 'v2',
        'rate_limit': 300,               # Requests per 15 minutes
        'timeout': 30,
        'max_results': 100               # Results per request
    }
}

# Database Configuration
DATABASE_CONFIG: Dict[str, Any] = {
    'scan_db_path': 'data/scanner.db',
    'cache_db_path': 'data/cache.db',
    'backup_interval_days': 7,
    'cleanup_old_data_days': 90,
    'vacuum_interval_days': 30
}

# Logging Configuration
LOGGING_CONFIG: Dict[str, Any] = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_handler': {
        'enabled': True,
        'filename': 'logs/scanner.log',
        'max_bytes': 10485760,           # 10MB
        'backup_count': 5
    },
    'console_handler': {
        'enabled': True
    }
}

# Market Hours Configuration (Eastern Time)
MARKET_HOURS: Dict[str, Any] = {
    'regular_hours': {
        'start': '09:30',
        'end': '16:00'
    },
    'pre_market': {
        'start': '04:00',
        'end': '09:30'
    },
    'after_hours': {
        'start': '16:00',
        'end': '20:00'
    },
    'timezone': 'US/Eastern'
}

# VIX Thresholds for Market Regime Detection
VIX_THRESHOLDS: Dict[str, float] = {
    'EXTREME_FEAR': 40.0,
    'HIGH_FEAR': 30.0,
    'ELEVATED_FEAR': 20.0,
    'NORMAL': 12.0,
    'LOW_FEAR': 0.0
}

# Sector Classifications
SECTOR_MAPPINGS: Dict[str, List[str]] = {
    'Technology': ['Software', 'Hardware', 'Semiconductors', 'Internet'],
    'Healthcare': ['Pharmaceuticals', 'Biotechnology', 'Medical Devices', 'Healthcare Services'],
    'Financial': ['Banks', 'Insurance', 'Real Estate', 'Asset Management'],
    'Consumer Discretionary': ['Retail', 'Automotive', 'Media', 'Hotels & Restaurants'],
    'Consumer Staples': ['Food & Beverages', 'Household Products', 'Personal Care'],
    'Industrial': ['Aerospace & Defense', 'Construction', 'Transportation', 'Machinery'],
    'Energy': ['Oil & Gas', 'Renewable Energy', 'Energy Equipment'],
    'Materials': ['Chemicals', 'Metals & Mining', 'Paper & Packaging'],
    'Utilities': ['Electric Utilities', 'Gas Utilities', 'Water Utilities'],
    'Communication': ['Telecommunications', 'Media', 'Entertainment'],
    'Real Estate': ['REITs', 'Real Estate Services']
}

# Export Settings
EXPORT_CONFIG: Dict[str, Any] = {
    'formats': ['csv', 'excel', 'json'],
    'default_format': 'csv',
    'include_metadata': True,
    'max_records_per_export': 10000,
    'export_directory': 'exports'
}

# Alert Settings
ALERT_CONFIG: Dict[str, Any] = {
    'high_score_threshold': 85.0,
    'volume_spike_threshold': 3.0,      # 3x average volume
    'momentum_acceleration_threshold': 20.0,  # 20% momentum increase
    'max_alerts_per_scan': 10,
    'alert_cooldown_hours': 6           # Hours between same-symbol alerts
}

# Environment Variables with Defaults
def get_env_config() -> Dict[str, str]:
    """Get environment variables with defaults"""
    return {
        'POLYGON_API_KEY': os.getenv('POLYGON_API_KEY', ''),
        'ALPHA_VANTAGE_API_KEY': os.getenv('ALPHA_VANTAGE_API_KEY', ''),
        'NEWS_API_KEY': os.getenv('NEWS_API_KEY', ''),
        'REDDIT_CLIENT_ID': os.getenv('REDDIT_CLIENT_ID', ''),
        'REDDIT_CLIENT_SECRET': os.getenv('REDDIT_CLIENT_SECRET', ''),
        'TWITTER_BEARER_TOKEN': os.getenv('TWITTER_BEARER_TOKEN', ''),
        'DEBUG_MODE': os.getenv('DEBUG_MODE', 'False').lower() == 'true',
        'LOG_LEVEL': os.getenv('LOG_LEVEL', 'INFO'),
        'MAX_WORKERS': int(os.getenv('MAX_WORKERS', '4')),
        'BATCH_SIZE': int(os.getenv('BATCH_SIZE', '25'))
    }

# Validation Functions
def validate_config() -> bool:
    """Validate configuration settings"""
    try:
        env_config = get_env_config()
        
        # Check required API keys
        required_keys = ['POLYGON_API_KEY']
        missing_keys = [key for key in required_keys if not env_config.get(key)]
        
        if missing_keys:
            logger.warning(f"Missing required API keys: {missing_keys}")
            logger.warning("Some features may be limited without proper API keys")
        
        # Validate numeric ranges
        if SCANNER_CONFIG['min_price'] >= SCANNER_CONFIG['max_price']:
            logger.error("Invalid price range: min_price >= max_price")
            return False
        
        if SCANNER_CONFIG['batch_size'] <= 0:
            logger.error("Invalid batch_size: must be > 0")
            return False
        
        if SCANNER_CONFIG['max_workers'] <= 0:
            logger.error("Invalid max_workers: must be > 0")
            return False
        
        # Validate scoring weights sum to 1.0
        total_weight = sum(SCANNER_CONFIG['scoring_weights'].values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Scoring weights don't sum to 1.0: {total_weight}")
        
        return True
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False

# Update configuration from environment
def update_config_from_env():
    """Update configuration from environment variables"""
    try:
        env_config = get_env_config()
        
        # Update scanner config from environment
        if env_config.get('MAX_WORKERS'):
            SCANNER_CONFIG['max_workers'] = int(env_config['MAX_WORKERS'])
        
        if env_config.get('BATCH_SIZE'):
            SCANNER_CONFIG['batch_size'] = int(env_config['BATCH_SIZE'])
        
        # Update logging level
        if env_config.get('LOG_LEVEL'):
            LOGGING_CONFIG['level'] = env_config['LOG_LEVEL']
        
        logger.info("Configuration updated from environment variables")
        
    except Exception as e:
        logger.error(f"Failed to update config from environment: {e}")

# Initialize configuration
if __name__ == "__main__":
    update_config_from_env()
    if validate_config():
        logger.info("Configuration validation passed")
    else:
        logger.error("Configuration validation failed")
