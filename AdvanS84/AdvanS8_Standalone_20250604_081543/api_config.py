"""
Central API Configuration Management
All API keys and external service configurations in one place
"""

import os

class APIConfig:
    """Centralized API configuration management"""
    
    def __init__(self):
        # Primary API keys - use only these verified values
        self.POLYGON_API_KEY = 'DKYSsJRspRnuO2N5pp7dJpznTpQ6OF4d'
        self.FRED_API_KEY = 'c2a2b890bd1ea280e5786eafabecafc5'
        
        # API endpoints
        self.POLYGON_BASE_URL = "https://api.polygon.io"
        self.FRED_BASE_URL = "https://api.stlouisfed.org/fred"
        
        # Rate limiting settings
        self.POLYGON_RATE_LIMIT = 0.1  # seconds between calls
        self.FRED_RATE_LIMIT = 0.1     # seconds between calls
        
        # Data settings
        self.DEFAULT_LOOKBACK_DAYS = 365
        self.MAX_SYMBOLS_PER_BATCH = 20
        
    def get_polygon_headers(self):
        """Get headers for Polygon API requests"""
        return {
            'Authorization': f'Bearer {self.POLYGON_API_KEY}',
            'Content-Type': 'application/json'
        }
    
    def get_polygon_params(self):
        """Get standard parameters for Polygon API"""
        return {
            'apikey': self.POLYGON_API_KEY
        }
    
    def get_fred_params(self, series_id='VIXCLS'):
        """Get standard parameters for FRED API"""
        return {
            'api_key': self.FRED_API_KEY,
            'file_type': 'json',
            'series_id': series_id
        }
    
    def validate_keys(self):
        """Validate that all required API keys are available"""
        missing_keys = []
        
        if not self.POLYGON_API_KEY or self.POLYGON_API_KEY == 'your_polygon_key_here':
            missing_keys.append('POLYGON_API_KEY')
        
        if not self.FRED_API_KEY or self.FRED_API_KEY == 'your_fred_key_here':
            missing_keys.append('FRED_API_KEY')
        
        return len(missing_keys) == 0, missing_keys

# Global configuration instance
config = APIConfig()

# Convenience functions for easy access
def get_polygon_key():
    """Get Polygon API key"""
    return config.POLYGON_API_KEY

def get_fred_key():
    """Get FRED API key"""
    return config.FRED_API_KEY

def get_polygon_params():
    """Get Polygon API parameters"""
    return config.get_polygon_params()

def get_fred_params(series_id='VIXCLS'):
    """Get FRED API parameters"""
    return config.get_fred_params(series_id)

def validate_api_configuration():
    """Validate API configuration"""
    return config.validate_keys()