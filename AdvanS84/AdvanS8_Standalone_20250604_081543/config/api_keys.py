"""
Centralized API key management for the momentum scanner system.
All external service credentials are managed here.
"""

import os
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

class APIKeyManager:
    """Centralized management of all API keys and external service credentials"""
    
    def __init__(self):
        """Initialize API key manager"""
        self._keys = {}
        self._load_keys()
    
    def _load_keys(self):
        """Load all API keys from environment variables"""
        # Load from .env file first
        from dotenv import load_dotenv
        load_dotenv()
        
        # Market data API keys
        self._keys['polygon'] = os.getenv('POLYGON_API_KEY') or os.getenv('reverent_nash') or 'DKYSsJRspRnuO2N5pp7dJpznTpQ6OF4d'
        
        # Communication API keys  
        self._keys['twilio_sid'] = os.getenv('TWILIO_ACCOUNT_SID')
        self._keys['twilio_token'] = os.getenv('TWILIO_AUTH_TOKEN')
        self._keys['twilio_phone'] = os.getenv('TWILIO_PHONE_NUMBER')
        
        # Database credentials
        self._keys['database_url'] = os.getenv('DATABASE_URL')
        
        # Log available keys (without revealing values)
        available_keys = [k for k, v in self._keys.items() if v is not None]
        logger.info(f"API keys loaded: {', '.join(available_keys)}")
    
    def get_polygon_key(self) -> Optional[str]:
        """Get Polygon API key for market data"""
        return self._keys.get('polygon')
    
    def get_twilio_credentials(self) -> Dict[str, Optional[str]]:
        """Get Twilio credentials for SMS alerts"""
        return {
            'account_sid': self._keys.get('twilio_sid'),
            'auth_token': self._keys.get('twilio_token'),
            'phone_number': self._keys.get('twilio_phone')
        }
    
    def get_database_url(self) -> Optional[str]:
        """Get database connection URL"""
        return self._keys.get('database_url')
    
    def is_polygon_available(self) -> bool:
        """Check if Polygon API key is available"""
        return self._keys.get('polygon') is not None
    
    def is_twilio_available(self) -> bool:
        """Check if Twilio credentials are available"""
        twilio_creds = self.get_twilio_credentials()
        return all(twilio_creds.values())
    
    def is_database_available(self) -> bool:
        """Check if database connection is available"""
        return self._keys.get('database_url') is not None
    
    def get_masked_polygon_key(self) -> str:
        """Get masked version of Polygon key for logging"""
        key = self._keys.get('polygon')
        if key and len(key) > 10:
            return f"{key[:10]}..."
        return "Not configured"
    
    def validate_all_keys(self) -> Dict[str, bool]:
        """Validate all API keys and return status"""
        return {
            'polygon': self.is_polygon_available(),
            'twilio': self.is_twilio_available(),
            'database': self.is_database_available()
        }

# Global instance for centralized access
api_keys = APIKeyManager()

def get_polygon_key() -> Optional[str]:
    """Quick access function for Polygon API key"""
    return api_keys.get_polygon_key()

def get_twilio_credentials() -> Dict[str, Optional[str]]:
    """Quick access function for Twilio credentials"""
    return api_keys.get_twilio_credentials()

def get_database_url() -> Optional[str]:
    """Quick access function for database URL"""
    return api_keys.get_database_url()