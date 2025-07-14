#!/usr/bin/env python3
"""
Setup Secure Environment Script
"""

import os
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_secure_environment():
    """Setup secure environment for mlTrainer"""
    logger.info("🔒 Setting up Secure Environment")
    logger.info("=" * 50)
    
    # Create secure directories
    create_secure_directories()
    
    # Setup environment variables
    setup_environment_variables()
    
    # Create security configuration
    create_security_config()
    
    logger.info("✅ Secure environment setup complete!")

def create_secure_directories():
    """Create secure directories"""
    secure_dirs = [
        'logs',
        'data',
        'config',
        'backups'
    ]
    
    for dir_name in secure_dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, mode=0o750)
            logger.info(f"✅ Created secure directory: {dir_name}")

def setup_environment_variables():
    """Setup environment variables"""
    env_vars = {
        'MLTRAINER_ENV': 'production',
        'MLTRAINER_LOG_LEVEL': 'INFO',
        'MLTRAINER_SECURE_MODE': 'true',
        'MLTRAINER_GOVERNANCE_ENABLED': 'true'
    }
    
    # Create .env file if it doesn't exist
    env_file = '.env'
    if not os.path.exists(env_file):
        with open(env_file, 'w') as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")
        logger.info(f"✅ Created {env_file} with secure defaults")

def create_security_config():
    """Create security configuration"""
    security_config = {
        'api_keys': {
            'polygon_api_key': 'YOUR_POLYGON_API_KEY',
            'fred_api_key': 'YOUR_FRED_API_KEY'
        },
        'security': {
            'enable_encryption': True,
            'enable_audit_logging': True,
            'enable_rate_limiting': True
        },
        'governance': {
            'enable_compliance_checks': True,
            'enable_permission_checks': True,
            'enable_data_validation': True
        }
    }
    
    config_file = 'config/security_config.json'
    if not os.path.exists(config_file):
        import json
        with open(config_file, 'w') as f:
            json.dump(security_config, f, indent=2)
        logger.info(f"✅ Created {config_file}")

def validate_environment():
    """Validate the secure environment"""
    logger.info("🔍 Validating Secure Environment")
    
    # Check required directories
    required_dirs = ['logs', 'data', 'config']
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            logger.info(f"✅ {dir_name} directory exists")
        else:
            logger.error(f"❌ {dir_name} directory missing")
    
    # Check required files
    required_files = ['.env', 'config/security_config.json']
    for file_name in required_files:
        if os.path.exists(file_name):
            logger.info(f"✅ {file_name} exists")
        else:
            logger.error(f"❌ {file_name} missing")
    
    # Check environment variables
    env_vars = ['MLTRAINER_ENV', 'MLTRAINER_SECURE_MODE']
    for var in env_vars:
        if os.getenv(var):
            logger.info(f"✅ Environment variable {var} is set")
        else:
            logger.warning(f"⚠️  Environment variable {var} not set")

def main():
    """Main function"""
    setup_secure_environment()
    validate_environment()

if __name__ == "__main__":
    main() 