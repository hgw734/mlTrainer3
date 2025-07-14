#!/usr/bin/env python3
"""
Validate Config Script
"""

import os
import json
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_config():
    """Validate all configuration files"""
    logger.info("🔍 Validating Configuration Files")
    logger.info("=" * 50)
    
    # Validate different config files
    validate_environment_config()
    validate_security_config()
    validate_models_config()
    validate_api_config()
    
    logger.info("✅ Configuration validation complete!")

def validate_environment_config():
    """Validate environment configuration"""
    logger.info("Checking environment configuration...")
    
    # Check .env file
    env_file = '.env'
    if os.path.exists(env_file):
        logger.info(f"✅ {env_file} exists")
        
        # Check required environment variables
        required_vars = [
            'MLTRAINER_ENV',
            'MLTRAINER_SECURE_MODE',
            'MLTRAINER_GOVERNANCE_ENABLED'
        ]
        
        for var in required_vars:
            if os.getenv(var):
                logger.info(f"✅ Environment variable {var} is set")
            else:
                logger.warning(f"⚠️  Environment variable {var} not set")
    else:
        logger.error(f"❌ {env_file} not found")

def validate_security_config():
    """Validate security configuration"""
    logger.info("Checking security configuration...")
    
    security_file = 'config/security_config.json'
    if os.path.exists(security_file):
        try:
            with open(security_file, 'r') as f:
                config = json.load(f)
            
            # Check required security settings
            required_settings = ['api_keys', 'security', 'governance']
            for setting in required_settings:
                if setting in config:
                    logger.info(f"✅ Security setting '{setting}' found")
                else:
                    logger.warning(f"⚠️  Security setting '{setting}' missing")
            
            logger.info(f"✅ {security_file} is valid JSON")
        except json.JSONDecodeError as e:
            logger.error(f"❌ {security_file} has invalid JSON: {e}")
    else:
        logger.error(f"❌ {security_file} not found")

def validate_models_config():
    """Validate models configuration"""
    logger.info("Checking models configuration...")
    
    models_file = 'config/models_config.py'
    if os.path.exists(models_file):
        logger.info(f"✅ {models_file} exists")
        
        # Check if file can be imported
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("models_config", models_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            logger.info(f"✅ {models_file} can be imported successfully")
        except Exception as e:
            logger.error(f"❌ {models_file} has import errors: {e}")
    else:
        logger.error(f"❌ {models_file} not found")

def validate_api_config():
    """Validate API configuration"""
    logger.info("Checking API configuration...")
    
    api_file = 'config/api_config.py'
    if os.path.exists(api_file):
        logger.info(f"✅ {api_file} exists")
        
        # Check if file can be imported
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("api_config", api_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            logger.info(f"✅ {api_file} can be imported successfully")
        except Exception as e:
            logger.error(f"❌ {api_file} has import errors: {e}")
    else:
        logger.error(f"❌ {api_file} not found")

def check_config_completeness():
    """Check if all required config files exist"""
    logger.info("Checking config completeness...")
    
    required_files = [
        '.env',
        'config/security_config.json',
        'config/models_config.py',
        'config/api_config.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        logger.warning(f"⚠️  Missing config files: {missing_files}")
    else:
        logger.info("✅ All required config files exist")

def main():
    """Main function"""
    validate_config()
    check_config_completeness()

if __name__ == "__main__":
    main() 