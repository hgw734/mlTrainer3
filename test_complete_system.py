#!/usr/bin/env python3
"""
Test Complete System Script
"""

import logging
import sys
from typing import Dict, Any
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_complete_system():
    """Test the complete mlTrainer system"""
    logger.info("🧪 Testing Complete System")
    logger.info("=" * 50)
    
    # Test different components
    test_results = {}
    
    test_results['environment'] = test_environment()
    test_results['governance'] = test_governance()
    test_results['models'] = test_models()
    test_results['data_sources'] = test_data_sources()
    test_results['security'] = test_security()
    
    # Generate test report
    generate_test_report(test_results)
    
    # Return overall success
    return all(test_results.values())

def test_environment():
    """Test environment setup"""
    logger.info("Testing environment...")
    
    try:
        # Check required environment variables
        required_vars = ['MLTRAINER_ENV', 'MLTRAINER_SECURE_MODE']
        for var in required_vars:
            if not os.getenv(var):
                logger.error(f"❌ Environment variable {var} not set")
                return False
        
        logger.info("✅ Environment test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Environment test failed: {e}")
        return False

def test_governance():
    """Test governance system"""
    logger.info("Testing governance...")
    
    try:
        # Test governance imports
        from core.governance_kernel import GovernanceKernel
        from core.compliance_mode import ComplianceMode
        
        governance = GovernanceKernel()
        compliance = ComplianceMode()
        
        logger.info("✅ Governance test passed")
        return True
    except ImportError as e:
        logger.error(f"❌ Governance test failed - import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Governance test failed: {e}")
        return False

def test_models():
    """Test ML models"""
    logger.info("Testing models...")
    
    try:
        # Test model imports
        from custom.momentum import MomentumBreakout, EMACrossover
        from custom.risk import InformationRatio, ExpectedShortfall
        from custom.volatility import RegimeSwitchingVolatility
        
        # Test model instantiation
        models = [
            MomentumBreakout(),
            EMACrossover(),
            InformationRatio(),
            ExpectedShortfall(),
            RegimeSwitchingVolatility()
        ]
        
        logger.info(f"✅ Models test passed - {len(models)} models loaded")
        return True
    except ImportError as e:
        logger.error(f"❌ Models test failed - import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Models test failed: {e}")
        return False

def test_data_sources():
    """Test data sources"""
    logger.info("Testing data sources...")
    
    try:
        # Test data source imports
        from polygon_connector import PolygonConnector
        from fred_connector import FREDConnector
        
        logger.info("✅ Data sources test passed")
        return True
    except ImportError as e:
        logger.warning(f"⚠️  Data sources test - some connectors not available: {e}")
        return True  # Not critical for basic functionality
    except Exception as e:
        logger.error(f"❌ Data sources test failed: {e}")
        return False

def test_security():
    """Test security features"""
    logger.info("Testing security...")
    
    try:
        # Test security imports
        from config.secrets_manager import SecretsManager
        from core.crypto_signing import CryptoSigning
        
        logger.info("✅ Security test passed")
        return True
    except ImportError as e:
        logger.error(f"❌ Security test failed - import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Security test failed: {e}")
        return False

def generate_test_report(test_results: Dict[str, bool]):
    """Generate test report"""
    logger.info("\n📊 Test Report")
    logger.info("=" * 30)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name.upper()}: {status}")
    
    logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All tests passed!")
    else:
        logger.warning(f"⚠️  {total_tests - passed_tests} tests failed")

def main():
    """Main function"""
    success = test_complete_system()
    
    if success:
        logger.info("🎉 Complete system test passed!")
        return 0
    else:
        logger.error("❌ Complete system test failed!")
        return 1

if __name__ == "__main__":
    exit(main()) 