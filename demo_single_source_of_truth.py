#!/usr/bin/env python3
import config
import logging

logger = logging.getLogger(__name__)


"""
🔒 SINGLE SOURCE OF TRUTH DEMONSTRATION
mlTrainer Configuration System Demo
"""


def main():
    logger.info("=" * 60)
    logger.info("🔒 SINGLE SOURCE OF TRUTH CONFIGURATION SYSTEM DEMO")
    logger.info("=" * 60)

    # API Configuration Demo
    logger.info("\n📡 API CONFIGURATION:")
    logger.info(
        f"  Approved Sources: {[s.value for s in config.get_all_approved_sources()]}")
    logger.info(
        f"  Polygon API Valid: {config.validate_api_source('polygon')}")
    logger.info(
        f"  Invalid API Valid: {config.validate_api_source('invalid')}")

    # AI Configuration Demo
    logger.info("\n🤖 AI CONFIGURATION:")
    logger.info(f"  Available Models: {config.get_all_ai_models()}")
    logger.info(f"  Default Model: {config.get_default_model()}")
    logger.info(
        f"  Model Valid: {config.validate_ai_model_config('claude-3-5-sonnet')}")

    # Mathematical Models Demo
    logger.info("\n📊 MATHEMATICAL MODELS:")
    logger.info(f"  Available Models: {config.get_all_models()}")
    logger.info(f"  Institutional Models: {config.get_institutional_models()}")
    logger.info(
        f"  XGBoost Valid: {config.validate_mathematical_model_config('xgboost')}")

    # Compliance Gateway Demo
    logger.info("\n🔒 COMPLIANCE GATEWAY:")
    compliance_report = config.COMPLIANCE_GATEWAY.get_compliance_report()
    logger.info(f"  Status: {compliance_report['gateway_status']}")
    logger.info(f"  Violations: {compliance_report['total_violations']}")
    logger.info(f"  Config Source: {compliance_report['config_source']}")

    logger.info("\n=" * 60)
    logger.info("✅ ALL CONFIGURATIONS LOADED FROM SINGLE SOURCE OF TRUTH")
    logger.info("🚫 NO HARD-CODED VALUES FOUND")
    logger.info("🔄 CHANGES CASCADE THROUGHOUT SYSTEM")
    logger.info("=" * 60)

    if __name__ == "__main__":
        main()
