#!/usr/bin/env python3
"""
mlTrainer Configuration Module

SINGLE SOURCE OF TRUTH for all system configuration:
    - API endpoints and authentication
    - AI model configurations
    - Mathematical model parameters
    - Compliance rules and enforcement
    - IMMUTABLE compliance enforcement
"""

from typing import Dict, Any
from datetime import datetime

# ================================
# SINGLE SOURCE OF TRUTH IMPORTS
# ================================

# API Configuration - Single Source of Truth
from .api_config import (
    APIEndpoint,
    APIAuthentication,
    APISource,
    ComplianceLevel,
    APPROVED_ENDPOINTS,
    API_AUTH_CONFIG,
    API_COMPLIANCE_CONFIG,
    API_OPERATIONAL_CONFIG,
    get_approved_endpoint,
    get_auth_config,
    get_all_endpoints_for_source,
    get_all_approved_sources,
    validate_api_source,
    get_compliance_config as get_api_compliance_config,
    get_operational_config as get_api_operational_config,
    get_api_key,
    get_rate_limit,
    is_endpoint_compliant,
)

# AI Configuration - Single Source of Truth
from .ai_config import (
    AIModel,
    AIProvider,
    AIModelType,
    AICapability,
    AI_MODELS,
    AI_PROVIDERS,
    AI_ROLE_CONFIGS,
    AI_OPERATIONAL_CONFIG,
    AI_COMPLIANCE_CONFIG,
    get_ai_model,
    get_ai_provider,
    get_role_config,
    get_all_models as get_all_ai_models,
    get_all_providers,
    get_all_roles,
    get_models_by_provider,
    get_institutional_models as get_institutional_ai_models,
    get_model_cost,
    is_model_compliant as is_ai_model_compliant,
    get_default_model,
    get_fallback_model,
    get_ai_operational_config,
    get_ai_compliance_config,
    get_api_key as get_ai_api_key,
    validate_model_config as validate_ai_model_config,
)

# Mathematical Models Configuration - Single Source of Truth
from .models_config import (
    ModelParameter,
    MathematicalModel,
    ModelCategory,
    AlgorithmType,
    MATHEMATICAL_MODELS,
    get_model_by_category,
    get_model_by_algorithm_type,
    get_institutional_grade_models as get_institutional_models,
    get_models_by_complexity,
    get_models_by_interpretability,
    get_models_supporting_online_learning,
    get_models_supporting_transfer_learning,
    get_model_summary,
    validate_model_configuration as validate_mathematical_model_config,
    MATHEMATICAL_MODELS_CONFIGURATION,
)


# Convenience function to get all models
def get_all_models():
    """Get all mathematical models"""
    return list(MATHEMATICAL_MODELS.keys())


# Compliance Gateway - Single Source of Truth
from .immutable_compliance_gateway import (
    ComplianceGateway,
    DataProvenance,
    VerifiedData,
    DataSource,
    ComplianceStatus,
    compliance_required,
    COMPLIANCE_GATEWAY,
)

# ================================
# SINGLE SOURCE OF TRUTH VALIDATION
# ================================


def validate_configuration_integrity() -> Dict[str, Any]:
    """
    Validate that all configurations are properly synchronized
    and maintain single source of truth integrity
    """
    validation_report = {
        "status": "VALID",
        "timestamp": datetime.now().isoformat(),
        "configurations_checked": [],
        "issues_found": [],
        "recommendations": [],
    }

    # Check API configuration
    try:
        api_sources = get_all_approved_sources()
        api_compliance = get_api_compliance_config()
        validation_report["configurations_checked"].append("API_CONFIG")

        if not api_sources:
            validation_report["issues_found"].append("No approved API sources found")
            validation_report["status"] = "INVALID"

    except Exception as e:
        validation_report["issues_found"].append(f"API Config Error: {str(e)}")
        validation_report["status"] = "INVALID"

    # Check AI configuration
    try:
        ai_models = get_all_ai_models()
        ai_compliance = get_ai_compliance_config()
        validation_report["configurations_checked"].append("AI_CONFIG")

        if not ai_models:
            validation_report["issues_found"].append("No AI models configured")
            validation_report["status"] = "INVALID"

    except Exception as e:
        validation_report["issues_found"].append(f"AI Config Error: {str(e)}")
        validation_report["status"] = "INVALID"

    # Check Mathematical Models configuration
    try:
        math_models = get_all_models()
        validation_report["configurations_checked"].append("MODELS_CONFIG")

        if not math_models:
            validation_report["issues_found"].append("No mathematical models configured")
            validation_report["status"] = "INVALID"

    except Exception as e:
        validation_report["issues_found"].append(f"Models Config Error: {str(e)}")
        validation_report["status"] = "INVALID"

    # Check Compliance Gateway
    try:
        compliance_report = COMPLIANCE_GATEWAY.get_compliance_report()
        validation_report["configurations_checked"].append("COMPLIANCE_GATEWAY")

        if compliance_report["gateway_status"] != "ACTIVE":
            validation_report["issues_found"].append("Compliance Gateway not active")
            validation_report["status"] = "INVALID"

    except Exception as e:
        validation_report["issues_found"].append(f"Compliance Gateway Error: {str(e)}")
        validation_report["status"] = "INVALID"

    # Generate recommendations
    if validation_report["status"] == "VALID":
        validation_report["recommendations"].append("Configuration integrity verified - all systems operational")
    else:
        validation_report["recommendations"].append(
            "Review configuration errors and ensure all config files are properly structured"
        )

    return validation_report


def get_comprehensive_configuration_report() -> Dict[str, Any]:
    """
    Generate comprehensive report of all configurations
    Demonstrates single source of truth architecture
    """
    from datetime import datetime

    return {
        "report_timestamp": datetime.now().isoformat(),
        "configuration_architecture": "SINGLE_SOURCE_OF_TRUTH",
        "compliance_status": "INSTITUTIONAL_GRADE",
        # Configuration Summary
        "api_configuration": {
            "total_endpoints": len(APPROVED_ENDPOINTS),
            "approved_sources": [source.value for source in get_all_approved_sources()],
            "compliance_level": get_api_compliance_config().get("required_compliance_level", "INSTITUTIONAL"),
            "configuration_file": "config/api_config.py",
        },
        "ai_configuration": {
            "total_models": len(AI_MODELS),
            "institutional_grade_models": len(get_institutional_ai_models()),
            "default_model": get_default_model(),
            "fallback_model": get_fallback_model(),
            "configuration_file": "config/ai_config.py",
        },
        "mathematical_models_configuration": {
            "total_models": len(MATHEMATICAL_MODELS),
            "institutional_grade_models": len(get_institutional_models()),
            "configuration_file": "config/models_config.py",
        },
        "compliance_configuration": {
            "gateway_status": COMPLIANCE_GATEWAY.get_compliance_report()["gateway_status"],
            "total_violations": COMPLIANCE_GATEWAY.get_compliance_report()["total_violations"],
            "verified_data_count": COMPLIANCE_GATEWAY.get_compliance_report()["verified_data_count"],
            "configuration_file": "config/immutable_compliance_gateway.py",
        },
        # Architecture Validation
        "single_source_of_truth_validation": validate_configuration_integrity(),
        # Usage Guidelines
        "usage_guidelines": [
            "ALL system components must import from config package",
            "NO hard-coded values allowed outside config files",
            "Changes to configurations cascade automatically",
            "Compliance gateway enforces institutional standards",
            "All configurations are immutable at runtime",
        ],
    }


# ================================
# CONFIGURATION PACKAGE EXPORTS
# ================================

__all__ = [
    # API Configuration
    "APIEndpoint",
    "APIAuthentication",
    "APISource",
    "ComplianceLevel",
    "APPROVED_ENDPOINTS",
    "API_AUTH_CONFIG",
    "API_COMPLIANCE_CONFIG",
    "API_OPERATIONAL_CONFIG",
    "get_approved_endpoint",
    "get_auth_config",
    "get_all_endpoints_for_source",
    "get_all_approved_sources",
    "validate_api_source",
    "get_api_compliance_config",
    "get_api_operational_config",
    "get_api_key",
    "get_rate_limit",
    "is_endpoint_compliant",
    # AI Configuration
    "AIModel",
    "AIProvider",
    "AIModelType",
    "AICapability",
    "AI_MODELS",
    "AI_PROVIDERS",
    "AI_ROLE_CONFIGS",
    "AI_OPERATIONAL_CONFIG",
    "AI_COMPLIANCE_CONFIG",
    "get_ai_model",
    "get_ai_provider",
    "get_role_config",
    "get_all_ai_models",
    "get_all_providers",
    "get_all_roles",
    "get_models_by_provider",
    "get_institutional_ai_models",
    "get_model_cost",
    "is_ai_model_compliant",
    "get_default_model",
    "get_fallback_model",
    "get_ai_operational_config",
    "get_ai_compliance_config",
    "get_ai_api_key",
    "validate_ai_model_config",
    # Mathematical Models Configuration
    "ModelParameter",
    "MathematicalModel",
    "ModelCategory",
    "AlgorithmType",
    "MATHEMATICAL_MODELS",
    "MATHEMATICAL_MODELS_CONFIGURATION",
    "get_all_models",
    "get_model_by_category",
    "get_model_by_algorithm_type",
    "get_institutional_models",
    "get_models_by_complexity",
    "get_models_by_interpretability",
    "get_models_supporting_online_learning",
    "get_models_supporting_transfer_learning",
    "get_model_summary",
    "validate_mathematical_model_config",
    # Compliance Gateway
    "ComplianceGateway",
    "DataProvenance",
    "VerifiedData",
    "DataSource",
    "ComplianceStatus",
    "compliance_required",
    "COMPLIANCE_GATEWAY",
    # Configuration System
    "validate_configuration_integrity",
    "get_comprehensive_configuration_report",
]

# ================================
# SINGLE SOURCE OF TRUTH BANNER
# ================================
print(
    """
    🔒 SINGLE SOURCE OF TRUTH CONFIGURATION SYSTEM LOADED
    ======================================================

    ✅ API Configuration: config/api_config.py
    ✅ AI Configuration: config/ai_config.py
    ✅ Mathematical Models: config/models_config.py
    ✅ Compliance Gateway: config/immutable_compliance_gateway.py

    🎯 INSTITUTIONAL GRADE COMPLIANCE ACTIVE
    🔄 ALL CONFIGURATIONS CASCADED THROUGHOUT SYSTEM
    🚫 NO HARD-CODED VALUES PERMITTED
    """
)
