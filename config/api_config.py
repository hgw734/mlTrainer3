#!/usr/bin/env python3
"""
🔒 SINGLE SOURCE OF TRUTH - API CONFIGURATION
mlTrainer Institutional Data Sources

CENTRALIZED API CONFIGURATION - NO HARD-CODED VALUES
All API endpoints, authentication, and parameters defined here ONLY
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from .secrets_manager import get_required_secret, get_secret


@dataclass
class APIEndpoint:
    """Approved API endpoint configuration"""

    name: str
    base_url: str
    requires_auth: bool
    auth_header: Optional[str] = None
    rate_limit_per_minute: int = 60
    compliance_verified: bool = True
    timeout_seconds: int = 30
    retry_attempts: int = 3
    retry_delay_seconds: int = 1


    @dataclass
    class APIAuthentication:
        """API authentication configuration"""

        api_key: str
        auth_method: str  # 'bearer_token', 'query_param', 'header'
        header_name: Optional[str] = None
        header_value: Optional[str] = None
        param_name: Optional[str] = None
        param_value: Optional[str] = None


        class APISource(Enum):
            """Approved API data sources"""

            POLYGON = "polygon"
            FRED = "fred"


            class ComplianceLevel(Enum):
                """API compliance levels"""

                INSTITUTIONAL = "institutional"
                VERIFIED = "verified"
                RESTRICTED = "restricted"


                # ================================
                # API KEYS - FROM SECURE SECRETS MANAGER
                # ================================
                # NO HARDCODED VALUES - All keys from environment variables via secrets manager
                def _get_polygon_api_key() -> str:
                    """Get Polygon API key from secure storage"""
                    return get_required_secret("polygon_api_key")


                def _get_fred_api_key() -> str:
                    """Get FRED API key from secure storage"""
                    return get_required_secret("fred_api_key")


                # ================================
                # API KEY EXPORTS - FOR BACKWARD COMPATIBILITY
                # ================================
                # These exports allow existing code to import API keys directly
                # while still using the secure secrets manager
                POLYGON_API_KEY = _get_polygon_api_key()
                FRED_API_KEY = _get_fred_api_key()

                # ================================
                # APPROVED API ENDPOINTS - SINGLE SOURCE OF TRUTH
                # ================================
                APPROVED_ENDPOINTS: Dict[APISource, Dict[str, APIEndpoint]] = {
                APISource.POLYGON: {
                "stocks_aggregates": APIEndpoint(
                name="Polygon Stocks Aggregates",
                base_url="https://api.polygon.io/v2/aggs/ticker",
                requires_auth=True,
                auth_header="Bearer",
                rate_limit_per_minute=100,
                compliance_verified=True,
                timeout_seconds=30,
                retry_attempts=3,
                retry_delay_seconds=1,
                ),
                "stocks_quotes": APIEndpoint(
                name="Polygon Stocks Quotes",
                base_url="https://api.polygon.io/v3/quotes",
                requires_auth=True,
                auth_header="Bearer",
                rate_limit_per_minute=100,
                compliance_verified=True,
                timeout_seconds=30,
                retry_attempts=3,
                retry_delay_seconds=1,
                ),
                "stocks_trades": APIEndpoint(
                name="Polygon Stocks Trades",
                base_url="https://api.polygon.io/v3/trades",
                requires_auth=True,
                auth_header="Bearer",
                rate_limit_per_minute=100,
                compliance_verified=True,
                timeout_seconds=30,
                retry_attempts=3,
                retry_delay_seconds=1,
                ),
                "market_status": APIEndpoint(
                name="Polygon Market Status",
                base_url="https://api.polygon.io/v1/marketstatus/now",
                requires_auth=True,
                auth_header="Bearer",
                rate_limit_per_minute=100,
                compliance_verified=True,
                timeout_seconds=10,
                retry_attempts=2,
                retry_delay_seconds=1,
                ),
                "ticker_details": APIEndpoint(
                name="Polygon Ticker Details",
                base_url="https://api.polygon.io/v3/reference/tickers",
                requires_auth=True,
                auth_header="Bearer",
                rate_limit_per_minute=100,
                compliance_verified=True,
                timeout_seconds=30,
                retry_attempts=3,
                retry_delay_seconds=1,
                ),
                "options_contracts": APIEndpoint(
                name="Polygon Options Contracts",
                base_url="https://api.polygon.io/v3/reference/options/contracts",
                requires_auth=True,
                auth_header="Bearer",
                rate_limit_per_minute=100,
                compliance_verified=True,
                timeout_seconds=30,
                retry_attempts=3,
                retry_delay_seconds=1,
                ),
                "forex_rates": APIEndpoint(
                name="Polygon Forex Rates",
                base_url="https://api.polygon.io/v2/aggs/ticker",
                requires_auth=True,
                auth_header="Bearer",
                rate_limit_per_minute=100,
                compliance_verified=True,
                timeout_seconds=30,
                retry_attempts=3,
                retry_delay_seconds=1,
                ),
                },
                APISource.FRED: {
                "series": APIEndpoint(
                name="FRED Economic Series",
                base_url="https://api.stlouisfed.org/fred/series",
                requires_auth=True,
                auth_header="api_key",
                rate_limit_per_minute=120,
                compliance_verified=True,
                timeout_seconds=30,
                retry_attempts=3,
                retry_delay_seconds=1,
                ),
                "series_observations": APIEndpoint(
                name="FRED Series Observations",
                base_url="https://api.stlouisfed.org/fred/series/observations",
                requires_auth=True,
                auth_header="api_key",
                rate_limit_per_minute=120,
                compliance_verified=True,
                timeout_seconds=30,
                retry_attempts=3,
                retry_delay_seconds=1,
                ),
                "series_search": APIEndpoint(
                name="FRED Series Search",
                base_url="https://api.stlouisfed.org/fred/series/search",
                requires_auth=True,
                auth_header="api_key",
                rate_limit_per_minute=120,
                compliance_verified=True,
                timeout_seconds=30,
                retry_attempts=3,
                retry_delay_seconds=1,
                ),
                "categories": APIEndpoint(
                name="FRED Categories",
                base_url="https://api.stlouisfed.org/fred/category",
                requires_auth=True,
                auth_header="api_key",
                rate_limit_per_minute=120,
                compliance_verified=True,
                timeout_seconds=30,
                retry_attempts=3,
                retry_delay_seconds=1,
                ),
                "series_categories": APIEndpoint(
                name="FRED Series Categories",
                base_url="https://api.stlouisfed.org/fred/series/categories",
                requires_auth=True,
                auth_header="api_key",
                rate_limit_per_minute=120,
                compliance_verified=True,
                timeout_seconds=30,
                retry_attempts=3,
                retry_delay_seconds=1,
                ),
                "releases": APIEndpoint(
                name="FRED Releases",
                base_url="https://api.stlouisfed.org/fred/releases",
                requires_auth=True,
                auth_header="api_key",
                rate_limit_per_minute=120,
                compliance_verified=True,
                timeout_seconds=30,
                retry_attempts=3,
                retry_delay_seconds=1,
                ),
                },
                }


                # ================================
                # API AUTHENTICATION - SINGLE SOURCE OF TRUTH
                # ================================
                def get_api_auth_config(source: APISource) -> APIAuthentication:
                    """
                    Get API authentication configuration dynamically
                    This ensures API keys are always fetched from secure storage
                    """
                    if source == APISource.POLYGON:
                        api_key = _get_polygon_api_key()
                        return APIAuthentication(
                            api_key=api_key, auth_method="bearer_token", header_name="Authorization", header_value=f"Bearer {api_key}"
                        )
                    elif source == APISource.FRED:
                        api_key = _get_fred_api_key()
                        return APIAuthentication(api_key=api_key, auth_method="query_param", param_name="api_key", param_value=api_key)
                    else:
                        raise ValueError(f"Unknown API source: {source}")


                # ================================
                # API COMPLIANCE CONFIGURATION
                # ================================
                API_COMPLIANCE_CONFIG = {
                    "max_data_age_seconds": 3600,  # 1 hour
                    "required_compliance_level": ComplianceLevel.INSTITUTIONAL,
                    "audit_all_requests": True,
                    "log_compliance_violations": True,
                    "reject_non_compliant_data": True,
                    "enforce_rate_limits": True,
                    "require_data_provenance": True,
                }

                # ================================
                # API OPERATIONAL SETTINGS
                # ================================
                API_OPERATIONAL_CONFIG = {
                    "default_timeout_seconds": 30,
                    "default_retry_attempts": 3,
                    "default_retry_delay_seconds": 1,
                    "connection_pool_size": 10,
                    "keep_alive_timeout": 30,
                    "verify_ssl": True,
                    "user_agent": "mlTrainer-Institutional/1.0",
                }


                # ================================
                # UTILITY FUNCTIONS - SINGLE SOURCE OF TRUTH ACCESS
                # ================================
                def get_approved_endpoint(source: APISource, endpoint_name: str) -> Optional[APIEndpoint]:
                    """Get approved endpoint configuration - SINGLE SOURCE OF TRUTH"""
                    if source in APPROVED_ENDPOINTS and endpoint_name in APPROVED_ENDPOINTS[source]:
                        return APPROVED_ENDPOINTS[source][endpoint_name]
                    return None


                def get_auth_config(source: APISource) -> Optional[APIAuthentication]:
                    """Get authentication configuration for approved source - SINGLE SOURCE OF TRUTH"""
                    try:
                        return get_api_auth_config(source)
                    except ValueError:
                        return None


                def get_all_endpoints_for_source(source: APISource) -> Dict[str, APIEndpoint]:
                    """Get all endpoints for a specific source - SINGLE SOURCE OF TRUTH"""
                    return APPROVED_ENDPOINTS.get(source, {})


                def get_all_approved_sources() -> List[APISource]:
                    """Get all approved API sources - SINGLE SOURCE OF TRUTH"""
                    return list(APPROVED_ENDPOINTS.keys())


                def validate_api_source(source_name: str) -> bool:
                    """Validate if API source is approved - SINGLE SOURCE OF TRUTH"""
                    try:
                        APISource(source_name.lower())
                        return True
                    except ValueError:
                        return False


                def get_compliance_config() -> Dict[str, Any]:
                    """Get API compliance configuration - SINGLE SOURCE OF TRUTH"""
                    return API_COMPLIANCE_CONFIG.copy()


                def get_operational_config() -> Dict[str, Any]:
                    """Get API operational configuration - SINGLE SOURCE OF TRUTH"""
                    return API_OPERATIONAL_CONFIG.copy()


                def get_api_key(source: APISource) -> Optional[str]:
                    """Get API key for source - SINGLE SOURCE OF TRUTH"""
                    auth_config = get_auth_config(source)
                    return auth_config.api_key if auth_config else None


                def get_rate_limit(source: APISource, endpoint_name: str) -> int:
                    """Get rate limit for specific endpoint - SINGLE SOURCE OF TRUTH"""
                    endpoint = get_approved_endpoint(source, endpoint_name)
                    return endpoint.rate_limit_per_minute if endpoint else 60


                def is_endpoint_compliant(source: APISource, endpoint_name: str) -> bool:
                    """Check if endpoint is compliance verified - SINGLE SOURCE OF TRUTH"""
                    endpoint = get_approved_endpoint(source, endpoint_name)
                    return endpoint.compliance_verified if endpoint else False


                # ====================================================================
                # EXPORT FOR COMPLIANCE SYSTEM
                # ====================================================================

                # Export authentication configuration for compliance system
                API_AUTH_CONFIG = {
                    "polygon": {
                        "auth": _get_polygon_api_key(),  # This should ideally be _get_polygon_api_auth()
                        "endpoints": APPROVED_ENDPOINTS[APISource.POLYGON],
                        "compliance_level": ComplianceLevel.INSTITUTIONAL.value,
                    },
                    "fred": {
                        "auth": _get_fred_api_key(),  # This should ideally be _get_fred_api_auth()
                        "endpoints": APPROVED_ENDPOINTS[APISource.FRED],
                        "compliance_level": ComplianceLevel.INSTITUTIONAL.value,
                    },
                }

                # ================================
                # EXPORT CONFIGURATION - SINGLE SOURCE OF TRUTH
                # ================================
                __all__ = [
                    # Core Classes
                    "APIEndpoint",
                    "APIAuthentication",
                    "APISource",
                    "ComplianceLevel",
                    # Configuration Dictionaries
                    "APPROVED_ENDPOINTS",
                    "API_COMPLIANCE_CONFIG",
                    "API_OPERATIONAL_CONFIG",
                    "API_AUTH_CONFIG",  # Added this line
                    # Utility Functions
                    "get_approved_endpoint",
                    "get_auth_config",
                    "get_all_endpoints_for_source",
                    "get_all_approved_sources",
                    "validate_api_source",
                    "get_compliance_config",
                    "get_operational_config",
                    "get_api_key",
                    "get_rate_limit",
                    "is_endpoint_compliant",
                ]
