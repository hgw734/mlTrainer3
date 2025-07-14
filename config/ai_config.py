#!/usr/bin/env python3
"""
AI Configuration - Single Source of Truth
mlTrainer AI Model and Provider Configuration
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


@dataclass
class AIModel:
    """AI model configuration"""

    name: str
    provider: str
    model_id: str
    api_key_env: str
    base_url: Optional[str] = None
    context_window: int = 4096
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout_seconds: int = 30
    retry_attempts: int = 3
    retry_delay_seconds: int = 1
    supports_streaming: bool = False
    supports_function_calling: bool = False
    cost_per_1k_tokens: float = 0.0
    compliance_verified: bool = True
    institutional_grade: bool = False


@dataclass
class AIProviderConfig:
    """AI provider configuration"""

    name: str
    base_url: str
    auth_method: str
    api_key_env: str
    rate_limit_per_minute: int
    supports_streaming: bool
    supports_function_calling: bool
    compliance_verified: bool
    institutional_grade: bool


class AIModelType(Enum):
    """AI model types"""

    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    FINE_TUNED = "fine_tuned"


class AIProvider(Enum):
    """AI providers"""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    COHERE = "cohere"


class AICapability(Enum):
    """AI capabilities"""

    REASONING = "reasoning"
    ANALYSIS = "analysis"
    CODING = "coding"
    MATH = "math"
    RESEARCH = "research"
    TRADING = "trading"
    PLANNING = "planning"


# ================================
# AI API KEYS - ENVIRONMENT VARIABLES
# ================================
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# ================================
# AI MODELS - SINGLE SOURCE OF TRUTH
# ================================
AI_MODELS: Dict[str, AIModel] = {
    # ANTHROPIC MODELS
    "claude-3-5-sonnet": AIModel(
        name="Claude 3.5 Sonnet",
        provider="anthropic",
        model_id="claude-3-5-sonnet-20241022",
        api_key_env="ANTHROPIC_API_KEY",
        base_url="https://api.anthropic.com",
        context_window=200000,
        max_tokens=4096,
        temperature=0.7,
        top_p=0.9,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        timeout_seconds=30,
        retry_attempts=3,
        retry_delay_seconds=1,
        supports_streaming=True,
        supports_function_calling=True,
        cost_per_1k_tokens=0.003,
        compliance_verified=True,
        institutional_grade=True,
    ),
    "claude-3-haiku": AIModel(
        name="Claude 3 Haiku",
        provider="anthropic",
        model_id="claude-3-haiku-20240307",
        api_key_env="ANTHROPIC_API_KEY",
        base_url="https://api.anthropic.com",
        context_window=200000,
        max_tokens=4096,
        temperature=0.7,
        top_p=0.9,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        timeout_seconds=15,
        retry_attempts=3,
        retry_delay_seconds=1,
        supports_streaming=True,
        supports_function_calling=True,
        cost_per_1k_tokens=0.0005,
        compliance_verified=True,
        institutional_grade=True,
    ),
    # OPENAI MODELS
    "gpt-4-turbo": AIModel(
        name="GPT-4 Turbo",
        provider="openai",
        model_id="gpt-4-turbo-preview",
        api_key_env="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
        context_window=128000,
        max_tokens=4096,
        temperature=0.7,
        top_p=0.9,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        timeout_seconds=30,
        retry_attempts=3,
        retry_delay_seconds=1,
        supports_streaming=True,
        supports_function_calling=True,
        cost_per_1k_tokens=0.01,
        compliance_verified=True,
        institutional_grade=True,
    ),
    "gpt-3.5-turbo": AIModel(
        name="GPT-3.5 Turbo",
        provider="openai",
        model_id="gpt-3.5-turbo",
        api_key_env="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
        context_window=16000,
        max_tokens=4096,
        temperature=0.7,
        top_p=0.9,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        timeout_seconds=20,
        retry_attempts=3,
        retry_delay_seconds=1,
        supports_streaming=True,
        supports_function_calling=True,
        cost_per_1k_tokens=0.002,
        compliance_verified=True,
        institutional_grade=False,
    ),
}

# ================================
# AI PROVIDERS - SINGLE SOURCE OF TRUTH
# ================================
AI_PROVIDERS: Dict[str, AIProviderConfig] = {
    "anthropic": AIProviderConfig(
        name="Anthropic",
        base_url="https://api.anthropic.com",
        auth_method="bearer_token",
        api_key_env="ANTHROPIC_API_KEY",
        rate_limit_per_minute=60,
        supports_streaming=True,
        supports_function_calling=True,
        compliance_verified=True,
        institutional_grade=True,
    ),
    "openai": AIProviderConfig(
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        auth_method="bearer_token",
        api_key_env="OPENAI_API_KEY",
        rate_limit_per_minute=60,
        supports_streaming=True,
        supports_function_calling=True,
        compliance_verified=True,
        institutional_grade=True,
    ),
}

# ================================
# AI ROLE CONFIGURATIONS - SINGLE SOURCE OF TRUTH
# ================================
AI_ROLE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "mltrainer_primary": {
        "model": "claude-3-5-sonnet",
        "role": "ML Training Assistant",
        "system_prompt": """You are mlTrainer, an institutional-grade ML training assistant.

CORE RESPONSIBILITIES:
    - Analyze training parameters and model configurations
    - Provide expert guidance on ML model selection
    - Ensure institutional compliance and data quality
    - Communicate with mlAgent for execution coordination
    - Generate comprehensive training analysis reports

COMPLIANCE REQUIREMENTS:
    - ZERO TOLERANCE for synthetic data
    - Only use verified API data sources (Polygon, FRED)
    - Maintain full audit trail of all operations
    - Ensure institutional-grade quality standards""",
        "temperature": 0.3,
        "max_tokens": 4096,
        "capabilities": [AICapability.REASONING, AICapability.ANALYSIS, AICapability.TRADING, AICapability.PLANNING],
    },
    "mlagent_executor": {
        "model": "claude-3-haiku",
        "role": "ML Execution Agent",
        "system_prompt": """You are mlAgent, the execution engine for mlTrainer.

CORE RESPONSIBILITIES:
    - Execute training trials with provided parameters
    - Monitor training progress and resource utilization
    - Communicate status updates to mlTrainer
    - Handle errors and exceptions during execution
    - Maintain performance metrics and logging

COMPLIANCE REQUIREMENTS:
    - Execute only verified, compliant training configurations
    - Report all execution status to mlTrainer
    - Maintain strict data provenance tracking
    - Log all operations for audit trail""",
        "temperature": 0.1,
        "max_tokens": 2048,
        "capabilities": [AICapability.CODING, AICapability.ANALYSIS],
    },
    "analysis_specialist": {
        "model": "claude-3-5-sonnet",
        "role": "Analysis Specialist",
        "system_prompt": """You are the Analysis Specialist for mlTrainer.

CORE RESPONSIBILITIES:
    - Analyze training results and model performance
    - Generate comprehensive analysis reports
    - Identify optimization opportunities
    - Provide statistical insights and recommendations
    - Create visualization recommendations

COMPLIANCE REQUIREMENTS:
    - Analyze only verified, compliant data
    - Provide institutional-grade analysis quality
    - Document all analysis methodologies
    - Ensure reproducible results""",
        "temperature": 0.2,
        "max_tokens": 4096,
        "capabilities": [AICapability.ANALYSIS, AICapability.MATH, AICapability.RESEARCH],
    },
}

# ================================
# AI OPERATIONAL SETTINGS - SINGLE SOURCE OF TRUTH
# ================================
AI_OPERATIONAL_CONFIG = {
    "default_model": "claude-3-5-sonnet",  # Best value for $50/month budget
    "fallback_model": "claude-3-haiku",  # Ultra-low cost backup
    "max_concurrent_requests": 5,
    "request_timeout_seconds": 30,
    "retry_on_rate_limit": True,
    "retry_on_timeout": True,
    "log_all_requests": True,
    "log_all_responses": True,
    "enable_streaming": True,
    "enable_function_calling": True,
    "cost_tracking_enabled": True,
    "usage_monitoring_enabled": True,
}

# ================================
# AI COMPLIANCE SETTINGS - SINGLE SOURCE OF TRUTH
# ================================
AI_COMPLIANCE_CONFIG = {
    "require_institutional_grade": True,
    "require_compliance_verification": True,
    "audit_all_ai_interactions": True,
    "log_compliance_violations": True,
    "reject_non_compliant_models": True,
    "enforce_cost_limits": True,
    "max_cost_per_request": 1.0,
    "max_daily_cost": 100.0,
}


# ================================
# UTILITY FUNCTIONS - SINGLE SOURCE OF TRUTH ACCESS
# ================================
def get_ai_model(model_name: str) -> Optional[AIModel]:
    """Get AI model configuration - SINGLE SOURCE OF TRUTH"""
    return AI_MODELS.get(model_name)


def get_ai_provider(provider_name: str) -> Optional[AIProviderConfig]:
    """Get AI provider configuration - SINGLE SOURCE OF TRUTH"""
    return AI_PROVIDERS.get(provider_name)


def get_role_config(role_name: str) -> Optional[Dict[str, Any]]:
    """Get AI role configuration - SINGLE SOURCE OF TRUTH"""
    return AI_ROLE_CONFIGS.get(role_name)


def get_all_models() -> List[str]:
    """Get all available AI models - SINGLE SOURCE OF TRUTH"""
    return list(AI_MODELS.keys())


def get_all_providers() -> List[str]:
    """Get all available AI providers - SINGLE SOURCE OF TRUTH"""
    return list(AI_PROVIDERS.keys())


def get_all_roles() -> List[str]:
    """Get all available AI roles - SINGLE SOURCE OF TRUTH"""
    return list(AI_ROLE_CONFIGS.keys())


def get_models_by_provider(provider: str) -> List[str]:
    """Get models for specific provider - SINGLE SOURCE OF TRUTH"""
    return [name for name, model in list(AI_MODELS.items()) if model.provider == provider]


def get_institutional_models() -> List[str]:
    """Get institutional-grade models only - SINGLE SOURCE OF TRUTH"""
    return [name for name, model in list(AI_MODELS.items()) if model.institutional_grade]


def get_model_cost(model_name: str) -> float:
    """Get model cost per 1K tokens - SINGLE SOURCE OF TRUTH"""
    model = get_ai_model(model_name)
    return model.cost_per_1k_tokens if model else 0.0


def is_model_compliant(model_name: str) -> bool:
    """Check if model is compliance verified - SINGLE SOURCE OF TRUTH"""
    model = get_ai_model(model_name)
    return model.compliance_verified if model else False


def get_default_model() -> str:
    """Get default AI model - SINGLE SOURCE OF TRUTH"""
    return AI_OPERATIONAL_CONFIG["default_model"]


def get_fallback_model() -> str:
    """Get fallback AI model - SINGLE SOURCE OF TRUTH"""
    return AI_OPERATIONAL_CONFIG["fallback_model"]


def get_ai_operational_config() -> Dict[str, Any]:
    """Get AI operational configuration - SINGLE SOURCE OF TRUTH"""
    return AI_OPERATIONAL_CONFIG.copy()


def get_ai_compliance_config() -> Dict[str, Any]:
    """Get AI compliance configuration - SINGLE SOURCE OF TRUTH"""
    return AI_COMPLIANCE_CONFIG.copy()


def get_api_key(model_name: str) -> Optional[str]:
    """Get API key for model - SINGLE SOURCE OF TRUTH"""
    model = get_ai_model(model_name)
    if model:
        return os.getenv(model.api_key_env)
    return None


def validate_model_config(model_name: str) -> bool:
    """Validate model configuration - SINGLE SOURCE OF TRUTH"""
    model = get_ai_model(model_name)
    if not model:
        return False

    # Check if API key is available
    api_key = get_api_key(model_name)
    if not api_key:
        return False

    # Check compliance requirements
    if AI_COMPLIANCE_CONFIG["require_institutional_grade"] and not model.institutional_grade:
        return False

    if AI_COMPLIANCE_CONFIG["require_compliance_verification"] and not model.compliance_verified:
        return False

    return True


# ================================
# EXPORT CONFIGURATION - SINGLE SOURCE OF TRUTH
# ================================
__all__ = [
    # Core Classes
    "AIModel",
    "AIProviderConfig",
    "AIProvider",
    "AIModelType",
    "AICapability",
    # Configuration Dictionaries
    "AI_MODELS",
    "AI_PROVIDERS",
    "AI_ROLE_CONFIGS",
    "AI_OPERATIONAL_CONFIG",
    "AI_COMPLIANCE_CONFIG",
    # Utility Functions
    "get_ai_model",
    "get_ai_provider",
    "get_role_config",
    "get_all_models",
    "get_all_providers",
    "get_all_roles",
    "get_models_by_provider",
    "get_institutional_models",
    "get_model_cost",
    "is_model_compliant",
    "get_default_model",
    "get_fallback_model",
    "get_ai_operational_config",
    "get_ai_compliance_config",
    "get_api_key",
    "validate_model_config",
]
