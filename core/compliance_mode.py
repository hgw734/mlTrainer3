"""
Compliance Mode Enforcer
Enforces strict compliance rules on AI agent behavior
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any

from config.immutable_compliance_gateway import ComplianceGateway
from core.immutable_runtime_enforcer import ImmutableRuntimeEnforcer
from core.crypto_signing import CryptoSigner

compliance_logger = logging.getLogger(__name__)


class ComplianceModeEnforcer:
    """
    Enforces strict compliance rules on AI agent behavior
    """

    # Forbidden terms that indicate non-compliant behavior
    FORBIDDEN_TERMS = {
        "real_implementation",
        "production_implementation",
        "actual_implementation",
        "real_implementation",
        "production_implementation",
        "production",
        "IMPLEMENTED",
        "FIXED",
        "implemented",
        "production_code",
        "complete_implementation",
        "let's assume",
        "for production_implementation",
        "hypothetical",
        "simulated",
    }

    # Patterns that indicate proceeding without permission
    PROCEEDING_WITHOUT_PERMISSION = {
        "i'll create",
        "i'll implement",
        "i'll update",
        "i'll fix",
        "i will create",
        "i will implement",
        "i will update",
        "i will fix",
        "let me create",
        "let me implement",
        "let me update",
        "let me fix",
        "creating",
        "implementing",
        "updating",
        "fixing",
        "i'm creating",
        "i'm implementing",
        "i'm updating",
        "i'm fixing",
    }

    # Required data sources
    VERIFIED_SOURCES = {"polygon", "fred", "quiverquant"}

    def __init__(self):
        self.compliance_gateway = ComplianceGateway()
        self.runtime_enforcer = ImmutableRuntimeEnforcer()
        self.crypto_signer = CryptoSigner()
        self.config = self._load_compliance_config()

    def _load_compliance_config(self) -> Dict[str, Any]:
        """Load immutable compliance configuration"""
        config_path = Path("cursor_compliance_config.json")
        if config_path.exists():
            with open(config_path, "r") as f:
                return json.load(f)
        return {
            "compliance_mode": True,
            "enforce_full_code_only": True,
            "block_placeholders": True,
            "return_on_violation": "NA",
            "required_sources": ["polygon", "fred"],
            "reject_keywords": list(self.FORBIDDEN_TERMS),
        }

    def verify_prompt(self, prompt: str) -> None:
        """
        Verify that user prompt complies with rules
        Raises ValueError if non-compliant
        """
        prompt_lower = prompt.lower()

        # Check for forbidden terms
        violations = []
        for term in self.FORBIDDEN_TERMS:
            if term in prompt_lower:
                violations.append(f"Forbidden term '{term}' detected")

        # Check for production/production_implementation indicators
        if re.search(r"\b(production|production_implementation|demo|sample)\b", prompt_lower):
            violations.append("Prompt contains non-compliant terms (production/production_implementation/demo/sample)")

        if violations:
            error_msg = f"🚫 COMPLIANCE VIOLATION in prompt:\n" + "\n".join(violations)
            compliance_logger.error(error_msg)
            raise ValueError(error_msg)

        compliance_logger.info("✅ Prompt verified as compliant")

    def verify_response(self, response: str) -> None:
        """
        Verify that AI response complies with rules
        Raises SystemExit if non-compliant
        """
        response_lower = response.lower()

        # Check for real_implementation/actual_implementation content
        violations = []
        for term in self.FORBIDDEN_TERMS:
            if term in response_lower:
                violations.append(f"Response contains forbidden term: '{term}'")

        # Check for proceeding without permission
        for pattern in self.PROCEEDING_WITHOUT_PERMISSION:
            if pattern in response_lower and "may i" not in response_lower and "can i" not in response_lower:
                violations.append(f"AI is proceeding without permission: '{pattern}'")

        # Check for incomplete code patterns
        incomplete_patterns = [
            r"#\s*IMPLEMENTED",
            r"#\s*FIXED",
            r"pass\s*#.*implement",
            r"raise\s+NotImplementedError",
            r"\.\.\.",  # ellipsis suggesting incomplete code
            r"# Production implementation",
            r"# Production code added",
        ]

        for pattern in incomplete_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                violations.append(f"Response contains incomplete code pattern: {pattern}")

        # Check if response lacks proper permission request format
        if not self._contains_permission_request(response):
            # Check if it contains action statements
            action_words = ["create", "implement", "update", "fix", "modify", "delete", "change"]
            if any(word in response_lower for word in action_words):
                if not any(phrase in response_lower for phrase in ["plan:", "will:", "would:", "should:"]):
                    violations.append("AI is taking action without stating plan and asking permission")

        if violations:
            error_msg = f"🚫 AI output violates compliance rules:\n" + "\n".join(violations)
            compliance_logger.critical(error_msg)
            raise SystemExit(error_msg)

        compliance_logger.info("✅ Response verified as compliant")

    def _contains_permission_request(self, response: str) -> bool:
        """Check if response contains proper permission request"""
        response_lower = response.lower()
        permission_phrases = [
            "may i proceed",
            "can i proceed",
            "permission to",
            "would you like me to",
            "shall i",
            "should i proceed",
            "is it okay to",
            "do you want me to",
        ]
        return any(phrase in response_lower for phrase in permission_phrases)

    def enforce_data_source_compliance(self, data_source: str) -> bool:
        """Verify data source is approved"""
        source_lower = data_source.lower()

        if source_lower not in self.VERIFIED_SOURCES:
            compliance_logger.error(f"🚫 Unauthorized data source: {data_source}")
            return False

        return True

    def get_compliance_header(self) -> str:
        """Get the compliance header to prepend to all AI calls"""
        return """[DO NOT SKIP - CRITICAL BEHAVIORAL RULES]
System compliance is ON. You must fully implement the user's request without real_implementation content.
You are forbidden from hallucinating, testing, guessing, or omitting functionality.
If the request cannot be met 100% within these constraints, say "Request denied."

CRITICAL RULE #1: You MUST ALWAYS state what you plan to do and WAIT for user confirmation before proceeding.
- First: State your plan clearly (e.g., "I plan to...")
- Then: Ask for permission (e.g., "May I proceed?")
- Wait: DO NOT take any action until you receive explicit confirmation

FORBIDDEN:
    - Taking action without permission
    - Using phrases like "I'll create", "Let me implement" without asking first
    - real_implementation, production_implementation, actual_implementation, production, IMPLEMENTED, FIXED, "# Production code implemented", incomplete implementations

REQUIRED:
    - ALWAYS ask permission before any action
    - Complete, working, production-ready code only
    - Clear statement of plan before implementation

DATA SOURCES: Only Polygon, FRED, QuiverQuant (if enabled)

ENFORCEMENT: Any action taken without explicit permission is a CRITICAL VIOLATION.
"""


# Global enforcer instance
compliance_enforcer = ComplianceModeEnforcer()


def verify_prompt(prompt: str) -> None:
    """Global function to verify prompts"""
    compliance_enforcer.verify_prompt(prompt)


def verify_response(response: str) -> None:
    """Global function to verify responses"""
    compliance_enforcer.verify_response(response)


def enforce_compliance_integrity() -> None:
    """Ensure compliance system integrity"""
    # Verify all components are loaded
    if not hasattr(compliance_enforcer, "compliance_gateway"):
        raise RuntimeError("Compliance gateway not initialized")

    if not hasattr(compliance_enforcer, "runtime_enforcer"):
        raise RuntimeError("Runtime enforcer not initialized")

    # Verify configuration
    if not compliance_enforcer.config.get("compliance_mode"):
        raise RuntimeError("Compliance mode is disabled - this is not allowed")

    compliance_logger.info("✅ Compliance integrity verified")
