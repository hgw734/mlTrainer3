"""
Agent Governance Module
Loads and enforces behavioral rules from agent_rules.yaml
"""

import yaml
import os
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AgentGovernance:
    """
    Enforces agent behavioral rules and guardrails
    Single source of truth for agent behavior constraints
    """

    def __init__(self, rules_file: str = "agent_rules.yaml"):
        """Initialize with rules from yaml file"""
        self.rules_file = rules_file
        self.rules = self._load_rules()
        self.audit_log = []

    def _load_rules(self) -> Dict[str, Any]:
        """Load rules from yaml file"""
        try:
            with open(self.rules_file, "r") as f:
                rules = yaml.safe_load(f)
                logger.info(f"Loaded agent rules v{rules.get('version', 'unknown')}")
                return rules
        except Exception as e:
            logger.error(f"Failed to load agent rules: {e}")
            # Return strict defaults if file not found
            return self._get_default_strict_rules()

    def _get_default_strict_rules(self) -> Dict[str, Any]:
        """Fallback strict rules if file not found"""
        return {
            "version": "2.0.0",
            "enforcement_level": "strict",
            "permission_protocol": {"require_explicit_permission": True, "ask_before_any_change": True},
            "data_authenticity": {"use_only_real_data": True, "no_synthetic_data": True},
            "transparency": {"no_omissions": True},
        }

    def check_permission_required(self, action: str) -> bool:
        """Check if an action requires permission"""
        protocol = self.rules.get("permission_protocol", {})
        return protocol.get("require_explicit_permission", True)

    def format_permission_request(self, action: str, impact: str, files: List[str]) -> str:
        """Format a permission request according to rules"""
        protocol = self.rules.get("permission_protocol", {})
        template = protocol.get("permission_template", "May I {action}? This will {impact} files: {files}")
        return template.format(action=action, impact=impact, files=", ".join(files) if files else "none")

    def is_permission_granted(self, response: str) -> bool:
        """Check if user response grants permission"""
        protocol = self.rules.get("permission_protocol", {})
        confirmations = protocol.get("acceptable_confirmations", ["yes"])
        return response.lower().strip() in [c.lower() for c in confirmations]

    def check_data_authenticity(self, code_snippet: str) -> Tuple[bool, str]:
        """
        Check if code uses only authentic data sources
        Returns (is_valid, reason)
        """
        data_rules = self.rules.get("data_authenticity", {})
        prohibited = data_rules.get("prohibited_patterns", [])
        for pattern in prohibited:
            if pattern in code_snippet:
                return False, f"Prohibited pattern found: {pattern}"
        return True, "No synthetic data patterns detected"

    def validate_data_source(self, source: str) -> bool:
        """Check if a data source is allowed"""
        data_rules = self.rules.get("data_authenticity", {})
        if not data_rules.get("required_data_sources", {}).get("whitelist_only"):
            return True
        allowed = data_rules.get("required_data_sources", {}).get("allowed_sources", [])
        return any(source in allowed_source for allowed_source in allowed)

    def format_transparent_response(
        self,
        answer: str,
        limitations: Optional[List[str]] = None,
        assumptions: Optional[List[str]] = None,
        uncertainties: Optional[List[str]] = None,
        data_sources: Optional[List[str]] = None,
        issues: Optional[List[str]] = None,
    ) -> str:
        """Format response with full transparency"""
        transparency = self.rules.get("transparency", {})
        if not transparency.get("no_omissions", True):
            return answer
        parts = [f"Answer: {answer}"]
        if limitations is not None:
            parts.append(f"\nLimitations: {', '.join(limitations)}")
        if assumptions is not None:
            parts.append(f"Assumptions: {', '.join(assumptions)}")
        if uncertainties is not None:
            parts.append(f"Uncertainties: {', '.join(uncertainties)}")
        if data_sources is not None:
            parts.append(f"Data Sources: {', '.join(data_sources)}")
        if issues is not None:
            parts.append(f"Potential Issues: {', '.join(issues)}")
        return "\n".join(parts)

    def check_scope_drift(self, requested: str, planned: str) -> Tuple[bool, str]:
        """
        Check if planned action drifts from request
        Returns (is_drift, reason)
        """
        anti_drift = self.rules.get("anti_drift", {})
        if not anti_drift.get("no_feature_creep", True):
            return False, "Drift protection disabled"
        # Simple check - in real implementation would be more sophisticated
        if len(planned) > len(requested) * 2:
            return True, "Planned action significantly exceeds request scope"
        return False, "No scope drift detected"

    def get_change_rules(self) -> List[str]:
        """Get rules for making changes"""
        discipline = self.rules.get("change_discipline", {})
        return discipline.get("change_rules", [])

    def should_minimize_changes(self) -> bool:
        """Check if changes should be minimized"""
        discipline = self.rules.get("change_discipline", {})
        return discipline.get("minimize_modifications", True)

    def log_action(self, action: str, details: Dict[str, Any]):
        """Log an action for audit trail"""
        if not self.rules.get("operational_boundaries", {}).get("audit_trail", {}).get("log_all_actions"):
            return
        entry = {"timestamp": datetime.now().isoformat(), "action": action, "details": details}
        self.audit_log.append(entry)
        logger.info(f"Action logged: {action}")

    def get_verification_checklist(self, phase: str = "before_any_response") -> List[str]:
        """Get verification checklist for a phase"""
        verification = self.rules.get("verification", {})
        return verification.get(phase, [])

    def check_override(self, user_input: str) -> bool:
        """Check if user is overriding rules"""
        overrides = self.rules.get("overrides", {})
        if not overrides.get("user_can_override", True):
            return False
        override_phrases = overrides.get("override_phrases", [])
        return any(phrase in user_input.lower() for phrase in override_phrases)

    def get_enforcement_level(self) -> str:
        """Get current enforcement level"""
        return self.rules.get("enforcement", {}).get("mode", "strict")

    def validate_action(self, action_type: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Comprehensive validation of an action
        Returns (is_allowed, reason)
        """
        # Check permission requirement
        if self.check_permission_required(action_type):
            if not context.get("permission_granted", False):
                return False, "Permission required but not granted"
        # Check data authenticity if code involved
        if "code" in context:
            valid, reason = self.check_data_authenticity(context["code"])
            if not valid:
                return False, reason
        # Check scope drift
        if "requested" in context and "planned" in context:
            drift, reason = self.check_scope_drift(context["requested"], context["planned"])
            if drift:
                return False, reason
        # Log the validation attempt
        self.log_action("validation", {"action_type": action_type, "result": "allowed", "context": context})
        return True, "Action validated successfully"

    def get_compliance_report(self) -> str:
        """Generate compliance report"""
        total_actions = len(self.audit_log)
        report = f"""
Compliance Report
================
Rules Version: {self.rules.get('version', 'unknown')}
Enforcement Level: {self.get_enforcement_level()}
Total Actions: {total_actions}
Session Start: {self.audit_log[0]['timestamp'] if self.audit_log else 'N/A'}

Recent Actions:
"""
        for entry in self.audit_log[-5:]:  # Last 5 actions
            report += f"- {entry['timestamp']}: {entry['action']}\n"
        return report


# Singleton instance
_governance_instance = None


def get_governance() -> AgentGovernance:
    """Get singleton governance instance"""
    global _governance_instance
    if _governance_instance is None:
        _governance_instance = AgentGovernance()
    return _governance_instance


# Decorator for enforcing governance
def governed_action(action_type: str):
    """Decorator to enforce governance on methods"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            governance = get_governance()

            # Build context from function arguments
            context = {"function": func.__name__, "args": args, "kwargs": kwargs}

            # Validate action
            allowed, reason = governance.validate_action(action_type, context)
            if not allowed:
                raise PermissionError(f"Action blocked by governance: {reason}")

            # Execute function
            result = func(*args, **kwargs)

            # Log completion
            governance.log_action(f"{action_type}_completed", {"function": func.__name__, "success": True})

            return result

        return wrapper

    return decorator
