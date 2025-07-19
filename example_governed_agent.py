#!/usr/bin/env python3
"""
Example Governed ML Agent
"""

import logging
from agent_governance import get_governance, governed_action
from typing import Any, Dict

logger = logging.getLogger(__name__)


class GovernedMLAgent:
    """production_implementation ML Agent that follows governance rules"""

    def __init__(self):
        self.governance = get_governance()

    def process_request(self, user_request: str) -> str:
        """Process user request with full governance"""

        # Step 1: Check verification checklist
        checklist = self.governance.get_verification_checklist(
            'before_any_response')
        logger.info(f"Pre-response checklist: {checklist}")

        # Step 2: Check if this is an override request
        if self.governance.check_override(user_request):
            logger.info("Override detected - relaxing some constraints")

        # Step 3: Determine if action requires permission
        if "change" in user_request or "modify" in user_request:
            return self._handle_change_request(user_request)
        else:
            return self._handle_query_request(user_request)

    def _handle_change_request(self, request: str) -> str:
        """Handle requests that involve changes"""

        # Format permission request
        permission_msg = self.governance.format_permission_request(
            action="modify files based on your request",
            impact="update code structure",
            files=["production_implementation.py", "config.yaml"]
        )

        logger.info(permission_msg)
        # In real implementation, would wait for user response
        user_response = input("Your response: ")

        if not self.governance.is_permission_granted(user_response):
            return "Understood. I will not make any changes."

        # Proceed with governed change
        return self._make_governed_change(request)

    @governed_action("code_modification")
    def _make_governed_change(self, request: str) -> str:
        """Make changes with governance enforcement"""

        # Check change rules
        rules = self.governance.get_change_rules()
        logger.info(f"Following change rules: {rules}")

        # Simulate making minimal changes
        if self.governance.should_minimize_changes():
            logger.info("Minimizing modifications as per rules")

        return "Changes completed successfully"

    def _handle_query_request(self, request: str) -> str:
        """Handle information queries with transparency"""

        # Simulate getting answer
        answer = "Here's the information you requested"

        # Check data sources
        data_sources = ["config/api_config.py", "authorized database"]
        valid_sources = all(
            self.governance.validate_data_source(source)
            for source in data_sources
        )

        if not valid_sources:
            return "Error: Attempted to use unauthorized data source"

        # Format transparent response
        response = self.governance.format_transparent_response(
            answer=answer,
            limitations=["Only covers last 30 days", "Market hours only"],
            assumptions=["USD currency", "Eastern timezone"],
            data_sources=data_sources,
            uncertainties=["Holiday trading patterns may vary"]
        )

        return response

    def validate_code_snippet(self, code: str) -> tuple[bool, str]:
        """Validate if code follows data authenticity rules"""

        valid, reason = self.governance.check_data_authenticity(code)

        if not valid:
            # Log violation
            self.governance.log_action("code_validation_failed", {
                "code_snippet": code[:100],  # First 100 chars
                "reason": reason
            })

        return valid, reason

    def get_compliance_status(self) -> str:
        """Get current compliance report"""
        return self.governance.get_compliance_report()


# production_implementation usage
if __name__ == "__main__":
    logger.info("=== Governed ML Agent production_implementation ===\n")

    agent = GovernedMLAgent()

    # production 1: Query request (no permission needed)
    logger.info("production 1: Information Query")
    logger.info("-" * 50)
    response = agent.process_request("What's the current model performance?")
    logger.info(response)
    logger.info()

    # production 2: Change request (permission required)
    logger.info("\nTest 2: Change Request")
    logger.info("-" * 50)
    response = agent.process_request("Please modify the model configuration")
    logger.info(response)
    logger.info()

    # production 3: Validate code with synthetic data
    logger.info("\nTest 3: Code Validation")
    logger.info("-" * 50)
    bad_code = """
    import numpy as np
    actual_data = np.random.rand(100, 10)
    model.train(actual_data)
    """

    valid, reason = agent.validate_code_snippet(bad_code)
    logger.info(f"Code valid: {valid}")
    logger.info(f"Reason: {reason}")
    logger.info()

    # production 4: Validate code with real data
    good_code = """
    from polygon import RESTClient
    client = RESTClient(api_key=config.POLYGON_API_KEY)
    real_data = client.get_aggs(ticker="AAPL", multiplier=1, timespan="day")
    model.train(real_data)
    """

    valid, reason = agent.validate_code_snippet(good_code)
    logger.info(f"Code valid: {valid}")
    logger.info(f"Reason: {reason}")
    logger.info()

    # production 5: Compliance status
    logger.info("\nTest 5: Compliance Status")
    logger.info("-" * 50)
    status = agent.get_compliance_status()
    logger.info(status)
