"""
Cursor Agent Wrapper - Intercepts all AI requests for compliance verification
"""

import json
import os
import sys
from typing import Dict, Any, Optional
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.compliance_mode import verify_prompt, verify_response, compliance_enforcer
from core.immutable_runtime_enforcer import (
get_system_prompt,
build_prompt,
detect_drift,
verify_response as verify_runtime_response,
SystemState,
save_system_state,
load_system_state,
)
from config.immutable_compliance_gateway import compliance_logger


class CursorAgentWrapper:
    """
    Wraps all AI agent interactions with compliance verification
    """

    def __init__(self):
        self.system_state = load_system_state()
        self.compliance_header = compliance_enforcer.get_compliance_header()

        def get_state(self) -> Dict[str, Any]:
            """Get current system state"""
            return self.system_state.to_dict()

            def build_compliant_prompt(self, user_input: str) -> str:
                """Build a fully compliant prompt with all safeguards"""
                # First verify the user input
                verify_prompt(user_input)

                # Build the full prompt with compliance header
                system_prompt = get_system_prompt()
                compliance_prompt = self.compliance_header

                full_prompt = f"{compliance_prompt}\n\n{system_prompt}\n\nUser Request: {user_input}"

                return full_prompt

                def verify_and_clean_response(self, response: str) -> str:
                    """Verify response compliance and clean if needed"""
                    # First check with compliance mode
                    verify_response(response)

                    # Then check for drift
                    if detect_drift(response):
                        compliance_logger.error("🚨 Drift detected in AI response")
                        raise SystemExit("AI response shows signs of drift/hallucination")

                        # Runtime verification
                        verified_response = verify_runtime_response(response, "cursor_agent")

                        return verified_response


                        # Global wrapper instance
                        cursor_wrapper = CursorAgentWrapper()


                        def guarded_completion(user_input: str) -> str:
                            """
                            Main entry point for guarded AI completions
                            All requests must go through this function
                            """
                            try:
                                # Build compliant prompt
                                full_prompt = cursor_wrapper.build_compliant_prompt(user_input)

                                # Call the AI (this would be replaced with actual API call)
                                response = call_ai_api(full_prompt)

                                # Verify and clean response
                                verified_response = cursor_wrapper.verify_and_clean_response(response)

                                compliance_logger.info("✅ AI interaction completed successfully")
                                return verified_response

                                except ValueError as e:
                                    # Prompt validation failed
                                    compliance_logger.error(f"Prompt validation failed: {e}")
                                    return "🚫 This cannot be completed under current compliance rules. Request denied."

                                    except SystemExit as e:
                                        # Response validation failed
                                        compliance_logger.error(f"Response validation failed: {e}")
                                        return "🚫 AI response violated compliance rules. Request denied."

                                        except Exception as e:
                                            # Unexpected error
                                            compliance_logger.error(f"Unexpected error in guarded completion: {e}")
                                            return "NA"


                                            def call_ai_api(prompt: str) -> str:
                                                """
                                                real_implementation for actual AI API call
                                                In production, this would call Claude/GPT/etc with the compliance-wrapped prompt
                                                """
                                                # This is where you'd integrate with the actual AI API
                                                # For now, we'll return a compliant response
                                                return "I understand your request. However, I cannot provide implementation without the actual AI API integration."


                                                # Export key functions
                                                __all__ = ["guarded_completion", "cursor_wrapper"]
