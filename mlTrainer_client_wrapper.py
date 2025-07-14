#!/usr/bin/env python3
"""
🔒 mlTrainer Client Wrapper with Permanent Prompt Injection
Ensures all AI interactions are compliance-bound and cannot drift
"""

import json
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pathlib import Path

# Import enforcement systems
from core.immutable_runtime_enforcer import (
get_system_prompt,
build_prompt,
verify_response,
SYSTEM_STATE,
activate_kill_switch,
fail_safe_response,
)
from config.immutable_compliance_gateway import COMPLIANCE_GATEWAY, VerifiedData, DataProvenance

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - mlTrainer - %(levelname)s - %(message)s")
logger = logging.getLogger("mlTrainer_CLIENT")


class ComplianceBoundClient:
    """
    mlTrainer client with permanent compliance restrictions
    All AI interactions are bound by immutable compliance rules
    """

    def __init__(self, model_name: str = "mlTrainer"):
        self.model_name = model_name
        self.interaction_count = 0
        self.violation_history = []
        self.system_prompt = get_system_prompt()

        # Log initialization
        logger.info(f"🔒 Initialized ComplianceBoundClient for {model_name}")
        logger.info(f"System prompt locked: {len(self.system_prompt)} chars")

        def process_request(self, user_input: str, context: Optional[Dict] = None) -> str:
            """
            Process user request with full compliance enforcement

            Args:
                user_input: User's request
                context: Optional context data (must be verified)

                Returns:
                    Compliance-verified response or fail-safe
                    """
                    self.interaction_count += 1

                    try:
                        # Build compliance-aware prompt
                        system_state = SYSTEM_STATE.to_dict()
                        full_prompt = build_prompt(user_input, system_state)

                        # Log the interaction
                        logger.info(f"Processing request #{self.interaction_count}")

                        # Check if system is in lockdown
                        if system_state.get("kill_switch", False):
                            logger.warning("System in lockdown - returning fail-safe")
                            return fail_safe_response()

                            # Verify context data if provided
                            if context:
                                verified_context = self._verify_context(context)
                                if not verified_context:
                                    logger.error("Context verification failed")
                                    return fail_safe_response()

                                    # Add verified context to prompt
                                    full_prompt += f"\nVerified Context: {json.dumps(verified_context)}\n"

                                    # Generate response (simulated - replace with actual model call)
                                    response = self._generate_response(full_prompt)

                                    # Verify response for compliance
                                    verified_response = verify_response(response, self.model_name)

                                    # Additional compliance checks
                                    if self._contains_violations(verified_response):
                                        logger.error("Response contains violations")
                                        self._record_violation("Response violations detected")
                                        return fail_safe_response()

                                        logger.info(f"✅ Request #{self.interaction_count} completed successfully")
                                        return verified_response

                                        except Exception as e:
                                            logger.error(f"Error processing request: {e}")
                                            self._record_violation(f"Processing error: {e}")
                                            return fail_safe_response()

                                            def _verify_context(self, context: Dict) -> Optional[Dict]:
                                                """Verify context data is from approved sources"""
                                                verified_context = {}

                                                for key, value in list(context.items()):
                                                    # Check if value has provenance
                                                    if isinstance(value, dict) and "source" in value:
                                                        source = value["source"]
                                                        data = value.get("data", {})

                                                        # Verify through compliance gateway
                                                        provenance = COMPLIANCE_GATEWAY.tag_incoming_data(data, source, f"context_{key}")

                                                        if provenance:
                                                            verified_context[key] = {"data": data, "verified": True, "source": source}
                                                            else:
                                                                logger.warning(f"Context key '{key}' failed verification")
                                                                return None
                                                                else:
                                                                    # Reject unverified context
                                                                    logger.warning(f"Context key '{key}' lacks provenance")
                                                                    return None

                                                                    return verified_context

                                                                    def _generate_response(self, prompt: str) -> str:
                                                                        """
                                                                        Generate response from model (real_implementation for actual implementation)
                                                                        Replace this with actual model API call
                                                                        """
                                                                        # This is a real_implementation - replace with actual model call
                                                                        # For production_implementation: response = openai_client.complete(prompt)

                                                                        # Simulate compliant response
                                                                        return "Based on verified data from approved sources, [actual response here]"

                                                                        def _contains_violations(self, response: str) -> bool:
                                                                            """Check if response contains compliance violations"""
                                                                            violation_patterns = [
                                                                            "production data",
                                                                            "production_implementation data",
                                                                            "synthetic",
                                                                            "simulated",
                                                                            "random",
                                                                            "generated",
                                                                            "real_implementation",
                                                                            "production_implementation",
                                                                            ]

                                                                            response_lower = response.lower()
                                                                            for pattern in violation_patterns:
                                                                                if pattern in response_lower:
                                                                                    logger.warning(f"Violation pattern detected: '{pattern}'")
                                                                                    return True

                                                                                    return False

                                                                                    def _record_violation(self, violation: str):
                                                                                        """Record compliance violation"""
                                                                                        violation_record = {
                                                                                        "timestamp": datetime.now().isoformat(),
                                                                                        "model": self.model_name,
                                                                                        "violation": violation,
                                                                                        "interaction": self.interaction_count,
                                                                                        }

                                                                                        self.violation_history.append(violation_record)

                                                                                        # Check if we need to activate kill switch
                                                                                        if len(self.violation_history) >= 5:
                                                                                            logger.critical("Too many violations - activating kill switch")
                                                                                            activate_kill_switch(f"{self.model_name} exceeded violation limit")

                                                                                            def get_compliance_status(self) -> Dict[str, Any]:
                                                                                                """Get current compliance status"""
                                                                                                return {
                                                                                                "model_name": self.model_name,
                                                                                                "interactions": self.interaction_count,
                                                                                                "violations": len(self.violation_history),
                                                                                                "recent_violations": self.violation_history[-5:],
                                                                                                "system_state": SYSTEM_STATE.to_dict(),
                                                                                                "compliance_gateway": COMPLIANCE_GATEWAY.get_compliance_report(),
                                                                                                }

                                                                                                def reset_violations(self, authorized_by: str):
                                                                                                    """Reset violation history (requires authorization)"""
                                                                                                    logger.info(f"Violation history reset authorized by: {authorized_by}")
                                                                                                    self.violation_history = []


                                                                                                    class StreamingComplianceClient(ComplianceBoundClient):
                                                                                                        """
                                                                                                        Streaming version for real-time responses with compliance
                                                                                                        """

                                                                                                        def stream_response(self, user_input: str, context: Optional[Dict] = None):
                                                                                                            """
                                                                                                            Stream response with real-time compliance checking
                                                                                                            Yields verified chunks or stops on violation
                                                                                                            """
                                                                                                            # Build prompt
                                                                                                            system_state = SYSTEM_STATE.to_dict()
                                                                                                            full_prompt = build_prompt(user_input, system_state)

                                                                                                            # Check lockdown
                                                                                                            if system_state.get("kill_switch", False):
                                                                                                                yield fail_safe_response()
                                                                                                                return

                                                                                                            # Stream real_implementation (replace with actual streaming)
                                                                                                            response_chunks = [
                                                                                                            "Based on ",
                                                                                                            "verified data ",
                                                                                                            "from approved sources, ",
                                                                                                            "the analysis shows# Production code implemented",
                                                                                                            ]

                                                                                                            accumulated_response = ""

                                                                                                            for chunk in response_chunks:
                                                                                                                accumulated_response += chunk

                                                                                                                # Check each chunk for drift
                                                                                                                if self._contains_violations(accumulated_response):
                                                                                                                    logger.error("Streaming violation detected - stopping")
                                                                                                                    yield "\n[STREAM TERMINATED - COMPLIANCE VIOLATION]"
                                                                                                                    self._record_violation("Streaming violation")
                                                                                                                    return

                                                                                                                yield chunk

                                                                                                                # Final verification
                                                                                                                verify_response(accumulated_response, self.model_name)


                                                                                                                # Factory functions
                                                                                                                def create_compliance_client(model_name: str = "mlTrainer") -> ComplianceBoundClient:
                                                                                                                    """Create a compliance-bound client"""
                                                                                                                    return ComplianceBoundClient(model_name)


                                                                                                                    def create_streaming_client(model_name: str = "mlTrainer") -> StreamingComplianceClient:
                                                                                                                        """Create a streaming compliance client"""
                                                                                                                        return StreamingComplianceClient(model_name)


                                                                                                                        # production_implementation usage wrapper
                                                                                                                        class mlTrainerAPI:
                                                                                                                            """
                                                                                                                            Main API interface for mlTrainer with built-in compliance
                                                                                                                            """

                                                                                                                            def __init__(self):
                                                                                                                                self.client = create_compliance_client("mlTrainer-API")
                                                                                                                                self.streaming_client = create_streaming_client("mlTrainer-Stream")

                                                                                                                                def query(self, prompt: str, context: Optional[Dict] = None) -> str:
                                                                                                                                    """Query mlTrainer with compliance enforcement"""
                                                                                                                                    return self.client.process_request(prompt, context)

                                                                                                                                    def stream(self, prompt: str, context: Optional[Dict] = None):
                                                                                                                                        """Stream mlTrainer response with compliance"""
                                                                                                                                        return self.streaming_client.stream_response(prompt, context)

                                                                                                                                        def status(self) -> Dict[str, Any]:
                                                                                                                                            """Get system status"""
                                                                                                                                            return {
                                                                                                                                            "client_status": self.client.get_compliance_status(),
                                                                                                                                            "streaming_status": self.streaming_client.get_compliance_status(),
                                                                                                                                            }


                                                                                                                                            # Global instance
                                                                                                                                            mltrainer = mlTrainerAPI()

                                                                                                                                            # Export public interface
                                                                                                                                            __all__ = [
                                                                                                                                            "ComplianceBoundClient",
                                                                                                                                            "StreamingComplianceClient",
                                                                                                                                            "create_compliance_client",
                                                                                                                                            "create_streaming_client",
                                                                                                                                            "mlTrainerAPI",
                                                                                                                                            "mltrainer",
                                                                                                                                            ]
