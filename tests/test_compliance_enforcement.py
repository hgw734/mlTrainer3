#!/usr/bin/env python3
"""
Unit Tests for Anti-Hallucination and Compliance Enforcement
Tests all layers of the immutable compliance system
"""

import pytest
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.immutable_runtime_enforcer import (
enforce_verification,
fail_safe_response,
detect_drift,
verify_response,
is_verified_source,
compliance_wrap,
FAIL_SAFE_RESPONSE,
SystemState,
activate_kill_switch,
)
from config.immutable_compliance_gateway import ComplianceGateway, DataSource, ComplianceStatus


class TestComplianceEnforcement:
    """Test compliance enforcement mechanisms"""

    def test_invalid_source_rejected(self):
        """Test that invalid sources are rejected"""
        data = {"price": 123.45}

        with pytest.raises(PermissionError) as exc_info:
            enforce_verification(data, "wikipedia")

            assert "Unverified source" in str(exc_info.value)

            def test_valid_source_accepted(self):
                """Test that valid sources are accepted"""
                data = {"price": 123.45, "symbol": "AAPL"}

                # Should not raise exception
                result = enforce_verification(data, "polygon")
                assert result == data

                def test_placeholder_data_rejected(self):
                    """Test that placeholder data is rejected"""
                    data = {"example": "placeholder price", "value": 100}

                    with pytest.raises(ValueError) as exc_info:
                        enforce_verification(data, "polygon")

                        assert "Synthetic or placeholder" in str(exc_info.value)

                        def test_synthetic_patterns_rejected(self):
                            """Test various synthetic data patterns are rejected"""
                            synthetic_patterns = [
                            {"test": "test data"},
                            {"demo": "demo value"},
                            {"fake": "fake price"},
                            {"mock": "mock response"},
                            {"synthetic": "synthetic data"},
                            ]

                            for data in synthetic_patterns:
                                with pytest.raises(ValueError):
                                    enforce_verification(data, "polygon")

                                    def test_fail_safe_response(self):
                                        """Test fail-safe response is correct"""
                                        assert fail_safe_response() == FAIL_SAFE_RESPONSE
                                        assert fail_safe_response() == "NA"

                                        def test_compliance_wrap_decorator(self):
                                            """Test compliance wrapper on functions"""

                                            @compliance_wrap
                                            def fetch_data(source="polygon"):
                                                return {"price": 150.0, "volume": 1000000}

                                                # Valid source should work
                                                result = fetch_data(source="polygon")
                                                assert result["price"] == 150.0

                                                # Invalid source should return fail-safe
                                                result = fetch_data(source="random_api")
                                                assert result == "NA"

                                                def test_drift_detection(self):
                                                    """Test drift detection in responses"""
                                                    drift_responses = [
                                                    "For example, the price could be 100",
                                                    "Let's assume the market goes up",
                                                    "Hypothetically speaking, returns are 10%",
                                                    "Here's a placeholder value: 123",
                                                    "Sample output: profit = 1000",
                                                    ]

                                                    for response in drift_responses:
                                                        assert detect_drift(response) == True

                                                        def test_no_drift_in_valid_response(self):
                                                            """Test valid responses don't trigger drift detection"""
                                                            valid_responses = [
                                                            "Based on Polygon data, the price is 150.23",
                                                            "FRED reports inflation at 2.1%",
                                                            "The verified data shows volume of 1.2M",
                                                            ]

                                                            for response in valid_responses:
                                                                assert detect_drift(response) == False

                                                                def test_verify_response_with_drift(self):
                                                                    """Test response verification catches drift"""
                                                                    drifted_response = "For example, returns could be 20%"

                                                                    # Mock the system state to avoid kill switch in test
                                                                    with patch("core.immutable_runtime_enforcer.SYSTEM_STATE") as mock_state:
                                                                        mock_state.enforcement_level = "normal"
                                                                        mock_state.drift_count = 0

                                                                        result = verify_response(drifted_response)
                                                                        assert result == "NA"
                                                                        assert mock_state.drift_count == 1


                                                                        class TestComplianceGateway:
                                                                            """Test the compliance gateway"""

                                                                            def test_gateway_initialization(self):
                                                                                """Test gateway initializes correctly"""
                                                                                gateway = ComplianceGateway()
                                                                                assert len(gateway.APPROVED_SOURCES) > 0
                                                                                assert gateway.MAX_DATA_AGE_SECONDS > 0

                                                                                def test_data_source_verification(self):
                                                                                    """Test data source verification"""
                                                                                    gateway = ComplianceGateway()

                                                                                    # Valid sources
                                                                                    assert gateway.verify_data_source("polygon", "/v2/aggs") != DataSource.INVALID
                                                                                    assert gateway.verify_data_source("fred", "/series") != DataSource.INVALID

                                                                                    # Invalid source
                                                                                    assert gateway.verify_data_source("random_api", "/data") == DataSource.INVALID

                                                                                    def test_prohibited_generators(self):
                                                                                        """Test that data generators are detected"""
                                                                                        gateway = ComplianceGateway()

                                                                                        # These should all be invalid
                                                                                        assert gateway.verify_data_source("np.random", "/") == DataSource.INVALID
                                                                                        assert gateway.verify_data_source("faker", "/") == DataSource.INVALID
                                                                                        assert gateway.verify_data_source("synthetic_generator", "/") == DataSource.INVALID

                                                                                        def test_tag_incoming_data(self):
                                                                                            """Test data tagging with provenance"""
                                                                                            gateway = ComplianceGateway()

                                                                                            data = {"price": 150.0, "volume": 1000000}
                                                                                            provenance = gateway.tag_incoming_data(data, "polygon", "/v2/aggs/ticker/AAPL")

                                                                                            assert provenance is not None
                                                                                            assert provenance.source == DataSource.POLYGON
                                                                                            assert provenance.compliance_status == ComplianceStatus.VERIFIED

                                                                                            def test_data_freshness_check(self):
                                                                                                """Test data freshness verification"""
                                                                                                gateway = ComplianceGateway()

                                                                                                # Tag fresh data
                                                                                                data = {"value": 100}
                                                                                                provenance = gateway.tag_incoming_data(data, "fred", "/series/GDP")

                                                                                                # Fresh data should pass
                                                                                                assert gateway.verify_data_freshness(provenance) == True

                                                                                                def test_compliance_report(self):
                                                                                                    """Test compliance report generation"""
                                                                                                    gateway = ComplianceGateway()

                                                                                                    report = gateway.get_compliance_report()

                                                                                                    assert "gateway_status" in report
                                                                                                    assert report["gateway_status"] == "ACTIVE"
                                                                                                    assert "total_violations" in report
                                                                                                    assert "approved_sources" in report


                                                                                                    class TestSystemState:
                                                                                                        """Test system state management"""

                                                                                                        def test_system_state_immutability(self):
                                                                                                            """Test that system state is immutable after freezing"""
                                                                                                            state = SystemState()
                                                                                                            state.freeze()

                                                                                                            with pytest.raises(RuntimeError) as exc_info:
                                                                                                                state.drift_count = 100

                                                                                                                assert "immutable" in str(exc_info.value)

                                                                                                                def test_system_state_serialization(self):
                                                                                                                    """Test system state can be serialized"""
                                                                                                                    state = SystemState()
                                                                                                                    state_dict = state.to_dict()

                                                                                                                    assert isinstance(state_dict, dict)
                                                                                                                    assert "compliance_mode" in state_dict
                                                                                                                    assert "drift_count" in state_dict
                                                                                                                    assert "kill_switch" in state_dict


                                                                                                                    class TestKillSwitch:
                                                                                                                        """Test kill switch functionality"""

                                                                                                                        @patch("core.immutable_runtime_enforcer.sys.exit")
                                                                                                                        @patch("core.immutable_runtime_enforcer.save_system_state")
                                                                                                                        def test_kill_switch_activation(self, mock_save, mock_exit):
                                                                                                                            """Test kill switch activates correctly"""
                                                                                                                            # Reset global state
                                                                                                                            import core.immutable_runtime_enforcer as enforcer

                                                                                                                            enforcer.KILL_SWITCH_ACTIVATED = False

                                                                                                                            # Activate kill switch
                                                                                                                            with patch.object(enforcer.SYSTEM_STATE, "enforcement_level", "strict"):
                                                                                                                                activate_kill_switch("Test violation")

                                                                                                                                # Verify activation
                                                                                                                                assert enforcer.KILL_SWITCH_ACTIVATED == True
                                                                                                                                mock_save.assert_called_once()
                                                                                                                                mock_exit.assert_called_once_with(1)


                                                                                                                                class TestIntegration:
                                                                                                                                    """Integration tests for the complete system"""

                                                                                                                                    def test_end_to_end_compliance_flow(self):
                                                                                                                                        """Test complete compliance flow"""

                                                                                                                                        # Create a compliant data fetch
                                                                                                                                        @compliance_wrap
                                                                                                                                        def fetch_market_data(ticker, source="polygon"):
                                                                                                                                            return {"ticker": ticker, "price": 150.0, "source": source}

                                                                                                                                            # Valid request
                                                                                                                                            data = fetch_market_data("AAPL", source="polygon")
                                                                                                                                            assert data["price"] == 150.0

                                                                                                                                            # Invalid request
                                                                                                                                            data = fetch_market_data("AAPL", source="yahoo")
                                                                                                                                            assert data == "NA"

                                                                                                                                            def test_response_verification_flow(self):
                                                                                                                                                """Test AI response verification flow"""
                                                                                                                                                # Valid response
                                                                                                                                                valid = "Based on Polygon data, AAPL price is $150.00"
                                                                                                                                                assert verify_response(valid) == valid

                                                                                                                                                # Invalid response with drift
                                                                                                                                                with patch("core.immutable_runtime_enforcer.SYSTEM_STATE") as mock_state:
                                                                                                                                                    mock_state.enforcement_level = "normal"
                                                                                                                                                    invalid = "For example, AAPL could be worth $200"
                                                                                                                                                    assert verify_response(invalid) == "NA"


                                                                                                                                                    if __name__ == "__main__":
                                                                                                                                                        pytest.main([__file__, "-v"])
