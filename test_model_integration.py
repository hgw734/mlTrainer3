"""
Test Model Integration
======================

Test script to verify ML and Financial model managers are working with mlTrainer.
"""

import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_ml_models():
    """Test ML model manager"""
    print(("\n" + "=" * 60))
    print("Testing ML Model Manager")
    print(("=" * 60))

    try:
        from mltrainer_models import get_ml_model_manager

        manager = get_ml_model_manager()
        print(f"✓ ML Model Manager initialized")

        # Get model summary
        summary = manager.get_summary_report()
        print(f"\nAvailable Models: {summary['total_models_available']}")

        # Show categories
        print("\nModel Categories:")
        for category, info in list(summary["categories"].items()):
            print(f"  • {category}: {info['available']} models")

            # Test training a simple model
            print("\nTest: Training linear regression model...")
            try:
                result = manager.train_model(
                    "linear_regression", symbol="AAPL", data_source="polygon")

                if result.compliance_status == "approved":
                    print(f"✓ Model trained successfully!")
                    print(
                        f"  R² Score: {result.performance_metrics.get('r2_score', 0):.4f}")
                    print(f"  Training time: {result.training_time:.2f}s")
                    else:
                        print(
                            f"✗ Model training failed compliance: {result.compliance_status}")

                        except Exception as e:
                            print(f"✗ Model training failed: {str(e)[:100]}")

                            # Show available models sample
                            print("\nSample Available Models:")
                            for model_id in manager.get_available_models()[:5]:
                                info = manager.get_model_info(model_id)
                                print(
                                    f"  • {model_id}: {info.get('category', 'unknown')} - complexity: {info.get('complexity', 'unknown')}"
                                )

                                return True

                                except Exception as e:
                                    print(
                                        f"✗ ML Model Manager test failed: {e}")
                                    logger.error(
                                        f"ML test error: {e}", exc_info=True)
                                    return False

                                    def test_financial_models():
                                        """Test Financial model manager"""
                                        print(("\n" + "=" * 60))
                                        print("Testing Financial Model Manager")
                                        print(("=" * 60))

                                        try:
                                            from mltrainer_financial_models import get_financial_model_manager

                                            manager = get_financial_model_manager()
                                            print(
                                                f"✓ Financial Model Manager initialized")

                                            # Show available models
                                            models = manager.get_available_models()
                                            print(
                                                f"\nAvailable Financial Models: {len(models)}")
                                            for model_id in models:
                                                info = manager.get_model_info(
                                                    model_id)
                                                print(
                                                    f"  • {model_id}: {info['description']}")

                                                # Test Black-Scholes
                                                print(
                                                    "\nTest: Black-Scholes Option Pricing...")
                                                try:
                                                    result = manager.run_model(
                                                        "black_scholes",
                                                        spot=100,
                                                        strike=105,
                                                        risk_free_rate=0.05,
                                                        volatility=0.2,
                                                        time_to_expiry=0.25,
                                                        option_type="call",
                                                    )

                                                    if result.compliance_status == "approved":
                                                        print(
                                                            f"✓ Black-Scholes calculated successfully!")
                                                        print(
                                                            f"  Option Price: ${result.option_price:.2f}")
                                                        print(
                                                            f"  Delta: {result.greeks['delta']:.4f}")
                                                        print(
                                                            f"  Gamma: {result.greeks['gamma']:.4f}")
                                                        else:
                                                            print(
                                                                f"✗ Calculation failed compliance")

                                                            except Exception as e:
                                                                print(
                                                                    f"✗ Black-Scholes test failed: {str(e)[:100]}")

                                                                # Test VaR
                                                                print(
                                                                    "\nTest: Value at Risk Calculation...")
                                                                try:
                                                                    import numpy as np
                                                                    import pandas as pd

                                                                    # Create
                                                                    # sample
                                                                    # returns
                                                                    returns = pd.Series(
                                                                        np.random.normal(0.001, 0.02, 252))

                                                                    result = manager.run_model(
                                                                        "value_at_risk", returns=returns, confidence_level=0.95, method="historical")

                                                                    if result.compliance_status == "approved":
                                                                        print(
                                                                            f"✓ VaR calculated successfully!")
                                                                        print(
                                                                            f"  VaR (95%): {result.risk_metrics['var']:.4f}")
                                                                        print(
                                                                            f"  Expected Shortfall: {result.risk_metrics['expected_shortfall']:.4f}")
                                                                        else:
                                                                            print(
                                                                                f"✗ VaR calculation failed compliance")

                                                                            except Exception as e:
                                                                                print(
                                                                                    f"✗ VaR test failed: {str(e)[:100]}")

                                                                                return True

                                                                                except Exception as e:
                                                                                    print(
                                                                                        f"✗ Financial Model Manager test failed: {e}")
                                                                                    logger.error(
                                                                                        f"Financial test error: {e}", exc_info=True)
                                                                                    return False

                                                                                    def test_model_integration():
                                                                                        """Test model integration with mlAgent"""
                                                                                        print(
                                                                                            ("\n" + "=" * 60))
                                                                                        print(
                                                                                            "Testing Model Integration")
                                                                                        print(
                                                                                            ("=" * 60))

                                                                                        try:
                                                                                            from mlagent_model_integration import get_model_integration

                                                                                            integration = get_model_integration()
                                                                                            print(
                                                                                                f"✓ Model Integration initialized")

                                                                                            # Test
                                                                                            # parsing
                                                                                            # model
                                                                                            # requests
                                                                                            print(
                                                                                                "\nTest: Parsing Model Requests...")

                                                                                            test_responses = [
                                                                                                "Let's train model random_forest_50 on AAPL data from Polygon",
                                                                                                "Calculate Black-Scholes option price with spot 100, strike 105, volatility 0.2",
                                                                                                "Show me the best models",
                                                                                                "List available models",
                                                                                            ]

                                                                                            for response in test_responses:
                                                                                                request = integration.parse_model_request(
                                                                                                    response)
                                                                                                if request["model_id"] or request["action"]:
                                                                                                    print(
                                                                                                        f"✓ Parsed: {response[:50]}...")
                                                                                                    print(
                                                                                                        f"  Type: {request['type']}, Action: {request['action']}, Model: {request['model_id']}")

                                                                                                    # Test
                                                                                                    # model
                                                                                                    # recommendations
                                                                                                    print(
                                                                                                        "\nTest: Model Recommendations...")
                                                                                                    recommendations = integration.get_model_recommendations(
                                                                                                        {"objective": "portfolio optimization", "data_type": "returns"}
                                                                                                    )

                                                                                                    if recommendations:
                                                                                                        print(
                                                                                                            "✓ Got recommendations:")
                                                                                                        for rec in recommendations:
                                                                                                            print(
                                                                                                                f"  • {rec}")

                                                                                                            return True

                                                                                                            except Exception as e:
                                                                                                                print(
                                                                                                                    f"✗ Model Integration test failed: {e}")
                                                                                                                logger.error(
                                                                                                                    f"Integration test error: {e}", exc_info=True)
                                                                                                                return False

                                                                                                                def main():
                                                                                                                    """Run all tests"""
                                                                                                                    print(
                                                                                                                        """
                                                                                                                    ╔══════════════════════════════════════════════════════════╗
                                                                                                                    ║                                                          ║
                                                                                                                    ║         mlTrainer Model Integration Test Suite           ║
                                                                                                                    ║                                                          ║
                                                                                                                    ╚══════════════════════════════════════════════════════════╝
                                                                                                                    """)

                                                                                                                    print(
                                                                                                                        f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                                                                                                                    # Run
                                                                                                                    # tests
                                                                                                                    ml_ok = test_ml_models()
                                                                                                                    financial_ok = test_financial_models()
                                                                                                                    integration_ok = test_model_integration()

                                                                                                                    # Summary
                                                                                                                    print(
                                                                                                                        ("\n" + "=" * 60))
                                                                                                                    print(
                                                                                                                        "TEST SUMMARY")
                                                                                                                    print(
                                                                                                                        ("=" * 60))
                                                                                                                    print(
                                                                                                                        f"ML Models: {'✅ PASS' if ml_ok else '❌ FAIL'}")
                                                                                                                    print(
                                                                                                                        f"Financial Models: {'✅ PASS' if financial_ok else '❌ FAIL'}")
                                                                                                                    print(
                                                                                                                        f"Model Integration: {'✅ PASS' if integration_ok else '❌ FAIL'}")

                                                                                                                    if ml_ok and financial_ok and integration_ok:
                                                                                                                        print(
                                                                                                                            "\n🎉 All tests passed! Model systems ready for use.")
                                                                                                                        print(
                                                                                                                            "\nYou can now:")
                                                                                                                        print(
                                                                                                                            "1. Ask Claude to train ML models (e.g., 'Train random_forest_100 on AAPL')")
                                                                                                                        print(
                                                                                                                            "2. Run financial models (e.g., 'Calculate Black-Scholes for...')")
                                                                                                                        print(
                                                                                                                            "3. Optimize portfolios (e.g., 'Optimize portfolio with AAPL, MSFT, GOOGL')")
                                                                                                                        print(
                                                                                                                            "4. Assess risk (e.g., 'Calculate VaR for my portfolio')")
                                                                                                                        else:
                                                                                                                            print(
                                                                                                                                "\n⚠️  Some tests failed. Check logs for details.")

                                                                                                                            return ml_ok and financial_ok and integration_ok

                                                                                                                            if __name__ == "__main__":
                                                                                                                                success = main()
                                                                                                                                exit(
                                                                                                                                    0 if success else 1)
