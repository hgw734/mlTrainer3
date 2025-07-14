#!/usr/bin/env python3
"""
🧠 COMPREHENSIVE SELF-LEARNING ENGINE TEST SUITE
Test all capabilities of the self-learning and self-correcting ML engine
"""

import numpy as np
import sys
import traceback
from datetime import datetime


# Test all core imports
def test_imports():
    """Test that all required modules can be imported"""
    try:
        print("🔄 Testing imports...")

        from self_learning_engine import (
        SelfLearningEngine,
        AdaptiveEnsemble,
        ModelPerformanceRecord,
        LearningContext,
        MetaKnowledge,
        initialize_self_learning_engine,
        )

        from self_learning_engine_helpers import (
        SelfLearningEngineHelpers,
        create_learning_context,
        analyze_model_compatibility,
        )

        print("✅ All imports successful!")
        return True

        except Exception as e:
            print(f"❌ Import failed: {e}")
            traceback.print_exc()
            return False


            def test_engine_initialization():
                """Test self-learning engine initialization"""
                try:
                    print("\n🔄 Testing engine initialization...")

                    from self_learning_engine import initialize_self_learning_engine

                    engine = initialize_self_learning_engine()

                    # Test basic properties
                    assert hasattr(engine, "system_models")
                    assert hasattr(engine, "meta_knowledge")
                    assert hasattr(engine, "learning_memory")

                    # Test learning status
                    status = engine.get_learning_status()
                    assert "learning_iterations" in status
                    assert "models_in_system" in status
                    assert "learning_engine_health" in status

                    print(f"✅ Engine initialized successfully!")
                    print(f"   Models in system: {status['models_in_system']}")
                    print(f"   Engine health: {status['learning_engine_health']}")

                    return engine

                    except Exception as e:
                        print(f"❌ Engine initialization failed: {e}")
                        traceback.print_exc()
                        return None


                        def test_learning_context():
                            """Test learning context creation and usage"""
                            try:
                                print("\n🔄 Testing learning context...")

                                from self_learning_engine_helpers import create_learning_context

                                context = create_learning_context(
                                market_regime="volatile", volatility_level="high", data_quality_score=0.85, prediction_horizon=120
                                )

                                assert context.market_regime == "volatile"
                                assert context.volatility_level == "high"
                                assert context.data_quality_score == 0.85
                                assert context.prediction_horizon == 120

                                print("✅ Learning context created successfully!")
                                return context

                                except Exception as e:
                                    print(f"❌ Learning context creation failed: {e}")
                                    traceback.print_exc()
                                    return None


                                    def test_adaptive_model_selection(engine, context):
                                        """Test adaptive model selection capability"""
                                        try:
                                            print("\n🔄 Testing adaptive model selection...")

                                            # Create synthetic data
                                            X = np.random.randn(100, 5)

                                            # Test model selection
                                            selection_result = engine.adaptive_model_selection(X, context)

                                            assert "strategy" in selection_result
                                            assert selection_result["strategy"] in ["single_model", "ensemble"]

                                            if selection_result["strategy"] == "ensemble":
                                                assert "models" in selection_result
                                                assert "selection_confidence" in selection_result
                                                print(f"✅ Ensemble strategy selected with {len(selection_result['models'])} models")
                                                else:
                                                    assert "model" in selection_result
                                                    assert "confidence" in selection_result
                                                    print(f"✅ Single model selected: {selection_result['model']}")

                                                    return selection_result

                                                    except Exception as e:
                                                        print(f"❌ Adaptive model selection failed: {e}")
                                                        traceback.print_exc()
                                                        return None


                                                        def test_learning_from_prediction(engine, context):
                                                            """Test learning from prediction feedback"""
                                                            try:
                                                                print("\n🔄 Testing learning from prediction...")

                                                                # Create synthetic prediction data
                                                                X = np.random.randn(50, 4)
                                                                y_true = np.random.randn(50)
                                                                y_pred = y_true + np.random.normal(0, 0.1, 50)  # Add some noise

                                                                # Test learning
                                                                learning_result = engine.learn_from_prediction(X, y_true, y_pred, model_name="random_forest", context=context)

                                                                assert "performance_record" in learning_result
                                                                assert "learning_iteration" in learning_result
                                                                assert "meta_knowledge_updated" in learning_result

                                                                record = learning_result["performance_record"]
                                                                assert record.model_name == "random_forest"
                                                                assert isinstance(record.prediction_accuracy, float)

                                                                print(f"✅ Learning completed successfully!")
                                                                print(f"   Prediction accuracy: {record.prediction_accuracy:.4f}")
                                                                print(f"   Learning iteration: {learning_result['learning_iteration']}")

                                                                return learning_result

                                                                except Exception as e:
                                                                    print(f"❌ Learning from prediction failed: {e}")
                                                                    traceback.print_exc()
                                                                    return None


                                                                    def test_self_correction(engine, context):
                                                                        """Test self-correction mechanism"""
                                                                        try:
                                                                            print("\n🔄 Testing self-correction...")

                                                                            # Create declining performance history
                                                                            performance_history = [
                                                                            {"accuracy": 0.85, "timestamp": "2024-01-01"},
                                                                            {"accuracy": 0.80, "timestamp": "2024-01-02"},
                                                                            {"accuracy": 0.75, "timestamp": "2024-01-03"},
                                                                            {"accuracy": 0.70, "timestamp": "2024-01-04"},
                                                                            {"accuracy": 0.65, "timestamp": "2024-01-05"},
                                                                            ]

                                                                            # Test self-correction
                                                                            correction_result = engine.self_correct_performance(performance_history, context)

                                                                            assert "corrections_applied" in correction_result
                                                                            assert "correction_count" in correction_result
                                                                            assert "expected_improvement" in correction_result

                                                                            print(f"✅ Self-correction completed!")
                                                                            print(f"   Corrections applied: {correction_result['correction_count']}")
                                                                            print(f"   Expected improvement: {correction_result['expected_improvement']:.2%}")

                                                                            return correction_result

                                                                            except Exception as e:
                                                                                print(f"❌ Self-correction failed: {e}")
                                                                                traceback.print_exc()
                                                                                return None


                                                                                def test_hyperparameter_evolution(engine, context):
                                                                                    """Test hyperparameter evolution capability"""
                                                                                    try:
                                                                                        print("\n🔄 Testing hyperparameter evolution...")

                                                                                        # Create synthetic data
                                                                                        X = np.random.randn(100, 6)
                                                                                        y = np.random.randn(100)

                                                                                        # Test hyperparameter evolution (with limited trials for testing)
                                                                                        evolution_result = engine.evolve_hyperparameters("ridge_regression", X, y, context)

                                                                                        assert "best_parameters" in evolution_result
                                                                                        assert "best_score" in evolution_result
                                                                                        assert "improvement" in evolution_result

                                                                                        print(f"✅ Hyperparameter evolution completed!")
                                                                                        print(f"   Best score: {evolution_result['best_score']:.4f}")
                                                                                        print(f"   Improvement: {evolution_result['improvement']:.4f}")
                                                                                        print(f"   Best parameters: {evolution_result['best_parameters']}")

                                                                                        return evolution_result

                                                                                        except Exception as e:
                                                                                            print(f"❌ Hyperparameter evolution failed: {e}")
                                                                                            traceback.print_exc()
                                                                                            return None


                                                                                            def test_ensemble_creation(engine, context):
                                                                                                """Test adaptive ensemble creation"""
                                                                                                try:
                                                                                                    print("\n🔄 Testing ensemble creation...")

                                                                                                    # Create synthetic data
                                                                                                    X = np.random.randn(150, 8)
                                                                                                    y = np.random.randn(150)

                                                                                                    # Test ensemble creation
                                                                                                    ensemble_result = engine.create_adaptive_ensemble(X, y, context)

                                                                                                    assert "ensemble" in ensemble_result
                                                                                                    assert "models" in ensemble_result
                                                                                                    assert "weights" in ensemble_result
                                                                                                    assert "individual_performances" in ensemble_result

                                                                                                    ensemble = ensemble_result["ensemble"]

                                                                                                    # Test ensemble prediction
                                                                                                    X_test = np.random.randn(10, 8)
                                                                                                    predictions = ensemble.predict(X_test)

                                                                                                    assert len(predictions) == 10
                                                                                                    assert isinstance(predictions[0], (int, float, np.number))

                                                                                                    print(f"✅ Ensemble creation completed!")
                                                                                                    print(f"   Ensemble size: {ensemble_result['ensemble_size']}")
                                                                                                    print(f"   Models: {ensemble_result['models']}")
                                                                                                    print(f"   Weights: {ensemble_result['weights']}")

                                                                                                    return ensemble_result

                                                                                                    except Exception as e:
                                                                                                        print(f"❌ Ensemble creation failed: {e}")
                                                                                                        traceback.print_exc()
                                                                                                        return None


                                                                                                        def test_model_compatibility():
                                                                                                            """Test model compatibility analysis"""
                                                                                                            try:
                                                                                                                print("\n🔄 Testing model compatibility...")

                                                                                                                from self_learning_engine_helpers import create_learning_context, analyze_model_compatibility

                                                                                                                context = create_learning_context(data_quality_score=0.9, prediction_horizon=60)

                                                                                                                # Test compatibility for different models
                                                                                                                models_to_test = ["linear_regression", "random_forest", "xgboost"]

                                                                                                                for model_name in models_to_test:
                                                                                                                    compatibility = analyze_model_compatibility(model_name, context)

                                                                                                                    assert "compatible" in compatibility

                                                                                                                    status = "✅ Compatible" if compatibility["compatible"] else "❌ Not Compatible"
                                                                                                                    print(f"   {model_name}: {status}")

                                                                                                                    print("✅ Model compatibility analysis completed!")
                                                                                                                    return True

                                                                                                                    except Exception as e:
                                                                                                                        print(f"❌ Model compatibility analysis failed: {e}")
                                                                                                                        traceback.print_exc()
                                                                                                                        return False


                                                                                                                        def run_comprehensive_test():
                                                                                                                            """Run comprehensive test suite"""
                                                                                                                            print("🧠 COMPREHENSIVE SELF-LEARNING ENGINE TEST SUITE")
                                                                                                                            print(("=" * 60))
                                                                                                                            print(f"Test started at: {datetime.now()}")

                                                                                                                            # Track test results
                                                                                                                            test_results = {}

                                                                                                                            # Test 1: Imports
                                                                                                                            test_results["imports"] = test_imports()

                                                                                                                            if not test_results["imports"]:
                                                                                                                                print("\n❌ CRITICAL FAILURE: Cannot proceed without successful imports")
                                                                                                                                return test_results

                                                                                                                                # Test 2: Engine initialization
                                                                                                                                engine = test_engine_initialization()
                                                                                                                                test_results["initialization"] = engine is not None

                                                                                                                                if not test_results["initialization"]:
                                                                                                                                    print("\n❌ CRITICAL FAILURE: Cannot proceed without engine initialization")
                                                                                                                                    return test_results

                                                                                                                                    # Test 3: Learning context
                                                                                                                                    context = test_learning_context()
                                                                                                                                    test_results["learning_context"] = context is not None

                                                                                                                                    if not test_results["learning_context"]:
                                                                                                                                        print("\n❌ Cannot proceed without learning context")
                                                                                                                                        return test_results

                                                                                                                                        # Test 4: Adaptive model selection
                                                                                                                                        selection_result = test_adaptive_model_selection(engine, context)
                                                                                                                                        test_results["adaptive_model_selection"] = selection_result is not None

                                                                                                                                        # Test 5: Learning from prediction
                                                                                                                                        learning_result = test_learning_from_prediction(engine, context)
                                                                                                                                        test_results["learning_from_prediction"] = learning_result is not None

                                                                                                                                        # Test 6: Self-correction
                                                                                                                                        correction_result = test_self_correction(engine, context)
                                                                                                                                        test_results["self_correction"] = correction_result is not None

                                                                                                                                        # Test 7: Hyperparameter evolution
                                                                                                                                        evolution_result = test_hyperparameter_evolution(engine, context)
                                                                                                                                        test_results["hyperparameter_evolution"] = evolution_result is not None

                                                                                                                                        # Test 8: Ensemble creation
                                                                                                                                        ensemble_result = test_ensemble_creation(engine, context)
                                                                                                                                        test_results["ensemble_creation"] = ensemble_result is not None

                                                                                                                                        # Test 9: Model compatibility
                                                                                                                                        test_results["model_compatibility"] = test_model_compatibility()

                                                                                                                                        # Summary
                                                                                                                                        print(("\n" + "=" * 60))
                                                                                                                                        print("📊 TEST SUMMARY")
                                                                                                                                        print(("=" * 60))

                                                                                                                                        passed = sum(test_results.values())
                                                                                                                                        total = len(test_results)

                                                                                                                                        for test_name, result in list(test_results.items()):
                                                                                                                                            status = "✅ PASS" if result else "❌ FAIL"
                                                                                                                                            print(f"{test_name:.<40} {status}")

                                                                                                                                            print(f"\nOverall Result: {passed}/{total} tests passed")

                                                                                                                                            if passed == total:
                                                                                                                                                print("🎉 ALL TESTS PASSED! Self-Learning Engine is fully functional!")
                                                                                                                                                else:
                                                                                                                                                    print(f"⚠️  {total - passed} tests failed. Please review the errors above.")

                                                                                                                                                    print(f"\nTest completed at: {datetime.now()}")

                                                                                                                                                    return test_results


                                                                                                                                                    if __name__ == "__main__":
                                                                                                                                                        results = run_comprehensive_test()

                                                                                                                                                        # Exit with appropriate code
                                                                                                                                                        if all(results.values()):
                                                                                                                                                            sys.exit(0)
                                                                                                                                                            else:
                                                                                                                                                                sys.exit(1)
