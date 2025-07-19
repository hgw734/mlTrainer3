
# model_router_dashboard.py

import streamlit as st
import plotly.express as px
import pandas as pd
from model_router import ModelRouter

st.set_page_config(page_title="📡 Model Router – AdvanSng2", layout="wide")

# Load router
try:
    router = ModelRouter()
    st.success("✅ Model Router loaded successfully")
except Exception as e:
    st.error(f"❌ Error loading Model Router: {e}")
    st.stop()

# Custom CSS for better styling
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #ff6b6b;
}
.model-card {
    background-color: #e8f5e8;
    padding: 0.8rem;
    border-radius: 0.3rem;
    margin: 0.2rem 0;
    border-left: 3px solid #4caf50;
}
.weight-display {
    background-color: #fff3cd;
    padding: 0.5rem;
    border-radius: 0.3rem;
    margin: 0.2rem 0;
}
</style>
""", unsafe_allow_html=True)

st.title("📡 AdvanSng2 Model Router Dashboard")
st.markdown(
    "**Multi-dimensional regime analysis with real-time model activation**")

# Layout: Main content and sidebar
col1, col2 = st.columns([2, 1])

with col2:
    st.header("🧭 Regime Configuration")

    # Input controls
    volatility = st.selectbox(
        "Volatility Level",
        ["low", "medium", "high"],
        index=2,
        help="Market volatility assessment"
    )

    macro_signal = st.selectbox(
        "Macro Signal",
        ["neutral", "trending", "shock", "macro_shift", "irregular"],
        index=2,
        help="Macroeconomic signal classification"
    )

    regime_score = st.slider(
        "Regime Score",
        0, 100, 75,
        help="Composite regime strength score (0-100)"
    )

    st.markdown("---")

    # Real-time regime assessment
    st.subheader("📊 Current Regime")
    regime_type = router._determine_regime_type(
        volatility, macro_signal, regime_score)

    if regime_score > 70:
        score_color = "🔴"
    elif regime_score > 40:
        score_color = "🟡"
    else:
        score_color = "🟢"

    st.markdown(f"""
    <div class="metric-card">
        <strong>Regime Type:</strong> {regime_type.upper()}<br>
        <strong>Volatility:</strong> {volatility.upper()}<br>
        <strong>Macro Signal:</strong> {macro_signal.upper()}<br>
        <strong>Score:</strong> {score_color} {regime_score}/100
    </div>
    """, unsafe_allow_html=True)

with col1:
    st.header("🚀 Model Activation Results")

    # Get routing results
    try:
        explanation = router.explain_routing(
            volatility, macro_signal, regime_score)
        models = explanation["activated_models"]
        weights = explanation["ensemble_weights"]
        reasoning = explanation["routing_reasoning"]

        # Display activated models
        st.subheader("🎯 Active Models")
        if models:
            # Create columns for model display
            model_cols = st.columns(min(3, len(models)))
            for i, model in enumerate(models):
                with model_cols[i % 3]:
                    weight = weights.get(model, 0.0)
                    weight_display = f"Weight: {weight:.2f}" if weight > 0 else "Ensemble"
                    st.markdown(f"""
                    <div class="model-card">
                        <strong>{model}</strong><br>
                        <small>{weight_display}</small>
                    </div>
                    """, unsafe_allow_html=True)

            st.success(f"✅ {len(models)} models activated for current regime")
        else:
            st.warning("⚠️ No models matched the current regime conditions")

        # Ensemble weights visualization
        st.subheader("⚖️ Ensemble Weights Distribution")
        if weights:
            # Create weights dataframe for visualization
            weights_df = pd.DataFrame(
                list(
                    weights.items()),
                columns=[
                    'Model',
                    'Weight'])

            # Plotly pie chart
            fig = px.pie(
                weights_df,
                values='Weight',
                names='Model',
                title=f"Weight Distribution for {regime_type.title()} Regime"
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

            # Weight details table
            st.markdown("**Weight Details:**")
            for model, weight in weights.items():
                st.markdown(f"""
                <div class="weight-display">
                    <strong>{model}:</strong> {weight:.3f} ({weight*100:.1f}%)
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("ℹ️ No specific ensemble weights found for this regime")

        # Routing explanation
        st.subheader("🔍 Decision Reasoning")
        st.markdown(f"**Logic:** {reasoning}")

        # Technical details (expandable)
        with st.expander("🔧 Technical Details"):
            st.json(explanation)

    except Exception as e:
        st.error(f"❌ Error getting routing results: {e}")

# Bottom section: Configuration status and health
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📋 Configuration Status")
    try:
        # Check if config files exist
        import os
        yaml_exists = os.path.exists("model_routing.yaml")
        json_exists = os.path.exists("ensemble_weights.json")

        st.write(f"🔧 YAML Config: {'✅' if yaml_exists else '❌'}")
        st.write(f"⚖️ Weight Config: {'✅' if json_exists else '❌'}")

        if yaml_exists and json_exists:
            st.success("All config files loaded")
        else:
            st.warning("Using default configurations")

    except Exception as e:
        st.error(f"Config check failed: {e}")

with col2:
    st.subheader("🎛️ Router Statistics")
    try:
        # Get regime configurations
        regimes = router.routing_config.get("regimes", {})
        weight_configs = len(router.weight_config)

        st.metric("Regime Rules", len(regimes))
        st.metric("Weight Profiles", weight_configs)
        st.metric("Current Models", len(models) if 'models' in locals() else 0)

    except Exception as e:
        st.error(f"Stats error: {e}")

with col3:
    st.subheader("🔄 Actions")

    if st.button("🔄 Reload Router"):
        try:
            router = ModelRouter()
            st.success("Router reloaded!")
            st.rerun()
        except Exception as e:
            st.error(f"Reload failed: {e}")

    if st.button("📊 Run Stress Test"):
        st.info("Running regime stress test...")

        # Test multiple regime combinations
        test_scenarios = [
            ("low", "neutral", 25),
            ("medium", "trending", 50),
            ("high", "shock", 85),
            ("high", "macro_shift", 90)
        ]

        results = []
        for vol, macro, score in test_scenarios:
            try:
                test_result = router.explain_routing(vol, macro, score)
                results.append({
                    "Scenario": f"{vol}-{macro}-{score}",
                    "Models": len(test_result["activated_models"]),
                    "Regime": test_result["regime_type"]
                })
            except Exception:
                results.append({
                    "Scenario": f"{vol}-{macro}-{score}",
                    "Models": 0,
                    "Regime": "ERROR"
                })

        st.dataframe(pd.DataFrame(results))

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>
    📡 AdvanSng2 Model Router Dashboard |
    Real-time regime analysis with multi-dimensional classification |
    🔒 Compliance-enforced model selection
    </small>
</div>
""", unsafe_allow_html=True)
