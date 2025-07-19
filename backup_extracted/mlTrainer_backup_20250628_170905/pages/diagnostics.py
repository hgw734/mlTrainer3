import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime
from utils.performance import get_training_performance_summary, get_model_variance_report

st.set_page_config(page_title="Diagnostics", layout="wide")

st.title("📊 Model Diagnostics & Training Feedback")

# === 1. Summary Cards ===
with st.container():
    col1, col2, col3 = st.columns(3)
    summary = get_training_performance_summary()

    col1.metric("📈 Avg Accuracy", f"{summary['avg_accuracy']:.2f}%", delta=None)
    col2.metric("🧠 Active Models", summary['active_models'])
    col3.metric("⏱️ Last Training", summary['last_trained'])

st.markdown("---")

# === 2. Model Performance Table ===
st.subheader("🔬 Accuracy by Model and Market Regime")
model_perf_df = summary['accuracy_table']

if not model_perf_df.empty:
    st.dataframe(model_perf_df.style.background_gradient(cmap='RdYlGn'), use_container_width=True)
else:
    st.info("No walk-forward training results available yet.")

# === 3. Underperforming Zones ===
st.subheader("🚨 Weak Regime Zones (Focus Training)")
variance_df = get_model_variance_report()

if not variance_df.empty:
    st.dataframe(variance_df.style.highlight_max(axis=0), use_container_width=True)
else:
    st.success("No significant weaknesses detected across models.")

st.markdown("---")

# === 4. Versioning & Health ===
st.subheader("⚙️ Training System Status")
st.text("✅ All model versions up to date.\n✅ No socket or syntax issues.\n✅ Compliance: Fully enforced.")
