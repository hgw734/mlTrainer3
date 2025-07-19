import streamlit as st
import datetime
from core.system_router import SystemRouter
from monitoring.monitor import run_monitoring_cycle

# Style overrides
st.markdown(
    """
    <style>
    html, body {
        font-family: Georgia, 'Times New Roman', Times, serif;
    }
    .stApp {
        background-color: #ffffff;
        color: #2c3e50;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Instantiate router
router = SystemRouter()
st.set_page_config(page_title="FMT2 Trading Intelligence", layout="wide")

st.title("📈 FMT2 Trading Intelligence System")
st.subheader("Momentum Model Trainer · Live Market Interface")

# Sidebar
with st.sidebar:
    st.header("🔧 Trial Configuration")
    symbol = st.text_input("Ticker Symbol", "AAPL")
    start_date = st.date_input("Start Date", datetime.date(2022, 1, 1))
    end_date = st.date_input("End Date", datetime.date(2023, 1, 1))

    if st.button("🚀 Start Walk-Forward Trial"):
        with st.spinner("Running trial..."):
            try:
                result = router.route_training_trial(
                    symbol, str(start_date), str(end_date))
                st.session_state["last_result"] = result
                st.success("✅ Trial completed.")
            except Exception as e:
                st.error(f"❌ Trial failed: {e}")

# Main Area
st.markdown("## 🔍 Trial Results")
if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    st.json(result)

# Monitoring Panel
st.markdown("## 🩺 System Monitoring")
monitoring_report = run_monitoring_cycle()
st.json(monitoring_report)
