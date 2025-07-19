import streamlit as st
import datetime
from utils.data_loader import get_live_recommendations, get_current_holdings, add_to_holdings, remove_from_holdings
from utils.execution_metrics import compute_live_metrics
from monitoring.monitor import run_monitoring_cycle

st.set_page_config(page_title="📊 Recommendations", layout="wide")

# --- Header ---
st.title("📊 Live AI Recommendations Dashboard")

status = run_monitoring_cycle()
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        "✅ Compliance",
        "ON" if status.get(
            "system_health",
            {}).get(
            "compliance",
            True) else "OFF")
with col2:
    st.metric("📡 Data Feed", "Live (15-min delayed)")
with col3:
    st.metric(
        "🔄 Updates",
        f"{status.get('system_health', {}).get('update_count', 0)}")

st.divider()

# --- Metrics ---
metrics = compute_live_metrics()
col1, col2, col3, col4 = st.columns(4)
col1.metric("📌 Active Recos", metrics["active_recommendations"])
col2.metric("💼 Holdings", metrics["current_holdings"])
col3.metric("✅ Win Rate", f"{metrics['win_rate']}%")
col4.metric("📈 Avg Score", f"{metrics['avg_score']}")

# --- Live Recommendations Table ---
st.markdown("### 🔍 Live Recommendations")

recommendations = get_live_recommendations()

if not recommendations.empty:
    recommendations["✔"] = False
    color_map = {"High": "✅", "Medium": "⚠️", "Low": "❌"}

    def highlight_score(row):
        if row["Score"] >= 80:
            return "High"
        elif row["Score"] >= 60:
            return "Medium"
        else:
            return "Low"

    recommendations["Rating"] = recommendations.apply(highlight_score, axis=1)
    recommendations["Score"] = recommendations.apply(
        lambda r: f"{color_map[r['Rating']]} {r['Score']}", axis=1)
    recommendations = recommendations.drop(columns=["Rating"])

    # Checkbox session key
    if "selected_recommendations" not in st.session_state:
        st.session_state.selected_recommendations = set()

    edited_recos = st.data_editor(
        recommendations,
        use_container_width=True,
        key="recommendations_editor"
    )

    # Process checkboxes
    for idx, row in edited_recos.iterrows():
        if row["✔"]:
            symbol = row["Ticker"]
            if symbol not in st.session_state.selected_recommendations:
                add_to_holdings(row)
                st.session_state.selected_recommendations.add(symbol)
else:
    st.warning("No live recommendations at this time.")

# --- Holdings Tracker Table ---
st.markdown("### 💼 Current Holdings")
holdings = get_current_holdings()

if not holdings.empty:
    holdings["✔"] = False
    holdings["Progress to Target"] = holdings["Progress %"].apply(
        lambda pct: f"{pct}% [{'█' * int(pct // 10)}{'-' * (10 - int(pct // 10))}]")

    # Checkbox session key
    if "selected_holdings" not in st.session_state:
        st.session_state.selected_holdings = set()

    edited_holdings = st.data_editor(
        holdings.drop(columns=["Progress %"]),
        use_container_width=True,
        key="holdings_editor"
    )

    for idx, row in edited_holdings.iterrows():
        if row["✔"]:
            symbol = row["Ticker"]
            if symbol not in st.session_state.selected_holdings:
                remove_from_holdings(symbol)
                st.session_state.selected_holdings.add(symbol)
else:
    st.info("No holdings yet. Add from recommendations.")

# --- Learning Insight ---
st.divider()
st.markdown("### 🧠 Learning Insights")

st.markdown(f"""
- **Market Regime:** {status['system_health'].get('regime_score', 'NA')}
- **Trainer Status:** {status['system_health'].get('ml_trainer_status', 'Idle')}
- **Insight:** System adapting to high-volatility conditions with momentum preference.
- **Last Update:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")
