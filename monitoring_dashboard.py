"""
mlTrainer Monitoring Dashboard
==============================

Real-time monitoring dashboard for mlTrainer system health,
API status, and ML trial performance.
"""

import streamlit as st
import json
import os
from datetime import datetime, timedelta
import time
import psutil
from typing import Dict, List, Optional
import plotly.graph_objects as go
import plotly.express as px

# Import connectors and utilities
from polygon_connector import get_polygon_connector
from fred_connector import get_fred_connector
from polygon_rate_limiter import get_polygon_rate_limiter

# Page configuration
st.set_page_config(
    page_title="mlTrainer Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded")

# Custom CSS
st.markdown(
    """
<style>
.metric-card {
background-color: #f0f2f6;
padding: 20px;
border-radius: 10px;
margin: 10px 0;
}
.status-ok {
color: #28a745;
font-weight: bold;
}
.status-warning {
color: #ffc107;
font-weight: bold;
}
.status-error {
color: #dc3545;
font-weight: bold;
}
.big-font {
font-size: 24px !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# Helper functions
def get_system_metrics():
    """Get current system metrics"""
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage("/").percent,
        "process_count": len(psutil.pids()),
        "timestamp": datetime.now(),
    }

    def get_api_status():
        """Check API connection status"""
        status = {}

        # Check Polygon
        try:
            polygon_connector = get_polygon_connector()
            quote = polygon_connector.get_quote("AAPL")
            status["polygon"] = {
                "status": "OK" if quote else "ERROR",
                "message": f"AAPL: ${quote.price:.2f}" if quote else "Failed to get quote",
                "metrics": polygon_connector.get_quality_metrics(),
            }
        except Exception as e:
            status["polygon"] = {
                "status": "ERROR",
                "message": str(e),
                "metrics": None}

        # Check FRED
        try:
            fred_connector = get_fred_connector()
            gdp = fred_connector.get_series_data(
                "GDP", start_date="2024-01-01")
            status["fred"] = {
                "status": "OK" if gdp else "ERROR",
                "message": f"GDP data available: {len(gdp.data)} records" if gdp else "Failed to get data",
            }
        except Exception as e:
            status["fred"] = {"status": "ERROR", "message": str(e)}

        # Check Anthropic (Claude)
        try:
            from mltrainer_claude_integration import test_claude_connection
            claude_ok = test_claude_connection()
            status["claude"] = {
                "status": "OK" if claude_ok else "ERROR",
                "message": "Connected" if claude_ok else "Connection failed",
            }
        except Exception as e:
            status["claude"] = {"status": "ERROR", "message": str(e)}

        return status

    def get_trial_history():
        """Get ML trial history from logs"""
        history = []
        log_file = "logs/trial_history.json"

        if os.path.exists(log_file):
            try:
                with open(log_file, "r") as f:
                    for line in f:
                        try:
                            history.append(json.loads(line))
                        except BaseException:
                            pass
            except BaseException:
                pass

        return history[-20:]  # Last 20 trials

    def get_chat_activity():
        """Get chat activity metrics"""
        chat_file = "logs/chat_history.json"

        if os.path.exists(chat_file):
            try:
                with open(chat_file, "r") as f:
                    messages = json.load(f)

                    # Count messages by hour for last 24 hours
                    now = datetime.now()
                    hourly_counts = {i: 0 for i in range(24)}

                    for msg in messages:
                        msg_time = datetime.fromisoformat(msg["timestamp"])
                        if now - msg_time < timedelta(hours=24):
                            hour = msg_time.hour
                            hourly_counts[hour] += 1

                    return {
                        "total_messages": len(messages),
                        "last_24h": sum(hourly_counts.values()),
                        "hourly_counts": hourly_counts,
                    }
            except BaseException:
                pass

        return {
            "total_messages": 0,
            "last_24h": 0,
            "hourly_counts": {
                i: 0 for i in range(24)}}

    # Main dashboard

    def main():
        st.title("🎯 mlTrainer Monitoring Dashboard")
        st.markdown("Real-time system health and performance monitoring")

        # Sidebar
        with st.sidebar:
            st.header("⚙️ Controls")

            auto_refresh = st.checkbox("Auto-refresh", value=True)
            refresh_interval = st.slider(
                "Refresh interval (seconds)", 5, 60, 10)

            if st.button("🔄 Manual Refresh"):
                st.rerun()

            st.divider()

            # Current time
            st.caption(
                f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            # Main content
            tab1, tab2, tab3, tab4, tab5 = st.tabs(
                ["📊 System Health", "🔌 API Status", "🤖 ML Trials", "💬 Chat Activity", "📈 Market Data"]
            )

            # Tab 1: System Health
            with tab1:
                st.header("System Health Metrics")

                # Get system metrics
                metrics = get_system_metrics()

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    cpu_status = (
                        "status-ok"
                        if metrics["cpu_percent"] < 70
                        else "status-warning" if metrics["cpu_percent"] < 90 else "status-error"
                    )
                    st.metric("CPU Usage", f"{metrics['cpu_percent']}%")
                    st.markdown(
                        f"<p class='{cpu_status}'>{'Normal' if metrics['cpu_percent'] < 70 else 'High' if metrics['cpu_percent'] < 90 else 'Critical'}</p>",
                        unsafe_allow_html=True,
                    )

                with col2:
                    mem_status = (
                        "status-ok"
                        if metrics["memory_percent"] < 70
                        else "status-warning" if metrics["memory_percent"] < 90 else "status-error"
                    )
                    st.metric("Memory Usage", f"{metrics['memory_percent']}%")
                    st.markdown(
                        f"<p class='{mem_status}'>{'Normal' if metrics['memory_percent'] < 70 else 'High' if metrics['memory_percent'] < 90 else 'Critical'}</p>",
                        unsafe_allow_html=True,
                    )

                with col3:
                    disk_status = (
                        "status-ok"
                        if metrics["disk_percent"] < 80
                        else "status-warning" if metrics["disk_percent"] < 95 else "status-error"
                    )
                    st.metric("Disk Usage", f"{metrics['disk_percent']}%")
                    st.markdown(
                        f"<p class='{disk_status}'>{'Normal' if metrics['disk_percent'] < 80 else 'High' if metrics['disk_percent'] < 95 else 'Critical'}</p>",
                        unsafe_allow_html=True,
                    )

                with col4:
                    st.metric("Processes", metrics["process_count"])
                    st.markdown(
                        "<p class='status-ok'>Active</p>",
                        unsafe_allow_html=True)

                    # System resource gauges
                    st.subheader("Resource Usage Gauges")

                    fig = go.Figure()

                    # CPU Gauge
                    fig.add_trace(
                        go.Indicator(
                            mode="gauge+number",
                            value=metrics["cpu_percent"],
                            domain={"x": [0, 0.3], "y": [0, 1]},
                            title={"text": "CPU %"},
                            gauge={
                                "axis": {"range": [None, 100]},
                                "bar": {"color": "darkblue"},
                                "steps": [
                                    {"range": [0, 50], "color": "lightgray"},
                                    {"range": [50, 80], "color": "yellow"},
                                    {"range": [80, 100], "color": "red"},
                                ],
                                "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 90},
                            },
                        )
                    )

                    # Memory Gauge
                    fig.add_trace(
                        go.Indicator(
                            mode="gauge+number",
                            value=metrics["memory_percent"],
                            domain={"x": [0.35, 0.65], "y": [0, 1]},
                            title={"text": "Memory %"},
                            gauge={
                                "axis": {"range": [None, 100]},
                                "bar": {"color": "darkgreen"},
                                "steps": [
                                    {"range": [0, 50], "color": "lightgray"},
                                    {"range": [50, 80], "color": "yellow"},
                                    {"range": [80, 100], "color": "red"},
                                ],
                                "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 90},
                            },
                        )
                    )

                    # Disk Gauge
                    fig.add_trace(
                        go.Indicator(
                            mode="gauge+number",
                            value=metrics["disk_percent"],
                            domain={"x": [0.7, 1], "y": [0, 1]},
                            title={"text": "Disk %"},
                            gauge={
                                "axis": {"range": [None, 100]},
                                "bar": {"color": "darkviolet"},
                                "steps": [
                                    {"range": [0, 60], "color": "lightgray"},
                                    {"range": [60, 85], "color": "yellow"},
                                    {"range": [85, 100], "color": "red"},
                                ],
                                "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 95},
                            },
                        )
                    )

                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)

            # Tab 2: API Status
            with tab2:
                st.header("API Connection Status")

                api_status = get_api_status()

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.subheader("🔷 Polygon API")
                    if "polygon" in api_status:
                        status = api_status["polygon"]["status"]
                        st.markdown(
                            f"<p class='status-{status.lower()} big-font'>{status}</p>",
                            unsafe_allow_html=True)
                        st.caption(api_status["polygon"]["message"])

                        if api_status["polygon"].get("metrics"):
                            metrics = api_status["polygon"]["metrics"]
                            st.metric(
                                "Success Rate", f"{metrics['success_rate']:.1%}")
                            st.metric(
                                "Response Time",
                                f"{metrics['avg_response_time']:.2f}s")
                            st.metric(
                                "Dropout Rate", f"{metrics['dropout_rate']:.1%}")

                            with col2:
                                st.subheader("📈 FRED API")
                                if "fred" in api_status:
                                    status = api_status["fred"]["status"]
                                    st.markdown(
                                        f"<p class='status-{status.lower()} big-font'>{status}</p>",
                                        unsafe_allow_html=True)
                                    st.caption(api_status["fred"]["message"])

                                    with col3:
                                        st.subheader("🤖 Claude API")
                                        if "claude" in api_status:
                                            status = api_status["claude"]["status"]
                                            st.markdown(
                                                f"<p class='status-{status.lower()} big-font'>{status}</p>",
                                                unsafe_allow_html=True)
                                            st.caption(
                                                api_status["claude"]["message"])

                                            # Rate limiter status
                                            if "polygon" in api_status and api_status["polygon"].get(
                                                    "metrics"):
                                                st.divider()
                                                st.subheader(
                                                    "📊 Polygon Rate Limiter Status")

                                                metrics = api_status["polygon"]["metrics"]

                                                col1, col2, col3, col4 = st.columns(
                                                    4)

                                                with col1:
                                                    st.metric(
                                                        "Total Requests", metrics["total_requests"])
                                                    with col2:
                                                        st.metric(
                                                            "Circuit Breaker", "OPEN" if metrics["circuit_open"] else "CLOSED")
                                                        with col3:
                                                            st.metric(
                                                                "Rate Limit Active", "YES" if metrics["rate_limit_active"] else "NO")
                                                            with col4:
                                                                st.metric(
                                                                    "Avg Response", f"{metrics['avg_response_time']:.2f}s")

                                                # Tab 3: ML Trials
                                                with tab3:
                                                    st.header(
                                                        "Machine Learning Trials")

                                                    trials = get_trial_history()

                                                    if trials:
                                                        # Summary metrics
                                                        col1, col2, col3, col4 = st.columns(
                                                            4)

                                                        with col1:
                                                            st.metric(
                                                                "Total Trials", len(trials))

                                                            with col2:
                                                                successful = sum(
                                                                    1 for t in trials if t.get("status") == "completed")
                                                                st.metric(
                                                                    "Successful", successful)

                                                                with col3:
                                                                    avg_accuracy = sum(t.get("accuracy", 0) for t in trials if t.get(
                                                                        "accuracy")) / max(1, len([t for t in trials if t.get("accuracy")]))
                                                                    st.metric(
                                                                        "Avg Accuracy", f"{avg_accuracy:.2%}")

                                                                    with col4:
                                                                        avg_profit = sum(t.get("profit_loss", 0) for t in trials if t.get(
                                                                            "profit_loss")) / max(1, len([t for t in trials if t.get("profit_loss")]))
                                                                        st.metric(
                                                                            "Avg Profit/Loss", f"{avg_profit:+.2f}%")

                                                                        # Trial
                                                                        # history
                                                                        # table
                                                                        st.subheader(
                                                                            "Recent Trials")

                                                                        # Convert
                                                                        # to
                                                                        # table
                                                                        # format
                                                                        table_data = []
                                                                        # Last
                                                                        # 10
                                                                        # trials
                                                                        for trial in trials[-10:]:
                                                                            table_data.append(
                                                                                {
                                                                                    "Time": trial.get("timestamp", "N/A"),
                                                                                    "Symbol": trial.get("symbol", "N/A"),
                                                                                    "Model": trial.get("model", "N/A"),
                                                                                    "Status": trial.get("status", "N/A"),
                                                                                    "Accuracy": f"{trial.get('accuracy', 0):.2%}",
                                                                                    "P/L": f"{trial.get('profit_loss', 0):+.2f}%",
                                                                                }
                                                                            )

                                                                        st.dataframe(
                                                                            table_data, use_container_width=True)
                                                    else:
                                                        st.info(
                                                            "No trial history available yet")

                                                    # Tab 4: Chat Activity
                                                    with tab4:
                                                        st.header(
                                                            "Chat Activity")

                                                        activity = get_chat_activity()

                                                        # Summary metrics
                                                        col1, col2, col3 = st.columns(
                                                            3)

                                                        with col1:
                                                            st.metric(
                                                                "Total Messages", activity["total_messages"])

                                                            with col2:
                                                                st.metric(
                                                                    "Last 24 Hours", activity["last_24h"])

                                                                with col3:
                                                                    avg_per_hour = activity["last_24h"] / 24
                                                                    st.metric(
                                                                        "Avg per Hour", f"{avg_per_hour:.1f}")

                                                                    # Hourly
                                                                    # activity
                                                                    # chart
                                                                    st.subheader(
                                                                        "Messages by Hour (Last 24h)")

                                                                    hours = list(
                                                                        range(24))
                                                                    counts = [
                                                                        activity["hourly_counts"][h] for h in hours]

                                                                    fig = px.bar(
                                                                        x=hours, y=counts, labels={
                                                                            "x": "Hour", "y": "Messages"}, title="Chat Activity Pattern")

                                                                    fig.update_layout(
                                                                        showlegend=False)
                                                                    st.plotly_chart(
                                                                        fig, use_container_width=True)

                                                    # Tab 5: Market Data
                                                    with tab5:
                                                        st.header(
                                                            "Market Data Overview")

                                                        try:
                                                            polygon = get_polygon_connector()

                                                            # Popular stocks
                                                            symbols = [
                                                                "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

                                                            col_per_row = 3
                                                            rows = (
                                                                len(symbols) + col_per_row - 1) // col_per_row

                                                            for row in range(
                                                                    rows):
                                                                cols = st.columns(
                                                                    col_per_row)
                                                                for col_idx in range(
                                                                        col_per_row):
                                                                    symbol_idx = row * col_per_row + col_idx
                                                                    if symbol_idx < len(
                                                                            symbols):
                                                                        symbol = symbols[symbol_idx]
                                                                        with cols[col_idx]:
                                                                            quote = polygon.get_quote(
                                                                                symbol)
                                                                            if quote:
                                                                                st.metric(
                                                                                    symbol,
                                                                                    f"${quote.price:.2f}",
                                                                                    f"{quote.change:+.2f} ({quote.change_percent:+.2f}%)",
                                                                                    delta_color="normal",
                                                                                )
                                                                            else:
                                                                                st.metric(
                                                                                    symbol, "N/A", "No data")

                                                        except Exception as e:
                                                            st.error(
                                                                f"Error loading market data: {e}")

                                                    # Auto-refresh
                                                    if auto_refresh:
                                                        time.sleep(
                                                            refresh_interval)
                                                        st.rerun()

    if __name__ == "__main__":
        main()
