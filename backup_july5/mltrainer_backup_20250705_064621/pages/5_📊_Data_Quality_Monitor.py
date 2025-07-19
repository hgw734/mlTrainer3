"""
Data Quality Monitoring Dashboard
=================================

Real-time monitoring of data quality metrics, API rate limits, and dropout rates
for the mlTrainer system. Provides comprehensive validation before ML trial execution.
"""

import streamlit as st
import requests
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import time

# Configure page
st.set_page_config(
    page_title="Data Quality Monitor",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for data quality dashboard
st.markdown("""
<style>
.metric-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    margin: 0.5rem 0;
}

.quality-good {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
}

.quality-warning {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
}

.quality-critical {
    background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
}

.api-status {
    padding: 0.5rem;
    border-radius: 5px;
    margin: 0.25rem;
    text-align: center;
    font-weight: bold;
}

.status-ok { background-color: #d4edda; color: #155724; }
.status-warning { background-color: #fff3cd; color: #856404; }
.status-error { background-color: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

def get_data_quality_metrics():
    """Fetch data quality metrics from API"""
    try:
        response = requests.get("http://localhost:8000/api/data-quality", timeout=5)
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error: {response.status_code}"
    except Exception as e:
        return None, f"Connection Error: {str(e)}"

def validate_trial_data(symbols=None):
    """Validate data quality for trial execution"""
    if symbols is None:
        symbols = ['SPY']
    
    try:
        response = requests.post(
            "http://localhost:8000/api/data-quality/trial-validation",
            json={"symbols": symbols},
            timeout=10
        )
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error: {response.status_code}"
    except Exception as e:
        return None, f"Connection Error: {str(e)}"

def reset_polygon_metrics():
    """Reset Polygon API metrics"""
    try:
        response = requests.post("http://localhost:8000/api/data-quality/polygon/reset", timeout=5)
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error: {response.status_code}"
    except Exception as e:
        return None, f"Connection Error: {str(e)}"

def create_quality_gauge(value, title, thresholds=None):
    """Create a quality gauge chart"""
    if thresholds is None:
        thresholds = [0.8, 0.9]  # Warning below 80%, Good above 90%
    
    color = "red" if value < thresholds[0] else "yellow" if value < thresholds[1] else "green"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': thresholds[1]},
        gauge = {
            'axis': {'range': [None, 1]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, thresholds[0]], 'color': "lightgray"},
                {'range': [thresholds[0], thresholds[1]], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': thresholds[0]
            }
        }
    ))
    
    fig.update_layout(height=250)
    return fig

def main():
    st.title("📊 Data Quality Monitor")
    st.markdown("Real-time monitoring of API rate limits, data quality metrics, and trial validation")
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=True)
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()
    
    # Reset metrics button
    if st.sidebar.button("🔄 Reset Polygon Metrics"):
        result, error = reset_polygon_metrics()
        if result:
            st.sidebar.success("✅ Polygon metrics reset successfully")
        else:
            st.sidebar.error(f"❌ Reset failed: {error}")
    
    # Get data quality metrics
    data_quality, error = get_data_quality_metrics()
    
    if error:
        st.error(f"❌ **System Connection Error**: {error}")
        st.info("💡 Make sure the Flask backend is running on port 8000")
        return
    
    if not data_quality:
        st.warning("⚠️ No data quality metrics available")
        return
    
    # Overall system status
    overall_status = data_quality.get("overall_status", "unknown")
    status_color = "green" if overall_status == "healthy" else "red"
    
    st.markdown(f"""
    <div class="metric-container quality-{'good' if overall_status == 'healthy' else 'critical'}">
        <h2>🏥 System Status: {overall_status.upper()}</h2>
        <p>Last updated: {data_quality.get('timestamp', 'Unknown')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    polygon_data = data_quality.get("polygon_api", {})
    quality_summary = polygon_data.get("quality_summary", {})
    
    with col1:
        success_rate = quality_summary.get("success_rate", 0)
        st.metric("API Success Rate", f"{success_rate:.1%}", 
                 delta=f"{success_rate - 0.95:.1%}" if success_rate > 0 else None)
    
    with col2:
        dropout_rate = quality_summary.get("dropout_rate", 0)
        st.metric("Dropout Rate", f"{dropout_rate:.1%}", 
                 delta=f"{dropout_rate - 0.05:.1%}" if dropout_rate > 0 else None,
                 delta_color="inverse")
    
    with col3:
        avg_response_time = quality_summary.get("avg_response_time", 0)
        st.metric("Avg Response Time", f"{avg_response_time:.2f}s", 
                 delta=f"{avg_response_time - 2.0:.2f}s" if avg_response_time > 0 else None,
                 delta_color="inverse")
    
    with col4:
        total_requests = quality_summary.get("total_requests", 0)
        st.metric("Total API Requests", f"{total_requests:,}")
    
    # Quality gauges
    st.subheader("📊 Quality Metrics")
    
    gauge_col1, gauge_col2, gauge_col3 = st.columns(3)
    
    with gauge_col1:
        success_gauge = create_quality_gauge(success_rate, "Success Rate", [0.85, 0.95])
        st.plotly_chart(success_gauge, use_container_width=True)
    
    with gauge_col2:
        completeness_rate = 1.0 - dropout_rate
        completeness_gauge = create_quality_gauge(completeness_rate, "Data Completeness", [0.80, 0.90])
        st.plotly_chart(completeness_gauge, use_container_width=True)
    
    with gauge_col3:
        # Response time quality (inverted - lower is better)
        response_quality = max(0, 1 - (avg_response_time / 10))  # 10s = 0 quality
        response_gauge = create_quality_gauge(response_quality, "Response Quality", [0.5, 0.8])
        st.plotly_chart(response_gauge, use_container_width=True)
    
    # API Status Section
    st.subheader("🔌 API Connection Status")
    
    # Polygon API status
    polygon_status = polygon_data.get("is_valid", False)
    rate_limit_status = polygon_data.get("rate_limit_status", "unknown")
    circuit_open = quality_summary.get("circuit_open", False)
    
    status_col1, status_col2 = st.columns(2)
    
    with status_col1:
        st.markdown("**Polygon API**")
        status_class = "status-ok" if polygon_status else "status-error"
        status_text = "✅ Operational" if polygon_status else "❌ Issues Detected"
        st.markdown(f'<div class="api-status {status_class}">{status_text}</div>', unsafe_allow_html=True)
        
        if circuit_open:
            st.error("⚠️ Circuit Breaker OPEN - API calls temporarily suspended")
        
        # Rate limit status
        rate_limit_class = "status-ok" if rate_limit_status == "ok" else "status-warning"
        rate_limit_text = "🟢 Rate Limit OK" if rate_limit_status == "ok" else "🟡 Approaching Limit"
        st.markdown(f'<div class="api-status {rate_limit_class}">{rate_limit_text}</div>', unsafe_allow_html=True)
    
    with status_col2:
        # Other API connections
        api_connections = data_quality.get("api_connections", {})
        if api_connections:
            st.markdown("**Other APIs**")
            for api_name, status in api_connections.items():
                if api_name == "error":
                    continue
                status_class = "status-ok" if status else "status-error"
                status_text = f"✅ {api_name.upper()}" if status else f"❌ {api_name.upper()}"
                st.markdown(f'<div class="api-status {status_class}">{status_text}</div>', unsafe_allow_html=True)
    
    # Trial Validation Section
    st.subheader("🧪 Trial Data Validation")
    
    # Symbol selection for validation
    test_symbols = st.multiselect(
        "Select symbols to validate",
        options=["SPY", "QQQ", "TSLA", "AAPL", "MSFT", "GOOGL", "NVDA"],
        default=["SPY"]
    )
    
    if st.button("🔍 Validate Trial Data"):
        if test_symbols:
            with st.spinner("Validating data quality for trial execution..."):
                validation_result, validation_error = validate_trial_data(test_symbols)
            
            if validation_error:
                st.error(f"❌ Validation failed: {validation_error}")
            elif validation_result:
                trial_approved = validation_result.get("trial_approved", False)
                
                if trial_approved:
                    st.success("✅ **Trial Data Validation PASSED**")
                    st.info("🚀 Data quality is sufficient for ML trial execution")
                else:
                    st.error("❌ **Trial Data Validation FAILED**")
                    reason = validation_result.get("reason", "Unknown reason")
                    st.warning(f"⚠️ {reason}")
                    
                    recommendation = validation_result.get("recommendation")
                    if recommendation:
                        st.info(f"💡 **Recommendation**: {recommendation}")
                
                # Show detailed validation results
                with st.expander("📋 Detailed Validation Results"):
                    validation_results = validation_result.get("validation_results", {})
                    for symbol, result in validation_results.items():
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            status = "✅" if result.get("is_valid", False) else "❌"
                            st.write(f"{status} **{symbol}**")
                        with col2:
                            if result.get("is_valid", False):
                                st.write("Data quality OK")
                            else:
                                issues = result.get("issues", ["Unknown issue"])
                                st.write(f"Issues: {', '.join(issues)}")
    
    # Quality History (if available)
    st.subheader("📈 Quality Trends")
    
    # Create sample trend data (in real implementation, this would be stored)
    hours = list(range(24))
    success_rates = [0.95 + (i % 3) * 0.02 for i in hours]
    dropout_rates = [0.05 - (i % 4) * 0.01 for i in hours]
    
    trend_data = pd.DataFrame({
        'Hour': hours,
        'Success Rate': success_rates,
        'Dropout Rate': dropout_rates
    })
    
    fig = px.line(trend_data, x='Hour', y=['Success Rate', 'Dropout Rate'], 
                  title='24-Hour Quality Trends')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Auto-refresh implementation
    if auto_refresh:
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()