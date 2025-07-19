"""
mlTrainer - Main Streamlit Application Entry Point
=================================================

Purpose: Main entry point for the mlTrainer trading intelligence system.
This file initializes the multi-page Streamlit application with compliance
enforcement and real-time monitoring capabilities.

Compliance: All data sources are verified and non-synthetic. Any unverified
data triggers "I don't know" responses with data-backed suggestions.
"""

import streamlit as st
import os
import sys
from datetime import datetime
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.compliance_engine import ComplianceEngine
from utils.monitoring import SystemMonitor
from utils.config_manager import ConfigManager

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="mlTrainer - Trading Intelligence System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize core components
@st.cache_resource
def initialize_system():
    """Initialize core system components"""
    try:
        config_manager = ConfigManager()
        compliance_engine = ComplianceEngine()
        system_monitor = SystemMonitor()
        
        return {
            "config": config_manager,
            "compliance": compliance_engine,
            "monitor": system_monitor
        }
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        return None

# Main application styling
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: Georgia, 'Times New Roman', Times, serif !important;
    }
    
    .stApp {
        background-color: #ffffff;
        color: #2c3e50;
    }
    
    .compliance-warning {
        background-color: #dc3545;
        color: white;
        padding: 1rem;
        border-radius: 0.25rem;
        margin-bottom: 1rem;
        text-align: center;
        font-weight: bold;
    }
    
    .compliance-ok {
        background-color: #28a745;
        color: white;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 0.5rem;
    }
    
    .status-green { background-color: #28a745; }
    .status-yellow { background-color: #ffc107; }
    .status-red { background-color: #dc3545; }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application entry point"""
    
    # Initialize system
    system = initialize_system()
    if not system:
        st.error("❌ System initialization failed. Please check configuration.")
        st.stop()
    
    compliance_engine = system["compliance"]
    system_monitor = system["monitor"]
    
    # Initialize compliance state in session
    if "compliance_override" not in st.session_state:
        st.session_state.compliance_override = True
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### ⚙️ System Controls")
        
        # Compliance toggle - only user can change this
        compliance_enabled = st.toggle(
            "🔒 Compliance Mode", 
            value=st.session_state.compliance_override,
            help="Toggle compliance enforcement. Only you can control this setting.",
            key="main_compliance_toggle"
        )
        
        # Update compliance mode if changed
        if compliance_enabled != st.session_state.compliance_override:
            st.session_state.compliance_override = compliance_enabled
            
            import requests
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/api/compliance/toggle",
                    json={"enabled": compliance_enabled},
                    timeout=5
                )
                if response.status_code == 200:
                    st.success(f"✅ Compliance {'enabled' if compliance_enabled else 'disabled'}")
                else:
                    st.error("❌ Failed to update compliance mode")
            except Exception as e:
                st.error(f"❌ Connection error: {e}")
        
        st.divider()
        
        # mlTrainer status
        st.markdown("### 🤖 mlTrainer Status")
        st.markdown("**Status:** 🟢 Online (Exempt from compliance)")
        st.markdown("**Role:** ML Training Coordinator")
        st.markdown("**Access:** Full system capabilities")
    
    # Display compliance banner based on toggle state
    if compliance_enabled:
        st.markdown("""
        <div class="compliance-ok">
            ✅ COMPLIANCE MODE: ON - FULL VERIFICATION ACTIVE
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="compliance-warning">
            ⚠️ COMPLIANCE MODE: OFF - SYSTEM OPERATING WITH FULL ML CAPABILITIES
        </div>
        """, unsafe_allow_html=True)
    
    # Main header
    st.title("📈 mlTrainer - Trading Intelligence System")
    st.markdown("**Multi-Model ML Pipeline · Comprehensive Analytics · Systematic Trading Intelligence**")
    
    # System status overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        monitor_status = system_monitor.get_system_health()
        status_color = "green" if monitor_status["healthy"] else "red"
        st.markdown(f"""
        <div class="metric-card">
            <span class="status-indicator status-{status_color}"></span>
            <strong>System Health:</strong> {"Healthy" if monitor_status["healthy"] else "Issues Detected"}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        data_status = compliance_engine.get_data_source_status()
        active_sources = len([s for s in data_status.values() if s])
        st.markdown(f"""
        <div class="metric-card">
            <span class="status-indicator status-green"></span>
            <strong>Data Sources:</strong> {active_sources}/3 Active
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <span class="status-indicator status-green"></span>
            <strong>Last Update:</strong> {datetime.now().strftime('%H:%M:%S')}
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        compliance_score = 100 if compliance_enabled else 75
        score_color = "green" if compliance_score > 80 else "yellow" if compliance_score > 60 else "red"
        st.markdown(f"""
        <div class="metric-card">
            <span class="status-indicator status-{score_color}"></span>
            <strong>Compliance Score:</strong> {compliance_score}%
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Welcome message and navigation
    st.markdown("""
    ### 🎯 Welcome to mlTrainer
    
    Navigate through the system using the sidebar to access:
    
    - **📊 Recommendations**: Live stock recommendations with real-time scoring
    - **🤖 mlTrainer Chat**: AI-powered trading assistant with verified responses
    - **📈 Analytics**: Regime analysis, model performance, and strategy insights
    - **🔔 Alerts**: Real-time notifications and signal monitoring
    
    #### 🔒 Compliance Commitment
    This system operates with **zero synthetic data**. All recommendations, analysis, 
    and insights are based on verified real-time market data from authorized sources:
    - Polygon API (15-min delayed market data)
    - FRED API (macroeconomic indicators)
    - QuiverQuant API (insider activity and sentiment)
    
    When data cannot be verified, responses follow the format: 
    *"I don't know. But based on the data, I would suggest..."*
    """)
    
    # Quick system diagnostics
    with st.expander("🔧 System Diagnostics", expanded=False):
        st.markdown("### Real-time System Status")
        
        # API connectivity
        st.markdown("**API Connectivity:**")
        api_status = compliance_engine.test_api_connections()
        for api_name, status in api_status.items():
            status_icon = "✅" if status else "❌"
            st.write(f"{status_icon} {api_name}: {'Connected' if status else 'Connection Failed'}")
        
        # Model status
        st.markdown("**ML Models:**")
        try:
            from core.model_manager import ModelManager
            model_manager = ModelManager()
            model_status = model_manager.get_model_status()
            for model_name, status in model_status.items():
                status_icon = "✅" if status.get("loaded") else "❌"
                accuracy = status.get("accuracy", "N/A")
                st.write(f"{status_icon} {model_name}: Loaded, Accuracy: {accuracy}")
        except Exception as e:
            st.warning(f"Model status unavailable: {e}")
        
        # Memory and performance
        st.markdown("**Performance Metrics:**")
        perf_metrics = system_monitor.get_performance_metrics()
        st.write(f"📊 Memory Usage: {perf_metrics.get('memory_percent', 'N/A')}%")
        st.write(f"⚡ CPU Usage: {perf_metrics.get('cpu_percent', 'N/A')}%")
        st.write(f"🔄 Uptime: {perf_metrics.get('uptime', 'N/A')}")

if __name__ == "__main__":
    main()
