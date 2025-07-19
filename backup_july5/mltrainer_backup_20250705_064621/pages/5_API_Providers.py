"""
mlTrainer - API Provider Management
===================================

Purpose: Admin page for managing API providers, switching between different
AI models and data sources, and viewing provider status.

Features:
- View current API provider configuration
- Switch between AI providers (Anthropic, etc.)
- Switch between data providers (Polygon, FRED)
- Monitor API key status and health
- Real-time configuration updates
"""

import streamlit as st
import requests
import json
from datetime import datetime
from typing import Dict, Any, Optional

# Page configuration
st.set_page_config(
    page_title="mlTrainer - API Providers",
    page_icon="🔧",
    layout="wide"
)

# Backend URL
BACKEND_URL = "http://127.0.0.1:8000"

def get_provider_status() -> Optional[Dict[str, Any]]:
    """Get current API provider status from backend"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/providers/status", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to get provider status: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error connecting to backend: {str(e)}")
        return None

def get_available_providers() -> Optional[Dict[str, Any]]:
    """Get list of all available providers"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/providers/available", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to get available providers: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error getting available providers: {str(e)}")
        return None

def switch_ai_provider(provider_id: str) -> bool:
    """Switch to a different AI provider"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/providers/ai/switch",
            json={"provider_id": provider_id},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            st.success(f"✅ {result.get('message', 'AI provider switched successfully')}")
            return True
        else:
            error_data = response.json()
            st.error(f"❌ Failed to switch AI provider: {error_data.get('message', 'Unknown error')}")
            return False
    except Exception as e:
        st.error(f"❌ Error switching AI provider: {str(e)}")
        return False

def switch_data_provider(data_type: str, provider_id: str) -> bool:
    """Switch to a different data provider"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/providers/data/switch",
            json={"data_type": data_type, "provider_id": provider_id},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            st.success(f"✅ {result.get('message', 'Data provider switched successfully')}")
            return True
        else:
            error_data = response.json()
            st.error(f"❌ Failed to switch data provider: {error_data.get('message', 'Unknown error')}")
            return False
    except Exception as e:
        st.error(f"❌ Error switching data provider: {str(e)}")
        return False

def main():
    """Main API provider management interface"""
    
    # Header
    st.title("🔧 API Provider Management")
    st.markdown("---")
    
    # Auto-refresh controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("**Real-time provider configuration and status monitoring**")
    with col2:
        auto_refresh = st.checkbox("Auto-refresh", value=True)
    with col3:
        if st.button("🔄 Refresh Now"):
            st.rerun()
    
    # Auto-refresh timer
    if auto_refresh:
        # Refresh every 30 seconds
        st.empty()
        
    # Get current status
    provider_status = get_provider_status()
    available_providers = get_available_providers()
    
    if not provider_status or not available_providers:
        st.error("Unable to connect to backend API. Please ensure the Flask backend is running on port 8000.")
        st.info("Check the backend logs for connection issues.")
        return
    
    # Current Status Overview
    st.subheader("📊 Current Configuration")
    
    # Status cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "System Status",
            provider_status.get("status", "Unknown").title(),
            delta="Healthy" if provider_status.get("status") == "healthy" else "Issues"
        )
    
    with col2:
        api_keys = provider_status.get("api_keys", {})
        available_count = api_keys.get("available_count", 0)
        st.metric(
            "API Keys",
            f"{available_count} Available",
            delta="✓ Configured" if available_count > 0 else "⚠ Missing"
        )
    
    with col3:
        ai_provider = provider_status.get("active_providers", {}).get("ai", {})
        st.metric(
            "AI Provider",
            ai_provider.get("name", "None"),
            delta=ai_provider.get("model", "No model")
        )
    
    with col4:
        market_provider = provider_status.get("active_providers", {}).get("market_data", {})
        st.metric(
            "Market Data",
            market_provider.get("name", "None"),
            delta="✓ Active" if market_provider.get("enabled") else "⚠ Inactive"
        )
    
    st.markdown("---")
    
    # Provider Management Sections
    tab1, tab2, tab3, tab4 = st.tabs(["🤖 AI Providers", "📈 Data Providers", "🔑 API Keys", "📋 Configuration"])
    
    # AI Providers Tab
    with tab1:
        st.subheader("AI Model Management")
        
        current_ai = provider_status.get("active_providers", {}).get("ai", {})
        ai_providers = available_providers.get("ai_providers", [])
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Current AI Provider:**")
            if current_ai.get("name"):
                st.info(f"**{current_ai['name']}**\n\nModel: {current_ai.get('model', 'Unknown')}\nStatus: {'✅ Active' if current_ai.get('enabled') else '❌ Inactive'}")
            else:
                st.warning("No AI provider active")
        
        with col2:
            st.markdown("**Switch AI Provider:**")
            
            if ai_providers:
                # Create selection options
                provider_options = {}
                for provider in ai_providers:
                    status = "🟢" if provider["enabled"] else "🔴"
                    provider_options[f"{status} {provider['name']}"] = provider["id"]
                
                selected_display = st.selectbox(
                    "Choose AI Provider:",
                    options=list(provider_options.keys()),
                    index=0
                )
                
                selected_provider_id = provider_options[selected_display]
                
                if st.button("🔄 Switch AI Provider", type="primary"):
                    if switch_ai_provider(selected_provider_id):
                        st.rerun()
            else:
                st.warning("No AI providers available")
    
    # Data Providers Tab
    with tab2:
        st.subheader("Data Source Management")
        
        current_providers = provider_status.get("active_providers", {})
        
        # Market Data Providers
        st.markdown("**Market Data Providers**")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            market_provider = current_providers.get("market_data", {})
            if market_provider.get("name"):
                st.info(f"**Current:** {market_provider['name']}\nStatus: {'✅ Active' if market_provider.get('enabled') else '❌ Inactive'}")
            else:
                st.warning("No market data provider active")
        
        with col2:
            market_providers = available_providers.get("market_data_providers", [])
            if market_providers:
                market_options = {}
                for provider in market_providers:
                    status = "🟢" if provider["enabled"] else "🔴"
                    market_options[f"{status} {provider['name']}"] = provider["id"]
                
                selected_market = st.selectbox(
                    "Choose Market Data Provider:",
                    options=list(market_options.keys()),
                    key="market_provider"
                )
                
                if st.button("🔄 Switch Market Provider", key="switch_market"):
                    provider_id = market_options[selected_market]
                    if switch_data_provider("market_data", provider_id):
                        st.rerun()
        
        st.markdown("---")
        
        # Economic Data Providers
        st.markdown("**Economic Data Providers**")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            economic_provider = current_providers.get("economic_data", {})
            if economic_provider.get("name"):
                st.info(f"**Current:** {economic_provider['name']}\nStatus: {'✅ Active' if economic_provider.get('enabled') else '❌ Inactive'}")
            else:
                st.warning("No economic data provider active")
        
        with col2:
            economic_providers = available_providers.get("economic_data_providers", [])
            if economic_providers:
                economic_options = {}
                for provider in economic_providers:
                    status = "🟢" if provider["enabled"] else "🔴"
                    economic_options[f"{status} {provider['name']}"] = provider["id"]
                
                selected_economic = st.selectbox(
                    "Choose Economic Data Provider:",
                    options=list(economic_options.keys()),
                    key="economic_provider"
                )
                
                if st.button("🔄 Switch Economic Provider", key="switch_economic"):
                    provider_id = economic_options[selected_economic]
                    if switch_data_provider("economic_data", provider_id):
                        st.rerun()
    
    # API Keys Tab
    with tab3:
        st.subheader("API Key Status")
        
        api_validation = provider_status.get("api_keys", {}).get("validation", {})
        
        if api_validation:
            for key_name, is_valid in api_validation.items():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(key_name)
                with col2:
                    if is_valid:
                        st.success("✅ Valid")
                    else:
                        st.error("❌ Missing")
        else:
            st.warning("No API key validation data available")
        
        st.markdown("---")
        st.info("💡 **Tip:** API keys are loaded from environment variables. Make sure all required keys are set before starting the backend.")
    
    # Configuration Tab
    with tab4:
        st.subheader("System Configuration")
        
        # Display raw configuration (for debugging)
        with st.expander("📋 Raw Configuration Data"):
            st.json(provider_status)
        
        # Configuration summary
        config_summary = provider_status.get("configuration", {})
        if config_summary:
            st.markdown("**Configuration Summary:**")
            for key, value in config_summary.items():
                st.text(f"{key}: {value}")
        
        # Last updated
        timestamp = provider_status.get("timestamp", "Unknown")
        if timestamp != "Unknown":
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                st.text(f"Last Updated: {dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            except:
                st.text(f"Last Updated: {timestamp}")

if __name__ == "__main__":
    main()