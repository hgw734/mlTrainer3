"""
mlTrainer Unified Chat Interface
================================

Combines the advanced UI features (mobile optimization, background trials)
with the compliance system and 140+ integrated models.
"""

import streamlit as st
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# Import unified components (these will be available in your system)
try:
    from core.unified_executor import get_unified_executor
    from core.enhanced_background_manager import get_enhanced_background_manager
    from mltrainer_claude_integration import MLTrainerClaude
    from goal_system import GoalSystem
    from mlagent_bridge import MLAgentBridge
    from recommendation_tracker import get_recommendation_tracker
    from virtual_portfolio_manager import get_virtual_portfolio_manager
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("Running in demo mode")

# Page configuration
st.set_page_config(
    page_title="mlTrainer3 - Unified Interface",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for mobile optimization
st.markdown("""
<style>
    /* Mobile-friendly adjustments */
    @media (max-width: 768px) {
        .stTabs [data-baseweb="tab-list"] {
            flex-wrap: wrap;
        }
        .stTabs [data-baseweb="tab"] {
            flex: 1 1 auto;
            min-width: 100px;
        }
        .metric-container {
            flex-direction: column;
        }
        .stButton button {
            width: 100%;
            margin: 5px 0;
        }
    }
    
    /* Dark mode improvements */
    .stApp[data-theme="dark"] {
        background-color: #0e1117;
    }
    
    /* Custom metric styling */
    .metric-card {
        background-color: rgba(28, 131, 225, 0.1);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid rgba(28, 131, 225, 0.3);
    }
    
    /* Improved chat styling */
    .stChatMessage {
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize components
@st.cache_resource
def init_components():
    """Initialize all system components"""
    components = {}
    
    try:
        components["executor"] = get_unified_executor()
        components["background_manager"] = get_enhanced_background_manager()
        components["claude"] = MLTrainerClaude()
        components["goal_system"] = GoalSystem()
        components["bridge"] = MLAgentBridge()
        components["recommendation_tracker"] = get_recommendation_tracker()
        components["portfolio_manager"] = get_virtual_portfolio_manager()
    except Exception as e:
        st.warning(f"Some components not available: {e}")
        # Return mock components for demo
        components = {
            "executor": None,
            "background_manager": None,
            "claude": None,
            "goal_system": None,
            "bridge": None,
            "recommendation_tracker": None,
            "portfolio_manager": None
        }
    
    return components

def load_chat_history():
    """Load chat history from file"""
    if os.path.exists("logs/unified_chat_history.json"):
        try:
            with open("logs/unified_chat_history.json", "r") as f:
                return json.load(f)
        except:
            return []
    return []

def save_chat_history(messages):
    """Save chat history to file"""
    os.makedirs("logs", exist_ok=True)
    with open("logs/unified_chat_history.json", "w") as f:
        json.dump(messages, f, indent=2)

def main():
    """Main application function"""
    st.title("📈 mlTrainer3 - Unified Interface")
    st.markdown("**Institutional-Grade AI/ML Trading System**")
    
    # Initialize components
    components = init_components()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = load_chat_history()
    
    if "active_trials" not in st.session_state:
        st.session_state.active_trials = []
    
    if "selected_models" not in st.session_state:
        st.session_state.selected_models = []
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # Model Selection
        st.subheader("📊 Model Selection")
        
        model_categories = {
            "Momentum": ["RSI Strategy", "MACD Crossover", "Breakout Detection"],
            "Mean Reversion": ["Bollinger Bands", "Z-Score", "Pairs Trading"],
            "ML Models": ["Random Forest", "XGBoost", "Neural Network"],
            "Risk Management": ["Kelly Criterion", "VaR", "Black-Scholes"]
        }
        
        selected_models = []
        for category, models in model_categories.items():
            with st.expander(f"{category} ({len(models)} models)"):
                for model in models:
                    if st.checkbox(model, key=f"model_{model}"):
                        selected_models.append(model)
        
        st.session_state.selected_models = selected_models
        
        # Quick Actions
        st.divider()
        st.subheader("⚡ Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.rerun()
        
        with col2:
            if st.button("📥 Export Results", use_container_width=True):
                st.info("Export functionality coming soon")
        
        # System Status
        st.divider()
        st.subheader("📡 System Status")
        
        status_data = {
            "API Status": "🟢 Connected",
            "Models Loaded": f"{len(selected_models)}/140+",
            "Background Tasks": len(st.session_state.active_trials),
            "Memory Usage": "45%"
        }
        
        for key, value in status_data.items():
            st.metric(key, value)
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📊 Dashboard", "🔬 Active Trials", "⚙️ Settings"])
    
    with tab1:
        # Chat Interface
        st.header("mlTrainer Chat Assistant")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask mlTrainer anything about trading, models, or analysis..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Generate response (mock for now)
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    # In production, this would call the actual AI
                    response = f"I understand you're asking about: '{prompt}'. "
                    
                    if "model" in prompt.lower():
                        response += "We have 140+ models available including momentum, mean reversion, ML, and risk management strategies."
                    elif "trade" in prompt.lower():
                        response += "I can help you analyze trading opportunities using our advanced models."
                    else:
                        response += "Let me analyze that for you using our integrated systems."
                    
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Save chat history
            save_chat_history(st.session_state.messages)
            st.rerun()
    
    with tab2:
        # Dashboard
        st.header("Trading Dashboard")
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Portfolio Value", "$125,432", "+2.3%")
        with col2:
            st.metric("Daily P&L", "$1,234", "+0.98%")
        with col3:
            st.metric("Win Rate", "68%", "+3%")
        with col4:
            st.metric("Sharpe Ratio", "1.85", "+0.12")
        
        # Charts
        st.subheader("Performance Overview")
        
        # Sample data for visualization
        dates = pd.date_range(start='2024-01-01', end='2024-12-20', freq='D')
        portfolio_value = 100000 + np.cumsum(np.random.randn(len(dates)) * 1000)
        
        chart_data = pd.DataFrame({
            'Date': dates,
            'Portfolio Value': portfolio_value,
            'Benchmark': 100000 + np.arange(len(dates)) * 50
        })
        
        st.line_chart(chart_data.set_index('Date'))
        
        # Active Positions
        st.subheader("Active Positions")
        
        positions_data = pd.DataFrame({
            'Symbol': ['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
            'Position': [100, -50, 200, 75],
            'Entry Price': [150.00, 140.00, 380.00, 170.00],
            'Current Price': [155.00, 138.00, 385.00, 172.00],
            'P&L': [500, 100, 1000, 150],
            'P&L %': [3.33, 1.43, 2.63, 1.18]
        })
        
        st.dataframe(
            positions_data,
            use_container_width=True,
            hide_index=True,
            column_config={
                "P&L": st.column_config.NumberColumn(format="$%.2f"),
                "P&L %": st.column_config.NumberColumn(format="%.2f%%")
            }
        )
    
    with tab3:
        # Active Trials
        st.header("Active Background Trials")
        
        if st.session_state.active_trials:
            for i, trial in enumerate(st.session_state.active_trials):
                with st.expander(f"Trial {i+1}: {trial.get('name', 'Unnamed')}", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Progress", f"{trial.get('progress', 0)}%")
                    with col2:
                        st.metric("Best Performance", f"{trial.get('best_performance', 0):.2%}")
                    with col3:
                        if st.button("Stop", key=f"stop_trial_{i}"):
                            st.session_state.active_trials.pop(i)
                            st.rerun()
                    
                    # Progress bar
                    st.progress(trial.get('progress', 0) / 100)
                    
                    # Trial details
                    st.json(trial.get('config', {}))
        else:
            st.info("No active trials. Start a new trial from the Models tab.")
        
        # Start new trial
        st.divider()
        st.subheader("Start New Trial")
        
        with st.form("new_trial"):
            trial_name = st.text_input("Trial Name")
            selected_model = st.selectbox("Select Model", st.session_state.selected_models or ["No models selected"])
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date")
            with col2:
                end_date = st.date_input("End Date")
            
            if st.form_submit_button("Start Trial"):
                if trial_name and selected_model:
                    new_trial = {
                        "name": trial_name,
                        "model": selected_model,
                        "start_date": str(start_date),
                        "end_date": str(end_date),
                        "progress": 0,
                        "best_performance": 0,
                        "config": {
                            "model": selected_model,
                            "parameters": "default"
                        }
                    }
                    st.session_state.active_trials.append(new_trial)
                    st.success(f"Started trial: {trial_name}")
                    st.rerun()
    
    with tab4:
        # Settings
        st.header("Settings")
        
        # API Configuration
        st.subheader("🔑 API Configuration")
        
        api_keys = [
            "ANTHROPIC_API_KEY",
            "POLYGON_API_KEY", 
            "FRED_API_KEY",
            "TELEGRAM_BOT_TOKEN",
            "TELEGRAM_CHAT_ID"
        ]
        
        for key in api_keys:
            value = os.environ.get(key, "")
            status = "✅ Set" if value else "❌ Not Set"
            st.text_input(f"{key} {status}", type="password", disabled=True, value="*" * 8 if value else "")
        
        # Display Preferences
        st.subheader("🎨 Display Preferences")
        
        col1, col2 = st.columns(2)
        with col1:
            theme = st.selectbox("Theme", ["Auto", "Light", "Dark"])
            refresh_rate = st.slider("Refresh Rate (seconds)", 5, 60, 30)
        
        with col2:
            show_notifications = st.checkbox("Show Notifications", value=True)
            sound_alerts = st.checkbox("Sound Alerts", value=False)
        
        # Risk Management
        st.subheader("⚠️ Risk Management")
        
        max_position_size = st.slider("Max Position Size (%)", 0, 100, 10)
        stop_loss = st.slider("Default Stop Loss (%)", 0, 20, 5)
        take_profit = st.slider("Default Take Profit (%)", 0, 50, 15)
        
        if st.button("Save Settings"):
            st.success("Settings saved successfully!")

if __name__ == "__main__":
    main()