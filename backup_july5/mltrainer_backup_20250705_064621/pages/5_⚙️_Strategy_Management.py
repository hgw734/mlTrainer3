"""
Strategy Management Page - mlTrainer Overriding Goal Configuration
================================================================

Purpose: Allow user to view and modify mlTrainer's primary objective
and strategy parameters in real-time through the technical facilitator.
"""

import streamlit as st
import requests
import json
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Strategy Management - mlTrainer",
    page_icon="⚙️",
    layout="wide"
)

# Custom CSS for strategy management interface
st.markdown("""
<style>
.strategy-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 10px;
    color: white;
    margin-bottom: 2rem;
}

.objective-box {
    background: rgba(255, 255, 255, 0.1);
    padding: 1.5rem;
    border-radius: 8px;
    margin: 1rem 0;
    border-left: 4px solid #4CAF50;
}

.timeframe-card {
    background: rgba(255, 255, 255, 0.05);
    padding: 1rem;
    border-radius: 6px;
    margin: 0.5rem 0;
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.status-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 8px;
}

.status-active { background-color: #4CAF50; }
.status-inactive { background-color: #f44336; }
.status-warning { background-color: #ff9800; }
</style>
""", unsafe_allow_html=True)

def get_backend_url():
    """Get backend URL for API calls"""
    return "http://localhost:8000"

def load_current_objective():
    """Load current mlTrainer objective from API"""
    try:
        response = requests.get(f"{get_backend_url()}/api/facilitator/primary-objective")
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Error loading objective: {e}")
        return None

def update_objective(new_objective_text):
    """Update mlTrainer objective through technical facilitator"""
    try:
        # Parse the new objective and create updated configuration
        config_path = "config/mltrainer_primary_objective.json"
        
        # Load existing config
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            # Create default structure if file doesn't exist
            config = {
                "mltrainer_primary_objective": {
                    "version": "1.0",
                    "last_updated": datetime.now().isoformat(),
                    "description": "Core objective configuration for mlTrainer"
                }
            }
        
        # Update the overriding goal
        config["mltrainer_primary_objective"]["overriding_goal"] = new_objective_text
        config["mltrainer_primary_objective"]["last_updated"] = datetime.now().isoformat()
        config["mltrainer_primary_objective"]["user_modified"] = True
        
        # Save updated configuration
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return True
        
    except Exception as e:
        st.error(f"Error updating objective: {e}")
        return False

def display_current_strategy():
    """Display current strategy configuration"""
    objective_data = load_current_objective()
    
    if objective_data and objective_data.get('success'):
        config = objective_data['primary_objective']['mltrainer_primary_objective']
        
        st.markdown('<div class="strategy-container">', unsafe_allow_html=True)
        st.markdown("### 🎯 Current mlTrainer Strategy")
        
        # Current overriding goal
        current_goal = config.get('overriding_goal', 'No goal defined')
        st.markdown(f'<div class="objective-box"><strong>Overriding Goal:</strong><br>{current_goal}</div>', 
                   unsafe_allow_html=True)
        
        # Timeframe specifications if available
        if 'momentum_identification_framework' in config:
            framework = config['momentum_identification_framework']
            if 'timeframe_specifications' in framework:
                st.markdown("**Performance Targets:**")
                timeframes = framework['timeframe_specifications']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if 'short_term' in timeframes:
                        st.markdown(f"""
                        <div class="timeframe-card">
                            <strong>Short-term</strong><br>
                            Duration: {timeframes['short_term'].get('duration', 'N/A')}<br>
                            Target: {timeframes['short_term'].get('minimum_target', 'N/A')}
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    if 'medium_term' in timeframes:
                        st.markdown(f"""
                        <div class="timeframe-card">
                            <strong>Medium-term</strong><br>
                            Duration: {timeframes['medium_term'].get('duration', 'N/A')}<br>
                            Target: {timeframes['medium_term'].get('minimum_target', 'N/A')}
                        </div>
                        """, unsafe_allow_html=True)
                
                with col3:
                    if 'long_term' in timeframes:
                        st.markdown(f"""
                        <div class="timeframe-card">
                            <strong>Long-term</strong><br>
                            Duration: {timeframes['long_term'].get('duration', 'N/A')}<br>
                            Target: {timeframes['long_term'].get('minimum_target', 'N/A')}
                        </div>
                        """, unsafe_allow_html=True)
        
        # Last updated
        last_updated = config.get('last_updated', 'Unknown')
        st.markdown(f"*Last updated: {last_updated}*")
        st.markdown('</div>', unsafe_allow_html=True)
        
        return current_goal
    else:
        st.warning("Could not load current strategy configuration")
        return "identify momentum stocks within short medium and long timeframes (7-10 days, up to 3 months, up to 9 months) that have a very high probability of reaching a calculated price within the estimated timeframe. We are looking for high performers: a minimum + 7 % price increase minimum short term, a minimum + 25% increase mid term, a minimum 75% increase long term."

def display_system_status():
    """Display system status indicators"""
    try:
        # Check backend health
        health_response = requests.get(f"{get_backend_url()}/health", timeout=5)
        backend_status = health_response.status_code == 200
        
        # Check technical facilitator
        facilitator_response = requests.get(f"{get_backend_url()}/api/facilitator/system-status", timeout=5)
        facilitator_status = facilitator_response.status_code == 200
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_class = "status-active" if backend_status else "status-inactive"
            st.markdown(f'<span class="status-indicator {status_class}"></span>Backend API', 
                       unsafe_allow_html=True)
        
        with col2:
            status_class = "status-active" if facilitator_status else "status-inactive"
            st.markdown(f'<span class="status-indicator {status_class}"></span>Technical Facilitator', 
                       unsafe_allow_html=True)
        
        with col3:
            # Always show mlTrainer as active since it's accessed through chat
            st.markdown('<span class="status-indicator status-active"></span>mlTrainer Access', 
                       unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Status check failed: {e}")

# Main page layout
st.title("⚙️ Strategy Management")
st.markdown("Configure and monitor mlTrainer's overriding objective and strategy parameters")

# System status section
st.markdown("### 🔧 System Status")
display_system_status()

st.markdown("---")

# Current strategy display
current_goal = display_current_strategy()

st.markdown("---")

# Strategy modification section
st.markdown("### ✏️ Modify Strategy")
st.markdown("Update mlTrainer's overriding goal. Changes are applied immediately to the technical facilitator.")

# Text area for strategy input
new_objective = st.text_area(
    "Overriding Goal for mlTrainer:",
    value=current_goal,
    height=150,
    help="Define mlTrainer's primary objective. This will be implemented immediately through the technical facilitator."
)

# Update buttons
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("Update Strategy", type="primary"):
        if new_objective.strip():
            if update_objective(new_objective.strip()):
                st.success("✅ Strategy updated successfully!")
                st.rerun()
            else:
                st.error("❌ Failed to update strategy")
        else:
            st.warning("Please enter a valid strategy objective")

with col2:
    if st.button("Reset to Default"):
        default_objective = "identify momentum stocks within short medium and long timeframes (7-10 days, up to 3 months, up to 9 months) that have a very high probability of reaching a calculated price within the estimated timeframe. We are looking for high performers: a minimum + 7 % price increase minimum short term, a minimum + 25% increase mid term, a minimum 75% increase long term."
        if update_objective(default_objective):
            st.success("✅ Strategy reset to default!")
            st.rerun()
        else:
            st.error("❌ Failed to reset strategy")

with col3:
    st.markdown("*Changes are applied immediately to mlTrainer through the technical facilitator*")

# Strategy validation section
st.markdown("---")
st.markdown("### 🔍 Strategy Validation")

if st.button("Validate Current Strategy"):
    try:
        # Test API endpoint access
        response = requests.get(f"{get_backend_url()}/api/facilitator/primary-objective")
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                st.success("✅ Strategy configuration is valid and accessible")
                st.json(data['primary_objective'])
            else:
                st.error("❌ Strategy configuration validation failed")
        else:
            st.error(f"❌ API validation failed: {response.status_code}")
    except Exception as e:
        st.error(f"❌ Validation error: {e}")

# Information section
st.markdown("---")
st.markdown("### ℹ️ How It Works")
st.markdown("""
1. **Strategy Definition**: Enter mlTrainer's overriding goal in plain language
2. **Immediate Implementation**: Changes are saved to the technical facilitator configuration
3. **API Access**: mlTrainer accesses the strategy through `/api/facilitator/primary-objective`
4. **Autonomous Execution**: mlTrainer implements the strategy independently through walk-forward testing
5. **Role Separation**: You define objectives, mlTrainer develops and executes strategies
""")

# Footer
st.markdown("---")
st.markdown("*Strategy Management Interface - Technical Facilitator System*")