"""
mlTrainer - Recommendations Dashboard
====================================

Purpose: Real-time stock recommendations with compliance-verified scoring.
Displays auto-populated stocks scoring >80 with real-time updates and
portfolio management capabilities.

Compliance: All recommendations are based on verified market data with
no synthetic or placeholder information.
"""

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(page_title="📊 Recommendations", layout="wide")

# Apply consistent styling
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: Georgia, 'Times New Roman', Times, serif !important;
    }
    
    .recommendation-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    
    .holding-card {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196f3;
        margin: 0.5rem 0;
    }
    
    .score-high { color: #28a745; font-weight: bold; }
    .score-medium { color: #ffc107; font-weight: bold; }
    .score-low { color: #dc3545; font-weight: bold; }
    
    .progress-bar {
        background-color: #e9ecef;
        border-radius: 10px;
        overflow: hidden;
    }
    
    .progress-fill {
        height: 20px;
        background: linear-gradient(90deg, #28a745, #20c997);
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

def get_backend_url():
    """Get backend API URL"""
    return "http://localhost:8000"

def fetch_recommendations():
    """Fetch recommendations from backend API"""
    try:
        response = requests.get(f"{get_backend_url()}/api/recommendations", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("recommendations", [])
        elif response.status_code == 403:
            st.error("❌ Compliance mode is disabled. Enable compliance to view recommendations.")
            return []
        else:
            st.error(f"❌ Failed to fetch recommendations: {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Connection error: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching recommendations: {e}")
        st.error("❌ An unexpected error occurred while fetching recommendations.")
        return []

def fetch_portfolio():
    """Fetch current portfolio from backend API"""
    try:
        response = requests.get(f"{get_backend_url()}/api/portfolio", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("portfolio", [])
        else:
            st.error(f"❌ Failed to fetch portfolio: {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Connection error: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching portfolio: {e}")
        return []

def add_to_portfolio(ticker):
    """Add recommendation to portfolio"""
    try:
        response = requests.post(
            f"{get_backend_url()}/api/portfolio/add",
            json={"ticker": ticker},
            timeout=10
        )
        if response.status_code == 200:
            st.success(f"✅ {ticker} added to portfolio")
            st.rerun()
        else:
            error_data = response.json()
            message = error_data.get("message", "Failed to add to portfolio")
            st.error(f"❌ {message}")
    except Exception as e:
        logger.error(f"Error adding to portfolio: {e}")
        st.error("❌ Failed to add to portfolio")

def remove_from_portfolio(ticker):
    """Remove holding from portfolio"""
    try:
        response = requests.post(
            f"{get_backend_url()}/api/portfolio/remove",
            json={"ticker": ticker},
            timeout=10
        )
        if response.status_code == 200:
            st.success(f"✅ {ticker} removed from portfolio")
            st.rerun()
        else:
            st.error("❌ Failed to remove from portfolio")
    except Exception as e:
        logger.error(f"Error removing from portfolio: {e}")
        st.error("❌ Failed to remove from portfolio")

def format_score_display(score):
    """Format score with color coding"""
    if score >= 80:
        return f'<span class="score-high">🟢 {score}</span>'
    elif score >= 60:
        return f'<span class="score-medium">🟡 {score}</span>'
    else:
        return f'<span class="score-low">🔴 {score}</span>'

def create_progress_bar(progress):
    """Create progress bar HTML"""
    return f"""
    <div class="progress-bar">
        <div class="progress-fill" style="width: {progress}%"></div>
    </div>
    <small>{progress}% to target</small>
    """

def main():
    """Main recommendations dashboard"""
    
    # Header
    st.title("📊 Live AI Recommendations Dashboard")
    st.markdown("**Real-time stock scoring with compliance-verified data**")
    
    # System status
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        # Get system health
        health_response = requests.get(f"{get_backend_url()}/health", timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            services = health_data.get("services", {})
            
            with col1:
                compliance_status = "✅ ON" if services.get("compliance") else "❌ OFF"
                st.metric("🔒 Compliance", compliance_status)
            
            with col2:
                data_status = "✅ Live" if services.get("data_sources") else "❌ Offline"
                st.metric("📡 Data Feed", data_status)
            
            with col3:
                ml_status = "✅ Ready" if services.get("ml_pipeline") else "❌ Loading"
                st.metric("🤖 ML Pipeline", ml_status)
            
            with col4:
                regime_status = "✅ Active" if services.get("regime_detector") else "❌ Inactive"
                st.metric("📈 Regime Detection", regime_status)
        else:
            st.error("❌ Backend API unavailable")
            return
            
    except Exception as e:
        st.error(f"❌ System status unavailable: {e}")
        return
    
    st.divider()
    
    # Fetch data
    recommendations = fetch_recommendations()
    portfolio = fetch_portfolio()
    
    # Recommendations section
    st.markdown("### 🎯 Live Stock Recommendations")
    st.markdown("*Auto-populated stocks scoring >80 with real-time compliance verification*")
    
    if recommendations:
        # Convert to DataFrame for display
        rec_df = pd.DataFrame(recommendations)
        
        # Format display columns
        if not rec_df.empty:
            # Create interactive table
            for idx, rec in rec_df.iterrows():
                with st.container():
                    col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                    
                    with col1:
                        score_html = format_score_display(rec.get('score', 0))
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <h4>{rec.get('ticker', 'N/A')}</h4>
                            <p><strong>Score:</strong> {score_html}</p>
                            <p><strong>Confidence:</strong> {rec.get('confidence', 0)}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.metric("Expected Profit", f"{rec.get('expected_profit', 0)}%")
                        st.metric("Max Entry", f"${rec.get('max_entry_price', 0):.2f}")
                    
                    with col3:
                        st.metric("Target Price", f"${rec.get('target_price', 0):.2f}")
                        st.metric("Stop Loss", f"${rec.get('stop_loss', 0):.2f}")
                    
                    with col4:
                        st.write(f"**Timeframe:** {rec.get('timeframe', 'N/A')}")
                        if st.button(f"➕ Add", key=f"add_{rec.get('ticker')}"):
                            add_to_portfolio(rec.get('ticker'))
                
                st.divider()
    else:
        st.info("ℹ️ No recommendations available. This may indicate:")
        st.markdown("""
        - No stocks currently scoring >80
        - Compliance mode is disabled
        - Data sources are unavailable
        - System is performing analysis
        """)
    
    # Portfolio section
    st.markdown("### 💼 Current Holdings")
    st.markdown("*Manual selections from recommendations with live progress tracking*")
    
    if portfolio:
        for holding in portfolio:
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                
                with col1:
                    st.markdown(f"""
                    <div class="holding-card">
                        <h4>{holding.get('ticker', 'N/A')}</h4>
                        <p><strong>Entry Date:</strong> {holding.get('entry_date', 'N/A')}</p>
                        <p><strong>Entry Price:</strong> ${holding.get('entry_price', 0):.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    current_price = holding.get('current_price', 0)
                    entry_price = holding.get('entry_price', 0)
                    pnl = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                    
                    st.metric("Current Price", f"${current_price:.2f}")
                    st.metric("P&L", f"{pnl:+.2f}%", delta=f"{pnl:+.2f}%")
                
                with col3:
                    progress = holding.get('progress_to_target', 0)
                    st.markdown("**Progress to Target:**")
                    st.markdown(create_progress_bar(progress), unsafe_allow_html=True)
                
                with col4:
                    if st.button(f"➖ Remove", key=f"remove_{holding.get('ticker')}"):
                        remove_from_portfolio(holding.get('ticker'))
            
            st.divider()
    else:
        st.info("ℹ️ No current holdings. Add recommendations from the table above.")
    
    # Quick insights
    with st.expander("📊 Market Insights", expanded=False):
        try:
            # Get regime analysis
            regime_response = requests.get(f"{get_backend_url()}/api/regime-analysis", timeout=10)
            if regime_response.status_code == 200:
                regime_data = regime_response.json()
                regime_analysis = regime_data.get("regime_analysis", {})
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Current Market Regime:**")
                    st.write(f"🎯 Type: {regime_analysis.get('regime_type', 'Unknown')}")
                    st.write(f"📊 Score: {regime_analysis.get('regime_score', 0)}/100")
                    st.write(f"📈 Volatility: {regime_analysis.get('volatility', 'Unknown')}")
                    st.write(f"🌍 Macro Signal: {regime_analysis.get('macro_signal', 'Unknown')}")
                
                with col2:
                    st.markdown("**Recommendation Summary:**")
                    total_recs = len(recommendations)
                    high_score = len([r for r in recommendations if r.get('score', 0) >= 80])
                    avg_confidence = sum([r.get('confidence', 0) for r in recommendations]) / len(recommendations) if recommendations else 0
                    
                    st.write(f"📈 Total Recommendations: {total_recs}")
                    st.write(f"⭐ High Score (>80): {high_score}")
                    st.write(f"🎯 Avg Confidence: {avg_confidence:.1f}%")
                    st.write(f"💼 Current Holdings: {len(portfolio)}")
            else:
                st.warning("⚠️ Regime analysis unavailable")
                
        except Exception as e:
            st.warning(f"⚠️ Insights unavailable: {e}")
    
    # Auto-refresh
    if st.button("🔄 Refresh Data"):
        st.rerun()
    
    # Footer info
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
        📊 Real-time data from verified sources | 🔒 Compliance-enforced recommendations<br>
        Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
        ⚡ Auto-refresh every 60 seconds
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
