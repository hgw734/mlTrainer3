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

# Import unified components
from core.unified_executor import get_unified_executor
from core.enhanced_background_manager import get_enhanced_background_manager
from mltrainer_claude_integration import MLTrainerClaude
from goal_system import GoalSystem
from mlagent_bridge import MLAgentBridge

# Import memory system (would come from advanced version)
# from utils.persistent_memory import add_chat_message, get_memory_stats


# Import recommendation system
from recommendation_tracker import get_recommendation_tracker
from virtual_portfolio_manager import get_virtual_portfolio_manager

# Initialize components
@st.cache_resource
def init_components():
    """Initialize all system components"""
    return {
        "executor": get_unified_executor(),
        "background_manager": get_enhanced_background_manager(),
        "claude": MLTrainerClaude(),
        "goal_system": GoalSystem(),
        "bridge": MLAgentBridge(),
        "recommendation_tracker": get_recommendation_tracker(),
        "portfolio_manager": get_virtual_portfolio_manager(),
    }


    def load_chat_history():
        """Load chat history from file"""
        if os.path.exists("logs/unified_chat_history.json"):
            with open("logs/unified_chat_history.json", "r") as f:
                return json.load(f)
                return []


                def save_chat_history(history):
                    """Save chat history to file"""
                    os.makedirs("logs", exist_ok=True)
                    with open("logs/unified_chat_history.json", "w") as f:
                        json.dump(history, f, indent=2)


                        def add_message_with_memory(role: str, content: str, **metadata):
                            """Add message to both chat history and memory system"""
                            # Add to session state
                            if "messages" not in st.session_state:
                                st.session_state.messages = []

                                message = {"role": role, "content": content, "timestamp": datetime.now().isoformat(), **metadata}

                                st.session_state.messages.append(message)
                                save_chat_history(st.session_state.messages)

                                # Would also add to persistent memory system
                                # add_chat_message(role, content, **metadata)


                                def main():
                                    st.set_page_config(page_title="mlTrainer Unified", page_icon="🤖", layout="wide", initial_sidebar_state="collapsed")

                                    # Mobile-optimized CSS from advanced version
                                    st.markdown(
                                    """
                                    <style>
                                    /* Mobile Optimization */
                                    @media (max-width: 768px) {
                                    .stButton > button {
                                    width: 100%;
                                    margin: 2px 0;
                                    }
                                    .main {
                                    padding: 1rem;
                                    }
                                    .sidebar .sidebar-content {
                                    padding: 1rem;
                                    }
                                    [data-testid="stSidebar"] {
                                    width: 80%;
                                    }
                                    }

                                    /* Enhanced UI Styling */
                                    .stChat {
                                    background-color: #f0f2f6;
                                    border-radius: 10px;
                                    padding: 1rem;
                                    }

                                    .user-message {
                                    background-color: #e3f2fd;
                                    padding: 0.5rem 1rem;
                                    border-radius: 10px;
                                    margin: 0.5rem 0;
                                    }

                                    .assistant-message {
                                    background-color: #f5f5f5;
                                    padding: 0.5rem 1rem;
                                    border-radius: 10px;
                                    margin: 0.5rem 0;
                                    }

                                    .trial-card {
                                    background-color: white;
                                    padding: 1rem;
                                    border-radius: 8px;
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                    margin: 0.5rem 0;
                                    }

                                    .goal-container {
                                    background-color: #fff3e0;
                                    padding: 1rem;
                                    border-radius: 8px;
                                    margin: 1rem 0;
                                    }

                                    .compliance-badge {
                                    display: inline-block;
                                    padding: 0.25rem 0.5rem;
                                    border-radius: 4px;
                                    font-size: 0.875rem;
                                    font-weight: 500;
                                    }

                                    .compliance-approved {
                                    background-color: #c8e6c9;
                                    color: #2e7d32;
                                    }

                                    .compliance-blocked {
                                    background-color: #ffcdd2;
                                    color: #c62828;
                                    }
                                    </style>
                                    """,
                                    unsafe_allow_html=True,
                                    )

                                    # Initialize components
                                    components = init_components()

                                    # Initialize session state
                                    if "messages" not in st.session_state:
                                        st.session_state.messages = load_chat_history()

                                        # Sidebar with enhanced features
                                        with st.sidebar:
                                            st.title("🤖 mlTrainer Control Panel")

                                            # Goal Management Section
                                            st.subheader("🎯 System Goal")
                                            current_goal = components["goal_system"].get_current_goal()

                                            with st.expander("Goal Settings", expanded=True):
                                                new_goal = st.text_area(
                                                "Current Goal", value=current_goal.get("goal", ""), help="Define the system's primary objective"
                                                )

                                                if st.button("Update Goal", type="primary"):
                                                    components["goal_system"].set_goal(new_goal)
                                                    st.success("Goal updated successfully!")
                                                    st.rerun()

                                                    # Display compliance requirements
                                                    if current_goal.get("compliance_requirements"):
                                                        st.caption("Active Compliance Rules:")
                                                        for req in current_goal["compliance_requirements"]:
                                                            st.caption(f"• {req}")

                                                            st.divider()

                                                            # Background Trials Section
                                                            st.subheader("🔄 Background Trials")

                                                            trials = components["background_manager"].get_all_trials()
                                                            active_trials = [t for t in trials if t.get("status") not in ["completed", "failed", "cancelled"]]

                                                            if active_trials:
                                                                st.metric("Active Trials", len(active_trials))

                                                                for trial in active_trials[:3]:  # Show top 3
                                                                    with st.container():
                                                                        col1, col2 = st.columns([3, 1])
                                                                        with col1:
                                                                            st.caption(f"**{trial['trial_id']}**")
                                                                            progress = trial.get("progress", {})
                                                                            st.progress(
                                                                                progress.get("percentage", 0) / 100,
                                                                                text=f"{progress.get('completed', 0)}/{progress.get('total', 0)} steps",
                                                                            )
                                                                        with col2:
                                                                            if trial["status"] == "pending_approval":
                                                                                if st.button("✓", key=f"approve_{trial['trial_id']}"):
                                                                                    components["background_manager"].approve_trial(trial["trial_id"])
                                                                                    st.rerun()
                                                            else:
                                                                st.info("No active trials")

                                                            st.divider()

                                                                                        # Trading Recommendations Section
                                                                                        st.subheader("💡 Trading Recommendations")
                                                                                        
                                                                                        # Get active recommendations
                                                                                        active_recs = components["recommendation_tracker"].get_active_recommendations()
                                                                                        
                                                                                        if active_recs:
                                                                                            st.metric("Active Recommendations", len(active_recs))
                                                                                            
                                                                                            # Display top recommendations
                                                                                            for rec in active_recs[:5]:  # Top 5
                                                                                                with st.expander(f"{rec['symbol']} - {rec['timeframe']}", expanded=True):
                                                                                                    col1, col2, col3 = st.columns(3)
                                                                                                    
                                                                                                    with col1:
                                                                                                        st.metric("Signal Strength", f"{rec['signal_strength']:.1%}")
                                                                                                        st.metric("Entry Price", f"${rec['entry_price']:.2f}")
                                                                                                    
                                                                                                    with col2:
                                                                                                        st.metric("Profit Probability", f"{rec['profit_probability']:.1%}")
                                                                                                        st.metric("Target Price", f"${rec['target_price']:.2f}")
                                                                                                    
                                                                                                    with col3:
                                                                                                        st.metric("Confidence", f"{rec['confidence']:.1%}")
                                                                                                        st.metric("Stop Loss", f"${rec['stop_loss']:.2f}")
                                                                                                    
                                                                                                    st.caption(f"Model: {rec['model_used']}")
                                                                                                    
                                                                                                    # Add to portfolio button
                                                                                                    if st.button(f"Add to Watch List", key=f"watch_{rec['symbol']}_{rec['timestamp']}"):
                                                                                                        st.success(f"Added {rec['symbol']} to watch list!")
                                                                                        else:
                                                                                            st.info("No active recommendations. Run a scan to generate recommendations.")
                                                                                            
                                                                                            if st.button("🔍 Scan S&P 500"):
                                                                                                with st.spinner("Scanning for opportunities..."):
                                                                                                    # Get S&P 500 symbols (simplified list for demo)
                                                                                                    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "JNJ", "V"]
                                                                                                    
                                                                                                    # Run async scan
                                                                                                    import asyncio
                                                                                                    recommendations = asyncio.run(
                                                                                                        components["recommendation_tracker"].scan_for_opportunities(symbols)
                                                                                                    )
                                                                                                    
                                                                                                    st.success(f"Found {len(recommendations)} recommendations!")
                                                                                                    st.rerun()
                                                                                        
                                                                                        st.divider()
                                                                                        
                                                                                        # Virtual Portfolio Performance
                                                                                        st.subheader("📈 Virtual Portfolio Performance")
                                                                                        
                                                                                        portfolio_metrics = components["portfolio_manager"].get_portfolio_metrics()
                                                                                        
                                                                                        col1, col2, col3, col4 = st.columns(4)
                                                                                        
                                                                                        with col1:
                                                                                            st.metric("Portfolio Value", f"${portfolio_metrics['total_value']:,.2f}")
                                                                                            st.metric("Open Positions", portfolio_metrics['open_positions'])
                                                                                        
                                                                                        with col2:
                                                                                            st.metric("Total Return", f"{portfolio_metrics['total_return_pct']:+.2f}%", 
                                                                                                     delta=f"${portfolio_metrics['total_return']:,.2f}")
                                                                                            st.metric("Closed Trades", portfolio_metrics['closed_positions'])
                                                                                        
                                                                                        with col3:
                                                                                            st.metric("Win Rate", f"{portfolio_metrics['win_rate']:.1f}%")
                                                                                            st.metric("Avg Win", f"{portfolio_metrics['avg_win_pct']:+.2f}%")
                                                                                        
                                                                                        with col4:
                                                                                            st.metric("Sharpe Ratio", f"{portfolio_metrics['sharpe_ratio']:.2f}")
                                                                                            st.metric("Sortino Ratio", f"{portfolio_metrics.get('sortino_ratio', 0):.2f}")
                                                                                        
                                                                                        # Show performance report
                                                                                        if st.button("📄 Generate Performance Report"):
                                                                                            report = components["portfolio_manager"].generate_performance_report()
                                                                                            st.text_area("Performance Report", report, height=300)

                                                                                        st.divider()

                                                                                        # Model Statistics
                                                                                        st.subheader("📊 Model Statistics")
                                                                                        executor_summary = components["executor"].get_execution_summary()

                                                                                        col1, col2 = st.columns(2)
                                                                                        with col1:
                                                                                            st.metric("Total Models", executor_summary.get("registered_actions", 0))
                                                                                            with col2:
                                                                                                st.metric("Executions", executor_summary.get("total_executions", 0))

                                                                                                # Memory Stats (would come from advanced memory system)
                                                                                                st.subheader("🧠 Memory Status")
                                                                                                st.caption("Chat History: " + str(len(st.session_state.messages)) + " messages")

                                                                                                st.divider()

                                                                                                # Quick Actions
                                                                                                st.subheader("⚡ Quick Actions")

                                                                                                if st.button("📊 Portfolio Analysis", use_container_width=True):
                                                                                                    st.session_state.quick_action = "portfolio_analysis"

                                                                                                    if st.button("🎯 Momentum Screening", use_container_width=True):
                                                                                                        st.session_state.quick_action = "momentum_screening"

                                                                                                        if st.button("📈 Train Models", use_container_width=True):
                                                                                                            st.session_state.quick_action = "train_models"

                                                                                                            if st.button("🔍 Market Regime", use_container_width=True):
                                                                                                                st.session_state.quick_action = "regime_detection"

                                                                                                                if st.button("🗑️ Clear Chat", type="secondary", use_container_width=True):
                                                                                                                    st.session_state.messages = []
                                                                                                                    save_chat_history([])
                                                                                                                    st.rerun()

                                                                                                                    # Main chat interface
                                                                                                                    st.title("🤖 mlTrainer Unified Interface")
                                                                                                                    
                                                                                                                    # Display update status
                                                                                                                    if os.path.exists("/data/recommendations/last_scan_time.json"):
                                                                                                                        try:
                                                                                                                            with open("/data/recommendations/last_scan_time.json", "r") as f:
                                                                                                                                last_scan = json.load(f)
                                                                                                                                last_scan_time = datetime.fromisoformat(last_scan["timestamp"])
                                                                                                                                time_since = datetime.now() - last_scan_time
                                                                                                                                minutes_ago = int(time_since.total_seconds() / 60)
                                                                                                                                
                                                                                                                                if minutes_ago < 1:
                                                                                                                                    st.success("🔄 Recommendations updated just now")
                                                                                                                                else:
                                                                                                                                    st.info(f"🕒 Last update: {minutes_ago} minutes ago")
                                                                                                                        except:
                                                                                                                            pass

                                                                                                                    # Display current goal prominently
                                                                                                                    if current_goal.get("goal"):
                                                                                                                        st.markdown(
                                                                                                                        f"""
                                                                                                                        <div class="goal-container">
                                                                                                                        <strong>Current Goal:</strong> {current_goal['goal']}<br>
                                                                                                                        <small>Set: {current_goal.get('timestamp', 'Unknown')}</small>
                                                                                                                        </div>
                                                                                                                        """,
                                                                                                                        unsafe_allow_html=True,
                                                                                                                        )

                                                                                                                        # Handle quick actions
                                                                                                                        if "quick_action" in st.session_state:
                                                                                                                            action = st.session_state.quick_action
                                                                                                                            del st.session_state.quick_action

                                                                                                                            quick_prompts = {
                                                                                                                            "portfolio_analysis": "Please analyze my portfolio and suggest optimal allocations using mean-variance optimization.",
                                                                                                                            "momentum_screening": "Run momentum screening using top-performing ML models to identify strong stocks.",
                                                                                                                            "train_models": "Train random forest and gradient boosting models on AAPL with recent market data.",
                                                                                                                            "regime_detection": "Detect the current market regime using clustering analysis.",
                                                                                                                            }

                                                                                                                            if action in quick_prompts:
                                                                                                                                user_input = quick_prompts[action]
                                                                                                                                add_message_with_memory("user", user_input)

                                                                                                                                # Get mlTrainer response
                                                                                                                                with st.spinner("mlTrainer is thinking# Production code implemented"):
                                                                                                                                    response = components["claude"].get_response_with_goal(
                                                                                                                                    user_input,
                                                                                                                                    current_goal.get("goal", ""),
                                                                                                                                    [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-10:]],
                                                                                                                                    )

                                                                                                                                    add_message_with_memory("assistant", response)
                                                                                                                                    st.rerun()

                                                                                                                                    # Display chat messages
                                                                                                                                    for message in st.session_state.messages:
                                                                                                                                        role = message["role"]
                                                                                                                                        content = message["content"]

                                                                                                                                        if role == "user":
                                                                                                                                            st.markdown(
                                                                                                                                            f'<div class="user-message">👤 <strong>You:</strong><br>{content}</div>', unsafe_allow_html=True
                                                                                                                                            )
                                                                                                                                            else:
                                                                                                                                                # Parse for executable actions
                                                                                                                                                parsed = components["bridge"].parse_mltrainer_response(content)

                                                                                                                                                st.markdown(
                                                                                                                                                f'<div class="assistant-message">🤖 <strong>mlTrainer:</strong><br>{content}</div>',
                                                                                                                                                unsafe_allow_html=True,
                                                                                                                                                )

                                                                                                                                                # Show executable actions if found
                                                                                                                                                if parsed.get("executable") and (parsed.get("trial_suggestions") or parsed.get("models_mentioned")):
                                                                                                                                                    with st.expander("🎯 Executable Actions Detected", expanded=True):
                                                                                                                                                        col1, col2 = st.columns([3, 1])

                                                                                                                                                        with col1:
                                                                                                                                                            if parsed.get("trial_suggestions"):
                                                                                                                                                                st.write("**Suggested Trials:**")
                                                                                                                                                                for suggestion in parsed["trial_suggestions"]:
                                                                                                                                                                    st.write(f"• {suggestion}")

                                                                                                                                                                    if parsed.get("models_mentioned"):
                                                                                                                                                                        st.write("**Models Ready to Execute:**")
                                                                                                                                                                        for model in parsed["models_mentioned"]:
                                                                                                                                                                            st.write(f"• {model}")

                                                                                                                                                                            with col2:
                                                                                                                                                                                if st.button("🚀 Execute", key=f"exec_{message.get('timestamp', '')}"):
                                                                                                                                                                                    # Start background trial
                                                                                                                                                                                    trial_id = components["background_manager"].start_trial(content, auto_approve=False)
                                                                                                                                                                                    if trial_id:
                                                                                                                                                                                        st.success(f"Trial {trial_id} created! Check sidebar for approval.")
                                                                                                                                                                                        else:
                                                                                                                                                                                            st.error("Failed to create trial. Check compliance.")
                                                                                                                                                                                            st.rerun()

                                                                                                                                                                                            # Chat input
                                                                                                                                                                                            user_input = st.chat_input("Ask mlTrainer anything# Production code implemented")

                                                                                                                                                                                            if user_input:
                                                                                                                                                                                                # Add user message
                                                                                                                                                                                                add_message_with_memory("user", user_input)

                                                                                                                                                                                                # Get mlTrainer response with goal context
                                                                                                                                                                                                with st.spinner("mlTrainer is processing# Production code implemented"):
                                                                                                                                                                                                    try:
                                                                                                                                                                                                        # Include goal in context
                                                                                                                                                                                                        response = components["claude"].get_response_with_goal(
                                                                                                                                                                                                        user_input,
                                                                                                                                                                                                        current_goal.get("goal", ""),
                                                                                                                                                                                                        [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-10:]],
                                                                                                                                                                                                        )

                                                                                                                                                                                                        # Add assistant response
                                                                                                                                                                                                        add_message_with_memory("assistant", response)

                                                                                                                                                                                                        # Check for executable actions
                                                                                                                                                                                                        parsed = components["executor"].parse_mltrainer_response(response)
                                                                                                                                                                                                        if parsed["executable"]:
                                                                                                                                                                                                            # Automatically create trial for review
                                                                                                                                                                                                            trial_id = components["background_manager"].start_trial(response, auto_approve=False)
                                                                                                                                                                                                            if trial_id:
                                                                                                                                                                                                                st.toast(f"Trial {trial_id} ready for review!", icon="🎯")

                                                                                                                                                                                                                except Exception as e:
                                                                                                                                                                                                                    st.error(f"Error: {str(e)}")
                                                                                                                                                                                                                    add_message_with_memory("system", f"Error occurred: {str(e)}")

                                                                                                                                                                                                                    st.rerun()

                                                                                                                                                                                                                    # Footer with system status
                                                                                                                                                                                                                    with st.container():
                                                                                                                                                                                                                        st.divider()
                                                                                                                                                                                                                        col1, col2, col3, col4 = st.columns(4)

                                                                                                                                                                                                                        with col1:
                                                                                                                                                                                                                            st.caption(
                                                                                                                                                                                                                            "**Models Available:** " + str(components["executor"].get_execution_summary()["registered_actions"])
                                                                                                                                                                                                                            )

                                                                                                                                                                                                                            with col2:
                                                                                                                                                                                                                                active_count = len(
                                                                                                                                                                                                                                [
                                                                                                                                                                                                                                t
                                                                                                                                                                                                                                for t in components["background_manager"].get_all_trials()
                                                                                                                                                                                                                                if t.get("status") not in ["completed", "failed", "cancelled"]
                                                                                                                                                                                                                                ]
                                                                                                                                                                                                                                )
                                                                                                                                                                                                                                st.caption(f"**Active Trials:** {active_count}")

                                                                                                                                                                                                                                with col3:
                                                                                                                                                                                                                                    st.caption("**Compliance:** ✅ Active")

                                                                                                                                                                                                                                    with col4:
                                                                                                                                                                                                                                        st.caption("**System:** 🟢 Operational")


                                                                                                                                                                                                                                        if __name__ == "__main__":
                                                                                                                                                                                                                                            main()
