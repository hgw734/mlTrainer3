"""
mlTrainer - Trial Results Dashboard
==================================

Purpose: Comprehensive display of completed trial results with detailed
analysis of model performance, accuracy rates, and timeframe predictions.
Shows what worked, what didn't, and why specific models succeeded.

Displays:
- Completed trial summaries with success/failure analysis
- Model accuracy rates per prediction timeframe (7-10 days, 3 months, 9 months)
- Performance comparisons across different models and market conditions
- Detailed reasoning for model selection and outcomes
- Real-time integration with trial completion data
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

# Page configuration
st.set_page_config(
    page_title="Trial Results - mlTrainer",
    page_icon="📊",
    layout="wide"
)

# API configuration
API_BASE_URL = "http://localhost:8000"

def get_trial_results() -> Dict[str, Any]:
    """Get completed trial results from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/trials/results", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API returned status {response.status_code}", "trials": []}
    except Exception as e:
        return {"error": f"Connection error: {str(e)}", "trials": []}

def get_model_performance() -> Dict[str, Any]:
    """Get model performance statistics"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/trials/performance", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API returned status {response.status_code}", "performance": {}}
    except Exception as e:
        return {"error": f"Connection error: {str(e)}", "performance": {}}

def format_timestamp(timestamp_str: str) -> str:
    """Format timestamp for display"""
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return timestamp_str

def display_trial_summary(trial: Dict[str, Any]):
    """Display individual trial summary card"""
    with st.container():
        # Trial header
        status_color = "🟢" if trial["status"] == "completed_success" else "🔴" if trial["status"] == "completed_failure" else "🟡"
        st.subheader(f"{status_color} Trial #{trial['id']} - {trial['objective']}")
        
        # Basic info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Duration", trial.get("duration", "N/A"))
        with col2:
            st.metric("Symbols Analyzed", trial.get("symbols_count", 0))
        with col3:
            st.metric("Models Used", len(trial.get("models_used", [])))
        with col4:
            st.metric("Overall Score", f"{trial.get('overall_score', 0):.1f}%")
        
        # Results breakdown
        if trial["status"] == "completed_success":
            st.success("✅ **Trial Completed Successfully**")
            
            # Model performance breakdown
            models_used = trial.get("models_used", [])
            if models_used:
                st.write("**Model Performance:**")
                for model in models_used:
                    accuracy = model.get("accuracy", 0)
                    timeframe = model.get("timeframe", "Unknown")
                    confidence = model.get("confidence", 0)
                    
                    # Color code by performance
                    if accuracy >= 85:
                        st.success(f"🎯 **{model['name']}** ({timeframe}): {accuracy:.1f}% accuracy, {confidence:.1f}% confidence")
                    elif accuracy >= 70:
                        st.warning(f"⚠️ **{model['name']}** ({timeframe}): {accuracy:.1f}% accuracy, {confidence:.1f}% confidence")
                    else:
                        st.error(f"❌ **{model['name']}** ({timeframe}): {accuracy:.1f}% accuracy, {confidence:.1f}% confidence")
            
            # What went well
            what_worked = trial.get("what_worked", [])
            if what_worked:
                st.write("**What Went Well:**")
                for item in what_worked:
                    st.write(f"• {item}")
            
            # Key insights
            insights = trial.get("insights", [])
            if insights:
                st.write("**Key Insights:**")
                for insight in insights:
                    st.write(f"• {insight}")
                    
        else:
            st.error("❌ **Trial Failed**")
            
            # What didn't work
            what_failed = trial.get("what_failed", [])
            if what_failed:
                st.write("**What Didn't Work:**")
                for item in what_failed:
                    st.write(f"• {item}")
            
            # Lessons learned
            lessons = trial.get("lessons_learned", [])
            if lessons:
                st.write("**Lessons Learned:**")
                for lesson in lessons:
                    st.write(f"• {lesson}")
        
        # Detailed reasoning
        reasoning = trial.get("detailed_reasoning", "")
        if reasoning:
            with st.expander("📝 Detailed Analysis & Reasoning"):
                st.write(reasoning)
        
        st.markdown("---")

def display_model_performance_chart(performance_data: Dict[str, Any]):
    """Display model performance comparison chart"""
    st.subheader("📈 Model Performance Comparison")
    
    models = performance_data.get("models", [])
    if not models:
        st.info("No model performance data available yet")
        return
    
    # Create DataFrame for plotting
    df_data = []
    for model in models:
        for timeframe in model.get("timeframes", []):
            df_data.append({
                "Model": model["name"],
                "Timeframe": timeframe["period"],
                "Accuracy": timeframe["accuracy"],
                "Confidence": timeframe["confidence"],
                "Trials": timeframe["trial_count"]
            })
    
    if df_data:
        df = pd.DataFrame(df_data)
        
        # Create grouped bar chart
        fig = px.bar(
            df, 
            x="Model", 
            y="Accuracy", 
            color="Timeframe",
            title="Model Accuracy by Timeframe",
            labels={"Accuracy": "Accuracy (%)", "Model": "ML Model"},
            height=400
        )
        fig.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed performance table
        st.subheader("📋 Detailed Performance Metrics")
        
        # Format for better display
        display_df = df.copy()
        display_df["Accuracy"] = display_df["Accuracy"].round(1).astype(str) + "%"
        display_df["Confidence"] = display_df["Confidence"].round(1).astype(str) + "%"
        
        st.dataframe(display_df, use_container_width=True)

def display_success_rate_analysis(performance_data: Dict[str, Any]):
    """Display success rate analysis by timeframe"""
    st.subheader("🎯 Success Rate Analysis by Timeframe")
    
    timeframe_stats = performance_data.get("timeframe_statistics", {})
    if not timeframe_stats:
        st.info("No timeframe statistics available yet")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        short_term = timeframe_stats.get("7-10_days", {})
        accuracy = short_term.get("average_accuracy", 0)
        trials = short_term.get("total_trials", 0)
        st.metric(
            "📅 Short-term (7-10 days)",
            f"{accuracy:.1f}%",
            f"{trials} trials"
        )
        if accuracy >= 85:
            st.success("✅ Meeting mlTrainer standards")
        else:
            st.warning("⚠️ Below target threshold")
    
    with col2:
        mid_term = timeframe_stats.get("3_months", {})
        accuracy = mid_term.get("average_accuracy", 0)
        trials = mid_term.get("total_trials", 0)
        st.metric(
            "📊 Mid-term (3 months)",
            f"{accuracy:.1f}%",
            f"{trials} trials"
        )
        if accuracy >= 85:
            st.success("✅ Meeting mlTrainer standards")
        else:
            st.warning("⚠️ Below target threshold")
    
    with col3:
        long_term = timeframe_stats.get("9_months", {})
        accuracy = long_term.get("average_accuracy", 0)
        trials = long_term.get("total_trials", 0)
        st.metric(
            "📈 Long-term (9 months)",
            f"{accuracy:.1f}%",
            f"{trials} trials"
        )
        if accuracy >= 85:
            st.success("✅ Meeting mlTrainer standards")
        else:
            st.warning("⚠️ Below target threshold")

def display_top_performing_models(performance_data: Dict[str, Any]):
    """Display top performing models section"""
    st.subheader("🏆 Top Performing Models")
    
    top_models = performance_data.get("top_performers", [])
    if not top_models:
        st.info("No top performer data available yet")
        return
    
    for i, model in enumerate(top_models[:5], 1):
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**#{i}. {model['name']}**")
                st.write(f"Best for: {model.get('best_conditions', 'Various conditions')}")
                st.write(f"Strength: {model.get('key_strength', 'Consistent performance')}")
                
            with col2:
                st.metric("Avg Accuracy", f"{model.get('average_accuracy', 0):.1f}%")
                st.metric("Success Rate", f"{model.get('success_rate', 0):.1f}%")
            
            st.markdown("---")

def main():
    """Main Trial Results page"""
    st.title("📊 Trial Results Dashboard")
    st.markdown("Comprehensive analysis of completed mlTrainer trials with detailed performance metrics")
    
    # Auto-refresh controls
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Refresh Results"):
            st.rerun()
    
    # Get trial results data
    with st.spinner("Loading trial results..."):
        results_data = get_trial_results()
        performance_data = get_model_performance()
    
    # Handle API errors
    if "error" in results_data:
        st.error(f"Error loading trial results: {results_data['error']}")
        st.info("Make sure the mlTrainer backend is running and try refreshing the page.")
        return
    
    if "error" in performance_data:
        st.warning(f"Error loading performance data: {performance_data['error']}")
        performance_data = {"performance": {}}
    
    trials = results_data.get("trials", [])
    performance_stats = performance_data.get("performance", {})
    
    if not trials:
        st.info("🔄 No completed trials yet. Start a trial from the mlTrainer Chat to see results here.")
        return
    
    # Summary metrics
    st.subheader("📈 Overall Statistics")
    
    total_trials = len(trials)
    successful_trials = len([t for t in trials if t.get("status") == "completed_success"])
    failed_trials = len([t for t in trials if t.get("status") == "completed_failure"])
    success_rate = (successful_trials / total_trials * 100) if total_trials > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Trials", total_trials)
    with col2:
        st.metric("Successful", successful_trials, f"{success_rate:.1f}%")
    with col3:
        st.metric("Failed", failed_trials)
    with col4:
        avg_score = sum(t.get("overall_score", 0) for t in trials) / len(trials) if trials else 0
        st.metric("Avg Score", f"{avg_score:.1f}%")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Recent Trials", "📊 Performance Analysis", "🎯 Success Rates", "🏆 Top Models"])
    
    with tab1:
        st.subheader("🔍 Recent Trial Results")
        
        # Sort trials by completion time (most recent first)
        sorted_trials = sorted(trials, key=lambda x: x.get("completed_at", ""), reverse=True)
        
        # Display each trial
        for trial in sorted_trials[:10]:  # Show last 10 trials
            display_trial_summary(trial)
    
    with tab2:
        display_model_performance_chart(performance_stats)
    
    with tab3:
        display_success_rate_analysis(performance_stats)
    
    with tab4:
        display_top_performing_models(performance_stats)
    
    # Footer
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Auto-refreshing every 30 seconds")

if __name__ == "__main__":
    main()