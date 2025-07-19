"""
Simple mlTrainer Chat Interface
==============================

A simplified chat interface for mlTrainer that ensures reliable display and functionality.
"""

import streamlit as st
import requests
import json
from datetime import datetime

# Page configuration
st.set_page_config(page_title="💬 Simple Chat", layout="wide")

def get_backend_url():
    """Get backend API URL"""
    return "http://127.0.0.1:8000"

def send_message_to_api(message):
    """Send message to backend chat API"""
    try:
        response = requests.post(
            f"{get_backend_url()}/api/chat",
            json={"message": message},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("response", "No response received")
        else:
            return f"Error: Backend returned status {response.status_code}"
    except Exception as e:
        return f"Connection error: {e}"

def main():
    st.title("💬 mlTrainer Simple Chat")
    st.markdown("Direct communication with your AI trading assistant")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Chat input form
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area("Your message:", height=100, placeholder="Ask me about market conditions, stocks, or trading strategies...")
        submitted = st.form_submit_button("Send Message")
        
        if submitted and user_input:
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now()
            })
            
            # Get AI response
            with st.spinner("mlTrainer is thinking..."):
                ai_response = send_message_to_api(user_input)
            
            # Add AI response
            st.session_state.messages.append({
                "role": "assistant",
                "content": ai_response,
                "timestamp": datetime.now()
            })
    
    # Display conversation
    st.markdown("### Conversation")
    
    if st.session_state.messages:
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                st.markdown(f"**You ({message['timestamp'].strftime('%H:%M:%S')}):**")
                st.info(message["content"])
            else:
                st.markdown(f"**mlTrainer ({message['timestamp'].strftime('%H:%M:%S')}):**")
                st.success(message["content"])
            st.markdown("---")
    else:
        st.markdown("*No messages yet. Start a conversation above!*")
    
    # Clear conversation button
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()
    
    # System status
    st.markdown("### System Status")
    try:
        health_response = requests.get(f"{get_backend_url()}/health", timeout=5)
        if health_response.status_code == 200:
            st.success("✅ Backend system is online")
        else:
            st.error("❌ Backend system issues")
    except:
        st.error("❌ Cannot connect to backend")

if __name__ == "__main__":
    main()