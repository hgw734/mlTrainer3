import streamlit as st
import os
import json
import pandas as pd
from datetime import datetime
import requests
from core.mlTrainer_engine import mlTrainerEngine
from core.compliance_mode import enable_compliance, is_compliance_enabled

# Configure Streamlit page immediately
st.set_page_config(
    page_title="mlTrainer Interactive Chat",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Initialize compliance
enable_compliance()

# Initialize session state
if "mltrainer_engine" not in st.session_state:
    st.session_state.mltrainer_engine = mlTrainerEngine()

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

def check_backend_status():
    """Check if Flask backend is running"""
    try:
        import requests
        # Use environment variable set by launcher, fallback to default
        flask_port = os.environ.get('FLASK_RUN_PORT', '5000')
        response = requests.get(f"http://0.0.0.0:{flask_port}/health", timeout=2)
        return response.status_code == 200
    except:
        # Flask may be starting up in background - return False but don't error
        return False

# Custom CSS for chat interface
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    html, body, [class*="css"], .stApp {
        font-family: Georgia, 'Times New Roman', Times, serif !important;
        background: #ffffff !important;
        color: #2c3e50 !important;
        line-height: 1.6 !important;
    }

    .main > div {
        max-width: 900px !important;
        margin: 0 auto !important;
        padding: 10px 15px 120px 15px !important;
    }

    .header {
        text-align: center;
        margin-bottom: 20px;
        padding: 20px 0;
        border-bottom: 2px solid #3498db;
    }

    .header h1 {
        color: #2c3e50;
        font-size: 2.5em;
        margin-bottom: 10px;
    }

    .header p {
        color: #7f8c8d;
        font-size: 1.1em;
    }

    .status-row {
        display: flex;
        justify-content: center;
        gap: 15px;
        margin-bottom: 20px;
        padding: 15px 0;
    }

    .status-indicator {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 15px;
        font-size: 0.9em;
        font-family: Georgia, 'Times New Roman', Times, serif;
        font-weight: 400;
    }

    .status-online {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }

    .status-offline {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)



def main():
    # Status indicators above header
    backend_status = check_backend_status()

    status_html = '<div class="status-row">'

    if backend_status:
        status_html += '<span class="status-indicator status-online">Flask Backend Online</span>'
    else:
        status_html += '<span class="status-indicator status-offline">Flask Backend Starting...</span>'

    if is_compliance_enabled():
        status_html += '<span class="status-indicator status-online">Compliance Active</span>'
    else:
        status_html += '<span class="status-indicator status-offline">Compliance Inactive</span>'

    status_html += '<span class="status-indicator status-online">mlTrainer Ready</span>'
    status_html += '</div>'

    st.markdown(status_html, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="header">
        <h1>mlTrainer Interactive Chat</h1>
        <p>Advanced Financial AI with Compliance-First Architecture</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Clear buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat", help="Clear current conversation"):
            st.session_state.chat_messages = []
            st.rerun()

    with col2:
        if st.button("Clear All", help="Clear all data"):
            st.session_state.chat_messages = []
            if hasattr(st.session_state, 'mltrainer_engine'):
                st.session_state.mltrainer_engine.trial_history = []
            st.rerun()

    # Chat interface
    chat_container = st.container(height=500)

    with chat_container:
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message.get("timestamp"):
                    st.caption(f"{message['timestamp']}")

    # File upload
    uploaded_files = st.file_uploader(
        "Upload files (images/PDFs)",
        type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'pdf'],
        accept_multiple_files=True,
        key="file_uploader"
    )

    # Chat input
    user_input = st.chat_input("Tell me about the trial you want to begin, or ask about ML models...")

    if user_input or uploaded_files:
        # Process uploaded files
        file_info = []
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_details = {
                    "name": uploaded_file.name,
                    "type": uploaded_file.type,
                    "size": uploaded_file.size
                }

                uploads_dir = "uploads"
                if not os.path.exists(uploads_dir):
                    os.makedirs(uploads_dir)

                file_path = os.path.join(uploads_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                file_details["path"] = file_path
                file_info.append(file_details)

        # Create user message content
        message_content = user_input if user_input.strip() else "File upload"
        if file_info:
            file_list = "\n".join([f"{f['name']} ({f['type']}, {f['size']:,} bytes)" for f in file_info])
            message_content += f"\n\n**Uploaded Files:**\n{file_list}"

        # Add user message
        st.session_state.chat_messages.append({
            "role": "user",
            "content": message_content,
            "timestamp": datetime.now().strftime("%H:%M"),
            "files": file_info if file_info else None
        })

        # Process with mlTrainer
        try:
            communication_status = st.empty()
            communication_status.info("Processing with mlTrainer...")

            mltrainer_engine = st.session_state.mltrainer_engine

            # Process request through mlTrainer engine
            result = mltrainer_engine.start_trial(
                user_prompt=user_input,
                trial_config={
                    "files": file_info,
                    "timestamp": datetime.now().isoformat(),
                    "compliance_mode": is_compliance_enabled()
                }
            )

            # Add assistant response
            if "response" in result:
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": result["response"],
                    "timestamp": datetime.now().strftime("%H:%M")
                })

            if "error" in result:
                st.session_state.chat_messages.append({
                    "role": "assistant", 
                    "content": f"**Error**: {result['error']}",
                    "timestamp": datetime.now().strftime("%H:%M")
                })

            communication_status.empty()

        except Exception as e:
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": f"**System Error**: {str(e)}",
                "timestamp": datetime.now().strftime("%H:%M")
            })

        st.rerun()

# Call main() directly - Streamlit doesn't execute __main__ blocks
main()