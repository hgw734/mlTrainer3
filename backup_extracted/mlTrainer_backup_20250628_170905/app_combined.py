
import streamlit as st
import os
import json
import requests
from datetime import datetime
from flask import Flask, jsonify

# Import your existing components
from core.mlTrainer_engine import mlTrainerEngine
from core.compliance_mode import enable_compliance, is_compliance_enabled
from core.immutable_gateway import is_override_authorized, enable_override, disable_override
from ml.ml_tracker import load_all_model_metrics

# ------------------------------
# Flask App Setup (Background API)
# ------------------------------
flask_app = Flask(__name__)


@flask_app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "mlTrainer"})


@flask_app.route("/api/models")
def get_models():
    try:
        engine = mlTrainerEngine()
        models = engine.get_available_models()
        return jsonify(models)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@flask_app.route("/api/compliance")
def compliance_status():
    return jsonify({
        "compliance_enabled": is_compliance_enabled(),
        "override_authorized": is_override_authorized()
    })


@flask_app.route("/api/metrics")
def get_metrics():
    try:
        metrics = load_all_model_metrics()
        return jsonify(metrics)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------------
# Optimized Streamlit Configuration
# ------------------------------
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
os.environ["STREAMLIT_GLOBAL_DISABLE_PROGRESS_BAR_FOR_GATHERING_USAGE_STATS"] = "true"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION"] = "false"
os.environ["STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION"] = "false"
os.environ["STREAMLIT_SERVER_MAX_MESSAGE_SIZE"] = "200"

# ------------------------------
# Streamlit Configuration
# ------------------------------
st.set_page_config(
    page_title="mlTrainer2 Trading Intelligence",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Enable compliance mode
enable_compliance()

# Initialize mlTrainer system and persistent chat history
CHAT_MEMORY_FILE = "chat_memory.json"
MAX_CHAT_MESSAGES = 200


def load_chat_memory():
    """Load chat messages from persistent storage"""
    if os.path.exists(CHAT_MEMORY_FILE):
        try:
            with open(CHAT_MEMORY_FILE, "r") as f:
                messages = json.load(f)
                return messages[-MAX_CHAT_MESSAGES:]
        except Exception:
            pass
    return []


def save_chat_memory(messages):
    """Save chat messages to persistent storage"""
    try:
        trimmed_messages = messages[-MAX_CHAT_MESSAGES:] if len(
            messages) > MAX_CHAT_MESSAGES else messages
        with open(CHAT_MEMORY_FILE, "w") as f:
            json.dump(trimmed_messages, f, indent=2)
    except Exception:
        pass


if "mltrainer_engine" not in st.session_state:
    st.session_state.mltrainer_engine = mlTrainerEngine()

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = load_chat_memory()

# Memory optimization: Clean old session data periodically
if len(st.session_state.chat_messages) > MAX_CHAT_MESSAGES:
    st.session_state.chat_messages = st.session_state.chat_messages[-MAX_CHAT_MESSAGES:]
    gc.collect()  # Force garbage collection

mltrainer = st.session_state.mltrainer_engine

# ------------------------------
# Custom CSS
# ------------------------------
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
    }

    .header h1 {
        color: #2c3e50 !important;
        font-size: 1.8rem !important;
        font-weight: 300 !important;
        margin-bottom: 5px !important;
    }

    .status-bar {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        padding: 8px;
        background: #f8f9fa;
        border-radius: 6px;
        margin-bottom: 20px;
        border: 1px solid #e8e8e8;
    }

    .status-indicator {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #27ae60;
    }

    .status-text {
        color: #7f8c8d;
        font-size: 0.7rem;
        font-weight: 500;
    }

    .stButton button {
        background: #3498db !important;
        color: white !important;
        border: 1px solid #3498db !important;
        border-radius: 8px !important;
        font-family: Georgia, 'Times New Roman', Times, serif !important;
    }

    #MainMenu { visibility: hidden !important; }
    footer { visibility: hidden !important; }
    header { visibility: hidden !important; }
    </style>
""", unsafe_allow_html=True)

# ------------------------------
# Header with Backend Status
# ------------------------------
st.markdown("""
<div class="header">
    <h1>🧠 mlTrainer2 Trading Intelligence</h1>
    <p>Unified Flask + Streamlit Architecture</p>
</div>
""", unsafe_allow_html=True)

# Backend status check
flask_status = "🟢"
flask_text = "Flask OK"

try:
    response = requests.get("http://localhost:5000/health", timeout=2)
    if response.status_code == 200:
        data = response.json()
        if data.get("status") == "ok":
            flask_status = "🟢"
            flask_text = f"Flask OK ({data.get('service', 'Unknown')})"
        else:
            flask_status = "🟡"
            flask_text = "Flask Warning"
    else:
        flask_status = "🔴"
        flask_text = f"Flask Error ({response.status_code})"
except Exception:
    flask_status = "🔴"
    flask_text = "Flask Down"

st.markdown(f"""
<div class="status-bar">
    <span class="status-indicator"></span>
    <span class="status-text">mlTrainer System - Unified Mode | {flask_status} {flask_text}</span>
</div>
""", unsafe_allow_html=True)

# ------------------------------
# Clear Chat Buttons
# ------------------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    clear_col1, clear_col2 = st.columns(2)
    with clear_col1:
        if st.button("Clear Chat", key="clear_btn"):
            st.session_state.chat_messages = []
            save_chat_memory(st.session_state.chat_messages)
            st.rerun()

    with clear_col2:
        if st.button("Clear All", key="clear_all_btn"):
            st.session_state.chat_messages = []
            if hasattr(st.session_state, 'mltrainer_engine'):
                st.session_state.mltrainer_engine.trial_history = []
                st.session_state.mltrainer_engine._save_memory()
            save_chat_memory([])
            st.rerun()

# ------------------------------
# Chat Interface
# ------------------------------
chat_container = st.container(height=500)

with chat_container:
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("timestamp"):
                st.caption(f"⏱️ {message['timestamp']}")

# File upload
uploaded_files = st.file_uploader(
    "📎 Upload files (images/PDFs)",
    type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'pdf'],
    accept_multiple_files=True,
    key="file_uploader"
)

# Chat input
user_input = st.chat_input(
    "Tell me about the trial you want to begin, or ask about ML models...")

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

    # Create user message
    message_content = user_input if user_input.strip() else "File upload"
    if file_info:
        file_list = "\n".join(
            [f"📎 {f['name']} ({f['type']}, {f['size']:,} bytes)" for f in file_info])
        message_content += f"\n\n**Uploaded Files:**\n{file_list}"

    # Add user message
    st.session_state.chat_messages.append({
        "role": "user",
        "content": message_content,
        "timestamp": datetime.now().strftime("%H:%M"),
        "files": file_info if file_info else None
    })

    save_chat_memory(st.session_state.chat_messages)

    try:
        with st.spinner("mlTrainer is analyzing your request..."):
            # Smart symbol detection
            words = user_input.upper().split()
            trial_config = {}
            common_symbols = [
                "AAPL",
                "MSFT",
                "NVDA",
                "TSLA",
                "GOOGL",
                "AMZN",
                "META",
                "NFLX"]
            for word in words:
                if word in common_symbols:
                    trial_config["symbol"] = word
                    trial_config.update({
                        "model": "LSTM",
                        "start_date": "2023-01-01",
                        "end_date": "2024-01-01",
                        "train_ratio": 0.8,
                        "paper_mode": True
                    })
                    break

            # Enhanced prompt with file information
            enhanced_prompt = user_input
            if file_info:
                file_descriptions = []
                for file in file_info:
                    if file['type'].startswith('image/'):
                        file_descriptions.append(f"Image file: {file['name']}")
                    elif file['type'] == 'application/pdf':
                        file_descriptions.append(
                            f"PDF document: {file['name']}")
                enhanced_prompt += f"\n\nFiles attached: {', '.join(file_descriptions)}"

            # Process with mlTrainer
            result = mltrainer.start_trial(
                user_prompt=enhanced_prompt,
                trial_config=trial_config,
                chat_context=st.session_state.chat_messages
            )

            # Format response
            if "error" in result:
                response_content = f"❌ **Error**\n\n{result['error']}"
            else:
                response_content = result.get(
                    "response", "No response received")

                if result.get("ml_analysis"):
                    ml_data = result["ml_analysis"]
                    response_content += f"""

📊 **ML Analysis Results:**
• Model: {ml_data.get('model', 'Unknown')}
• Symbol: {ml_data.get('symbol', 'Unknown')}
• Score: {ml_data.get('score', 'Unknown')}
• Total Return: ${ml_data.get('total_return', 0):,.2f}
"""

        # Add mlTrainer response
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": response_content,
            "timestamp": datetime.now().strftime("%H:%M")
        })

        save_chat_memory(st.session_state.chat_messages)

        # Display ML analysis metrics if available
        if "ml_analysis" in result:
            analysis = result["ml_analysis"]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Symbol", analysis.get("symbol", "N/A"))
            with col2:
                st.metric("Model", analysis.get("model", "N/A"))
            with col3:
                st.metric("Accuracy", f"{analysis.get('score', 0):.1%}")
            with col4:
                st.metric("Returns", f"${analysis.get('total_return', 0):.0f}")

    except Exception as e:
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": f"❌ **System Error**\n\nI encountered an issue: {str(e)}\n\nPlease try rephrasing your request.",
            "timestamp": datetime.now().strftime("%H:%M")
        })
        save_chat_memory(st.session_state.chat_messages)

    st.rerun()

# ------------------------------
# Sidebar with Flask API Testing
# ------------------------------
with st.sidebar:
    st.header("🔧 System Status")

    # Test Flask API endpoints
    if st.button("Test Flask APIs"):
        with st.spinner("Testing Flask endpoints..."):
            # Test health
            try:
                r = requests.get("http://localhost:5000/health", timeout=2)
                st.success(f"✅ Health: {r.json()}")
            except Exception as e:
                st.error(f"❌ Health failed: {e}")

            # Test models
            try:
                r = requests.get("http://localhost:5000/api/models", timeout=2)
                st.success(f"✅ Models: {len(r.json().get('models', {}))}")
            except Exception as e:
                st.error(f"❌ Models failed: {e}")

            # Test compliance
            try:
                r = requests.get(
                    "http://localhost:5000/api/compliance", timeout=2)
                compliance = r.json()
                st.success(f"✅ Compliance: {compliance}")
            except Exception as e:
                st.error(f"❌ Compliance failed: {e}")

    st.markdown("---")

    # System configuration
    compliance_status = "🔒 ACTIVE" if is_compliance_enabled() else "🔓 DISABLED"
    override_status = "🚨 OVERRIDE ACTIVE" if is_override_authorized() else "🔒 ENFORCED"
    st.write(f"**Compliance:** {compliance_status}")
    st.write(f"**Override:** {override_status}")

    if is_override_authorized():
        if st.button("🔒 Restore Compliance"):
            disable_override()
            st.success("Compliance restored")
            st.rerun()
    else:
        if st.button("🚨 Override Compliance"):
            enable_override()
            st.success("Override enabled")
            st.rerun()
