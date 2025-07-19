import streamlit as st
import json
import os
import socket
from datetime import datetime
from core.mlTrainer_engine import mlTrainerEngine
from core.compliance_mode import enable_compliance, is_compliance_enabled, is_override_authorized
from core.immutable_gateway import enable_override, disable_override, load_allowed_apis


# Port configuration handled by fix_ports_replit.py

st.set_page_config(
    page_title="mlTrainer2 Trading Intelligence",
    layout="centered",
    initial_sidebar_state="collapsed"
)

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
                # Keep only last 200 messages
                return messages[-MAX_CHAT_MESSAGES:]
        except Exception:
            pass

    # Return empty if no history exists
    return []


def save_chat_memory(messages):
    """Save chat messages to persistent storage"""
    try:
        # Keep only last 200 messages
        trimmed_messages = messages[-MAX_CHAT_MESSAGES:] if len(
            messages) > MAX_CHAT_MESSAGES else messages
        with open(CHAT_MEMORY_FILE, "w") as f:
            json.dump(trimmed_messages, f, indent=2)
    except Exception:
        pass


if "mltrainer_engine" not in st.session_state:
    st.session_state.mltrainer_engine = mlTrainerEngine()
# Load persistent chat messages
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = load_chat_memory()

mltrainer = st.session_state.mltrainer_engine

# Custom CSS matching the HTML design exactly
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
        background-color: #ffffff !important;
        color: #2c3e50 !important;
        line-height: 1.6 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    .main > div {
        max-width: 900px !important;
        width: 100% !important;
        margin: 0 auto !important;
        padding: 10px 15px 120px 15px !important;
        min-height: auto !important;
        display: flex !important;
        flex-direction: column !important;
        box-sizing: border-box !important;
    }

    /* Header styling */
    .header {
        text-align: center;
        margin: 0;
        padding: 0;
    }

    .header h1 {
        color: #2c3e50 !important;
        font-size: 1.3rem !important;
        font-weight: 300 !important;
        margin: 0 0 1px 0 !important;
        letter-spacing: -0.5px !important;
        font-family: Georgia, 'Times New Roman', Times, serif !important;
    }

    .header p {
        color: #7f8c8d !important;
        font-size: 0.75rem !important;
        font-weight: 400 !important;
        margin: 0 0 1px 0 !important;
    }

    /* Clear buttons */
    .clear-buttons {
        display: flex;
        justify-content: center;
        gap: 6px;
        margin: 0;
    }

    .clear-btn {
        padding: 6px 11px !important;
        border: none !important;
        border-radius: 6px !important;
        background: #ffa366 !important;
        color: white !important;
        cursor: pointer !important;
        font-size: 0.58rem !important;
        font-family: Georgia, 'Times New Roman', Times, serif !important;
        transition: background 0.2s ease !important;
    }

    .clear-btn:hover {
        background: #ff9447 !important;
    }

    .clear-btn.danger {
        background: #dc3545 !important;
    }

    .clear-btn.danger:hover {
        background: #c82333 !important;
    }

    /* Status bar */
    .status-bar {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        padding: 4px;
        background: #f8f9fa;
        border-radius: 6px;
        margin-bottom: 5px;
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
        font-size: 0.6rem;
        font-weight: 500;
    }

    /* Chat container styling for native components */
    .stChatMessage {
        margin-bottom: 1rem;
    }

    .stChatMessage [data-testid="user-avatar"] {
        background: #27ae60 !important;
    }

    .stChatMessage [data-testid="assistant-avatar"] {
        background: #3498db !important;
    }

    /* Chat input styling */
    .stChatInput {
        border: 1px solid #e8e8e8 !important;
        border-radius: 12px !important;
        background: #ffffff !important;
    }

    .stButton button {
        background: #3498db !important;
        color: white !important;
        border: 1px solid #3498db !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-family: Georgia, 'Times New Roman', Times, serif !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        transition: background 0.2s ease !important;
    }

    .stButton button:hover {
        background: #2980b9 !important;
        border-color: #2980b9 !important;
    }

    /* File uploader styling */
    .stFileUploader {
        margin-top: 8px !important;
        margin-bottom: 8px !important;
    }

    .stFileUploader > div {
        border: 1px dashed #e8e8e8 !important;
        border-radius: 8px !important;
        background: #f8f9fa !important;
        padding: 8px 12px !important;
    }

    .stFileUploader label {
        font-size: 0.75rem !important;
        color: #7f8c8d !important;
        font-family: Georgia, 'Times New Roman', Times, serif !important;
    }

    /* Bottom navigation */
    .bottom-nav {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: #ffffff;
        border-top: 1px solid #e8e8e8;
        display: flex;
        justify-content: space-around;
        padding: 8px 0 12px 0;
        z-index: 1000;
        box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
    }

    .nav-item {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-decoration: none;
        color: #7f8c8d;
        transition: color 0.2s ease;
        padding: 12px 8px;
        border-radius: 8px;
        min-width: 60px;
        font-size: 13px;
        font-weight: 400;
        font-family: Georgia, 'Times New Roman', Times, serif;
    }

    .nav-item:hover {
        color: #3498db;
        background: rgba(52, 152, 219, 0.1);
    }

    .nav-item.active {
        color: #3498db;
        font-weight: 500;
    }

    /* Typing indicator */
    .typing-indicator {
        padding: 16px 40px;
        color: #95a5a6;
        font-style: italic;
        font-size: 0.8rem;
        display: none;
    }

    /* Hide Streamlit elements */
    .stDeployButton {
        display: none !important;
    }

    .stDecoration {
        display: none !important;
    }

    #MainMenu {
        visibility: hidden !important;
    }

    footer {
        visibility: hidden !important;
    }

    header {
        visibility: hidden !important;
    }

    /* Enhanced Mobile responsiveness */
    @media (max-width: 768px) {
        .main > div {
            max-width: 100% !important;
            padding: 8px 8px 110px 8px !important;
            box-sizing: border-box !important;
        }

        .header h1 {
            font-size: 1.1rem !important;
            margin-bottom: 2px !important;
        }

        .header p {
            font-size: 0.7rem !important;
            margin-bottom: 8px !important;
        }

        .message-content {
            max-width: 85% !important;
            padding: 12px 16px !important;
            font-size: 0.8rem !important;
        }

        .message-avatar {
            width: 24px !important;
            height: 24px !important;
            font-size: 0.6rem !important;
        }

        .chat-container {
            height: calc(100vh - 180px) !important;
            max-height: calc(100vh - 180px) !important;
            min-height: 200px !important;
            border-radius: 8px !important;
        }

        .messages {
            padding: 12px !important;
        }

        .input-container {
            padding: 10px !important;
            min-height: 85px !important;
        }

        .stTextArea textarea {
            resize: none !important;
            min-height: 50px !important;
            font-size: 0.8rem !important;
            padding: 8px 12px !important;
        }

        .stButton button {
            padding: 6px 12px !important;
            font-size: 0.8rem !important;
        }

        .clear-btn {
            padding: 4px 8px !important;
            font-size: 0.55rem !important;
        }

        .bottom-nav {
            padding: 4px 0 6px 0 !important;
        }

        .nav-item {
            padding: 8px 4px !important;
            font-size: 11px !important;
            min-width: 50px !important;
        }
    }

    /* Tablet responsiveness */
    @media (min-width: 768px) and (max-width: 1024px) {
        .main > div {
            max-width: 95% !important;
            padding: 12px 15px 120px 15px !important;
        }

        .chat-container {
            height: calc(100vh - 240px) !important;
            max-height: calc(100vh - 240px) !important;
        }

        .message-content {
            max-width: 85% !important;
        }
    }

    /* Large screen optimizations */
    @media (min-width: 1200px) {
        .main > div {
            max-width: 1000px !important;
        }

        .message-content {
            max-width: 75% !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Header with clear buttons
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    clear_col1, clear_col2 = st.columns(2)
    with clear_col1:
        if st.button(
            "Clear",
            key="clear_btn",
                help="Clear current conversation"):
            st.session_state.chat_messages = []
            save_chat_memory(st.session_state.chat_messages)
            st.rerun()

    with clear_col2:
        if st.button("Clear All", key="clear_all_btn", help="Clear all data"):
            st.session_state.chat_messages = []
            if hasattr(st.session_state, 'mltrainer_engine'):
                st.session_state.mltrainer_engine.trial_history = []
                st.session_state.mltrainer_engine._save_memory()
            save_chat_memory([])
            st.rerun()

# Header
st.markdown("""
<div class="header">
    <h1>mlTrainer2 Trading Intelligence</h1>
    <p>Advanced Financial AI with Autonomous ML Training & Genesis Conversations</p>
</div>
""", unsafe_allow_html=True)

# Status bar with WebSocket and Flask backend health
websocket_status = "🟢"
websocket_text = "WebSocket OK"
flask_status = "🟢"
flask_text = "Flask OK"

# Check WebSocket status
if os.path.exists("websocket_status.json"):
    try:
        with open("websocket_status.json", "r") as f:
            ws_status = json.load(f)
        if not ws_status.get("websocket_ok", False):
            websocket_status = "🔴"
            websocket_text = "WebSocket Error"
    except Exception:
        websocket_status = "🟡"
        websocket_text = "WebSocket Unknown"

# Check Flask backend health  
try:
    import requests
    # Flask runs on separate port from fix_ports_replit.py
    flask_port = os.environ.get("FLASK_RUN_PORT", "5000")
    response = requests.get(f"http://127.0.0.1:{flask_port}/health", timeout=2)
    if response.status_code != 200:
        flask_status = "🔴"
        flask_text = "Flask Error"
except Exception:
    flask_status = "🔴"
    flask_text = "Flask Down"

st.markdown(f"""
<div class="status-bar">
    <span class="status-indicator"></span>
    <span class="status-text">mlTrainer System - Active | {flask_status} {flask_text} | {websocket_status} {websocket_text}</span>
</div>
""", unsafe_allow_html=True)

# Chat container with native Streamlit components
chat_container = st.container(height=500)

with chat_container:
    for message in st.session_state.chat_messages:
        # Use Streamlit's native chat message component
        with st.chat_message(message["role"]):
            st.write(message["content"])

            # Display timestamp
            if message.get("timestamp"):
                st.caption(f"⏱️ {message['timestamp']}")

            # Handle file attachments
            if message.get("files"):
                for file_info in message["files"]:
                    if file_info["type"].startswith(
                            "image/") and os.path.exists(file_info["path"]):
                        try:
                            from PIL import Image
                            img = Image.open(file_info["path"])
                            img.thumbnail((400, 400))
                            st.image(img, caption=file_info["name"], width=300)
                        except Exception:
                            st.caption(
                                f"📷 {file_info['name']} (preview unavailable)")
                    else:
                        st.caption(
                            f"📎 {file_info['name']} ({file_info['size']:,} bytes)")

# File upload section (outside the chat input)
uploaded_files = st.file_uploader(
    "📎 Upload files (images/PDFs)",
    type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'pdf'],
    accept_multiple_files=True,
    key="file_uploader"
)

# Use Streamlit's native chat input
user_input = st.chat_input(
    "Tell me about the trial you want to begin, or ask about ML models, strategies...")

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

            # Save file to uploads directory
            uploads_dir = "uploads"
            if not os.path.exists(uploads_dir):
                os.makedirs(uploads_dir)

            file_path = os.path.join(uploads_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            file_details["path"] = file_path
            file_info.append(file_details)

    # Create user message content with file information
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

    # Save after user message
    save_chat_memory(st.session_state.chat_messages)

    try:
        # Process with mlTrainer following the data flow principles
        communication_status = st.empty()
        with st.spinner("mlTrainer is analyzing your request..."):
            # Smart symbol detection for ML analysis
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

            # Display communication status
            if trial_config:
                communication_status.info(
                    "🤖 mlTrainer <-> ML Engine communication initiated...")

            # Prepare prompt with file information if available
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

            # Send to mlTrainer (Claude) - works for any conversation
            result = mltrainer.start_trial(
                user_prompt=enhanced_prompt,
                trial_config=trial_config,
                chat_context=st.session_state.chat_messages)

            # Format response
            if "error" in result:
                response_content = f"❌ **Error**\n\n{result['error']}"
            else:
                # Get the main response from Claude
                response_content = result.get(
                    "response", "No response received")

                # If ML analysis was performed, add summary
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

        # Save after mlTrainer response
        save_chat_memory(st.session_state.chat_messages)

        # Display results
        st.markdown("---")

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

        # Display communication logs if available
        if hasattr(
                mltrainer_engine,
                '_communication_log') and mltrainer_engine._communication_log:
            with st.expander("🔄 mlTrainer <-> ML Engine Communication Log"):
                for i, log_entry in enumerate(
                        mltrainer_engine._communication_log[-5:]):  # Show last 5 entries
                    if log_entry.get("type") == "ml_to_trainer":
                        st.info(
                            f"**ML Engine → mlTrainer**: {log_entry.get('question', {}).get('message', 'Communication')}")
                    elif log_entry.get("type") == "trainer_to_ml":
                        st.success(
                            f"**mlTrainer → ML Engine**: {log_entry.get('response', 'Response')[:100]}...")

        if "error" in result:
            st.error(f"Analysis Error: {result['error']}")

        # Clear communication status
        communication_status.empty()

    except Exception as e:
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": f"❌ **System Error**\n\nI encountered an issue processing your request: {str(e)}\n\nPlease try rephrasing your request or contact support if the issue persists.",
            "timestamp": datetime.now().strftime("%H:%M")
        })

        # Save after error response
        save_chat_memory(st.session_state.chat_messages)

    st.rerun()

# Chat interface complete

# Enhanced JavaScript for chat functionality
st.markdown("""
<script>
// Enhanced auto-scroll with smooth behavior
function scrollToBottom(smooth = false) {
    const messagesContainer = document.querySelector('.messages');
    if (messagesContainer) {
        messagesContainer.scrollTo({
            top: messagesContainer.scrollHeight,
            behavior: smooth ? 'smooth' : 'auto'
        });
    }
}

// Debounced scroll function
let scrollTimeout;
function debouncedScroll() {
    clearTimeout(scrollTimeout);
    scrollTimeout = setTimeout(() => scrollToBottom(true), 100);
}

// Auto-scroll management
let isUserScrolling = false;
let lastScrollTop = 0;

function handleScroll() {
    const messagesContainer = document.querySelector('.messages');
    if (messagesContainer) {
        const currentScrollTop = messagesContainer.scrollTop;
        const maxScrollTop = messagesContainer.scrollHeight - messagesContainer.clientHeight;

        // Check if user is near bottom (within 50px)
        isUserScrolling = currentScrollTop < maxScrollTop - 50;
        lastScrollTop = currentScrollTop;
    }
}

// Initialize chat functionality
function initChat() {
    const messagesContainer = document.querySelector('.messages');
    if (messagesContainer) {
        messagesContainer.addEventListener('scroll', handleScroll);

        // Auto-scroll to bottom on load
        setTimeout(() => scrollToBottom(false), 100);

        // Observe DOM changes for new messages
        const observer = new MutationObserver(() => {
            if (!isUserScrolling) {
                debouncedScroll();
            }
        });

        observer.observe(messagesContainer, {
            childList: true,
            subtree: true,
            characterData: true
        });
    }
}

// Function to check WebSocket connection
function checkWebSocketConnection() {
    // Replace with the correct WebSocket endpoint for your Streamlit app
    const ws = new WebSocket(`ws://${window.location.hostname}:${window.location.port}/stream`);

    ws.onopen = () => {
        console.log('WebSocket connection established');
    };

    ws.onclose = () => {
        console.warn('WebSocket connection closed. Retrying in 3 seconds...');
        setTimeout(() => checkWebSocketConnection(), 3000);
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        initChat();
        checkWebSocketConnection();
    });
} else {
    initChat();
    checkWebSocketConnection();
}

// Re-initialize on Streamlit reruns
window.addEventListener('load', () => {
    setTimeout(() => {
        initChat();
        checkWebSocketConnection();
    }, 500);
});

// Handle textarea auto-resize
function setupTextareaResize() {
    const textarea = document.querySelector('.stTextArea textarea');
    if (textarea) {
        textarea.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 120) + 'px';
        });
    }
}

# Setup textarea on load
setTimeout(setupTextareaResize, 1000);
</script>
""", unsafe_allow_html=True)

# Sidebar configuration
engine = st.session_state.mltrainer_engine

with st.sidebar:
    st.sidebar.header("⚙️ Configuration")

    # Display compliance status
    compliance_status = "🔒 ACTIVE" if is_compliance_enabled() else "🔓 DISABLED"
    override_status = "🚨 OVERRIDE ACTIVE" if is_override_authorized() else "🔒 ENFORCED"
    st.sidebar.write(f"**Compliance Mode:** {compliance_status}")
    st.sidebar.write(f"**Immutable Gateway:** {override_status}")

    # CRITICAL: System Owner Override Controls
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 System Owner Controls")
    st.sidebar.markdown("*For maintenance, debugging & development only*")

    if is_override_authorized():
        st.sidebar.warning("🚨 OVERRIDE MODE ACTIVE")
        if st.sidebar.button("🔒 Restore Full Compliance"):
            disable_override()
            st.sidebar.success("Full compliance restored")
            st.rerun()
    else:
        st.sidebar.markdown("**Override Status:** 🔒 Enforced")
        if st.sidebar.button(
            "🚨 Disable All Compliance",
                key="disable_compliance_btn"):
            enable_override()
            st.sidebar.success("Override enabled - Maintenance mode active")
            st.rerun()
        st.sidebar.caption("⚠️ This disables ALL compliance enforcement")

    st.sidebar.markdown("---")

    # Display allowed APIs
    try:
        from core.immutable_gateway import load_allowed_apis
        allowed_apis = load_allowed_apis()
        st.sidebar.write(f"**Authorized APIs:** {', '.join(allowed_apis)}")
    except Exception:
        st.sidebar.write(f"**API Status:** ❌ Error loading config")

    # Display available models
    try:
        models_info = engine.get_available_models()
        model_list = ", ".join(models_info.get("models", {}).keys())
        st.sidebar.write(f"**Available Models:** {model_list}")
    except Exception:
        st.sidebar.write(f"**Models:** ❌ Error loading models")

# Bottom navigation
st.markdown("""
<div class="bottom-nav">
    <a href="#" class="nav-item active">
        <div class="nav-label">mlTrainer Chat</div>
    </a>
    <a href="#" class="nav-item">
        <div class="nav-label">Active Trials</div>
    </a>
    <a href="#" class="nav-item">
        <div class="nav-label">Recommendations</div>
    </a>
    <a href="#" class="nav-item">
        <div class="nav-label">Monitoring</div>
    </a>
    <a href="#" class="nav-item">
        <div class="nav-label">Compliance</div>
    </a>
</div>
""", unsafe_allow_html=True)
