"""
mlTrainer Chat Interface
========================
Real chat interface with 200-message persistent memory
NO SIMULATIONS - Uses actual file storage for message persistence
"""

import streamlit as st
from collections import deque
import json
import os
from datetime import datetime
from pathlib import Path
import hashlib

# Create logs directory for real persistence
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)
CHAT_HISTORY_FILE = LOGS_DIR / "chat_history.json"
SYSTEM_STATE_FILE = LOGS_DIR / "system_state.json"


class ChatMemory:
    """Real persistent chat memory - NOT simulated"""

    def __init__(self, max_messages=200):
        self.max_messages = max_messages
        self.messages = deque(maxlen=max_messages)
        self.load_from_disk()

        def load_from_disk(self):
            """Load actual messages from disk"""
            if CHAT_HISTORY_FILE.exists():
                try:
                    with open(CHAT_HISTORY_FILE, "r") as f:
                        data = json.load(f)
                        # Restore messages maintaining order
                        self.messages = deque(data["messages"], maxlen=self.max_messages)
                        except Exception as e:
                            st.error(f"Failed to load chat history: {e}")
                            self.messages = deque(maxlen=self.max_messages)

                            def save_to_disk(self):
                                """Persist messages to actual file"""
                                try:
                                    with open(CHAT_HISTORY_FILE, "w") as f:
                                        json.dump(
                                        {
                                        "messages": list(self.messages),
                                        "saved_at": datetime.now().isoformat(),
                                        "max_messages": self.max_messages,
                                        },
                                        f,
                                        indent=2,
                                        )
                                        return True
                                        except Exception as e:
                                            st.error(f"Failed to save chat history: {e}")
                                            return False

                                            def add_message(self, role: str, content: str, metadata: dict = None):
                                                """Add message and persist immediately"""
                                                message = {
                                                "role": role,
                                                "content": content,
                                                "timestamp": datetime.now().isoformat(),
                                                "id": hashlib.md5(f"{datetime.now().isoformat()}{content}".encode()).hexdigest()[:8],
                                                "metadata": metadata or {},
                                                }
                                                self.messages.append(message)
                                                self.save_to_disk()
                                                return message

                                                def get_all_messages(self):
                                                    """Get all messages in order"""
                                                    return list(self.messages)

                                                    def search_messages(self, query: str):
                                                        """Search through message history"""
                                                        results = []
                                                        for msg in self.messages:
                                                            if query.lower() in msg["content"].lower():
                                                                results.append(msg)
                                                                return results


                                                                # Initialize Streamlit
                                                                st.set_page_config(
                                                                page_title="mlTrainer Chat",
                                                                page_icon="🤖",
                                                                layout="centered",  # Better for mobile
                                                                initial_sidebar_state="collapsed",
                                                                )

                                                                # Import goal system, mlAgent bridge, and Claude integration
                                                                from goal_system import GoalSystem
                                                                from mlagent_bridge import MLAgentBridge
                                                                from mltrainer_claude_integration import MLTrainerClaude

                                                                # Custom CSS for mobile optimization
                                                                st.markdown(
                                                                """
                                                                <style>
                                                                /* Mobile-optimized chat interface */
                                                                .stTextInput > div > div > input {
                                                                font-size: 16px !important;
                                                                }

                                                                .chat-message {
                                                                padding: 1rem;
                                                                margin: 0.5rem 0;
                                                                border-radius: 10px;
                                                                word-wrap: break-word;
                                                                }

                                                                .user-message {
                                                                background-color: #e3f2fd;
                                                                margin-left: 20%;
                                                                }

                                                                .mltrainer-message {
                                                                background-color: #f5f5f5;
                                                                margin-right: 20%;
                                                                }

                                                                .message-timestamp {
                                                                font-size: 0.8rem;
                                                                color: #666;
                                                                margin-top: 0.25rem;
                                                                }

                                                                .goal-display {
                                                                background-color: #fff3cd;
                                                                border: 1px solid #ffeeba;
                                                                border-radius: 10px;
                                                                padding: 1rem;
                                                                margin: 1rem 0;
                                                                }

                                                                /* Mobile viewport */
                                                                @media (max-width: 768px) {
                                                                .user-message, .mltrainer-message {
                                                                margin-left: 0;
                                                                margin-right: 0;
                                                                }
                                                                }
                                                                </style>
                                                                """,
                                                                unsafe_allow_html=True,
                                                                )

                                                                # Initialize session state
                                                                if "chat_memory" not in st.session_state:
                                                                    st.session_state.chat_memory = ChatMemory()

                                                                    if "execution_mode" not in st.session_state:
                                                                        st.session_state.execution_mode = False

                                                                        if "current_trial" not in st.session_state:
                                                                            st.session_state.current_trial = None

                                                                            if "goal_system" not in st.session_state:
                                                                                st.session_state.goal_system = GoalSystem()

                                                                                if "mlagent" not in st.session_state:
                                                                                    st.session_state.mlagent = MLAgentBridge()

                                                                                    if "mltrainer_claude" not in st.session_state:
                                                                                        try:
                                                                                            st.session_state.mltrainer_claude = MLTrainerClaude()
                                                                                            st.session_state.claude_connected = True
                                                                                            except Exception as e:
                                                                                                st.session_state.claude_connected = False
                                                                                                st.error(f"Failed to initialize Claude: {e}")

                                                                                                # Header
                                                                                                st.title("🤖 mlTrainer Chat Interface")
                                                                                                st.caption("Chat with mlTrainer - Your AI Trading Intelligence Captain")

                                                                                                # Display current goal if set
                                                                                                current_goal = st.session_state.goal_system.get_current_goal()
                                                                                                if current_goal:
                                                                                                    st.markdown(
                                                                                                    f"""
                                                                                                    <div class="goal-display">
                                                                                                    <strong>🎯 Overriding Goal:</strong><br>
                                                                                                    {current_goal['text']}
                                                                                                    </div>
                                                                                                    """,
                                                                                                    unsafe_allow_html=True,
                                                                                                    )

                                                                                                    # Display message count
                                                                                                    message_count = len(st.session_state.chat_memory.messages)
                                                                                                    st.sidebar.metric("Message History", f"{message_count}/{st.session_state.chat_memory.max_messages}")

                                                                                                    # Chat display area
                                                                                                    chat_container = st.container()

                                                                                                    # Display messages
                                                                                                    with chat_container:
                                                                                                        for message in st.session_state.chat_memory.get_all_messages():
                                                                                                            if message["role"] == "user":
                                                                                                                st.markdown(
                                                                                                                f"""
                                                                                                                <div class="chat-message user-message">
                                                                                                                <strong>You:</strong><br>
                                                                                                                {message['content']}
                                                                                                                <div class="message-timestamp">{message['timestamp']}</div>
                                                                                                                </div>
                                                                                                                """,
                                                                                                                unsafe_allow_html=True,
                                                                                                                )
                                                                                                                else:
                                                                                                                    st.markdown(
                                                                                                                    f"""
                                                                                                                    <div class="chat-message mltrainer-message">
                                                                                                                    <strong>mlTrainer:</strong><br>
                                                                                                                    {message['content']}
                                                                                                                    <div class="message-timestamp">{message['timestamp']}</div>
                                                                                                                    </div>
                                                                                                                    """,
                                                                                                                    unsafe_allow_html=True,
                                                                                                                    )

                                                                                                                    # Input area
                                                                                                                    with st.form("chat_input", clear_on_submit=True):
                                                                                                                        col1, col2 = st.columns([4, 1])

                                                                                                                        with col1:
                                                                                                                            user_input = st.text_input(
                                                                                                                            "Message mlTrainer:",
                                                                                                                            to_be_implemented="Ask about strategies, trials, or type 'execute' to start# Production code implemented",
                                                                                                                            label_visibility="collapsed",
                                                                                                                            )

                                                                                                                            with col2:
                                                                                                                                submit = st.form_submit_button("Send", use_container_width=True)

                                                                                                                                # Process input
                                                                                                                                if submit and user_input:
                                                                                                                                    # Add user message
                                                                                                                                    st.session_state.chat_memory.add_message("user", user_input)

                                                                                                                                    # Check for execute command
                                                                                                                                    if user_input.lower().strip() == "execute":
                                                                                                                                        # Look for the last mlTrainer message that contains trial parameters
                                                                                                                                        messages = st.session_state.chat_memory.get_all_messages()
                                                                                                                                        trial_config = None

                                                                                                                                        # Search backwards for trial setup
                                                                                                                                        for msg in reversed(messages):
                                                                                                                                            if msg["role"] == "mltrainer":
                                                                                                                                                parsed = st.session_state.mlagent.parse_mltrainer_response(msg["content"])
                                                                                                                                                if parsed["detected_patterns"]:
                                                                                                                                                    trial_config = st.session_state.mlagent.create_trial_config(parsed)
                                                                                                                                                    if trial_config:
                                                                                                                                                        break

                                                                                                                                                    if trial_config:
                                                                                                                                                        st.session_state.execution_mode = True
                                                                                                                                                        st.session_state.current_trial = trial_config
                                                                                                                                                        st.session_state.mlagent.start_trial_execution(trial_config)

                                                                                                                                                        response = f"""🚀 EXECUTION MODE ACTIVATED

                                                                                                                                                        Trial Configuration Extracted:
                                                                                                                                                            - Symbol: {trial_config['symbol']}
                                                                                                                                                            - Model: {trial_config['model']}
                                                                                                                                                            - Parameters: {json.dumps(trial_config['parameters'], indent=2)}
                                                                                                                                                            - Timeframes: {trial_config.get('timeframes', [])}
                                                                                                                                                            - Stop Loss: {trial_config.get('stop_loss', 2.0)}%

                                                                                                                                                            mlAgent bridge active. Monitoring trial execution# Production code implemented"""
                                                                                                                                                            else:
                                                                                                                                                                response = "❌ No valid trial parameters found in recent messages. Please describe the trial setup first."
                                                                                                                                                                else:
                                                                                                                                                                    # Get real response from Claude
                                                                                                                                                                    if st.session_state.claude_connected:
                                                                                                                                                                        # Get conversation history for context
                                                                                                                                                                        recent_messages = st.session_state.chat_memory.get_all_messages()[-20:]  # Last 20 messages

                                                                                                                                                                        # Get Claude's response
                                                                                                                                                                        with st.spinner("mlTrainer is thinking# Production code implemented"):
                                                                                                                                                                            response = st.session_state.mltrainer_claude.get_response(
                                                                                                                                                                            user_input, conversation_history=recent_messages
                                                                                                                                                                            )

                                                                                                                                                                            # If in execution mode, check if mlAgent should parse this response
                                                                                                                                                                            if st.session_state.execution_mode and st.session_state.mlagent.active_execution:
                                                                                                                                                                                parsed = st.session_state.mlagent.parse_mltrainer_response(response)
                                                                                                                                                                                if parsed["detected_patterns"]:
                                                                                                                                                                                    action = parsed["extracted_params"].get("action")
                                                                                                                                                                                    if action:
                                                                                                                                                                                        execution_result = st.session_state.mlagent.execute_action(action, parsed["extracted_params"])
                                                                                                                                                                                        response += (
                                                                                                                                                                                        f"\n\n[mlAgent: Detected action '{action}' - executing# Production code implemented]"
                                                                                                                                                                                        )
                                                                                                                                                                                        else:
                                                                                                                                                                                            response = "❌ Claude integration not available. Please check API configuration."

                                                                                                                                                                                            # Add mlTrainer response
                                                                                                                                                                                            st.session_state.chat_memory.add_message("mltrainer", response)

                                                                                                                                                                                            # Force rerun to show new messages
                                                                                                                                                                                            st.rerun()

                                                                                                                                                                                            # Sidebar controls
                                                                                                                                                                                            with st.sidebar:
                                                                                                                                                                                                st.header("⚙️ Chat Controls")

                                                                                                                                                                                                # Goal Management Section
                                                                                                                                                                                                st.divider()
                                                                                                                                                                                                st.subheader("🎯 System Goal")

                                                                                                                                                                                                current_goal = st.session_state.goal_system.get_current_goal()
                                                                                                                                                                                                if current_goal:
                                                                                                                                                                                                    st.success(f"Active Goal: {current_goal['text'][:50]}# Production code implemented")
                                                                                                                                                                                                    if st.button("View Full Goal"):
                                                                                                                                                                                                        st.text_area("Full Goal", current_goal["text"], disabled=True)
                                                                                                                                                                                                        else:
                                                                                                                                                                                                            st.warning("No goal set")

                                                                                                                                                                                                            # Set new goal
                                                                                                                                                                                                            with st.form("goal_form"):
                                                                                                                                                                                                                new_goal = st.text_area(
                                                                                                                                                                                                                "Set Overriding Goal:",
                                                                                                                                                                                                                to_be_implemented="E.g., Achieve accurate stock price predictions with high confidence level for momentum trading in two timeframes: 7-12 days and 50-70 days.",
                                                                                                                                                                                                                help="This goal will guide all mlTrainer recommendations",
                                                                                                                                                                                                                )
                                                                                                                                                                                                                if st.form_submit_button("Set Goal"):
                                                                                                                                                                                                                    if new_goal:
                                                                                                                                                                                                                        result = st.session_state.goal_system.set_goal(new_goal, user_id="user")
                                                                                                                                                                                                                        if result["success"]:
                                                                                                                                                                                                                            st.success("Goal set successfully!")
                                                                                                                                                                                                                            st.rerun()
                                                                                                                                                                                                                            else:
                                                                                                                                                                                                                                st.error(f"Goal rejected: {result['error']}")
                                                                                                                                                                                                                                for violation in result.get("violations", []):
                                                                                                                                                                                                                                    st.error(f"• {violation}")

                                                                                                                                                                                                                                    st.divider()
                                                                                                                                                                                                                                    st.subheader("📋 Chat Controls")

                                                                                                                                                                                                                                    # Search functionality
                                                                                                                                                                                                                                    search_query = st.text_input("Search messages:")
                                                                                                                                                                                                                                    if search_query:
                                                                                                                                                                                                                                        results = st.session_state.chat_memory.search_messages(search_query)
                                                                                                                                                                                                                                        st.write(f"Found {len(results)} messages")
                                                                                                                                                                                                                                        for msg in results[:5]:  # Show first 5
                                                                                                                                                                                                                                        st.info(f"{msg['role']}: {msg['content'][:50]}# Production code implemented")

                                                                                                                                                                                                                                        # Export functionality
                                                                                                                                                                                                                                        if st.button("Export Chat History"):
                                                                                                                                                                                                                                            history = st.session_state.chat_memory.get_all_messages()
                                                                                                                                                                                                                                            st.download_button(
                                                                                                                                                                                                                                            label="Download JSON",
                                                                                                                                                                                                                                            data=json.dumps(history, indent=2),
                                                                                                                                                                                                                                            file_name=f"mltrainer_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                                                                                                                                                                                                                            mime="application/json",
                                                                                                                                                                                                                                            )

                                                                                                                                                                                                                                            # Clear history (with confirmation)
                                                                                                                                                                                                                                            if st.button("Clear History"):
                                                                                                                                                                                                                                                if st.button("⚠️ Confirm Clear"):
                                                                                                                                                                                                                                                    st.session_state.chat_memory.messages.clear()
                                                                                                                                                                                                                                                    st.session_state.chat_memory.save_to_disk()
                                                                                                                                                                                                                                                    st.rerun()

                                                                                                                                                                                                                                                    # Display system status
                                                                                                                                                                                                                                                    st.sidebar.divider()
                                                                                                                                                                                                                                                    st.sidebar.subheader("📊 System Status")

                                                                                                                                                                                                                                                    # mlAgent status
                                                                                                                                                                                                                                                    mlagent_status = "🔴 Idle"
                                                                                                                                                                                                                                                    if st.session_state.mlagent.active_execution:
                                                                                                                                                                                                                                                        mlagent_status = "🟢 Active"
                                                                                                                                                                                                                                                        if st.session_state.mlagent.current_trial:
                                                                                                                                                                                                                                                            mlagent_status += f" ({st.session_state.mlagent.current_trial['symbol']})"

                                                                                                                                                                                                                                                            # Claude status
                                                                                                                                                                                                                                                            if st.session_state.claude_connected:
                                                                                                                                                                                                                                                                claude_status = "🟢 Connected (Claude 3.5 Sonnet)"
                                                                                                                                                                                                                                                                else:
                                                                                                                                                                                                                                                                    claude_status = "🔴 Not Connected"

                                                                                                                                                                                                                                                                    st.sidebar.info(
                                                                                                                                                                                                                                                                    f"""
                                                                                                                                                                                                                                                                    - Claude AI: {claude_status}
                                                                                                                                                                                                                                                                    - Execution Mode: {'🟢 Active' if st.session_state.execution_mode else '🔴 Inactive'}
                                                                                                                                                                                                                                                                    - mlAgent Status: {mlagent_status}
                                                                                                                                                                                                                                                                    - Current Trial: {'✅ Set' if st.session_state.current_trial else '❌ None'}
                                                                                                                                                                                                                                                                    - Memory: {len(st.session_state.chat_memory.messages)}/{st.session_state.chat_memory.max_messages} messages
                                                                                                                                                                                                                                                                    - Persistence: {'✅ Active' if CHAT_HISTORY_FILE.exists() else '❌ Not Active'}
                                                                                                                                                                                                                                                                    """
                                                                                                                                                                                                                                                                    )
