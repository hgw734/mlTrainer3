import re

# Read the file
with open("pages/0_🤖_mlTrainer_Chat.py", "r") as f:
    content = f.read()

# Replace the problematic section with proper AI client initialization
pattern = r'# Initialize background trial manager if needed\s+if not st\.session_state\.background_trial_manager:\s+st\.session_state\.background_trial_manager = get_background_trial_manager\(\s+st\.session_state\.executor,\s+ai_client\s+\)'

replacement = '''# Initialize background trial manager if needed
                            if not st.session_state.background_trial_manager:
                                try:
                                    from core.ai_client import AIClient
                                    ai_client = AIClient()
                                    st.session_state.background_trial_manager = get_background_trial_manager(
                                        st.session_state.executor, 
                                        ai_client
                                    )
                                except Exception as e:
                                    st.error(f"Failed to initialize background trial manager: {e}")
                                    st.session_state.background_trial_manager = None'''

# Replace all occurrences
content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)

# Write back
with open("pages/0_🤖_mlTrainer_Chat.py", "w") as f:
    f.write(content)

print("Fixed AI client initialization")
