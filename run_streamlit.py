import os
import sys

print("=== STARTING RUN_STREAMLIT.PY ===", flush=True)
print(f"Python version: {sys.version}", flush=True)
print(f"Current directory: {os.getcwd()}", flush=True)
print(f"Files in directory: {os.listdir('.')}", flush=True)

# Get port from environment
port = os.environ.get('PORT', '8501')
print(f"PORT environment variable: {port}", flush=True)

# Check if mltrainer_unified_chat.py exists
if os.path.exists('mltrainer_unified_chat.py'):
    print("✓ mltrainer_unified_chat.py found", flush=True)
else:
    print("✗ mltrainer_unified_chat.py NOT FOUND!", flush=True)
    sys.exit(1)

print("Starting Streamlit...", flush=True)

# Run streamlit directly
cmd = f"streamlit run mltrainer_unified_chat.py --server.port {port} --server.address 0.0.0.0 --server.headless true"
print(f"Command: {cmd}", flush=True)

exit_code = os.system(cmd)
print(f"Streamlit exited with code: {exit_code}", flush=True)
