import os
import sys
import subprocess

print("=== STARTING RUN_STREAMLIT.PY ===", flush=True)

# Get port from environment
port = os.environ.get('PORT', '8501')
print(f"PORT environment variable: {port}", flush=True)

# FORCE remove any STREAMLIT env vars
for key in list(os.environ.keys()):
    if key.startswith('STREAMLIT_'):
        print(f"Removing {key}={os.environ[key]}", flush=True)
        del os.environ[key]

print("Starting Streamlit...", flush=True)

# Use subprocess instead of os.system
subprocess.run([
    sys.executable, '-m', 'streamlit', 'run',
    'mltrainer_unified_chat.py',
    '--server.port', port,
    '--server.address', '0.0.0.0',
    '--server.headless', 'true'
])
