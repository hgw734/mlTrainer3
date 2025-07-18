import os
import sys
import subprocess

print("=== STARTING RUN.PY ===")
print(f"Python version: {sys.version}")
print(f"PORT from environment: {os.environ.get('PORT', 'NOT SET')}")
print(f"All environment variables: {list(os.environ.keys())}")

# Get PORT from environment or default to 8501
port = os.environ.get('PORT', '8501')
print(f"Using port: {port}")

# Run streamlit with the correct port
cmd = [
    sys.executable, '-m', 'streamlit', 'run',
    'mltrainer_unified_chat.py',
    '--server.port=' + port,  # Try concatenating instead
    '--server.address=0.0.0.0',
    '--server.headless=true'
]

print(f"Running command: {' '.join(cmd)}")
subprocess.run(cmd)
