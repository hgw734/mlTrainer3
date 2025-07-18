import os
import sys
import subprocess

# Get PORT from environment or default to 8501
port = os.environ.get('PORT', '8501')
print(f"Starting Streamlit on port {port}")

# Run streamlit with the correct port
cmd = [
    'streamlit', 'run',
    'mltrainer_unified_chat.py',
    '--server.port', port,
    '--server.address', '0.0.0.0',
    '--server.headless', 'true',
    '--browser.gatherUsageStats', 'false'
]

subprocess.run(cmd)
