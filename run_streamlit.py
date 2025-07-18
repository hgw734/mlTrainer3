#!/usr/bin/env python3
import os
import sys
import subprocess

# Get the PORT from environment variable, default to 8501
port = os.environ.get('PORT', '8501')

# Run streamlit with the correct configuration
cmd = [
    sys.executable, '-m', 'streamlit', 'run',
    'mltrainer_unified_chat.py',
    '--server.port', port,
    '--server.address', '0.0.0.0',
    '--server.headless', 'true'
]

# Execute the command
subprocess.run(cmd)
