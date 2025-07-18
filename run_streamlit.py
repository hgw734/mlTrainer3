import os
import sys

# Get port from environment
port = os.environ.get('PORT', '8501')
print(f"Starting Streamlit on port {port}", flush=True)

# Run streamlit directly
os.system(f"streamlit run mltrainer_unified_chat.py --server.port {port} --server.address 0.0.0.0 --server.headless true")
