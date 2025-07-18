#!/bin/bash

# Get the PORT environment variable, default to 8501 if not set
PORT="${PORT:-8501}"

# Debug: Print the PORT value
echo "Starting Streamlit on port: $PORT"

# Run streamlit with the evaluated port
exec streamlit run mltrainer_unified_chat.py --server.port=$PORT --server.address=0.0.0.0
