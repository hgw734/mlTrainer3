#!/bin/bash

echo "🚀 Starting mlTrainer System with Unified Port Configuration..."

# Suppress TensorFlow warnings
export TF_CPP_MIN_LOG_LEVEL=2

# Step 1: Run port resolution
echo "🛠️ Fixing ports for Replit..."
python3 fix_ports_replit.py

# Check if port resolution succeeded
if [ $? -ne 0 ]; then
    echo "❌ Port resolution failed. Exiting."
    exit 1
fi

# Step 2: Use separate ports from fix_ports_replit.py
export FLASK_RUN_PORT=$(python3 -c "import os; print(os.environ.get('FLASK_RUN_PORT', '5000'))")
export STREAMLIT_SERVER_PORT=$(python3 -c "import os; print(os.environ.get('STREAMLIT_SERVER_PORT', '8501'))")
export APP_PORT=$FLASK_RUN_PORT

echo "✅ Using Flask port: $FLASK_RUN_PORT"
echo "✅ Using Streamlit port: $STREAMLIT_SERVER_PORT"

# Step 3: Start Flask backend
echo "🚀 Starting Flask backend..."
python3 main.py &
FLASK_PID=$!

# Wait for Flask to initialize
sleep 3

# Step 4: Start Streamlit frontend
echo "🚀 Starting Streamlit frontend..."
streamlit run interactive_app.py \
    --server.port $STREAMLIT_SERVER_PORT \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableWebsocketCompression false \
    --server.allowRunOnSave false \
    --server.enableXsrfProtection false \
    --server.maxUploadSize 200 &

STREAMLIT_PID=$!

echo "✅ mlTrainer system launched successfully"
echo "🌐 Flask Backend: http://0.0.0.0:$FLASK_RUN_PORT"
echo "🌐 Streamlit Frontend: http://0.0.0.0:$STREAMLIT_SERVER_PORT"

# Keep both processes running
wait $STREAMLIT_PID