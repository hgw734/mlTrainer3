
#!/bin/bash

echo "🔧 Starting AdvanSng2 Full System with fix_ports_replit..."

### 1. Check Python ###
if ! command -v python3 &> /dev/null; then
  echo "❌ Python3 not found. Aborting."
  exit 1
fi

### 2. Use fix_ports_replit for all port management ###
echo "🛠️ Using fix_ports_replit for port configuration..."
python3 fix_ports_replit.py

# Check if port resolution succeeded
if [ $? -ne 0 ]; then
    echo "❌ Port resolution failed. Exiting."
    exit 1
fi

### 3. Get ports from fix_ports_replit ###
FLASK_PORT=$(python3 -c "import os; print(os.environ.get('FLASK_RUN_PORT', '5000'))")
STREAMLIT_PORT=$(python3 -c "import os; print(os.environ.get('STREAMLIT_SERVER_PORT', '8501'))")

echo "✅ Flask will use port: $FLASK_PORT"
echo "✅ Streamlit will use port: $STREAMLIT_PORT"

### 4. Fix API config ###
echo "🧠 Using centralized API configuration..."
echo "✅ API keys will be loaded from Replit Secrets via ai_config.json"

### 5. Create Streamlit config using dynamic port ###
echo "🔧 Creating Streamlit configuration for port $STREAMLIT_PORT..."
mkdir -p .streamlit

cat <<EOF > .streamlit/config.toml
[browser]
gatherUsageStats = false

[global]
disableWatchdogWarning = true

[server]
headless = true
enableXsrfProtection = false
enableWebsocketCompression = false
address = "0.0.0.0"
port = $STREAMLIT_PORT
maxUploadSize = 50
maxMessageSize = 200
enableStaticServing = false
runOnSave = false
allowRunOnSave = false

[theme]
base = "light"
EOF

### 6. Install dependencies ###
echo "📦 Installing WebSocket dependencies..."
pip install --upgrade websocket-client requests streamlit

### 7. Set environment variables from fix_ports_replit ###
export FLASK_RUN_PORT=$FLASK_PORT
export STREAMLIT_SERVER_PORT=$STREAMLIT_PORT
export STREAMLIT_SERVER_ADDRESS="0.0.0.0"
export STREAMLIT_SERVER_HEADLESS="true"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS="false"
export STREAMLIT_SERVER_ENABLE_CORS="false"
export STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION="false"

### 8. Start Flask backend ###
echo "🚀 Starting Flask backend on port $FLASK_PORT..."
python3 main.py &
FLASK_PID=$!

# Wait for Flask to initialize
sleep 3

### 9. Launch Streamlit ###
echo "🚀 Launching Streamlit application on port $STREAMLIT_PORT..."
streamlit run interactive_app.py \
  --server.port=$STREAMLIT_PORT \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableWebsocketCompression=false \
  --server.allowRunOnSave=false \
  --server.enableXsrfProtection=false \
  --browser.gatherUsageStats=false &

STREAMLIT_PID=$!

echo "✅ System launched with fix_ports_replit configuration"
echo "🌐 Flask Backend: http://0.0.0.0:$FLASK_PORT"
echo "🌐 Streamlit Frontend: http://0.0.0.0:$STREAMLIT_PORT"

# Keep both processes running
wait $STREAMLIT_PID
