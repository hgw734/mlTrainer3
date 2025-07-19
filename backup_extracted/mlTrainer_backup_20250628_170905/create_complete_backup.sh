
#!/bin/bash

echo "🔄 Creating COMPLETE mlTrainer System Backup..."

# Create timestamped backup directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_DIR="mlTrainer_backup_${TIMESTAMP}"

echo "📦 Creating backup directory: ${BACKUP_DIR}"
mkdir -p "${BACKUP_DIR}"

# Backup all source code and configurations
echo "📁 Backing up core directories..."
cp -r core ml monitoring data_sources feature_engineering "${BACKUP_DIR}/" 2>/dev/null
cp -r strategy finance utils pages routes signals tests "${BACKUP_DIR}/" 2>/dev/null

# Backup all configuration files
echo "📄 Backing up configuration files..."
cp ai_config.json port_config.json chat_memory.json "${BACKUP_DIR}/" 2>/dev/null
cp requirements.txt pyproject.toml .replit "${BACKUP_DIR}/" 2>/dev/null
cp model_routing.yaml ensemble_weights.json "${BACKUP_DIR}/" 2>/dev/null

# Backup all Python scripts
echo "🐍 Backing up Python scripts..."
cp *.py "${BACKUP_DIR}/" 2>/dev/null

# Backup all shell scripts
echo "📜 Backing up shell scripts..."
cp *.sh "${BACKUP_DIR}/" 2>/dev/null

# Backup data and logs (if they exist)
echo "💾 Backing up data and logs..."
cp -r data logs uploads "${BACKUP_DIR}/" 2>/dev/null

# Backup Streamlit config
echo "🎨 Backing up Streamlit configuration..."
cp -r .streamlit "${BACKUP_DIR}/" 2>/dev/null

# Create backup manifest
echo "📋 Creating backup manifest..."
cat > "${BACKUP_DIR}/backup_manifest.txt" << EOF
mlTrainer System Backup
Created: $(date)
Timestamp: ${TIMESTAMP}

Included Components:
- Core modules (core/, ml/, monitoring/)
- Data sources and feature engineering
- Strategy and finance modules
- Configuration files
- Python scripts and utilities
- Shell scripts and launchers
- Data and logs
- Streamlit configuration

Port Configuration Status:
- fix_ports_replit.py: $(test -f fix_ports_replit.py && echo "✅ Present" || echo "❌ Missing")
- port_config.json: $(test -f port_config.json && echo "✅ Present" || echo "❌ Missing")

System Status at Backup:
$(python3 fix_ports_replit.py 2>/dev/null | grep "✅" || echo "Port config check failed")
EOF

# Create archive
echo "📦 Creating compressed archive..."
tar -czf "${BACKUP_DIR}.tar.gz" "${BACKUP_DIR}"

# Summary
echo "✅ Backup completed successfully!"
echo "📁 Directory backup: ${BACKUP_DIR}/"
echo "📦 Compressed archive: ${BACKUP_DIR}.tar.gz"
echo ""
echo "📊 Backup Size:"
du -sh "${BACKUP_DIR}" "${BACKUP_DIR}.tar.gz"

echo ""
echo "🔧 To restore from backup:"
echo "1. Extract: tar -xzf ${BACKUP_DIR}.tar.gz"
echo "2. Copy files: cp -r ${BACKUP_DIR}/* ."
echo "3. Run port fix: python3 fix_ports_replit.py"
echo "4. Start system: bash start_system.sh"
EOF
