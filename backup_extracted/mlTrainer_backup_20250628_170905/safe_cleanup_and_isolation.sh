#!/bin/bash

echo "🔄 Starting SAFE CLEANUP and ISOLATION for AdvanSng2..."

# Step 1: Backup current project
echo "📦 Backing up current files to backup_before_cleanup/"
mkdir -p backup_before_cleanup
cp -r * backup_before_cleanup/ 2>/dev/null

# Step 2: Create new clean folder
echo "🧹 Creating clean workspace: advansng2_clean/"
mkdir -p advansng2_clean

# Step 3: Move critical folders and files
echo "📁 Moving source code folders..."
mv core ml monitoring data_sources advansng2_clean/ 2>/dev/null

echo "📄 Moving config and logic files..."
mv ai_config.json fmt_memory.json startup_diagnostics.py advansng2_clean/ 2>/dev/null
mv requirements.txt main.py app.py index.py advansng2_clean/ 2>/dev/null

# Step 4: Confirmation message
echo "✅ Done! Your clean system is now in: advansng2_clean/"
echo "👉 Run your diagnostics with: cd advansng2_clean && python3 startup_diagnostics.py"
