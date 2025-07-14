#!/bin/bash
# mlTrainer3 Immutable Compliance System Deployment Script
# Deploy with extreme caution - this system has real consequences

set -e  # Exit on error

echo "🔒 mlTrainer3 Immutable Compliance System Deployment"
echo "==================================================="

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root (required for system directories)
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}This script must be run as root for production deployment${NC}"
   echo "Run: sudo ./deploy_immutable_compliance.sh"
   exit 1
fi

# Confirm deployment
echo -e "${YELLOW}⚠️  WARNING: This will deploy the IMMUTABLE compliance system${NC}"
echo "Once activated:"
echo "  • Violations have REAL consequences"
echo "  • No warnings, only actions"
echo "  • Cannot be disabled"
echo ""
read -p "Are you ABSOLUTELY sure you want to proceed? (type 'DEPLOY' to confirm): " confirm

if [ "$confirm" != "DEPLOY" ]; then
    echo "Deployment cancelled"
    exit 1
fi

echo ""
echo "🔍 Checking prerequisites..."

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo -e "${RED}❌ Python 3.8+ required (found $python_version)${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python $python_version${NC}"

# Check Docker (optional but recommended)
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✅ Docker available${NC}"
    DOCKER_AVAILABLE=true
else
    echo -e "${YELLOW}⚠️  Docker not found (some features will be limited)${NC}"
    DOCKER_AVAILABLE=false
fi

echo ""
echo "📦 Installing dependencies..."

# Install system packages
apt-get update
apt-get install -y libc6-dev gcc python3-pip python3-venv

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv /opt/mltrainer3-venv
source /opt/mltrainer3-venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-immutable.txt

echo ""
echo "📁 Creating system directories..."

# Create system directories with proper permissions
directories=(
    "/var/log/mltrainer"
    "/var/lib/mltrainer"
    "/var/lib/mltrainer/lockouts"
    "/etc/mltrainer"
)

for dir in "${directories[@]}"; do
    mkdir -p "$dir"
    chmod 755 "$dir"
    echo -e "${GREEN}✅ Created $dir${NC}"
done

echo ""
echo "🔧 Deploying compliance components..."

# Copy core components
cp -r core /opt/mltrainer3/
cp -r scripts /opt/mltrainer3/
cp test_immutable_kernel.py /opt/mltrainer3/

# Set proper permissions
chmod -R 755 /opt/mltrainer3
chmod +x /opt/mltrainer3/scripts/activate_immutable_compliance.py

echo ""
echo "🚀 Activating immutable compliance system..."

cd /opt/mltrainer3
python3 scripts/activate_immutable_compliance.py

echo ""
echo "🐳 Building Docker validation image..."

if [ "$DOCKER_AVAILABLE" = true ]; then
    docker build -f Dockerfile.immutable -t mltrainer3/validation:latest .
    echo -e "${GREEN}✅ Docker image built${NC}"
else
    echo -e "${YELLOW}⚠️  Skipping Docker image (Docker not available)${NC}"
fi

echo ""
echo "📝 Creating systemd service..."

cat > /etc/systemd/system/mltrainer3-compliance.service << EOF
[Unit]
Description=mlTrainer3 Immutable Compliance Monitor
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/mltrainer3
Environment="PYTHONPATH=/opt/mltrainer3"
ExecStart=/opt/mltrainer3-venv/bin/python -c "from core import IMMUTABLE_COMPLIANCE_ACTIVE; print('Compliance Active:', IMMUTABLE_COMPLIANCE_ACTIVE); import time; time.sleep(86400)"
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable mltrainer3-compliance.service
systemctl start mltrainer3-compliance.service

echo ""
echo "🔍 Verifying deployment..."

# Run verification
cd /opt/mltrainer3
if ./verify_compliance.py; then
    echo -e "${GREEN}✅ Deployment verification passed${NC}"
else
    echo -e "${RED}❌ Deployment verification failed${NC}"
    exit 1
fi

echo ""
echo "📋 Setting up cron jobs..."

# Add hourly compliance check
echo "0 * * * * root cd /opt/mltrainer3 && /opt/mltrainer3-venv/bin/python verify_compliance.py >> /var/log/mltrainer/compliance-check.log 2>&1" > /etc/cron.d/mltrainer3-compliance

echo ""
echo "🎯 Final steps..."

# Create profile script
cat > /etc/profile.d/mltrainer3.sh << 'EOF'
# mlTrainer3 Immutable Compliance
export PYTHONSTARTUP=/opt/mltrainer3/mltrainer_compliance_startup.py
export MLTRAINER_ENFORCEMENT_LEVEL=STRICT

# Warning on login
echo ""
echo "🔒 mlTrainer3 Immutable Compliance System ACTIVE"
echo "   • All code operations are monitored"
echo "   • Violations have immediate consequences"
echo "   • No exemptions or bypasses"
echo ""
EOF

chmod +x /etc/profile.d/mltrainer3.sh

echo ""
echo "============================================="
echo -e "${GREEN}✅ DEPLOYMENT COMPLETE${NC}"
echo "============================================="
echo ""
echo "The mlTrainer3 Immutable Compliance System is now:"
echo "  • Monitoring all Python operations"
echo "  • Enforcing strict compliance rules"
echo "  • Recording all violations"
echo "  • Ready to enforce consequences"
echo ""
echo "⚠️  CRITICAL REMINDERS:"
echo "  • Test all code before deployment"
echo "  • Violations result in immediate action"
echo "  • Repeated violations lead to bans"
echo "  • This system cannot be disabled"
echo ""
echo "System service: systemctl status mltrainer3-compliance"
echo "Logs: /var/log/mltrainer/"
echo "Verify: /opt/mltrainer3/verify_compliance.py"
echo ""
echo "May your code be compliant! 🚀"