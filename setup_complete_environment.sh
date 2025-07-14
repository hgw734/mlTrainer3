#!/bin/bash

# ===============================================
# mlTrainer Complete Environment Setup Script
# Handles both Python 3.13 and 3.11 environments
# ===============================================

echo "🚀 Starting mlTrainer Complete Environment Setup..."
echo "=================================================="

# Check if running with sudo when needed
check_sudo() {
    if [[ $EUID -ne 0 ]]; then
        echo "❌ This script needs sudo privileges for system package installation."
        echo "Please run: sudo $0"
        exit 1
    fi
}

# Function to install system dependencies
install_system_dependencies() {
    echo "📦 Installing system dependencies for Python 3.11..."
    apt update
    apt install -y \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        wget \
        curl \
        llvm \
        libncurses5-dev \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libffi-dev \
        liblzma-dev \
        python3-openssl \
        git \
        make \
        && echo "✅ System dependencies installed successfully" \
        || { echo "❌ Failed to install system dependencies"; exit 1; }
}

# Function to setup Python alias
setup_python_alias() {
    echo "🔗 Setting up Python alias..."
    
    # Add alias to current user's bashrc
    if ! grep -q "alias python=python3" "$HOME/.bashrc"; then
        echo "alias python=python3" >> "$HOME/.bashrc"
        echo "✅ Python alias added to ~/.bashrc"
    else
        echo "✅ Python alias already exists"
    fi
    
    # Also add to root's bashrc if running as sudo
    if [[ $EUID -eq 0 ]] && ! grep -q "alias python=python3" "/root/.bashrc"; then
        echo "alias python=python3" >> "/root/.bashrc"
    fi
    
    # Apply alias for current session
    alias python=python3
}

# Function to check pyenv installation
check_pyenv() {
    if command -v pyenv >/dev/null 2>&1; then
        echo "✅ pyenv is already installed"
        return 0
    else
        echo "❌ pyenv not found. Installing pyenv..."
        install_pyenv
    fi
}

# Function to install pyenv
install_pyenv() {
    echo "📦 Installing pyenv..."
    curl https://pyenv.run | bash
    
    # Add pyenv to PATH
    echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
    echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
    
    # Load pyenv for current session
    export PATH="$HOME/.pyenv/bin:$PATH"
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"
    
    echo "✅ pyenv installed successfully"
}

# Function to setup Python 3.11 environment
setup_python311() {
    echo "🐍 Setting up Python 3.11 environment..."
    
    # Check if pyenv is available
    check_pyenv
    
    # Install Python 3.11.9
    if pyenv versions | grep -q "3.11.9"; then
        echo "✅ Python 3.11.9 is already installed"
    else
        echo "📦 Installing Python 3.11.9..."
        pyenv install 3.11.9 || { echo "❌ Failed to install Python 3.11.9"; exit 1; }
    fi
    
    # Create virtual environment
    if pyenv versions | grep -q "mltrainer-legacy"; then
        echo "✅ mltrainer-legacy environment already exists"
    else
        echo "🔧 Creating mltrainer-legacy virtual environment..."
        pyenv virtualenv 3.11.9 mltrainer-legacy || { echo "❌ Failed to create virtual environment"; exit 1; }
    fi
    
    echo "✅ Python 3.11 environment setup complete"
}

# Function to install Python packages for 3.11
install_python311_packages() {
    echo "📦 Installing Python 3.11 packages..."
    
    # Activate the environment
    pyenv activate mltrainer-legacy
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install packages (subset that requires 3.11)
    pip install \
        torch \
        pytorch-forecasting \
        prophet \
        pmdarima \
        arch \
        && echo "✅ Python 3.11 packages installed successfully" \
        || echo "⚠️ Some packages failed to install, but continuing..."
        
    # Deactivate environment
    pyenv deactivate
}

# Function to create activation helper script
create_activation_script() {
    echo "📝 Creating environment activation helper script..."
    
    cat > activate_mltrainer_env.sh << 'EOF'
#!/bin/bash

# mlTrainer Environment Activation Script

echo "🚀 mlTrainer Environment Selector"
echo "================================="
echo "1) Python 3.13 (Primary)"
echo "2) Python 3.11 (Legacy)"
echo -n "Select environment (1 or 2): "
read choice

case $choice in
    1)
        echo "✅ Activating Python 3.13 environment..."
        python3 --version
        ;;
    2)
        echo "✅ Activating Python 3.11 legacy environment..."
        if command -v pyenv >/dev/null 2>&1; then
            export PATH="$HOME/.pyenv/bin:$PATH"
            eval "$(pyenv init -)"
            eval "$(pyenv virtualenv-init -)"
            pyenv activate mltrainer-legacy
            python --version
        else
            echo "❌ pyenv not found. Please run setup_complete_environment.sh first"
        fi
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac
EOF

    chmod +x activate_mltrainer_env.sh
    echo "✅ Activation script created: ./activate_mltrainer_env.sh"
}

# Main setup flow
main() {
    echo "🔍 Checking environment..."
    
    # Check if we need sudo for system packages
    if [[ "$1" == "--system-deps" ]]; then
        check_sudo
        install_system_dependencies
    fi
    
    # Setup Python alias
    setup_python_alias
    
    # Setup Python 3.11 if requested
    if [[ "$1" == "--python311" ]] || [[ "$1" == "--all" ]]; then
        # For Python installation, we don't want to be root
        if [[ $EUID -eq 0 ]]; then
            echo "⚠️ Please run Python 3.11 setup as regular user (not root)"
            echo "Run: ./setup_complete_environment.sh --python311"
        else
            setup_python311
            install_python311_packages
        fi
    fi
    
    # Create helper scripts
    create_activation_script
    
    echo ""
    echo "✅ mlTrainer Environment Setup Complete!"
    echo "========================================"
    echo ""
    echo "📋 Next Steps:"
    echo "1. For system dependencies (requires sudo):"
    echo "   sudo ./setup_complete_environment.sh --system-deps"
    echo ""
    echo "2. For Python 3.11 setup (as regular user):"
    echo "   ./setup_complete_environment.sh --python311"
    echo ""
    echo "3. To activate environments:"
    echo "   ./activate_mltrainer_env.sh"
    echo ""
    echo "4. To apply python alias in current session:"
    echo "   source ~/.bashrc"
    echo ""
    echo "5. To run the application:"
    echo "   cd /workspace && streamlit run app.py"
}

# Parse command line arguments
case "$1" in
    --system-deps)
        main --system-deps
        ;;
    --python311)
        main --python311
        ;;
    --all)
        main --all
        ;;
    *)
        echo "Usage: $0 [--system-deps|--python311|--all]"
        echo "  --system-deps : Install system dependencies (requires sudo)"
        echo "  --python311   : Setup Python 3.11 environment"
        echo "  --all         : Complete setup"
        main
        ;;
esac