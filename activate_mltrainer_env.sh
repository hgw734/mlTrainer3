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
