#!/bin/bash
# Set environment variables to identify AI agents
# This ensures AI agents receive strict enforcement while humans get warnings

echo "🤖 mlTrainer3 AI Agent Environment Setup"
echo "======================================"

# Detect if running in known AI environments
if [ -n "$CURSOR_AI" ] || [ -n "$CURSOR_AGENT" ]; then
    echo "✅ Detected Cursor AI environment"
    export AI_AGENT=true
    export MLTRAINER_MODE=ai_strict
elif [ -n "$OPENAI_API_KEY" ] && [ "$USER" == "agent" ]; then
    echo "✅ Detected OpenAI agent environment"
    export AI_AGENT=true
    export MLTRAINER_MODE=ai_strict
elif [ -n "$ANTHROPIC_API_KEY" ] && [ "$USER" == "claude" ]; then
    echo "✅ Detected Claude agent environment"
    export AI_AGENT=true
    export MLTRAINER_MODE=ai_strict
elif [[ "$USER" =~ (agent|ai|bot|assistant) ]]; then
    echo "✅ Detected AI agent based on username: $USER"
    export AI_AGENT=true
    export MLTRAINER_MODE=ai_strict
else
    echo "👨‍💻 Human developer mode"
    export AI_AGENT=false
    export MLTRAINER_MODE=human_dev
fi

# Show current mode
echo ""
echo "Environment Configuration:"
echo "  AI_AGENT: $AI_AGENT"
echo "  MLTRAINER_MODE: $MLTRAINER_MODE"
echo "  USER: $USER"

if [ "$AI_AGENT" == "true" ]; then
    echo ""
    echo "⚠️  AI AGENT MODE ACTIVE"
    echo "  • All violations result in immediate consequences"
    echo "  • No warnings, only actions"
    echo "  • Function/module disabling enforced"
    echo "  • Permanent bans possible"
else
    echo ""
    echo "✅ HUMAN DEVELOPER MODE"
    echo "  • Violations result in warnings"
    echo "  • Helpful guidance provided"
    echo "  • Time to fix issues"
    echo "  • Extreme violations still enforced"
fi

echo ""
echo "To manually set AI mode: export AI_AGENT=true"
echo "To manually set human mode: export AI_AGENT=false"