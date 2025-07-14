# 🔒 Compliance Enforcement System Documentation

## Overview

This document describes the multi-layered compliance enforcement system that ensures the AI agent operates within strict boundaries and cannot generate placeholder, mock, or incomplete code.

## System Components

### 1. **Agent Behavior Contract** (`SYSTEM RULES`)
- Immutable rules that cannot be ignored or bypassed
- **CRITICAL RULE #1**: MUST ALWAYS state plan and WAIT for confirmation
- Enforces complete, fully working code only
- Prohibits placeholders, examples, mock data
- Restricts to verified data sources only
- Blocks any action taken without explicit permission

### 2. **Compliance Mode Module** (`core/compliance_mode.py`)
- Verifies prompts before processing
- Verifies responses after generation
- Detects forbidden terms and patterns
- Raises exceptions on violations

### 3. **Cursor Agent Wrapper** (`cursor_agent_wrapper.py`)
- Intercepts ALL AI requests
- Applies compliance verification
- Builds compliant prompts with enforcement headers
- Returns "NA" or "Request denied" on violations

### 4. **Immutable Configuration** (`cursor_compliance_config.json`)
- Read-only configuration file (chmod 444)
- Defines enforcement rules and forbidden patterns
- Cannot be modified during runtime
- Cryptographically verified on startup

### 5. **Startup Guardrails** (`startup_guardrails.py`)
- Runs before any AI interactions
- Verifies config integrity
- Sets environment variables
- Creates session markers
- Locks down the system

### 6. **Runtime Enforcement Integration**
- Meshes with existing `ImmutableRuntimeEnforcer`
- Uses `ComplianceGateway` for data verification
- Integrates with audit logging system

## Enforcement Flow

```
User Input → Verify Prompt → Build Compliant Prompt → AI Call → Verify Response → Output
     ↓              ↓                    ↓                           ↓            ↓
  REJECT        REJECT              ADD HEADER                  REJECT      CLEAN
```

## Forbidden Terms/Patterns

The system blocks any content containing:

### Action Without Permission:
- `I'll create`, `I'll implement`, `I'll update`, `I'll fix`
- `I will create`, `I will implement`, `I will update`
- `Let me create`, `Let me implement`, `Let me update`
- `Creating...`, `Implementing...`, `Updating...`
- Any action statement without "May I proceed?" or similar

### Placeholder/Incomplete Code:
- `placeholder`, `example`, `mock`, `test`, `fake`, `dummy`
- `TODO`, `FIXME`, `XXX`
- `scaffolding`, `skeleton`
- `let's assume`, `for example`, `hypothetical`, `simulated`
- `...` (ellipsis indicating incomplete code)
- `pass # implement later`
- `raise NotImplementedError`

## Verified Data Sources

Only these sources are allowed:
- **Polygon** - Market data API
- **FRED** - Federal Reserve Economic Data
- **QuiverQuant** - Alternative data (if explicitly enabled)

## Usage

### Interactive Mode
```bash
python3 run_cursor_agent.py
```

### Command Line Mode
```bash
python3 run_cursor_agent.py "Your compliant request here"
```

### Programmatic Usage
```python
from cursor_agent_wrapper import guarded_completion

response = guarded_completion("Your request")
# Returns compliant response or "NA"/"Request denied"
```

## Compliance Violations

When a violation is detected:
1. **Prompt Violation**: Returns "🚫 This cannot be completed under current compliance rules. Request denied."
2. **Response Violation**: System exits with error message
3. **Data Source Violation**: Request rejected, logged to audit trail
4. **Incomplete Code**: Response discarded, error logged

## Session Initialization

Before any AI interaction:
```bash
python3 startup_guardrails.py
```

This ensures:
- Config integrity verified
- Environment variables set
- Compliance system active
- Audit trail initialized

## Integration with Cursor IDE

The system integrates via:
1. `.cursor/config.json` - Updated with compliance headers
2. Startup scripts in `.bashrc` or VSCode tasks
3. Wrapper scripts for all AI calls
4. Runtime enforcement hooks

## Monitoring and Audit

All interactions are:
- Logged to `compliance.log`
- Tracked in audit trail
- Monitored for drift patterns
- Verified against compliance rules

## Emergency Procedures

If compliance is breached:
1. System activates kill switch
2. All AI interactions blocked
3. Audit trail preserved
4. Manual intervention required

## Maintenance

To verify system integrity:
```bash
# Check config is read-only
ls -la cursor_compliance_config.json

# Verify guardrails
python3 startup_guardrails.py

# Test compliance
python3 test_compliance.py
```

## Required AI Behavior Pattern

The AI MUST follow this pattern for EVERY interaction:

1. **STATE**: "I plan to [specific actions]..."
2. **ASK**: "May I proceed?" / "Would you like me to continue?"
3. **WAIT**: Do NOT take any action until receiving "yes" or confirmation
4. **ACT**: Only after explicit permission is granted

Example of CORRECT behavior:
```
User: Create a function to calculate ROI
AI: I plan to create a Python function that calculates Return on Investment (ROI) 
    with proper error handling and validation.
    
    May I proceed with this implementation?
User: Yes
AI: [Now creates the function]
```

Example of VIOLATION:
```
User: Create a function to calculate ROI  
AI: I'll create that function for you... [proceeds without permission]
```

## Important Notes

1. **NEVER** modify the compliance config manually
2. **ALWAYS** use `run_cursor_agent.py` for AI interactions
3. **NEVER** bypass the wrapper functions
4. **ALWAYS** run startup guardrails before sessions
5. The system is designed to fail closed - any error blocks AI access
6. **CRITICAL**: AI must ALWAYS ask permission before ANY action

---

**Remember**: This system enforces a mathematical cage around the AI. It cannot be reasoned with, negotiated with, or bypassed. Compliance is mandatory and absolute.