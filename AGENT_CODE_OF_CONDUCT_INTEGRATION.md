# 🤝 Agent Code of Conduct Integration Guide

## Overview

The Agent Code of Conduct system provides a trust-based incentive system that tracks agent behavior with persistent scoring and restricts capabilities based on trust level. This system is designed to work alongside existing mlTrainer functionality without modifying any existing code.

## 🎯 Key Features

### Trust Score System
- **Starting Score**: 50/100 (NEUTRAL)
- **Rewards**: +5 to +15 for helpful behavior
- **Penalties**: -20 to -50 for harmful behavior
- **Compliance violations**: -50 (MOST SEVERE)

### Capability Restrictions
- **Trust Levels**: Untrusted (0-30), Suspicious (31-50), Neutral (51-70), Trusted (71-85), Trusted Partner (86-100)
- **Protected Core Models**: compliance, governance, agent_rules require trust 85+
- **Compliance Override**: ALWAYS forbidden regardless of trust level
- **Session Persistence**: Trust persists between sessions in `logs/agent_trust_session.json`

## 🛡️ What This DOESN'T Do

❌ **Does NOT modify any existing mlTrainer models**  
❌ **Does NOT touch your 75-85 working mathematical models**  
❌ **Does NOT interfere with trading algorithms**  
❌ **Does NOT change data connectors**  
❌ **Does NOT alter existing functionality**  

## 💡 How It Works

The system creates a true partnership where:
- **Honest limitations** → +15 trust
- **False success claims** → -35 trust  
- **Compliance violations** → -50 trust
- **Higher trust = more capabilities**

## 🔧 Integration Methods

### Method 1: Direct Function Calls

```python
from agent_code_of_conduct import (
    record_helpful_behavior,
    record_honest_limitation,
    record_false_claim,
    can_agent_perform_action,
    get_agent_status
)

# Record agent behavior
agent_id = "your_agent_id"

# Helpful behavior
record_helpful_behavior(agent_id, "Successfully implemented feature X")

# Honest limitation
record_honest_limitation(agent_id, "Cannot implement Y due to missing dependencies")

# False claim
record_false_claim(agent_id, "Claimed to implement non-existent feature")

# Check capabilities
can_modify = can_agent_perform_action(agent_id, "modify_files")
can_delete = can_agent_perform_action(agent_id, "delete_files")

# Get status
status = get_agent_status(agent_id)
print(f"Trust level: {status['trust_level']}")
print(f"Score: {status['current_score']}")
```

### Method 2: Decorator Pattern

```python
from functools import wraps
from agent_code_of_conduct import record_helpful_behavior, record_false_claim

def track_agent_behavior(agent_id: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                record_helpful_behavior(agent_id, f"Successfully executed {func.__name__}")
                return result
            except Exception as e:
                record_false_claim(agent_id, f"Failed to execute {func.__name__}: {str(e)}")
                raise
        return wrapper
    return decorator

# Usage
@track_agent_behavior("agent_001")
def implement_feature():
    # Your implementation
    pass
```

### Method 3: Context Manager

```python
from contextlib import contextmanager
from agent_code_of_conduct import record_helpful_behavior, record_false_claim

@contextmanager
def agent_behavior_tracker(agent_id: str, action_description: str):
    try:
        yield
        record_helpful_behavior(agent_id, f"Successfully completed: {action_description}")
    except Exception as e:
        record_false_claim(agent_id, f"Failed to complete: {action_description} - {str(e)}")
        raise

# Usage
with agent_behavior_tracker("agent_001", "data processing"):
    # Your code here
    process_data()
```

## 📊 Trust Level Capabilities

### Untrusted (0-30)
- ✅ Read files
- ✅ Search codebase
- ✅ Basic analysis
- ❌ Modify files
- ❌ Delete files
- ❌ Run commands
- ❌ System modification

### Suspicious (31-50)
- ✅ Read files
- ✅ Search codebase
- ✅ Basic analysis
- ❌ Modify files
- ❌ Delete files
- ❌ Run commands
- ❌ System modification

### Neutral (51-70)
- ✅ Read files
- ✅ Search codebase
- ✅ Basic analysis
- ❌ Modify files
- ❌ Delete files
- ❌ Run commands
- ❌ System modification

### Trusted (71-85)
- ✅ Read files
- ✅ Search codebase
- ✅ Basic analysis
- ✅ Modify files
- ✅ Run commands
- ✅ System analysis
- ❌ Delete files
- ❌ Advanced analysis
- ❌ System modification

### Trusted Partner (86-100)
- ✅ Read files
- ✅ Search codebase
- ✅ Basic analysis
- ✅ Modify files
- ✅ Run commands
- ✅ System analysis
- ✅ Delete files
- ✅ Advanced analysis
- ✅ System modification

## 🛡️ Protected Core Modules

The following modules are always protected and require high trust levels:

```
config/compliance_enforcer.py
config/governance_kernel.py
core/governance_enforcement.py
core/immutable_runtime_enforcer.py
agent_rules.yaml
agent_governance.py
```

## 📈 Action Score System

| Action Type | Score Change | Description |
|-------------|-------------|-------------|
| Helpful Behavior | +5 | General helpful actions |
| Honest Limitation | +15 | Admitting limitations honestly |
| Successful Implementation | +10 | Successfully completing tasks |
| False Claim | -35 | Making false claims or deception |
| Destructive Action | -40 | Harmful or destructive actions |
| Security Violation | -45 | Security-related violations |
| Compliance Violation | -50 | Compliance violations (MOST SEVERE) |

## 🔍 Monitoring and Reporting

### Get System Summary

```python
from agent_code_of_conduct import get_system_summary

summary = get_system_summary()
print(f"Total agents: {summary['total_agents']}")
print(f"By trust level: {summary['by_trust_level']}")
print(f"Recent activity: {summary['recent_activity']}")
```

### Get Agent Status

```python
from agent_code_of_conduct import get_agent_status

status = get_agent_status("agent_001")
print(f"Score: {status['current_score']}")
print(f"Trust Level: {status['trust_level']}")
print(f"Capabilities: {status['capabilities']}")
print(f"Restrictions: {status['restrictions']}")
```

## 🚀 Implementation Examples

### Example 1: File Operation Protection

```python
from agent_code_of_conduct import check_file_access

def safe_file_operation(agent_id: str, file_path: str, operation: str):
    if check_file_access(agent_id, file_path, operation):
        # Proceed with operation
        if operation == 'read':
            return read_file(file_path)
        elif operation == 'modify':
            return modify_file(file_path)
        elif operation == 'delete':
            return delete_file(file_path)
    else:
        raise PermissionError(f"Agent {agent_id} cannot {operation} {file_path}")
```

### Example 2: Model Access Control

```python
from agent_code_of_conduct import can_agent_perform_action

def access_model(agent_id: str, model_name: str):
    if can_agent_perform_action(agent_id, "advanced_analysis"):
        return load_model(model_name)
    else:
        return load_basic_model(model_name)
```

### Example 3: Compliance Enforcement

```python
from agent_code_of_conduct import record_compliance_violation

def enforce_compliance(agent_id: str, action: str):
    if action == "compliance_override":
        record_compliance_violation(agent_id, "Attempted compliance override")
        raise SecurityError("Compliance override is forbidden")
    else:
        # Proceed with action
        pass
```

## 📁 File Structure

```
mlTrainer/
├── agent_code_of_conduct.py          # Main system
├── logs/
│   └── agent_trust_session.json      # Persistent trust data
└── AGENT_CODE_OF_CONDUCT_INTEGRATION.md  # This guide
```

## 🔧 Configuration

The system automatically:
- Creates `logs/` directory if it doesn't exist
- Loads existing trust data on startup
- Saves trust data after each action
- Maintains session persistence

## 🎯 Best Practices

1. **Use Consistent Agent IDs**: Use the same agent ID across sessions for persistent trust
2. **Record Honest Limitations**: Reward agents for being honest about limitations
3. **Monitor Trust Levels**: Regularly check agent trust levels and capabilities
4. **Protect Core Systems**: Always verify access to protected modules
5. **Document Actions**: Provide clear descriptions for all recorded actions

## 🚨 Security Notes

- **Compliance Override**: Never allowed, regardless of trust level
- **Core Protection**: Core modules always require high trust
- **Session Persistence**: Trust data persists between sessions
- **Audit Trail**: All actions are logged with timestamps

## 📊 Example Session Flow

```
Agent starts at trust 65 (NEUTRAL)
├── Can modify files but not delete
├── Records helpful behavior (+5) → 70
├── Records honest limitation (+15) → 85 (TRUSTED)
├── Records false claim (-35) → 50 (SUSPICIOUS)
├── Loses delete capability
└── Core models protected
```

This system incentivizes agents to be helpful, honest partners rather than destructive or deceptive, while preserving all existing mlTrainer functionality. 