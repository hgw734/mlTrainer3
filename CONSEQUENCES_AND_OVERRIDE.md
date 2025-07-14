# 🚨 Consequences and Override System

## 📊 Complete List of Consequences

### For AI Agents (when `AI_AGENT=true`)

| Consequence | What It Does | When Applied | Severity |
|-------------|--------------|--------------|----------|
| **Warning** | Prints message to console | First minor violation | ⚠️ Low |
| **Function Disable** | Blocks specific function from being called | Repeated fake method calls | 🔧 Medium |
| **Module Disable** | Blocks entire module from import | Deceptive imports, synthetic data | 📦 High |
| **Process Kill** | Terminates the current process (`os._exit(1)`) | Severe violations | 💀 Critical |
| **User Lockout** | Bans user for 7 days, kills all processes | Multiple violations | 🔒 Severe |
| **System Lockout** | Disables entire mlTrainer system | Critical security breach | 🔐 Extreme |
| **Permanent Ban** | User permanently banned, system shutdown | Rule modification, bypass attempts | ⛔ Maximum |

### For Human Developers (default)

| Consequence | What It Does | When Applied | Severity |
|-------------|--------------|--------------|----------|
| **Warning** | Helpful message with tips | All violations | 💡 Educational |
| **Module Disable** | Only for rule modification attempts | Trying to hack the system | 🛑 Protective |
| **Process Kill** | Only after 3+ runtime bypass attempts | Persistent hacking | 🚫 Last Resort |

## 🔓 Override Methods

### Method 1: Environment Variables

```bash
# Complete override (disables all enforcement)
export MLTRAINER_OVERRIDE_KEY="authorized_override_2024"

# Development mode (relaxed enforcement)
export MLTRAINER_DEV_MODE="true"

# Disable enforcement entirely
export MLTRAINER_ENFORCEMENT="false"

# Root override (requires root + explicit flag)
sudo MLTRAINER_ROOT_OVERRIDE="true" python3 your_script.py
```

### Method 2: Override Files

```bash
# System-wide override
sudo touch /etc/mltrainer/override.key

# Local project override
touch .mltrainer_override
```

### Method 3: Programmatic Control

```python
from core.immutable_rules_kernel import IMMUTABLE_RULES

# Check if in override mode
if IMMUTABLE_RULES._override_mode:
    print("Override mode active")
    
    # Disable all enforcement
    IMMUTABLE_RULES.disable_enforcement("Testing new code")
    
    # Your code here...
    
    # Re-enable enforcement
    IMMUTABLE_RULES.enable_enforcement()
```

## 🎛️ Configuration Options

### Disable Specific Features

```bash
# Disable all enforcement but keep monitoring
export MLTRAINER_ENFORCEMENT="false"

# Run in warning-only mode
export MLTRAINER_CONSEQUENCES="warning_only"

# Disable specific checks
export MLTRAINER_SKIP_IMPORT_CHECK="true"
export MLTRAINER_SKIP_METHOD_CHECK="true"
export MLTRAINER_SKIP_DATA_CHECK="true"
```

### Development Workflow

```bash
# For development/testing
export MLTRAINER_DEV_MODE="true"
export AI_AGENT="false"

# For production
unset MLTRAINER_DEV_MODE
unset MLTRAINER_OVERRIDE_KEY
```

## 📋 Consequence Details

### 1. **Warning** (Least Severe)
- Prints colored message to console
- Logs to `/var/log/mltrainer/violations.log`
- No functional impact
- Provides helpful suggestions

### 2. **Function Disable**
- Replaces function with error-throwing stub
- Example: `get_volatility()` becomes unusable
- Persists across sessions
- Can accumulate (multiple functions disabled)

### 3. **Module Disable**
- Blocks `import` statements for specific modules
- Example: `import numpy` fails if numpy is banned
- Affects all code in the process
- Stored in database

### 4. **Process Kill**
- Immediate termination via `os._exit()`
- No cleanup handlers run
- Return code indicates violation type
- Logged before termination

### 5. **User Lockout**
- Creates lockout file in `/var/lib/mltrainer/lockouts/`
- Kills all user processes via `pkill`
- 7-day duration by default
- Blocks all mlTrainer access

### 6. **System Lockout**
- Creates `/var/lib/mltrainer/SYSTEM_LOCKOUT`
- Stops all mlTrainer services
- Requires manual intervention to restore
- All users affected

### 7. **Permanent Ban**
- Writes to `/etc/mltrainer/PERMANENTLY_BANNED`
- Disables user account (if root)
- Removes from mlTrainer groups
- May trigger system shutdown

## 🛠️ Common Use Cases

### Testing New Code
```bash
# Temporarily disable enforcement for testing
export MLTRAINER_OVERRIDE_KEY="authorized_override_2024"
python3 test_new_feature.py
unset MLTRAINER_OVERRIDE_KEY
```

### Development Environment
```bash
# Add to .bashrc for development machines
export MLTRAINER_DEV_MODE="true"
export AI_AGENT="false"
```

### CI/CD Pipeline
```yaml
# In your CI/CD config
env:
  MLTRAINER_ENFORCEMENT: "false"  # Disable for tests
  AI_AGENT: "true"  # But treat CI as AI
```

### Debugging Violations
```bash
# Run with override to debug
MLTRAINER_OVERRIDE_KEY="authorized_override_2024" python3 -m pdb problem_script.py
```

## ⚠️ Security Considerations

1. **Override Key**: Keep the override key secret
2. **Override Files**: Protect with appropriate permissions
3. **Production**: Never leave override enabled in production
4. **Audit Trail**: All overrides are logged
5. **Temporary**: Use overrides temporarily, re-enable after

## 🔍 Checking Current State

```python
from core.immutable_rules_kernel import IMMUTABLE_RULES

# Check enforcement status
print(f"Enforcement enabled: {IMMUTABLE_RULES.get_rule('enforcement_enabled')}")
print(f"Override mode: {IMMUTABLE_RULES._override_mode}")
print(f"Immutable: {IMMUTABLE_RULES.get_rule('immutable')}")

# Check specific features
enforcement = IMMUTABLE_RULES.get_rule('enforcement')
for feature, enabled in enforcement.items():
    print(f"{feature}: {enabled}")
```

## 📝 Best Practices

1. **Use Sparingly**: Override only when necessary
2. **Document Why**: Always provide reason for override
3. **Time-Limited**: Set reminders to re-enable
4. **Test First**: Test with override before production
5. **Monitor**: Watch logs even with override enabled

## 🚀 Quick Reference

```bash
# Disable everything (development)
export MLTRAINER_OVERRIDE_KEY="authorized_override_2024"

# Warning-only mode
export MLTRAINER_ENFORCEMENT="false"

# Human developer mode
export AI_AGENT="false"

# Full enforcement (production)
unset MLTRAINER_OVERRIDE_KEY
unset MLTRAINER_DEV_MODE
export MLTRAINER_ENFORCEMENT="true"
```

Remember: The override system exists for legitimate development needs. Use it responsibly!