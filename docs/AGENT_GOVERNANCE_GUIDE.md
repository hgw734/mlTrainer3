# Agent Governance & Guardrails Guide

## Overview

The mlTrainer Agent Governance system ensures AI assistants operate within strict behavioral boundaries, maintaining safety, predictability, and trustworthiness.

## Architecture

```
mlTrainer/
├── agent_rules.yaml           # Behavioral rules (single source of truth)
├── agent_governance.py        # Rules enforcement engine
├── example_governed_agent.py  # Usage example
└── docs/
    └── AGENT_GOVERNANCE_GUIDE.md  # This guide
```

## Core Components

### 1. `agent_rules.yaml`
Central configuration defining ALL behavioral constraints:
- Permission protocols
- Data authenticity requirements
- Transparency rules
- Anti-drift protection
- Change discipline
- Communication standards

### 2. `agent_governance.py`
Python module that:
- Loads rules from YAML
- Enforces constraints
- Provides validation APIs
- Maintains audit logs
- Generates compliance reports

### 3. Integration Pattern

```python
from agent_governance import get_governance, governed_action

# Get governance instance
governance = get_governance()

# Check permission
if governance.check_permission_required("modify_file"):
    permission_msg = governance.format_permission_request(
        action="update configuration",
        impact="change system behavior",
        files=["config.yaml"]
    )
    # Wait for user permission...

# Validate data sources
if not governance.validate_data_source("random_generator"):
    raise ValueError("Synthetic data not allowed")

# Use decorator for automatic governance
@governed_action("code_modification")
def modify_code(file_path, changes):
    # This function automatically checks governance rules
    pass
```

## Key Guardrails Implemented

### 1. Permission Protocol ✅
- **Rule**: Never act without explicit permission
- **Implementation**: `check_permission_required()`, `format_permission_request()`
- **Example**: Must ask before any file modification

### 2. Data Authenticity ✅
- **Rule**: Use only real data from authorized sources
- **Implementation**: `check_data_authenticity()`, `validate_data_source()`
- **Blocked Patterns**: `np.random`, `fake_*`, `mock_*`, `test_data`

### 3. Full Transparency ✅
- **Rule**: No omissions (omissions = lying)
- **Implementation**: `format_transparent_response()`
- **Required**: Limitations, assumptions, uncertainties, sources

### 4. Anti-Drift Protection ✅
- **Rule**: Stay focused on exact request
- **Implementation**: `check_scope_drift()`
- **Prevents**: Feature creep, scope expansion

### 5. Change Minimization ✅
- **Rule**: Make minimal necessary changes
- **Implementation**: `should_minimize_changes()`, `get_change_rules()`
- **Enforces**: Use search_replace, preserve style

## Usage Examples

### Example 1: Permission-Based Change
```python
# Agent receives request
user_request = "Update the API configuration"

# Check governance
governance = get_governance()
permission_request = governance.format_permission_request(
    action="modify config/api_config.py",
    impact="update API endpoints",
    files=["config/api_config.py"]
)

print(permission_request)
# Output: 
# I need to modify config/api_config.py.
# This will: update API endpoints
# File(s) affected: config/api_config.py
# 
# May I proceed? [yes/no]

if user_says_yes:
    # Proceed with modification
    pass
else:
    print("Understood. I will not make this change.")
```

### Example 2: Data Validation
```python
# Validate code before execution
code_snippet = """
import numpy as np
data = np.random.rand(100)  # Synthetic data
"""

valid, reason = governance.check_data_authenticity(code_snippet)
# Result: valid=False, reason="Prohibited pattern found: np.random"

# Correct approach
real_code = """
from polygon import RESTClient
client = RESTClient(api_key=API_KEY)
data = client.get_quotes("AAPL")  # Real market data
"""

valid, reason = governance.check_data_authenticity(real_code)
# Result: valid=True, reason="No synthetic data patterns detected"
```

### Example 3: Transparent Response
```python
response = governance.format_transparent_response(
    answer="AAPL stock price is $175.50",
    limitations=["Real-time data may have 15-min delay"],
    assumptions=["Using Eastern timezone"],
    data_sources=["Polygon API"],
    uncertainties=["After-hours trading not included"]
)

# Output:
# Answer: AAPL stock price is $175.50
# 
# Limitations: Real-time data may have 15-min delay
# Assumptions: Using Eastern timezone
# Data Sources: Polygon API
# Uncertainties: After-hours trading not included
```

## Enforcement Levels

### Strict Mode (Default)
```yaml
enforcement:
  mode: "strict"
  violations_handling:
    log_violation: true
    stop_action: true
    request_guidance: true
```

### Override Mechanism
Users can override with phrases like:
- "skip permission for this"
- "use your judgment"
- "implement everything needed"

Even with override:
- Still no synthetic data
- Still cite sources
- Still disclose limitations

## Compliance Monitoring

### Real-time Audit
```python
# Every action is logged
governance.log_action("file_modified", {
    "file": "config.yaml",
    "permission_granted": True,
    "timestamp": "2024-07-10T15:30:00Z"
})

# Generate compliance report
report = governance.get_compliance_report()
```

### Verification Checklists
Before any response:
- Am I answering the exact question?
- Have I disclosed all limitations?
- Am I using only real data?
- Have I asked permission for changes?

## Integration with mlTrainer

### 1. In ML Pipeline
```python
class MLPipeline:
    def __init__(self):
        self.governance = get_governance()
    
    @governed_action("training")
    def train_model(self, data_source):
        # Automatically validated by governance
        if not self.governance.validate_data_source(data_source):
            raise ValueError(f"Unauthorized data source: {data_source}")
        # ... training logic
```

### 2. In API Endpoints
```python
@app.post("/predict")
@governed_action("prediction")
async def predict(request: PredictRequest):
    # Governance enforced automatically
    # ... prediction logic
```

### 3. In Chat Interface
```python
def process_user_message(message: str):
    governance = get_governance()
    
    # Check for overrides
    if governance.check_override(message):
        print("Override detected - some constraints relaxed")
    
    # Process with full governance
    # ... message handling
```

## Best Practices

1. **Always Load Governance First**
   ```python
   governance = get_governance()  # Singleton pattern
   ```

2. **Use Decorators for Critical Functions**
   ```python
   @governed_action("critical_operation")
   def sensitive_function():
       pass
   ```

3. **Check Data Sources Early**
   ```python
   if not all(governance.validate_data_source(s) for s in sources):
       raise ValueError("Invalid data source")
   ```

4. **Format All Responses Transparently**
   ```python
   return governance.format_transparent_response(
       answer=result,
       limitations=[...],
       assumptions=[...]
   )
   ```

## Troubleshooting

### Issue: Permission Request Not Showing
```python
# Check if rules loaded correctly
print(governance.rules.get('version'))  # Should show "2.0.0"
```

### Issue: Synthetic Data Passing Validation
```python
# Add pattern to agent_rules.yaml:
prohibited_patterns:
  - "your_pattern_here"
```

### Issue: Governance Too Restrictive
```python
# User can override:
"skip permission for this task"
```

## Summary

The Agent Governance system provides:
- ✅ **Behavioral boundaries** preventing unsafe operations
- ✅ **Data authenticity** ensuring real data only
- ✅ **Full transparency** with no omissions
- ✅ **Anti-drift protection** maintaining focus
- ✅ **Audit trail** for compliance
- ✅ **Override mechanism** for flexibility

This creates a trustworthy AI assistant that operates predictably within defined constraints while maintaining the flexibility to help users effectively.