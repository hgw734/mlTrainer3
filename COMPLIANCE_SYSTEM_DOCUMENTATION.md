# mlTrainer Compliance System Documentation

## Overview

The mlTrainer compliance system ensures that all code follows strict security, data integrity, and governance standards. This system prevents the use of synthetic data, hardcoded API keys, and other security risks while maintaining production-ready code quality.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Compliance Rules](#compliance-rules)
3. [Setup and Installation](#setup-and-installation)
4. [Usage](#usage)
5. [CI/CD Integration](#cicd-integration)
6. [Team Guidelines](#team-guidelines)
7. [Troubleshooting](#troubleshooting)
8. [Security Exceptions](#security-exceptions)

## System Architecture

### Core Components

- **Agent Rules (`agent_rules.yaml`)**: Defines compliance policies
- **Compliance Auditor (`scripts/production_audit.py`)**: Main audit engine
- **Governance Framework**: Core governance components in `core/` and `config/`
- **Fix Scripts**: Automated compliance fixes

### Audit Categories

1. **Synthetic Data Detection**: Prevents use of random/synthetic data
2. **API Key Security**: Prevents hardcoded credentials
3. **Security Patterns**: Blocks dangerous code patterns
4. **Data Sources**: Ensures approved data sources only
5. **Error Handling**: Validates proper error management
6. **Code Quality**: Checks for best practices
7. **Governance**: Ensures governance framework usage

## Compliance Rules

### Critical Violations (Block Deployment)

- ❌ **Synthetic Data**: No `np.random`, `random.random()`, or synthetic data generation
- ❌ **Hardcoded API Keys**: No credentials in code
- ❌ **Security Risks**: No `eval()`, `exec()`, `pickle`, or `__import__`

### Warnings (Review Required)

- ⚠️ **Data Sources**: Ensure approved sources (Polygon, FRED, Redis, Database)
- ⚠️ **Error Handling**: Avoid bare except clauses
- ⚠️ **Code Quality**: Use logging instead of print statements
- ⚠️ **Governance**: Core components should use governance framework

## Setup and Installation

### Prerequisites

```bash
# Install required packages
pip install -r requirements_unified.txt

# Install additional compliance tools
pip install pyyaml ast-comments
```

### Environment Setup

1. **Create `.env` file**:
```bash
# API Keys (use real keys, not placeholders)
POLYGON_API_KEY=your_polygon_key_here
FRED_API_KEY=your_fred_key_here
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Database
REDIS_URL=redis://localhost:6379

# Security
ENCRYPTION_KEY=your_encryption_key_here
```

2. **Verify setup**:
```bash
python3 scripts/setup_secure_environment.py
```

## Usage

### Running Compliance Audits

#### Quick Check
```bash
python3 scripts/final_compliance_check.py
```

#### Full Production Audit
```bash
python3 scripts/production_audit.py
```

#### Manual Fix Application
```bash
# Fix specific issues
python3 scripts/fix_security_api_keys.py
python3 scripts/fix_synthetic_data_patterns.py
```

### Audit Output

```
🔍 mlTrainer Comprehensive Compliance Audit
======================================================================
Started at: 2024-01-15 14:30:00

📁 Found 127 Python files to audit

📦 Auditing Dependencies...
⚙️  Auditing Configuration Files...

======================================================================
📊 AUDIT REPORT
======================================================================

📈 Statistics:
   Files scanned: 127
   Total lines: 45,230
   Dependencies checked: 2

✅ COMPLIANCE SUMMARY
--------------------------------------------------
   ✅ No Synthetic Data
   ✅ No Hardcoded Keys
   ✅ Secure Code

======================================================================
✅ AUDIT PASSED - No critical violations found
```

## CI/CD Integration

### GitHub Actions Workflow

Create `.github/workflows/compliance.yml`:

```yaml
name: Compliance Check

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  compliance:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements_unified.txt
        pip install pyyaml ast-comments
    
    - name: Run compliance audit
      run: |
        python3 scripts/production_audit.py
    
    - name: Upload audit report
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: compliance-report
        path: audit_report.json
```

### Pre-commit Hook

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: compliance-check
        name: Compliance Audit
        entry: python3 scripts/production_audit.py
        language: system
        pass_filenames: false
        always_run: true
```

Install pre-commit:
```bash
pip install pre-commit
pre-commit install
```

## Team Guidelines

### Development Workflow

1. **Before Committing**:
   ```bash
   python3 scripts/production_audit.py
   ```

2. **If Violations Found**:
   - Fix critical violations immediately
   - Document warnings for review
   - Update code to use approved patterns

3. **For New Features**:
   - Use approved data sources only
   - Implement proper error handling
   - Follow governance framework

### Approved Patterns

#### ✅ Good: Real Data Sources
```python
# Approved data sources
from polygon import RESTClient
from fredapi import Fred
import redis

# Fetch real market data
client = RESTClient(api_key=os.getenv('POLYGON_API_KEY'))
data = client.get_aggs('AAPL', 1, 'day', '2024-01-01', '2024-01-15')
```

#### ❌ Bad: Synthetic Data
```python
# Prohibited - synthetic data
import numpy as np
data = np.random.randn(100)  # This will be flagged
```

#### ✅ Good: Secure Configuration
```python
# Approved - environment variables
import os
api_key = os.getenv('POLYGON_API_KEY')
```

#### ❌ Bad: Hardcoded Keys
```python
# Prohibited - hardcoded credentials
api_key = "pk_1234567890abcdef"  # This will be flagged
```

### Code Review Checklist

- [ ] No synthetic data generation
- [ ] No hardcoded API keys or secrets
- [ ] Proper error handling (no bare except)
- [ ] Uses logging instead of print statements
- [ ] Core components use governance framework
- [ ] Approved data sources only
- [ ] Security patterns avoided (eval, exec, pickle)

## Troubleshooting

### Common Issues

#### 1. "Synthetic Data" Violations

**Problem**: Code flagged for using `np.random` or similar
**Solution**: Replace with real data sources

```python
# Instead of:
data = np.random.randn(100)

# Use:
from polygon import RESTClient
client = RESTClient(api_key=os.getenv('POLYGON_API_KEY'))
data = client.get_aggs('AAPL', 1, 'day', '2024-01-01', '2024-01-15')
```

#### 2. "API Key" Violations

**Problem**: Hardcoded credentials detected
**Solution**: Use environment variables

```python
# Instead of:
api_key = "pk_1234567890abcdef"

# Use:
import os
api_key = os.getenv('POLYGON_API_KEY')
```

#### 3. "Security" Violations

**Problem**: Dangerous patterns like `eval()` detected
**Solution**: Use safer alternatives

```python
# Instead of:
result = eval(user_input)

# Use:
import ast
result = ast.literal_eval(user_input)  # Safer for literals only
```

### Audit Script Issues

#### Syntax Errors
```bash
# Check for syntax errors
python3 -m py_compile scripts/production_audit.py
```

#### Missing Dependencies
```bash
# Install missing packages
pip install -r requirements_unified.txt
pip install pyyaml ast-comments
```

## Security Exceptions

### When Dynamic Execution is Required

If you absolutely need dynamic execution (rare), document the security review:

```python
# SECURITY REVIEW REQUIRED
# This code uses dynamic execution for the following reasons:
# 1. Feature X requires runtime code generation
# 2. Mitigation: Input validation and sandboxing implemented
# 3. Risk assessment: Low (controlled inputs only)
# 4. Approval: [Your Name] - [Date]

def safe_dynamic_execution(code: str, context: dict):
    """
    SECURITY: This function uses dynamic execution with strict controls.
    
    Mitigations:
    - Input validation against whitelist
    - Sandboxed execution environment
    - Audit logging of all executions
    - Timeout limits
    """
    # Implementation with security controls
    pass
```

### Exception Documentation

All security exceptions must be documented with:
1. **Justification**: Why the exception is necessary
2. **Risk Assessment**: Security risks and mitigations
3. **Approval**: Who approved the exception and when
4. **Review Schedule**: When to re-evaluate the exception

## Maintenance

### Regular Tasks

1. **Weekly**: Run full compliance audit
2. **Monthly**: Review and update agent rules
3. **Quarterly**: Security review of exceptions
4. **Annually**: Comprehensive compliance review

### Updating Rules

To modify compliance rules, edit `agent_rules.yaml`:

```yaml
# Example rule modification
synthetic_data:
  prohibited_patterns:
    - "np.random"
    - "random.random()"
  exceptions:
    - "test_files"
    - "documentation"
```

### Monitoring

- Audit reports are saved to `audit_report.json`
- Compliance status tracked in CI/CD
- Violations trigger automated alerts
- Regular compliance metrics reporting

## Support

For compliance issues:
1. Check this documentation first
2. Review `audit_report.json` for specific violations
3. Consult the team lead for exceptions
4. Update documentation for new patterns

---

**Last Updated**: January 2024
**Version**: 1.0
**Maintainer**: mlTrainer Team 