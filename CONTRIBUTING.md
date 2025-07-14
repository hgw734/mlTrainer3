# Contributing to mlTrainer

Thank you for your interest in contributing to mlTrainer! As this is a proprietary system with strict compliance requirements, please review these guidelines carefully.

## Code of Conduct

By participating in this project, you agree to maintain professionalism and respect for all contributors.

## Getting Started

1. **License Agreement**: Contributors must sign a Contributor License Agreement (CLA) before any contributions can be accepted.

2. **Access Request**: Request access by emailing contribute@mltrainer.ai with:
   - Your GitHub username
   - Organization affiliation
   - Intended contribution area

3. **Environment Setup**:
   ```bash
   git clone https://github.com/yourusername/mlTrainer.git
   cd mlTrainer
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e ".[dev]"
   pre-commit install
   ```

## Compliance Requirements

### Absolute Rules

1. **NO Synthetic Data**: Never introduce synthetic, fake, or generated data
2. **NO Hardcoded Secrets**: All API keys must use environment variables
3. **NO Security Bypasses**: Never attempt to bypass compliance checks
4. **NO Unapproved Sources**: Only use Polygon, FRED, and QuiverQuant APIs

### Before Submitting

Run ALL compliance checks:

```bash
# 1. Run production audit
python scripts/production_audit_final.py

# 2. Run tests
pytest tests/ -v

# 3. Check code quality
black .
isort .
flake8 .
mypy .

# 4. Verify compliance system
python verify_compliance_system.py
```

## Development Process

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Write clean, documented code
- Add type hints
- Include docstrings
- Follow PEP 8

### 3. Add Tests

- All new features must have tests
- Maintain >90% code coverage
- Test edge cases
- Include compliance tests

### 4. Document Your Changes

- Update relevant documentation
- Add docstrings to all functions
- Update README if needed
- Add to CHANGELOG

### 5. Commit Your Changes

```bash
git add .
git commit -m "feat: add new feature

- Detailed description
- Why this change is needed
- Any breaking changes"
```

Follow [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Test additions
- `chore:` Maintenance

### 6. Push and Create PR

```bash
git push origin feature/your-feature-name
```

## Pull Request Process

1. **Title**: Use conventional commit format
2. **Description**: Explain what, why, and how
3. **Checklist**:
   - [ ] Compliance audit passes
   - [ ] All tests pass
   - [ ] Code coverage maintained
   - [ ] Documentation updated
   - [ ] No hardcoded values
   - [ ] No synthetic data

4. **Review Process**:
   - Automated checks must pass
   - Code review by 2 maintainers
   - Security review if applicable
   - Final compliance check

## Code Style

### Python Style Guide

```python
"""Module docstring explaining purpose."""

from typing import Dict, List, Optional

import external_library
import another_library

from mltrainer.core import SomeClass


class ExampleClass:
    """Class docstring with description.
    
    Attributes:
        attribute: Description of attribute
    """
    
    def __init__(self, param: str) -> None:
        """Initialize ExampleClass.
        
        Args:
            param: Description of parameter
        """
        self.attribute = param
    
    def method(self, arg: int) -> Dict[str, Any]:
        """Method description.
        
        Args:
            arg: Description of argument
            
        Returns:
            Description of return value
            
        Raises:
            ValueError: When validation fails
        """
        if arg < 0:
            raise ValueError("Argument must be positive")
        
        return {"result": arg * 2}
```

### Import Order

1. Standard library imports
2. Third-party imports
3. Local application imports

### Testing Standards

```python
import pytest
from unittest.mock import Mock, patch

from mltrainer.module import function_to_test


class TestFunction:
    """Test class for function."""
    
    def test_normal_operation(self):
        """Test normal operation."""
        result = function_to_test("input")
        assert result == "expected"
    
    def test_edge_case(self):
        """Test edge case."""
        with pytest.raises(ValueError):
            function_to_test(None)
    
    @patch('mltrainer.module.external_call')
    def test_with_mock(self, mock_call):
        """Test with mocked dependency."""
        mock_call.return_value = "mocked"
        result = function_to_test("input")
        assert result == "expected"
        mock_call.assert_called_once()
```

## Security Considerations

1. **Never commit sensitive data**
2. **Use secrets manager for credentials**
3. **Validate all inputs**
4. **Log security events**
5. **Follow OWASP guidelines**

## Questions?

- Technical questions: dev@mltrainer.ai
- Security concerns: security@mltrainer.ai
- General inquiries: info@mltrainer.ai

## License

By contributing, you agree that your contributions will be licensed under the same proprietary license as the project.