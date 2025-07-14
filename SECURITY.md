# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 2.0.x   | :white_check_mark: |
| 1.x.x   | :x:                |

## Reporting a Vulnerability

We take the security of mlTrainer seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### Please do NOT:
- Create a public GitHub issue for security vulnerabilities
- Post about the vulnerability on social media
- Attempt to exploit the vulnerability in production systems

### Please DO:
- Email security@mltrainer.ai with details
- Include steps to reproduce if possible
- Allow us reasonable time to respond and fix the issue

## What to Report

### High Priority
- Authentication/authorization bypasses
- Data leakage or unauthorized access
- Compliance system bypasses
- API key exposure
- Code injection vulnerabilities

### Medium Priority
- Denial of service vulnerabilities
- Information disclosure
- Cross-site scripting (XSS)
- Session management issues

### Low Priority
- Best practice violations
- Minor information leaks
- Performance issues

## Response Timeline

- **Initial Response**: Within 24 hours
- **Assessment**: Within 72 hours
- **Fix Timeline**: Depends on severity
  - Critical: Within 7 days
  - High: Within 14 days
  - Medium: Within 30 days
  - Low: Next release cycle

## Compliance & Audit

mlTrainer includes several security features:

1. **Immutable Compliance Gateway**: Prevents unauthorized data access
2. **Runtime Enforcement**: Real-time security monitoring
3. **Kill Switch**: Automatic shutdown on critical violations
4. **Audit Logging**: Complete audit trail of all operations
5. **Cryptographic Verification**: SHA-256 signatures on all data

## Security Best Practices

When using mlTrainer:

1. **API Keys**: 
   - Never commit API keys to version control
   - Use environment variables
   - Rotate keys regularly

2. **Data Sources**:
   - Only use approved data sources (Polygon, FRED, QuiverQuant)
   - Verify data provenance
   - Check data freshness

3. **Access Control**:
   - Use principle of least privilege
   - Enable two-factor authentication
   - Review access logs regularly

4. **Monitoring**:
   - Monitor runtime_enforcement.log
   - Set up alerts for violations
   - Review audit trails

## Compliance Requirements

mlTrainer enforces strict compliance:
- NO synthetic or fake data
- NO unauthorized data sources
- NO bypassing of security controls
- ALL operations must be audited

Attempting to bypass these controls will trigger the kill switch.

## Contact

Security Team: security@mltrainer.ai
Enterprise Support: enterprise@mltrainer.ai

PGP Key: [Available on request]