# mlTrainer3 Immutable Compliance System Makefile
# Use with caution - violations have real consequences

.PHONY: help test install activate deploy docker clean verify monitor

# Default target
help:
	@echo "🔒 mlTrainer3 Immutable Compliance System"
	@echo "========================================"
	@echo ""
	@echo "Available commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make test         - Run compliance tests (safe)"
	@echo "  make verify       - Verify system status"
	@echo "  make activate     - Activate compliance (PERMANENT!)"
	@echo "  make deploy       - Deploy to production (requires root)"
	@echo "  make docker       - Build Docker images"
	@echo "  make docker-test  - Run tests in Docker"
	@echo "  make monitor      - Monitor violations"
	@echo "  make clean        - Clean temporary files"
	@echo ""
	@echo "⚠️  WARNING: This system has REAL consequences!"

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	pip install -r requirements-immutable.txt
	@echo "✅ Dependencies installed"

# Run compliance tests
test:
	@echo "🧪 Running compliance tests..."
	python test_immutable_kernel.py
	@if [ -f "./verify_compliance.py" ]; then \
		echo ""; \
		echo "Verifying compliance status..."; \
		./verify_compliance.py || true; \
	fi

# Verify system status
verify:
	@echo "🔍 Verifying compliance system..."
	@if [ -f "./verify_compliance.py" ]; then \
		./verify_compliance.py; \
	else \
		echo "❌ Verification script not found"; \
		echo "Run 'make activate' first"; \
	fi

# Activate compliance system (DANGEROUS!)
activate:
	@echo "⚠️  WARNING: This will PERMANENTLY activate the immutable compliance system!"
	@echo "Once activated:"
	@echo "  • Violations have REAL consequences"
	@echo "  • Cannot be disabled"
	@echo "  • No exemptions"
	@echo ""
	@read -p "Type 'ACTIVATE' to proceed: " confirm; \
	if [ "$$confirm" = "ACTIVATE" ]; then \
		sudo python scripts/activate_immutable_compliance.py; \
	else \
		echo "Activation cancelled"; \
	fi

# Deploy to production
deploy:
	@echo "🚀 Deploying to production..."
	@if [ ! -f "./deploy_immutable_compliance.sh" ]; then \
		echo "❌ Deployment script not found"; \
		exit 1; \
	fi
	@chmod +x ./deploy_immutable_compliance.sh
	sudo ./deploy_immutable_compliance.sh

# Docker operations
docker:
	@echo "🐳 Building Docker images..."
	docker build -f Dockerfile.immutable -t mltrainer3/compliance:latest .

docker-test:
	@echo "🧪 Running tests in Docker..."
	docker-compose -f docker-compose.immutable.yml --profile test run compliance-tester

docker-up:
	@echo "🚀 Starting Docker services..."
	docker-compose -f docker-compose.immutable.yml up -d

docker-down:
	@echo "🛑 Stopping Docker services..."
	docker-compose -f docker-compose.immutable.yml down

docker-logs:
	@echo "📋 Showing Docker logs..."
	docker-compose -f docker-compose.immutable.yml logs -f

# Monitor violations
monitor:
	@echo "👀 Monitoring violations..."
	@if [ -f "./verify_compliance.py" ]; then \
		watch -n 5 "./verify_compliance.py"; \
	else \
		echo "❌ Monitoring not available - activate compliance first"; \
	fi

# Clean temporary files
clean:
	@echo "🧹 Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name ".DS_Store" -delete
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	@echo "✅ Cleanup complete"

# Check for violations in current code
check:
	@echo "🔍 Checking for compliance violations..."
	@violations=0; \
	echo "Checking for prohibited patterns..."; \
	if grep -r "get_volatility" --include="*.py" . 2>/dev/null | grep -v "^Binary"; then \
		echo "❌ Found get_volatility calls"; \
		violations=$$((violations + 1)); \
	fi; \
	if grep -r "np\.random\|random\.random" --include="*.py" . 2>/dev/null | grep -v "^Binary"; then \
		echo "❌ Found random generation"; \
		violations=$$((violations + 1)); \
	fi; \
	if grep -r "mock_\|fake_\|dummy_" --include="*.py" . 2>/dev/null | grep -v "^Binary"; then \
		echo "❌ Found mock patterns"; \
		violations=$$((violations + 1)); \
	fi; \
	if [ $$violations -eq 0 ]; then \
		echo "✅ No violations found"; \
	else \
		echo ""; \
		echo "❌ Found $$violations violation types"; \
		echo "Fix these before activating compliance!"; \
		exit 1; \
	fi

# Install git hooks
install-hooks:
	@echo "🔗 Installing git hooks..."
	@mkdir -p .git/hooks
	@if [ -f "hooks/pre-commit" ]; then \
		cp hooks/pre-commit .git/hooks/pre-commit; \
		chmod +x .git/hooks/pre-commit; \
		echo "✅ Pre-commit hook installed"; \
	else \
		echo "❌ Pre-commit hook not found"; \
	fi

# Run security scan
security-scan:
	@echo "🔐 Running security scan..."
	@if command -v bandit >/dev/null 2>&1; then \
		bandit -r core/ -f json -o security-report.json || true; \
		echo "✅ Security report generated: security-report.json"; \
	else \
		echo "Installing bandit..."; \
		pip install bandit; \
		bandit -r core/ -f json -o security-report.json || true; \
	fi

# Full CI/CD pipeline
ci: clean install test check security-scan
	@echo "✅ CI pipeline complete"

# Development setup
dev-setup: install install-hooks
	@echo "🛠️  Development environment ready"
	@echo "Remember: All commits will be checked for compliance!"

# Show current violations (if any)
show-violations:
	@echo "📋 Current violations:"
	@python -c "from core.consequence_enforcement_system import CONSEQUENCE_ENFORCER; \
		report = CONSEQUENCE_ENFORCER.get_violation_report(); \
		print(f'Total: {report[\"total_violations\"]}'); \
		print(f'Banned functions: {report[\"banned_functions\"]}'); \
		print(f'Banned modules: {report[\"banned_modules\"]}'); \
		print(f'Banned users: {report[\"banned_users\"]}')" 2>/dev/null || \
		echo "No violation data available"