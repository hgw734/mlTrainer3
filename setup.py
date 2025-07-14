"""
mlTrainer - Institutional-Grade AI/ML Trading System
Setup configuration for package installation
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

    # Read requirements
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

        # Package metadata
        setup(
        name="mltrainer",
        version="2.0.0",
        author="mlTrainer Development Team",
        author_email="dev@mltrainer.ai",
        description="Institutional-grade AI/ML trading system with immutable compliance enforcement",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/yourusername/mlTrainer",
        project_urls={
        "Bug Tracker": "https://github.com/yourusername/mlTrainer/issues",
        "Documentation": "https://docs.mltrainer.ai",
        "Source Code": "https://github.com/yourusername/mlTrainer",
        },
        classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        ],
        packages=find_packages(exclude=["tests", "tests.*", "scripts", "docs"]),
        python_requires=">=3.8",
        install_requires=requirements,
        extras_require={
        "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "pytest-asyncio>=0.20.0",
        "black>=22.0.0",
        "isort>=5.10.0",
        "flake8>=4.0.0",
        "mypy>=0.990",
        "sphinx>=4.5.0",
        "sphinx-rtd-theme>=1.0.0",
        ],
        "monitoring": [
        "prometheus-client>=0.15.0",
        "grafana-api>=1.0.0",
        ],
        },
        entry_points={
        "console_scripts": [
        "mltrainer=app:main",
        "mltrainer-audit=scripts.production_audit_final:main",
        "mltrainer-verify=verify_compliance_system:main",
        ],
        },
        package_data={
        "config": ["*.json", "*.yaml"],
        "core": ["*.json"],
        "tests": ["*.json", "*.yaml"],
        },
        include_package_data=True,
        zip_safe=False,
        )
