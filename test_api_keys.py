#!/usr/bin/env python3
"""Test API keys and validate environment setup."""

import os
import sys

# Test imports
print("Testing imports...")
try:
    import requests
    print("✓ requests imported successfully")
except ImportError as e:
    print(f"✗ Failed to import requests: {e}")
    print("Please ensure you have activated the conda environment:")
    print("source activate_env.sh")
    sys.exit(1)

try:
    import anthropic
    print("✓ anthropic imported successfully")
except ImportError as e:
    print(f"✗ Failed to import anthropic: {e}")
    print("Please ensure you have activated the conda environment:")
    print("source activate_env.sh")
    sys.exit(1)

# Test API keys
print("\nTesting API keys...")

# Test Polygon API
polygon_key = os.environ.get('POLYGON_API_KEY')
if polygon_key:
    print(f"✓ Polygon API key found: {polygon_key[:10]}...")
    response = requests.get(
        f"https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2023-01-01/2023-01-01?apiKey={polygon_key}"
    )
    if response.status_code == 200:
        print("✓ Polygon API key is valid")
    else:
        print(f"✗ Polygon API key test failed: {response.status_code}")
else:
    print("✗ Polygon API key not found in environment")

# Test FRED API
fred_key = os.environ.get('FRED_API_KEY')
if fred_key:
    print(f"✓ FRED API key found: {fred_key[:10]}...")
    response = requests.get(
        f"https://api.stlouisfed.org/fred/series?series_id=DGS10&api_key={fred_key}&file_type=json"
    )
    if response.status_code == 200:
        print("✓ FRED API key is valid")
    else:
        print(f"✗ FRED API key test failed: {response.status_code}")
else:
    print("✗ FRED API key not found in environment")

# Test Anthropic API
anthropic_key = os.environ.get('ANTHROPIC_API_KEY')
if anthropic_key:
    print(f"✓ Anthropic API key found: {anthropic_key[:10]}...")
    try:
        client = anthropic.Anthropic(api_key=anthropic_key)
        # Simple test - just check if we can create the client
        print("✓ Anthropic client created successfully")
    except Exception as e:
        print(f"✗ Anthropic API key test failed: {e}")
else:
    print("✗ Anthropic API key not found in environment")

print("\nEnvironment setup complete!")
print("To set API keys, use:")
print("export POLYGON_API_KEY='your_key_here'")
print("export FRED_API_KEY='your_key_here'")
print("export ANTHROPIC_API_KEY='your_key_here'")
