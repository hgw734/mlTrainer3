"""
mlTrainer Complete Modal Deployment
===================================
Full-featured deployment with recommendation system and virtual portfolio
"""
import modal
import os
from datetime import datetime
import json
from typing import Dict, Any, Optional

# Create Modal app
app = modal.App(
    "mltrainer-complete",
    secrets=[
        # You'll need to set these in Modal
        modal.Secret.from_name("mltrainer-secrets"),
    ],
)

# Create the Docker image with all dependencies
mltrainer_image = (
    modal.Image.debian_slim()
    .pip_install([
        # Core dependencies
        "streamlit==1.28.0",
        "pandas==2.0.0",
        "numpy==1.24.0",
        "scikit-learn==1.3.0",
        "xgboost==2.0.0",
        "lightgbm==4.1.0",

        # API and data
        "anthropic==0.7.0",
        "polygon-api-client==1.12.0",
        "fredapi==0.5.1",
        "requests==2.31.0",
        "aiohttp==3.9.0",

        # UI and visualization
        "plotly==5.17.0",
        "matplotlib==3.7.0",

        # ML and finance
        "statsmodels==0.14.0",
        "scipy==1.11.0",
        "ta==0.10.2",  # Technical analysis
        "yfinance==0.2.28",

        # Infrastructure
        "pyyaml==6.0.1",
        "python-dotenv==1.0.0",
        "asyncio==3.4.3",
        "prometheus-client==0.18.0",
        "pyjwt==2.8.0",
    ])
    .copy_local_dir(".", "/app")  # Copy entire workspace
    .env({
        "PYTHONUNBUFFERED": "1",
        "STREAMLIT_SERVER_HEADLESS": "true",
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
        "STREAMLIT_SERVER_ENABLE_CORS": "true",
        "STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION": "false",
    })
)

# Create a persistent volume for data storage
volume = modal.Volume.from_name("mltrainer-data", create_if_missing=True)


@app.function(
    image=mltrainer_image,
    gpu=None,  # No GPU needed for this app
    cpu=2.0,
    memory=4096,
    timeout=3600,
    volumes={"/data": volume},
    allow_concurrent_inputs=100,
    keep_warm=1,  # Keep one instance always warm
)
@modal.web_endpoint(method="GET", label="mltrainer-chat")
def run_mltrainer_chat():
    """Run the mlTrainer chat interface"""
    import subprocess
    import sys

    # Set up environment variables from secrets
    secrets = modal.Secret.from_name("mltrainer-secrets").dict()
    for key, value in secrets.items():
        os.environ[key] = value

    # Ensure we're in the app directory
    os.chdir("/app")

    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("database", exist_ok=True)
    os.makedirs("/data/recommendations", exist_ok=True)
    os.makedirs("/data/portfolio", exist_ok=True)

    # Run the unified chat interface
    subprocess.Popen([
        sys.executable, "-m", "streamlit", "run",
        "mltrainer_unified_chat.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false"
    ])

    # Return a redirect to the Streamlit app
    return {
        "statusCode": 302,
        "headers": {
            "Location": f"https://{os.environ.get('MODAL_WORKSPACE_NAME', 'default')}-mltrainer-chat.modal.run/"
        }
    }


@app.function(
    image=mltrainer_image,
    schedule=modal.Period(minutes=15),  # Run every 15 minutes
    volumes={"/data": volume},
    timeout=600,
)
def scan_for_recommendations():
    """Periodic scan for trading recommendations"""
    import asyncio
    from recommendation_tracker import get_recommendation_tracker

    # S&P 500 symbols (you'd want to expand this)
    symbols = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "TSLA", "NVDA", "JPM", "JNJ", "V",
        "MA", "PG", "HD", "DIS", "PYPL",
        "NFLX", "ADBE", "CRM", "NKE", "INTC"
    ]

    tracker = get_recommendation_tracker()

    # Run the scan
    recommendations = asyncio.run(
        tracker.scan_for_opportunities(symbols)
    )

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "count": len(recommendations),
        # Top 10
        "recommendations": [r.to_dict() for r in recommendations[:10]]
    }

    with open("/data/recommendations/latest.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Found {len(recommendations)} recommendations at {datetime.now()}")
    return results


@app.function(
    image=mltrainer_image,
    schedule=modal.Period(minutes=5),  # Update positions every 5 minutes
    volumes={"/data": volume},
    timeout=300,
)
def update_virtual_positions():
    """Update virtual portfolio positions"""
    from virtual_portfolio_manager import get_virtual_portfolio_manager

    portfolio = get_virtual_portfolio_manager()

    # Update all positions with current prices
    portfolio.update_positions()

    # Get metrics
    metrics = portfolio.get_portfolio_metrics()

    # Save metrics
    with open("/data/portfolio/metrics.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        }, f, indent=2)

    print(
        f"Updated portfolio: {metrics['total_value']:.2f} ({metrics['total_return_pct']:+.2f}%)")
    return metrics


@app.function(
    image=mltrainer_image,
    volumes={"/data": volume},
)
@modal.web_endpoint(method="GET", label="mltrainer-api-recommendations")
def get_recommendations():
    """API endpoint to get latest recommendations"""
    try:
        with open("/data/recommendations/latest.json", "r") as f:
            data = json.load(f)
        return {
            "success": True,
            "data": data
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.function(
    image=mltrainer_image,
    volumes={"/data": volume},
)
@modal.web_endpoint(method="GET", label="mltrainer-api-portfolio")
def get_portfolio_metrics():
    """API endpoint to get portfolio metrics"""
    try:
        with open("/data/portfolio/metrics.json", "r") as f:
            data = json.load(f)
        return {
            "success": True,
            "data": data
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.local_entrypoint()
def main():
    """Deploy the complete mlTrainer system"""
    print("🚀 Deploying mlTrainer to Modal...")

    # Deploy the app
    with app.run():
        print("✅ mlTrainer deployed successfully!")
        print("\nAccess your mlTrainer system at:")
        print(
            f"https://{os.environ.get('MODAL_WORKSPACE_NAME', 'your-workspace')}-mltrainer-chat.modal.run/")
        print("\nAPI Endpoints:")
        print(
            f"- Recommendations: https://{os.environ.get('MODAL_WORKSPACE_NAME', 'your-workspace')}-mltrainer-api-recommendations.modal.run/")
        print(
            f"- Portfolio: https://{os.environ.get('MODAL_WORKSPACE_NAME', 'your-workspace')}-mltrainer-api-portfolio.modal.run/")
        print("\nScheduled Jobs:")
        print("- Recommendation scan: Every 15 minutes")
        print("- Portfolio update: Every 5 minutes")
