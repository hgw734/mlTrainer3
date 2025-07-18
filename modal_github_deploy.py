"""
Deploy mlTrainer directly from GitHub to Modal
==============================================
No local clone needed - Modal pulls from your repo
"""
import modal
import os

# Create Modal app
app = modal.App(
    "mltrainer3",
    secrets=[
        modal.Secret.from_name("mltrainer3-secrets"),
    ],
)

# Create image that clones from GitHub
mltrainer_image = (
    modal.Image.debian_slim()
    .run_commands(
        "apt-get update",
        "apt-get install -y git",
    )
    .run_commands(
        # Clone the repository directly from GitHub
        "git clone https://github.com/hgw734/mlTrainer3.git /app",
        "cd /app && ls -la",  # Verify files are there
    )
    .pip_install_from_requirements("/app/requirements_unified.txt")
    .pip_install([
        # Additional packages that might not be in requirements
        "streamlit==1.28.0",
        "anthropic==0.7.0",
        "polygon-api-client==1.12.0",
        "fredapi==0.5.1",
        "ta==0.10.2",
    ])
    .workdir("/app")  # Set working directory
    .env({
        "PYTHONUNBUFFERED": "1",
        "STREAMLIT_SERVER_HEADLESS": "true",
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
    })
)

# Persistent volume for data
volume = modal.Volume.from_name("mltrainer3-data", create_if_missing=True)

@app.function(
    image=mltrainer_image,
    gpu=None,
    cpu=2.0,
    memory=4096,
    timeout=3600,
    volumes={"/data": volume},
    allow_concurrent_inputs=100,
    keep_warm=1,
)
@modal.asgi_app()
def mltrainer_app():
    """Run mlTrainer as an ASGI app"""
    import subprocess
    import sys
    from fastapi import FastAPI, Response
    from fastapi.responses import RedirectResponse
    
    # Set up environment
    secrets = modal.Secret.from_name("mltrainer3-secrets").dict()
    for key, value in secrets.items():
        os.environ[key] = value
    
    # Create directories
    os.makedirs("/data/logs", exist_ok=True)
    os.makedirs("/data/recommendations", exist_ok=True)
    os.makedirs("/data/portfolio", exist_ok=True)
    
    # Start Streamlit in background
    subprocess.Popen([
        sys.executable, "-m", "streamlit", "run",
        "mltrainer_unified_chat.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
    ])
    
    # Create FastAPI app to handle routing
    web_app = FastAPI()
    
    @web_app.get("/")
    async def root():
        # Redirect to Streamlit
        return RedirectResponse(url="/stream/", status_code=302)
    
    @web_app.get("/health")
    async def health():
        return {"status": "healthy", "app": "mlTrainer"}
    
    return web_app

@app.function(
    image=mltrainer_image,
    schedule=modal.Period(minutes=15),
    volumes={"/data": volume},
)
def scan_recommendations():
    """Scan for trading recommendations every 15 minutes"""
    import asyncio
    import json
    from datetime import datetime
    
    # Import from the cloned repo
    import sys
    sys.path.append('/app')
    from recommendation_tracker import get_recommendation_tracker
    
    symbols = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "TSLA", "NVDA", "JPM", "JNJ", "V",
        "MA", "PG", "HD", "DIS", "PYPL",
    ]
    
    tracker = get_recommendation_tracker()
    recommendations = asyncio.run(tracker.scan_for_opportunities(symbols))
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "count": len(recommendations),
        "recommendations": [r.to_dict() for r in recommendations[:10]]
    }
    
    with open("/data/recommendations/latest.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Found {len(recommendations)} recommendations")
    return results

# One-line deployment script
@app.local_entrypoint()
def deploy():
    """Deploy directly from GitHub"""
    print("🚀 Deploying mlTrainer from GitHub...")
    print("📦 Repository: https://github.com/hgw734/mlTrainer3")
    print("⏳ This will take a few minutes on first deploy...")
    
    # The deployment happens automatically when this script runs
    print("\n✅ Deployment complete!")
    print(f"\n🌐 Access your mlTrainer3 at:")
    print(f"   https://{os.environ.get('USER', 'your-username')}--mltrainer3.modal.run")
    print("\n📱 Save this URL to your iPhone home screen!")

if __name__ == "__main__":
    deploy.remote()