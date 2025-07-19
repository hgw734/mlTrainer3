#!/usr/bin/env python3
"""
One-Click Modal Deployment for mlTrainer
========================================
Run this single file to deploy mlTrainer from GitHub to Modal
No need to clone the repository!
"""

# First, save this file locally and run:
# pip install modal
# modal token new
# python deploy_to_modal.py

import subprocess
import sys
import os


def install_modal():
    """Install Modal if not already installed"""
    try:
        import modal
        print("✅ Modal is already installed")
    except ImportError:
        print("📦 Installing Modal...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "modal"])
        print("✅ Modal installed successfully")


def check_modal_auth():
    """Check if Modal is authenticated"""
    try:
        result = subprocess.run(["modal", "profile", "current"],
                                capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Authenticated as: {result.stdout.strip()}")
            return True
        else:
            print("🔑 Need to authenticate with Modal...")
            subprocess.run(["modal", "token", "new"])
            return True
    except Exception as e:
        print(f"❌ Error checking Modal auth: {e}")
        return False


def deploy_from_github():
    """Deploy mlTrainer directly from GitHub"""
    print("\n🚀 Deploying mlTrainer from GitHub to Modal...")
    print("📦 Repository: https://github.com/hgw734/mlTrainer3")
    print("⏳ This will take 3-5 minutes on first deploy...\n")

    # Create the deployment script content
    deployment_script = '''
import modal
import os

app = modal.App(
    "mltrainer3",
    secrets=[modal.Secret.from_name("mltrainer3-secrets")],
)

image = (
    modal.Image.debian_slim()
    .run_commands(
        "apt-get update",
        "apt-get install -y git",
        "git clone https://github.com/hgw734/mlTrainer3.git /app",
    )
    .pip_install_from_requirements("/app/requirements_unified.txt")
    .pip_install(["streamlit", "anthropic", "polygon-api-client", "fredapi"])
    .workdir("/app")
)

@app.function(
    image=image,
    cpu=2.0,
    memory=4096,
    timeout=3600,
    keep_warm=1,
)
@modal.web_endpoint()
def run():
    import subprocess
    import sys
    subprocess.Popen([
        sys.executable, "-m", "streamlit", "run",
        "mltrainer_unified_chat.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
    ])
    return {"status": "mlTrainer is running!"}

@app.local_entrypoint()
def main():
    print("Deploying mlTrainer...")
'''

    # Write temporary deployment file
    with open("temp_deploy.py", "w") as f:
        f.write(deployment_script)

    try:
        # Run the deployment
        result = subprocess.run([sys.executable, "temp_deploy.py"],
                                capture_output=True, text=True)

        if result.returncode == 0:
            print("\n✅ Deployment successful!")
            print("\n🌐 Your mlTrainer3 is now live at:")
            print("   https://YOUR-MODAL-USERNAME--mltrainer3.modal.run")
            print("\n📱 Save this URL to your iPhone home screen!")
            print("\n⚠️  Don't forget to set up your API keys in Modal:")
            print("   1. Go to https://modal.com/secrets")
            print("   2. Create secret 'mltrainer3-secrets' with:")
            print("      - POLYGON_API_KEY")
            print("      - FRED_API_KEY")
            print("      - ANTHROPIC_API_KEY")
        else:
            print(f"\n❌ Deployment failed: {result.stderr}")

    finally:
        # Clean up
        if os.path.exists("temp_deploy.py"):
            os.remove("temp_deploy.py")


def main():
    """Main deployment process"""
    print("🎯 mlTrainer Modal Deployment Tool")
    print("=" * 50)

    # Step 1: Install Modal
    install_modal()

    # Step 2: Check authentication
    if not check_modal_auth():
        print("❌ Modal authentication failed. Please try again.")
        return

    # Step 3: Deploy from GitHub
    deploy_from_github()


if __name__ == "__main__":
    main()
