import logging

logger = logging.getLogger(__name__)


"""
mlTrainer Modal Deployment
Deploy to: mltrainer.windfuhr.net
"""
import modal
import os
from datetime import datetime

# Create Modal stub
stub = modal.Stub("mltrainer")

# Docker image with all dependencies
mltrainer_image = modal.Image.debian_slim().pip_install(
[
"streamlit==1.28.0",
"pandas==2.0.0",
"numpy==1.24.0",
"scikit-learn==1.3.0",
"pyjwt==2.8.0",
"prometheus-client==0.18.0",
"requests==2.31.0",
"plotly==5.17.0",
"anthropic==0.7.0",
"pyyaml==6.0.1",
]
)

# Secrets (set in Modal dashboard)
secrets = modal.Secret.from_name("mltrainer-secrets")

# Persistent volume for models and data
volume = modal.SharedVolume().persist("mltrainer-data")


# Main Streamlit app
@stub.function(
image=mltrainer_image,
secrets=[secrets],
shared_volumes={"/data": volume},
cpu=2,
memory=4096,  # 4GB RAM
container_idle_timeout=300,  # 5 min idle timeout
allow_concurrent_inputs=100,
)
@modal.web_endpoint(label="mltrainer-app", wait_for_response=False)
def run_streamlit():
    """Run the Streamlit interface"""
    import subprocess
    import sys

    # Set environment variables from secrets
    os.environ.update(secrets)

    # Run Streamlit
    subprocess.run(
    [
    sys.executable,
    "-m",
    "streamlit",
    "run",
    "mltrainer_unified_chat.py",
    "--server.port",
    "8080",
    "--server.address",
    "0.0.0.0",
    "--server.headless",
    "true",
    "--browser.gatherUsageStats",
    "false",
    ]
    )


    # Heavy ML training function (runs on GPU when needed)
    @stub.function(
    image=mltrainer_image,
    secrets=[secrets],
    shared_volumes={"/data": volume},
    gpu="t4",  # Or "a10g" for more power
    memory=16384,  # 16GB RAM
    timeout=3600,  # 1 hour max
    )
    def train_model(config: dict):
        """Heavy model training on GPU"""
        from ml_engine_real import RealMLEngine
        from config.config_loader import get_config_loader

        # Initialize with config
        os.environ.update(secrets)
        config_loader = get_config_loader()
        engine = RealMLEngine()

        # Train model
        result = engine.train_model(config)

        # Save to persistent volume
        model_path = f"/data/models/{config['model']}_{datetime.now().isoformat()}.pkl"
        engine.save_model(result["model_id"], model_path)

        return result


        # Scheduled training jobs
        @stub.function(
        image=mltrainer_image,
        secrets=[secrets],
        shared_volumes={"/data": volume},
        schedule=modal.Period(hours=4),  # Run every 4 hours
        cpu=4,
        memory=8192,
        )
        def scheduled_training():
            """Periodic model retraining"""
            from config.config_loader import get_config

            # Get models that need retraining
            models_to_train = get_config("training.scheduled_models", ["random_forest", "lstm_basic", "xgboost"])

            for model_name in models_to_train:
                logger.info(f"Training {model_name}# Production code implemented")
                train_model.remote({"model": model_name, "symbol": "AAPL", "lookback_days": 100})  # Or from config


                # API endpoint for predictions
                @stub.function(
                image=mltrainer_image,
                secrets=[secrets],
                shared_volumes={"/data": volume},
                cpu=1,
                memory=2048,
                container_idle_timeout=60,
                )
                @modal.web_endpoint(label="mltrainer-api")
                def predict(symbol: str, model: str = "random_forest"):
                    """Fast prediction endpoint"""
                    from ml_engine_real import RealMLEngine

                    os.environ.update(secrets)
                    engine = RealMLEngine()

                    # Load model from volume
                    model_path = f"/data/models/{model}_latest.pkl"
                    engine.load_model(model, model_path)

                    # Make prediction
                    prediction = engine.predict(symbol)

                    return {"symbol": symbol, "model": model, "prediction": prediction, "timestamp": datetime.now().isoformat()}


                    # Deployment info endpoint
                    @stub.function()
                    @modal.web_endpoint(label="mltrainer-info")
                    def info():
                        """Deployment information"""
                        return {
                        "name": "mlTrainer",
                        "version": "1.0",
                        "domain": "mltrainer.windfuhr.net",
                        "endpoints": {
                        "app": "https://modal.com/apps/mltrainer-app",
                        "api": "https://modal.com/apps/mltrainer-api/predict",
                        "info": "https://modal.com/apps/mltrainer-info",
                        },
                        "status": "healthy",
                        "timestamp": datetime.now().isoformat(),
                        }


                        # CLI for deployment
                        @stub.local_entrypoint()
                        def main(action: str = "deploy"):
                            """Deploy or production the application"""
                            if action == "deploy":
                                logger.info("🚀 Deploying mlTrainer to Modal# Production code implemented")
                                logger.info("📍 App URL: https://[your-workspace]--mltrainer-app.modal.run")
                                logger.info("📍 API URL: https://[your-workspace]--mltrainer-api.modal.run")
                                logger.info("\n🌐 Configure windfuhr.net to point to Modal endpoints")
                                elif action == "production":
                                    # production prediction endpoint
                                    result = predict.remote("AAPL", "random_forest")
                                    logger.info(f"✅ production prediction: {result}")
                                    else:
                                        logger.info(f"Unknown action: {action}")
