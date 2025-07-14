import logging

logger = logging.getLogger(__name__)


"""
mlTrainer Optimized Modal Deployment
Production-ready configuration with all optimizations
"""
import modal
import os
from datetime import datetime, time
import json
from typing import Dict, Any, Optional

# Create Modal stub with optimization settings
stub = modal.Stub(
"mltrainer",
secrets=[
modal.Secret.from_name("mltrainer-secrets"),
modal.Secret.from_dict({"MLTRAINER_ENV": "production", "DOMAIN": "windfuhr.net"}),
],
)

# Optimized Docker image with caching
mltrainer_image = (
modal.Image.debian_slim()
.pip_install(
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
"redis==5.0.0",  # For caching
"asyncio==3.4.3",
"aiohttp==3.9.0",  # For async operations
]
)
.copy_local_dir("config", "/app/config")  # Pre-copy config files
.copy_local_dir("core", "/app/core")  # Pre-copy core modules
.copy_local_dir("utils", "/app/utils")  # Pre-copy utils
.copy_local_dir("backend", "/app/backend")  # Pre-copy backend
.env(
{"PYTHONUNBUFFERED": "1", "STREAMLIT_SERVER_HEADLESS": "true", "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false"}
)
)

# Persistent volumes with optimized mount points
model_volume = modal.SharedVolume().persist("mltrainer-models")
cache_volume = modal.SharedVolume().persist("mltrainer-cache")
data_volume = modal.SharedVolume().persist("mltrainer-data")

# Redis for distributed caching
redis_image = modal.Image.from_registry("redis:7-alpine")


# Time-based scaling configuration
def get_container_count():
    """Dynamic container count based on market hours"""
    current_hour = datetime.now().hour
    current_day = datetime.now().weekday()

    # Market hours (9:30 AM - 4:00 PM EST, Mon-Fri)
    if current_day < 5:  # Monday-Friday
    if 9 <= current_hour <= 16:  # Market hours
    return 5  # High availability
    elif 7 <= current_hour <= 18:  # Extended hours
    return 3  # Medium availability

    return 1  # Low availability off-hours


    # Cache service for fast data access
    @stub.function(
    image=redis_image,
    shared_volumes={"/data": cache_volume},
    cpu=1,
    memory=1024,
    container_idle_timeout=86400,  # 24 hours
    keep_warm=1,
    )
    @modal.asgi_app(label="mltrainer-cache")
    def cache_service():
        """Redis cache for fast data access"""
        import redis

        return redis.Redis(host="localhost", port=6379, decode_responses=True)


        # Optimized Streamlit app with smart scaling
        @stub.function(
        image=mltrainer_image,
        shared_volumes={"/models": model_volume, "/cache": cache_volume, "/data": data_volume},
        cpu=2.0,
        memory=4096,
        keep_warm=get_container_count(),  # Dynamic scaling
        container_idle_timeout=3600,  # 1 hour
        allow_concurrent_inputs=100,
        retries=3,
        )
        @modal.web_endpoint(label="mltrainer-app", wait_for_response=False)
        def run_streamlit():
            """Optimized Streamlit interface"""
            import subprocess
            import sys

            # Pre-warm model loading
            _preload_models()

            # Run Streamlit with optimized settings
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
            "--server.maxUploadSize",
            "200",
            "--server.enableCORS",
            "false",
            "--server.enableXsrfProtection",
            "true",
            "--browser.serverAddress",
            "mltrainer.windfuhr.net",
            "--browser.gatherUsageStats",
            "false",
            "--global.dataFrameSerialization",
            "arrow",
            ]
            )


            # API with connection pooling and caching
            @stub.function(
            image=mltrainer_image,
            shared_volumes={"/models": model_volume, "/cache": cache_volume, "/data": data_volume},
            cpu=1.0,
            memory=2048,
            keep_warm=get_container_count(),
            container_idle_timeout=3600,
            concurrency_limit=200,
            retries=3,
            )
            @modal.web_endpoint(label="mltrainer-api")
            async def api_endpoint(request: Dict[str, Any]):
                """Fast API endpoint with caching"""
                import aioredis
                from ml_engine_real import RealMLEngine

                # Check cache first
                cache_key = f"prediction:{request['symbol']}:{request.get('model', 'default')}"

                # Try cache
                cache = await _get_cache_connection()
                cached_result = await cache.get(cache_key)

                if cached_result:
                    return json.loads(cached_result)

                    # Not in cache, compute
                    engine = _get_ml_engine()  # Reuse engine instance
                    result = await engine.predict_async(symbol=request["symbol"], model=request.get("model", "random_forest"))

                    # Cache result (5 minute TTL)
                    await cache.setex(cache_key, 300, json.dumps(result))

                    return result


                    # Optimized training with GPU scheduling
                    @stub.function(
                    image=mltrainer_image,
                    shared_volumes={"/models": model_volume, "/data": data_volume},
                    gpu=modal.gpu.T4(count=1),  # Single T4 GPU
                    cpu=4.0,
                    memory=16384,
                    timeout=3600,
                    retries=1,  # Don't retry expensive GPU operations
                    )
                    async def train_model_optimized(config: Dict[str, Any]):
                        """GPU-accelerated training with optimization"""
                        from ml_engine_real import RealMLEngine
                        from config.config_loader import get_config_loader

                        # Set up optimized training
                        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                        os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"

                        # Initialize
                        config_loader = get_config_loader()
                        engine = RealMLEngine(use_gpu=True)

                        # Train with progress tracking
                        result = await engine.train_model_async(config, callbacks=[_progress_callback])

                        # Save model with compression
                        model_path = f"/models/{config['model']}_{datetime.now().isoformat()}.pkl.gz"
                        await engine.save_model_compressed(result["model_id"], model_path)

                        # Update model registry
                        await _update_model_registry(result)

                        return result


                        # Health monitoring endpoint
                        @stub.function(cpu=0.5, memory=512, keep_warm=1, container_idle_timeout=86400)
                        @modal.web_endpoint(label="mltrainer-health")
                        def health_check():
                            """Health and readiness check"""
                            checks = {
                            "status": "healthy",
                            "timestamp": datetime.now().isoformat(),
                            "checks": {
                            "models": _check_models(),
                            "cache": _check_cache(),
                            "memory": _check_memory(),
                            "api": _check_api_health(),
                            },
                            "metrics": {
                            "container_count": get_container_count(),
                            "uptime": _get_uptime(),
                            "request_count": _get_request_count(),
                            },
                            }

                            # Return 503 if any critical check fails
                            if not all(checks["checks"].values()):
                                return {"status": 503, "body": checks}

                                return checks


                                # Scheduled optimization tasks
                                @stub.function(image=mltrainer_image, schedule=modal.Cron("0 2 * * *"), cpu=2.0, memory=4096)  # 2 AM daily
                                async def optimize_and_cleanup():
                                    """Daily optimization and cleanup tasks"""
                                    # Clean old cache entries
                                    await _cleanup_cache()

                                    # Compress old models
                                    await _compress_old_models()

                                    # Update performance metrics
                                    await _calculate_performance_metrics()

                                    # Optimize model registry
                                    await _optimize_model_registry()


                                    # Helper functions
                                    def _preload_models():
                                        """Preload frequently used models into memory"""
                                        from config.config_loader import get_config

                                        models_to_preload = get_config(
                                        "optimization.preload_models", ["random_forest_default", "lstm_basic", "xgboost_balanced"]
                                        )

                                        # Load models into memory
                                        for model_name in models_to_preload:
                                            try:
                                                _load_model_to_memory(model_name)
                                                except Exception as e:
                                                    logger.error(f"Failed to preload {model_name}: {e}")


                                                    async def _get_cache_connection():
                                                        """Get Redis connection with pooling"""
                                                        import aioredis

                                                        return await aioredis.create_redis_pool("redis://mltrainer-cache:6379", minsize=5, maxsize=10)


                                                        def _get_ml_engine():
                                                            """Get ML engine instance (singleton pattern)"""
                                                            if not hasattr(_get_ml_engine, "_instance"):
                                                                from ml_engine_real import RealMLEngine

                                                                _get_ml_engine._instance = RealMLEngine(cache_enabled=True, connection_pool_size=20)
                                                                return _get_ml_engine._instance


                                                                # Deployment configuration
                                                                @stub.local_entrypoint()
                                                                def deploy():
                                                                    """Deploy with optimization settings"""
                                                                    logger.info("🚀 Deploying optimized mlTrainer to Modal# Production code implemented")
                                                                    logger.info("⚡ Optimizations enabled:")
                                                                    logger.info("  - Dynamic container scaling based on market hours")
                                                                    logger.info("  - Redis caching with 5-minute TTL")
                                                                    logger.info("  - Model preloading for common models")
                                                                    logger.info("  - Connection pooling for APIs")
                                                                    logger.info("  - GPU optimization for training")
                                                                    logger.info("  - Health monitoring and auto-recovery")
                                                                    logger.info("\n📍 Custom domains:")
                                                                    logger.info("  - App: https://mltrainer.windfuhr.net")
                                                                    logger.info("  - API: https://api.mltrainer.windfuhr.net")
                                                                    logger.info("  - Health: https://monitor.mltrainer.windfuhr.net/health")
