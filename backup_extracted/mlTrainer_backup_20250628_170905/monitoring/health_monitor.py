import logging
import os
import time
import psutil
import socket
from datetime import datetime

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def check_internet_connection(host="8.8.8.8", port=53, timeout=3) -> bool:
    """
    Checks basic internet connection via DNS ping (Google).
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except Exception:
        return False

def get_system_health() -> dict:
    """
    Returns a dict summarizing system resource and status health.
    """
    try:
        cpu = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory().percent
        disk = psutil.disk_usage("/").percent
        net_ok = check_internet_connection()

        health_report = {
            "timestamp": datetime.utcnow().isoformat(),
            "cpu_usage_percent": round(cpu, 2),
            "memory_usage_percent": round(memory, 2),
            "disk_usage_percent": round(disk, 2),
            "internet_ok": net_ok,
            "status": "OK" if all([cpu < 90, memory < 90, disk < 90, net_ok]) else "DEGRADED"
        }

        logger.info(f"📡 System health check: {health_report}")
        return health_report

    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "ERROR",
            "error": str(e)
        }

def get_health_status() -> dict:
    """
    Alias for get_system_health for backward compatibility.
    """
    return get_system_health()

def log_status(message: str, context: dict = None) -> None:
    """
    Log a status message with optional context.
    """
    if context:
        logger.info(f"📊 {message}: {context}")
    else:
        logger.info(f"📊 {message}")