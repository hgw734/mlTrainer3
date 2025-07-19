import logging
import json
from datetime import datetime
from monitoring.health_monitor import get_system_health
from monitoring.error_monitor import get_recent_errors

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def run_monitoring_cycle() -> dict:
    """
    Runs a full monitoring sweep and returns the combined system status report.
    Includes system health and recent error logs.
    """
    logger.info("🩺 Running monitoring cycle...")

    health = get_system_health()
    errors = get_recent_errors(limit=5)

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "system_health": health,
        "recent_errors": errors
    }

    logger.info(f"📊 Monitoring report generated")
    return report

def print_monitoring_report():
    """
    Print the current system status to console (or log viewer).
    """
    report = run_monitoring_cycle()
    print("\n🔍 System Monitoring Report:")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    print_monitoring_report()
