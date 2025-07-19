import logging
import traceback
import json
from datetime import datetime
from typing import Optional, Dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ERROR_LOG_FILE = "logs/system_errors.log"
ERROR_MEMORY = []

def log_exception(e: Exception, context: Optional[str] = None) -> Dict[str, any]:
    """
    Logs full exception details with traceback and optional context tag.
    Also stores a short record in memory for recent errors.
    """
    timestamp = datetime.utcnow().isoformat()
    error_type = type(e).__name__
    tb = traceback.format_exc()

    error_entry = {
        "timestamp": timestamp,
        "type": error_type,
        "message": str(e),
        "traceback": tb,
        "context": context or "unspecified"
    }

    logger.error(f"❌ {error_type} at {timestamp}: {str(e)}")
    logger.debug(tb)

    # Append to in-memory tracker
    ERROR_MEMORY.append(error_entry)
    if len(ERROR_MEMORY) > 100:
        ERROR_MEMORY.pop(0)

    # Also persist to file
    try:
        with open(ERROR_LOG_FILE, "a") as f:
            f.write(json.dumps(error_entry, indent=2) + "\n")
    except Exception as file_err:
        logger.warning(f"⚠️ Could not write to error log file: {file_err}")

    return error_entry

def get_recent_errors(limit: int = 10) -> list:
    """
    Returns the most recent system errors (in-memory).
    """
    return ERROR_MEMORY[-limit:]

def log_error(message: str, details: str = None, context: str = None) -> None:
    """
    Log an error message with optional details and context.
    """
    timestamp = datetime.utcnow().isoformat()
    context_info = f" [{context}]" if context else ""
    logger.error(f"❌ {message}{context_info} at {timestamp}")
    if details:
        logger.error(f"Details: {details}")

def catch_and_log(func):
    """
    Decorator to wrap a function with exception logging.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            log_exception(e, context=func.__name__)
            return None
    return wrapper
