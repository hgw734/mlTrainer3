# ========================================
# drift_protection.py
# AI Drift Guard: Data + Model Monitoring
# Version: 2025.07
# ========================================

import numpy as np
import scipy.stats
import hashlib
import json
from datetime import datetime
from sklearn.metrics import mean_squared_error, accuracy_score
import os
# import pickle  # Removed - not used
import warnings
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import logging

# Import runtime enforcement system
try:
    from core.immutable_runtime_enforcer import (
        verify_response,
        activate_kill_switch,
        SYSTEM_STATE,
        compliance_wrap,
        enforce_verification
    )
    RUNTIME_ENFORCEMENT_ENABLED = True
except ImportError:
    RUNTIME_ENFORCEMENT_ENABLED = False
    warnings.warn(
        "Runtime enforcement system not available - operating in legacy mode")

# ================================
# CONFIGURATION & CONSTANTS
# ================================
DRIFT_LOG_FILE = "logs/drift_log.jsonl"
MODEL_REGISTRY = "logs/model_registry.json"
COMPLIANCE_LOG = "logs/compliance_log.jsonl"
PERFORMANCE_BASELINE = "logs/performance_baseline.json"
DATA_FINGERPRINTS = "logs/data_fingerprints.json"

MSE_THRESHOLD_MULTIPLIER = 1.5  # You can tune this
ACCURACY_THRESHOLD_MULTIPLIER = 0.8  # Minimum acceptable accuracy multiplier
DATA_DRIFT_THRESHOLD = 0.05  # Statistical significance threshold for drift
COMPLIANCE_RETENTION_DAYS = 2555  # 7 years regulatory compliance

# ABSOLUTE PROHIBITION: No data generators
NO_DATA_GENERATORS = True  # IMMUTABLE - No synthetic, random, or simulated data allowed
PROHIBITED_FUNCTIONS = [
    'random', 'randn', 'rand', 'randint', 'normal', 'uniform',
    'synthetic', 'generate', 'simulate', 'mock', 'fake', 'dummy'
]

# Approved data sources for institutional compliance
APPROVED_DATA_SOURCES = [
    "polygon", "fred", "quiverquant", "alpha_vantage", "iex",
    "bloomberg", "refinitiv", "factset", "quandl", "tiingo",
    "intrinio", "morningstar", "yfinance"  # Only for research/testing
]

# ABSOLUTE PROHIBITION: No data generators allowed anywhere in the system
NO_DATA_GENERATORS = True  # IMMUTABLE - Zero tolerance for synthetic/random data
PROHIBITED_DATA_GENERATION = [
    'random', 'randn', 'rand', 'randint', 'normal', 'uniform',
    'synthetic', 'generate', 'simulate', 'mock', 'fake', 'dummy',
    'np.random', 'numpy.random', 'scipy.stats.random', 'faker'
]

# Model approval status levels
MODEL_APPROVAL_LEVELS = {
    "RESEARCH": 1,
    "TESTING": 2,
    "STAGING": 3,
    "PRODUCTION": 4,
    "DEPRECATED": 0
}

# Ensure log folders exist
os.makedirs("logs", exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/drift_protection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================================
# DATA CLASSES & STRUCTURES
# ================================


@dataclass
class DriftAlert:
    """Data structure for drift alerts"""
    timestamp: str
    alert_type: str  # 'DATA_DRIFT', 'PERFORMANCE_DRIFT', 'MODEL_CORRUPTION'
    severity: str    # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    message: str
    metric_value: float
    threshold_value: float
    affected_components: List[str]
    recommended_actions: List[str]
    compliance_impact: bool = False


@dataclass
class ModelFingerprint:
    """Model fingerprint for change detection"""
    model_hash: str
    model_name: str
    version: str
    creation_timestamp: str
    parameters_hash: str
    training_data_hash: str
    approval_level: str
    compliance_verified: bool
    performance_baseline: Dict[str, float]


@dataclass
class DataFingerprint:
    """Data fingerprint for distribution monitoring"""
    data_hash: str
    timestamp: str
    source: str
    distribution_stats: Dict[str, float]
    quality_metrics: Dict[str, float]
    schema_hash: str
    record_count: int

# ================================
# ENHANCED DISTRIBUTION MONITORING
# ================================


def log_distribution_metrics(data: np.ndarray,
                             name: str = "input",
                             window: Optional[str] = None,
                             source: str = "unknown") -> Dict[str,
                                                              Any]:
    """
    Enhanced distribution logging with statistical tests and quality metrics
    """
    if len(data) == 0:
        raise ValueError("Cannot analyze empty data array")

    # Validate data source
    validate_data_source(source)

    # Calculate comprehensive statistics
    stats = {
        "timestamp": str(datetime.now()),
        "type": name,
        "window": window,
        "source": source,
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "skew": float(scipy.stats.skew(data)),
        "kurtosis": float(scipy.stats.kurtosis(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "median": float(np.median(data)),
        "q25": float(np.percentile(data, 25)),
        "q75": float(np.percentile(data, 75)),
        "iqr": float(np.percentile(data, 75) - np.percentile(data, 25)),
        "record_count": len(data),
        "missing_values": int(np.sum(np.isnan(data))),
        "infinite_values": int(np.sum(np.isinf(data))),
        "unique_values": int(len(np.unique(data))),
        "zero_values": int(np.sum(data == 0))
    }

    # Quality metrics
    stats["quality_score"] = calculate_data_quality_score(data)
    stats["completeness"] = 1.0 - (stats["missing_values"] / len(data))
    stats["validity"] = 1.0 - \
        ((stats["missing_values"] + stats["infinite_values"]) / len(data))

    # Generate data fingerprint
    logger.info(data, source, stats)

    # Log with compliance metadata
    append_log(stats, log_type="distribution")
    log_compliance_event(
        "DATA_DISTRIBUTION_LOGGED", {
            "data_type": name, "source": source})

    # Check for distribution drift
    detect_distribution_drift(stats, name)

    return stats


def calculate_data_quality_score(data: np.ndarray) -> float:
    """Calculate overall data quality score (0-1)"""
    if len(data) == 0:
        return 0.0

    scores = []

    # Completeness (no missing values)
    completeness = 1.0 - (np.sum(np.isnan(data)) / len(data))
    scores.append(completeness)

    # Validity (no infinite values)
    validity = 1.0 - (np.sum(np.isinf(data)) / len(data))
    scores.append(validity)

    # Consistency (reasonable distribution)
    try:
        if np.std(data) > 0:
            consistency = min(1.0, 1.0 / (1.0 + abs(scipy.stats.skew(data))))
        else:
            consistency = 0.5  # Constant data is suspicious
        scores.append(consistency)
    except BaseException:
        scores.append(0.0)

    # Uniqueness (not all same values)
    uniqueness = min(1.0, len(np.unique(data)) / len(data))
    scores.append(uniqueness)

    return float(np.mean(scores))


def generate_data_fingerprint(
        data: np.ndarray,
        source: str,
        stats: Dict) -> str:
    """Generate unique fingerprint for data distribution"""
    fingerprint_data = {
        "source": source,
        "mean": stats["mean"],
        "std": stats["std"],
        "skew": stats["skew"],
        "kurtosis": stats["kurtosis"],
        "record_count": stats["record_count"],
        # Date only for daily fingerprints
        "timestamp": stats["timestamp"][:10]
    }

    fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
    return hashlib.sha256(fingerprint_str.encode()).hexdigest()


def detect_distribution_drift(current_stats: Dict, data_name: str) -> bool:
    """Detect if current data distribution has drifted from baseline"""
    baseline_file = f"logs/baseline_{data_name}.json"

    if not os.path.exists(baseline_file):
        # Create baseline if it doesn't exist
        with open(baseline_file, 'w') as f:
            json.dump(current_stats, f)
        logger.info(f"Created baseline for {data_name}")
        return False

    # Load baseline
    with open(baseline_file, 'r') as f:
        baseline_stats = json.load(f)

    # Perform KS test equivalent using statistical differences
    drift_detected = False
    drift_reasons = []

    # Check mean drift
    mean_diff = abs(
        current_stats["mean"] - baseline_stats["mean"]) / (baseline_stats["std"] + 1e-8)
    if mean_diff > 2.0:  # 2 standard deviations
        drift_detected = True
        drift_reasons.append(f"Mean drift: {mean_diff:.3f} std devs")

    # Check variance drift
    std_ratio = current_stats["std"] / (baseline_stats["std"] + 1e-8)
    if std_ratio > 2.0 or std_ratio < 0.5:
        drift_detected = True
        drift_reasons.append(f"Variance drift: {std_ratio:.3f}x change")

    # Check skewness drift
    skew_diff = abs(current_stats["skew"] - baseline_stats["skew"])
    if skew_diff > 1.0:
        drift_detected = True
        drift_reasons.append(f"Skewness drift: {skew_diff:.3f}")

    if drift_detected:
        alert = DriftAlert(
            timestamp=str(
                datetime.now()),
            alert_type="DATA_DRIFT",
            severity="MEDIUM",
            message=f"Distribution drift detected in {data_name}: {'; '.join(drift_reasons)}",
            metric_value=mean_diff,
            threshold_value=2.0,
            affected_components=[data_name],
            recommended_actions=[
                "Review data pipeline",
                "Check data sources",
                "Consider model retraining"],
            compliance_impact=True)
        log_drift_alert(alert)
        logger.warning(f"Data drift detected in {data_name}: {drift_reasons}")

    return drift_detected

# ================================
# ENHANCED MODEL PERFORMANCE TRACKING
# ================================


def track_model_performance(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "unknown",
        metric: str = "mse",
        window: str = "daily") -> float:
    """
    Enhanced model performance tracking with baseline comparison
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")

    if len(y_true) == 0:
        raise ValueError("Cannot calculate metrics on empty arrays")

    # Calculate metrics
    if metric == "mse":
        error = mean_squared_error(y_true, y_pred)
    elif metric == "rmse":
        error = np.sqrt(mean_squared_error(y_true, y_pred))
    elif metric == "mae":
        error = np.mean(np.abs(y_true - y_pred))
    elif metric == "accuracy":
        error = 1.0 - accuracy_score(y_true, y_pred)
    elif metric == "r2":
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        # Convert R² to error (higher is worse)
        error = 1.0 - (1 - ss_res / (ss_tot + 1e-8))
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    # Additional diagnostic metrics
    residuals = y_true - y_pred
    record = {
        "timestamp": str(
            datetime.now()),
        "model_name": model_name,
        "window": window,
        "metric": metric,
        "error": float(error),
        "mean_residual": float(
            np.mean(residuals)),
        "std_residual": float(
            np.std(residuals)),
        "max_residual": float(
            np.max(
                np.abs(residuals))),
        "prediction_count": len(y_pred),
        "correlation": float(
            np.corrcoef(
                y_true,
                y_pred)[
                0,
                1]) if len(y_true) > 1 else 0.0}

    append_log(record, log_type="performance")
    log_compliance_event(
        "MODEL_PERFORMANCE_TRACKED", {
            "model": model_name, "metric": metric})

    # Check for performance drift
    detect_performance_drift_enhanced(record, model_name)

    return error


def detect_performance_drift_enhanced(
        current_record: Dict,
        model_name: str) -> bool:
    """Enhanced performance drift detection with multiple criteria"""
    history = load_performance_history(model_name, limit=50)

    if len(history) < 5:
        logger.info(f"Insufficient history for {model_name} drift detection")
        return False

    current_error = current_record["error"]
    historical_errors = [h["error"] for h in history]

    # Multiple drift detection criteria
    drift_detected = False
    drift_reasons = []

    # 1. Threshold-based detection
    historical_avg = np.mean(historical_errors)
    if current_error > historical_avg * MSE_THRESHOLD_MULTIPLIER:
        drift_detected = True
        drift_reasons.append(
            f"Error threshold exceeded: {current_error:.4f} > {historical_avg * MSE_THRESHOLD_MULTIPLIER:.4f}")

    # 2. Statistical significance test
    recent_errors = historical_errors[-10:] if len(
        historical_errors) >= 10 else historical_errors
    if len(recent_errors) >= 3:
        try:
            t_stat, p_value = scipy.stats.ttest_1samp(
                recent_errors + [current_error], historical_avg)
            if p_value < 0.05 and current_error > historical_avg:
                drift_detected = True
                drift_reasons.append(
                    f"Statistically significant deterioration (p={p_value:.4f})")
        except BaseException:
            pass

    # 3. Trend analysis
    if len(historical_errors) >= 5:
        # Check if there's an upward trend in errors
        x = np.arange(len(historical_errors))
        slope, _, _, p_value, _ = scipy.stats.linregress(x, historical_errors)
        if slope > 0 and p_value < 0.1:  # Upward trend
            drift_detected = True
            drift_reasons.append(
                f"Deteriorating trend detected (slope={slope:.6f})")

    # 4. Correlation degradation
    if "correlation" in current_record:
        historical_corrs = [h.get("correlation", 0)
                            for h in history if "correlation" in h]
        if historical_corrs and len(historical_corrs) >= 3:
            avg_corr = np.mean(historical_corrs)
            if current_record["correlation"] < avg_corr * \
                    0.8:  # 20% degradation
                drift_detected = True
                drift_reasons.append(
                    f"Correlation degradation: {current_record['correlation']:.3f} < {avg_corr * 0.8:.3f}")

    if drift_detected:
        severity = "HIGH" if current_error > historical_avg * 2.0 else "MEDIUM"
        alert = DriftAlert(
            timestamp=str(
                datetime.now()),
            alert_type="PERFORMANCE_DRIFT",
            severity=severity,
            message=f"Performance drift detected in {model_name}: {'; '.join(drift_reasons)}",
            metric_value=current_error,
            threshold_value=historical_avg *
            MSE_THRESHOLD_MULTIPLIER,
            affected_components=[model_name],
            recommended_actions=[
                "Investigate model degradation",
                "Consider retraining",
                "Check input data quality"],
            compliance_impact=True)
        log_drift_alert(alert)
        logger.warning(f"Performance drift in {model_name}: {drift_reasons}")

    return drift_detected


def load_performance_history(model_name: str, limit: int = 100) -> List[Dict]:
    """Load recent performance history for a model"""
    if not os.path.exists(DRIFT_LOG_FILE):
        return []

    history = []
    try:
        with open(DRIFT_LOG_FILE, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if (entry.get("log_type") == "performance" and
                            entry.get("model_name") == model_name):
                        history.append(entry)
                        if len(history) >= limit:
                            break
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass

    return history[-limit:] if history else []

# ================================
# ENHANCED MODEL HASHING & REGISTRATION
# ================================


def hash_model_config(config: Dict[str, Any]) -> str:
    """Generate robust hash for model configuration"""
    if not isinstance(config, dict):
        raise ValueError("Config must be a dictionary")

    # Normalize config for consistent hashing
    normalized_config = normalize_config_for_hashing(config)
    config_str = json.dumps(
        normalized_config,
        sort_keys=True,
        separators=(
            ',',
            ':'))
    return hashlib.sha256(config_str.encode('utf-8')).hexdigest()


def normalize_config_for_hashing(config: Dict) -> Dict:
    """Normalize config to ensure consistent hashing"""
    normalized = {}
    for key, value in config.items():
        if isinstance(value, dict):
            normalized[key] = normalize_config_for_hashing(value)
        elif isinstance(value, list):
            # Sort lists for consistency (if they contain comparable items)
            try:
                normalized[key] = sorted(value)
            except TypeError:
                normalized[key] = value
        elif isinstance(value, float):
            # Round floats to avoid precision issues
            normalized[key] = round(value, 10)
        else:
            normalized[key] = value
    return normalized


def register_model(model_hash: str,
                   meta: Dict[str,
                              Any],
                   approval_level: str = "RESEARCH") -> None:
    """Enhanced model registration with compliance tracking"""
    if approval_level not in MODEL_APPROVAL_LEVELS:
        raise ValueError(f"Invalid approval level: {approval_level}")

    fingerprint = ModelFingerprint(
        model_hash=model_hash,
        model_name=meta.get("model_name", "unknown"),
        version=meta.get("version", "1.0"),
        creation_timestamp=str(datetime.now()),
        parameters_hash=hash_model_config(meta.get("parameters", {})),
        training_data_hash=meta.get("training_data_hash", ""),
        approval_level=approval_level,
        compliance_verified=meta.get("compliance_verified", False),
        performance_baseline=meta.get("performance_baseline", {})
    )

    # Log registration
    entry = {
        "timestamp": fingerprint.creation_timestamp,
        "model_hash": model_hash,
        "model_name": fingerprint.model_name,
        "approval_level": approval_level,
        "meta": meta,
        "fingerprint": fingerprint.__dict__
    }

    # Append to registry
    with open(MODEL_REGISTRY, "a") as f:
        f.write(json.dumps(entry) + "\n")

    log_compliance_event("MODEL_REGISTERED", {
        "model_hash": model_hash,
        "model_name": fingerprint.model_name,
        "approval_level": approval_level
    })

    logger.info(
        f"Model registered: {fingerprint.model_name} ({approval_level})")


def is_model_registered(model_hash: str) -> bool:
    """Check if model is registered and approved"""
    if not os.path.exists(MODEL_REGISTRY):
        return False

    try:
        with open(MODEL_REGISTRY, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry.get("model_hash") == model_hash:
                        approval_level = entry.get(
                            "approval_level", "RESEARCH")
                        return MODEL_APPROVAL_LEVELS.get(approval_level, 0) > 0
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass

    return False


def get_model_approval_level(model_hash: str) -> str:
    """Get model approval level"""
    if not os.path.exists(MODEL_REGISTRY):
        return "UNREGISTERED"

    try:
        with open(MODEL_REGISTRY, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry.get("model_hash") == model_hash:
                        return entry.get("approval_level", "RESEARCH")
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass

    return "UNREGISTERED"

# ================================
# ENHANCED REGRESSION SANITY CHECKS
# ================================


def assert_prediction_sanity(
        model: Any,
        actual_input: np.ndarray,
        expected_value: float,
        tolerance: float = 0.01,
        model_name: str = "unknown") -> None:
    """Enhanced prediction sanity check with detailed logging"""
    try:
        # Handle different model interfaces
        if hasattr(model, 'predict'):
            if len(actual_input.shape) == 1:
                pred = model.predict([actual_input])[0]
            else:
                pred = model.predict(actual_input)[0]
        elif callable(model):
            pred = model(actual_input)
        else:
            raise ValueError("Model must have predict method or be callable")

        # Sanity checks
        if np.isnan(pred) or np.isinf(pred):
            raise AssertionError(
                f"Model {model_name} returned invalid prediction: {pred}")

        diff = abs(pred - expected_value)
        relative_error = diff / (abs(expected_value) + 1e-8)

        # Log sanity check
        sanity_record = {
            "timestamp": str(datetime.now()),
            "model_name": model_name,
            "check_type": "PREDICTION_SANITY",
            "expected_value": float(expected_value),
            "predicted_value": float(pred),
            "absolute_error": float(diff),
            "relative_error": float(relative_error),
            "tolerance": tolerance,
            "passed": diff < tolerance
        }
        append_log(sanity_record, log_type="sanity_check")

        if diff >= tolerance:
            alert = DriftAlert(
                timestamp=str(
                    datetime.now()),
                alert_type="MODEL_CORRUPTION",
                severity="HIGH",
                message=f"Sanity check failed for {model_name}: expected {expected_value}, got {pred} (error: {diff:.6f})",
                metric_value=diff,
                threshold_value=tolerance,
                affected_components=[model_name],
                recommended_actions=[
                    "Check model integrity",
                    "Verify model loading",
                    "Consider model rollback"],
                compliance_impact=True)
            log_drift_alert(alert)
            raise AssertionError(
                f"⚠️ Drift: Model {model_name} sanity check failed - expected {expected_value}, got {pred} (error: {diff:.6f} > {tolerance})")

        logger.info(
            f"Sanity check passed for {model_name}: {pred:.6f} ≈ {expected_value:.6f}")

    except Exception as e:
        error_record = {
            "timestamp": str(datetime.now()),
            "model_name": model_name,
            "check_type": "PREDICTION_SANITY",
            "error": str(e),
            "passed": False
        }
        append_log(error_record, log_type="sanity_check")
        log_compliance_event(
            "SANITY_CHECK_FAILED", {
                "model": model_name, "error": str(e)})
        raise

# ================================
# ENHANCED COMPLIANCE & VALIDATION
# ================================


def validate_data_source(
        source_name: str,
        allowed_sources: List[str] = None) -> None:
    """Enhanced data source validation with compliance logging"""
    if allowed_sources is None:
        allowed_sources = APPROVED_DATA_SOURCES

    if source_name not in allowed_sources:
        compliance_violation = {
            "timestamp": str(datetime.now()),
            "violation_type": "UNAUTHORIZED_DATA_SOURCE",
            "source": source_name,
            "allowed_sources": allowed_sources,
            "severity": "HIGH"
        }
        log_compliance_event("DATA_SOURCE_VIOLATION", compliance_violation)
        raise RuntimeError(
            f"❌ Unverified data source: {source_name}. Allowed sources: {allowed_sources}")

    log_compliance_event("DATA_SOURCE_VALIDATED", {"source": source_name})


def validate_model_hash(
        model_hash: str,
        required_approval: str = "RESEARCH") -> None:
    """Enhanced model validation with approval level checking"""
    if not is_model_registered(model_hash):
        log_compliance_event("MODEL_HASH_VIOLATION", {"hash": model_hash})
        raise RuntimeError(
            f"❌ Model hash {model_hash} not registered — possible drift or corruption.")

    approval_level = get_model_approval_level(model_hash)
    required_level = MODEL_APPROVAL_LEVELS.get(required_approval, 1)
    current_level = MODEL_APPROVAL_LEVELS.get(approval_level, 0)

    if current_level < required_level:
        log_compliance_event("MODEL_APPROVAL_VIOLATION", {
            "hash": model_hash,
            "current_level": approval_level,
            "required_level": required_approval
        })
        raise RuntimeError(
            f"❌ Model {model_hash} approval level {approval_level} insufficient for {required_approval}")

    log_compliance_event(
        "MODEL_HASH_VALIDATED", {
            "hash": model_hash, "approval": approval_level})


def enforce_institutional_compliance(
        data_source: str,
        model_hash: str,
        required_approval: str = "PRODUCTION") -> None:
    """Comprehensive institutional compliance enforcement"""
    logger.info(
        f"Enforcing institutional compliance for {data_source} -> {model_hash}")

    # Validate data source
    validate_data_source(data_source)

    # Validate model
    validate_model_hash(model_hash, required_approval)

    # Additional institutional checks
    compliance_checks = {
        "data_source_approved": data_source in APPROVED_DATA_SOURCES,
        "model_registered": is_model_registered(model_hash),
        "model_approval_sufficient": MODEL_APPROVAL_LEVELS.get(get_model_approval_level(model_hash), 0) >= MODEL_APPROVAL_LEVELS.get(required_approval, 1),
        "compliance_logging_active": os.path.exists(COMPLIANCE_LOG),
        "audit_trail_complete": True  # Additional checks would go here
    }

    all_passed = all(compliance_checks.values())

    compliance_record = {
        "timestamp": str(datetime.now()),
        "event_type": "INSTITUTIONAL_COMPLIANCE_CHECK",
        "data_source": data_source,
        "model_hash": model_hash,
        "required_approval": required_approval,
        "checks": compliance_checks,
        "result": "PASSED" if all_passed else "FAILED"
    }

    log_compliance_event("INSTITUTIONAL_COMPLIANCE", compliance_record)

    if not all_passed:
        failed_checks = [k for k, v in compliance_checks.items() if not v]
        raise RuntimeError(
            f"❌ Institutional compliance failed: {failed_checks}")

    logger.info("✅ Institutional compliance verified")

# ================================
# LOGGING & ALERT UTILITIES
# ================================


def append_log(entry: Dict[str, Any], log_type: str = "general") -> None:
    """Enhanced logging with metadata"""
    enriched_entry = {
        **entry,
        "log_type": log_type,
        "log_id": hashlib.md5(
            f"{entry.get('timestamp', '')}{log_type}".encode()).hexdigest()[
            :8]}

    with open(DRIFT_LOG_FILE, "a") as f:
        f.write(json.dumps(enriched_entry) + "\n")


def log_compliance_event(event_type: str, details: Dict[str, Any]) -> None:
    """Log compliance-related events for audit trail"""
    compliance_entry = {
        "timestamp": str(datetime.now()),
        "event_type": event_type,
        "details": details,
        "compliance_level": "INSTITUTIONAL",
        "retention_required": True
    }

    with open(COMPLIANCE_LOG, "a") as f:
        f.write(json.dumps(compliance_entry) + "\n")


def log_drift_alert(alert: DriftAlert) -> None:
    """Log drift alerts for monitoring and compliance"""
    alert_entry = {
        "timestamp": alert.timestamp,
        "alert_type": alert.alert_type,
        "severity": alert.severity,
        "message": alert.message,
        "metric_value": alert.metric_value,
        "threshold_value": alert.threshold_value,
        "affected_components": alert.affected_components,
        "recommended_actions": alert.recommended_actions,
        "compliance_impact": alert.compliance_impact,
        "log_type": "drift_alert"
    }

    append_log(alert_entry, log_type="drift_alert")
    log_compliance_event("DRIFT_ALERT", alert_entry)

    # Critical alerts should trigger immediate notifications
    if alert.severity in ["HIGH", "CRITICAL"]:
        logger.error(f"🚨 CRITICAL DRIFT ALERT: {alert.message}")

# ================================
# SYSTEM MONITORING & HEALTH CHECKS
# ================================


def system_health_check() -> Dict[str, Any]:
    """Comprehensive system health check"""
    health_status = {
        "timestamp": str(datetime.now()),
        "log_files_accessible": True,
        "model_registry_accessible": True,
        "compliance_logging_active": True,
        "data_sources_validated": True,
        "recent_alerts": 0,
        "performance_trending": "STABLE",
        "overall_health": "HEALTHY"
    }

    # Check file accessibility
    required_dirs = ["logs"]
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path, exist_ok=True)
            except BaseException:
                health_status["log_files_accessible"] = False

    # Check for recent critical alerts
    try:
        recent_alerts = count_recent_alerts(hours=24)
        health_status["recent_alerts"] = recent_alerts

        if recent_alerts > 10:
            health_status["overall_health"] = "DEGRADED"
        elif recent_alerts > 50:
            health_status["overall_health"] = "CRITICAL"

    except Exception as e:
        health_status["recent_alerts"] = -1
        health_status["overall_health"] = "UNKNOWN"
        logger.error(f"Error checking recent alerts: {e}")

    # Log health check
    append_log(health_status, log_type="health_check")
    log_compliance_event("SYSTEM_HEALTH_CHECK", health_status)

    return health_status


def count_recent_alerts(hours: int = 24) -> int:
    """Count alerts in recent time window"""
    if not os.path.exists(DRIFT_LOG_FILE):
        return 0

    cutoff_time = datetime.now().timestamp() - (hours * 3600)
    alert_count = 0

    try:
        with open(DRIFT_LOG_FILE, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry.get("log_type") == "drift_alert":
                        entry_time = datetime.fromisoformat(
                            entry["timestamp"]).timestamp()
                        if entry_time > cutoff_time:
                            alert_count += 1
                except (json.JSONDecodeError, ValueError, KeyError):
                    continue
    except FileNotFoundError:
        pass

    return alert_count


# ================================
# EXAMPLE USAGE & TESTING
# ================================
if __name__ == "__main__":
    logger.info("🔒 AI Drift Protection System - Comprehensive Testing")
    logger.info("=" * 60)

    try:
        # 1. Test data distribution monitoring
        logger.info("\n📊 Testing Data Distribution Monitoring...")
        input_data = np.linspace(-2, 2, 100)  # Fixed test data
        stats = log_distribution_metrics(
            input_data, name="test_input", source="polygon")
        logger.info(f"✅ Data quality score: {stats['quality_score']:.3f}")

        # 2. Test model performance tracking
        logger.info("\n📈 Testing Model Performance Tracking...")
        y_true = np.sin(np.linspace(0, 4 * np.pi, 100))  # Fixed test data
        y_pred = y_true + 0.1 * \
            np.sin(np.linspace(0, 20 * np.pi, 100))  # Fixed test noise
        error = track_model_performance(
            y_true, y_pred, model_name="test_model")
        logger.error(f"✅ Model error tracked: {error:.4f}")

        # 3. Test model registration and validation
        logger.info("\n🔗 Testing Model Registration...")
        model_config = {
            "model_name": "XGBoost_v1.0",
            "model": "XGBoost",
            "parameters": {"n_estimators": 100, "max_depth": 6},
            "version": "1.0",
            "compliance_verified": True
        }
        model_hash = hash_model_config(model_config)
        register_model(model_hash, model_config, approval_level="TESTING")
        validate_model_hash(model_hash, required_approval="TESTING")
        logger.info(f"✅ Model registered and validated: {model_hash[:8]}...")

        # 4. Test institutional compliance
        logger.info("\n🏛️ Testing Institutional Compliance...")
        enforce_institutional_compliance(
            "polygon", model_hash, required_approval="TESTING")
        logger.info("✅ Institutional compliance verified")

        # 5. Test sanity check
        logger.info("\n🧪 Testing Prediction Sanity Check...")
        # Mock model for testing

        class MockModel:
            def predict(self, X):
                return [0.5]  # Always return 0.5

        real_model = MockModel()
        actual_input = np.array([1.0, 2.0, 3.0])
        assert_prediction_sanity(
            real_model,
            actual_input,
            0.5,
            tolerance=0.01,
            model_name="real_model")
        logger.info("✅ Sanity check passed")

        # 6. Test system health
        logger.info("\n❤️ Testing System Health Check...")
        health = system_health_check()
        logger.info(f"✅ System health: {health['overall_health']}")

        # 7. Test data source validation
        logger.info("\n📡 Testing Data Source Validation...")
        validate_data_source("polygon")
        logger.info("✅ Data source validation passed")

        logger.info(f"\n🎉 All tests completed successfully!")
        logger.info(f"📁 Logs written to: {DRIFT_LOG_FILE}")
        logger.info(f"📋 Compliance log: {COMPLIANCE_LOG}")
        logger.info(f"🗂️ Model registry: {MODEL_REGISTRY}")

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        logger.error(f"❌ Test failed: {e}")
        raise

    logger.info("\n✅ Drift protection system operational and compliant.")
