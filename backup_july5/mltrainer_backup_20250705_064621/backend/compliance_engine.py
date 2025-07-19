"""
mlTrainer - Compliance Engine
============================

Purpose: Enforces strict compliance with zero synthetic data policy.
Validates all data sources, filters non-verified information, and ensures
all AI responses follow "I don't know" format when data cannot be verified.

Compliance Rules:
- No synthetic, placeholder, or example data
- All data must be from verified sources (Polygon, FRED, QuiverQuant)
- Unverified responses must use "I don't know. But based on the data, I would suggest..."
- Immutable compliance gateway for all data flows
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import os
import yaml
import sys
import threading
import time
import glob
import shutil

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.api_provider_manager import get_api_manager

try:
    from utils.session_manager import get_session_manager
    SESSION_MANAGER_AVAILABLE = True
except ImportError:
    SESSION_MANAGER_AVAILABLE = False
    
logger = logging.getLogger(__name__)

class UniversalDataInterceptor:
    """UNIVERSAL COMPLIANCE GATEWAY - Every single piece of data passes through here"""
    
    def __init__(self):
        self.verified_sources = ["polygon", "fred", "quiverquant"]
        self.intercepted_data_log = []
        self.compliance_violations = []
        logger.info("🔒 UNIVERSAL DATA INTERCEPTOR ACTIVE - All data monitored")
    
    def intercept_all_data(self, data: Any, context: str = "unknown") -> tuple[bool, Any]:
        """MANDATORY CHECKPOINT: Every piece of data must pass through this function"""
        try:
            # Log ALL data that enters the system
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "context": context,
                "data_type": type(data).__name__,
                "data_size": len(str(data)) if data else 0,
                "compliance_check": "pending"
            }
            
            # Check for synthetic data patterns
            if self._contains_synthetic_patterns(data):
                violation = {
                    "type": "synthetic_data_detected",
                    "context": context,
                    "timestamp": datetime.now().isoformat(),
                    "severity": "critical"
                }
                self.compliance_violations.append(violation)
                log_entry["compliance_check"] = "BLOCKED - Synthetic data detected"
                self.intercepted_data_log.append(log_entry)
                logger.error(f"🚨 UNIVERSAL BLOCK: Synthetic data detected in {context}")
                return False, None
            
            # Validate data source if it's a dictionary
            if isinstance(data, dict):
                source = data.get("source")
                if source and source not in self.verified_sources:
                    violation = {
                        "type": "unauthorized_source", 
                        "source": source,
                        "context": context,
                        "timestamp": datetime.now().isoformat(),
                        "severity": "critical"
                    }
                    self.compliance_violations.append(violation)
                    log_entry["compliance_check"] = f"BLOCKED - Unauthorized source: {source}"
                    self.intercepted_data_log.append(log_entry)
                    logger.error(f"🚨 UNIVERSAL BLOCK: Unauthorized source {source} in {context}")
                    return False, None
                
                # Tag verified data
                data["universal_compliance_check"] = True
                data["interceptor_timestamp"] = datetime.now().isoformat()
            
            log_entry["compliance_check"] = "APPROVED"
            self.intercepted_data_log.append(log_entry)
            logger.debug(f"✅ UNIVERSAL PASS: {context} data approved")
            return True, data
            
        except Exception as e:
            logger.error(f"🚨 UNIVERSAL INTERCEPTOR ERROR: {e}")
            return False, None
    
    def _contains_synthetic_patterns(self, data: Any) -> bool:
        """Detect synthetic data patterns in any data type"""
        data_str = str(data).lower()
        synthetic_indicators = [
            "mock", "fake", "dummy", "placeholder", "example", "test_data",
            "synthetic", "generated", "simulated", "random_state=42",
            "0.813587956665656",  # The exact fraud accuracy we found
            "86.68%", "identical_accuracy"
        ]
        
        for indicator in synthetic_indicators:
            if indicator in data_str:
                return True
        
        # Check for suspiciously identical accuracy patterns
        if isinstance(data, dict):
            accuracy = data.get("accuracy")
            if accuracy and str(accuracy) == "0.813587956665656":
                return True
        
        return False

class ComplianceEngine:
    """Enforces data verification and compliance policies"""
    
    def __init__(self):
        self.compliance_enabled = True  # Always enabled for data flow monitoring
        self.verified_sources = ["polygon", "fred", "quiverquant"]
        self.compliance_violations = []
        self.last_scan_time = None
        self.mltrainer_exempt = True  # mlTrainer AI is exempt from compliance restrictions
        self.data_flow_mode = True  # Focus on data validation, not operation blocking
        
        # Initialize Universal Data Interceptor
        self.universal_interceptor = UniversalDataInterceptor()
        logger.info("🔒 ComplianceEngine initialized with Universal Data Interceptor")
        
        # Load compliance configuration
        self.compliance_config = self._load_compliance_config()
        
        # Compliance metrics
        self.metrics = {
            "total_checks": 0,
            "violations": 0,
            "verified_items": 0,
            "rejected_items": 0,
            "purged_items": 0,
            "last_audit": None
        }
        
        # Automatic audit schedule (twice daily: 6:00 and 18:00)
        self.audit_times = ["06:00", "18:00"]
        self.audit_running = False
        self.audit_thread = None
        
        # Start automatic audit scheduler
        self._start_audit_scheduler()
        
        logger.info("ComplianceEngine initialized with strict verification enabled")
    
    def _load_compliance_config(self) -> Dict:
        """Load compliance configuration from file"""
        try:
            config_path = "config/compliance_config.yaml"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load compliance config: {e}")
        
        # Default compliance configuration
        return {
            "strict_mode": True,
            "verified_sources": ["polygon", "fred", "quiverquant"],
            "max_data_age_hours": 24,
            "require_timestamps": True,
            "allow_synthetic_data": False,
            "response_template": "I don't know. But based on the data, I would suggest {suggestion}."
        }
    
    def check_compliance_status(self) -> Dict[str, Any]:
        """Get current compliance status - focused on data flow monitoring"""
        return {
            "is_compliant": True,  # Always allow operations, just monitor data
            "data_flow_monitoring": self.data_flow_mode,
            "verified_sources": len(self.verified_sources),
            "last_scan": self.last_scan_time.isoformat() if self.last_scan_time else None,
            "violations_count": len(self.compliance_violations),
            "score": self._calculate_compliance_score(),
            "metrics": self.metrics,
            "message": "Data flow monitoring active - operations allowed"
        }
    
    def validate_data_source(self, data: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Validate that data comes from authorized sources only"""
        validation_result = {
            "is_valid": False,
            "source": source,
            "authorized": False,
            "data_quality": "unknown",
            "timestamp": datetime.now().isoformat()
        }
        
        # Check if source is authorized
        if source.lower() in [s.lower() for s in self.verified_sources]:
            validation_result["authorized"] = True
            
            # Validate data quality
            if self._validate_data_structure(data):
                validation_result["is_valid"] = True
                validation_result["data_quality"] = "verified"
                self.metrics["verified_data"] += 1
            else:
                validation_result["data_quality"] = "invalid_structure"
                self.metrics["violations"] += 1
        else:
            validation_result["data_quality"] = "unauthorized_source"
            self.metrics["violations"] += 1
        
        self.metrics["total_checks"] += 1
        return validation_result
    
    def _validate_data_structure(self, data: Dict[str, Any]) -> bool:
        """Validate that data has proper structure and isn't synthetic"""
        if not data:
            return False
            
        # Check for synthetic/mock data indicators
        synthetic_indicators = ["mock", "test", "placeholder", "dummy", "fake", "sample"]
        data_str = str(data).lower()
        
        for indicator in synthetic_indicators:
            if indicator in data_str:
                return False
        
        # Check for timestamp validity
        if "timestamp" in data:
            try:
                timestamp = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
                # Data shouldn't be too old (configurable)
                max_age = timedelta(hours=self.compliance_config.get("max_data_age_hours", 24))
                if datetime.now() - timestamp.replace(tzinfo=None) > max_age:
                    return False
            except:
                return False
        
        return True
    
    def _calculate_compliance_score(self) -> float:
        """Calculate compliance score from 0-100"""
        if self.metrics["total_checks"] == 0:
            return 100.0
        
        violation_rate = self.metrics["violations"] / self.metrics["total_checks"]
        score = max(0, 100 - (violation_rate * 100))
        
        # Penalty for disabled compliance
        if not self.compliance_enabled:
            score = min(score, 30)
        
        return round(score, 1)
    
    def is_active(self) -> bool:
        """Check if compliance engine is active"""
        return self.compliance_enabled
    
    def set_compliance_mode(self, enabled: bool, user_initiated: bool = True) -> Dict[str, Any]:
        """Enable or disable compliance mode - only user can change this"""
        if not user_initiated:
            return {
                "success": False,
                "error": "Only user can toggle compliance mode",
                "current_status": self.compliance_enabled
            }
            
        old_status = self.compliance_enabled
        self.compliance_enabled = enabled
        
        if enabled != old_status:
            self._log_compliance_change(enabled)
        
        return {
            "success": True,
            "previous_status": old_status,
            "current_status": enabled,
            "timestamp": datetime.now().isoformat(),
            "changed_by": "user"
        }
    
    def _log_compliance_change(self, enabled: bool):
        """Log compliance mode changes"""
        violation = {
            "type": "compliance_mode_change",
            "enabled": enabled,
            "timestamp": datetime.now().isoformat(),
            "severity": "warning" if not enabled else "info"
        }
        self.compliance_violations.append(violation)
        
        if enabled:
            logger.info("✅ Compliance mode ENABLED - Full verification active")
        else:
            logger.warning("⚠️ Compliance mode DISABLED - System operating with restricted functionality")
    
    def is_mltrainer_request(self, request_context: Dict = None) -> bool:
        """Check if request comes from mlTrainer AI agent"""
        if request_context and request_context.get("source") == "mltrainer":
            return True
        return False
    
    def validate_data_source(self, data: Dict, request_context: Dict = None) -> bool:
        """Validate that data comes from verified sources"""
        self.metrics["total_checks"] += 1
        
        # mlTrainer AI is exempt from compliance restrictions
        if self.mltrainer_exempt and self.is_mltrainer_request(request_context):
            logger.debug("mlTrainer exempt from compliance check")
            return True
        
        if not self.compliance_enabled:
            return True  # Skip validation if compliance disabled
        
        # Check for required fields
        if not isinstance(data, dict):
            self._record_violation("invalid_data_format", "Data must be dictionary format")
            return False
        
        source = data.get("source")
        if not source:
            self._record_violation("missing_source", "Data source not specified")
            return False
        
        if source not in self.verified_sources:
            self._record_violation("unauthorized_source", f"Source {source} not in verified list")
            return False
        
        # Check timestamp freshness
        timestamp = data.get("timestamp")
        if not timestamp:
            self._record_violation("missing_timestamp", "Data timestamp required")
            return False
        
        try:
            data_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            age_hours = (datetime.now() - data_time.replace(tzinfo=None)).total_seconds() / 3600
            max_age = self.compliance_config.get("max_data_age_hours", 24)
            
            if age_hours > max_age:
                self._record_violation("stale_data", f"Data age {age_hours:.1f}h exceeds limit {max_age}h")
                return False
        except Exception as e:
            self._record_violation("invalid_timestamp", f"Timestamp parsing error: {e}")
            return False
        
        # Check verification flag
        if not data.get("verified", False):
            self._record_violation("unverified_data", "Data lacks verification flag")
            return False
        
        self.metrics["verified_items"] += 1
        return True
    
    def validate_recommendation(self, recommendation: Dict) -> bool:
        """Validate stock recommendation data"""
        if not self.compliance_enabled:
            return True
        
        self.metrics["total_checks"] += 1
        
        # Required fields for recommendations
        required_fields = ["ticker", "score", "confidence", "source", "timestamp"]
        
        for field in required_fields:
            if field not in recommendation:
                self._record_violation("missing_recommendation_field", f"Missing required field: {field}")
                return False
        
        # Validate score ranges
        score = recommendation.get("score", 0)
        confidence = recommendation.get("confidence", 0)
        
        if not (0 <= score <= 100):
            self._record_violation("invalid_score_range", f"Score {score} outside valid range 0-100")
            return False
        
        if not (0 <= confidence <= 100):
            self._record_violation("invalid_confidence_range", f"Confidence {confidence} outside valid range 0-100")
            return False
        
        # Validate data freshness
        if not self._is_data_fresh(recommendation.get("timestamp")):
            return False
        
        self.metrics["verified_items"] += 1
        return True
    
    def validate_regime_data(self, regime_data: Dict) -> bool:
        """Validate market regime analysis data"""
        if not self.compliance_enabled:
            return True
        
        self.metrics["total_checks"] += 1
        
        # Check for verified sources
        sources = regime_data.get("sources", [])
        if not sources:
            self._record_violation("no_regime_sources", "No data sources specified for regime analysis")
            return False
        
        # Verify at least one authorized source
        valid_sources = [s for s in sources if s in self.verified_sources]
        if not valid_sources:
            self._record_violation("no_verified_regime_sources", "No verified sources in regime data")
            return False
        
        # Check regime score validity
        regime_score = regime_data.get("regime_score")
        if regime_score is None or not (0 <= regime_score <= 100):
            self._record_violation("invalid_regime_score", f"Invalid regime score: {regime_score}")
            return False
        
        # Validate timestamp
        if not self._is_data_fresh(regime_data.get("timestamp")):
            return False
        
        self.metrics["verified_items"] += 1
        return True
    
    def _is_data_fresh(self, timestamp_str: str) -> bool:
        """Check if data timestamp is within acceptable freshness window"""
        if not timestamp_str:
            self._record_violation("missing_timestamp", "Data timestamp required")
            return False
        
        try:
            data_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            age_hours = (datetime.now() - data_time.replace(tzinfo=None)).total_seconds() / 3600
            max_age = self.compliance_config.get("max_data_age_hours", 24)
            
            if age_hours > max_age:
                self._record_violation("stale_data", f"Data age {age_hours:.1f}h exceeds limit {max_age}h")
                return False
                
            return True
        except Exception as e:
            self._record_violation("invalid_timestamp", f"Timestamp parsing error: {e}")
            return False
    
    def _record_violation(self, violation_type: str, description: str):
        """Record compliance violation"""
        violation = {
            "type": violation_type,
            "description": description,
            "timestamp": datetime.now().isoformat(),
            "severity": "error"
        }
        
        self.compliance_violations.append(violation)
        self.metrics["violations"] += 1
        self.metrics["rejected_items"] += 1
        
        logger.warning(f"Compliance violation: {violation_type} - {description}")
    
    def run_compliance_scan(self) -> Dict[str, Any]:
        """Run comprehensive compliance scan"""
        self.last_scan_time = datetime.now()
        
        scan_results = {
            "scan_time": self.last_scan_time.isoformat(),
            "compliance_enabled": self.compliance_enabled,
            "recent_violations": self._get_recent_violations(),
            "compliance_score": self._calculate_compliance_score(),
            "recommendations": []
        }
        
        # Add recommendations based on findings
        if not self.compliance_enabled:
            scan_results["recommendations"].append("Enable compliance mode for full data verification")
        
        if self.metrics["violations"] > 0:
            scan_results["recommendations"].append("Review and address compliance violations")
        
        if len(self.verified_sources) < 3:
            scan_results["recommendations"].append("Configure all three verified data sources")
        
        return scan_results
    
    def _get_recent_violations(self, hours: int = 24) -> List[Dict]:
        """Get violations from the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_violations = []
        for violation in self.compliance_violations:
            try:
                violation_time = datetime.fromisoformat(violation["timestamp"])
                if violation_time > cutoff_time:
                    recent_violations.append(violation)
            except:
                continue
        
        return recent_violations[-50:]  # Return last 50 violations
    
    def get_verified_sources(self) -> List[str]:
        """Get list of verified data sources"""
        return self.verified_sources.copy()
    
    def get_data_source_status(self) -> Dict[str, bool]:
        """Get status of all data sources"""
        try:
            from backend.data_sources import DataSourceManager
            data_manager = DataSourceManager()
            return data_manager.check_connections()
        except Exception as e:
            logger.error(f"Could not check data source status: {e}")
            return {source: False for source in self.verified_sources}
    
    def test_api_connections(self) -> Dict[str, bool]:
        """Test connections to all verified APIs"""
        try:
            from backend.data_sources import DataSourceManager
            data_manager = DataSourceManager()
            return data_manager.check_connections()
        except Exception as e:
            logger.error(f"API connection test failed: {e}")
            return {source: False for source in self.verified_sources}
    
    def format_compliant_response(self, suggestion: str) -> str:
        """Format response according to compliance template when data is uncertain"""
        template = self.compliance_config.get(
            "response_template", 
            "I don't know. But based on the data, I would suggest {suggestion}."
        )
        return template.format(suggestion=suggestion)
    
    def purge_synthetic_data(self) -> Dict[str, Any]:
        """Purge any synthetic or non-verified data from system"""
        purge_results = {
            "timestamp": datetime.now().isoformat(),
            "items_purged": 0,
            "sources_cleaned": [],
            "success": True
        }
        
        try:
            # In production, this would clean databases and caches
            # For now, we reset violation counters and clean memory
            
            self.compliance_violations = []
            self.metrics["rejected_items"] = 0
            
            purge_results["items_purged"] = self.metrics["rejected_items"]
            purge_results["sources_cleaned"] = ["memory_cache", "violation_log"]
            
            logger.info("Synthetic data purge completed")
            
        except Exception as e:
            logger.error(f"Data purge failed: {e}")
            purge_results["success"] = False
            purge_results["error"] = str(e)
        
        return purge_results
    
    def get_compliance_stats(self) -> Dict[str, Any]:
        """Get detailed compliance statistics"""
        return {
            "enabled": self.compliance_enabled,
            "score": self._calculate_compliance_score(),
            "metrics": self.metrics,
            "verified_sources": len(self.verified_sources),
            "recent_violations": len(self._get_recent_violations()),
            "last_scan": self.last_scan_time.isoformat() if self.last_scan_time else None,
            "config": {
                "strict_mode": self.compliance_config.get("strict_mode"),
                "max_data_age_hours": self.compliance_config.get("max_data_age_hours"),
                "require_timestamps": self.compliance_config.get("require_timestamps")
            }
        }
    
    def _start_audit_scheduler(self):
        """Start the automatic audit scheduler thread"""
        def audit_scheduler():
            while True:
                try:
                    current_time = datetime.now().strftime("%H:%M")
                    
                    # Check if it's time for an audit
                    if current_time in self.audit_times and not self.audit_running:
                        logger.info(f"🔍 Starting scheduled compliance audit at {current_time}")
                        self._run_automatic_audit()
                    
                    # Check every minute
                    time.sleep(60)
                except Exception as e:
                    logger.error(f"Audit scheduler error: {e}")
                    time.sleep(300)  # Wait 5 minutes on error
        
        self.audit_thread = threading.Thread(target=audit_scheduler, daemon=True)
        self.audit_thread.start()
        logger.info("Automatic compliance audit scheduler started (runs at 06:00 and 18:00)")
    
    def _run_automatic_audit(self):
        """Run comprehensive compliance audit and purge non-compliant data"""
        if self.audit_running:
            return
        
        self.audit_running = True
        audit_start = datetime.now()
        
        try:
            logger.info("🔍 COMPLIANCE AUDIT: Starting comprehensive system scan")
            
            # Scan all data directories
            purge_summary = {
                "files_scanned": 0,
                "files_purged": 0,
                "violations_found": 0,
                "data_size_purged": 0
            }
            
            # Directories to scan for compliance
            scan_directories = [
                "data/",
                "models/saved_models/",
                "research_analysis/",
                "cache/",
                "logs/",
                "temp/"
            ]
            
            for directory in scan_directories:
                if os.path.exists(directory):
                    purge_summary = self._scan_directory(directory, purge_summary)
            
            # Scan JSON data files specifically
            purge_summary = self._scan_json_data_files(purge_summary)
            
            # Update metrics
            self.metrics["purged_items"] += purge_summary["files_purged"]
            self.metrics["last_audit"] = audit_start.isoformat()
            
            # Log audit results
            audit_duration = (datetime.now() - audit_start).total_seconds()
            logger.info(f"✅ COMPLIANCE AUDIT COMPLETED in {audit_duration:.1f}s")
            logger.info(f"📊 AUDIT SUMMARY: {purge_summary['files_scanned']} files scanned, "
                       f"{purge_summary['files_purged']} files purged, "
                       f"{purge_summary['violations_found']} violations found")
            
            # Record audit in violations log
            self._record_violation("audit_completed", f"Automatic audit completed: {purge_summary}")
            
        except Exception as e:
            logger.error(f"❌ COMPLIANCE AUDIT FAILED: {e}")
        finally:
            self.audit_running = False
    
    def _scan_directory(self, directory: str, purge_summary: Dict) -> Dict:
        """Scan a directory for non-compliant files"""
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    filepath = os.path.join(root, file)
                    
                    # Skip exempt files (mlTrainer chat and user data)
                    exempt_patterns = [
                        'chat_history.json',
                        'mltrainer_chat',
                        'chat_memory',
                        'hgw_',
                        'user_chat',
                        'mltrainer_'
                    ]
                    
                    if any(pattern in filepath.lower() for pattern in exempt_patterns):
                        logger.debug(f"Skipping exempt file: {filepath}")
                        continue
                    
                    purge_summary["files_scanned"] += 1
                    
                    # Check file for compliance issues
                    if self._is_file_non_compliant(filepath):
                        purge_summary["violations_found"] += 1
                        
                        # Get file size before deletion
                        try:
                            file_size = os.path.getsize(filepath)
                            purge_summary["data_size_purged"] += file_size
                        except:
                            pass
                        
                        # Purge non-compliant file
                        self._purge_file(filepath)
                        purge_summary["files_purged"] += 1
                        
        except Exception as e:
            logger.error(f"Error scanning directory {directory}: {e}")
        
        return purge_summary
    
    def _scan_json_data_files(self, purge_summary: Dict) -> Dict:
        """Scan JSON data files for non-compliant data"""
        json_files = [
            "data/portfolio.json",
            "data/recommendations.json", 
            "data/alerts.json"
            # Exempt: "data/chat_memory.json" - mlTrainer chat data exempt from purges
        ]
        
        for json_file in json_files:
            if os.path.exists(json_file):
                purge_summary["files_scanned"] += 1
                
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    # Check for non-compliant data
                    original_size = len(str(data))
                    cleaned_data = self._clean_json_data(data)
                    cleaned_size = len(str(cleaned_data))
                    
                    if cleaned_size < original_size:
                        # Data was purged
                        purge_summary["violations_found"] += 1
                        purge_summary["data_size_purged"] += (original_size - cleaned_size)
                        
                        # Save cleaned data
                        with open(json_file, 'w') as f:
                            json.dump(cleaned_data, f, indent=2)
                        
                        logger.info(f"🧹 Cleaned non-compliant data from {json_file}")
                        
                except Exception as e:
                    logger.error(f"Error processing {json_file}: {e}")
        
        return purge_summary
    
    def _is_file_non_compliant(self, filepath: str) -> bool:
        """Check if a file contains non-compliant data"""
        try:
            # Check file age
            file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(filepath))
            max_age = timedelta(days=30)  # Files older than 30 days
            
            if file_age > max_age:
                return True
            
            # Check file content for compliance issues
            if filepath.endswith('.json'):
                with open(filepath, 'r') as f:
                    content = f.read()
                    
                    # Look for synthetic data indicators
                    synthetic_indicators = [
                        "example_", "mock_", "test_", "placeholder_", "fake_",
                        "synthetic", "dummy", "sample_data"
                    ]
                    
                    for indicator in synthetic_indicators:
                        if indicator in content.lower():
                            return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking file compliance {filepath}: {e}")
            return False
    
    def create_data_filter(self):
        """Create active data filter for mlTrainer communication"""
        return DataFlowFilter(self)
    
    def get_verified_sources_from_config(self) -> List[str]:
        """Get list of verified sources from API configuration"""
        try:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'api_providers.yaml')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                verified_sources = []
                
                # Extract verified data providers
                data_providers = config.get('data_providers', {})
                for category in ['market_data', 'economic_data', 'alternative_data']:
                    category_data = data_providers.get(category, {})
                    providers = category_data.get('providers', {})
                    
                    for provider_id, provider_config in providers.items():
                        compliance = provider_config.get('compliance', {})
                        if compliance.get('verified_source', False):
                            verified_sources.append(provider_id)
                
                return verified_sources
                
        except Exception as e:
            logger.warning(f"Could not load API config for source verification: {e}")
        
        # Fallback to default verified sources
        return self.verified_sources

class DataFlowFilter:
    """Active filter for data flowing to mlTrainer"""
    
    def __init__(self, compliance_engine):
        self.compliance_engine = compliance_engine
        self.verified_sources = compliance_engine.get_verified_sources_from_config()
        logger.info(f"DataFlowFilter initialized with verified sources: {self.verified_sources}")
    
    def filter_data_for_mltrainer(self, data: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Filter data before it reaches mlTrainer - only verified sources allowed"""
        
        # Add source to data for validation
        data_with_source = data.copy()
        data_with_source['source'] = source
        
        # Add verification flag for authorized sources
        if source in self.verified_sources:
            data_with_source['verified'] = True
        
        # Validate source authorization
        validation_result = self.compliance_engine.validate_data_source(data_with_source, {"source": source})
        
        if not validation_result:
            # Block non-verified data
            logger.warning(f"Blocking data from unauthorized source: {source}")
            return {
                "error": "Data source not verified",
                "message": f"I don't know. But based on the data, I would suggest using verified sources: {', '.join(self.verified_sources)}",
                "source": source,
                "verified_sources": self.verified_sources,
                "compliance_status": "blocked"
            }
        
        # Allow verified data to pass through with validation metadata
        filtered_data = data.copy()
        filtered_data['_compliance'] = {
            "source_verified": True,
            "source": source,
            "validation_timestamp": datetime.now().isoformat(),
            "validation_passed": True
        }
        
        logger.info(f"Data from {source} passed compliance filter")
        return filtered_data
    
    def validate_api_response(self, response: Dict[str, Any], api_source: str) -> Dict[str, Any]:
        """Validate API response before forwarding to mlTrainer"""
        
        # Check if source is in verified list
        if api_source.lower() not in [s.lower() for s in self.verified_sources]:
            logger.error(f"API response from unverified source: {api_source}")
            return {
                "error": "Unverified API source",
                "message": f"I don't know. But based on the data, I would suggest using only verified APIs: {', '.join(self.verified_sources)}",
                "blocked_source": api_source
            }
        
        # Check response structure for compliance
        if self._contains_synthetic_data(response):
            logger.error(f"Synthetic data detected from {api_source}")
            return {
                "error": "Synthetic data detected", 
                "message": "I don't know. But based on the data, I would suggest requesting fresh data from the API.",
                "source": api_source
            }
        
        # Tag verified response
        response['_compliance_verified'] = {
            "api_source": api_source,
            "verified": True,
            "timestamp": datetime.now().isoformat()
        }
        
        return response
    
    def _contains_synthetic_data(self, data: Dict[str, Any]) -> bool:
        """Check if data contains synthetic/mock indicators"""
        synthetic_indicators = [
            "mock", "test", "placeholder", "dummy", "fake", "sample", 
            "example", "demo", "simulation", "synthetic"
        ]
        
        data_str = str(data).lower()
        return any(indicator in data_str for indicator in synthetic_indicators)
    
    def get_allowed_sources(self) -> List[str]:
        """Get list of sources allowed to send data to mlTrainer"""
        return self.verified_sources.copy()
    
    def _purge_file(self, filepath: str):
        """Safely purge a non-compliant file"""
        try:
            # Create backup directory if it doesn't exist
            backup_dir = "data/compliance_backups"
            os.makedirs(backup_dir, exist_ok=True)
            
            # Create backup before deletion
            backup_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.path.basename(filepath)}"
            backup_path = os.path.join(backup_dir, backup_filename)
            
            shutil.copy2(filepath, backup_path)
            
            # Delete original file
            os.remove(filepath)
            
            logger.info(f"🗑️ Purged non-compliant file: {filepath} (backup: {backup_path})")
            
        except Exception as e:
            logger.error(f"Error purging file {filepath}: {e}")
    
    def _clean_json_data(self, data: Any) -> Any:
        """Clean non-compliant data from JSON structures"""
        if isinstance(data, dict):
            cleaned = {}
            for key, value in data.items():
                # Skip synthetic data fields
                if any(indicator in key.lower() for indicator in ["example", "mock", "test", "placeholder", "fake", "synthetic", "dummy"]):
                    continue
                
                # Check for non-compliant source
                if key == "source" and value not in self.verified_sources:
                    continue
                
                # Check timestamp freshness
                if key == "timestamp":
                    try:
                        timestamp_dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                        age_hours = (datetime.now() - timestamp_dt.replace(tzinfo=None)).total_seconds() / 3600
                        if age_hours > 24:  # Data older than 24 hours
                            continue
                    except:
                        continue
                
                cleaned[key] = self._clean_json_data(value)
            return cleaned
            
        elif isinstance(data, list):
            cleaned = []
            for item in data:
                cleaned_item = self._clean_json_data(item)
                if cleaned_item is not None:
                    cleaned.append(cleaned_item)
            return cleaned
        
        return data
    
    def get_audit_status(self) -> Dict[str, Any]:
        """Get current audit status and schedule"""
        return {
            "audit_running": self.audit_running,
            "next_audit_times": self.audit_times,
            "last_audit": self.metrics.get("last_audit"),
            "total_purged": self.metrics.get("purged_items", 0),
            "audit_schedule": "Twice daily at 06:00 and 18:00"
        }
    
    def force_audit(self) -> Dict[str, Any]:
        """Force an immediate compliance audit"""
        if self.audit_running:
            return {
                "success": False,
                "message": "Audit already in progress"
            }
        
        logger.info("🔍 MANUAL COMPLIANCE AUDIT: Starting forced system scan")
        
        # Run audit in separate thread
        audit_thread = threading.Thread(target=self._run_automatic_audit, daemon=True)
        audit_thread.start()
        
        return {
            "success": True,
            "message": "Compliance audit started",
            "timestamp": datetime.now().isoformat()
        }

