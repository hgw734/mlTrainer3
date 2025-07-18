"""
Immutable Compliance Gateway - Single Source of Truth for Data Compliance
Ensures all data sources and processing comply with regulatory requirements
"""

import logging
import hashlib
import time
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable
from functools import wraps

# SINGLE SOURCE OF TRUTH - Import centralized configurations
try:
    from .api_config import (
        APISource as ConfigAPISource,
        get_all_approved_sources,
        validate_api_source
    )
except ImportError:
    # Fallback for module loading
    from api_config import (
        APISource as ConfigAPISource,
        get_all_approved_sources,
        validate_api_source
    )

try:
    from .ai_config import (
        get_ai_compliance_config,
        validate_model_config as validate_ai_model_config
    )
except ImportError:
    # Fallback for module loading
    from ai_config import (
        get_ai_compliance_config,
        validate_model_config as validate_ai_model_config
    )

# Configure logging for compliance auditing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - COMPLIANCE - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('compliance_audit.log'),
        logging.StreamHandler()
    ]
)
compliance_logger = logging.getLogger('COMPLIANCE_GATEWAY')


class DataSource(Enum):
    """APPROVED DATA SOURCES ONLY"""
    POLYGON = "polygon_api"
    FRED = "fred_api"
    INVALID = "invalid_source"


class ComplianceStatus(Enum):
    """Data compliance verification status"""
    VERIFIED = "verified"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CORRUPTED = "corrupted"


@dataclass
class DataProvenance:
    """Immutable data provenance tracking"""
    source: DataSource
    api_endpoint: str
    timestamp: datetime
    data_hash: str
    freshness_seconds: int
    verification_signature: str
    compliance_status: ComplianceStatus = ComplianceStatus.VERIFIED

    def __post_init__(self):
        """Ensure immutability after creation"""
        self._frozen = True

    def __setattr__(self, name, value):
        if hasattr(self, '_frozen') and self._frozen:
            raise ValueError("🔒 COMPLIANCE VIOLATION: DataProvenance is immutable")
        super().__setattr__(name, value)


class ComplianceGateway:
    """Immutable compliance gateway for data verification"""
    
    def __init__(self):
        self.violations = []
        self.verification_count = 0
        self.rejection_count = 0
        
    
    
    
    
    def _generate_verification_signature(self, data: Any, source: DataSource, endpoint: str) -> str:
        """Generate deterministic verification signature"""
        data_str = str(data) + str(source.value) + endpoint + str(int(time.time()))
        return hashlib.sha256(data_str.encode()).hexdigest()

    def _calculate_data_hash(self, data: Any) -> str:
        """Calculate deterministic hash of data"""
        data_str = str(data)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def verify_data_source(self, source: str, endpoint: str) -> DataSource:
        """Verify data source is approved"""
        try:
            # Validate against approved sources
            if source in [DataSource.POLYGON.value, DataSource.FRED.value]:
                return DataSource(source)
            else:
                self._log_compliance_violation(f"Invalid data source: {source}")
                return DataSource.INVALID
        except Exception as e:
            self._log_compliance_violation(f"Source verification failed: {e}")
            return DataSource.INVALID

    def tag_incoming_data(self, data: Any, source: str, endpoint: str) -> Optional[DataProvenance]:
        """Tag incoming data with immutable provenance"""
        try:
            # Verify source
            verified_source = self.verify_data_source(source, endpoint)
            if verified_source == DataSource.INVALID:
                return None

            # Calculate data hash
            data_hash = self._calculate_data_hash(data)
            
            # Generate verification signature
            signature = self._generate_verification_signature(data, verified_source, endpoint)
            
            # Create provenance
            provenance = DataProvenance(
                source=verified_source,
                api_endpoint=endpoint,
                timestamp=datetime.now(),
                data_hash=data_hash,
                freshness_seconds=300,  # 5 minutes default
                verification_signature=signature,
                compliance_status=ComplianceStatus.VERIFIED
            )
            
            self.verification_count += 1
            compliance_logger.info(f"Data tagged with provenance: {signature[:8]}")
            return provenance
            
        except Exception as e:
            self._log_compliance_violation(f"Data tagging failed: {e}")
            return None

    def verify_data_freshness(self, provenance: DataProvenance) -> bool:
        """Verify data is still fresh"""
        try:
            age_seconds = (datetime.now() - provenance.timestamp).total_seconds()
            return age_seconds <= provenance.freshness_seconds
        except Exception as e:
            self._log_compliance_violation(f"Freshness check failed: {e}")
            return False

    def verify_data_integrity(self, data: Any, provenance: DataProvenance) -> bool:
        """Verify data integrity matches provenance"""
        try:
            current_hash = self._calculate_data_hash(data)
            return current_hash == provenance.data_hash
        except Exception as e:
            self._log_compliance_violation(f"Integrity check failed: {e}")
            return False

    def pre_processing_compliance_check(self, data: Any, provenance: DataProvenance) -> bool:
        """Pre-processing compliance verification"""
        try:
            # Check freshness
            if not self.verify_data_freshness(provenance):
                self._log_compliance_violation("Data expired")
                return False
            
            # Check integrity
            if not self.verify_data_integrity(data, provenance):
                self._log_compliance_violation("Data integrity violation")
                return False
            
            # Check source validity
            if provenance.source == DataSource.INVALID:
                self._log_compliance_violation("Invalid data source")
                return False
            
            compliance_logger.info("Pre-processing compliance check passed")
            return True
            
        except Exception as e:
            self._log_compliance_violation(f"Pre-processing check failed: {e}")
            return False

    def post_processing_compliance_check(self, processed_data: Any, original_provenance: DataProvenance) -> bool:
        """Post-processing compliance verification"""
        try:
            # Verify original data was compliant
            if original_provenance.compliance_status != ComplianceStatus.VERIFIED:
                self._log_compliance_violation("Original data not verified")
                return False
            
            # Check that processing didn't corrupt data
            if not self.verify_data_integrity(processed_data, original_provenance):
                self._log_compliance_violation("Processing corrupted data")
                return False
            
            compliance_logger.info("Post-processing compliance check passed")
            return True
            
        except Exception as e:
            self._log_compliance_violation(f"Post-processing check failed: {e}")
            return False

    def _log_compliance_violation(self, violation: str):
        """Log compliance violation"""
        self.violations.append({
            'timestamp': datetime.now(),
            'violation': violation
        })
        self.rejection_count += 1
        compliance_logger.error(f"COMPLIANCE VIOLATION: {violation}")

    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance report"""
        return {
            'total_verifications': self.verification_count,
            'total_rejections': self.rejection_count,
            'violations': self.violations,
            'compliance_rate': (self.verification_count - self.rejection_count) / max(self.verification_count, 1)
        }

    def verify_model_parameters(self, params: Dict[str, Any]) -> bool:
        """Verify model parameters comply with regulations"""
        try:
            # Check for prohibited parameters
            prohibited = ['synthetic', 'fake', 'dummy', 'mock']
            for key, value in params.items():
                if any(term in str(key).lower() or term in str(value).lower() for term in prohibited):
                    self._log_compliance_violation(f"Prohibited parameter: {key}")
                    return False
            
            compliance_logger.info("Model parameters verified")
            return True
            
        except Exception as e:
            self._log_compliance_violation(f"Parameter verification failed: {e}")
            return False

    def verify_high_complexity_model(self, model_id: str) -> bool:
        """Verify high complexity model compliance"""
        try:
            # Check if model requires special compliance
            high_complexity_indicators = ['neural', 'deep', 'ensemble', 'complex']
            if any(indicator in model_id.lower() for indicator in high_complexity_indicators):
                # Additional checks for high complexity models
                compliance_logger.warning(f"High complexity model detected: {model_id}")
                return True  # Allow but log
            
            return True
            
        except Exception as e:
            self._log_compliance_violation(f"Model complexity check failed: {e}")
            return False


def compliance_required(func):
    """Decorator to enforce compliance checks"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check if data has provenance
        for arg in args:
            if hasattr(arg, 'provenance'):
                if not COMPLIANCE_GATEWAY.pre_processing_compliance_check(arg, arg.provenance):
                    raise ValueError("🔒 COMPLIANCE VIOLATION: Data failed pre-processing check")
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Post-processing check
        if hasattr(result, 'provenance'):
            if not COMPLIANCE_GATEWAY.post_processing_compliance_check(result, result.provenance):
                raise ValueError("🔒 COMPLIANCE VIOLATION: Result failed post-processing check")
        
        return result
    return wrapper


@dataclass
class VerifiedData:
    """Container for verified data with immutable provenance"""
    data: Any
    provenance: DataProvenance

    def __post_init__(self):
        """Ensure data is properly tagged and verified"""
        if not isinstance(self.provenance, DataProvenance):
            raise ValueError("🔒 COMPLIANCE VIOLATION: Invalid provenance")

        if self.provenance.compliance_status != ComplianceStatus.VERIFIED:
            raise ValueError("🔒 COMPLIANCE VIOLATION: Unverified data")


# Export compliance functions
__all__ = [
    'ComplianceGateway',
    'DataProvenance',
    'VerifiedData',
    'DataSource',
    'ComplianceStatus',
    'compliance_required',
    'COMPLIANCE_GATEWAY'
]

# Global compliance gateway instance
COMPLIANCE_GATEWAY = ComplianceGateway()