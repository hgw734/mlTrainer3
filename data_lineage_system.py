#!/usr/bin/env python3
"""
🔗 Data Lineage System - Complete Data Provenance Tracking
Tracks ALL data from source through every calculation with full computational lineage
"""

import pandas as pd
import numpy as np
import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Valid data sources that can be used in calculations"""
    POLYGON_API = "polygon_api"
    FRED_API = "fred_api"
    WIKIPEDIA = "wikipedia"
    INTERNAL_CALCULATION = "internal_calculation"
    USER_INPUT = "user_input"
    MODEL_OUTPUT = "model_output"


class LineageStatus(Enum):
    """Status of data lineage verification"""
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    BLOCKED = "blocked"
    EXPIRED = "expired"


@dataclass
class DataLineage:
    """Complete data lineage information"""
    source: DataSource
    timestamp: datetime
    original_data_hash: str
    calculation_path: List[str] = field(default_factory=list)
    input_lineages: List['DataLineage'] = field(default_factory=list)
    freshness_threshold: timedelta = field(
        default_factory=lambda: timedelta(hours=24))
    verification_status: LineageStatus = LineageStatus.UNVERIFIED
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.calculation_path:
            self.calculation_path = [f"source_{self.source.value}"]

    def add_calculation_step(
            self,
            operation: str,
            inputs: List['DataLineage'] = None):
        """Add a calculation step to the lineage"""
        self.calculation_path.append(operation)
        if inputs:
            self.input_lineages.extend(inputs)

    def is_fresh(self) -> bool:
        """Check if data is within freshness threshold"""
        return datetime.now() - self.timestamp < self.freshness_threshold

    def get_full_lineage(self) -> str:
        """Get complete computational lineage as string"""
        lineage_parts = []
        for step in self.calculation_path:
            lineage_parts.append(step)
        return " -> ".join(lineage_parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert lineage to dictionary for serialization"""
        return {
            'source': self.source.value,
            'timestamp': self.timestamp.isoformat(),
            'original_data_hash': self.original_data_hash,
            'calculation_path': self.calculation_path,
            'input_lineages': [
                lineage.to_dict() for lineage in self.input_lineages],
            'freshness_threshold_hours': self.freshness_threshold.total_seconds() / 3600,
            'verification_status': self.verification_status.value,
            'metadata': self.metadata}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataLineage':
        """Create lineage from dictionary"""
        return cls(
            source=DataSource(
                data['source']),
            timestamp=datetime.fromisoformat(
                data['timestamp']),
            original_data_hash=data['original_data_hash'],
            calculation_path=data['calculation_path'],
            input_lineages=[
                cls.from_dict(lineage) for lineage in data['input_lineages']],
            freshness_threshold=timedelta(
                hours=data['freshness_threshold_hours']),
            verification_status=LineageStatus(
                data['verification_status']),
            metadata=data['metadata'])


class TrackedArray(np.ndarray):
    """NumPy array with embedded lineage tracking"""

    def __new__(cls, input_array, lineage: DataLineage, *args, **kwargs):
        obj = np.asarray(input_array).view(cls)
        obj.lineage = lineage
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.lineage = getattr(obj, 'lineage', None)

    def __array_wrap__(self, out_arr, context=None):
        """Preserve lineage through array operations"""
        if context is not None:
            operation = context[0].__name__
            inputs = [self.lineage] if hasattr(self, 'lineage') else []
            # Add input lineages from context if available
            for arg in context[1]:
                if hasattr(arg, 'lineage'):
                    inputs.append(arg.lineage)

            new_lineage = DataLineage(
                source=DataSource.INTERNAL_CALCULATION,
                timestamp=datetime.now(),
                original_data_hash=hashlib.sha256(
                    out_arr.tobytes()).hexdigest(),
                calculation_path=[operation],
                input_lineages=inputs)
            out_arr.lineage = new_lineage

        return out_arr


class TrackedDataFrame(pd.DataFrame):
    """Pandas DataFrame with embedded lineage tracking"""

    def __init__(
            self,
            data=None,
            lineage: DataLineage = None,
            *args,
            **kwargs):
        super().__init__(data, *args, **kwargs)
        self.lineage = lineage or DataLineage(
            source=DataSource.USER_INPUT,
            timestamp=datetime.now(),
            original_data_hash=hashlib.sha256(str(data).encode()).hexdigest()
        )

    def __getitem__(self, key):
        """Preserve lineage through DataFrame operations"""
        result = super().__getitem__(key)
        if isinstance(result, pd.DataFrame):
            result.lineage = self.lineage
        elif isinstance(result, pd.Series):
            result.lineage = self.lineage
        return result

    def __setitem__(self, key, value):
        """Update lineage when modifying DataFrame"""
        super().__setitem__(key, value)
        if hasattr(value, 'lineage'):
            # Merge lineages if value has its own lineage
            self.lineage.input_lineages.append(value.lineage)

    def _update_lineage(
            self,
            operation: str,
            inputs: List[DataLineage] = None):
        """Update lineage after operation"""
        new_lineage = DataLineage(
            source=DataSource.INTERNAL_CALCULATION,
            timestamp=datetime.now(),
            original_data_hash=hashlib.sha256(
                self.to_string().encode()).hexdigest(),
            calculation_path=[operation],
            input_lineages=inputs or [
                self.lineage])
        self.lineage = new_lineage


class TrackedSeries(pd.Series):
    """Pandas Series with embedded lineage tracking"""

    def __init__(
            self,
            data=None,
            lineage: DataLineage = None,
            *args,
            **kwargs):
        super().__init__(data, *args, **kwargs)
        self.lineage = lineage or DataLineage(
            source=DataSource.USER_INPUT,
            timestamp=datetime.now(),
            original_data_hash=hashlib.sha256(str(data).encode()).hexdigest()
        )

    def _update_lineage(
            self,
            operation: str,
            inputs: List[DataLineage] = None):
        """Update lineage after operation"""
        new_lineage = DataLineage(
            source=DataSource.INTERNAL_CALCULATION,
            timestamp=datetime.now(),
            original_data_hash=hashlib.sha256(
                self.to_string().encode()).hexdigest(),
            calculation_path=[operation],
            input_lineages=inputs or [
                self.lineage])
        self.lineage = new_lineage


class DataLineageSystem:
    """Complete data lineage tracking system"""

    def __init__(self, config_path: str = "config/"):
        self.config_path = Path(config_path)
        self.allowed_sources = self._load_allowed_sources()
        self.lineage_registry: Dict[str, DataLineage] = {}
        self.blocked_sources: Set[str] = set()
        self.freshness_thresholds: Dict[DataSource, timedelta] = {
            DataSource.POLYGON_API: timedelta(minutes=5),
            DataSource.FRED_API: timedelta(hours=1),
            DataSource.WIKIPEDIA: timedelta(days=1),
            DataSource.INTERNAL_CALCULATION: timedelta(hours=24),
            DataSource.USER_INPUT: timedelta(hours=24),
            DataSource.MODEL_OUTPUT: timedelta(hours=24)
        }

        # Load existing lineage data
        self._load_lineage_registry()

    def _load_allowed_sources(self) -> Set[str]:
        """Load allowed data sources from configuration"""
        allowed_sources = set()

        # Load from api_config.py if available
        api_config_path = self.config_path / "api_config.py"
        if api_config_path.exists():
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "api_config", api_config_path)
                api_config = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(api_config)

                # Extract API sources
                if hasattr(api_config, 'POLYGON_API_KEY'):
                    allowed_sources.add('polygon_api')
                if hasattr(api_config, 'FRED_API_KEY'):
                    allowed_sources.add('fred_api')
            except Exception as e:
                logger.warning(f"Could not load api_config.py: {e}")

        # Load from ai_config.py if available
        ai_config_path = self.config_path / "ai_config.py"
        if ai_config_path.exists():
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "ai_config", ai_config_path)
                ai_config = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(ai_config)

                # Extract AI sources
                if hasattr(ai_config, 'ALLOWED_DATA_SOURCES'):
                    allowed_sources.update(ai_config.ALLOWED_DATA_SOURCES)
            except Exception as e:
                logger.warning(f"Could not load ai_config.py: {e}")

        # Default sources if none found
        if not allowed_sources:
            allowed_sources = {
                'polygon_api',
                'fred_api',
                'wikipedia',
                'internal_calculation'}

        return allowed_sources

    def _load_lineage_registry(self):
        """Load existing lineage registry from file"""
        registry_path = Path("logs/data_lineage_registry.json")
        if registry_path.exists():
            try:
                with open(registry_path, 'r') as f:
                    data = json.load(f)
                    for key, lineage_data in data.items():
                        self.lineage_registry[key] = DataLineage.from_dict(
                            lineage_data)
                logger.info(
                    f"Loaded {len(self.lineage_registry)} lineage records")
            except Exception as e:
                logger.error(f"Error loading lineage registry: {e}")

    def _save_lineage_registry(self):
        """Save lineage registry to file"""
        registry_path = Path("logs/data_lineage_registry.json")
        registry_path.parent.mkdir(exist_ok=True)

        try:
            with open(registry_path, 'w') as f:
                data = {
                    key: lineage.to_dict() for key,
                    lineage in self.lineage_registry.items()}
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving lineage registry: {e}")

    def create_lineage(self, data: Any, source: DataSource,
                       metadata: Dict[str, Any] = None) -> DataLineage:
        """Create new data lineage"""
        # Verify source is allowed
        if source.value not in self.allowed_sources:
            raise ValueError(
                f"Data source '{source.value}' is not in allowed sources: {self.allowed_sources}")

        # Create data hash
        if isinstance(data, (pd.DataFrame, pd.Series)):
            data_hash = hashlib.sha256(data.to_string().encode()).hexdigest()
        elif isinstance(data, np.ndarray):
            data_hash = hashlib.sha256(data.tobytes()).hexdigest()
        else:
            data_hash = hashlib.sha256(str(data).encode()).hexdigest()

        # Create lineage
        lineage = DataLineage(
            source=source,
            timestamp=datetime.now(),
            original_data_hash=data_hash,
            metadata=metadata or {}
        )

        # Store in registry
        self.lineage_registry[data_hash] = lineage
        self._save_lineage_registry()

        return lineage

    def verify_lineage(self, lineage: DataLineage) -> bool:
        """Verify data lineage is valid and fresh"""
        # Check if source is allowed
        if lineage.source.value not in self.allowed_sources:
            logger.warning(
                f"Blocked data from unauthorized source: {lineage.source.value}")
            lineage.verification_status = LineageStatus.BLOCKED
            return False

        # Check if source is blocked
        if lineage.source.value in self.blocked_sources:
            logger.warning(
                f"Blocked data from blocked source: {lineage.source.value}")
            lineage.verification_status = LineageStatus.BLOCKED
            return False

        # Check freshness
        if not lineage.is_fresh():
            logger.warning(f"Data lineage expired: {lineage.source.value}")
            lineage.verification_status = LineageStatus.EXPIRED
            return False

        # Verify calculation path
        for step in lineage.calculation_path:
            if not self._verify_calculation_step(step):
                logger.warning(f"Invalid calculation step: {step}")
                lineage.verification_status = LineageStatus.BLOCKED
                return False

        lineage.verification_status = LineageStatus.VERIFIED
        return True

    def _verify_calculation_step(self, step: str) -> bool:
        """Verify a calculation step is valid"""
        # Define allowed operations
        allowed_operations = {
            'add',
            'subtract',
            'multiply',
            'divide',
            'mean',
            'std',
            'sum',
            'momentum',
            'volatility',
            'correlation',
            'regression',
            'clustering',
            'normalize',
            'standardize',
            'filter',
            'sort',
            'rank'}

        # Check if step contains allowed operation
        for operation in allowed_operations:
            if operation in step.lower():
                return True

        # Allow source operations
        if step.startswith('source_'):
            return True

        return False

    def create_tracked_array(self,
                             data: np.ndarray,
                             source: DataSource,
                             metadata: Dict[str,
                                            Any] = None) -> TrackedArray:
        """Create tracked NumPy array"""
        lineage = self.create_lineage(data, source, metadata)
        return TrackedArray(data, lineage)

    def create_tracked_dataframe(self,
                                 data: pd.DataFrame,
                                 source: DataSource,
                                 metadata: Dict[str,
                                                Any] = None) -> TrackedDataFrame:
        """Create tracked pandas DataFrame"""
        lineage = self.create_lineage(data, source, metadata)
        return TrackedDataFrame(data, lineage)

    def create_tracked_series(self,
                              data: pd.Series,
                              source: DataSource,
                              metadata: Dict[str,
                                             Any] = None) -> TrackedSeries:
        """Create tracked pandas Series"""
        lineage = self.create_lineage(data, source, metadata)
        return TrackedSeries(data, lineage)

    def block_source(self, source: str):
        """Block a data source"""
        self.blocked_sources.add(source)
        logger.warning(f"Blocked data source: {source}")

    def unblock_source(self, source: str):
        """Unblock a data source"""
        self.blocked_sources.discard(source)
        logger.info(f"Unblocked data source: {source}")

    def get_lineage_summary(self) -> Dict[str, Any]:
        """Get summary of all lineages"""
        summary = {
            'total_lineages': len(self.lineage_registry),
            'verified': 0,
            'unverified': 0,
            'blocked': 0,
            'expired': 0,
            'sources': {},
            'recent_operations': []
        }

        for lineage in self.lineage_registry.values():
            summary[lineage.verification_status.value] += 1

            # Count by source
            source = lineage.source.value
            if source not in summary['sources']:
                summary['sources'][source] = 0
            summary['sources'][source] += 1

        return summary

    def cleanup_expired_lineages(self):
        """Remove expired lineages from registry"""
        current_time = datetime.now()
        expired_keys = []

        for key, lineage in self.lineage_registry.items():
            if not lineage.is_fresh():
                expired_keys.append(key)

        for key in expired_keys:
            del self.lineage_registry[key]

        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired lineages")
            self._save_lineage_registry()


# Global lineage system instance
lineage_system = DataLineageSystem()


def track_data(data: Any, source: DataSource, metadata: Dict[str, Any] = None):
    """Convenience function to track data with lineage"""
    if isinstance(data, np.ndarray):
        return lineage_system.create_tracked_array(data, source, metadata)
    elif isinstance(data, pd.DataFrame):
        return lineage_system.create_tracked_dataframe(data, source, metadata)
    elif isinstance(data, pd.Series):
        return lineage_system.create_tracked_series(data, source, metadata)
    else:
        return lineage_system.create_lineage(data, source, metadata)


def verify_data_lineage(data: Any) -> bool:
    """Verify data lineage is valid"""
    if hasattr(data, 'lineage'):
        return lineage_system.verify_lineage(data.lineage)
    return False


def get_data_lineage(data: Any) -> Optional[DataLineage]:
    """Get lineage from tracked data"""
    if hasattr(data, 'lineage'):
        return data.lineage
    return None


def block_data_source(source: str):
    """Block a data source"""
    lineage_system.block_source(source)


def unblock_data_source(source: str):
    """Unblock a data source"""
    lineage_system.unblock_source(source)


def get_lineage_summary() -> Dict[str, Any]:
    """Get lineage system summary"""
    return lineage_system.get_lineage_summary()


if __name__ == "__main__":
    # Example usage
    print("🔗 Data Lineage System - Complete Data Provenance Tracking")

    # Create some test data
    test_data = np.array([1, 2, 3, 4, 5])
    tracked_data = track_data(test_data, DataSource.USER_INPUT, {'test': True})

    print(
        f"Created tracked data with lineage: {tracked_data.lineage.get_full_lineage()}")
    print(f"Verification status: {verify_data_lineage(tracked_data)}")

    # Get summary
    summary = get_lineage_summary()
    print(f"Lineage summary: {summary}")
