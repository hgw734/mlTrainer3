#!/usr/bin/env python3
"""
🛡️ Anti-Drift Protection System
Continuous monitoring of core files with SHA-256 fingerprinting and modification blocking
"""

import hashlib
import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import shutil
import stat

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FileFingerprint:
    """File fingerprint with metadata"""
    path: str
    hash: str
    size: int
    last_modified: datetime
    permissions: int
    is_protected: bool = False
    backup_path: Optional[str] = None

    def __post_init__(self):
        if self.backup_path is None and self.is_protected:
            self.backup_path = self._create_backup_path()

    def _create_backup_path(self) -> str:
        """Create backup path for protected file"""
        backup_dir = Path("logs/protected_backups")
        backup_dir.mkdir(exist_ok=True)

        file_path = Path(self.path)
        backup_name = f"{file_path.stem}_{int(time.time())}{file_path.suffix}"
        return str(backup_dir / backup_name)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'path': self.path,
            'hash': self.hash,
            'size': self.size,
            'last_modified': self.last_modified.isoformat(),
            'permissions': self.permissions,
            'is_protected': self.is_protected,
            'backup_path': self.backup_path
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileFingerprint':
        """Create from dictionary"""
        return cls(
            path=data['path'],
            hash=data['hash'],
            size=data['size'],
            last_modified=datetime.fromisoformat(data['last_modified']),
            permissions=data['permissions'],
            is_protected=data.get('is_protected', False),
            backup_path=data.get('backup_path')
        )


class AntiDriftProtection:
    """Anti-drift protection system with continuous monitoring"""

    def __init__(self, config_path: str = "logs/anti_drift_config.json"):
        self.config_path = Path(config_path)
        self.fingerprints: Dict[str, FileFingerprint] = {}
        self.protected_files: Set[str] = set()
        self.monitoring_active = False
        self.monitor_thread = None
        self.monitor_interval = 30  # seconds

        # Core protected files
        self.core_protected_files = {
            'config/compliance_enforcer.py',
            'config/governance_kernel.py',
            'core/governance_enforcement.py',
            'core/immutable_runtime_enforcer.py',
            'agent_rules.yaml',
            'agent_governance.py',
            'data_lineage_system.py',
            'modular_integration_system.py',
            'anti_drift_protection.py',
            'agent_code_of_conduct.py'
        }

        # Load configuration
        self._load_config()

        # Initialize protected files
        self._initialize_protected_files()

    def _load_config(self):
        """Load anti-drift configuration"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    data = json.load(f)

                    # Load fingerprints
                    for path, fingerprint_data in data.get(
                            'fingerprints', {}).items():
                        self.fingerprints[path] = FileFingerprint.from_dict(
                            fingerprint_data)

                    # Load protected files
                    self.protected_files = set(data.get('protected_files', []))

                    logger.info(
                        f"Loaded {len(self.fingerprints)} file fingerprints")
                    logger.info(
                        f"Loaded {len(self.protected_files)} protected files")

            except Exception as e:
                logger.error(f"Error loading anti-drift config: {e}")

    def _save_config(self):
        """Save anti-drift configuration"""
        self.config_path.parent.mkdir(exist_ok=True)

        try:
            data = {
                'fingerprints': {
                    path: fp.to_dict() for path,
                    fp in self.fingerprints.items()},
                'protected_files': list(
                    self.protected_files),
                'last_updated': datetime.now().isoformat()}

            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error saving anti-drift config: {e}")

    def _initialize_protected_files(self):
        """Initialize protected files with fingerprints"""
        for file_path in self.core_protected_files:
            if Path(file_path).exists():
                self.protect_file(file_path)

    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            return hashlib.sha256(content).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return ""

    def get_file_info(
            self, file_path: str) -> Optional[Tuple[int, datetime, int]]:
        """Get file information (size, modified time, permissions)"""
        try:
            stat_info = os.stat(file_path)
            return (
                stat_info.st_size,
                datetime.fromtimestamp(stat_info.st_mtime),
                stat_info.st_mode
            )
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {e}")
            return None

    def create_fingerprint(
            self,
            file_path: str,
            is_protected: bool = False) -> Optional[FileFingerprint]:
        """Create fingerprint for a file"""
        if not Path(file_path).exists():
            logger.warning(f"File does not exist: {file_path}")
            return None

        file_hash = self.calculate_file_hash(file_path)
        if not file_hash:
            return None

        file_info = self.get_file_info(file_path)
        if not file_info:
            return None

        size, modified_time, permissions = file_info

        fingerprint = FileFingerprint(
            path=file_path,
            hash=file_hash,
            size=size,
            last_modified=modified_time,
            permissions=permissions,
            is_protected=is_protected
        )

        self.fingerprints[file_path] = fingerprint
        self._save_config()

        logger.info(f"Created fingerprint for {file_path}")
        return fingerprint

    def protect_file(self, file_path: str) -> bool:
        """Protect a file from modification"""
        if not Path(file_path).exists():
            logger.warning(f"Cannot protect non-existent file: {file_path}")
            return False

        # Create fingerprint
        fingerprint = self.create_fingerprint(file_path, is_protected=True)
        if not fingerprint:
            return False

        # Add to protected set
        self.protected_files.add(file_path)

        # Create backup
        if fingerprint.backup_path:
            try:
                shutil.copy2(file_path, fingerprint.backup_path)
                logger.info(f"Created backup: {fingerprint.backup_path}")
            except Exception as e:
                logger.error(f"Error creating backup: {e}")

        self._save_config()
        logger.info(f"Protected file: {file_path}")
        return True

    def unprotect_file(self, file_path: str) -> bool:
        """Remove protection from a file"""
        if file_path not in self.protected_files:
            logger.warning(f"File not protected: {file_path}")
            return False

        self.protected_files.remove(file_path)

        # Update fingerprint
        if file_path in self.fingerprints:
            self.fingerprints[file_path].is_protected = False

        self._save_config()
        logger.info(f"Unprotected file: {file_path}")
        return True

    def verify_file_integrity(self, file_path: str) -> Tuple[bool, str]:
        """Verify file integrity against stored fingerprint"""
        if file_path not in self.fingerprints:
            return False, "No fingerprint found"

        fingerprint = self.fingerprints[file_path]

        if not Path(file_path).exists():
            return False, "File does not exist"

        # Calculate current hash
        current_hash = self.calculate_file_hash(file_path)
        if not current_hash:
            return False, "Could not calculate current hash"

        # Compare hashes
        if current_hash != fingerprint.hash:
            return False, f"Hash mismatch: expected {fingerprint.hash[:8]}..., got {current_hash[:8]}..."

        # Check file info
        file_info = self.get_file_info(file_path)
        if not file_info:
            return False, "Could not get file info"

        size, modified_time, permissions = file_info

        # Check if file was modified (but hash is same - potential collision)
        if modified_time != fingerprint.last_modified:
            return False, f"File modified: {fingerprint.last_modified} vs {modified_time}"

        return True, "File integrity verified"

    def detect_drift(self, file_path: str) -> Tuple[bool, str]:
        """Detect if file has drifted from expected state"""
        is_integrity_ok, message = self.verify_file_integrity(file_path)

        if not is_integrity_ok:
            return True, f"Drift detected: {message}"

        return False, "No drift detected"

    def restore_file(self, file_path: str) -> bool:
        """Restore file from backup"""
        if file_path not in self.fingerprints:
            logger.error(f"No fingerprint for file: {file_path}")
            return False

        fingerprint = self.fingerprints[file_path]
        if not fingerprint.backup_path or not Path(
                fingerprint.backup_path).exists():
            logger.error(f"No backup available for file: {file_path}")
            return False

        try:
            # Restore from backup
            shutil.copy2(fingerprint.backup_path, file_path)

            # Update fingerprint
            self.create_fingerprint(file_path, fingerprint.is_protected)

            logger.info(f"Restored file from backup: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error restoring file {file_path}: {e}")
            return False

    def start_monitoring(self):
        """Start continuous monitoring"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Started anti-drift monitoring")

    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("Stopped anti-drift monitoring")

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self._check_all_protected_files()
                time.sleep(self.monitor_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitor_interval)

    def _check_all_protected_files(self):
        """Check all protected files for drift"""
        violations = []

        for file_path in self.protected_files:
            has_drift, message = self.detect_drift(file_path)
            if has_drift:
                violations.append({
                    'file': file_path,
                    'message': message,
                    'timestamp': datetime.now().isoformat()
                })

                # Log violation
                self._log_violation(file_path, message)

                # Auto-restore if enabled
                if self._should_auto_restore(file_path):
                    self.restore_file(file_path)

        if violations:
            logger.warning(f"Found {len(violations)} drift violations")
            self._save_violations(violations)

    def _should_auto_restore(self, file_path: str) -> bool:
        """Determine if file should be auto-restored"""
        # Auto-restore core files
        if file_path in self.core_protected_files:
            return True

        # Check if file is marked for auto-restore
        if file_path in self.fingerprints:
            return self.fingerprints[file_path].metadata.get(
                'auto_restore', False)

        return False

    def _log_violation(self, file_path: str, message: str):
        """Log a violation"""
        violation_log = Path("logs/anti_drift_violations.log")
        violation_log.parent.mkdir(exist_ok=True)

        with open(violation_log, 'a') as f:
            f.write(f"{datetime.now().isoformat()} - {file_path}: {message}\n")

    def _save_violations(self, violations: List[Dict[str, Any]]):
        """Save violations to JSON file"""
        violations_file = Path("logs/anti_drift_violations.json")
        violations_file.parent.mkdir(exist_ok=True)

        try:
            # Load existing violations
            existing_violations = []
            if violations_file.exists():
                with open(violations_file, 'r') as f:
                    existing_violations = json.load(f)

            # Add new violations
            existing_violations.extend(violations)

            # Save updated violations
            with open(violations_file, 'w') as f:
                json.dump(existing_violations, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error saving violations: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        status = {
            'monitoring_active': self.monitoring_active,
            'total_fingerprints': len(self.fingerprints),
            'protected_files': len(self.protected_files),
            'core_protected': len(self.core_protected_files),
            'recent_violations': [],
            'file_status': {}
        }

        # Check status of all protected files
        for file_path in self.protected_files:
            is_ok, message = self.verify_file_integrity(file_path)
            status['file_status'][file_path] = {
                'integrity_ok': is_ok,
                'message': message,
                'is_protected': True
            }

        # Load recent violations
        violations_file = Path("logs/anti_drift_violations.json")
        if violations_file.exists():
            try:
                with open(violations_file, 'r') as f:
                    all_violations = json.load(f)
                    # Get last 10 violations
                    status['recent_violations'] = all_violations[-10:]
            except Exception as e:
                logger.error(f"Error loading violations: {e}")

        return status

    def add_file_to_monitoring(
            self,
            file_path: str,
            auto_restore: bool = False):
        """Add file to monitoring (without protection)"""
        fingerprint = self.create_fingerprint(file_path, is_protected=False)
        if fingerprint and auto_restore:
            fingerprint.metadata['auto_restore'] = True
            self._save_config()

        logger.info(f"Added file to monitoring: {file_path}")

    def remove_file_from_monitoring(self, file_path: str):
        """Remove file from monitoring"""
        if file_path in self.fingerprints:
            del self.fingerprints[file_path]

        if file_path in self.protected_files:
            self.unprotect_file(file_path)

        self._save_config()
        logger.info(f"Removed file from monitoring: {file_path}")


# Global anti-drift system instance
anti_drift_system = AntiDriftProtection()


def protect_file(file_path: str) -> bool:
    """Protect a file from modification"""
    return anti_drift_system.protect_file(file_path)


def unprotect_file(file_path: str) -> bool:
    """Remove protection from a file"""
    return anti_drift_system.unprotect_file(file_path)


def verify_file_integrity(file_path: str) -> Tuple[bool, str]:
    """Verify file integrity"""
    return anti_drift_system.verify_file_integrity(file_path)


def detect_drift(file_path: str) -> Tuple[bool, str]:
    """Detect file drift"""
    return anti_drift_system.detect_drift(file_path)


def restore_file(file_path: str) -> bool:
    """Restore file from backup"""
    return anti_drift_system.restore_file(file_path)


def start_monitoring():
    """Start anti-drift monitoring"""
    anti_drift_system.start_monitoring()


def stop_monitoring():
    """Stop anti-drift monitoring"""
    anti_drift_system.stop_monitoring()


def get_system_status() -> Dict[str, Any]:
    """Get anti-drift system status"""
    return anti_drift_system.get_system_status()


if __name__ == "__main__":
    # Example usage
    print("🛡️ Anti-Drift Protection System")

    # Start monitoring
    start_monitoring()

    # Get system status
    status = get_system_status()
    print(f"System status: {status}")

    # Stop monitoring
    stop_monitoring()
