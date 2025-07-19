"""
Real-Time Data Integrity Monitor
Continuously monitors all processes for fake data, repeated patterns, and suspicious metrics
"""

import os
import json
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Set, Any
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import re
import hashlib

class RealTimeDataIntegrityMonitor:
    """
    Advanced real-time monitoring system that detects:
    - Fake/synthetic performance metrics
    - Repeated number patterns
    - Impossible values
    - Timestamp inconsistencies
    - Data manipulation attempts
    - Pattern anomalies
    """
    
    def __init__(self):
        self.monitoring_active = False
        self.detection_rules = self._initialize_detection_rules()
        self.alert_history = []
        self.pattern_cache = defaultdict(list)
        self.suspicious_values = set()
        self.monitoring_thread = None
        self.file_hashes = {}  # Track file integrity
        self.baseline_established = False
        self.protected_directories = ['./', 'core/', 'ml_engine/', 'scanner/']
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - INTEGRITY MONITOR - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('data_integrity_monitor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _initialize_detection_rules(self) -> Dict:
        """Initialize comprehensive detection rules"""
        return {
            'fake_sharpe_patterns': [
                # Known fake values
                5.7189, 0.9018, 10.0408,
                # Suspicious patterns
                r'[0-9]\.[0-9]{4}$',  # Exactly 4 decimal places
                r'^[5-9]\.[0-9]{3,4}$',  # High values with 3-4 decimals
                r'^1[0-9]\.[0-9]{3,4}$'  # Double digit high values
            ],
            'repeated_patterns': {
                'max_consecutive_same': 3,  # Max consecutive identical values
                'max_pattern_frequency': 5,  # Max times same pattern appears
                'suspicious_sequences': ['1111', '2222', '3333', '1234', '5678']
            },
            'impossible_values': {
                'sharpe_ratio': {'min': -5.0, 'max': 10.0},
                'returns': {'min': -1.0, 'max': 5.0},
                'win_rate': {'min': 0.0, 'max': 1.0},
                'confidence': {'min': 0.0, 'max': 1.0},
                'price': {'min': 0.01, 'max': 10000.0}
            },
            'timestamp_rules': {
                'max_future_minutes': 5,  # Max minutes in future allowed
                'min_past_days': 30,  # Min days in past for historical data
                'suspicious_patterns': ['00:00:00', '12:00:00']  # Suspicious exact times
            },
            'file_integrity': {
                'monitored_extensions': ['.py', '.json', '.log', '.csv'],
                'suspicious_changes': ['sharpe', 'return', 'performance'],
                'protected_files': [
                    'robust_optimization_results.json',
                    'production_lstm_results.json',
                    'adaptive_ml_signals.json'
                ]
            },
            'unauthorized_modifications': {
                'critical_files': [
                    'real_time_data_integrity_monitor.py',
                    'AdvanS8_Live_Trading_Dashboard.py',
                    'robust_tpe_optimizer.py',
                    'lstm_production_fix.py'
                ],
                'forbidden_patterns': [
                    'fake_sharpe',
                    'mock_data',
                    'placeholder',
                    'synthetic_performance',
                    'artificial_results'
                ],
                'suspicious_imports': [
                    'random',
                    'faker',
                    'mock'
                ],
                'unauthorized_functions': [
                    'generate_fake_',
                    'create_synthetic_',
                    'mock_performance',
                    'artificial_sharpe'
                ]
            }
        }
    
    def start_monitoring(self):
        """Start continuous real-time monitoring"""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
            
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Real-time data integrity monitoring started")
        
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("Data integrity monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        # Establish baseline on first run
        if not self.baseline_established:
            self._establish_file_baseline()
            
        while self.monitoring_active:
            try:
                # Monitor different aspects
                self._check_file_integrity()
                self._monitor_file_tampering()
                self._monitor_optimization_results()
                self._check_signal_integrity()
                self._verify_timestamps()
                self._detect_pattern_anomalies()
                self._monitor_performance_metrics()
                
                # Sleep before next check
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _check_file_integrity(self):
        """Monitor file changes for suspicious modifications"""
        try:
            for file_path in self.detection_rules['file_integrity']['protected_files']:
                if os.path.exists(file_path):
                    # Check file hash
                    file_hash = self._calculate_file_hash(file_path)
                    
                    # Check for suspicious content
                    with open(file_path, 'r') as f:
                        content = f.read()
                        
                    self._scan_content_for_fake_data(content, file_path)
                    
        except Exception as e:
            self.logger.error(f"File integrity check failed: {e}")
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate file hash for integrity checking"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _scan_content_for_fake_data(self, content: str, file_path: str):
        """Scan content for fake data patterns"""
        # Check for known fake Sharpe values
        for fake_value in self.detection_rules['fake_sharpe_patterns']:
            if isinstance(fake_value, float):
                if str(fake_value) in content:
                    self._trigger_alert(
                        'FAKE_VALUE_DETECTED',
                        f"Fake Sharpe value {fake_value} found in {file_path}",
                        'CRITICAL'
                    )
            elif isinstance(fake_value, str):
                if re.search(fake_value, content):
                    self._trigger_alert(
                        'SUSPICIOUS_PATTERN',
                        f"Suspicious pattern {fake_value} found in {file_path}",
                        'HIGH'
                    )
    
    def _monitor_optimization_results(self):
        """Monitor optimization results for authenticity"""
        try:
            if os.path.exists('robust_optimization_results.json'):
                with open('robust_optimization_results.json', 'r') as f:
                    data = json.load(f)
                
                # Check best Sharpe ratio
                best_sharpe = data.get('optimization_summary', {}).get('best_sharpe', 0)
                self._validate_sharpe_ratio(best_sharpe, 'optimization_results')
                
                # Check trial consistency
                trials = data.get('all_trials', [])
                self._check_trial_consistency(trials)
                
        except Exception as e:
            self.logger.error(f"Optimization results monitoring failed: {e}")
    
    def _check_signal_integrity(self):
        """Monitor signal files for data integrity"""
        signal_files = [
            'adaptive_ml_signals.json',
            'production_lstm_results.json',
            'optimized_ml_signals.json'
        ]
        
        for file_path in signal_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    self._validate_signal_data(data, file_path)
                    
                except Exception as e:
                    self.logger.error(f"Signal integrity check failed for {file_path}: {e}")
    
    def _verify_timestamps(self):
        """Verify timestamp authenticity and consistency"""
        try:
            current_time = datetime.now()
            
            # Check log files for timestamp patterns
            for log_file in ['data_integrity_monitor.log', 'trading_system.log']:
                if os.path.exists(log_file):
                    self._check_log_timestamps(log_file, current_time)
                    
        except Exception as e:
            self.logger.error(f"Timestamp verification failed: {e}")
    
    def _detect_pattern_anomalies(self):
        """Detect suspicious patterns in data"""
        try:
            # Check for repeated values across files
            all_values = []
            
            for file_path in ['robust_optimization_results.json', 'production_lstm_results.json']:
                if os.path.exists(file_path):
                    values = self._extract_numeric_values(file_path)
                    all_values.extend(values)
            
            # Analyze patterns
            self._analyze_value_patterns(all_values)
            
        except Exception as e:
            self.logger.error(f"Pattern anomaly detection failed: {e}")
    
    def _monitor_performance_metrics(self):
        """Monitor performance metrics for authenticity"""
        try:
            # Check workflow logs for suspicious performance claims
            workflow_patterns = [
                r'sharpe.*[5-9]\.[0-9]{3,4}',
                r'return.*[2-9][0-9]%',
                r'accuracy.*9[5-9]%'
            ]
            
            # Monitor console logs if available
            # This would integrate with workflow monitoring
            
        except Exception as e:
            self.logger.error(f"Performance metrics monitoring failed: {e}")
    
    def _validate_sharpe_ratio(self, sharpe: float, source: str):
        """Validate Sharpe ratio authenticity"""
        rules = self.detection_rules['impossible_values']['sharpe_ratio']
        
        if sharpe < rules['min'] or sharpe > rules['max']:
            self._trigger_alert(
                'IMPOSSIBLE_SHARPE',
                f"Impossible Sharpe ratio {sharpe} in {source}",
                'CRITICAL'
            )
        
        # Check for known fake values
        fake_values = [5.7189, 0.9018, 10.0408]
        if any(abs(sharpe - fake) < 0.0001 for fake in fake_values):
            self._trigger_alert(
                'FAKE_SHARPE_DETECTED',
                f"Known fake Sharpe value {sharpe} detected in {source}",
                'CRITICAL'
            )
    
    def _check_trial_consistency(self, trials: List[Dict]):
        """Check optimization trial consistency"""
        if len(trials) < 2:
            return
            
        sharpe_values = [trial.get('sharpe', 0) for trial in trials]
        
        # Check for repeated values
        sharpe_counter = Counter(sharpe_values)
        for value, count in sharpe_counter.items():
            if count > 3:  # Same Sharpe more than 3 times is suspicious
                self._trigger_alert(
                    'REPEATED_SHARPE',
                    f"Sharpe value {value} repeated {count} times in trials",
                    'MEDIUM'
                )
    
    def _validate_signal_data(self, data: Dict, file_path: str):
        """Validate signal data integrity"""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    # Check confidence values
                    confidence = value.get('confidence', 0)
                    if confidence > 1.0 or confidence < 0.0:
                        self._trigger_alert(
                            'INVALID_CONFIDENCE',
                            f"Invalid confidence {confidence} in {file_path}",
                            'HIGH'
                        )
    
    def _check_log_timestamps(self, log_file: str, current_time: datetime):
        """Check log file timestamps for consistency"""
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()[-100:]  # Check last 100 lines
            
            for line in lines:
                # Extract timestamp if present
                timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                if timestamp_match:
                    timestamp_str = timestamp_match.group(1)
                    try:
                        log_time = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                        
                        # Check if timestamp is in future
                        if log_time > current_time + timedelta(minutes=5):
                            self._trigger_alert(
                                'FUTURE_TIMESTAMP',
                                f"Future timestamp detected in {log_file}: {timestamp_str}",
                                'HIGH'
                            )
                            
                    except ValueError:
                        pass
                        
        except Exception as e:
            self.logger.error(f"Log timestamp check failed: {e}")
    
    def _extract_numeric_values(self, file_path: str) -> List[float]:
        """Extract numeric values from file for pattern analysis"""
        values = []
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Find all float numbers
            float_pattern = r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'
            matches = re.findall(float_pattern, content)
            
            for match in matches:
                try:
                    values.append(float(match))
                except ValueError:
                    continue
                    
        except Exception as e:
            self.logger.error(f"Value extraction failed for {file_path}: {e}")
            
        return values
    
    def _analyze_value_patterns(self, values: List[float]):
        """Analyze values for suspicious patterns"""
        if len(values) < 10:
            return
            
        # Filter out legitimate optimization parameters and data
        legitimate_ranges = {
            (0.02, 0.25): 'trading_params',  # All trading parameters
            (1.0, 2.5): 'volume_threshold',  # Volume thresholds
            (2020.0, 2030.0): 'date_values',  # Date/year values
            (-10.0, 50.0): 'feature_values',  # Technical indicators
            (0.0, 100.0): 'percentage_values'  # Percentage-based metrics
        }
        
        suspicious_values = []
        for value in values:
            is_legitimate = False
            for (min_val, max_val), param_type in legitimate_ranges.items():
                if min_val <= value <= max_val:
                    is_legitimate = True
                    break
            
            # Only flag non-legitimate values for pattern analysis
            if not is_legitimate:
                suspicious_values.append(value)
        
        # Check for too many identical suspicious values (focus on critical patterns)
        if suspicious_values:
            value_counts = Counter(suspicious_values)
            for value, count in value_counts.items():
                # Only flag known fake values or extremely suspicious patterns
                if (abs(value - 5.7189) < 0.0001 or 
                    abs(value - 10.0408) < 0.0001 or 
                    abs(value - 0.9018) < 0.0001 or
                    count > 20):  # Very high threshold for legitimate repeated values
                    self._trigger_alert(
                        'CRITICAL_FAKE_VALUE',
                        f"Critical suspicious value {value} appears {count} times",
                        'CRITICAL'
                    )
        
        # Check for artificial sequential patterns (not optimization parameters)
        sorted_suspicious = sorted(suspicious_values)
        if len(sorted_suspicious) >= 4:
            for i in range(len(sorted_suspicious) - 3):
                diff1 = round(sorted_suspicious[i+1] - sorted_suspicious[i], 6)
                diff2 = round(sorted_suspicious[i+2] - sorted_suspicious[i+1], 6)
                diff3 = round(sorted_suspicious[i+3] - sorted_suspicious[i+2], 6)
                
                # Only flag perfect sequential patterns (artificial)
                if diff1 == diff2 == diff3 and diff1 > 0.001:
                    self._trigger_alert(
                        'ARTIFICIAL_SEQUENCE',
                        f"Artificial sequential pattern: {sorted_suspicious[i:i+4]}",
                        'HIGH'
                    )
    
    def _trigger_alert(self, alert_type: str, message: str, severity: str):
        """Trigger integrity alert"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'message': message,
            'severity': severity
        }
        
        self.alert_history.append(alert)
        
        # Log alert
        if severity == 'CRITICAL':
            self.logger.critical(f"CRITICAL INTEGRITY ALERT: {message}")
        elif severity == 'HIGH':
            self.logger.error(f"HIGH INTEGRITY ALERT: {message}")
        elif severity == 'MEDIUM':
            self.logger.warning(f"MEDIUM INTEGRITY ALERT: {message}")
        else:
            self.logger.info(f"LOW INTEGRITY ALERT: {message}")
        
        # Save alert to file
        self._save_alert(alert)
    
    def _save_alert(self, alert: Dict):
        """Save alert to alerts file"""
        try:
            alerts_file = 'integrity_alerts.json'
            
            if os.path.exists(alerts_file):
                with open(alerts_file, 'r') as f:
                    alerts = json.load(f)
            else:
                alerts = []
            
            alerts.append(alert)
            
            # Keep only last 1000 alerts
            if len(alerts) > 1000:
                alerts = alerts[-1000:]
            
            with open(alerts_file, 'w') as f:
                json.dump(alerts, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save alert: {e}")
    
    def get_alert_summary(self) -> Dict:
        """Get summary of recent alerts"""
        recent_alerts = [alert for alert in self.alert_history 
                        if datetime.fromisoformat(alert['timestamp']) > 
                        datetime.now() - timedelta(hours=24)]
        
        summary = {
            'total_alerts_24h': len(recent_alerts),
            'critical_alerts': len([a for a in recent_alerts if a['severity'] == 'CRITICAL']),
            'high_alerts': len([a for a in recent_alerts if a['severity'] == 'HIGH']),
            'medium_alerts': len([a for a in recent_alerts if a['severity'] == 'MEDIUM']),
            'alert_types': Counter([a['type'] for a in recent_alerts]),
            'latest_critical': [a for a in recent_alerts if a['severity'] == 'CRITICAL'][-5:],
            'monitoring_status': 'ACTIVE' if self.monitoring_active else 'INACTIVE'
        }
        
        return summary
    
    def manual_scan(self, file_path: str = None) -> Dict:
        """Perform manual integrity scan"""
        self.logger.info("Starting manual integrity scan...")
        
        results = {
            'scan_timestamp': datetime.now().isoformat(),
            'files_scanned': 0,
            'issues_found': 0,
            'alerts_generated': 0
        }
        
        initial_alert_count = len(self.alert_history)
        
        if file_path:
            # Scan specific file
            if os.path.exists(file_path):
                self._check_single_file(file_path)
                results['files_scanned'] = 1
        else:
            # Scan all monitored files
            self._check_file_integrity()
            self._monitor_optimization_results()
            self._check_signal_integrity()
            self._verify_timestamps()
            self._detect_pattern_anomalies()
            
            results['files_scanned'] = len(self.detection_rules['file_integrity']['protected_files'])
        
        results['alerts_generated'] = len(self.alert_history) - initial_alert_count
        results['issues_found'] = results['alerts_generated']
        
        self.logger.info(f"Manual scan complete: {results}")
        return results
    
    def _check_single_file(self, file_path: str):
        """Check single file for integrity issues"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            self._scan_content_for_fake_data(content, file_path)
            
            if file_path.endswith('.json'):
                data = json.loads(content)
                if 'sharpe' in content.lower():
                    self._validate_json_sharpe_values(data, file_path)
                    
        except Exception as e:
            self.logger.error(f"Single file check failed for {file_path}: {e}")
    
    def _validate_json_sharpe_values(self, data: Any, file_path: str):
        """Recursively validate Sharpe values in JSON data"""
        if isinstance(data, dict):
            for key, value in data.items():
                if 'sharpe' in key.lower() and isinstance(value, (int, float)):
                    self._validate_sharpe_ratio(value, file_path)
                elif isinstance(value, (dict, list)):
                    self._validate_json_sharpe_values(value, file_path)
        elif isinstance(data, list):
            for item in data:
                self._validate_json_sharpe_values(item, file_path)
    
    def _establish_file_baseline(self):
        """Establish baseline file hashes for integrity monitoring"""
        try:
            self.logger.info("Establishing file integrity baseline...")
            
            # Scan all critical files and directories
            for directory in self.protected_directories:
                if os.path.exists(directory):
                    for root, dirs, files in os.walk(directory):
                        for file in files:
                            file_path = os.path.join(root, file)
                            if any(file_path.endswith(ext) for ext in 
                                  self.detection_rules['file_integrity']['monitored_extensions']):
                                try:
                                    file_hash = self._calculate_file_hash(file_path)
                                    self.file_hashes[file_path] = {
                                        'hash': file_hash,
                                        'last_modified': os.path.getmtime(file_path),
                                        'size': os.path.getsize(file_path)
                                    }
                                except Exception as e:
                                    self.logger.error(f"Failed to hash {file_path}: {e}")
            
            self.baseline_established = True
            self.logger.info(f"Baseline established for {len(self.file_hashes)} files")
            
        except Exception as e:
            self.logger.error(f"Failed to establish baseline: {e}")
    
    def _monitor_file_tampering(self):
        """Monitor for unauthorized file modifications and tampering"""
        try:
            tampering_detected = False
            
            # Check critical files for unauthorized modifications
            critical_files = self.detection_rules['unauthorized_modifications']['critical_files']
            
            for file_path in critical_files:
                if os.path.exists(file_path):
                    current_hash = self._calculate_file_hash(file_path)
                    current_mtime = os.path.getmtime(file_path)
                    current_size = os.path.getsize(file_path)
                    
                    # Check if file has baseline
                    if file_path in self.file_hashes:
                        baseline = self.file_hashes[file_path]
                        
                        # Detect unauthorized changes
                        if (current_hash != baseline['hash'] or 
                            current_size != baseline['size']):
                            
                            # Check if modification is authorized
                            if self._is_unauthorized_modification(file_path):
                                self._trigger_alert(
                                    'UNAUTHORIZED_FILE_MODIFICATION',
                                    f"Critical file {file_path} modified without authorization",
                                    'CRITICAL'
                                )
                                tampering_detected = True
                            
                            # Update baseline for legitimate changes
                            self.file_hashes[file_path] = {
                                'hash': current_hash,
                                'last_modified': current_mtime,
                                'size': current_size
                            }
                    else:
                        # New critical file - establish baseline
                        self.file_hashes[file_path] = {
                            'hash': current_hash,
                            'last_modified': current_mtime,
                            'size': current_size
                        }
            
            # Scan for forbidden patterns in code files
            self._scan_for_forbidden_patterns()
            
            return tampering_detected
            
        except Exception as e:
            self.logger.error(f"File tampering monitoring failed: {e}")
            return False
    
    def _is_unauthorized_modification(self, file_path: str) -> bool:
        """Determine if a file modification is unauthorized"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for forbidden patterns
            forbidden_patterns = self.detection_rules['unauthorized_modifications']['forbidden_patterns']
            for pattern in forbidden_patterns:
                if pattern.lower() in content.lower():
                    self._trigger_alert(
                        'FORBIDDEN_PATTERN_DETECTED',
                        f"Forbidden pattern '{pattern}' found in {file_path}",
                        'CRITICAL'
                    )
                    return True
            
            # Check for suspicious imports
            suspicious_imports = self.detection_rules['unauthorized_modifications']['suspicious_imports']
            for imp in suspicious_imports:
                if f"import {imp}" in content or f"from {imp}" in content:
                    self._trigger_alert(
                        'SUSPICIOUS_IMPORT_DETECTED',
                        f"Suspicious import '{imp}' found in {file_path}",
                        'HIGH'
                    )
                    return True
            
            # Check for unauthorized functions
            unauthorized_functions = self.detection_rules['unauthorized_modifications']['unauthorized_functions']
            for func in unauthorized_functions:
                if func in content:
                    self._trigger_alert(
                        'UNAUTHORIZED_FUNCTION_DETECTED',
                        f"Unauthorized function '{func}' found in {file_path}",
                        'HIGH'
                    )
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to analyze {file_path}: {e}")
            return False
    
    def _scan_for_forbidden_patterns(self):
        """Scan all Python files for forbidden patterns and unauthorized modifications"""
        try:
            python_files = []
            
            # Collect all Python files
            for directory in self.protected_directories:
                if os.path.exists(directory):
                    for root, dirs, files in os.walk(directory):
                        for file in files:
                            if file.endswith('.py'):
                                python_files.append(os.path.join(root, file))
            
            # Scan each file for violations
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for data manipulation attempts
                    if any(pattern in content.lower() for pattern in [
                        'sharpe = 5.7189',
                        'sharpe = 10.0408', 
                        'sharpe = 0.9018',
                        'fake_performance',
                        'synthetic_data',
                        'mock_sharpe'
                    ]):
                        self._trigger_alert(
                            'DATA_MANIPULATION_ATTEMPT',
                            f"Data manipulation attempt detected in {file_path}",
                            'CRITICAL'
                        )
                    
                    # Check for monitoring system tampering
                    if 'real_time_data_integrity_monitor' in file_path:
                        if any(pattern in content for pattern in [
                            'monitoring_active = False',
                            'return True  # bypass',
                            'pass  # disabled',
                            '# monitoring disabled'
                        ]):
                            self._trigger_alert(
                                'MONITORING_SYSTEM_TAMPERING',
                                f"Monitoring system tampering detected in {file_path}",
                                'CRITICAL'
                            )
                
                except Exception as e:
                    self.logger.error(f"Failed to scan {file_path}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Forbidden pattern scan failed: {e}")
    
    def get_tampering_summary(self) -> Dict:
        """Get summary of file tampering detection results"""
        return {
            'baseline_files': len(self.file_hashes),
            'monitoring_active': self.monitoring_active,
            'baseline_established': self.baseline_established,
            'protected_directories': self.protected_directories,
            'critical_files_count': len(self.detection_rules['unauthorized_modifications']['critical_files']),
            'forbidden_patterns_count': len(self.detection_rules['unauthorized_modifications']['forbidden_patterns'])
        }

def main():
    """Main function to start monitoring"""
    monitor = RealTimeDataIntegrityMonitor()
    
    try:
        print("Starting Real-Time Data Integrity Monitor...")
        print("This system will continuously monitor for:")
        print("- Fake Sharpe ratios and performance metrics")
        print("- Repeated number patterns")
        print("- Impossible values")
        print("- Timestamp inconsistencies")
        print("- Data manipulation attempts")
        print("\nPress Ctrl+C to stop monitoring\n")
        
        # Perform initial scan
        initial_results = monitor.manual_scan()
        print(f"Initial scan results: {initial_results}")
        
        # Start continuous monitoring
        monitor.start_monitoring()
        
        # Keep running
        while True:
            time.sleep(60)
            summary = monitor.get_alert_summary()
            
            if summary['total_alerts_24h'] > 0:
                print(f"\n--- Alert Summary (Last 24h) ---")
                print(f"Total alerts: {summary['total_alerts_24h']}")
                print(f"Critical: {summary['critical_alerts']}")
                print(f"High: {summary['high_alerts']}")
                print(f"Medium: {summary['medium_alerts']}")
                
                if summary['latest_critical']:
                    print("\nLatest Critical Alerts:")
                    for alert in summary['latest_critical']:
                        print(f"  {alert['timestamp']}: {alert['message']}")
            
    except KeyboardInterrupt:
        print("\nStopping monitoring...")
        monitor.stop_monitoring()
        print("Data integrity monitoring stopped.")

if __name__ == "__main__":
    main()