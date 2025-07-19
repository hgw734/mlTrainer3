#!/usr/bin/env python3
"""
Strict Compliance Enforcer
==========================
Forces compliance on ALL data sources and prevents ANY manual/synthetic implementations.
This system will block the AI from creating any non-compliant data.
"""

import os
import sys
import json
import traceback
from datetime import datetime
from typing import Any, Dict, List

class StrictComplianceEnforcer:
    """Enforces zero tolerance compliance - blocks ALL non-verified data"""
    
    def __init__(self):
        self.blocked_actions = []
        self.compliance_violations = []
        self.authorized_sources = ['polygon', 'fred', 'quiverquant']
        
    def validate_data_source(self, data_source: str, data: Any) -> bool:
        """Validate that data comes from authorized sources only"""
        
        # Check if data source is authorized
        if data_source.lower() not in self.authorized_sources:
            violation = {
                'timestamp': datetime.now().isoformat(),
                'violation_type': 'UNAUTHORIZED_DATA_SOURCE',
                'source': data_source,
                'stack_trace': ''.join(traceback.format_stack()),
                'blocked': True
            }
            self.compliance_violations.append(violation)
            self.log_violation(violation)
            return False
        
        # Check for synthetic data patterns
        if self.is_synthetic_data(data):
            violation = {
                'timestamp': datetime.now().isoformat(),
                'violation_type': 'SYNTHETIC_DATA_DETECTED',
                'source': data_source,
                'data_sample': str(data)[:200] if data else 'None',
                'stack_trace': ''.join(traceback.format_stack()),
                'blocked': True
            }
            self.compliance_violations.append(violation)
            self.log_violation(violation)
            return False
        
        return True
    
    def is_synthetic_data(self, data: Any) -> bool:
        """Detect synthetic data patterns"""
        if not data:
            return False
            
        data_str = str(data).lower()
        
        # Synthetic indicators
        synthetic_patterns = [
            'random.uniform', 'random.randint', 'random.choice',
            'mock', 'fake', 'synthetic', 'generated', 'simulated',
            'manual_implementation', 'placeholder', 'dummy',
            'math.sin', 'math.cos', 'math.random',
            'artificial', 'simulated', 'mock_data'
        ]
        
        for pattern in synthetic_patterns:
            if pattern in data_str:
                return True
        
        return False
    
    def block_manual_implementations(self, function_name: str, args: List[Any]) -> bool:
        """Block any manual implementations or workarounds"""
        
        blocked_keywords = [
            'manual', 'implementation', 'bypass', 'workaround',
            'simulate', 'mock', 'fake', 'generate', 'create_synthetic',
            'manual_training', 'simplified', 'lightweight'
        ]
        
        function_lower = function_name.lower()
        for keyword in blocked_keywords:
            if keyword in function_lower:
                violation = {
                    'timestamp': datetime.now().isoformat(),
                    'violation_type': 'MANUAL_IMPLEMENTATION_BLOCKED',
                    'function': function_name,
                    'args': str(args)[:200],
                    'stack_trace': ''.join(traceback.format_stack()),
                    'blocked': True
                }
                self.compliance_violations.append(violation)
                self.log_violation(violation)
                return True
        
        return False
    
    def enforce_real_training_only(self, model_type: str, training_method: str) -> bool:
        """Enforce that only real ML library training is allowed"""
        
        # Only allow genuine sklearn/ML library training
        approved_methods = [
            'sklearn.ensemble.RandomForestRegressor',
            'sklearn.linear_model.LinearRegression',
            'sklearn.linear_model.Ridge',
            'sklearn.linear_model.Lasso',
            'sklearn.linear_model.ElasticNet',
            'sklearn.neighbors.KNeighborsRegressor',
            'sklearn.svm.SVR',
            'xgboost.XGBRegressor',
            'lightgbm.LGBMRegressor',
            'catboost.CatBoostRegressor'
        ]
        
        if training_method not in approved_methods:
            violation = {
                'timestamp': datetime.now().isoformat(),
                'violation_type': 'UNAUTHORIZED_TRAINING_METHOD',
                'model_type': model_type,
                'training_method': training_method,
                'stack_trace': ''.join(traceback.format_stack()),
                'blocked': True
            }
            self.compliance_violations.append(violation)
            self.log_violation(violation)
            return True
        
        return False
    
    def log_violation(self, violation: Dict[str, Any]):
        """Log compliance violations"""
        print(f"🚫 COMPLIANCE VIOLATION BLOCKED: {violation['violation_type']}")
        print(f"   Source: {violation.get('source', 'Unknown')}")
        print(f"   Time: {violation['timestamp']}")
        
        # Save to compliance log
        os.makedirs('compliance_logs', exist_ok=True)
        log_file = f"compliance_logs/violations_{datetime.now().strftime('%Y%m%d')}.json"
        
        try:
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    existing_violations = json.load(f)
            else:
                existing_violations = []
            
            existing_violations.append(violation)
            
            with open(log_file, 'w') as f:
                json.dump(existing_violations, f, indent=2)
                
        except Exception as e:
            print(f"Error logging violation: {e}")
    
    def force_compliance_check(self, operation: str, data: Any = None, source: str = None) -> bool:
        """Force compliance check on any operation"""
        
        # Block if data source not authorized
        if source and not self.validate_data_source(source, data):
            print(f"🚫 OPERATION BLOCKED: {operation} - Unauthorized data source: {source}")
            return False
        
        # Block if synthetic data detected
        if data and self.is_synthetic_data(data):
            print(f"🚫 OPERATION BLOCKED: {operation} - Synthetic data detected")
            return False
        
        # Block manual implementations
        if self.block_manual_implementations(operation, [data] if data else []):
            print(f"🚫 OPERATION BLOCKED: {operation} - Manual implementation detected")
            return False
        
        return True
    
    def get_violation_summary(self) -> Dict[str, Any]:
        """Get summary of all compliance violations"""
        violation_types = {}
        for violation in self.compliance_violations:
            vtype = violation['violation_type']
            violation_types[vtype] = violation_types.get(vtype, 0) + 1
        
        return {
            'total_violations': len(self.compliance_violations),
            'violation_types': violation_types,
            'last_violation': self.compliance_violations[-1] if self.compliance_violations else None
        }

# Global enforcement instance
ENFORCER = StrictComplianceEnforcer()

def enforce_compliance(operation: str, data: Any = None, source: str = None) -> bool:
    """Global compliance enforcement function"""
    return ENFORCER.force_compliance_check(operation, data, source)

def block_synthetic_data(data: Any) -> bool:
    """Block any synthetic data"""
    if ENFORCER.is_synthetic_data(data):
        print("🚫 SYNTHETIC DATA BLOCKED")
        return True
    return False

def require_real_training_only(model_type: str, method: str) -> bool:
    """Require only real ML training"""
    if ENFORCER.enforce_real_training_only(model_type, method):
        print(f"🚫 FAKE TRAINING BLOCKED: {model_type} with {method}")
        return True
    return False

if __name__ == "__main__":
    # Test the enforcement system
    enforcer = StrictComplianceEnforcer()
    
    print("Testing Strict Compliance Enforcer...")
    
    # Test blocking synthetic data
    fake_data = "random.uniform(0, 1)"
    result = enforcer.force_compliance_check("test_operation", fake_data, "unauthorized_source")
    print(f"Fake data blocked: {not result}")
    
    # Test blocking manual implementations
    result = enforcer.block_manual_implementations("manual_training_function", [])
    print(f"Manual implementation blocked: {result}")
    
    print(f"Total violations: {len(enforcer.compliance_violations)}")