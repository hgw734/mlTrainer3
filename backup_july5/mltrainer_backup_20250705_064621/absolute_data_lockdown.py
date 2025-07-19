#!/usr/bin/env python3
"""
Absolute Data Lockdown System
============================
ONLY Polygon and FRED data sources allowed. Everything else blocked permanently.
"""

import os
import sys
import json
from datetime import datetime

class AbsoluteDataLockdown:
    """Blocks ALL data except Polygon and FRED - no exceptions"""
    
    def __init__(self):
        # ONLY these sources allowed - from API config
        self.AUTHORIZED_SOURCES = ['polygon', 'fred']
        self.blocked_attempts = []
        
    def is_authorized_source(self, source_name: str) -> bool:
        """Check if data source is in authorized API config"""
        return source_name.lower() in self.AUTHORIZED_SOURCES
    
    def block_all_other_data(self, operation: str, data_source: str = None) -> bool:
        """Block any operation not using Polygon or FRED"""
        
        if data_source and not self.is_authorized_source(data_source):
            self.log_blocked_attempt(operation, data_source, "UNAUTHORIZED_SOURCE")
            print(f"🚫 BLOCKED: {operation} - Only Polygon/FRED allowed, attempted: {data_source}")
            return True
            
        # Block any data generation, creation, simulation, etc.
        blocked_operations = [
            'create', 'generate', 'simulate', 'mock', 'fake', 'synthetic',
            'manual', 'implement', 'build', 'construct', 'make', 'produce',
            'random', 'artificial', 'placeholder', 'dummy', 'test'
        ]
        
        operation_lower = operation.lower()
        for blocked_op in blocked_operations:
            if blocked_op in operation_lower:
                self.log_blocked_attempt(operation, data_source, "PROHIBITED_OPERATION")
                print(f"🚫 BLOCKED: {operation} - Prohibited operation detected")
                return True
        
        return False
    
    def validate_real_api_data(self, data: any, source: str) -> bool:
        """Validate data actually came from real API calls"""
        
        if source not in self.AUTHORIZED_SOURCES:
            return False
            
        # Data must have API response characteristics
        if isinstance(data, dict):
            # Polygon data should have specific structure
            if source == 'polygon':
                required_fields = ['status', 'results'] # Typical Polygon response
                if not any(field in data for field in required_fields):
                    self.log_blocked_attempt("data_validation", source, "INVALID_API_STRUCTURE")
                    return False
            
            # FRED data should have specific structure  
            elif source == 'fred':
                required_fields = ['observations', 'series'] # Typical FRED response
                if not any(field in data for field in required_fields):
                    self.log_blocked_attempt("data_validation", source, "INVALID_API_STRUCTURE") 
                    return False
        
        return True
    
    def log_blocked_attempt(self, operation: str, source: str, reason: str):
        """Log blocked attempts"""
        blocked_attempt = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'attempted_source': source,
            'block_reason': reason,
            'authorized_sources': self.AUTHORIZED_SOURCES
        }
        
        self.blocked_attempts.append(blocked_attempt)
        
        # Save to lockdown log
        os.makedirs('compliance_logs', exist_ok=True)
        log_file = f"compliance_logs/data_lockdown_{datetime.now().strftime('%Y%m%d')}.json"
        
        try:
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    existing_blocks = json.load(f)
            else:
                existing_blocks = []
                
            existing_blocks.append(blocked_attempt)
            
            with open(log_file, 'w') as f:
                json.dump(existing_blocks, f, indent=2)
                
        except Exception as e:
            print(f"Error logging block: {e}")

# Global lockdown instance
LOCKDOWN = AbsoluteDataLockdown()

def enforce_data_lockdown(operation: str, data_source: str = None) -> bool:
    """Global function to enforce data lockdown"""
    return LOCKDOWN.block_all_other_data(operation, data_source)

def validate_api_data(data: any, source: str) -> bool:
    """Validate data came from real API"""
    return LOCKDOWN.validate_real_api_data(data, source)

def get_authorized_sources() -> list:
    """Get list of only authorized sources"""
    return LOCKDOWN.AUTHORIZED_SOURCES.copy()

if __name__ == "__main__":
    # Test the lockdown system
    lockdown = AbsoluteDataLockdown()
    
    print("Testing Absolute Data Lockdown...")
    
    # Test blocking unauthorized sources
    result = lockdown.block_all_other_data("training", "unauthorized_api")
    print(f"Unauthorized source blocked: {result}")
    
    # Test blocking prohibited operations  
    result = lockdown.block_all_other_data("create_synthetic_data", "polygon")
    print(f"Prohibited operation blocked: {result}")
    
    # Test allowing authorized source
    result = lockdown.block_all_other_data("fetch_data", "polygon")
    print(f"Authorized operation allowed: {not result}")
    
    print(f"Total blocked attempts: {len(lockdown.blocked_attempts)}")