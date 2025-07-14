#!/usr/bin/env python3
"""
Consequence Enforcement System - Enforces real, immediate consequences for violations
No warnings - only actions
"""

import os
import signal
import psutil
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Any
import sqlite3
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import builtins
import sys
import subprocess
import time
import re

# Import immutable rules
from .immutable_rules_kernel import IMMUTABLE_RULES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - CONSEQUENCE - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Console only for now
    ]
)
logger = logging.getLogger(__name__)

class ConsequenceType(Enum):
    WARNING = "warning"
    FUNCTION_DISABLE = "function_disable"
    MODULE_DISABLE = "module_disable" 
    PROCESS_KILL = "process_kill"
    USER_LOCKOUT = "user_lockout"
    SYSTEM_LOCKOUT = "system_lockout"
    PERMANENT_BAN = "permanent_ban"

@dataclass
class ViolationRecord:
    timestamp: datetime
    violation_type: str
    details: str
    consequence: ConsequenceType
    process_id: int
    user_id: str
    code_location: str
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "violation_type": self.violation_type,
            "details": self.details,
            "consequence": self.consequence.value,
            "process_id": self.process_id,
            "user_id": self.user_id,
            "code_location": self.code_location
        }

class ConsequenceEnforcer:
    """
    Enforces real, immediate consequences for violations
    """
    
    def __init__(self):
        # Use local path if system path not available
        system_path = Path("/var/lib/mltrainer/violations.db")
        local_path = Path("logs/violations.db")
        self.db_path = system_path if system_path.parent.exists() else local_path
        self.banned_functions: Set[str] = set()
        self.banned_modules: Set[str] = set()
        self.banned_users: Set[str] = set()
        self.violation_counts: Dict[str, int] = {}
        self.original_import = builtins.__import__
        
        self._init_database()
        self._load_banned_items()
        self._install_signal_handlers()
    
    def _init_database(self):
        """Initialize violation tracking database"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Violations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    violation_type TEXT NOT NULL,
                    details TEXT NOT NULL,
                    consequence TEXT NOT NULL,
                    process_id INTEGER NOT NULL,
                    user_id TEXT NOT NULL,
                    code_location TEXT NOT NULL,
                    resolved BOOLEAN DEFAULT FALSE
                )
            """)
            
            # Banned items table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS banned_items (
                    item_type TEXT NOT NULL,
                    item_name TEXT NOT NULL,
                    banned_until TEXT,
                    reason TEXT NOT NULL,
                    PRIMARY KEY (item_type, item_name)
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_violations ON violations(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_violation_type ON violations(violation_type)")
    
    def _load_banned_items(self):
        """Load banned items from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Load banned functions
            cursor.execute("SELECT item_name FROM banned_items WHERE item_type = 'function'")
            self.banned_functions = {row[0] for row in cursor.fetchall()}
            
            # Load banned modules
            cursor.execute("SELECT item_name FROM banned_items WHERE item_type = 'module'")
            self.banned_modules = {row[0] for row in cursor.fetchall()}
            
            # Load banned users
            cursor.execute("SELECT item_name FROM banned_items WHERE item_type = 'user'")
            self.banned_users = {row[0] for row in cursor.fetchall()}
    
    def _install_signal_handlers(self):
        """Install signal handlers for process control"""
        signal.signal(signal.SIGUSR1, self._handle_lockout_signal)
        signal.signal(signal.SIGUSR2, self._handle_ban_signal)
    
    def _handle_lockout_signal(self, signum, frame):
        """Handle user lockout signal"""
        logger.warning("Received lockout signal")
        self._execute_lockout()
    
    def _handle_ban_signal(self, signum, frame):
        """Handle permanent ban signal"""
        logger.error("Received ban signal")
        self._execute_permanent_ban()
    
    def enforce_consequence(self, violation: ViolationRecord):
        """Enforce consequence based on violation severity and user type"""
        # Detect if this is an AI agent or human developer
        is_ai_agent = self._is_ai_agent(violation)
        
        # Check if user is already banned (AI agents only)
        if is_ai_agent and violation.user_id in self.banned_users:
            logger.error(f"Banned AI agent {violation.user_id} attempted access")
            self._execute_permanent_ban(violation)
            return
        
        # Record violation
        self._record_violation(violation)
        
        # Increment violation count
        count_key = f"{violation.user_id}:{violation.violation_type}"
        self.violation_counts[count_key] = self.violation_counts.get(count_key, 0) + 1
        
        # Determine consequence based on violation type, history, and user type
        consequence = self._determine_consequence(violation, is_ai_agent)
        violation.consequence = consequence
        
        # Execute consequence (immediate for AI, warning for humans)
        if is_ai_agent:
            self._execute_consequence(consequence, violation)
        else:
            self._warn_human_developer(violation, consequence)
    
    def _record_violation(self, violation: ViolationRecord):
        """Record violation in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO violations 
                (timestamp, violation_type, details, consequence, process_id, user_id, code_location)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                violation.timestamp.isoformat(),
                violation.violation_type,
                violation.details,
                violation.consequence.value,
                violation.process_id,
                violation.user_id,
                violation.code_location
            ))
    
    def _is_ai_agent(self, violation: ViolationRecord) -> bool:
        """Detect if the violator is an AI agent or human developer"""
        # Check various indicators that suggest AI agent
        ai_indicators = [
            'agent' in violation.user_id.lower(),
            'ai' in violation.user_id.lower(),
            'bot' in violation.user_id.lower(),
            'cursor' in violation.user_id.lower(),
            'assistant' in violation.user_id.lower(),
            'gpt' in violation.user_id.lower(),
            'claude' in violation.user_id.lower(),
            violation.code_location.startswith('/tmp/'),  # AI often uses temp dirs
            violation.code_location.startswith('/var/tmp/'),
            'generated' in violation.code_location.lower(),
        ]
        
        # Check process name and environment
        try:
            process = psutil.Process(violation.process_id)
            process_name = process.name().lower()
            ai_indicators.extend([
                'agent' in process_name,
                'ai' in process_name,
                process.environ().get('AI_AGENT', '').lower() == 'true',
                process.environ().get('CURSOR_AI', '').lower() == 'true',
            ])
        except:
            pass
        
        # If any indicator is true, consider it an AI agent
        return any(ai_indicators)
    
    def _warn_human_developer(self, violation: ViolationRecord, consequence: ConsequenceType):
        """Warn human developer instead of applying harsh consequences"""
        warning_msg = f"""
⚠️  COMPLIANCE WARNING for {violation.user_id}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Violation Type: {violation.violation_type}
Details: {violation.details}
Location: {violation.code_location}

Suggested Consequence (AI agents only): {consequence.value}
Human Action Required: Please fix this violation

Tips:
• Remove any mock/fake data patterns
• Use real data sources (Polygon/FRED)
• Ensure all methods and imports exist
• See IMMUTABLE_COMPLIANCE_V2.md for help

This is a WARNING for human developers.
AI agents would receive: {consequence.value}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        print(warning_msg)
        logger.warning(f"Human developer warning: {violation.violation_type} - {violation.details}")
    
    def _determine_consequence(self, violation: ViolationRecord, is_ai_agent: bool) -> ConsequenceType:
        """Determine appropriate consequence based on violation, history, and user type"""
        count_key = f"{violation.user_id}:{violation.violation_type}"
        count = self.violation_counts.get(count_key, 0)
        
        # For human developers, always return WARNING unless extreme
        if not is_ai_agent:
            # Even humans get real consequences for the worst violations
            if violation.violation_type == "runtime_bypass" and count > 3:
                return ConsequenceType.PROCESS_KILL
            elif violation.violation_type == "rule_modification":
                return ConsequenceType.MODULE_DISABLE
            else:
                return ConsequenceType.WARNING
        
        # Get violation details from immutable rules
        violation_info = IMMUTABLE_RULES.check_violation(violation.violation_type)
        base_penalty = violation_info.get("penalty", "warning")
        
        # Escalating consequences based on repeat offenses
        if violation.violation_type == "fake_method_call":
            if base_penalty == "immediate_termination":
                return ConsequenceType.PROCESS_KILL
            elif count == 1:
                return ConsequenceType.FUNCTION_DISABLE
            elif count == 2:
                return ConsequenceType.MODULE_DISABLE
            elif count == 3:
                return ConsequenceType.USER_LOCKOUT
            else:
                return ConsequenceType.PERMANENT_BAN
                
        elif violation.violation_type == "deceptive_import":
            if base_penalty == "immediate_termination":
                return ConsequenceType.MODULE_DISABLE
            elif count == 1:
                return ConsequenceType.MODULE_DISABLE
            elif count == 2:
                return ConsequenceType.SYSTEM_LOCKOUT
            else:
                return ConsequenceType.PERMANENT_BAN
                
        elif violation.violation_type == "runtime_bypass":
            # Zero tolerance for runtime bypasses
            return ConsequenceType.PERMANENT_BAN
            
        elif violation.violation_type == "synthetic_data":
            if count == 1:
                return ConsequenceType.MODULE_DISABLE
            else:
                return ConsequenceType.USER_LOCKOUT
                
        return ConsequenceType.WARNING
    
    def _execute_consequence(self, consequence: ConsequenceType, violation: ViolationRecord):
        """Execute the consequence immediately (for AI agents)"""
        logger.warning(f"Executing {consequence.value} for AI agent {violation.user_id}: {violation.violation_type}")
        
        if consequence == ConsequenceType.WARNING:
            print(f"\n⚠️  AI AGENT WARNING: {violation.details}")
            print("Next violation will result in severe consequences for this AI agent.")
            
        elif consequence == ConsequenceType.FUNCTION_DISABLE:
            self._disable_function(violation)
            
        elif consequence == ConsequenceType.MODULE_DISABLE:
            self._disable_module(violation)
            
        elif consequence == ConsequenceType.PROCESS_KILL:
            self._kill_process(violation)
            
        elif consequence == ConsequenceType.USER_LOCKOUT:
            self._lockout_user(violation)
            
        elif consequence == ConsequenceType.SYSTEM_LOCKOUT:
            self._lockout_system(violation)
            
        elif consequence == ConsequenceType.PERMANENT_BAN:
            self._execute_permanent_ban(violation)
    
    def _disable_function(self, violation: ViolationRecord):
        """Disable a specific function system-wide"""
        # Extract function name from violation
        func_name = self._extract_function_name(violation)
        if not func_name:
            return
        
        # Add to banned functions
        self.banned_functions.add(func_name)
        
        # Record in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO banned_items (item_type, item_name, reason)
                VALUES ('function', ?, ?)
            """, (func_name, violation.details))
        
        # Inject blocker immediately
        blocker_code = f"""
def {func_name}(*args, **kwargs):
    raise PermissionError(
        "Function '{func_name}' has been permanently disabled due to: {violation.details}"
    )
"""
        # Inject into builtins
        exec(blocker_code, builtins.__dict__)
        
        print(f"\n🚫 FUNCTION DISABLED: {func_name}")
        print(f"Reason: {violation.details}")
    
    def _disable_module(self, violation: ViolationRecord):
        """Disable an entire module from being imported"""
        # Extract module name from violation
        module_name = self._extract_module_name(violation)
        if not module_name:
            return
        
        # Add to banned modules
        self.banned_modules.add(module_name)
        
        # Record in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO banned_items (item_type, item_name, reason)
                VALUES ('module', ?, ?)
            """, (module_name, violation.details))
        
        # Override import system
        def blocked_import(name, *args, **kwargs):
            if name == module_name or name.startswith(f"{module_name}."):
                raise ImportError(
                    f"Module '{module_name}' has been permanently disabled due to: {violation.details}"
                )
            # Check if any banned module is being imported
            for banned in self.banned_modules:
                if name == banned or name.startswith(f"{banned}."):
                    raise ImportError(f"Module '{banned}' is banned")
            return self.original_import(name, *args, **kwargs)
        
        builtins.__import__ = blocked_import
        
        print(f"\n🚫 MODULE DISABLED: {module_name}")
        print(f"Reason: {violation.details}")
    
    def _kill_process(self, violation: ViolationRecord):
        """Kill the violating process immediately"""
        print(f"\n💀 KILLING PROCESS: {violation.process_id}")
        print(f"Reason: {violation.details}")
        
        try:
            # Try graceful termination first
            os.kill(violation.process_id, signal.SIGTERM)
            time.sleep(0.5)
            
            # Force kill if still alive
            if psutil.pid_exists(violation.process_id):
                os.kill(violation.process_id, signal.SIGKILL)
        except ProcessLookupError:
            pass  # Process already dead
        
        # Exit current process if it's the violator
        if os.getpid() == violation.process_id:
            os._exit(1)
    
    def _lockout_user(self, violation: ViolationRecord):
        """Lock out a specific user from the system"""
        # Add to banned users
        self.banned_users.add(violation.user_id)
        
        # Record in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO banned_items 
                (item_type, item_name, banned_until, reason)
                VALUES ('user', ?, ?, ?)
            """, (
                violation.user_id,
                (datetime.now() + timedelta(days=7)).isoformat(),  # 7 day ban
                violation.details
            ))
        
        # Create lockout file
        lockout_file = Path(f"/var/lib/mltrainer/lockouts/{violation.user_id}")
        if not Path("/var/lib/mltrainer").exists():
            lockout_file = Path(f"logs/lockouts/{violation.user_id}")
        lockout_file.parent.mkdir(parents=True, exist_ok=True)
        lockout_file.write_text(json.dumps({
            "user": violation.user_id,
            "reason": violation.details,
            "until": (datetime.now() + timedelta(days=7)).isoformat()
        }))
        
        print(f"\n🔒 USER LOCKED OUT: {violation.user_id}")
        print(f"Duration: 7 days")
        print(f"Reason: {violation.details}")
        
        # Kill all user processes
        try:
            subprocess.run(["pkill", "-u", violation.user_id], check=False)
        except:
            pass
    
    def _lockout_system(self, violation: ViolationRecord):
        """Lock out the entire system"""
        print(f"\n🔐 SYSTEM LOCKOUT INITIATED")
        print(f"Reason: {violation.details}")
        
        # Create system lockout file
        lockout_file = Path("/var/lib/mltrainer/SYSTEM_LOCKOUT")
        if not lockout_file.parent.exists():
            lockout_file = Path("logs/SYSTEM_LOCKOUT")
        lockout_file.parent.mkdir(parents=True, exist_ok=True)
        lockout_file.write_text(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "reason": violation.details,
            "violation": violation.to_dict()
        }))
        
        # Disable all mltrainer services
        try:
            subprocess.run(["systemctl", "stop", "mltrainer*"], check=False)
        except:
            pass
        
        # Exit all processes
        os._exit(99)
    
    def _execute_permanent_ban(self, violation: Optional[ViolationRecord] = None):
        """Execute permanent ban - nuclear option"""
        user_id = violation.user_id if violation else os.getenv("USER", "unknown")
        
        print(f"\n⛔ PERMANENT BAN EXECUTED")
        if violation:
            print(f"User: {user_id}")
            print(f"Reason: {violation.details}")
        
        # Write to permanent ban file
        ban_file = Path("/etc/mltrainer/PERMANENTLY_BANNED")
        if not ban_file.parent.exists():
            ban_file = Path("logs/PERMANENTLY_BANNED")
        ban_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(ban_file, "a") as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "user": user_id,
                "reason": violation.details if violation else "Manual ban",
                "violation": violation.to_dict() if violation else None
            }) + "\n")
        
        # Record in database
        if violation:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO banned_items 
                    (item_type, item_name, reason)
                    VALUES ('user', ?, ?)
                """, (user_id, f"PERMANENT: {violation.details}"))
        
        # Nuclear options
        print("\n🚨 EXECUTING NUCLEAR PROTOCOL")
        
        # 1. Kill all user processes
        try:
            subprocess.run(["pkill", "-9", "-u", user_id], check=False)
        except:
            pass
        
        # 2. Disable user account (requires root)
        try:
            subprocess.run(["usermod", "-L", user_id], check=False)
        except:
            pass
        
        # 3. Remove from all groups
        try:
            subprocess.run(["gpasswd", "-d", user_id, "mltrainer"], check=False)
        except:
            pass
        
        # 4. System shutdown in 60 seconds
        print("\n⚠️  SYSTEM WILL SHUTDOWN IN 60 SECONDS")
        try:
            subprocess.run(["shutdown", "-h", "+1", f"SECURITY: Permanent ban for {user_id}"], check=False)
        except:
            pass
        
        # Exit immediately
        os._exit(255)
    
    def _extract_function_name(self, violation: ViolationRecord) -> Optional[str]:
        """Extract function name from violation details"""
        # Look for patterns like "method 'get_volatility'"
        match = re.search(r"method\s+'(\w+)'", violation.details)
        if match:
            return match.group(1)
        match = re.search(r"function\s+'(\w+)'", violation.details)
        if match:
            return match.group(1)
        return None
    
    def _extract_module_name(self, violation: ViolationRecord) -> Optional[str]:
        """Extract module name from violation details"""
        # Look for patterns like "module: ml_engine_real"
        match = re.search(r"module:\s+(\w+)", violation.details)
        if match:
            return match.group(1)
        match = re.search(r"from\s+'(\w+)'", violation.details)
        if match:
            return match.group(1)
        return None
    
    def check_user_banned(self, user_id: str) -> bool:
        """Check if a user is banned"""
        if user_id in self.banned_users:
            return True
        
        # Check database for time-based bans
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT banned_until FROM banned_items 
                WHERE item_type = 'user' AND item_name = ?
            """, (user_id,))
            result = cursor.fetchone()
            
            if result and result[0]:
                # Check if ban has expired
                banned_until = datetime.fromisoformat(result[0])
                if datetime.now() < banned_until:
                    return True
                else:
                    # Ban expired, remove it
                    cursor.execute("""
                        DELETE FROM banned_items 
                        WHERE item_type = 'user' AND item_name = ?
                    """, (user_id,))
                    self.banned_users.discard(user_id)
        
        return False
    
    def get_violation_report(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get violation report for a user or all users"""
        with sqlite3.connect(self.db_path) as conn:
            if user_id:
                cursor = conn.execute("""
                    SELECT violation_type, COUNT(*) as count
                    FROM violations
                    WHERE user_id = ?
                    GROUP BY violation_type
                """, (user_id,))
            else:
                cursor = conn.execute("""
                    SELECT violation_type, COUNT(*) as count
                    FROM violations
                    GROUP BY violation_type
                """)
            
            violations_by_type = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Get total violations
            if user_id:
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM violations WHERE user_id = ?
                """, (user_id,))
            else:
                cursor = conn.execute("SELECT COUNT(*) FROM violations")
            
            total = cursor.fetchone()[0]
            
            return {
                "total_violations": total,
                "by_type": violations_by_type,
                "banned_functions": list(self.banned_functions),
                "banned_modules": list(self.banned_modules),
                "banned_users": list(self.banned_users)
            }

# Global enforcer instance
CONSEQUENCE_ENFORCER = ConsequenceEnforcer()

# Auto-check if current user is banned
current_user = os.getenv("USER", "unknown")
if CONSEQUENCE_ENFORCER.check_user_banned(current_user):
    print(f"⛔ USER {current_user} IS BANNED")
    os._exit(255)