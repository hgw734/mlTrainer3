"""
Immutable Audit Log System
Provides blockchain-style immutable audit trail for all system actions
"""

import hashlib
import json
import logging
import os
import sqlite3
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import fcntl

logger = logging.getLogger(__name__)

# Configuration
AUDIT_DB_PATH = "logs/audit.db"
AUDIT_BACKUP_PATH = "logs/audit_backup"

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)


class ImmutableAuditLog:
    """
    Immutable audit log with blockchain-style integrity
    Ensures all actions are permanently recorded and verifiable
    """

    def __init__(self, db_path: str = AUDIT_DB_PATH):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize audit log database with blockchain structure"""
        with sqlite3.connect(self.db_path) as conn:
            # Create audit log table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    action TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    details TEXT NOT NULL,
                    hash TEXT NOT NULL UNIQUE,
                    previous_hash TEXT,
                    block_number INTEGER NOT NULL,
                    nonce INTEGER NOT NULL,
                    signature TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create integrity checks table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS integrity_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    check_timestamp TEXT NOT NULL,
                    last_verified_block INTEGER,
                    verification_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    issues TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_action ON audit_log(action)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_actor ON audit_log(actor)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_log(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_block_number ON audit_log(block_number)")

            conn.commit()

    def log_action(self, action: str, actor: str, details: Dict[str, Any], signature: Optional[str] = None) -> str:
        """
        Log an action with blockchain-style immutability

        Args:
            action: The action being performed
            actor: Who performed the action
            details: Additional details about the action
            signature: Optional cryptographic signature

        Returns:
            Hash of the logged entry
        """
        try:
            # Get next block number
            block_number = self._get_next_block_number()
            previous_hash = self._get_last_hash()

            # Create entry data
            entry_data = {
                "timestamp": datetime.now().isoformat(),
                "action": action,
                "actor": actor,
                "details": details,
                "previous_hash": previous_hash,
                "block_number": block_number,
            }

            # Calculate hash with proof of work
            entry_hash, nonce = self._calculate_hash_with_pow(entry_data)

            # Insert into database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO audit_log 
                    (timestamp, action, actor, details, hash, previous_hash, block_number, nonce, signature)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry_data["timestamp"],
                    entry_data["action"],
                    entry_data["actor"],
                    json.dumps(entry_data["details"]),
                    entry_hash,
                    entry_data["previous_hash"],
                    entry_data["block_number"],
                    nonce,
                    signature
                ))
                conn.commit()

            # Backup entry
            self._backup_entry(entry_data, entry_hash)

            logger.info(f"Audit log entry created: {action} by {actor} (block {block_number})")
            return entry_hash

        except Exception as e:
            logger.error(f"Failed to log action: {e}")
            raise

    def _calculate_hash_with_pow(self, entry_data: Dict[str, Any], difficulty: int = 2) -> Tuple[str, int]:
        """
        Calculate hash with proof of work (simple implementation)
        
        Args:
            entry_data: The data to hash
            difficulty: Number of leading zeros required
            
        Returns:
            Tuple of (hash, nonce)
        """
        target = "0" * difficulty
        nonce = 0
        
        while True:
            entry_data["nonce"] = nonce
            data_str = json.dumps(entry_data, sort_keys=True)
            hash_value = hashlib.sha256(data_str.encode()).hexdigest()
            
            if hash_value.startswith(target):
                return hash_value, nonce
            
            nonce += 1

    def _get_last_hash(self) -> str:
        """Get hash of the last entry in the chain"""
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute("SELECT hash FROM audit_log ORDER BY block_number DESC LIMIT 1").fetchone()
            return result[0] if result else "0" * 64

    def _get_next_block_number(self) -> int:
        """Get the next block number"""
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute("SELECT MAX(block_number) FROM audit_log").fetchone()
            return (result[0] or 0) + 1

    def verify_integrity(self, start_block: int = 0, end_block: Optional[int] = None) -> Tuple[bool, List[str]]:
        """
        Verify the integrity of the audit log chain
        
        Args:
            start_block: Starting block number
            end_block: Ending block number (None for all)
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Build query
                query = "SELECT * FROM audit_log WHERE block_number >= ?"
                params = [start_block]
                
                if end_block is not None:
                    query += " AND block_number <= ?"
                    params.append(end_block)
                
                query += " ORDER BY block_number"
                
                entries = conn.execute(query, params).fetchall()
                
                if not entries:
                    return True, []

                # Verify each entry
                previous_hash = "0" * 64 if start_block == 0 else None
                
                for entry in entries:
                    (
                        entry_id,
                        timestamp,
                        action,
                        actor,
                        details_json,
                        hash_value,
                        prev_hash,
                        block_number,
                        nonce,
                        signature,
                    ) = entry

                    # Verify previous hash linkage
                    if previous_hash and prev_hash != previous_hash:
                        issues.append(
                            f"Block {block_number}: Previous hash mismatch. "
                            f"Expected {previous_hash}, got {prev_hash}"
                        )

                    # Verify hash calculation
                    entry_data = {
                        "timestamp": timestamp,
                        "action": action,
                        "actor": actor,
                        "details": json.loads(details_json),
                        "previous_hash": prev_hash,
                        "block_number": block_number,
                        "nonce": nonce,
                    }

                    calculated_hash = hashlib.sha256(json.dumps(entry_data, sort_keys=True).encode()).hexdigest()

                    if calculated_hash != hash_value:
                        issues.append(
                            f"Block {block_number}: Hash mismatch. "
                            f"Expected {calculated_hash}, got {hash_value}"
                        )

                    previous_hash = hash_value

                # Log integrity check
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT INTO integrity_checks
                        (check_timestamp, last_verified_block, verification_hash, status)
                        VALUES (?, ?, ?, ?)
                    """, (
                        datetime.now().isoformat(),
                        end_block or entries[-1][7],  # block_number
                        hashlib.sha256(str(issues).encode()).hexdigest(),
                        "VALID" if not issues else "INVALID",
                    ))

                return len(issues) == 0, issues

        except Exception as e:
            logger.error(f"Integrity verification failed: {e}")
            return False, [f"Verification error: {e}"]

    def query_logs(
        self,
        action: Optional[str] = None,
        actor: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query audit logs with filters"""
        query = "SELECT * FROM audit_log WHERE 1=1"
        params = []

        if action:
            query += " AND action = ?"
            params.append(action)

        if actor:
            query += " AND actor = ?"
            params.append(actor)

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())

        query += " ORDER BY block_number DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            results = conn.execute(query, params).fetchall()

            return [dict(row) for row in results]

    def get_statistics(self) -> Dict[str, Any]:
        """Get audit log statistics"""
        with sqlite3.connect(self.db_path) as conn:
            # Total entries
            total = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]

            # Actions by type
            actions = conn.execute("""
                SELECT action, COUNT(*) as count
                FROM audit_log
                GROUP BY action
                ORDER BY count DESC
            """).fetchall()

            # Most active actors
            actors = conn.execute("""
                SELECT actor, COUNT(*) as count
                FROM audit_log
                GROUP BY actor
                ORDER BY count DESC
                LIMIT 10
            """).fetchall()

            # Recent integrity checks
            checks = conn.execute("""
                SELECT * FROM integrity_checks
                ORDER BY id DESC
                LIMIT 5
            """).fetchall()

            return {
                "total_entries": total,
                "actions": dict(actions),
                "top_actors": dict(actors),
                "recent_integrity_checks": [
                    {"timestamp": check[1], "last_block": check[2], "status": check[4]} 
                    for check in checks
                ],
            }

    def _backup_entry(self, entry_data: Dict[str, Any], entry_hash: str):
        """Backup entry to secondary location"""
        backup_file = f"{AUDIT_BACKUP_PATH}.{datetime.now().strftime('%Y%m%d')}"
        
        with open(backup_file, "a") as f:
            # Use file locking for consistency
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump({
                    "entry": entry_data,
                    "hash": entry_hash,
                    "backed_up_at": datetime.now().isoformat()
                }, f)
                f.write("\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def export_blockchain(self, output_file: str):
        """Export entire audit log as blockchain"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            entries = conn.execute("SELECT * FROM audit_log ORDER BY block_number").fetchall()

            blockchain = {
                "chain": [dict(row) for row in entries],
                "exported_at": datetime.now().isoformat(),
                "total_blocks": len(entries),
                "integrity_valid": self.verify_integrity()[0],
            }

            with open(output_file, "w") as f:
                json.dump(blockchain, f, indent=2)


# Global instance
_audit_log = None


def get_audit_log() -> ImmutableAuditLog:
    """Get global audit log instance"""
    global _audit_log
    if _audit_log is None:
        _audit_log = ImmutableAuditLog()
    return _audit_log


# Convenience functions
def audit_action(action: str, actor: str, **details):
    """Convenience function to log an action"""
    return get_audit_log().log_action(action, actor, details)


def verify_audit_integrity():
    """Convenience function to verify audit log integrity"""
    return get_audit_log().verify_integrity()
