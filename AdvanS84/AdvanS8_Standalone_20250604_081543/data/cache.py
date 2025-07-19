import sqlite3
import json
import pickle
import time
from typing import Any, Dict, Optional, List
import logging
import os
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)

class DataCache:
    """
    High-performance data caching system for institutional-grade
    stock scanning with TTL support and efficient storage.
    """
    
    def __init__(self, db_path: str = "data/cache.db"):
        """
        Initialize data cache with SQLite backend
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.hit_count = 0
        self.miss_count = 0
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Clean expired entries on startup
        self._cleanup_expired()
        
        logger.info(f"DataCache initialized with database: {db_path}")
    
    def _init_database(self):
        """Initialize cache database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create cache table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS cache (
                        key TEXT PRIMARY KEY,
                        value BLOB,
                        expires_at REAL,
                        created_at REAL,
                        hit_count INTEGER DEFAULT 0,
                        data_type TEXT,
                        size_bytes INTEGER
                    )
                """)
                
                # Create index for cleanup operations
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_expires_at 
                    ON cache(expires_at)
                """)
                
                # Create metadata table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS cache_metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT
                    )
                """)
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """
        Store value in cache with TTL
        
        Args:
            key: Cache key
            value: Value to store
            ttl: Time to live in seconds (default 1 hour)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            expires_at = time.time() + ttl
            created_at = time.time()
            
            # Serialize value based on type
            if isinstance(value, pd.DataFrame):
                serialized_value = pickle.dumps(value)
                data_type = "dataframe"
            elif isinstance(value, dict):
                serialized_value = json.dumps(value).encode('utf-8')
                data_type = "dict"
            elif isinstance(value, (list, tuple)):
                serialized_value = json.dumps(list(value)).encode('utf-8')
                data_type = "list"
            else:
                serialized_value = pickle.dumps(value)
                data_type = "pickle"
            
            size_bytes = len(serialized_value)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO cache 
                    (key, value, expires_at, created_at, hit_count, data_type, size_bytes)
                    VALUES (?, ?, ?, ?, 0, ?, ?)
                """, (key, serialized_value, expires_at, created_at, data_type, size_bytes))
                
                conn.commit()
            
            logger.debug(f"Cached value for key: {key} (TTL: {ttl}s, Size: {size_bytes} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"Cache set failed for key {key}: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        try:
            current_time = time.time()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT value, expires_at, data_type 
                    FROM cache 
                    WHERE key = ? AND expires_at > ?
                """, (key, current_time))
                
                result = cursor.fetchone()
                
                if result is None:
                    self.miss_count += 1
                    logger.debug(f"Cache miss for key: {key}")
                    return None
                
                value_blob, expires_at, data_type = result
                
                # Increment hit count
                cursor.execute("""
                    UPDATE cache 
                    SET hit_count = hit_count + 1 
                    WHERE key = ?
                """, (key,))
                
                conn.commit()
            
            # Deserialize value based on type
            if data_type == "dataframe":
                value = pickle.loads(value_blob)
            elif data_type == "dict":
                value = json.loads(value_blob.decode('utf-8'))
            elif data_type == "list":
                value = json.loads(value_blob.decode('utf-8'))
            else:  # pickle
                value = pickle.loads(value_blob)
            
            self.hit_count += 1
            logger.debug(f"Cache hit for key: {key}")
            return value
            
        except Exception as e:
            logger.error(f"Cache get failed for key {key}: {e}")
            self.miss_count += 1
            return None
    
    def delete(self, key: str) -> bool:
        """
        Delete value from cache
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("DELETE FROM cache WHERE key = ?", (key,))
                deleted_count = cursor.rowcount
                
                conn.commit()
            
            logger.debug(f"Deleted cache entry for key: {key}")
            return deleted_count > 0
            
        except Exception as e:
            logger.error(f"Cache delete failed for key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """
        Check if key exists and is not expired
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists and is valid, False otherwise
        """
        try:
            current_time = time.time()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 1 FROM cache 
                    WHERE key = ? AND expires_at > ?
                """, (key, current_time))
                
                return cursor.fetchone() is not None
                
        except Exception as e:
            logger.error(f"Cache exists check failed for key {key}: {e}")
            return False
    
    def clear(self) -> bool:
        """
        Clear all cache entries
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("DELETE FROM cache")
                deleted_count = cursor.rowcount
                
                conn.commit()
            
            logger.info(f"Cleared {deleted_count} cache entries")
            return True
            
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
            return False
    
    def _cleanup_expired(self) -> int:
        """
        Remove expired cache entries
        
        Returns:
            Number of entries removed
        """
        try:
            current_time = time.time()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("DELETE FROM cache WHERE expires_at <= ?", (current_time,))
                deleted_count = cursor.rowcount
                
                conn.commit()
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} expired cache entries")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total entries
                cursor.execute("SELECT COUNT(*) FROM cache")
                total_entries = cursor.fetchone()[0]
                
                # Expired entries
                current_time = time.time()
                cursor.execute("SELECT COUNT(*) FROM cache WHERE expires_at <= ?", (current_time,))
                expired_entries = cursor.fetchone()[0]
                
                # Total size
                cursor.execute("SELECT SUM(size_bytes) FROM cache")
                total_size = cursor.fetchone()[0] or 0
                
                # Most hit entries
                cursor.execute("""
                    SELECT key, hit_count 
                    FROM cache 
                    WHERE expires_at > ? 
                    ORDER BY hit_count DESC 
                    LIMIT 5
                """, (current_time,))
                top_entries = cursor.fetchall()
                
            hit_rate = (self.hit_count / (self.hit_count + self.miss_count) * 100) if (self.hit_count + self.miss_count) > 0 else 0
            
            return {
                'total_entries': total_entries,
                'active_entries': total_entries - expired_entries,
                'expired_entries': expired_entries,
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
                'hit_rate_percent': hit_rate,
                'top_entries': top_entries
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Alias for get_stats() for backward compatibility"""
        return self.get_stats()
    
    def get_hit_rate(self) -> float:
        """
        Get cache hit rate percentage
        
        Returns:
            Hit rate as percentage (0-100)
        """
        total_requests = self.hit_count + self.miss_count
        if total_requests == 0:
            return 0.0
        
        return (self.hit_count / total_requests) * 100
    
    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate cache entries matching a pattern
        
        Args:
            pattern: SQL LIKE pattern (e.g., "market_data_%")
            
        Returns:
            Number of entries invalidated
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("DELETE FROM cache WHERE key LIKE ?", (pattern,))
                deleted_count = cursor.rowcount
                
                conn.commit()
            
            logger.info(f"Invalidated {deleted_count} cache entries matching pattern: {pattern}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Pattern invalidation failed for {pattern}: {e}")
            return 0
    
    def set_metadata(self, key: str, value: str) -> bool:
        """
        Set metadata value
        
        Args:
            key: Metadata key
            value: Metadata value
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO cache_metadata (key, value)
                    VALUES (?, ?)
                """, (key, value))
                
                conn.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Metadata set failed for key {key}: {e}")
            return False
    
    def get_metadata(self, key: str) -> Optional[str]:
        """
        Get metadata value
        
        Args:
            key: Metadata key
            
        Returns:
            Metadata value or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT value FROM cache_metadata WHERE key = ?", (key,))
                result = cursor.fetchone()
                
                return result[0] if result else None
                
        except Exception as e:
            logger.error(f"Metadata get failed for key {key}: {e}")
            return None
    
    def optimize_database(self) -> bool:
        """
        Optimize database performance
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Clean up expired entries
                self._cleanup_expired()
                
                # Vacuum database to reclaim space
                cursor.execute("VACUUM")
                
                # Analyze to update statistics
                cursor.execute("ANALYZE")
                
                conn.commit()
            
            logger.info("Database optimization completed")
            return True
            
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
            return False
    
    def export_cache_data(self, filepath: str) -> bool:
        """
        Export cache data to JSON file
        
        Args:
            filepath: Export file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            current_time = time.time()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT key, data_type, created_at, expires_at, hit_count, size_bytes
                    FROM cache 
                    WHERE expires_at > ?
                    ORDER BY hit_count DESC
                """, (current_time,))
                
                entries = []
                for row in cursor.fetchall():
                    key, data_type, created_at, expires_at, hit_count, size_bytes = row
                    entries.append({
                        'key': key,
                        'data_type': data_type,
                        'created_at': datetime.fromtimestamp(created_at).isoformat(),
                        'expires_at': datetime.fromtimestamp(expires_at).isoformat(),
                        'hit_count': hit_count,
                        'size_bytes': size_bytes
                    })
            
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'total_entries': len(entries),
                'cache_stats': self.get_stats(),
                'entries': entries
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Cache data exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Cache export failed: {e}")
            return False
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup expired entries"""
        self._cleanup_expired()
