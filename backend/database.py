"""
Database Layer
==============

Provides database abstraction with support for SQLite (development) and PostgreSQL (production).
Includes Redis caching layer.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from contextlib import contextmanager
import sqlite3
import asyncio
from dataclasses import dataclass, asdict

# Optional imports for production
try:
    import asyncpg
    import redis
    import aioredis

    POSTGRES_AVAILABLE = True
    REDIS_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class Trial:
    trial_id: str
    status: str
    created_at: str
    updated_at: str
    goal_context: Dict[str, Any]
    actions: List[str]
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class ChatMessage:
    message_id: str
    role: str
    content: str
    timestamp: str
    metadata: Dict[str, Any]


@dataclass
class ModelResult:
    result_id: str
    model_id: str
    symbol: str
    metrics: Dict[str, Any]
    created_at: str


class DatabaseManager:
    """
    Unified database manager supporting multiple backends.
    """

    def __init__(self, database_url: str = None):
        self.database_url = database_url or os.getenv("DATABASE_URL", "sqlite:///mltrainer.db")
        self.is_postgres = self.database_url.startswith("postgresql://")
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

        # Initialize connections
        self._init_database()
        self._init_cache()

        logger.info(f"Database initialized: {'PostgreSQL' if self.is_postgres else 'SQLite'}")

    def _init_database(self):
        """Initialize database connection"""
        if self.is_postgres and POSTGRES_AVAILABLE:
            # PostgreSQL setup handled in async context
            self.pool = None
        else:
            # SQLite setup
            self.db_path = self.database_url.replace("sqlite:///", "")
            os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)
            self._create_sqlite_tables()

    def _init_cache(self):
        """Initialize cache connection"""
        if REDIS_AVAILABLE:
            try:
                self.redis = redis.from_url(self.redis_url, decode_responses=True)
                self.redis.ping()
                self.cache_available = True
                logger.info("Redis cache connected")
            except Exception as e:
                logger.warning(f"Redis not available: {e}")
                self.cache_available = False
        else:
            self.cache_available = False

    def _create_sqlite_tables(self):
        """Create SQLite tables"""
        with self._get_sqlite_connection() as conn:
            cursor = conn.cursor()

            # Trials table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS trials (
                trial_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                goal_context TEXT,
                actions TEXT,
                results TEXT,
                error TEXT
                )
                """
            )

            # Chat messages table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                message_id TEXT PRIMARY KEY,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT,
                importance REAL DEFAULT 0.5
                )
                """
            )

            # Model results table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS model_results (
                result_id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                symbol TEXT,
                metrics TEXT,
                created_at TEXT NOT NULL,
                trial_id TEXT,
                FOREIGN KEY (trial_id) REFERENCES trials(trial_id)
                )
                """
            )

            # Goals table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS goals (
                goal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                goal TEXT NOT NULL,
                created_at TEXT NOT NULL,
                is_active BOOLEAN DEFAULT 1
                )
                """
            )

            # Compliance events table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS compliance_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                details TEXT,
                timestamp TEXT NOT NULL
                )
                """
            )

            conn.commit()

    @contextmanager
    def _get_sqlite_connection(self):
        """Get SQLite connection context manager"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # Trial Management
    async def create_trial(self, trial: Trial) -> bool:
        """Create a new trial"""
        if self.is_postgres and POSTGRES_AVAILABLE:
            return await self._create_trial_postgres(trial)
        else:
            return self._create_trial_sqlite(trial)

    def _create_trial_sqlite(self, trial: Trial) -> bool:
        """Create trial in SQLite"""
        try:
            with self._get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO trials (trial_id, status, created_at, updated_at,
                    goal_context, actions, results, error)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trial.trial_id,
                        trial.status,
                        trial.created_at,
                        trial.updated_at,
                        json.dumps(trial.goal_context),
                        json.dumps(trial.actions),
                        json.dumps(trial.results) if trial.results else None,
                        trial.error,
                    ),
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to create trial: {e}")
            return False

    async def get_trial(self, trial_id: str) -> Optional[Trial]:
        """Get trial by ID"""
        # Check cache first
        if self.cache_available:
            cached = self._get_cached_trial(trial_id)
            if cached:
                return cached

        if self.is_postgres and POSTGRES_AVAILABLE:
            return await self._get_trial_postgres(trial_id)
        else:
            return self._get_trial_sqlite(trial_id)

    def _get_trial_sqlite(self, trial_id: str) -> Optional[Trial]:
        """Get trial from SQLite"""
        try:
            with self._get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM trials WHERE trial_id = ?", (trial_id,))
                row = cursor.fetchone()

                if row:
                    trial = Trial(
                        trial_id=row["trial_id"],
                        status=row["status"],
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                        goal_context=json.loads(row["goal_context"]) if row["goal_context"] else {},
                        actions=json.loads(row["actions"]) if row["actions"] else [],
                        results=json.loads(row["results"]) if row["results"] else None,
                        error=row["error"],
                    )

                    # Cache for next time
                    if self.cache_available:
                        self._cache_trial(trial)

                    return trial
        except Exception as e:
            logger.error(f"Failed to get trial: {e}")
            return None

    async def update_trial(self, trial_id: str, **updates) -> bool:
        """Update trial status"""
        updates["updated_at"] = datetime.now().isoformat()

        if self.is_postgres and POSTGRES_AVAILABLE:
            return await self._update_trial_postgres(trial_id, **updates)
        else:
            return self._update_trial_sqlite(trial_id, **updates)

    def _update_trial_sqlite(self, trial_id: str, **updates) -> bool:
        """Update trial in SQLite"""
        try:
            with self._get_sqlite_connection() as conn:
                cursor = conn.cursor()

                # Build update query
                set_clauses = []
                values = []
                for key, value in list(updates.items()):
                    if key in ["status", "results", "error", "updated_at"]:
                        set_clauses.append(f"{key} = ?")
                        if key == "results" and isinstance(value, dict):
                            values.append(json.dumps(value))
                        else:
                            values.append(value)

                values.append(trial_id)

                query = f"UPDATE trials SET {', '.join(set_clauses)} WHERE trial_id = ?"
                cursor.execute(query, values)
                conn.commit()

                # Invalidate cache
                if self.cache_available:
                    self._invalidate_trial_cache(trial_id)

                return True
        except Exception as e:
            logger.error(f"Failed to update trial: {e}")
            return False

    # Chat Message Management
    async def save_chat_message(self, message: ChatMessage) -> bool:
        """Save chat message"""
        if self.is_postgres and POSTGRES_AVAILABLE:
            return await self._save_chat_message_postgres(message)
        else:
            return self._save_chat_message_sqlite(message)

    def _save_chat_message_sqlite(self, message: ChatMessage) -> bool:
        """Save chat message to SQLite"""
        try:
            with self._get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO chat_messages (message_id, role, content, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        message.message_id,
                        message.role,
                        message.content,
                        message.timestamp,
                        json.dumps(message.metadata),
                    ),
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save chat message: {e}")
            return False

    async def get_chat_history(self, limit: int = 50) -> List[ChatMessage]:
        """Get recent chat history"""
        if self.is_postgres and POSTGRES_AVAILABLE:
            return await self._get_chat_history_postgres(limit)
        else:
            return self._get_chat_history_sqlite(limit)

    def _get_chat_history_sqlite(self, limit: int) -> List[ChatMessage]:
        """Get chat history from SQLite"""
        messages = []
        try:
            with self._get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM chat_messages
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (limit,),
                )

                for row in cursor.fetchall():
                    messages.append(
                        ChatMessage(
                            message_id=row["message_id"],
                            role=row["role"],
                            content=row["content"],
                            timestamp=row["timestamp"],
                            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                        )
                    )
        except Exception as e:
            logger.error(f"Failed to get chat history: {e}")
            return messages

    # Model Results Management
    async def save_model_result(self, result: ModelResult) -> bool:
        """Save model training/execution result"""
        if self.is_postgres and POSTGRES_AVAILABLE:
            return await self._save_model_result_postgres(result)
        else:
            return self._save_model_result_sqlite(result)

    def _save_model_result_sqlite(self, result: ModelResult) -> bool:
        """Save model result to SQLite"""
        try:
            with self._get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO model_results (result_id, model_id, symbol, metrics, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (result.result_id, result.model_id, result.symbol, json.dumps(result.metrics), result.created_at),
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save model result: {e}")
            return False

    # Goal Management
    async def save_goal(self, goal: str) -> bool:
        """Save a new goal"""
        if self.is_postgres and POSTGRES_AVAILABLE:
            return await self._save_goal_postgres(goal)
        else:
            return self._save_goal_sqlite(goal)

    def _save_goal_sqlite(self, goal: str) -> bool:
        """Save goal to SQLite"""
        try:
            with self._get_sqlite_connection() as conn:
                cursor = conn.cursor()

                # Deactivate previous goals
                cursor.execute("UPDATE goals SET is_active = 0 WHERE is_active = 1")

                # Insert new goal
                cursor.execute(
                    """
                    INSERT INTO goals (goal, created_at, is_active)
                    VALUES (?, ?, 1)
                    """,
                    (goal, datetime.now().isoformat()),
                )

                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save goal: {e}")
            return False

    # Cache Management
    def _cache_trial(self, trial: Trial):
        """Cache trial in Redis"""
        if self.cache_available:
            try:
                key = f"trial:{trial.trial_id}"
                self.redis.setex(key, 3600, json.dumps(asdict(trial)))  # 1 hour TTL
            except Exception as e:
                logger.warning(f"Failed to cache trial: {e}")

    def _get_cached_trial(self, trial_id: str) -> Optional[Trial]:
        """Get trial from cache"""
        if self.cache_available:
            try:
                key = f"trial:{trial_id}"
                data = self.redis.get(key)
                if data:
                    trial_dict = json.loads(data)
                    return Trial(**trial_dict)
            except Exception as e:
                logger.warning(f"Failed to get cached trial: {e}")
                return None

    def _invalidate_trial_cache(self, trial_id: str):
        """Invalidate trial cache"""
        if self.cache_available:
            try:
                key = f"trial:{trial_id}"
                self.redis.delete(key)
            except Exception as e:
                logger.warning(f"Failed to invalidate cache: {e}")

    # PostgreSQL implementations (stubs for when asyncpg is available)
    async def _create_trial_postgres(self, trial: Trial) -> bool:
        """Create trial in PostgreSQL"""
        # Implementation when asyncpg is available
        logger.warning("PostgreSQL support not fully implemented")
        return self._create_trial_sqlite(trial)

    async def _get_trial_postgres(self, trial_id: str) -> Optional[Trial]:
        """Get trial from PostgreSQL"""
        logger.warning("PostgreSQL support not fully implemented")
        return self._get_trial_sqlite(trial_id)

    # Migration utilities
    def migrate_from_files(self, logs_dir: str = "logs"):
        """Migrate data from file-based storage to database"""
        migrated = {"trials": 0, "messages": 0, "goals": 0}

        # Migrate trials
        trials_dir = os.path.join(logs_dir, "trials")
        if os.path.exists(trials_dir):
            for filename in os.listdir(trials_dir):
                if filename.endswith(".json"):
                    filepath = os.path.join(trials_dir, filename)
                    try:
                        with open(filepath, "r") as f:
                            trial_data = json.load(f)

                            trial = Trial(
                                trial_id=trial_data["trial_id"],
                                status=trial_data["status"],
                                created_at=trial_data["created_at"],
                                updated_at=trial_data.get("last_update", trial_data["created_at"]),
                                goal_context=trial_data.get("goal_context", {}),
                                actions=trial_data.get("actions", []),
                                results=trial_data.get("results"),
                                error=trial_data.get("error"),
                            )

                            if self._create_trial_sqlite(trial):
                                migrated["trials"] += 1
                    except Exception as e:
                        logger.error(f"Failed to migrate trial {filename}: {e}")

        # Migrate chat history
        chat_file = os.path.join(logs_dir, "unified_chat_history.json")
        if os.path.exists(chat_file):
            try:
                with open(chat_file, "r") as f:
                    messages = json.load(f)

                    for msg in messages:
                        message = ChatMessage(
                            message_id=msg.get("id", f"msg_{msg['timestamp']}"),
                            role=msg["role"],
                            content=msg["content"],
                            timestamp=msg["timestamp"],
                            metadata=msg.get("metadata", {}),
                        )

                        if self._save_chat_message_sqlite(message):
                            migrated["messages"] += 1
            except Exception as e:
                logger.error(f"Failed to migrate chat history: {e}")

        logger.info(f"Migration complete: {migrated}")
        return migrated


# Singleton instance
_db_manager = None


def get_database_manager() -> DatabaseManager:
    """Get the database manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager
