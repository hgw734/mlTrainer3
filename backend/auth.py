"""
Authentication System
====================

Provides JWT-based authentication for the unified mlTrainer system.
"""

import os
import jwt
import bcrypt
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import sqlite3
from contextlib import contextmanager
import json

logger = logging.getLogger(__name__)

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Security
security = HTTPBearer()


# Pydantic models
class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    full_name: Optional[str] = None


    class UserLogin(BaseModel):
        username: str
        password: str


        class Token(BaseModel):
            access_token: str
            refresh_token: str
            token_type: str = "bearer"


            class User(BaseModel):
                user_id: int
                username: str
                email: str
                full_name: Optional[str] = None
                is_active: bool = True
                is_admin: bool = False
                created_at: str


                class AuthManager:
                    """
                    Manages user authentication and authorization.
                    """

                    def __init__(self, db_path: str = "mltrainer_auth.db"):
                        self.db_path = db_path
                        self._init_database()

                        def _init_database(self):
                            """Initialize authentication database"""
                            with self._get_connection() as conn:
                                cursor = conn.cursor()

                                # Users table
                                cursor.execute(
                                """
                                CREATE TABLE IF NOT EXISTS users (
                                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                                username TEXT UNIQUE NOT NULL,
                                email TEXT UNIQUE NOT NULL,
                                password_hash TEXT NOT NULL,
                                full_name TEXT,
                                is_active BOOLEAN DEFAULT 1,
                                is_admin BOOLEAN DEFAULT 0,
                                created_at TEXT NOT NULL,
                                last_login TEXT
                                )
                                """
                                )

                                # Sessions table for tracking active sessions
                                cursor.execute(
                                """
                                CREATE TABLE IF NOT EXISTS sessions (
                                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                                user_id INTEGER NOT NULL,
                                refresh_token TEXT UNIQUE NOT NULL,
                                expires_at TEXT NOT NULL,
                                created_at TEXT NOT NULL,
                                FOREIGN KEY (user_id) REFERENCES users(user_id)
                                )
                                """
                                )

                                # User preferences table
                                cursor.execute(
                                """
                                CREATE TABLE IF NOT EXISTS user_preferences (
                                user_id INTEGER PRIMARY KEY,
                                theme TEXT DEFAULT 'light',
                                notifications_enabled BOOLEAN DEFAULT 1,
                                default_models TEXT,
                                FOREIGN KEY (user_id) REFERENCES users(user_id)
                                )
                                """
                                )

                                conn.commit()

                                @contextmanager
                                def _get_connection(self):
                                    """Get database connection"""
                                    conn = sqlite3.connect(self.db_path)
                                    conn.row_factory = sqlite3.Row
                                    try:
                                        yield conn
                                    finally:
                                        conn.close()

    def create_user(self, user_data: UserCreate) -> Optional[User]:
        """Create a new user"""
        try:
            # Hash password
            password_hash = bcrypt.hashpw(user_data.password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Check if user exists
                cursor.execute(
                    "SELECT * FROM users WHERE username = ? OR email = ?", (user_data.username, user_data.email)
                )
                if cursor.fetchone():
                    return None

                # Insert new user
                cursor.execute(
                    """
                    INSERT INTO users (username, email, password_hash, full_name, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        user_data.username,
                        user_data.email,
                        password_hash,
                        user_data.full_name,
                        datetime.now().isoformat(),
                    ),
                )

                user_id = cursor.lastrowid

                # Create default preferences
                cursor.execute(
                    """
                    INSERT INTO user_preferences (user_id)
                    VALUES (?)
                    """,
                    (user_id,),
                )

                conn.commit()

                # Return user object
                return User(
                    user_id=user_id,
                    username=user_data.username,
                    email=user_data.email,
                    full_name=user_data.full_name,
                    created_at=datetime.now().isoformat(),
                )

        except Exception as e:
            logger.error(f"Failed to create user: {e}")
            return None

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Get user
                cursor.execute("SELECT * FROM users WHERE username = ? AND is_active = 1", (username,))
                user_row = cursor.fetchone()

                if not user_row:
                    return None

                # Verify password
                if not bcrypt.checkpw(password.encode("utf-8"), user_row["password_hash"].encode("utf-8")):
                    return None

                # Update last login
                cursor.execute(
                    "UPDATE users SET last_login = ? WHERE user_id = ?",
                    (datetime.now().isoformat(), user_row["user_id"]),
                )
                conn.commit()

                # Return user object
                return User(
                    user_id=user_row["user_id"],
                    username=user_row["username"],
                    email=user_row["email"],
                    full_name=user_row["full_name"],
                    is_active=bool(user_row["is_active"]),
                    is_admin=bool(user_row["is_admin"]),
                    created_at=user_row["created_at"],
                )

        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return None

    def create_tokens(self, user: User) -> Token:
        """Create access and refresh tokens for user"""
        # Access token payload
        access_payload = {
            "user_id": user.user_id,
            "username": user.username,
            "is_admin": user.is_admin,
            "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        }

        # Refresh token payload
        refresh_payload = {
            "user_id": user.user_id,
            "username": user.username,
            "exp": datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
        }

        # Create tokens
        access_token = jwt.encode(access_payload, SECRET_KEY, algorithm=ALGORITHM)
        refresh_token = jwt.encode(refresh_payload, SECRET_KEY, algorithm=ALGORITHM)

        # Store refresh token
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO sessions (user_id, refresh_token, expires_at, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (user.user_id, refresh_token, refresh_payload["exp"].isoformat(), datetime.now().isoformat()),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to store session: {e}")

        return Token(access_token=access_token, refresh_token=refresh_token)

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def get_current_user(self, credentials: HTTPAuthorizationCredentials = Security(security)) -> User:
        """Get current user from JWT token"""
        token = credentials.credentials
        payload = self.verify_token(token)

        if not payload:
            raise HTTPException(status_code=401, detail="Invalid or expired token")

        # Get user from database
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM users WHERE user_id = ? AND is_active = 1", (payload["user_id"],))
                user_row = cursor.fetchone()

                if not user_row:
                    raise HTTPException(status_code=401, detail="User not found")

                return User(
                    user_id=user_row["user_id"],
                    username=user_row["username"],
                    email=user_row["email"],
                    full_name=user_row["full_name"],
                    is_active=bool(user_row["is_active"]),
                    is_admin=bool(user_row["is_admin"]),
                    created_at=user_row["created_at"],
                )

        except Exception as e:
            logger.error(f"Failed to get user: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """Refresh access token using refresh token"""
        payload = self.verify_token(refresh_token)

        if not payload:
            return None

        # Verify refresh token exists in database
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM sessions WHERE refresh_token = ? AND expires_at > ?",
                    (refresh_token, datetime.now().isoformat()),
                )

                if not cursor.fetchone():
                    return None

                # Create new access token
                access_payload = {
                    "user_id": payload["user_id"],
                    "username": payload["username"],
                    "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
                }

                return jwt.encode(access_payload, SECRET_KEY, algorithm=ALGORITHM)

        except Exception as e:
            logger.error(f"Failed to refresh token: {e}")
            return None

    def logout(self, user_id: int, refresh_token: str) -> bool:
        """Logout user by invalidating refresh token"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM sessions WHERE user_id = ? AND refresh_token = ?", (user_id, refresh_token))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to logout: {e}")
            return False

    def get_user_preferences(self, user_id: int) -> Dict[str, Any]:
        """Get user preferences"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM user_preferences WHERE user_id = ?", (user_id,))
                row = cursor.fetchone()

                if row:
                    return {
                        "theme": row["theme"],
                        "notifications_enabled": bool(row["notifications_enabled"]),
                        "default_models": json.loads(row["default_models"]) if row["default_models"] else [],
                    }

                return {"theme": "light", "notifications_enabled": True, "default_models": []}

        except Exception as e:
            logger.error(f"Failed to get preferences: {e}")
            return {}

    def update_user_preferences(self, user_id: int, preferences: Dict[str, Any]) -> bool:
        """Update user preferences"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Build update query
                updates = []
                values = []

                if "theme" in preferences:
                    updates.append("theme = ?")
                    values.append(preferences["theme"])

                if "notifications_enabled" in preferences:
                    updates.append("notifications_enabled = ?")
                    values.append(int(preferences["notifications_enabled"]))

                if "default_models" in preferences:
                    updates.append("default_models = ?")
                    values.append(json.dumps(preferences["default_models"]))

                if updates:
                    values.append(user_id)
                    query = f"UPDATE user_preferences SET {', '.join(updates)} WHERE user_id = ?"
                    cursor.execute(query, values)
                    conn.commit()

                    return True

        except Exception as e:
            logger.error(f"Failed to update preferences: {e}")
            return False


    # Singleton instance
    _auth_manager = None


    def get_auth_manager() -> AuthManager:
        """Get the auth manager instance"""
        global _auth_manager
        if _auth_manager is None:
            _auth_manager = AuthManager()
        return _auth_manager


    # Dependency for FastAPI
    async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)) -> User:
        """FastAPI dependency to get current user"""
        auth_manager = get_auth_manager()
        return auth_manager.get_current_user(credentials)


    async def require_admin(current_user: User = Depends(get_current_user)) -> User:
        """FastAPI dependency to require admin user"""
        if not current_user.is_admin:
            raise HTTPException(status_code=403, detail="Admin access required")
        return current_user
