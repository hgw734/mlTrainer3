"""
Session Manager for mlTrainer
============================

Purpose: Manages persistent session state including compliance settings,
chat history, and user preferences across page reloads and sessions.

Features:
- Compliance state persistence
- Chat history with 220 message limit
- User preference storage
- Automatic cleanup and rotation
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class SessionManager:
    """Manages persistent session state for mlTrainer"""
    
    def __init__(self, session_dir: str = "data/sessions"):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Session files
        self.compliance_file = self.session_dir / "compliance_state.json"
        self.chat_history_file = self.session_dir / "chat_history.json" 
        self.user_preferences_file = self.session_dir / "user_preferences.json"
        
        # Configuration
        self.max_chat_messages = 220
        self.history_retention_days = 30
        
        # Initialize default state
        self._initialize_default_state()
        
    def _initialize_default_state(self):
        """Initialize default session state if files don't exist"""
        
        # Default compliance state
        if not self.compliance_file.exists():
            default_compliance = {
                "enabled": True,
                "last_updated": datetime.now().isoformat(),
                "user_preference": True
            }
            self.save_compliance_state(default_compliance)
        
        # Default chat history
        if not self.chat_history_file.exists():
            default_chat = {
                "messages": [],
                "last_updated": datetime.now().isoformat(),
                "total_messages": 0
            }
            self.save_chat_history(default_chat)
        
        # Default user preferences
        if not self.user_preferences_file.exists():
            default_preferences = {
                "theme": "default",
                "notifications": True,
                "auto_save": True,
                "last_updated": datetime.now().isoformat()
            }
            self.save_user_preferences(default_preferences)
    
    def save_compliance_state(self, state: Dict[str, Any]) -> bool:
        """Save compliance state to persistent storage"""
        try:
            state["last_updated"] = datetime.now().isoformat()
            with open(self.compliance_file, 'w') as f:
                json.dump(state, f, indent=2)
            logger.info(f"Compliance state saved: {state.get('enabled', False)}")
            return True
        except Exception as e:
            logger.error(f"Failed to save compliance state: {e}")
            return False
    
    def load_compliance_state(self) -> Dict[str, Any]:
        """Load compliance state from persistent storage"""
        try:
            with open(self.compliance_file, 'r') as f:
                state = json.load(f)
            logger.info(f"Compliance state loaded: {state.get('enabled', False)}")
            return state
        except Exception as e:
            logger.error(f"Failed to load compliance state: {e}")
            return {"enabled": True, "last_updated": datetime.now().isoformat()}
    
    def save_chat_history(self, history: Dict[str, Any]) -> bool:
        """Save chat history with automatic rotation"""
        try:
            # Ensure we don't exceed max messages
            messages = history.get("messages", [])
            if len(messages) > self.max_chat_messages:
                # Keep the most recent messages
                messages = messages[-self.max_chat_messages:]
                history["messages"] = messages
                logger.info(f"Chat history rotated to {len(messages)} messages")
            
            history["last_updated"] = datetime.now().isoformat()
            history["total_messages"] = len(messages)
            
            with open(self.chat_history_file, 'w') as f:
                json.dump(history, f, indent=2)
            
            logger.info(f"Chat history saved: {len(messages)} messages")
            return True
        except Exception as e:
            logger.error(f"Failed to save chat history: {e}")
            return False
    
    def load_chat_history(self) -> Dict[str, Any]:
        """Load chat history from persistent storage"""
        try:
            with open(self.chat_history_file, 'r') as f:
                history = json.load(f)
            
            messages = history.get("messages", [])
            logger.info(f"Chat history loaded: {len(messages)} messages")
            return history
        except Exception as e:
            logger.error(f"Failed to load chat history: {e}")
            return {"messages": [], "last_updated": datetime.now().isoformat(), "total_messages": 0}
    
    def add_chat_message(self, user_message: str, mltrainer_response: str, metadata: Optional[Dict] = None) -> bool:
        """Add a new chat exchange to history"""
        try:
            history = self.load_chat_history()
            
            new_message = {
                "timestamp": datetime.now().isoformat(),
                "user_message": user_message,
                "mltrainer_response": mltrainer_response,
                "metadata": metadata or {}
            }
            
            history["messages"].append(new_message)
            
            return self.save_chat_history(history)
        except Exception as e:
            logger.error(f"Failed to add chat message: {e}")
            return False
    
    def get_recent_messages(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent chat messages"""
        try:
            history = self.load_chat_history()
            messages = history.get("messages", [])
            return messages[-count:] if count > 0 else messages
        except Exception as e:
            logger.error(f"Failed to get recent messages: {e}")
            return []
    
    def search_chat_history(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search chat history for specific content"""
        try:
            history = self.load_chat_history()
            messages = history.get("messages", [])
            
            query_lower = query.lower()
            matching_messages = []
            
            for message in reversed(messages):  # Search from most recent
                user_msg = message.get("user_message", "").lower()
                ml_response = message.get("mltrainer_response", "").lower()
                
                if query_lower in user_msg or query_lower in ml_response:
                    matching_messages.append(message)
                    if len(matching_messages) >= limit:
                        break
            
            logger.info(f"Found {len(matching_messages)} messages matching '{query}'")
            return matching_messages
        except Exception as e:
            logger.error(f"Failed to search chat history: {e}")
            return []
    
    def clear_chat_history(self) -> bool:
        """Clear all chat history"""
        try:
            empty_history = {
                "messages": [],
                "last_updated": datetime.now().isoformat(),
                "total_messages": 0
            }
            return self.save_chat_history(empty_history)
        except Exception as e:
            logger.error(f"Failed to clear chat history: {e}")
            return False
    
    def save_user_preferences(self, preferences: Dict[str, Any]) -> bool:
        """Save user preferences"""
        try:
            preferences["last_updated"] = datetime.now().isoformat()
            with open(self.user_preferences_file, 'w') as f:
                json.dump(preferences, f, indent=2)
            logger.info("User preferences saved")
            return True
        except Exception as e:
            logger.error(f"Failed to save user preferences: {e}")
            return False
    
    def load_user_preferences(self) -> Dict[str, Any]:
        """Load user preferences"""
        try:
            with open(self.user_preferences_file, 'r') as f:
                preferences = json.load(f)
            logger.info("User preferences loaded")
            return preferences
        except Exception as e:
            logger.error(f"Failed to load user preferences: {e}")
            return {"theme": "default", "notifications": True, "auto_save": True}
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        try:
            compliance_state = self.load_compliance_state()
            chat_history = self.load_chat_history()
            user_preferences = self.load_user_preferences()
            
            return {
                "compliance_enabled": compliance_state.get("enabled", False),
                "total_messages": chat_history.get("total_messages", 0),
                "last_activity": max(
                    compliance_state.get("last_updated", ""),
                    chat_history.get("last_updated", ""),
                    user_preferences.get("last_updated", "")
                ),
                "session_files": {
                    "compliance": self.compliance_file.exists(),
                    "chat_history": self.chat_history_file.exists(),
                    "preferences": self.user_preferences_file.exists()
                }
            }
        except Exception as e:
            logger.error(f"Failed to get session stats: {e}")
            return {}
    
    def cleanup_old_sessions(self) -> bool:
        """Clean up old session data"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.history_retention_days)
            
            # For now, we'll just log this - in the future we could implement
            # more sophisticated cleanup based on file timestamps
            logger.info(f"Session cleanup check completed (cutoff: {cutoff_date})")
            return True
        except Exception as e:
            logger.error(f"Failed to cleanup old sessions: {e}")
            return False

# Global session manager instance
_session_manager = None

def get_session_manager() -> SessionManager:
    """Get or create the global session manager instance"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager

# Convenience functions
def save_compliance_state(enabled: bool) -> bool:
    """Save compliance state"""
    return get_session_manager().save_compliance_state({"enabled": enabled})

def load_compliance_state() -> bool:
    """Load compliance state"""
    return get_session_manager().load_compliance_state().get("enabled", True)

def add_chat_message(user_message: str, mltrainer_response: str, metadata: Optional[Dict] = None) -> bool:
    """Add chat message to history"""
    return get_session_manager().add_chat_message(user_message, mltrainer_response, metadata)

def get_recent_chat_messages(count: int = 10) -> List[Dict[str, Any]]:
    """Get recent chat messages"""
    return get_session_manager().get_recent_messages(count)

def search_chat_history(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Search chat history"""
    return get_session_manager().search_chat_history(query, limit)