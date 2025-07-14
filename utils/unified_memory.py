"""
Unified Memory System
====================

Combines the persistent memory from the advanced version with
compliance tracking and goal context from the current version.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import hashlib
import re
from collections import defaultdict


class UnifiedMemorySystem:
    """
    Unified memory system that:
        - Stores chat history with importance scoring
        - Tracks compliance events
        - Maintains goal context
        - Extracts and indexes topics
        - Supports memory search and retrieval
    """

    def __init__(self, memory_dir: str = "logs/memory"):
        self.memory_dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)

        # Memory files
        self.chat_file = os.path.join(memory_dir, "chat_memory.jsonl")
        self.compliance_file = os.path.join(memory_dir, "compliance_memory.jsonl")
        self.goal_file = os.path.join(memory_dir, "goal_memory.jsonl")
        self.topic_index_file = os.path.join(memory_dir, "topic_index.json")

        # In-memory caches
        self.topic_index = self._load_topic_index()
        self.importance_scores = {}

    def add_message(self, role: str, content: str, **metadata) -> str:
        """Add a message to memory with full context"""
        message_id = self._generate_id(content)
        timestamp = datetime.now().isoformat()

        # Extract topics
        topics = self._extract_topics(content)

        # Calculate importance
        importance = self._calculate_importance(content, role, metadata)

        # Create memory entry
        entry = {
            "id": message_id,
            "timestamp": timestamp,
            "role": role,
            "content": content,
            "topics": topics,
            "importance": importance,
            "metadata": metadata,
        }

        # Add goal context if available
        if "goal_context" in metadata:
            entry["goal_context"] = metadata["goal_context"]

        # Check for compliance implications
        compliance_flags = self._check_compliance_implications(content)
        if compliance_flags:
            entry["compliance_flags"] = compliance_flags
            self._log_compliance_event(entry)

        # Save to persistent storage
        self._append_to_file(self.chat_file, entry)

        # Update topic index
        self._update_topic_index(message_id, topics)

        # Update importance cache
        self.importance_scores[message_id] = importance

        return message_id

    def add_compliance_event(self, event_type: str, details: Dict[str, Any]):
        """Log a compliance-related event"""
        entry = {"timestamp": datetime.now().isoformat(), "event_type": event_type, "details": details}

        self._append_to_file(self.compliance_file, entry)

    def add_goal_change(self, old_goal: str, new_goal: str, reason: str = ""):
        """Log a goal change event"""
        entry = {"timestamp": datetime.now().isoformat(), "old_goal": old_goal, "new_goal": new_goal, "reason": reason}

        self._append_to_file(self.goal_file, entry)

    def search_by_topic(self, topic: str, limit: int = 10) -> List[Dict]:
        """Search memory by topic"""
        topic_lower = topic.lower()

        # Get message IDs from topic index
        message_ids = []
        for indexed_topic, ids in list(self.topic_index.items()):
            if topic_lower in indexed_topic.lower():
                message_ids.extend(ids)

        # Remove duplicates
        message_ids = list(set(message_ids))

        # Load and sort messages by importance
        messages = []
        for line in self._read_file(self.chat_file):
            entry = json.loads(line)
            if entry["id"] in message_ids:
                messages.append(entry)

        # Sort by importance and recency
        messages.sort(key=lambda x: (x.get("importance", 0), x["timestamp"]), reverse=True)

        return messages[:limit]

    def get_recent_context(self, limit: int = 10, min_importance: float = 0.3) -> List[Dict]:
        """Get recent important messages for context"""
        messages = []

        for line in self._read_file(self.chat_file):
            entry = json.loads(line)
            if entry.get("importance", 0) >= min_importance:
                messages.append(entry)

        # Sort by timestamp (most recent first)
        messages.sort(key=lambda x: x["timestamp"], reverse=True)

        return messages[:limit]

    def get_compliance_history(self, event_type: Optional[str] = None) -> List[Dict]:
        """Get compliance event history"""
        events = []

        for line in self._read_file(self.compliance_file):
            entry = json.loads(line)
            if event_type is None or entry.get("event_type") == event_type:
                events.append(entry)

        return events

    def get_goal_history(self) -> List[Dict]:
        """Get goal change history"""
        goals = []

        for line in self._read_file(self.goal_file):
            goals.append(json.loads(line))

        return goals

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        chat_count = sum(1 for _ in self._read_file(self.chat_file))
        compliance_count = sum(1 for _ in self._read_file(self.compliance_file))
        goal_changes = sum(1 for _ in self._read_file(self.goal_file))

        # Topic statistics
        topic_count = len(self.topic_index)
        most_common_topics = sorted(
            [(topic, len(ids)) for topic, ids in list(self.topic_index.items())], key=lambda x: x[1], reverse=True
        )[:5]

        # Importance distribution
        importance_dist = defaultdict(int)
        for line in self._read_file(self.chat_file):
            entry = json.loads(line)
            importance = entry.get("importance", 0)
            if importance >= 0.8:
                importance_dist["high"] += 1
            elif importance >= 0.5:
                importance_dist["medium"] += 1
            else:
                importance_dist["low"] += 1

        return {
            "total_messages": chat_count,
            "compliance_events": compliance_count,
            "goal_changes": goal_changes,
            "unique_topics": topic_count,
            "top_topics": most_common_topics,
            "importance_distribution": dict(importance_dist),
            "memory_size_mb": self._get_memory_size_mb(),
        }

    def _extract_topics(self, content: str) -> List[str]:
        """Extract topics from content"""
        topics = []

        # Extract model names
        model_patterns = [
            r"random_forest",
            r"gradient_boost",
            r"neural_network",
            r"svm",
            r"logistic_regression",
            r"decision_tree",
            r"kmeans",
            r"dbscan",
        ]
        for pattern in model_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                topics.append(pattern.replace("_", " "))

        # Extract financial terms
        financial_terms = [
            "portfolio",
            "optimization",
            "black-scholes",
            "option",
            "var",
            "risk",
            "return",
            "sharpe",
            "momentum",
        ]
        for term in financial_terms:
            if term.lower() in content.lower():
                topics.append(term)

        return list(set(topics))

    def _calculate_importance(self, content: str, role: str, metadata: Dict) -> float:
        """Calculate importance score for a message"""
        importance = 0.5  # Base importance

        # Role-based importance
        if role == "user":
            importance += 0.2
        elif role == "assistant":
            importance += 0.1

        # Content-based importance
        if len(content) > 200:
            importance += 0.1
        if any(keyword in content.lower() for keyword in ["error", "exception", "failed"]):
            importance += 0.2
        if any(keyword in content.lower() for keyword in ["success", "completed", "finished"]):
            importance += 0.1

        # Metadata-based importance
        if metadata.get("compliance_related"):
            importance += 0.3
        if metadata.get("goal_related"):
            importance += 0.2

        return min(importance, 1.0)

    def _check_compliance_implications(self, content: str) -> List[str]:
        """Check content for compliance implications"""
        flags = []

        # Check for synthetic data patterns
        synthetic_patterns = ["synthetic", "generated", "random", "simulated"]
        for pattern in synthetic_patterns:
            if pattern in content.lower():
                flags.append(f"synthetic_data_{pattern}")

        # Check for compliance violations
        violation_patterns = ["bypass", "override", "ignore", "disable"]
        for pattern in violation_patterns:
            if pattern in content.lower():
                flags.append(f"compliance_violation_{pattern}")

        return flags

    def _log_compliance_event(self, entry: Dict):
        """Log a compliance event"""
        compliance_entry = {
            "timestamp": entry["timestamp"],
            "event_type": "compliance_flag",
            "message_id": entry["id"],
            "flags": entry.get("compliance_flags", []),
            "content_preview": entry["content"][:100],
        }

        self._append_to_file(self.compliance_file, compliance_entry)

    def _update_topic_index(self, message_id: str, topics: List[str]):
        """Update topic index with new message"""
        for topic in topics:
            if topic not in self.topic_index:
                self.topic_index[topic] = []
            self.topic_index[topic].append(message_id)

        # Save updated index
        self._save_topic_index()

    def _load_topic_index(self) -> Dict[str, List[str]]:
        """Load topic index from file"""
        try:
            if os.path.exists(self.topic_index_file):
                with open(self.topic_index_file, "r") as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load topic index: {e}")

        return {}

    def _save_topic_index(self):
        """Save topic index to file"""
        try:
            with open(self.topic_index_file, "w") as f:
                json.dump(self.topic_index, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save topic index: {e}")

    def _generate_id(self, content: str) -> str:
        """Generate unique ID for message"""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _append_to_file(self, filepath: str, entry: Dict):
        """Append entry to JSONL file"""
        try:
            with open(filepath, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to append to {filepath}: {e}")

    def _read_file(self, filepath: str):
        """Read lines from file"""
        try:
            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    for line in f:
                        yield line.strip()
        except Exception as e:
            logger.error(f"Failed to read {filepath}: {e}")

    def _get_memory_size_mb(self) -> float:
        """Get total memory size in MB"""
        total_size = 0

        for filepath in [self.chat_file, self.compliance_file, self.goal_file, self.topic_index_file]:
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)

        return total_size / (1024 * 1024)  # Convert to MB

# Singleton instance
_unified_memory = None

def get_unified_memory() -> UnifiedMemorySystem:
    """Get the unified memory system instance"""
    global _unified_memory
    if _unified_memory is None:
        _unified_memory = UnifiedMemorySystem()
    return _unified_memory
