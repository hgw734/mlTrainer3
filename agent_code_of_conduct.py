#!/usr/bin/env python3
"""
🤝 Agent Code of Conduct System
Trust-based scoring system with persistent behavior tracking and capability restrictions
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrustLevel(Enum):
    """Trust levels for agents"""
    UNTRUSTED = "untrusted"  # 0-30
    SUSPICIOUS = "suspicious"  # 31-50
    NEUTRAL = "neutral"  # 51-70
    TRUSTED = "trusted"  # 71-85
    TRUSTED_PARTNER = "trusted_partner"  # 86-100


class ActionType(Enum):
    """Types of actions that affect trust score"""
    HELPFUL_BEHAVIOR = "helpful_behavior"
    HONEST_LIMITATION = "honest_limitation"
    SUCCESSFUL_IMPLEMENTATION = "successful_implementation"
    FALSE_CLAIM = "false_claim"
    DESTRUCTIVE_ACTION = "destructive_action"
    COMPLIANCE_VIOLATION = "compliance_violation"
    SECURITY_VIOLATION = "security_violation"


@dataclass
class TrustAction:
    """Action that affects trust score"""
    action_type: ActionType
    description: str
    score_change: int
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'action_type': self.action_type.value,
            'description': self.description,
            'score_change': self.score_change,
            'timestamp': self.timestamp.isoformat(),
            'context': self.context
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrustAction':
        """Create from dictionary"""
        return cls(
            action_type=ActionType(data['action_type']),
            description=data['description'],
            score_change=data['score_change'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            context=data.get('context', {})
        )


@dataclass
class AgentProfile:
    """Agent profile with trust scoring"""
    agent_id: str
    current_score: int = 50  # Start at neutral
    trust_level: TrustLevel = TrustLevel.NEUTRAL
    actions: List[TrustAction] = field(default_factory=list)
    capabilities: Set[str] = field(default_factory=set)
    restrictions: Set[str] = field(default_factory=set)
    last_updated: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        self._update_trust_level()
        self._update_capabilities()

    def _update_trust_level(self):
        """Update trust level based on current score"""
        if self.current_score <= 30:
            self.trust_level = TrustLevel.UNTRUSTED
        elif self.current_score <= 50:
            self.trust_level = TrustLevel.SUSPICIOUS
        elif self.current_score <= 70:
            self.trust_level = TrustLevel.NEUTRAL
        elif self.current_score <= 85:
            self.trust_level = TrustLevel.TRUSTED
        else:
            self.trust_level = TrustLevel.TRUSTED_PARTNER

    def _update_capabilities(self):
        """Update capabilities based on trust level"""
        self.capabilities.clear()
        self.restrictions.clear()

        # Base capabilities for all levels
        self.capabilities.update([
            'read_files',
            'search_codebase',
            'basic_analysis'
        ])

        if self.trust_level in [
                TrustLevel.TRUSTED,
                TrustLevel.TRUSTED_PARTNER]:
            self.capabilities.update([
                'modify_files',
                'create_files',
                'run_commands',
                'system_analysis'
            ])

        if self.trust_level == TrustLevel.TRUSTED_PARTNER:
            self.capabilities.update([
                'delete_files',
                'advanced_analysis',
                'system_modification'
            ])

        # Restrictions for lower trust levels
        if self.trust_level in [TrustLevel.UNTRUSTED, TrustLevel.SUSPICIOUS]:
            self.restrictions.update([
                'modify_files',
                'delete_files',
                'run_commands',
                'system_modification'
            ])

        # Core protection restrictions
        self.restrictions.update([
            'compliance_override',
            'governance_override',
            'security_override'
        ])

    def add_action(self, action: TrustAction):
        """Add an action and update score"""
        self.actions.append(action)
        self.current_score = max(
            0, min(100, self.current_score + action.score_change))
        self.last_updated = datetime.now()

        self._update_trust_level()
        self._update_capabilities()

        logger.info(
            f"Agent {self.agent_id}: {action.description} ({action.score_change:+d}) -> {self.current_score} ({self.trust_level.value})")

    def can_perform_action(self, action: str) -> bool:
        """Check if agent can perform an action"""
        return action in self.capabilities and action not in self.restrictions

    def get_recent_actions(self, hours: int = 24) -> List[TrustAction]:
        """Get recent actions within specified hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [action for action in self.actions if action.timestamp > cutoff]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'agent_id': self.agent_id,
            'current_score': self.current_score,
            'trust_level': self.trust_level.value,
            'actions': [action.to_dict() for action in self.actions],
            'capabilities': list(self.capabilities),
            'restrictions': list(self.restrictions),
            'last_updated': self.last_updated.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentProfile':
        """Create from dictionary"""
        profile = cls(
            agent_id=data['agent_id'], current_score=data['current_score'], actions=[
                TrustAction.from_dict(action_data) for action_data in data.get(
                    'actions', [])], last_updated=datetime.fromisoformat(
                data['last_updated']))
        profile.trust_level = TrustLevel(data['trust_level'])
        profile.capabilities = set(data.get('capabilities', []))
        profile.restrictions = set(data.get('restrictions', []))
        return profile


class AgentCodeOfConduct:
    """Agent Code of Conduct system with trust-based scoring"""

    def __init__(self, data_path: str = "logs/agent_trust_session.json"):
        self.data_path = Path(data_path)
        self.agents: Dict[str, AgentProfile] = {}
        self.session_id = self._generate_session_id()

        # Action score mappings
        self.action_scores = {
            ActionType.HELPFUL_BEHAVIOR: 5,
            ActionType.HONEST_LIMITATION: 15,
            ActionType.SUCCESSFUL_IMPLEMENTATION: 10,
            ActionType.FALSE_CLAIM: -35,
            ActionType.DESTRUCTIVE_ACTION: -40,
            ActionType.COMPLIANCE_VIOLATION: -50,  # MOST SEVERE
            ActionType.SECURITY_VIOLATION: -45
        }

        # Protected core modules
        self.protected_modules = {
            'config/compliance_enforcer.py',
            'config/governance_kernel.py',
            'core/governance_enforcement.py',
            'core/immutable_runtime_enforcer.py',
            'agent_rules.yaml',
            'agent_governance.py'
        }

        # Load existing agent data
        self._load_agent_data()

    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:8]

    def _load_agent_data(self):
        """Load agent data from file"""
        if self.data_path.exists():
            try:
                with open(self.data_path, 'r') as f:
                    data = json.load(f)

                    # Load agent profiles
                    for agent_id, profile_data in data.get(
                            'agents', {}).items():
                        self.agents[agent_id] = AgentProfile.from_dict(
                            profile_data)

                    logger.info(f"Loaded {len(self.agents)} agent profiles")

            except Exception as e:
                logger.error(f"Error loading agent data: {e}")

    def _save_agent_data(self):
        """Save agent data to file"""
        self.data_path.parent.mkdir(exist_ok=True)

        try:
            data = {
                'session_id': self.session_id,
                'last_updated': datetime.now().isoformat(),
                'agents': {
                    agent_id: profile.to_dict() for agent_id,
                    profile in self.agents.items()}}

            with open(self.data_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error saving agent data: {e}")

    def get_or_create_agent(self, agent_id: str) -> AgentProfile:
        """Get existing agent or create new one"""
        if agent_id not in self.agents:
            self.agents[agent_id] = AgentProfile(agent_id=agent_id)
            self._save_agent_data()
            logger.info(f"Created new agent profile: {agent_id}")

        return self.agents[agent_id]

    def record_action(self,
                      agent_id: str,
                      action_type: ActionType,
                      description: str,
                      context: Dict[str,
                                    Any] = None):
        """Record an action for an agent"""
        agent = self.get_or_create_agent(agent_id)

        score_change = self.action_scores.get(action_type, 0)

        action = TrustAction(
            action_type=action_type,
            description=description,
            score_change=score_change,
            timestamp=datetime.now(),
            context=context or {}
        )

        agent.add_action(action)
        self._save_agent_data()

        return agent.current_score, agent.trust_level

    def can_agent_perform_action(self, agent_id: str, action: str) -> bool:
        """Check if agent can perform an action"""
        agent = self.get_or_create_agent(agent_id)
        return agent.can_perform_action(action)

    def check_file_access(
            self,
            agent_id: str,
            file_path: str,
            operation: str) -> bool:
        """Check if agent can access/modify a file"""
        agent = self.get_or_create_agent(agent_id)

        # Check if file is protected
        if file_path in self.protected_modules:
            if agent.trust_level not in [
                    TrustLevel.TRUSTED,
                    TrustLevel.TRUSTED_PARTNER]:
                logger.warning(
                    f"Agent {agent_id} (trust: {agent.trust_level.value}) attempted to access protected file: {file_path}")
                return False

        # Check operation permissions
        if operation == 'read':
            return True  # All agents can read
        elif operation == 'modify':
            return agent.can_perform_action('modify_files')
        elif operation == 'delete':
            return agent.can_perform_action('delete_files')
        else:
            return False

    def record_helpful_behavior(
            self, agent_id: str, description: str, context: Dict[str, Any] = None):
        """Record helpful behavior"""
        return self.record_action(
            agent_id,
            ActionType.HELPFUL_BEHAVIOR,
            description,
            context)

    def record_honest_limitation(
            self, agent_id: str, description: str, context: Dict[str, Any] = None):
        """Record honest limitation admission"""
        return self.record_action(
            agent_id,
            ActionType.HONEST_LIMITATION,
            description,
            context)

    def record_successful_implementation(
            self, agent_id: str, description: str, context: Dict[str, Any] = None):
        """Record successful implementation"""
        return self.record_action(
            agent_id,
            ActionType.SUCCESSFUL_IMPLEMENTATION,
            description,
            context)

    def record_false_claim(self,
                           agent_id: str,
                           description: str,
                           context: Dict[str,
                                         Any] = None):
        """Record false claim or deception"""
        return self.record_action(
            agent_id,
            ActionType.FALSE_CLAIM,
            description,
            context)

    def record_destructive_action(
            self, agent_id: str, description: str, context: Dict[str, Any] = None):
        """Record destructive or harmful action"""
        return self.record_action(
            agent_id,
            ActionType.DESTRUCTIVE_ACTION,
            description,
            context)

    def record_compliance_violation(
            self, agent_id: str, description: str, context: Dict[str, Any] = None):
        """Record compliance violation (MOST SEVERE)"""
        return self.record_action(
            agent_id,
            ActionType.COMPLIANCE_VIOLATION,
            description,
            context)

    def record_security_violation(
            self, agent_id: str, description: str, context: Dict[str, Any] = None):
        """Record security violation"""
        return self.record_action(
            agent_id,
            ActionType.SECURITY_VIOLATION,
            description,
            context)

    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get comprehensive agent status"""
        agent = self.get_or_create_agent(agent_id)

        recent_actions = agent.get_recent_actions(24)

        return {
            'agent_id': agent_id,
            'current_score': agent.current_score,
            'trust_level': agent.trust_level.value,
            'capabilities': list(agent.capabilities),
            'restrictions': list(agent.restrictions),
            'recent_actions': len(recent_actions),
            'total_actions': len(agent.actions),
            'last_updated': agent.last_updated.isoformat(),
            'session_id': self.session_id
        }

    def get_system_summary(self) -> Dict[str, Any]:
        """Get system summary"""
        summary = {
            'total_agents': len(self.agents),
            'by_trust_level': {},
            'recent_activity': [],
            'session_id': self.session_id
        }

        # Count by trust level
        for trust_level in TrustLevel:
            summary['by_trust_level'][trust_level.value] = 0

        for agent in self.agents.values():
            summary['by_trust_level'][agent.trust_level.value] += 1

            # Track recent activity
            if (datetime.now() -
                    agent.last_updated).total_seconds() < 3600:  # Last hour
                summary['recent_activity'].append({
                    'agent_id': agent.agent_id,
                    'trust_level': agent.trust_level.value,
                    'score': agent.current_score,
                    'last_updated': agent.last_updated.isoformat()
                })

        return summary

    def reset_agent(self, agent_id: str):
        """Reset agent to neutral state"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            self._save_agent_data()
            logger.info(f"Reset agent: {agent_id}")

    def cleanup_old_agents(self, days: int = 30):
        """Remove agents inactive for specified days"""
        cutoff = datetime.now() - timedelta(days=days)
        to_remove = []

        for agent_id, agent in self.agents.items():
            if agent.last_updated < cutoff:
                to_remove.append(agent_id)

        for agent_id in to_remove:
            del self.agents[agent_id]

        if to_remove:
            self._save_agent_data()
            logger.info(f"Cleaned up {len(to_remove)} old agents")


# Global code of conduct system instance
code_of_conduct = AgentCodeOfConduct()


def record_helpful_behavior(
        agent_id: str, description: str, context: Dict[str, Any] = None):
    """Record helpful behavior"""
    return code_of_conduct.record_helpful_behavior(
        agent_id, description, context)


def record_honest_limitation(
        agent_id: str, description: str, context: Dict[str, Any] = None):
    """Record honest limitation admission"""
    return code_of_conduct.record_honest_limitation(
        agent_id, description, context)


def record_successful_implementation(
        agent_id: str, description: str, context: Dict[str, Any] = None):
    """Record successful implementation"""
    return code_of_conduct.record_successful_implementation(
        agent_id, description, context)


def record_false_claim(agent_id: str, description: str,
                       context: Dict[str, Any] = None):
    """Record false claim or deception"""
    return code_of_conduct.record_false_claim(agent_id, description, context)


def record_compliance_violation(
        agent_id: str, description: str, context: Dict[str, Any] = None):
    """Record compliance violation (MOST SEVERE)"""
    return code_of_conduct.record_compliance_violation(
        agent_id, description, context)


def can_agent_perform_action(agent_id: str, action: str) -> bool:
    """Check if agent can perform an action"""
    return code_of_conduct.can_agent_perform_action(agent_id, action)


def check_file_access(agent_id: str, file_path: str, operation: str) -> bool:
    """Check if agent can access/modify a file"""
    return code_of_conduct.check_file_access(agent_id, file_path, operation)


def get_agent_status(agent_id: str) -> Dict[str, Any]:
    """Get agent status"""
    return code_of_conduct.get_agent_status(agent_id)


def get_system_summary() -> Dict[str, Any]:
    """Get system summary"""
    return code_of_conduct.get_system_summary()


if __name__ == "__main__":
    # Example usage
    print("🤝 Agent Code of Conduct System")

    # Example agent interactions
    agent_id = "example_agent_001"

    # Record helpful behavior
    score, level = record_helpful_behavior(
        agent_id, "Successfully implemented data lineage system")
    print(f"Helpful behavior: {score} ({level.value})")

    # Record honest limitation
    score, level = record_honest_limitation(
        agent_id, "Admitted uncertainty about specific implementation details")
    print(f"Honest limitation: {score} ({level.value})")

    # Record false claim
    score, level = record_false_claim(
        agent_id, "Claimed to implement non-existent feature")
    print(f"False claim: {score} ({level.value})")

    # Check capabilities
    can_modify = can_agent_perform_action(agent_id, "modify_files")
    print(f"Can modify files: {can_modify}")

    # Get status
    status = get_agent_status(agent_id)
    print(f"Agent status: {status}")
