#!/usr/bin/env python3
"""
🔧 Modular Integration System - Central Component Registry
Central registry for all components with automatic cascading changes and protected core modules
"""

import importlib
import importlib.util
import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Type, Callable, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import time
import hashlib
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComponentType(Enum):
    """Types of components in the system"""
    API = "api"
    MODEL = "model"
    COMPLIANCE = "compliance"
    GOVERNANCE = "governance"
    DATA = "data"
    UTILITY = "utility"
    CORE = "core"
    CUSTOM = "custom"


class ComponentStatus(Enum):
    """Status of components"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PROTECTED = "protected"
    DEPRECATED = "deprecated"
    ERROR = "error"


@dataclass
class ComponentInfo:
    """Information about a component"""
    name: str
    type: ComponentType
    module_path: str
    class_name: Optional[str] = None
    function_name: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    status: ComponentStatus = ComponentStatus.ACTIVE
    last_modified: datetime = field(default_factory=datetime.now)
    hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.hash:
            self.hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """Calculate hash of component file"""
        try:
            file_path = Path(self.module_path)
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    content = f.read()
                return hashlib.sha256(content).hexdigest()
        except Exception as e:
            logger.warning(
                f"Could not calculate hash for {self.module_path}: {e}")
        return ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'type': self.type.value,
            'module_path': self.module_path,
            'class_name': self.class_name,
            'function_name': self.function_name,
            'dependencies': self.dependencies,
            'dependents': self.dependents,
            'status': self.status.value,
            'last_modified': self.last_modified.isoformat(),
            'hash': self.hash,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComponentInfo':
        """Create from dictionary"""
        return cls(
            name=data['name'],
            type=ComponentType(data['type']),
            module_path=data['module_path'],
            class_name=data.get('class_name'),
            function_name=data.get('function_name'),
            dependencies=data.get('dependencies', []),
            dependents=data.get('dependents', []),
            status=ComponentStatus(data['status']),
            last_modified=datetime.fromisoformat(data['last_modified']),
            hash=data.get('hash', ''),
            metadata=data.get('metadata', {})
        )


class ModularIntegrationSystem:
    """Central registry for all components with automatic cascading"""

    def __init__(self, registry_path: str = "logs/component_registry.json"):
        self.registry_path = Path(registry_path)
        self.components: Dict[str, ComponentInfo] = {}
        self.protected_modules: Set[str] = {
            'config/compliance_enforcer.py',
            'config/governance_kernel.py',
            'core/governance_enforcement.py',
            'core/immutable_runtime_enforcer.py',
            'agent_rules.yaml',
            'agent_governance.py'
        }
        self.type_organizations: Dict[ComponentType, List[str]] = {
            component_type: [] for component_type in ComponentType
        }

        # Load existing registry
        self._load_registry()

        # Auto-discover components
        self._discover_components()

    def _load_registry(self):
        """Load component registry from file"""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)
                    for key, component_data in data.items():
                        self.components[key] = ComponentInfo.from_dict(
                            component_data)

                # Rebuild type organizations
                self._rebuild_type_organizations()
                logger.info(
                    f"Loaded {len(self.components)} components from registry")
            except Exception as e:
                logger.error(f"Error loading component registry: {e}")

    def _save_registry(self):
        """Save component registry to file"""
        self.registry_path.parent.mkdir(exist_ok=True)

        try:
            with open(self.registry_path, 'w') as f:
                data = {key: component.to_dict()
                        for key, component in self.components.items()}
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving component registry: {e}")

    def _discover_components(self):
        """Auto-discover components in the project"""
        project_root = Path(".")

        # Discover Python files
        for py_file in project_root.rglob("*.py"):
            if py_file.name.startswith("__"):
                continue

            relative_path = py_file.relative_to(project_root)
            component_name = relative_path.stem

            # Determine component type
            component_type = self._determine_component_type(relative_path)

            # Check if component is protected
            status = ComponentStatus.PROTECTED if str(
                relative_path) in self.protected_modules else ComponentStatus.ACTIVE

            # Create component info
            component_info = ComponentInfo(
                name=component_name,
                type=component_type,
                module_path=str(relative_path),
                status=status
            )

            # Add to registry if not exists or if hash changed
            if component_name not in self.components:
                self.components[component_name] = component_info
                logger.info(
                    f"Discovered new component: {component_name} ({component_type.value})")
            else:
                existing = self.components[component_name]
                if existing.hash != component_info.hash:
                    existing.hash = component_info.hash
                    existing.last_modified = datetime.now()
                    logger.info(f"Updated component: {component_name}")

        self._save_registry()

    def _determine_component_type(self, file_path: Path) -> ComponentType:
        """Determine component type based on file path"""
        path_str = str(file_path)

        if path_str.startswith("config/"):
            if "compliance" in path_str or "governance" in path_str:
                return ComponentType.COMPLIANCE
            elif "api" in path_str:
                return ComponentType.API
            else:
                return ComponentType.CORE
        elif path_str.startswith("core/"):
            if "governance" in path_str or "compliance" in path_str:
                return ComponentType.GOVERNANCE
            else:
                return ComponentType.CORE
        elif path_str.startswith("custom/"):
            return ComponentType.MODEL
        elif path_str.startswith("backend/"):
            return ComponentType.API
        elif path_str.startswith("utils/"):
            return ComponentType.UTILITY
        elif "data" in path_str or "connector" in path_str:
            return ComponentType.DATA
        elif "model" in path_str or "ml" in path_str:
            return ComponentType.MODEL
        else:
            return ComponentType.CUSTOM

    def _rebuild_type_organizations(self):
        """Rebuild type-based component organizations"""
        for component_type in ComponentType:
            self.type_organizations[component_type] = []

        for component_name, component in self.components.items():
            self.type_organizations[component.type].append(component_name)

    def register_component(self,
                           name: str,
                           component_type: ComponentType,
                           module_path: str,
                           class_name: Optional[str] = None,
                           function_name: Optional[str] = None,
                           dependencies: List[str] = None,
                           metadata: Dict[str,
                                          Any] = None) -> ComponentInfo:
        """Register a new component"""

        # Check if component is protected
        if module_path in self.protected_modules:
            logger.warning(
                f"Attempted to register protected component: {name}")
            return None

        component_info = ComponentInfo(
            name=name,
            type=component_type,
            module_path=module_path,
            class_name=class_name,
            function_name=function_name,
            dependencies=dependencies or [],
            metadata=metadata or {}
        )

        self.components[name] = component_info
        self.type_organizations[component_type].append(name)

        # Update dependents
        for dep in component_info.dependencies:
            if dep in self.components:
                self.components[dep].dependents.append(name)

        self._save_registry()
        logger.info(f"Registered component: {name} ({component_type.value})")

        return component_info

    def get_component(self, name: str) -> Optional[ComponentInfo]:
        """Get component by name"""
        return self.components.get(name)

    def get_components_by_type(
            self,
            component_type: ComponentType) -> List[ComponentInfo]:
        """Get all components of a specific type"""
        component_names = self.type_organizations.get(component_type, [])
        return [self.components[name]
                for name in component_names if name in self.components]

    def load_component(self, name: str) -> Optional[Any]:
        """Dynamically load a component"""
        component = self.get_component(name)
        if not component:
            logger.error(f"Component not found: {name}")
            return None

        try:
            # Import module
            spec = importlib.util.spec_from_file_location(
                component.name, component.module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Return class or function if specified
            if component.class_name:
                return getattr(module, component.class_name)
            elif component.function_name:
                return getattr(module, component.function_name)
            else:
                return module

        except Exception as e:
            logger.error(f"Error loading component {name}: {e}")
            component.status = ComponentStatus.ERROR
            self._save_registry()
            return None

    def update_component(self, name: str, **kwargs) -> bool:
        """Update component information"""
        component = self.get_component(name)
        if not component:
            logger.error(f"Component not found: {name}")
            return False

        # Check if component is protected
        if component.module_path in self.protected_modules:
            logger.warning(f"Cannot update protected component: {name}")
            return False

        # Update fields
        for key, value in kwargs.items():
            if hasattr(component, key):
                setattr(component, key, value)

        component.last_modified = datetime.now()
        component.hash = component._calculate_hash()

        self._save_registry()
        logger.info(f"Updated component: {name}")

        # Cascade changes to dependents
        self._cascade_changes(name)

        return True

    def _cascade_changes(self, component_name: str):
        """Cascade changes to dependent components"""
        component = self.get_component(component_name)
        if not component:
            return

        for dependent_name in component.dependents:
            dependent = self.get_component(dependent_name)
            if dependent:
                dependent.last_modified = datetime.now()
                logger.info(f"Cascaded change to dependent: {dependent_name}")

        self._save_registry()

    def remove_component(self, name: str) -> bool:
        """Remove a component"""
        component = self.get_component(name)
        if not component:
            logger.error(f"Component not found: {name}")
            return False

        # Check if component is protected
        if component.module_path in self.protected_modules:
            logger.warning(f"Cannot remove protected component: {name}")
            return False

        # Remove from type organization
        if name in self.type_organizations[component.type]:
            self.type_organizations[component.type].remove(name)

        # Remove from dependents
        for dep in component.dependencies:
            if dep in self.components:
                self.components[dep].dependents.remove(name)

        # Remove component
        del self.components[name]

        self._save_registry()
        logger.info(f"Removed component: {name}")

        return True

    def get_dependency_tree(self, name: str) -> Dict[str, Any]:
        """Get dependency tree for a component"""
        component = self.get_component(name)
        if not component:
            return {}

        tree = {
            'component': component.name,
            'dependencies': [],
            'dependents': [],
            'depth': 0
        }

        # Get dependencies recursively
        for dep_name in component.dependencies:
            dep_tree = self.get_dependency_tree(dep_name)
            tree['dependencies'].append(dep_tree)

        # Get dependents
        for dep_name in component.dependents:
            dep = self.get_component(dep_name)
            if dep:
                tree['dependents'].append({
                    'name': dep.name,
                    'type': dep.type.value,
                    'status': dep.status.value
                })

        return tree

    def validate_dependencies(self, name: str) -> List[str]:
        """Validate component dependencies"""
        component = self.get_component(name)
        if not component:
            return [f"Component not found: {name}"]

        errors = []

        for dep_name in component.dependencies:
            dep = self.get_component(dep_name)
            if not dep:
                errors.append(f"Missing dependency: {dep_name}")
            elif dep.status == ComponentStatus.ERROR:
                errors.append(f"Dependency has error: {dep_name}")
            elif dep.status == ComponentStatus.INACTIVE:
                errors.append(f"Dependency is inactive: {dep_name}")

        return errors

    def get_system_summary(self) -> Dict[str, Any]:
        """Get summary of the modular system"""
        summary = {
            'total_components': len(self.components),
            'by_type': {},
            'by_status': {},
            'protected_components': [],
            'recent_changes': []
        }

        # Count by type
        for component_type in ComponentType:
            summary['by_type'][component_type.value] = len(
                self.get_components_by_type(component_type))

        # Count by status
        for component in self.components.values():
            status = component.status.value
            if status not in summary['by_status']:
                summary['by_status'][status] = 0
            summary['by_status'][status] += 1

            # Track protected components
            if component.module_path in self.protected_modules:
                summary['protected_components'].append(component.name)

            # Track recent changes (last 24 hours)
            if (datetime.now() - component.last_modified).total_seconds() < 86400:
                summary['recent_changes'].append({
                    'name': component.name,
                    'type': component.type.value,
                    'last_modified': component.last_modified.isoformat()
                })

        return summary

    def protect_module(self, module_path: str):
        """Add a module to protected list"""
        self.protected_modules.add(module_path)
        logger.info(f"Protected module: {module_path}")

    def unprotect_module(self, module_path: str):
        """Remove a module from protected list"""
        self.protected_modules.discard(module_path)
        logger.info(f"Unprotected module: {module_path}")


# Global modular system instance
modular_system = ModularIntegrationSystem()


def register_component(
        name: str,
        component_type: ComponentType,
        module_path: str,
        **kwargs) -> ComponentInfo:
    """Convenience function to register a component"""
    return modular_system.register_component(
        name, component_type, module_path, **kwargs)


def get_component(name: str) -> Optional[ComponentInfo]:
    """Get component by name"""
    return modular_system.get_component(name)


def load_component(name: str) -> Optional[Any]:
    """Load a component dynamically"""
    return modular_system.load_component(name)


def get_components_by_type(
        component_type: ComponentType) -> List[ComponentInfo]:
    """Get components by type"""
    return modular_system.get_components_by_type(component_type)


def get_system_summary() -> Dict[str, Any]:
    """Get system summary"""
    return modular_system.get_system_summary()


if __name__ == "__main__":
    # Example usage
    print("🔧 Modular Integration System - Central Component Registry")

    # Get system summary
    summary = get_system_summary()
    print(f"System summary: {summary}")

    # Show components by type
    for component_type in ComponentType:
        components = get_components_by_type(component_type)
        print(f"{component_type.value}: {len(components)} components")
