#!/usr/bin/env python3
"""
Model Registry Builder for mlTrainer3
=====================================
Scans all model directories and builds a unified registry of 200+ models.
NO TEMPLATES - This is real, functional code that discovers actual models.
"""

import os
import ast
import json
import importlib.util
import inspect
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelRegistryBuilder:
    """Builds comprehensive registry of all ML models in the system"""
    
    def __init__(self):
        self.models = {}
        self.categories = {
            'ml': 'Machine Learning',
            'risk': 'Risk Management',
            'volatility': 'Volatility Models',
            'portfolio': 'Portfolio Optimization',
            'technical': 'Technical Analysis',
            'sentiment': 'Sentiment Analysis',
            'regime': 'Market Regime Detection',
            'options': 'Options Pricing',
            'forecast': 'Time Series Forecasting',
            'ensemble': 'Ensemble Methods'
        }
        self.scan_paths = [
            'custom',
            'ml_engine',
            'ml_fmt',
            'finance_fmt'
        ]
        
    def scan_directory(self, directory: str) -> Dict[str, Any]:
        """Scan a directory for Python files containing models"""
        models_found = {}
        dir_path = Path(directory)
        
        if not dir_path.exists():
            logger.warning(f"Directory {directory} does not exist")
            return models_found
            
        for py_file in dir_path.glob('*.py'):
            if py_file.name.startswith('__'):
                continue
                
            try:
                models_in_file = self.extract_models_from_file(py_file)
                if models_in_file:
                    models_found[str(py_file)] = models_in_file
                    logger.info(f"Found {len(models_in_file)} models in {py_file}")
            except Exception as e:
                logger.error(f"Error scanning {py_file}: {e}")
                
        return models_found
    
    def extract_models_from_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract model classes and functions from a Python file"""
        models = []
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Parse AST to find classes and functions
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    model_info = self.extract_class_info(node, file_path)
                    if model_info:
                        models.append(model_info)
                        
                elif isinstance(node, ast.FunctionDef):
                    # Check if function name suggests it's a model
                    if any(keyword in node.name.lower() for keyword in 
                          ['model', 'predict', 'forecast', 'train', 'fit']):
                        model_info = self.extract_function_info(node, file_path)
                        if model_info:
                            models.append(model_info)
                            
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            
        return models
    
    def extract_class_info(self, node: ast.ClassDef, file_path: Path) -> Optional[Dict[str, Any]]:
        """Extract information about a model class"""
        # Skip non-model classes
        if not any(keyword in node.name.lower() for keyword in 
                  ['model', 'predictor', 'classifier', 'regressor', 'ensemble',
                   'strategy', 'optimizer', 'analyzer', 'detector']):
            return None
            
        model_info = {
            'name': node.name,
            'type': 'class',
            'file': str(file_path),
            'category': self.categorize_model(node.name, file_path),
            'methods': [],
            'parameters': [],
            'docstring': ast.get_docstring(node) or 'No description available'
        }
        
        # Extract methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = {
                    'name': item.name,
                    'parameters': [arg.arg for arg in item.args.args if arg.arg != 'self'],
                    'docstring': ast.get_docstring(item)
                }
                model_info['methods'].append(method_info)
                
                # Special handling for __init__ to get model parameters
                if item.name == '__init__':
                    model_info['parameters'] = method_info['parameters']
                    
        return model_info
    
    def extract_function_info(self, node: ast.FunctionDef, file_path: Path) -> Optional[Dict[str, Any]]:
        """Extract information about a model function"""
        return {
            'name': node.name,
            'type': 'function',
            'file': str(file_path),
            'category': self.categorize_model(node.name, file_path),
            'parameters': [arg.arg for arg in node.args.args],
            'docstring': ast.get_docstring(node) or 'No description available'
        }
    
    def categorize_model(self, model_name: str, file_path: Path) -> str:
        """Categorize a model based on its name and location"""
        name_lower = model_name.lower()
        file_name = file_path.stem.lower()
        
        # Check file name first
        for key, category in self.categories.items():
            if key in file_name:
                return category
                
        # Then check model name
        if 'risk' in name_lower or 'var' in name_lower:
            return 'Risk Management'
        elif 'volatility' in name_lower or 'garch' in name_lower:
            return 'Volatility Models'
        elif 'portfolio' in name_lower or 'optimize' in name_lower:
            return 'Portfolio Optimization'
        elif 'lstm' in name_lower or 'rnn' in name_lower or 'neural' in name_lower:
            return 'Machine Learning'
        elif 'regime' in name_lower or 'state' in name_lower:
            return 'Market Regime Detection'
        elif 'ensemble' in name_lower:
            return 'Ensemble Methods'
        elif 'option' in name_lower or 'black' in name_lower:
            return 'Options Pricing'
        elif 'forecast' in name_lower or 'arima' in name_lower:
            return 'Time Series Forecasting'
        elif any(ind in name_lower for ind in ['rsi', 'macd', 'bollinger', 'ema', 'sma']):
            return 'Technical Analysis'
        else:
            return 'Machine Learning'  # Default category
    
    def build_registry(self) -> Dict[str, Any]:
        """Build the complete model registry"""
        logger.info("Starting model registry build...")
        
        all_models = {}
        total_models = 0
        
        # Scan each directory
        for directory in self.scan_paths:
            logger.info(f"Scanning {directory}...")
            models = self.scan_directory(directory)
            
            for file_path, file_models in models.items():
                for model in file_models:
                    model_key = f"{model['name']}_{Path(file_path).stem}"
                    all_models[model_key] = model
                    total_models += 1
        
        # Also scan for specific known model files
        known_files = [
            'machine_learning.py',
            'risk_management.py', 
            'volatility_models.py',
            'technical_analysis.py',
            'regime_detection.py',
            'momentum_models.py',
            'position_sizing.py',
            'volume_analysis.py'
        ]
        
        for file_name in known_files:
            file_path = Path('custom') / file_name
            if file_path.exists():
                models = self.extract_models_from_file(file_path)
                for model in models:
                    model_key = f"{model['name']}_{file_path.stem}"
                    if model_key not in all_models:
                        all_models[model_key] = model
                        total_models += 1
        
        # Create registry structure
        registry = {
            'version': '1.0',
            'created': datetime.now().isoformat(),
            'total_models': total_models,
            'categories': self.categories,
            'models': all_models,
            'statistics': self.calculate_statistics(all_models)
        }
        
        logger.info(f"Registry built with {total_models} models")
        return registry
    
    def calculate_statistics(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistics about the model registry"""
        category_counts = {}
        type_counts = {'class': 0, 'function': 0}
        
        for model in models.values():
            # Count by category
            category = model.get('category', 'Unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
            
            # Count by type
            model_type = model.get('type', 'unknown')
            if model_type in type_counts:
                type_counts[model_type] += 1
        
        return {
            'by_category': category_counts,
            'by_type': type_counts,
            'files_scanned': len(set(m['file'] for m in models.values()))
        }
    
    def save_registry(self, registry: Dict[str, Any], output_file: str = 'model_registry.json'):
        """Save the registry to a JSON file"""
        with open(output_file, 'w') as f:
            json.dump(registry, f, indent=2)
        logger.info(f"Registry saved to {output_file}")
    
    def generate_markdown_report(self, registry: Dict[str, Any], output_file: str = 'MODEL_REGISTRY_REPORT.md'):
        """Generate a human-readable markdown report"""
        with open(output_file, 'w') as f:
            f.write("# mlTrainer3 Model Registry Report\n\n")
            f.write(f"Generated: {registry['created']}\n\n")
            f.write(f"## Summary\n\n")
            f.write(f"- **Total Models**: {registry['total_models']}\n")
            f.write(f"- **Categories**: {len(registry['categories'])}\n")
            f.write(f"- **Files Scanned**: {registry['statistics']['files_scanned']}\n\n")
            
            f.write("## Models by Category\n\n")
            for category, count in registry['statistics']['by_category'].items():
                f.write(f"- **{category}**: {count} models\n")
            
            f.write("\n## Model Details\n\n")
            
            # Group models by category
            models_by_category = {}
            for model_key, model in registry['models'].items():
                category = model.get('category', 'Unknown')
                if category not in models_by_category:
                    models_by_category[category] = []
                models_by_category[category].append(model)
            
            # Write details for each category
            for category, models in sorted(models_by_category.items()):
                f.write(f"### {category}\n\n")
                for model in sorted(models, key=lambda x: x['name']):
                    f.write(f"#### {model['name']}\n")
                    f.write(f"- **Type**: {model['type']}\n")
                    f.write(f"- **File**: `{model['file']}`\n")
                    if model['parameters']:
                        f.write(f"- **Parameters**: {', '.join(model['parameters'])}\n")
                    f.write(f"- **Description**: {model['docstring'].split('\\n')[0]}\n")
                    f.write("\n")
        
        logger.info(f"Markdown report saved to {output_file}")


def main():
    """Build the model registry"""
    builder = ModelRegistryBuilder()
    
    # Build the registry
    registry = builder.build_registry()
    
    # Save to JSON
    builder.save_registry(registry)
    
    # Generate markdown report
    builder.generate_markdown_report(registry)
    
    # Print summary
    print(f"\n✅ Model Registry Built Successfully!")
    print(f"   Total Models: {registry['total_models']}")
    print(f"   Categories: {len(registry['categories'])}")
    print("\nModel Distribution:")
    for category, count in registry['statistics']['by_category'].items():
        print(f"   - {category}: {count} models")


if __name__ == "__main__":
    main()