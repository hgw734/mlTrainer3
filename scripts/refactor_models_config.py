#!/usr/bin/env python3
"""
Refactor Models Config Script
"""

import os
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def refactor_models_config():
    """Refactor models configuration"""
    logger.info("🔧 Refactoring Models Config")
    logger.info("=" * 50)
    
    # Define new model configurations
    model_configs = {
        "momentum": {
            "MomentumBreakout": {
                "window": 20,
                "threshold": 0.02,
                "description": "Momentum breakout system"
            },
            "EMACrossover": {
                "short_window": 12,
                "long_window": 26,
                "description": "EMA crossover system"
            }
        },
        "risk": {
            "InformationRatio": {
                "benchmark_return": 0.0,
                "description": "Information ratio calculation"
            },
            "ExpectedShortfall": {
                "confidence_level": 0.95,
                "description": "Expected shortfall (CVaR)"
            },
            "MaximumDrawdown": {
                "window": 252,
                "description": "Maximum drawdown calculation"
            }
        },
        "volatility": {
            "RegimeSwitchingVolatility": {
                "window": 20,
                "description": "Regime switching volatility model"
            },
            "VolatilitySurface": {
                "maturity_steps": 5,
                "description": "Volatility surface model"
            }
        }
    }
    
    # Update config files
    for category, models in model_configs.items():
        update_model_config(category, models)
    
    logger.info("✅ Models config refactored successfully!")

def update_model_config(category: str, models: Dict[str, Any]):
    """Update model configuration for a category"""
    config_file = f"config/models_config.py"
    
    if not os.path.exists(config_file):
        logger.warning(f"⚠️  Config file {config_file} does not exist")
        return
    
    try:
        with open(config_file, 'r') as f:
            content = f.read()
        
        # Add or update model configurations
        for model_name, config in models.items():
            if model_name not in content:
                # Add new model config
                config_section = f'''
# {model_name} Configuration
{model_name.upper()}_CONFIG = {{
    "description": "{config['description']}",
    "parameters": {config}
}}
'''
                content += config_section
                logger.info(f"✅ Added {model_name} config to {config_file}")
        
        # Write updated content
        with open(config_file, 'w') as f:
            f.write(content)
        
    except Exception as e:
        logger.error(f"❌ Error updating {config_file}: {e}")

def create_models_config():
    """Create models configuration file"""
    config_content = '''#!/usr/bin/env python3
"""
Models Configuration
"""

# Momentum Models
MOMENTUM_BREAKOUT_CONFIG = {
    "description": "Momentum breakout system",
    "parameters": {
        "window": 20,
        "threshold": 0.02
    }
}

EMA_CROSSOVER_CONFIG = {
    "description": "EMA crossover system",
    "parameters": {
        "short_window": 12,
        "long_window": 26
    }
}

# Risk Models
INFORMATION_RATIO_CONFIG = {
    "description": "Information ratio calculation",
    "parameters": {
        "benchmark_return": 0.0
    }
}

EXPECTED_SHORTFALL_CONFIG = {
    "description": "Expected shortfall (CVaR)",
    "parameters": {
        "confidence_level": 0.95
    }
}

MAXIMUM_DRAWDOWN_CONFIG = {
    "description": "Maximum drawdown calculation",
    "parameters": {
        "window": 252
    }
}

# Volatility Models
REGIME_SWITCHING_VOLATILITY_CONFIG = {
    "description": "Regime switching volatility model",
    "parameters": {
        "window": 20
    }
}

VOLATILITY_SURFACE_CONFIG = {
    "description": "Volatility surface model",
    "parameters": {
        "maturity_steps": 5
    }
}

# Model Registry
MODEL_REGISTRY = {
    "momentum": {
        "MomentumBreakout": MOMENTUM_BREAKOUT_CONFIG,
        "EMACrossover": EMA_CROSSOVER_CONFIG
    },
    "risk": {
        "InformationRatio": INFORMATION_RATIO_CONFIG,
        "ExpectedShortfall": EXPECTED_SHORTFALL_CONFIG,
        "MaximumDrawdown": MAXIMUM_DRAWDOWN_CONFIG
    },
    "volatility": {
        "RegimeSwitchingVolatility": REGIME_SWITCHING_VOLATILITY_CONFIG,
        "VolatilitySurface": VOLATILITY_SURFACE_CONFIG
    }
}
'''
    
    config_file = "config/models_config.py"
    try:
        with open(config_file, 'w') as f:
            f.write(config_content)
        logger.info(f"✅ Created {config_file}")
    except Exception as e:
        logger.error(f"❌ Error creating {config_file}: {e}")

def main():
    """Main function"""
    # Create models config if it doesn't exist
    if not os.path.exists("config/models_config.py"):
        create_models_config()
    
    # Refactor existing config
    refactor_models_config()

if __name__ == "__main__":
    main() 