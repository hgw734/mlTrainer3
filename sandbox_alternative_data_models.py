#!/usr/bin/env python3
"""
🔒 SANDBOX FOR ALTERNATIVE DATA MODELS
Models requiring data sources beyond Polygon.io and FRED are sandboxed here.
These models cannot be used in production until proper data sources are integrated.
"""

import logging
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from config.models_config import MATHEMATICAL_MODELS
from config.api_config import APISource, get_all_approved_sources
from core.compliance_mode import ComplianceMode
from data_lineage_system import DataLineageSystem

logger = logging.getLogger(__name__)

@dataclass
class SandboxedModel:
    """Information about a sandboxed model"""
    model_name: str
    category: str
    subcategory: str
    required_data_sources: List[str]
    reason: str
    sandboxed_date: datetime
    compliance_notes: str

class AlternativeDataModelSandbox:
    """
    Sandbox for models requiring alternative data sources.
    These models are identified and disabled until proper data integration.
    """
    
    # Define models that require alternative data sources
    ALTERNATIVE_DATA_MODELS = {
        # Models requiring satellite imagery
        "satellite_based_models": [
            "alternative_data_model",  # Requires satellite imagery APIs
        ],
        
        # Models requiring web scraping
        "web_scraping_models": [
            "finbert_sentiment_classifier",  # Needs real news/social media APIs
            "bert_classification_head",  # Needs real text data from web
            "sentence_transformer_embedding",  # Needs real text data
        ],
        
        # Models requiring supply chain data
        "supply_chain_models": [
            # Currently none explicitly defined, but would include:
            # - Shipping/port activity models
            # - Supply chain disruption models
        ],
        
        # Models requiring weather data
        "weather_data_models": [
            # Currently none explicitly defined, but would include:
            # - Weather-based commodity models
            # - Climate risk models
        ],
        
        # Models requiring crypto/DeFi data
        "crypto_defi_models": [
            # Currently none explicitly defined, but would include:
            # - On-chain analytics models
            # - DeFi protocol models
            # - MEV models
        ],
        
        # Models requiring market microstructure data
        "microstructure_models": [
            "market_impact_models",  # Needs real order book data
            "order_flow_analysis",  # Needs real order flow data
            "bid_ask_spread_analysis",  # Needs real quote data
            "liquidity_assessment",  # Needs real market depth data
            "order_book_imbalance_model",  # Needs real order book data
        ],
        
        # Models requiring alternative financial data
        "alternative_financial_models": [
            "network_topology_analysis",  # Needs graph/network data
            "vision_transformer_chart",  # Needs chart images
            "graph_neural_network",  # Needs graph data
        ]
    }
    
    def __init__(self):
        self.compliance_mode = ComplianceMode()
        self.lineage_system = DataLineageSystem()
        self.sandboxed_models: Dict[str, SandboxedModel] = {}
        self._identify_and_sandbox_models()
    
    def _identify_and_sandbox_models(self):
        """Identify all models requiring alternative data and sandbox them"""
        logger.info("Identifying models requiring alternative data sources...")
        
        approved_sources = {source.value for source in get_all_approved_sources()}
        logger.info(f"Approved data sources: {approved_sources}")
        
        # Check all models in the configuration
        for model_name, model_config in MATHEMATICAL_MODELS.items():
            data_requirements = model_config.data_requirements
            
            # Check if model requires alternative data
            requires_alternative = False
            required_sources = []
            
            # Check for specific data requirements
            for requirement in data_requirements:
                requirement_lower = requirement.lower()
                
                # Check for alternative data indicators
                if any(indicator in requirement_lower for indicator in [
                    'satellite', 'social', 'news', 'web', 'scraping',
                    'supply_chain', 'shipping', 'port', 'weather', 'climate',
                    'crypto', 'blockchain', 'defi', 'ethereum', 'on_chain',
                    'order_book', 'order_flow', 'quote_data', 'market_depth',
                    'graph_data', 'network_data', 'image', 'chart_image',
                    'alternative_data', 'text_data', 'sentiment_data'
                ]):
                    requires_alternative = True
                    required_sources.append(requirement)
            
            # Check model category and subcategory
            if model_config.category in ['nlp_sentiment', 'market_microstructure']:
                requires_alternative = True
                required_sources.append(f"{model_config.category}_data")
            
            # Check specific model names
            for category, model_list in self.ALTERNATIVE_DATA_MODELS.items():
                if model_name in model_list:
                    requires_alternative = True
                    required_sources.append(category)
            
            # Sandbox the model if it requires alternative data
            if requires_alternative:
                self._sandbox_model(model_name, model_config, required_sources)
    
    def _sandbox_model(self, model_name: str, model_config: Any, required_sources: List[str]):
        """Sandbox a specific model"""
        sandboxed_model = SandboxedModel(
            model_name=model_name,
            category=model_config.category,
            subcategory=model_config.subcategory,
            required_data_sources=required_sources,
            reason=f"Requires data sources not available through Polygon.io or FRED: {', '.join(required_sources)}",
            sandboxed_date=datetime.now(),
            compliance_notes="Model disabled until proper data source integration"
        )
        
        self.sandboxed_models[model_name] = sandboxed_model
        
        # Log the sandboxing
        logger.warning(f"SANDBOXED MODEL: {model_name} - {sandboxed_model.reason}")
        
        # Record in compliance system
        self.compliance_mode.log_action(
            action="model_sandboxed",
            details={
                "model": model_name,
                "reason": sandboxed_model.reason,
                "required_sources": required_sources
            }
        )
    
    def is_model_sandboxed(self, model_name: str) -> bool:
        """Check if a model is sandboxed"""
        return model_name in self.sandboxed_models
    
    def get_sandboxed_models(self) -> Dict[str, SandboxedModel]:
        """Get all sandboxed models"""
        return self.sandboxed_models.copy()
    
    def get_sandbox_report(self) -> Dict[str, Any]:
        """Generate a comprehensive sandbox report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_models": len(MATHEMATICAL_MODELS),
            "sandboxed_models": len(self.sandboxed_models),
            "sandboxed_percentage": f"{(len(self.sandboxed_models) / len(MATHEMATICAL_MODELS) * 100):.1f}%",
            "categories_affected": {},
            "required_data_sources": {},
            "models": {}
        }
        
        # Analyze by category
        category_counts = {}
        source_counts = {}
        
        for model_name, sandboxed in self.sandboxed_models.items():
            # Count by category
            if sandboxed.category not in category_counts:
                category_counts[sandboxed.category] = 0
            category_counts[sandboxed.category] += 1
            
            # Count by required sources
            for source in sandboxed.required_data_sources:
                if source not in source_counts:
                    source_counts[source] = 0
                source_counts[source] += 1
            
            # Add model details
            report["models"][model_name] = {
                "category": sandboxed.category,
                "subcategory": sandboxed.subcategory,
                "required_sources": sandboxed.required_data_sources,
                "reason": sandboxed.reason,
                "sandboxed_date": sandboxed.sandboxed_date.isoformat()
            }
        
        report["categories_affected"] = category_counts
        report["required_data_sources"] = source_counts
        
        return report
    
    def enforce_sandbox(self, model_name: str) -> bool:
        """
        Enforce sandbox restrictions on a model.
        Returns True if model is allowed, False if sandboxed.
        """
        if self.is_model_sandboxed(model_name):
            logger.error(f"Model {model_name} is SANDBOXED and cannot be used")
            
            # Log compliance violation
            self.compliance_mode.log_action(
                action="sandbox_violation_prevented",
                details={
                    "model": model_name,
                    "reason": self.sandboxed_models[model_name].reason
                }
            )
            
            return False
        
        return True
    
    def get_alternative_data_requirements(self) -> Dict[str, List[str]]:
        """Get a summary of all alternative data requirements"""
        requirements = {}
        
        for model_name, sandboxed in self.sandboxed_models.items():
            for source in sandboxed.required_data_sources:
                if source not in requirements:
                    requirements[source] = []
                requirements[source].append(model_name)
        
        return requirements

# Initialize the sandbox
sandbox = AlternativeDataModelSandbox()

def check_model_availability(model_name: str) -> bool:
    """Check if a model is available for use (not sandboxed)"""
    return sandbox.enforce_sandbox(model_name)

def get_sandboxed_models() -> List[str]:
    """Get list of all sandboxed model names"""
    return list(sandbox.get_sandboxed_models().keys())

def generate_sandbox_report():
    """Generate and save sandbox report"""
    report = sandbox.get_sandbox_report()
    
    # Save report
    import json
    with open("SANDBOX_REPORT.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Create markdown report
    with open("SANDBOX_REPORT.md", "w") as f:
        f.write("# Alternative Data Models Sandbox Report\n\n")
        f.write(f"Generated: {report['timestamp']}\n\n")
        f.write("## Summary\n\n")
        f.write(f"- Total Models: {report['total_models']}\n")
        f.write(f"- Sandboxed Models: {report['sandboxed_models']}\n")
        f.write(f"- Sandboxed Percentage: {report['sandboxed_percentage']}\n\n")
        
        f.write("## Categories Affected\n\n")
        for category, count in report['categories_affected'].items():
            f.write(f"- {category}: {count} models\n")
        
        f.write("\n## Required Data Sources\n\n")
        for source, count in report['required_data_sources'].items():
            f.write(f"- {source}: {count} models\n")
        
        f.write("\n## Sandboxed Models\n\n")
        for model_name, details in report['models'].items():
            f.write(f"### {model_name}\n")
            f.write(f"- Category: {details['category']}\n")
            f.write(f"- Subcategory: {details['subcategory']}\n")
            f.write(f"- Required Sources: {', '.join(details['required_sources'])}\n")
            f.write(f"- Reason: {details['reason']}\n")
            f.write(f"- Sandboxed Date: {details['sandboxed_date']}\n\n")
    
    logger.info("Sandbox report generated: SANDBOX_REPORT.md and SANDBOX_REPORT.json")
    return report

if __name__ == "__main__":
    # Generate sandbox report
    report = generate_sandbox_report()
    
    print(f"\nSandbox Report Summary:")
    print(f"Total Models: {report['total_models']}")
    print(f"Sandboxed Models: {report['sandboxed_models']} ({report['sandboxed_percentage']})")
    print(f"\nTop Required Data Sources:")
    for source, count in sorted(report['required_data_sources'].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  - {source}: {count} models")