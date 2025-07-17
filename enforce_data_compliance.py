#!/usr/bin/env python3
"""
🔒 ENFORCE DATA COMPLIANCE ACROSS ALL MODELS
This script enforces strict data compliance rules:
1. Only Polygon.io and FRED data sources allowed
2. No synthetic or placeholder data
3. All data must be tagged with source
4. Models requiring alternative data are sandboxed
"""

import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Set, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Models identified as requiring alternative data sources
SANDBOXED_MODELS = {
    # NLP/Sentiment Models
    "finbert_sentiment_classifier": "Requires real news/social media data",
    "bert_classification_head": "Requires labeled text data from web",
    "sentence_transformer_embedding": "Requires text data pairs",
    
    # Market Microstructure Models
    "market_impact_models": "Requires real order book data",
    "order_flow_analysis": "Requires real order flow data",
    "bid_ask_spread_analysis": "Requires real quote/trade data",
    "liquidity_assessment": "Requires real market depth data",
    "order_book_imbalance_model": "Requires real order book data",
    
    # Alternative Data Models
    "alternative_data_model": "Explicitly requires satellite/alternative data",
    "network_topology_analysis": "Requires graph/network data",
    "vision_transformer_chart": "Requires chart images",
    "graph_neural_network": "Requires graph data structures",
    
    # Reinforcement Learning Models (often used for crypto)
    "q_learning": "Often used with crypto/DeFi data",
    "double_q_learning": "Often used with crypto/DeFi data",
    "dueling_dqn": "Often used with crypto/DeFi data",
    "dqn": "Often used with crypto/DeFi data",
    "regime_aware_dqn": "Often used with market regime data",
    
    # Advanced Models
    "adversarial_momentum_net": "Requires specialized momentum data",
    "fractal_model": "Requires high-frequency fractal analysis data",
    "wavelet_transform_model": "Requires signal processing data",
    "empirical_mode_decomposition": "Requires decomposition data",
    "neural_ode_financial": "Requires continuous time series data",
    "hurst_exponent_fractal": "Requires fractal time series data",
    "threshold_autoregressive": "Requires nonlinear time series data",
    "model_architecture_search": "Requires diverse training data",
    "lempel_ziv_complexity": "Requires binary sequence data",
    
    # Transformer Models
    "transformer": "Requires tokenized sequential data",
    "temporal_fusion_transformer": "Requires multi-modal time series data",
    
    # Risk Models
    "market_stress_indicators": "Requires stress scenario data"
}

class DataComplianceEnforcer:
    """Enforce data compliance across the entire system"""
    
    def __init__(self):
        self.approved_sources = ["polygon", "fred"]
        self.violations = []
        self.sandboxed_count = 0
        self.compliant_count = 0
        
    def check_data_source_compliance(self, file_path: str) -> List[Dict[str, Any]]:
        """Check a file for data source compliance violations"""
        violations = []
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Check for synthetic data generators
            synthetic_patterns = [
                r'np\.random\.',
                r'random\.',
                r'torch\.randn',
                r'tf\.random',
                r'generate.*data',
                r'synthetic.*data',
                r'fake.*data',
                r'mock.*data',
                r'placeholder.*data',
                r'dummy.*data'
            ]
            
            import re
            for pattern in synthetic_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    violations.append({
                        'file': file_path,
                        'type': 'synthetic_data',
                        'pattern': pattern,
                        'severity': 'HIGH'
                    })
            
            # Check for unapproved data sources
            unapproved_sources = [
                'satellite', 'weather', 'social_media', 'twitter',
                'blockchain', 'ethereum', 'crypto', 'defi',
                'web_scraping', 'beautifulsoup', 'scrapy',
                'alternative_data', 'quandl', 'alpha_vantage'
            ]
            
            for source in unapproved_sources:
                if source in content.lower():
                    violations.append({
                        'file': file_path,
                        'type': 'unapproved_source',
                        'source': source,
                        'severity': 'HIGH'
                    })
            
            # Check for proper data tagging
            if 'get_data' in content or 'fetch_data' in content:
                if 'tag_data_source' not in content and 'data_lineage' not in content:
                    violations.append({
                        'file': file_path,
                        'type': 'missing_data_tagging',
                        'severity': 'MEDIUM'
                    })
                    
        except Exception as e:
            logger.error(f"Error checking {file_path}: {e}")
            
        return violations
    
    def sandbox_model(self, model_name: str, reason: str):
        """Sandbox a model that requires alternative data"""
        logger.warning(f"SANDBOXING MODEL: {model_name}")
        logger.warning(f"  Reason: {reason}")
        
        # Create sandbox marker file
        sandbox_dir = "sandboxed_models"
        os.makedirs(sandbox_dir, exist_ok=True)
        
        sandbox_file = os.path.join(sandbox_dir, f"{model_name}.sandbox")
        with open(sandbox_file, 'w') as f:
            json.dump({
                'model': model_name,
                'reason': reason,
                'sandboxed_date': datetime.now().isoformat(),
                'status': 'DISABLED'
            }, f, indent=2)
        
        self.sandboxed_count += 1
    
    def enforce_compliance(self):
        """Main enforcement function"""
        logger.info("Starting data compliance enforcement...")
        
        # 1. Sandbox all models requiring alternative data
        logger.info("\n=== SANDBOXING MODELS ===")
        for model_name, reason in SANDBOXED_MODELS.items():
            self.sandbox_model(model_name, reason)
        
        # 2. Check all Python files for violations
        logger.info("\n=== CHECKING SOURCE FILES ===")
        for root, dirs, files in os.walk('.'):
            # Skip virtual environments and cache
            if any(skip in root for skip in ['.venv', '__pycache__', '.git', 'sandboxed_models']):
                continue
                
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    violations = self.check_data_source_compliance(file_path)
                    if violations:
                        self.violations.extend(violations)
        
        # 3. Generate compliance report
        self.generate_compliance_report()
        
        # 4. Update model configurations
        self.update_model_configs()
        
        return len(self.violations) == 0
    
    def generate_compliance_report(self):
        """Generate comprehensive compliance report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'sandboxed_models': self.sandboxed_count,
            'violations_found': len(self.violations),
            'enforcement_actions': [],
            'compliance_status': 'FAIL' if self.violations else 'PASS'
        }
        
        # Group violations by type
        violations_by_type = {}
        for violation in self.violations:
            vtype = violation['type']
            if vtype not in violations_by_type:
                violations_by_type[vtype] = []
            violations_by_type[vtype].append(violation)
        
        report['violations_by_type'] = violations_by_type
        
        # Save JSON report
        with open('DATA_COMPLIANCE_REPORT.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate Markdown report
        with open('DATA_COMPLIANCE_REPORT.md', 'w') as f:
            f.write("# Data Compliance Enforcement Report\n\n")
            f.write(f"Generated: {report['timestamp']}\n\n")
            f.write(f"## Status: {report['compliance_status']}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- Models Sandboxed: {self.sandboxed_count}\n")
            f.write(f"- Violations Found: {len(self.violations)}\n\n")
            
            f.write("## Sandboxed Models\n\n")
            f.write("The following models have been disabled due to alternative data requirements:\n\n")
            
            for model, reason in sorted(SANDBOXED_MODELS.items()):
                f.write(f"- **{model}**: {reason}\n")
            
            if self.violations:
                f.write("\n## Violations Found\n\n")
                for vtype, violations in violations_by_type.items():
                    f.write(f"### {vtype.replace('_', ' ').title()}\n\n")
                    for v in violations[:5]:  # Show first 5
                        f.write(f"- File: `{v['file']}`\n")
                        if 'pattern' in v:
                            f.write(f"  - Pattern: `{v['pattern']}`\n")
                        if 'source' in v:
                            f.write(f"  - Source: `{v['source']}`\n")
                    if len(violations) > 5:
                        f.write(f"  - ... and {len(violations) - 5} more\n")
                    f.write("\n")
            
            f.write("## Compliance Rules Enforced\n\n")
            f.write("1. **Approved Data Sources Only**: Polygon.io and FRED\n")
            f.write("2. **No Synthetic Data**: All random/generated data prohibited\n")
            f.write("3. **Data Lineage Required**: All data must be tagged with source\n")
            f.write("4. **Alternative Data Models Sandboxed**: 27 models disabled\n\n")
            
            f.write("## Next Steps\n\n")
            f.write("1. Review and fix all violations\n")
            f.write("2. Integrate approved data sources for sandboxed models\n")
            f.write("3. Ensure all data flows through compliance system\n")
            f.write("4. Re-run compliance check\n")
        
        logger.info(f"Compliance report generated: DATA_COMPLIANCE_REPORT.md")
    
    def update_model_configs(self):
        """Update model configurations to reflect sandboxing"""
        config_update = {
            'sandboxed_models': list(SANDBOXED_MODELS.keys()),
            'enforcement_date': datetime.now().isoformat(),
            'approved_data_sources': self.approved_sources
        }
        
        with open('sandboxed_models_config.json', 'w') as f:
            json.dump(config_update, f, indent=2)
        
        logger.info("Model configurations updated with sandbox information")

def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("DATA COMPLIANCE ENFORCEMENT SYSTEM")
    print("="*60 + "\n")
    
    enforcer = DataComplianceEnforcer()
    
    # Run enforcement
    success = enforcer.enforce_compliance()
    
    print(f"\n{'='*60}")
    print(f"ENFORCEMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Models Sandboxed: {enforcer.sandboxed_count}")
    print(f"Violations Found: {len(enforcer.violations)}")
    print(f"Status: {'COMPLIANT' if success else 'NON-COMPLIANT'}")
    print(f"\nReports generated:")
    print(f"  - DATA_COMPLIANCE_REPORT.md")
    print(f"  - DATA_COMPLIANCE_REPORT.json")
    print(f"  - sandboxed_models_config.json")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()