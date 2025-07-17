#!/usr/bin/env python3
"""
Analyze models requiring alternative data sources
This script identifies models that need data beyond Polygon.io and FRED
"""

import json
import re
from datetime import datetime
from typing import Dict, List, Set, Tuple

def analyze_models_config():
    """Analyze models configuration file for alternative data requirements"""
    
    # Read the models configuration file directly
    with open('config/models_config.py', 'r') as f:
        content = f.read()
    
    # Extract model definitions using regex
    model_pattern = r'\"(\w+)\"\s*:\s*MathematicalModel\s*\((.*?)\),\s*(?=\"|$)'
    matches = re.findall(model_pattern, content, re.DOTALL)
    
    alternative_data_models = {}
    alternative_data_indicators = [
        # Satellite/Alternative Data
        'satellite', 'alternative_data', 'alternative_datasets',
        
        # Web/Social Media Data
        'social', 'news', 'web', 'scraping', 'text_data', 'sentiment_data',
        'financial_text', 'labeled_data', 'sentence_pairs',
        
        # Supply Chain Data
        'supply_chain', 'shipping', 'port',
        
        # Weather Data
        'weather', 'climate', 'noaa',
        
        # Crypto/DeFi Data
        'crypto', 'blockchain', 'defi', 'ethereum', 'on_chain',
        'state_action_pairs', 'reward_signals',  # RL models often used for crypto
        
        # Market Microstructure Data
        'order_book', 'order_flow', 'quote_data', 'market_depth',
        'order_data', 'trade_data', 'order_book_data',
        
        # Graph/Network Data
        'graph_data', 'network_data', 'graph_adjacency',
        
        # Image/Chart Data
        'image', 'chart_image', 'chart_images', '2d_sequential_data',
        
        # Options/Derivatives Data (beyond basic Polygon)
        'options_data', 'option_parameters', 'underlying_price',
        
        # Other Alternative Sources
        'meta_features', 'multiple_tasks', 'distributed_data',
        'continuous_time_series', 'binary_sequences'
    ]
    
    # Categories that typically require alternative data
    alternative_categories = [
        'nlp_sentiment',
        'market_microstructure',
        'cutting_edge_ai',
        'reinforcement_learning'
    ]
    
    for model_name, model_def in matches:
        requires_alternative = False
        reasons = []
        
        # Check data requirements
        data_req_match = re.search(r'data_requirements\s*=\s*\[(.*?)\]', model_def, re.DOTALL)
        if data_req_match:
            data_reqs = data_req_match.group(1).lower()
            for indicator in alternative_data_indicators:
                if indicator in data_reqs:
                    requires_alternative = True
                    reasons.append(f"requires {indicator}")
        
        # Check category
        category_match = re.search(r'category\s*=\s*\"(.*?)\"', model_def)
        if category_match:
            category = category_match.group(1)
            if category in alternative_categories:
                requires_alternative = True
                reasons.append(f"category: {category}")
        
        # Check specific model types
        if any(keyword in model_name.lower() for keyword in [
            'sentiment', 'bert', 'transformer', 'nlp',
            'order', 'market_impact', 'liquidity', 'microstructure',
            'graph', 'network', 'alternative', 'satellite',
            'crypto', 'defi', 'blockchain'
        ]):
            requires_alternative = True
            reasons.append(f"model type: {model_name}")
        
        if requires_alternative:
            alternative_data_models[model_name] = {
                'reasons': list(set(reasons)),
                'category': category_match.group(1) if category_match else 'unknown'
            }
    
    return alternative_data_models

def generate_sandbox_report(models: Dict):
    """Generate a comprehensive sandbox report"""
    
    # Count by category
    category_counts = {}
    for model_info in models.values():
        category = model_info['category']
        if category not in category_counts:
            category_counts[category] = 0
        category_counts[category] += 1
    
    # Count by reason type
    reason_counts = {}
    for model_info in models.values():
        for reason in model_info['reasons']:
            reason_type = reason.split(':')[0].strip()
            if reason_type not in reason_counts:
                reason_counts[reason_type] = 0
            reason_counts[reason_type] += 1
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_models_analyzed': 'approximately 140+',
        'models_requiring_alternative_data': len(models),
        'percentage_requiring_alternative_data': f'{len(models)}+ models',
        'categories_affected': category_counts,
        'reason_types': reason_counts,
        'models': models
    }
    
    # Save JSON report
    with open('ALTERNATIVE_DATA_MODELS_ANALYSIS.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate Markdown report
    with open('ALTERNATIVE_DATA_MODELS_ANALYSIS.md', 'w') as f:
        f.write("# Alternative Data Models Analysis Report\n\n")
        f.write(f"Generated: {report['timestamp']}\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write("This report identifies all models in the mlTrainer system that require ")
        f.write("data sources beyond Polygon.io and FRED APIs.\n\n")
        
        f.write("## Key Findings\n\n")
        f.write(f"- **Models Requiring Alternative Data**: {len(models)}\n")
        f.write("- **Main Data Gaps**:\n")
        f.write("  - Satellite imagery and alternative data feeds\n")
        f.write("  - Web scraping and social media APIs\n")
        f.write("  - Real-time order book and market microstructure data\n")
        f.write("  - Crypto/DeFi blockchain data\n")
        f.write("  - Weather and climate data\n\n")
        
        f.write("## Categories Affected\n\n")
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"- **{category}**: {count} models\n")
        
        f.write("\n## Reason Analysis\n\n")
        for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"- **{reason}**: {count} models\n")
        
        f.write("\n## Sandboxed Models\n\n")
        f.write("The following models must be sandboxed until proper data sources are integrated:\n\n")
        
        # Group by category
        models_by_category = {}
        for model_name, model_info in models.items():
            category = model_info['category']
            if category not in models_by_category:
                models_by_category[category] = []
            models_by_category[category].append((model_name, model_info))
        
        for category in sorted(models_by_category.keys()):
            f.write(f"### {category}\n\n")
            for model_name, model_info in sorted(models_by_category[category]):
                f.write(f"- **{model_name}**\n")
                for reason in model_info['reasons']:
                    f.write(f"  - {reason}\n")
            f.write("\n")
        
        f.write("## Compliance Requirements\n\n")
        f.write("According to the compliance system:\n\n")
        f.write("1. **No Placeholder Data**: All data must originate from approved APIs\n")
        f.write("2. **No Synthetic Data**: No data generators or synthetic data allowed\n")
        f.write("3. **Data Tagging**: All data must be tagged with source indefinitely\n")
        f.write("4. **Current Approved Sources**:\n")
        f.write("   - Polygon.io (market data)\n")
        f.write("   - FRED (economic data)\n\n")
        
        f.write("## Recommendations\n\n")
        f.write("1. **Immediate Action**: Sandbox all identified models\n")
        f.write("2. **Integration Priority**:\n")
        f.write("   - Market microstructure data (for trading models)\n")
        f.write("   - NLP/sentiment data feeds (for sentiment models)\n")
        f.write("   - Alternative data APIs (for advanced models)\n")
        f.write("3. **Compliance**: Ensure all new data sources are properly integrated ")
        f.write("with the data lineage and compliance systems\n")
    
    return report

def main():
    """Main analysis function"""
    print("Analyzing models for alternative data requirements...")
    
    # Analyze models
    alternative_models = analyze_models_config()
    
    print(f"\nFound {len(alternative_models)} models requiring alternative data sources")
    
    # Generate report
    report = generate_sandbox_report(alternative_models)
    
    print("\nReport generated:")
    print("- ALTERNATIVE_DATA_MODELS_ANALYSIS.json")
    print("- ALTERNATIVE_DATA_MODELS_ANALYSIS.md")
    
    # Print summary
    print("\nTop categories requiring alternative data:")
    category_counts = report['categories_affected']
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  - {category}: {count} models")
    
    print("\nModels to sandbox immediately:")
    for model_name in sorted(alternative_models.keys())[:10]:
        print(f"  - {model_name}")
    print(f"  ... and {len(alternative_models) - 10} more")

if __name__ == "__main__":
    main()