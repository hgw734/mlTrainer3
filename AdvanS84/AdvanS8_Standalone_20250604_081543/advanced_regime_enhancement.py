"""
Advanced Regime Enhancement System
Implements micro-regime layers, multi-timeframe inference, and confidence-adaptive switching
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

class AdvancedRegimeEnhancer:
    """
    Advanced regime classification with micro-regimes and multi-timeframe analysis
    """
    
    def __init__(self):
        self.macro_regimes = ['bull', 'bear', 'neutral', 'volatile']
        self.micro_regimes = [
            'bull_trending', 'bull_consolidating', 'bull_stable',
            'bear_trending', 'bear_bouncing', 'bear_stabilizing',
            'neutral_range', 'neutral_drift', 'volatile_choppy'
        ]
        self.regime_memory = {}
        self.confidence_threshold = 0.85
        self.hysteresis_buffer = 3
        self.regime_counters = {}
        
    def load_current_signals(self):
        """Load current signal data for enhancement"""
        try:
            with open('optimized_ml_signals.json', 'r') as f:
                data = json.load(f)
                return data.get('current_signals', {})
        except:
            return {}
    
    def calculate_multi_timeframe_features(self, symbol_data):
        """Calculate features across multiple timeframes"""
        features = {}
        
        # Hourly momentum features
        if len(symbol_data) >= 24:
            hourly_returns = np.diff(symbol_data[-24:])  # Last 24 periods
            features['hourly_momentum'] = np.mean(hourly_returns[-6:])  # Last 6 periods
            features['hourly_volatility'] = np.std(hourly_returns[-6:])
            features['hourly_trend'] = np.polyfit(range(6), symbol_data[-6:], 1)[0]
        
        # Daily momentum features
        if len(symbol_data) >= 5:
            daily_change = (symbol_data[-1] - symbol_data[-5]) / symbol_data[-5]
            features['daily_momentum'] = daily_change
            features['daily_volatility'] = np.std(symbol_data[-5:]) / np.mean(symbol_data[-5:])
            features['daily_trend'] = np.polyfit(range(5), symbol_data[-5:], 1)[0]
        
        # Weekly features
        if len(symbol_data) >= 20:
            weekly_change = (symbol_data[-1] - symbol_data[-20]) / symbol_data[-20]
            features['weekly_momentum'] = weekly_change
            features['weekly_volatility'] = np.std(symbol_data[-20:]) / np.mean(symbol_data[-20:])
        
        return features
    
    def classify_macro_regime(self, features):
        """Classify macro regime using enhanced logic"""
        daily_momentum = features.get('daily_momentum', 0)
        daily_volatility = features.get('daily_volatility', 0)
        weekly_momentum = features.get('weekly_momentum', 0)
        
        # Enhanced regime classification
        if daily_momentum > 0.03 and weekly_momentum > 0.05:
            if daily_volatility < 0.02:
                return 'bull', 0.9
            else:
                return 'bull', 0.75
        elif daily_momentum < -0.03 and weekly_momentum < -0.05:
            if daily_volatility < 0.02:
                return 'bear', 0.9
            else:
                return 'bear', 0.75
        elif daily_volatility > 0.04:
            return 'volatile', 0.8
        else:
            return 'neutral', 0.7
    
    def classify_micro_regime(self, macro_regime, features):
        """Classify micro-regime within macro regime"""
        hourly_momentum = features.get('hourly_momentum', 0)
        hourly_volatility = features.get('hourly_volatility', 0)
        daily_trend = features.get('daily_trend', 0)
        
        if macro_regime == 'bull':
            if abs(daily_trend) > 0.001:
                return 'bull_trending', 0.85
            elif hourly_volatility < 0.01:
                return 'bull_stable', 0.9
            else:
                return 'bull_consolidating', 0.8
        
        elif macro_regime == 'bear':
            if daily_trend < -0.001:
                return 'bear_trending', 0.85
            elif hourly_momentum > 0:
                return 'bear_bouncing', 0.75
            else:
                return 'bear_stabilizing', 0.8
        
        elif macro_regime == 'neutral':
            if abs(hourly_momentum) < 0.005:
                return 'neutral_range', 0.8
            else:
                return 'neutral_drift', 0.7
        
        else:  # volatile
            return 'volatile_choppy', 0.85
    
    def apply_confidence_adaptive_switching(self, symbol, new_regime, confidence):
        """Apply hysteresis and confidence thresholds for regime switching"""
        current_regime = self.regime_memory.get(symbol, {}).get('regime', 'neutral')
        counter = self.regime_counters.get(symbol, 0)
        
        if new_regime != current_regime:
            if confidence > self.confidence_threshold or counter >= self.hysteresis_buffer:
                # Switch regime
                self.regime_memory[symbol] = {
                    'regime': new_regime,
                    'confidence': confidence,
                    'timestamp': datetime.now().isoformat(),
                    'switch_reason': 'high_confidence' if confidence > self.confidence_threshold else 'hysteresis_met'
                }
                self.regime_counters[symbol] = 0
                return new_regime, True  # Regime changed
            else:
                # Hold current regime, increment counter
                self.regime_counters[symbol] = counter + 1
                return current_regime, False  # No change
        else:
            # Same regime, reset counter
            self.regime_counters[symbol] = 0
            self.regime_memory[symbol] = {
                'regime': new_regime,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat(),
                'switch_reason': 'stability'
            }
            return new_regime, False
    
    def calculate_regime_attention_weights(self, features):
        """Calculate attention weights for regime decision explainability"""
        feature_importance = {}
        
        # Simple attention mechanism based on feature variance and correlation
        for key, value in features.items():
            if abs(value) > 0.01:  # Significant feature
                feature_importance[key] = min(1.0, abs(value) * 10)
            else:
                feature_importance[key] = 0.1
        
        # Normalize to sum to 1
        total_importance = sum(feature_importance.values())
        if total_importance > 0:
            feature_importance = {k: v/total_importance for k, v in feature_importance.items()}
        
        return feature_importance
    
    def generate_realistic_price_data(self, symbol, periods=60):
        """DISABLED - Only authentic Polygon API data allowed"""
        # This function has been disabled to enforce authentic data integrity
        # All regime analysis must use fetch_authentic_polygon_data() instead
        return None
        
        # Convert to prices
        prices = [base_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        return prices
    
    def enhance_symbol_regime(self, symbol, transformer_signal=None):
        """Enhanced regime classification for a single symbol"""
        try:
            # Generate realistic market data
            price_data = self.generate_realistic_price_data(symbol)
            
            # Calculate multi-timeframe features
            features = self.calculate_multi_timeframe_features(price_data)
            
            # Classify macro regime
            macro_regime, macro_confidence = self.classify_macro_regime(features)
            
            # Classify micro regime
            micro_regime, micro_confidence = self.classify_micro_regime(macro_regime, features)
            
            # Apply confidence-adaptive switching
            final_regime, regime_changed = self.apply_confidence_adaptive_switching(
                symbol, macro_regime, macro_confidence
            )
            
            # Calculate attention weights for explainability
            attention_weights = self.calculate_regime_attention_weights(features)
            
            # Enhanced regime data
            enhanced_data = {
                'symbol': symbol,
                'macro_regime': final_regime,
                'micro_regime': micro_regime,
                'macro_confidence': macro_confidence,
                'micro_confidence': micro_confidence,
                'regime_changed': regime_changed,
                'features': features,
                'attention_weights': attention_weights,
                'enhancement_timestamp': datetime.now().isoformat(),
                'transformer_signal': transformer_signal
            }
            
            return enhanced_data
            
        except Exception as e:
            print(f"Error enhancing regime for {symbol}: {e}")
            return None
    
    def run_advanced_enhancement_cycle(self):
        """Run advanced regime enhancement cycle"""
        print(f"Advanced Regime Enhancement - {datetime.now().strftime('%H:%M:%S')}")
        
        # Load current signals
        current_signals = self.load_current_signals()
        transformer_signals = current_signals.get('transformer', {})
        
        if not transformer_signals:
            print("No transformer signals available for enhancement")
            return {}
        
        enhanced_regimes = {}
        regime_changes = 0
        confidence_improvements = 0
        
        print(f"Enhancing regime classification for {len(transformer_signals)} symbols...")
        
        for symbol, signal_data in transformer_signals.items():
            enhanced_regime = self.enhance_symbol_regime(symbol, signal_data)
            
            if enhanced_regime:
                enhanced_regimes[symbol] = enhanced_regime
                
                if enhanced_regime['regime_changed']:
                    regime_changes += 1
                
                if enhanced_regime['macro_confidence'] > 0.8:
                    confidence_improvements += 1
        
        # Calculate enhancement statistics
        enhancement_stats = {
            'total_symbols': len(transformer_signals),
            'enhanced_symbols': len(enhanced_regimes),
            'regime_changes': regime_changes,
            'confidence_improvements': confidence_improvements,
            'change_rate': regime_changes / len(enhanced_regimes) if enhanced_regimes else 0,
            'confidence_rate': confidence_improvements / len(enhanced_regimes) if enhanced_regimes else 0
        }
        
        print(f"Enhanced {len(enhanced_regimes)} symbols, {regime_changes} regime changes")
        print(f"Confidence improvements: {confidence_improvements}")
        print(f"Change rate: {enhancement_stats['change_rate']:.1%}")
        
        return {
            'enhanced_regimes': enhanced_regimes,
            'statistics': enhancement_stats,
            'timestamp': datetime.now().isoformat()
        }
    
    def save_advanced_results(self, results):
        """Save advanced enhancement results"""
        with open('advanced_regime_enhancement.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Advanced enhancement results saved to: advanced_regime_enhancement.json")
    
    def display_enhancement_summary(self, results):
        """Display comprehensive enhancement summary"""
        stats = results.get('statistics', {})
        enhanced_regimes = results.get('enhanced_regimes', {})
        
        print(f"\nAdvanced Regime Enhancement Summary:")
        print(f"=" * 60)
        print(f"Total symbols processed: {stats.get('total_symbols', 0)}")
        print(f"Successfully enhanced: {stats.get('enhanced_symbols', 0)}")
        print(f"Regime changes: {stats.get('regime_changes', 0)} ({stats.get('change_rate', 0):.1%})")
        print(f"High confidence signals: {stats.get('confidence_improvements', 0)}")
        
        # Regime distribution
        if enhanced_regimes:
            macro_regimes = [r['macro_regime'] for r in enhanced_regimes.values()]
            micro_regimes = [r['micro_regime'] for r in enhanced_regimes.values()]
            
            from collections import Counter
            macro_dist = Counter(macro_regimes)
            micro_dist = Counter(micro_regimes)
            
            print(f"\nMacro Regime Distribution:")
            for regime, count in macro_dist.items():
                print(f"  {regime}: {count} ({count/len(enhanced_regimes):.1%})")
            
            print(f"\nMicro Regime Distribution:")
            for regime, count in micro_dist.most_common(5):
                print(f"  {regime}: {count} ({count/len(enhanced_regimes):.1%})")
            
            # Show high-confidence examples
            high_conf_examples = [(s, r) for s, r in enhanced_regimes.items() 
                                if r['macro_confidence'] > 0.85][:5]
            
            if high_conf_examples:
                print(f"\nHigh Confidence Examples:")
                for symbol, regime_data in high_conf_examples:
                    print(f"  {symbol}: {regime_data['macro_regime']}/{regime_data['micro_regime']} "
                          f"(conf: {regime_data['macro_confidence']:.3f})")

def main():
    """Main function for advanced regime enhancement"""
    print("Advanced Regime Enhancement System")
    print("Multi-timeframe analysis with micro-regimes and confidence-adaptive switching")
    print("=" * 80)
    
    # Initialize advanced enhancer
    enhancer = AdvancedRegimeEnhancer()
    
    # Run enhancement cycle
    results = enhancer.run_advanced_enhancement_cycle()
    
    if results and results.get('enhanced_regimes'):
        # Save results
        enhancer.save_advanced_results(results)
        
        # Display summary
        enhancer.display_enhancement_summary(results)
        
        print(f"\nAdvanced regime enhancement complete!")
        print(f"Enhanced regime data available for trading strategy optimization.")
        
    else:
        print("No enhancement results generated. Check signal availability.")

if __name__ == "__main__":
    main()