"""
Visualization Module
Performance analysis and trading visualization for Meta-Enhanced TPE-ML system
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import logging

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

logger = logging.getLogger(__name__)

class TradingVisualizer:
    """
    Comprehensive visualization system for trading performance and model analysis
    """
    
    def __init__(self, output_dir='visualizations'):
        """Initialize visualizer with output directory"""
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set visualization style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        logger.info(f"Trading visualizer initialized - Output: {self.output_dir}")
    
    def plot_model_confusion_matrix(self, y_true, y_pred, labels, title="Exit Strategy Confusion Matrix"):
        """
        Plot confusion matrix for ML model performance
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Label names
            title: Plot title
        
        Returns:
            str: Saved plot filename
        """
        from sklearn.metrics import confusion_matrix, classification_report
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Confusion matrix heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels, ax=ax1)
        ax1.set_title(title)
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # Classification report visualization
        report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
        
        metrics_df = pd.DataFrame(report).iloc[:-1, :].T
        sns.heatmap(metrics_df.iloc[:, :-1], annot=True, fmt='.3f', 
                   cmap='Greens', ax=ax2)
        ax2.set_title('Classification Metrics')
        
        plt.tight_layout()
        filename = f"{self.output_dir}/confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {filename}")
        return filename
    
    def plot_trades_timeline(self, trades_log, market_data=None):
        """
        Plot timeline of trades with regime overlays
        
        Args:
            trades_log: List of trade records
            market_data: Market data for context
        
        Returns:
            str: Saved plot filename
        """
        if not trades_log:
            logger.warning("No trades to visualize")
            return None
        
        trades_df = pd.DataFrame(trades_log)
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
        trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
        
        if PLOTLY_AVAILABLE:
            return self._plot_interactive_timeline(trades_df, market_data)
        else:
            return self._plot_static_timeline(trades_df)
    
    def _plot_interactive_timeline(self, trades_df, market_data):
        """Create interactive timeline with Plotly"""
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Portfolio Performance', 'Trade Returns', 'Market Regime'),
            row_heights=[0.5, 0.3, 0.2]
        )
        
        # Portfolio cumulative returns
        trades_df = trades_df.sort_values('entry_date')
        cumulative_returns = (1 + trades_df['return']).cumprod()
        
        fig.add_trace(
            go.Scatter(x=trades_df['entry_date'], y=cumulative_returns,
                      mode='lines', name='Portfolio Value', line=dict(width=3)),
            row=1, col=1
        )
        
        # Individual trade returns
        colors = ['green' if r > 0 else 'red' for r in trades_df['return']]
        fig.add_trace(
            go.Scatter(x=trades_df['entry_date'], y=trades_df['return'],
                      mode='markers', name='Trade Returns',
                      marker=dict(color=colors, size=8),
                      text=trades_df['symbol'],
                      hovertemplate='<b>%{text}</b><br>Return: %{y:.2%}<extra></extra>'),
            row=2, col=1
        )
        
        # Market regime indicators
        regime_colors = {
            'crisis_regime': 'red',
            'high_volatility_rising': 'orange',
            'high_volatility_stable': 'yellow',
            'moderate_volatility_normal': 'lightblue',
            'low_volatility_normal': 'lightgreen',
            'low_volatility_complacent': 'green'
        }
        
        for regime in trades_df['market_regime'].unique():
            regime_trades = trades_df[trades_df['market_regime'] == regime]
            fig.add_trace(
                go.Scatter(x=regime_trades['entry_date'], 
                          y=[regime] * len(regime_trades),
                          mode='markers', name=regime,
                          marker=dict(color=regime_colors.get(regime, 'gray'), size=6)),
                row=3, col=1
            )
        
        fig.update_layout(height=800, title_text="Trading Performance Analysis")
        
        filename = f"{self.output_dir}/trades_timeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(filename)
        
        logger.info(f"Interactive timeline saved to {filename}")
        return filename
    
    def _plot_static_timeline(self, trades_df):
        """Create static timeline with matplotlib"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        # Portfolio performance
        trades_df = trades_df.sort_values('entry_date')
        cumulative_returns = (1 + trades_df['return']).cumprod()
        
        ax1.plot(trades_df['entry_date'], cumulative_returns, linewidth=2, color='blue')
        ax1.set_title('Portfolio Cumulative Returns')
        ax1.set_ylabel('Cumulative Return')
        ax1.grid(True, alpha=0.3)
        
        # Trade returns scatter
        colors = ['green' if r > 0 else 'red' for r in trades_df['return']]
        ax2.scatter(trades_df['entry_date'], trades_df['return'], c=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_title('Individual Trade Returns')
        ax2.set_ylabel('Return')
        ax2.grid(True, alpha=0.3)
        
        # Regime analysis
        regime_counts = trades_df['market_regime'].value_counts()
        ax3.bar(range(len(regime_counts)), regime_counts.values)
        ax3.set_xticks(range(len(regime_counts)))
        ax3.set_xticklabels(regime_counts.index, rotation=45)
        ax3.set_title('Trades by Market Regime')
        ax3.set_ylabel('Count')
        
        plt.tight_layout()
        filename = f"{self.output_dir}/trades_timeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Static timeline saved to {filename}")
        return filename
    
    def plot_regime_performance(self, trades_log):
        """
        Plot performance analysis by market regime
        
        Args:
            trades_log: List of trade records
        
        Returns:
            str: Saved plot filename
        """
        if not trades_log:
            return None
        
        trades_df = pd.DataFrame(trades_log)
        
        # Analyze performance by regime
        regime_analysis = trades_df.groupby('market_regime').agg({
            'return': ['mean', 'std', 'count'],
            'hold_days': 'mean'
        }).round(4)
        
        regime_analysis.columns = ['Avg_Return', 'Return_Std', 'Trade_Count', 'Avg_Hold_Days']
        regime_analysis['Win_Rate'] = trades_df.groupby('market_regime')['return'].apply(
            lambda x: (x > 0).sum() / len(x)
        ).round(3)
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Average returns by regime
        regime_analysis['Avg_Return'].plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title('Average Return by Market Regime')
        ax1.set_ylabel('Average Return')
        ax1.tick_params(axis='x', rotation=45)
        
        # Win rate by regime
        regime_analysis['Win_Rate'].plot(kind='bar', ax=ax2, color='lightgreen')
        ax2.set_title('Win Rate by Market Regime')
        ax2.set_ylabel('Win Rate')
        ax2.tick_params(axis='x', rotation=45)
        
        # Trade count by regime
        regime_analysis['Trade_Count'].plot(kind='bar', ax=ax3, color='coral')
        ax3.set_title('Trade Count by Market Regime')
        ax3.set_ylabel('Number of Trades')
        ax3.tick_params(axis='x', rotation=45)
        
        # Return distribution by regime
        for regime in trades_df['market_regime'].unique():
            regime_returns = trades_df[trades_df['market_regime'] == regime]['return']
            ax4.hist(regime_returns, alpha=0.6, label=regime, bins=20)
        
        ax4.set_title('Return Distribution by Regime')
        ax4.set_xlabel('Return')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        
        plt.tight_layout()
        filename = f"{self.output_dir}/regime_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Regime performance analysis saved to {filename}")
        return filename
    
    def plot_strategy_comparison(self, trades_log):
        """
        Plot comparison of different exit strategies
        
        Args:
            trades_log: List of trade records
        
        Returns:
            str: Saved plot filename
        """
        if not trades_log:
            return None
        
        trades_df = pd.DataFrame(trades_log)
        
        # Analyze by exit strategy
        strategy_analysis = trades_df.groupby('exit_strategy').agg({
            'return': ['mean', 'std', 'count'],
            'hold_days': 'mean'
        }).round(4)
        
        strategy_analysis.columns = ['Avg_Return', 'Return_Std', 'Trade_Count', 'Avg_Hold_Days']
        strategy_analysis['Win_Rate'] = trades_df.groupby('exit_strategy')['return'].apply(
            lambda x: (x > 0).sum() / len(x)
        ).round(3)
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Strategy performance metrics
        strategy_analysis[['Avg_Return', 'Win_Rate']].plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Strategy Performance Comparison')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Hold days comparison
        strategy_analysis['Avg_Hold_Days'].plot(kind='bar', ax=axes[0,1], color='orange')
        axes[0,1].set_title('Average Holding Period by Strategy')
        axes[0,1].set_ylabel('Days')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Return distributions
        for strategy in trades_df['exit_strategy'].unique():
            strategy_returns = trades_df[trades_df['exit_strategy'] == strategy]['return']
            axes[1,0].hist(strategy_returns, alpha=0.6, label=strategy, bins=15)
        
        axes[1,0].set_title('Return Distribution by Exit Strategy')
        axes[1,0].set_xlabel('Return')
        axes[1,0].legend()
        
        # Trade count by strategy
        strategy_analysis['Trade_Count'].plot(kind='pie', ax=axes[1,1], autopct='%1.1f%%')
        axes[1,1].set_title('Trade Distribution by Strategy')
        
        plt.tight_layout()
        filename = f"{self.output_dir}/strategy_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Strategy comparison saved to {filename}")
        return filename
    
    def create_performance_dashboard(self, trades_log, model_metrics=None):
        """
        Create comprehensive performance dashboard
        
        Args:
            trades_log: Trade records
            model_metrics: ML model performance metrics
        
        Returns:
            list: List of generated visualization files
        """
        logger.info("Creating comprehensive performance dashboard...")
        
        generated_files = []
        
        if trades_log:
            # Trading performance visualizations
            timeline_file = self.plot_trades_timeline(trades_log)
            if timeline_file:
                generated_files.append(timeline_file)
            
            regime_file = self.plot_regime_performance(trades_log)
            if regime_file:
                generated_files.append(regime_file)
            
            strategy_file = self.plot_strategy_comparison(trades_log)
            if strategy_file:
                generated_files.append(strategy_file)
        
        logger.info(f"Dashboard created with {len(generated_files)} visualizations")
        return generated_files

def create_visualizer(output_dir='visualizations'):
    """Factory function to create visualizer"""
    return TradingVisualizer(output_dir)