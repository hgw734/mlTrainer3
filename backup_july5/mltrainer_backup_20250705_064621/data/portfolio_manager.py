"""
mlTrainer - Portfolio Manager
============================

Purpose: Manages portfolio holdings, tracks performance, and handles
position management for the mlTrainer system. Provides real-time
portfolio monitoring and performance analytics.

Features:
- Holdings management and tracking
- Real-time P&L calculation
- Progress to target monitoring
- Risk management and position sizing
- Portfolio analytics and reporting
"""

import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class PortfolioManager:
    """Manages portfolio holdings and performance tracking"""
    
    def __init__(self):
        self.portfolio_file = "data/portfolio.json"
        self.holdings_history_file = "data/holdings_history.json"
        self.current_portfolio = {}
        self.holdings_history = []
        
        # Ensure data directory exists
        Path("data").mkdir(parents=True, exist_ok=True)
        
        # Risk management parameters
        self.risk_params = {
            "max_position_size": 0.10,  # 10% max per position
            "max_portfolio_risk": 0.20,  # 20% max portfolio risk
            "default_stop_loss": 0.05,  # 5% default stop loss
            "position_sizing_method": "equal_weight"
        }
        
        # Load existing portfolio
        self._load_portfolio()
        
        logger.info("PortfolioManager initialized")
    
    def _load_portfolio(self):
        """Load portfolio from disk"""
        try:
            if os.path.exists(self.portfolio_file):
                with open(self.portfolio_file, 'r') as f:
                    self.current_portfolio = json.load(f)
                logger.info(f"Loaded portfolio with {len(self.current_portfolio)} holdings")
            else:
                self.current_portfolio = {}
                logger.info("Created new portfolio")
        except Exception as e:
            logger.error(f"Failed to load portfolio: {e}")
            self.current_portfolio = {}
        
        # Load holdings history
        try:
            if os.path.exists(self.holdings_history_file):
                with open(self.holdings_history_file, 'r') as f:
                    self.holdings_history = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load holdings history: {e}")
            self.holdings_history = []
    
    def _save_portfolio(self):
        """Save portfolio to disk"""
        try:
            with open(self.portfolio_file, 'w') as f:
                json.dump(self.current_portfolio, f, indent=2, default=str)
            logger.info("Portfolio saved")
        except Exception as e:
            logger.error(f"Failed to save portfolio: {e}")
    
    def _save_holdings_history(self):
        """Save holdings history to disk"""
        try:
            with open(self.holdings_history_file, 'w') as f:
                json.dump(self.holdings_history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save holdings history: {e}")
    
    def add_holding(self, recommendation: Dict[str, Any]) -> Dict[str, Any]:
        """Add a new holding to the portfolio"""
        ticker = recommendation.get("ticker")
        if not ticker:
            return {"success": False, "error": "No ticker specified"}
        
        if ticker in self.current_portfolio:
            return {"success": False, "error": f"{ticker} already in portfolio"}
        
        try:
            # Create holding entry
            holding = {
                "ticker": ticker,
                "entry_date": datetime.now().isoformat(),
                "entry_price": recommendation.get("max_entry_price", 0),
                "target_price": recommendation.get("target_price", 0),
                "stop_loss": recommendation.get("stop_loss", 0),
                "confidence": recommendation.get("confidence", 0),
                "score": recommendation.get("score", 0),
                "timeframe": recommendation.get("timeframe", "unknown"),
                "reasoning": recommendation.get("reasoning", ""),
                "current_price": recommendation.get("max_entry_price", 0),  # Will be updated
                "quantity": 0,  # To be set based on position sizing
                "unrealized_pnl": 0,
                "unrealized_pnl_pct": 0,
                "progress_to_target": 0,
                "status": "active",
                "source": recommendation.get("source", "mlTrainer")
            }
            
            # Calculate position size
            position_size = self._calculate_position_size(holding)
            holding["position_size_pct"] = position_size
            
            # Add to portfolio
            self.current_portfolio[ticker] = holding
            
            # Record in history
            history_entry = {
                "action": "add",
                "ticker": ticker,
                "timestamp": datetime.now().isoformat(),
                "details": holding.copy()
            }
            self.holdings_history.append(history_entry)
            
            # Save changes
            self._save_portfolio()
            self._save_holdings_history()
            
            logger.info(f"Added holding: {ticker}")
            return {"success": True, "holding": holding}
            
        except Exception as e:
            logger.error(f"Failed to add holding {ticker}: {e}")
            return {"success": False, "error": str(e)}
    
    def remove_holding(self, ticker: str, reason: str = "manual_removal") -> Dict[str, Any]:
        """Remove a holding from the portfolio"""
        if ticker not in self.current_portfolio:
            return {"success": False, "error": f"{ticker} not found in portfolio"}
        
        try:
            holding = self.current_portfolio[ticker].copy()
            holding["exit_date"] = datetime.now().isoformat()
            holding["exit_reason"] = reason
            
            # Calculate final P&L
            if holding.get("current_price", 0) > 0 and holding.get("entry_price", 0) > 0:
                final_pnl_pct = (holding["current_price"] - holding["entry_price"]) / holding["entry_price"] * 100
                holding["final_pnl_pct"] = final_pnl_pct
            
            # Remove from current portfolio
            del self.current_portfolio[ticker]
            
            # Record in history
            history_entry = {
                "action": "remove",
                "ticker": ticker,
                "timestamp": datetime.now().isoformat(),
                "reason": reason,
                "details": holding
            }
            self.holdings_history.append(history_entry)
            
            # Save changes
            self._save_portfolio()
            self._save_holdings_history()
            
            logger.info(f"Removed holding: {ticker} (reason: {reason})")
            return {"success": True, "removed_holding": holding}
            
        except Exception as e:
            logger.error(f"Failed to remove holding {ticker}: {e}")
            return {"success": False, "error": str(e)}
    
    def update_holding_price(self, ticker: str, current_price: float) -> bool:
        """Update current price for a holding"""
        if ticker not in self.current_portfolio:
            return False
        
        try:
            holding = self.current_portfolio[ticker]
            holding["current_price"] = current_price
            holding["last_updated"] = datetime.now().isoformat()
            
            # Recalculate metrics
            entry_price = holding.get("entry_price", 0)
            target_price = holding.get("target_price", 0)
            
            if entry_price > 0:
                # Calculate unrealized P&L
                pnl_pct = (current_price - entry_price) / entry_price * 100
                holding["unrealized_pnl_pct"] = round(pnl_pct, 2)
                
                # Calculate progress to target
                if target_price > entry_price:
                    progress = (current_price - entry_price) / (target_price - entry_price) * 100
                    holding["progress_to_target"] = max(0, min(100, round(progress, 1)))
                
                # Check for stop loss or target hit
                stop_loss = holding.get("stop_loss", 0)
                if stop_loss > 0 and current_price <= stop_loss:
                    holding["status"] = "stop_loss_triggered"
                elif target_price > 0 and current_price >= target_price:
                    holding["status"] = "target_reached"
            
            self._save_portfolio()
            return True
            
        except Exception as e:
            logger.error(f"Failed to update price for {ticker}: {e}")
            return False
    
    def update_portfolio_prices(self, portfolio_data: List[Dict]) -> List[Dict]:
        """Update all portfolio prices with live data"""
        # This is a placeholder - in production, this would fetch real-time prices
        # For now, return the input data as-is since we don't have live price feeds
        return portfolio_data
    
    def get_current_portfolio(self) -> List[Dict[str, Any]]:
        """Get current portfolio holdings"""
        portfolio_list = []
        
        for ticker, holding in self.current_portfolio.items():
            portfolio_list.append(holding.copy())
        
        # Sort by entry date (newest first)
        portfolio_list.sort(key=lambda x: x.get("entry_date", ""), reverse=True)
        
        return portfolio_list
    
    def _calculate_position_size(self, holding: Dict[str, Any]) -> float:
        """Calculate position size based on risk management rules"""
        if self.risk_params["position_sizing_method"] == "equal_weight":
            # Equal weight for all positions
            return self.risk_params["max_position_size"]
        
        elif self.risk_params["position_sizing_method"] == "confidence_weighted":
            # Weight by confidence score
            confidence = holding.get("confidence", 50) / 100
            base_size = self.risk_params["max_position_size"]
            return base_size * confidence
        
        elif self.risk_params["position_sizing_method"] == "risk_adjusted":
            # Adjust by stop loss distance
            entry_price = holding.get("entry_price", 0)
            stop_loss = holding.get("stop_loss", 0)
            
            if entry_price > 0 and stop_loss > 0:
                risk_per_share = (entry_price - stop_loss) / entry_price
                if risk_per_share > 0:
                    # Target 2% portfolio risk per position
                    target_risk = 0.02
                    position_size = target_risk / risk_per_share
                    return min(position_size, self.risk_params["max_position_size"])
            
            return self.risk_params["max_position_size"]
        
        else:
            return self.risk_params["max_position_size"]
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio performance summary"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_holdings": len(self.current_portfolio),
            "active_holdings": 0,
            "total_unrealized_pnl": 0,
            "total_unrealized_pnl_pct": 0,
            "winners": 0,
            "losers": 0,
            "targets_reached": 0,
            "stop_losses_triggered": 0,
            "holdings_by_status": {},
            "top_performers": [],
            "worst_performers": []
        }
        
        if not self.current_portfolio:
            return summary
        
        holdings_data = []
        
        for ticker, holding in self.current_portfolio.items():
            status = holding.get("status", "active")
            summary["holdings_by_status"][status] = summary["holdings_by_status"].get(status, 0) + 1
            
            if status == "active":
                summary["active_holdings"] += 1
            elif status == "target_reached":
                summary["targets_reached"] += 1
            elif status == "stop_loss_triggered":
                summary["stop_losses_triggered"] += 1
            
            # P&L analysis
            pnl_pct = holding.get("unrealized_pnl_pct", 0)
            if pnl_pct > 0:
                summary["winners"] += 1
            elif pnl_pct < 0:
                summary["losers"] += 1
            
            holdings_data.append({
                "ticker": ticker,
                "pnl_pct": pnl_pct,
                "confidence": holding.get("confidence", 0)
            })
        
        # Calculate average P&L
        if holdings_data:
            pnl_values = [h["pnl_pct"] for h in holdings_data]
            summary["total_unrealized_pnl_pct"] = round(np.mean(pnl_values), 2)
            
            # Top and worst performers
            holdings_data.sort(key=lambda x: x["pnl_pct"], reverse=True)
            summary["top_performers"] = holdings_data[:3]
            summary["worst_performers"] = holdings_data[-3:] if len(holdings_data) > 3 else []
        
        return summary
    
    def get_portfolio_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get detailed portfolio analytics"""
        analytics = {
            "timestamp": datetime.now().isoformat(),
            "period_days": days,
            "portfolio_summary": self.get_portfolio_summary(),
            "historical_analysis": {},
            "risk_metrics": {},
            "performance_attribution": {}
        }
        
        # Historical analysis from holdings history
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_actions = []
        for entry in self.holdings_history:
            try:
                entry_date = datetime.fromisoformat(entry["timestamp"])
                if entry_date > cutoff_date:
                    recent_actions.append(entry)
            except:
                continue
        
        analytics["historical_analysis"] = {
            "total_actions": len(recent_actions),
            "additions": len([a for a in recent_actions if a["action"] == "add"]),
            "removals": len([a for a in recent_actions if a["action"] == "remove"]),
            "recent_activity": recent_actions[-10:]  # Last 10 actions
        }
        
        # Risk metrics
        current_holdings = list(self.current_portfolio.values())
        if current_holdings:
            position_sizes = [h.get("position_size_pct", 0) for h in current_holdings]
            pnl_values = [h.get("unrealized_pnl_pct", 0) for h in current_holdings]
            
            analytics["risk_metrics"] = {
                "max_position_size": max(position_sizes) if position_sizes else 0,
                "avg_position_size": round(np.mean(position_sizes), 3) if position_sizes else 0,
                "portfolio_concentration": len([p for p in position_sizes if p > 0.05]),  # Positions > 5%
                "portfolio_volatility": round(np.std(pnl_values), 2) if len(pnl_values) > 1 else 0,
                "sharpe_estimate": self._calculate_sharpe_estimate(pnl_values) if pnl_values else 0
            }
        
        # Performance attribution
        if current_holdings:
            confidence_bins = {"high": [], "medium": [], "low": []}
            
            for holding in current_holdings:
                confidence = holding.get("confidence", 0)
                pnl = holding.get("unrealized_pnl_pct", 0)
                
                if confidence >= 80:
                    confidence_bins["high"].append(pnl)
                elif confidence >= 60:
                    confidence_bins["medium"].append(pnl)
                else:
                    confidence_bins["low"].append(pnl)
            
            analytics["performance_attribution"] = {
                "by_confidence": {
                    bin_name: {
                        "count": len(values),
                        "avg_pnl": round(np.mean(values), 2) if values else 0
                    }
                    for bin_name, values in confidence_bins.items()
                }
            }
        
        return analytics
    
    def _calculate_sharpe_estimate(self, returns: List[float]) -> float:
        """Calculate simplified Sharpe ratio estimate"""
        if len(returns) < 2:
            return 0
        
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
        
        # Simplified Sharpe (assuming 0% risk-free rate)
        return round(avg_return / std_return, 2)
    
    def check_portfolio_alerts(self) -> List[Dict[str, Any]]:
        """Check for portfolio-related alerts"""
        alerts = []
        
        for ticker, holding in self.current_portfolio.items():
            # Check stop loss
            current_price = holding.get("current_price", 0)
            stop_loss = holding.get("stop_loss", 0)
            entry_price = holding.get("entry_price", 0)
            
            if current_price > 0 and stop_loss > 0 and current_price <= stop_loss:
                alerts.append({
                    "type": "stop_loss_hit",
                    "ticker": ticker,
                    "message": f"{ticker} hit stop loss at ${current_price:.2f}",
                    "severity": "critical",
                    "timestamp": datetime.now().isoformat()
                })
            
            # Check target reached
            target_price = holding.get("target_price", 0)
            if current_price > 0 and target_price > 0 and current_price >= target_price:
                alerts.append({
                    "type": "target_reached",
                    "ticker": ticker,
                    "message": f"{ticker} reached target price ${current_price:.2f}",
                    "severity": "success",
                    "timestamp": datetime.now().isoformat()
                })
            
            # Check large moves
            if entry_price > 0 and current_price > 0:
                move_pct = abs((current_price - entry_price) / entry_price * 100)
                if move_pct > 10:  # Large move threshold
                    direction = "up" if current_price > entry_price else "down"
                    alerts.append({
                        "type": "large_move",
                        "ticker": ticker,
                        "message": f"{ticker} moved {move_pct:.1f}% {direction}",
                        "severity": "info",
                        "timestamp": datetime.now().isoformat()
                    })
        
        return alerts
    
    def export_portfolio(self, export_path: str) -> bool:
        """Export portfolio data to file"""
        try:
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "current_portfolio": self.current_portfolio,
                "portfolio_summary": self.get_portfolio_summary(),
                "holdings_history": self.holdings_history[-100:],  # Last 100 entries
                "risk_parameters": self.risk_params
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Portfolio exported to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Portfolio export failed: {e}")
            return False
