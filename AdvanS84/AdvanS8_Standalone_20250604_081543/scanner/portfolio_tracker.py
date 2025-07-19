"""
Portfolio Tracking & Alert System for Manual Trading
Bridges the gap between scan results and real-world trading decisions.
"""

import json
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class PortfolioTracker:
    """
    Tracks user's actual positions and provides real-time exit alerts
    """
    
    def __init__(self, db_path: str = "data/portfolio.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize portfolio tracking database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Portfolio positions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                entry_date TEXT NOT NULL,
                entry_price REAL NOT NULL,
                quantity INTEGER NOT NULL,
                scanner_score REAL,
                original_signal TEXT,
                status TEXT DEFAULT 'active',
                exit_date TEXT,
                exit_price REAL,
                exit_reason TEXT,
                profit_loss REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Exit alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS exit_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                alert_message TEXT NOT NULL,
                current_price REAL,
                current_return REAL,
                days_held INTEGER,
                urgency_level TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                acknowledged BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_position(self, symbol: str, entry_price: float, quantity: int, 
                    scanner_score: float = None, signal_data: Dict = None) -> bool:
        """
        Add a new position to track
        
        Args:
            symbol: Stock symbol
            entry_price: Price at which you bought
            quantity: Number of shares
            scanner_score: Original scanner score
            signal_data: Original signal information
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO positions 
                (symbol, entry_date, entry_price, quantity, scanner_score, original_signal)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                datetime.now().strftime("%Y-%m-%d"),
                entry_price,
                quantity,
                scanner_score,
                json.dumps(signal_data) if signal_data else None
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Added position: {symbol} - {quantity} shares at ${entry_price}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding position {symbol}: {e}")
            return False
    
    def get_active_positions(self) -> List[Dict]:
        """Get all active positions"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM positions 
                WHERE status = 'active'
                ORDER BY entry_date DESC
            ''')
            
            columns = [desc[0] for desc in cursor.description]
            positions = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            conn.close()
            return positions
            
        except Exception as e:
            logger.error(f"Error getting active positions: {e}")
            return []
    
    def close_position(self, symbol: str, exit_price: float, exit_reason: str = "Manual exit") -> bool:
        """
        Close a position
        
        Args:
            symbol: Stock symbol to close
            exit_price: Exit price
            exit_reason: Reason for exit
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get position info
            cursor.execute('''
                SELECT entry_price, quantity FROM positions 
                WHERE symbol = ? AND status = 'active'
                ORDER BY entry_date DESC LIMIT 1
            ''', (symbol,))
            
            result = cursor.fetchone()
            if not result:
                return False
            
            entry_price, quantity = result
            profit_loss = (exit_price - entry_price) * quantity
            
            # Update position
            cursor.execute('''
                UPDATE positions 
                SET status = 'closed', exit_date = ?, exit_price = ?, 
                    exit_reason = ?, profit_loss = ?
                WHERE symbol = ? AND status = 'active'
            ''', (
                datetime.now().strftime("%Y-%m-%d"),
                exit_price,
                exit_reason,
                profit_loss,
                symbol
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Closed position: {symbol} - P&L: ${profit_loss:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error closing position {symbol}: {e}")
            return False
    
    def generate_exit_alerts(self, market_data_provider) -> List[Dict]:
        """
        Generate real-time exit alerts for active positions
        
        Args:
            market_data_provider: Data provider to get current prices
            
        Returns:
            List of exit alerts
        """
        alerts = []
        active_positions = self.get_active_positions()
        
        for position in active_positions:
            try:
                symbol = position['symbol']
                entry_price = position['entry_price']
                entry_date = datetime.strptime(position['entry_date'], "%Y-%m-%d")
                days_held = (datetime.now() - entry_date).days
                
                # Get current market data
                current_data = self._get_current_market_data(symbol, market_data_provider)
                if not current_data:
                    continue
                
                current_price = current_data['current_price']
                current_return = (current_price - entry_price) / entry_price
                
                # Apply aggressive exit logic
                alert = self._check_exit_conditions(
                    symbol, current_return, days_held, current_price, current_data
                )
                
                if alert:
                    alert.update({
                        'symbol': symbol,
                        'current_price': current_price,
                        'current_return': current_return,
                        'days_held': days_held,
                        'entry_price': entry_price,
                        'quantity': position['quantity']
                    })
                    alerts.append(alert)
                    
                    # Save alert to database
                    self._save_alert(alert)
                
            except Exception as e:
                logger.error(f"Error generating alert for {symbol}: {e}")
                continue
        
        return alerts
    
    def _check_exit_conditions(self, symbol: str, current_return: float, 
                              days_held: int, current_price: float, 
                              market_data: Dict) -> Optional[Dict]:
        """Check if position meets exit conditions"""
        
        if current_return < 0:  # LOSING TRADE - Aggressive exit
            if days_held == 1 and current_return <= -0.02:
                return {
                    'alert_type': 'IMMEDIATE_EXIT',
                    'alert_message': f'Day 1 loss cut: {current_return:.1%} loss - EXIT NOW',
                    'urgency_level': 'CRITICAL',
                    'exit_reason': 'Day 1 aggressive loss cut (-2%)'
                }
            elif days_held == 2 and current_return <= -0.03:
                return {
                    'alert_type': 'IMMEDIATE_EXIT',
                    'alert_message': f'Day 2 loss cut: {current_return:.1%} loss - EXIT NOW',
                    'urgency_level': 'CRITICAL',
                    'exit_reason': 'Day 2 aggressive loss cut (-3%)'
                }
            elif days_held == 3 and current_return <= -0.04:
                return {
                    'alert_type': 'IMMEDIATE_EXIT',
                    'alert_message': f'Day 3 loss cut: {current_return:.1%} loss - EXIT NOW',
                    'urgency_level': 'CRITICAL',
                    'exit_reason': 'Day 3 aggressive loss cut (-4%)'
                }
            elif days_held > 3 and current_return <= -0.015:
                return {
                    'alert_type': 'IMMEDIATE_EXIT',
                    'alert_message': f'Sustained loss: {current_return:.1%} - EXIT NOW',
                    'urgency_level': 'HIGH',
                    'exit_reason': 'Sustained loss cut (-1.5%)'
                }
            elif current_return <= -0.06:
                return {
                    'alert_type': 'IMMEDIATE_EXIT',
                    'alert_message': f'HARD STOP LOSS: {current_return:.1%} - EXIT IMMEDIATELY',
                    'urgency_level': 'CRITICAL',
                    'exit_reason': 'Hard stop loss (-6%)'
                }
        
        else:  # WINNING TRADE - Let winners run
            rsi = market_data.get('rsi', 50)
            volume_ratio = market_data.get('volume_ratio', 1.0)
            
            if rsi > 80 and volume_ratio < 0.3 and current_return > 0.15:
                return {
                    'alert_type': 'PROFIT_WARNING',
                    'alert_message': f'Winner showing weakness: {current_return:.1%} gain - Consider exit',
                    'urgency_level': 'MEDIUM',
                    'exit_reason': 'Severe deterioration on winner'
                }
            elif days_held >= 30:
                return {
                    'alert_type': 'TIME_EXIT',
                    'alert_message': f'30-day hold reached: {current_return:.1%} - Time to exit',
                    'urgency_level': 'MEDIUM',
                    'exit_reason': 'Maximum hold period (30 days)'
                }
        
        return None
    
    def _get_current_market_data(self, symbol: str, data_provider) -> Optional[Dict]:
        """Get current market data for a symbol"""
        try:
            # Use the same data provider as the scanner
            return data_provider.get_current_market_data(symbol)
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    def _save_alert(self, alert: Dict):
        """Save alert to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO exit_alerts 
                (symbol, alert_type, alert_message, current_price, current_return, 
                 days_held, urgency_level)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert['symbol'],
                alert['alert_type'],
                alert['alert_message'],
                alert['current_price'],
                alert['current_return'],
                alert['days_held'],
                alert['urgency_level']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving alert: {e}")
    
    def get_unacknowledged_alerts(self) -> List[Dict]:
        """Get all unacknowledged alerts"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM exit_alerts 
                WHERE acknowledged = FALSE
                ORDER BY urgency_level DESC, created_at DESC
            ''')
            
            columns = [desc[0] for desc in cursor.description]
            alerts = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            conn.close()
            return alerts
            
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return []
    
    def acknowledge_alert(self, alert_id: int) -> bool:
        """Mark an alert as acknowledged"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE exit_alerts 
                SET acknowledged = TRUE 
                WHERE id = ?
            ''', (alert_id,))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error acknowledging alert {alert_id}: {e}")
            return False
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio performance summary"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Active positions
            cursor.execute('SELECT COUNT(*) FROM positions WHERE status = "active"')
            active_count = cursor.fetchone()[0]
            
            # Closed positions performance
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(profit_loss) as total_pnl,
                    AVG(profit_loss) as avg_pnl
                FROM positions 
                WHERE status = "closed"
            ''')
            
            result = cursor.fetchone()
            total_trades, winning_trades, total_pnl, avg_pnl = result
            
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            conn.close()
            
            return {
                'active_positions': active_count,
                'total_trades': total_trades or 0,
                'winning_trades': winning_trades or 0,
                'win_rate': win_rate,
                'total_pnl': total_pnl or 0,
                'avg_pnl': avg_pnl or 0
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return {}