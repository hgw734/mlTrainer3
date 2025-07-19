"""
Notification System Module
Alert regime and signal notifications for Meta-Enhanced TPE-ML Trading System
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
from .regime_logic import get_market_regime, get_regime_adaptive_rsi_threshold
from .data_loader import get_stock_data, get_vix_data

try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False

logger = logging.getLogger(__name__)

class TradingNotificationSystem:
    """
    Comprehensive notification system for trading signals and regime alerts
    """
    
    def __init__(self, config=None):
        """Initialize notification system with configuration"""
        self.config = config or self._default_config()
        from api_config import get_polygon_key, get_fred_key
        self.polygon_api_key = get_polygon_key()
        self.fred_api_key = get_fred_key()
        
        # Twilio configuration for SMS alerts
        self.twilio_sid = os.getenv('TWILIO_ACCOUNT_SID')
        self.twilio_token = os.getenv('TWILIO_AUTH_TOKEN')
        self.twilio_phone = os.getenv('TWILIO_PHONE_NUMBER')
        
        self.notification_log = []
        logger.info("Trading notification system initialized")
    
    def _default_config(self):
        """Default notification configuration"""
        return {
            'universe': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'META', 'AMZN', 'NFLX'],
            'alert_thresholds': {
                'momentum_spike': 0.03,
                'volume_surge': 2.5,
                'volatility_breakout': 0.04,
                'regime_change': True
            },
            'notification_methods': ['console', 'file', 'sms'],
            'alert_frequency': 'hourly'  # hourly, daily, realtime
        }
    
    def check_market_regime_alerts(self):
        """
        Check for market regime changes and generate alerts
        
        Returns:
            list: List of regime alerts
        """
        if not self.polygon_api_key:
            logger.warning("Polygon API key required for regime alerts")
            return []
        
        alerts = []
        
        try:
            # Get recent VIX data for regime detection
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5)
            
            vix_data = get_vix_data(start_date, end_date, self.fred_api_key)
            
            if vix_data is not None and len(vix_data) >= 2:
                current_regime = get_market_regime(vix_data.index[-1], vix_data)
                previous_regime = get_market_regime(vix_data.index[-2], vix_data)
                
                if current_regime != previous_regime:
                    alert = {
                        'type': 'regime_change',
                        'timestamp': datetime.now(),
                        'message': f"Market regime changed: {previous_regime} → {current_regime}",
                        'current_regime': current_regime,
                        'previous_regime': previous_regime,
                        'vix_level': vix_data.iloc[-1]['vix'],
                        'priority': 'high' if 'crisis' in current_regime else 'medium'
                    }
                    alerts.append(alert)
                    logger.info(f"Regime change detected: {previous_regime} → {current_regime}")
                
                # VIX spike alerts
                current_vix = vix_data.iloc[-1]['vix']
                if current_vix > 30:
                    alert = {
                        'type': 'vix_spike',
                        'timestamp': datetime.now(),
                        'message': f"VIX spike alert: {current_vix:.2f} (Crisis level)",
                        'vix_level': current_vix,
                        'regime': current_regime,
                        'priority': 'urgent'
                    }
                    alerts.append(alert)
                elif current_vix > 25:
                    alert = {
                        'type': 'vix_elevated',
                        'timestamp': datetime.now(),
                        'message': f"VIX elevated: {current_vix:.2f} (High volatility)",
                        'vix_level': current_vix,
                        'regime': current_regime,
                        'priority': 'high'
                    }
                    alerts.append(alert)
        
        except Exception as e:
            logger.error(f"Error checking regime alerts: {e}")
        
        return alerts
    
    def check_trading_signals(self, optimized_params=None):
        """
        Check for trading signals across the universe using authentic data
        
        Args:
            optimized_params: Optimized parameters from TPE system
        
        Returns:
            list: List of trading signal alerts
        """
        if not self.polygon_api_key:
            logger.warning("Polygon API key required for trading signals")
            return []
        
        signals = []
        default_params = {
            'momentum_threshold': 0.02,
            'volume_multiplier': 2.0,
            'risk_tolerance': 0.03
        }
        
        params = optimized_params if optimized_params else default_params
        
        try:
            # Get recent market data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            # Get VIX for regime context
            vix_data = get_vix_data(start_date, end_date, self.fred_api_key)
            current_regime = get_market_regime(vix_data.index[-1], vix_data) if vix_data is not None else 'moderate_volatility_normal'
            
            for symbol in self.config['universe']:
                try:
                    # Get recent stock data
                    stock_data = get_stock_data(symbol, start_date, end_date, self.polygon_api_key, '1day')
                    
                    if stock_data is not None and len(stock_data) >= 20:
                        from .data_loader import add_technical_indicators
                        stock_data = add_technical_indicators(stock_data)
                        
                        current_row = stock_data.iloc[-1]
                        signal = self._evaluate_signal(symbol, current_row, params, current_regime)
                        
                        if signal:
                            signals.append(signal)
                
                except Exception as e:
                    logger.warning(f"Error checking signals for {symbol}: {e}")
        
        except Exception as e:
            logger.error(f"Error checking trading signals: {e}")
        
        return signals
    
    def _evaluate_signal(self, symbol, current_row, params, market_regime):
        """
        Evaluate if current market conditions generate a trading signal
        
        Args:
            symbol: Stock symbol
            current_row: Current market data row
            params: Trading parameters
            market_regime: Current market regime
        
        Returns:
            dict or None: Signal information if conditions are met
        """
        # Skip if essential data is missing
        if pd.isna(current_row['momentum_5']) or pd.isna(current_row['rsi_14']):
            return None
        
        # Entry criteria evaluation
        momentum_ok = current_row['momentum_5'] >= params['momentum_threshold']
        volume_ok = current_row['volume_ratio'] >= params['volume_multiplier']
        trend_ok = current_row['close'] > current_row['sma_20']
        
        # Regime-adaptive RSI threshold
        rsi_threshold = get_regime_adaptive_rsi_threshold(market_regime)
        rsi_ok = current_row['rsi_14'] > rsi_threshold
        
        # Risk management
        risk_ok = True
        if not pd.isna(current_row['atr_pct_14']):
            risk_tolerance = params['risk_tolerance']
            if 'crisis' in market_regime:
                risk_tolerance *= 0.7  # Tighter risk in crisis
            risk_ok = current_row['atr_pct_14'] < risk_tolerance
        
        # Generate signal if all conditions met
        if momentum_ok and volume_ok and trend_ok and rsi_ok and risk_ok:
            # Calculate signal strength
            signal_strength = self._calculate_signal_strength(current_row, params, market_regime)
            
            return {
                'type': 'trading_signal',
                'symbol': symbol,
                'timestamp': datetime.now(),
                'price': current_row['close'],
                'signal_strength': signal_strength,
                'market_regime': market_regime,
                'momentum': current_row['momentum_5'],
                'volume_ratio': current_row['volume_ratio'],
                'rsi': current_row['rsi_14'],
                'atr_pct': current_row['atr_pct_14'],
                'message': f"{symbol} BUY signal - Strength: {signal_strength:.2f} - Regime: {market_regime}",
                'priority': 'high' if signal_strength > 0.7 else 'medium'
            }
        
        return None
    
    def _calculate_signal_strength(self, current_row, params, market_regime):
        """Calculate signal strength score (0-1)"""
        score = 0
        
        # Momentum component (0-0.3)
        momentum_score = min(current_row['momentum_5'] / 0.05, 1.0) * 0.3
        score += momentum_score
        
        # Volume component (0-0.2)
        volume_score = min(current_row['volume_ratio'] / 3.0, 1.0) * 0.2
        score += volume_score
        
        # RSI component (0-0.2)
        rsi_norm = (current_row['rsi_14'] - 50) / 50
        rsi_score = max(0, rsi_norm) * 0.2
        score += rsi_score
        
        # Trend strength component (0-0.2)
        if not pd.isna(current_row.get('trend_strength', 0)):
            trend_score = min(abs(current_row['trend_strength']) / 0.1, 1.0) * 0.2
            score += trend_score
        
        # Regime bonus (0-0.1)
        if 'low_volatility' in market_regime:
            score += 0.1  # Favorable regime
        elif 'crisis' in market_regime:
            score *= 0.8  # Penalty for crisis regime
        
        return min(score, 1.0)
    
    def send_notification(self, alert_data):
        """
        Send notification through configured channels
        
        Args:
            alert_data: Alert information dictionary
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"[{timestamp}] {alert_data['message']}"
        
        # Console notification
        if 'console' in self.config['notification_methods']:
            priority_prefix = {
                'urgent': "🚨 URGENT",
                'high': "⚠️ HIGH",
                'medium': "📊 MEDIUM",
                'low': "ℹ️ INFO"
            }
            prefix = priority_prefix.get(alert_data.get('priority', 'medium'), "📊")
            print(f"{prefix}: {message}")
        
        # File logging
        if 'file' in self.config['notification_methods']:
            self._log_to_file(alert_data)
        
        # SMS notification
        if 'sms' in self.config['notification_methods'] and alert_data.get('priority') in ['urgent', 'high']:
            self._send_sms(message)
        
        # Store in notification log
        self.notification_log.append({
            'timestamp': timestamp,
            'alert_data': alert_data,
            'sent_methods': self.config['notification_methods']
        })
    
    def _log_to_file(self, alert_data):
        """Log alert to file"""
        log_dir = 'data/notifications'
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = f"{log_dir}/trading_alerts_{datetime.now().strftime('%Y%m%d')}.log"
        
        with open(log_file, 'a') as f:
            f.write(f"{datetime.now().isoformat()} - {alert_data}\n")
    
    def _send_sms(self, message):
        """Send SMS notification via Twilio"""
        if not TWILIO_AVAILABLE:
            logger.warning("Twilio not available for SMS notifications")
            return
        
        if not all([self.twilio_sid, self.twilio_token, self.twilio_phone]):
            logger.warning("Twilio credentials not configured for SMS")
            return
        
        try:
            client = Client(self.twilio_sid, self.twilio_token)
            
            # Truncate message for SMS
            sms_message = message[:160] + "..." if len(message) > 160 else message
            
            message = client.messages.create(
                body=sms_message,
                from_=self.twilio_phone,
                to=os.getenv('NOTIFICATION_PHONE_NUMBER')
            )
            
            logger.info(f"SMS sent with SID: {message.sid}")
        
        except Exception as e:
            logger.error(f"Error sending SMS: {e}")
    
    def run_alert_scan(self, optimized_params=None):
        """
        Run complete alert scan for regime changes and trading signals
        
        Args:
            optimized_params: Optimized parameters from TPE system
        
        Returns:
            dict: Summary of alerts generated
        """
        logger.info("Running comprehensive alert scan...")
        
        # Check regime alerts
        regime_alerts = self.check_market_regime_alerts()
        
        # Check trading signals
        trading_signals = self.check_trading_signals(optimized_params)
        
        # Send notifications
        all_alerts = regime_alerts + trading_signals
        
        for alert in all_alerts:
            self.send_notification(alert)
        
        # Generate summary
        summary = {
            'total_alerts': len(all_alerts),
            'regime_alerts': len(regime_alerts),
            'trading_signals': len(trading_signals),
            'urgent_alerts': sum(1 for alert in all_alerts if alert.get('priority') == 'urgent'),
            'high_priority': sum(1 for alert in all_alerts if alert.get('priority') == 'high'),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Alert scan complete - {summary['total_alerts']} alerts generated")
        return summary
    
    def get_recent_notifications(self, hours=24):
        """Get recent notifications within specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent = [
            notif for notif in self.notification_log
            if datetime.fromisoformat(notif['timestamp']) > cutoff_time
        ]
        
        return recent
    
    def configure_notification_methods(self, methods):
        """Configure notification delivery methods"""
        valid_methods = ['console', 'file', 'sms', 'email']
        self.config['notification_methods'] = [m for m in methods if m in valid_methods]
        logger.info(f"Notification methods configured: {self.config['notification_methods']}")

def create_notification_system(config=None):
    """Factory function to create notification system"""
    return TradingNotificationSystem(config)