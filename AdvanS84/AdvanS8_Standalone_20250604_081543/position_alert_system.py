"""
Position Alert System for AdvanS8
Monitors positions and triggers notifications when target or stop loss levels are reached
"""

import json
import pandas as pd
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Tuple
from api_config import get_polygon_key, get_polygon_params
import time

class PositionAlertSystem:
    """
    Monitors portfolio positions and triggers alerts for target/stop loss events
    """
    
    def __init__(self):
        self.polygon_api_key = get_polygon_key()
        self.base_url = "https://api.polygon.io"
        self.alerts_file = "position_alerts.json"
        self.positions_file = "portfolio_positions.json"
        
    def load_portfolio_positions(self) -> pd.DataFrame:
        """Load current portfolio positions from saved data"""
        try:
            with open(self.positions_file, 'r') as f:
                positions_data = json.load(f)
            return pd.DataFrame(positions_data)
        except:
            # Return empty DataFrame if no positions file exists
            return pd.DataFrame()
    
    def save_portfolio_positions(self, positions_df: pd.DataFrame):
        """Save portfolio positions to file"""
        try:
            positions_data = positions_df.to_dict('records')
            with open(self.positions_file, 'w') as f:
                json.dump(positions_data, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving positions: {e}")
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price from Polygon API with timestamp verification"""
        try:
            current_time = datetime.now()
            print(f"TIMESTAMP VERIFICATION: Fetching {symbol} price at {current_time.isoformat()}")
            
            url = f"{self.base_url}/v2/aggs/ticker/{symbol}/prev"
            params = get_polygon_params()
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('results') and len(data['results']) > 0:
                    result = data['results'][0]
                    
                    # Verify timestamp authenticity
                    if 't' in result:
                        timestamp = pd.to_datetime(result['t'], unit='ms')
                        print(f"TIMESTAMP VERIFICATION: {symbol} - Price data from {timestamp}")
                        
                        # Check for suspicious patterns
                        if timestamp > current_time:
                            print(f"WARNING: {symbol} - Future timestamp detected: {timestamp}")
                            return None
                        
                        # Check if data is too old
                        days_old = (current_time - timestamp).days
                        if days_old > 5:
                            print(f"WARNING: {symbol} - Price data is {days_old} days old")
                    
                    return float(result['c'])  # Close price
        except Exception as e:
            print(f"Error fetching price for {symbol}: {e}")
        
        return None
    
    def check_position_alerts(self, positions_df: pd.DataFrame) -> List[Dict]:
        """Check all positions for target/stop loss triggers"""
        alerts = []
        
        for idx, row in positions_df.iterrows():
            symbol = row['symbol']
            original_price = float(row.get('original_price', row.get('entry_price', 0)))
            original_target = float(row.get('original_target', original_price * 1.08))
            original_stop_loss = float(row.get('original_stop_loss', original_price * 0.97))
            
            # Get current market price
            current_price = self.get_current_price(symbol)
            
            if current_price is None:
                continue
                
            # Check for target achievement
            if current_price >= original_target:
                alert = {
                    'symbol': symbol,
                    'alert_type': 'TARGET_REACHED',
                    'current_price': current_price,
                    'trigger_price': original_target,
                    'original_price': original_price,
                    'gain_percent': ((current_price - original_price) / original_price * 100),
                    'timestamp': datetime.now().isoformat(),
                    'message': f"🎯 {symbol} reached target! ${current_price:.2f} (target: ${original_target:.2f})"
                }
                alerts.append(alert)
            
            # Check for stop loss trigger
            elif current_price <= original_stop_loss:
                alert = {
                    'symbol': symbol,
                    'alert_type': 'STOP_LOSS_HIT',
                    'current_price': current_price,
                    'trigger_price': original_stop_loss,
                    'original_price': original_price,
                    'loss_percent': ((current_price - original_price) / original_price * 100),
                    'timestamp': datetime.now().isoformat(),
                    'message': f"🛑 {symbol} hit stop loss! ${current_price:.2f} (stop: ${original_stop_loss:.2f})"
                }
                alerts.append(alert)
            
            # Update current price in positions
            positions_df.at[idx, 'current_price'] = current_price
            positions_df.at[idx, 'last_updated'] = datetime.now().isoformat()
        
        return alerts
    
    def send_telegram_notification(self, alert: Dict):
        """Send alert via Telegram using existing notification system"""
        try:
            from notifications.telegram_notifier import send_trading_alert
            
            symbol = alert['symbol']
            alert_type = alert['alert_type'].replace('_', ' ').title()
            message = alert['message']
            
            # Set priority based on alert type
            if alert['alert_type'] == 'STOP_LOSS_HIT':
                priority = "urgent"
            elif alert['alert_type'] == 'TARGET_REACHED':
                priority = "high"
            else:
                priority = "normal"
            
            success, result = send_trading_alert(symbol, alert_type, message, priority)
            
            if success:
                print(f"📱 Telegram alert sent for {symbol}")
            else:
                print(f"📱 Telegram failed: {result}")
                
            return success
            
        except Exception as e:
            print(f"Telegram notification error: {e}")
            return False
    
    def save_alerts(self, alerts: List[Dict]):
        """Save alerts to file and send notifications"""
        try:
            # Load existing alerts
            existing_alerts = []
            try:
                with open(self.alerts_file, 'r') as f:
                    existing_alerts = json.load(f)
            except:
                pass
            
            # Process each new alert
            for alert in alerts:
                # Send Telegram notification
                self.send_telegram_notification(alert)
                
                # Add to existing alerts
                existing_alerts.append(alert)
            
            # Keep only last 100 alerts
            existing_alerts = existing_alerts[-100:]
            
            # Save updated alerts
            with open(self.alerts_file, 'w') as f:
                json.dump(existing_alerts, f, indent=2)
                
        except Exception as e:
            print(f"Error saving alerts: {e}")
    
    def run_position_monitoring(self):
        """Main monitoring function"""
        print("Position Alert System - Starting Monitoring")
        print("=" * 50)
        
        # Load current positions
        positions_df = self.load_portfolio_positions()
        
        if positions_df.empty:
            print("No positions to monitor")
            return
        
        print(f"Monitoring {len(positions_df)} positions...")
        
        # Check for alerts
        alerts = self.check_position_alerts(positions_df)
        
        # Save updated positions
        self.save_portfolio_positions(positions_df)
        
        # Process alerts
        if alerts:
            print(f"\n🚨 {len(alerts)} ALERTS TRIGGERED:")
            for alert in alerts:
                print(f"  {alert['message']}")
            
            # Save alerts for dashboard
            self.save_alerts(alerts)
        else:
            print("No alerts triggered")
        
        print(f"\nMonitoring complete - {datetime.now().strftime('%H:%M:%S')}")
        return alerts
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Get alerts from the last N hours"""
        try:
            with open(self.alerts_file, 'r') as f:
                all_alerts = json.load(f)
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            recent_alerts = []
            for alert in all_alerts:
                alert_time = datetime.fromisoformat(alert['timestamp'])
                if alert_time >= cutoff_time:
                    recent_alerts.append(alert)
            
            return recent_alerts
        except:
            return []

def main():
    """Run position monitoring cycle"""
    alert_system = PositionAlertSystem()
    alerts = alert_system.run_position_monitoring()
    
    return {
        'alerts_count': len(alerts) if alerts else 0,
        'alerts': alerts if alerts else [],
        'timestamp': datetime.now().isoformat()
    }

if __name__ == "__main__":
    main()