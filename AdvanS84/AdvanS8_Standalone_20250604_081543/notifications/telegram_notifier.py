import requests
from utils.config_loader import CONFIG

def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{CONFIG['telegram_token']}/sendMessage"
    data = {"chat_id": CONFIG['telegram_chat_id'], "text": text}
    try:
        response = requests.post(url, json=data, timeout=10)
        if response.status_code == 200:
            return True, "Message sent successfully"
        else:
            return False, f"HTTP {response.status_code}: {response.text}"
    except Exception as e:
        print("Telegram error:", e)
        return False, str(e)

def send_trading_alert(symbol, alert_type, message, priority="normal"):
    """Send formatted trading alert via Telegram"""
    if priority == "urgent":
        formatted_message = f"🚨 URGENT ALERT 🚨\n\n{symbol} - {alert_type}\n\n{message}"
    elif priority == "high":
        formatted_message = f"⚠️ HIGH PRIORITY ⚠️\n\n{symbol} - {alert_type}\n\n{message}"
    else:
        formatted_message = f"📊 AdvanS8 Alert\n\n{symbol} - {alert_type}\n\n{message}"
    
    return send_telegram_message(formatted_message)