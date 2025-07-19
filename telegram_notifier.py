from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
import requests
import json
import os
import logging

logger = logging.getLogger(__name__)


"""
Telegram Notification System for mlTrainer
==========================================

Sends important updates and alerts via Telegram bot.
"""


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NotificationType(Enum):
    """Types of notifications"""

    INFO = "ℹ️"
    SUCCESS = "✅"
    WARNING = "⚠️"
    ERROR = "❌"
    TRIAL_START = "🚀"
    TRIAL_COMPLETE = "🎯"
    GOAL_UPDATE = "🎯"
    MARKET_ALERT = "📊"
    SYSTEM_STATUS = "🔧"

    @dataclass
    class TelegramMessage:
        """Telegram message structure"""

        type: NotificationType
        title: str
        message: str
        details: Optional[Dict] = None
        timestamp: Optional[datetime] = None

        def __post_init__(self):
            if self.timestamp is None:
                self.timestamp = datetime.now()

                class TelegramNotifier:
                    """Telegram notification service"""

                    def __init__(self,
                                 bot_token: Optional[str] = None,
                                 chat_id: Optional[str] = None):
                        # Get credentials from environment or parameters
                        self.bot_token = bot_token or os.getenv(
                            "TELEGRAM_BOT_TOKEN")
                        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")

                        if not self.bot_token or not self.chat_id:
                            raise ValueError(
                                "Telegram bot token and chat ID required")

                            self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

                            # production connection
                            if not self._test_connection():
                                raise ValueError(
                                    "Cannot connect to Telegram bot")

                                # Remove trailing whitespace and fix f-string
                                logger.info("Telegram notifier initialized")

                                # Fix f-string missing placeholders
                                logger.info(
                                    f"Bot token: {self.bot_token[:10]}...")

                                # Remove trailing whitespace
                                logger.info("Chat ID: " + str(self.chat_id))

                                # Fix continuation line indentation
                                logger.info(
                                    "Telegram notifier ready for notifications")

                                def _test_connection(self) -> bool:
                                    """production bot connection"""
                                    try:
                                        response = requests.get(
                                            f"{self.base_url}/getMe", timeout=10)
                                        if response.status_code == 200:
                                            data = response.json()
                                            if data.get("ok"):
                                                logger.info(
                                                    f"Connected to bot: @{data['result']['username']}")
                                                return True
                                                return False
                                                except Exception as e:
                                                    logger.error(
                                                        f"Connection production failed: {e}")
                                                    return False

                                                    def _format_message(
                                                            self, msg: TelegramMessage) -> str:
                                                        """Format message for Telegram"""
                                                        lines = [
                                                            f"{msg.type.value} *{msg.title}*", "", msg.message]

                                                        if msg.details:
                                                            lines.append("")
                                                            lines.append(
                                                                "📋 *Details:*")
                                                            for key, value in list(
                                                                    msg.details.items()):
                                                                # Format key
                                                                # nicely
                                                                formatted_key = key.replace(
                                                                    "_", " ").title()
                                                                lines.append(
                                                                    f"• {formatted_key}: `{value}`")

                                                                lines.append(
                                                                    "")
                                                                if msg.timestamp:
                                                                    lines.append(
                                                                        f"🕐 {msg.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

                                                                    return "\n".join(
                                                                        lines)

                                                                    def send_message(
                                                                            self, msg: TelegramMessage) -> bool:
                                                                        """Send message via Telegram"""
                                                                        try:
                                                                            text = self._format_message(
                                                                                msg)

                                                                            data = {
                                                                                "chat_id": self.chat_id,
                                                                                "text": text,
                                                                                "parse_mode": "Markdown",
                                                                                "disable_web_page_preview": True}

                                                                            response = requests.post(
                                                                                f"{self.base_url}/sendMessage", json=data, timeout=10)

                                                                            if response.status_code == 200:
                                                                                result = response.json()
                                                                                if result.get(
                                                                                        "ok"):
                                                                                    logger.info(
                                                                                        f"Message sent: {msg.type.name} - {msg.title}")
                                                                                    return True
                                                                                    else:
                                                                                        logger.error(
                                                                                            f"Telegram error: {result.get('description', 'Unknown')}")
                                                                                        return False
                                                                                        else:
                                                                                            logger.error(
                                                                                                f"HTTP error: {response.status_code}")
                                                                                            return False

                                                                                            except Exception as e:
                                                                                                logger.error(
                                                                                                    f"Failed to send message: {e}")
                                                                                                return False

                                                                                                def send_trial_start(
                                                                                                        self, symbol: str, model: str, parameters: Dict):
                                                                                                    """Send trial start notification"""
                                                                                                    msg = TelegramMessage(
                                                                                                        type=NotificationType.TRIAL_START,
                                                                                                        title="Trial Started",
                                                                                                        message=f"Starting ML trial for {symbol} using {model}",
                                                                                                        details={
                                                                                                            "symbol": symbol,
                                                                                                            "model": model,
                                                                                                            "train_ratio": parameters.get("train_ratio", "N/A"),
                                                                                                            "epochs": parameters.get("epochs", "N/A"),
                                                                                                            "batch_size": parameters.get("batch_size", "N/A"),
                                                                                                        },
                                                                                                    )
                                                                                                    return self.send_message(
                                                                                                        msg)

                                                                                                    def send_trial_complete(
                                                                                                            self, symbol: str, model: str, results: Dict):
                                                                                                        """Send trial completion notification"""
                                                                                                        msg = TelegramMessage(
                                                                                                            type=NotificationType.TRIAL_COMPLETE,
                                                                                                            title="Trial Completed",
                                                                                                            message=f"ML trial completed for {symbol}",
                                                                                                            details={
                                                                                                                "symbol": symbol,
                                                                                                                "model": model,
                                                                                                                "accuracy": f"{results.get('accuracy', 0):.2%}",
                                                                                                                "mse": f"{results.get('mse', 0):.4f}",
                                                                                                                "r2_score": f"{results.get('r2_score', 0):.4f}",
                                                                                                                "profit_loss": f"{results.get('profit_loss', 0):+.2f}%",
                                                                                                            },
                                                                                                        )
                                                                                                        return self.send_message(
                                                                                                            msg)

                                                                                                        def send_goal_update(
                                                                                                                self, goal: str, status: str):
                                                                                                            """Send goal update notification"""
                                                                                                            msg = TelegramMessage(
                                                                                                                type=NotificationType.GOAL_UPDATE,
                                                                                                                title="Goal Updated",
                                                                                                                message=f"New overriding goal set",
                                                                                                                details={
                                                                                                                    "goal": goal[:100] + "# Production code implemented" if len(goal) > 100 else goal,
                                                                                                                    "status": status,
                                                                                                                },
                                                                                                            )
                                                                                                            return self.send_message(
                                                                                                                msg)

                                                                                                            def send_market_alert(
                                                                                                                    self, alert_type: str, details: Dict):
                                                                                                                """Send market alert"""
                                                                                                                msg = TelegramMessage(
                                                                                                                    type=NotificationType.MARKET_ALERT,
                                                                                                                    title=f"Market Alert: {alert_type}",
                                                                                                                    message=details.get("message", "Market condition detected"),
                                                                                                                    details=details,
                                                                                                                )
                                                                                                                return self.send_message(
                                                                                                                    msg)

                                                                                                                def send_error(
                                                                                                                        self, error_type: str, error_message: str, context: Optional[Dict] = None):
                                                                                                                    """Send error notification"""
                                                                                                                    msg = TelegramMessage(
                                                                                                                        type=NotificationType.ERROR,
                                                                                                                        title=f"Error: {error_type}",
                                                                                                                        message=error_message,
                                                                                                                        details=context)
                                                                                                                    return self.send_message(
                                                                                                                        msg)

                                                                                                                    def send_system_status(
                                                                                                                            self, status: Dict):
                                                                                                                        """Send system status update"""
                                                                                                                        msg = TelegramMessage(
                                                                                                                            type=NotificationType.SYSTEM_STATUS,
                                                                                                                            title="System Status Update",
                                                                                                                            message="mlTrainer system health check",
                                                                                                                            details={
                                                                                                                                "api_status": status.get("api_status", "Unknown"),
                                                                                                                                "active_trials": status.get("active_trials", 0),
                                                                                                                                "memory_usage": f"{status.get('memory_usage', 0):.1f}%",
                                                                                                                                "cpu_usage": f"{status.get('cpu_usage', 0):.1f}%",
                                                                                                                                "uptime": status.get("uptime", "Unknown"),
                                                                                                                            },
                                                                                                                        )
                                                                                                                        return self.send_message(
                                                                                                                            msg)

                                                                                                                        def send_custom(
                                                                                                                            self,
                                                                                                                            title: str,
                                                                                                                            message: str,
                                                                                                                            notification_type: NotificationType = NotificationType.INFO,
                                                                                                                            details: Optional[Dict] = None,
                                                                                                                        ):
                                                                                                                            """Send custom notification"""
                                                                                                                            msg = TelegramMessage(
                                                                                                                                type=notification_type, title=title, message=message, details=details)
                                                                                                                            return self.send_message(
                                                                                                                                msg)

                                                                                                                            # Global
                                                                                                                            # notifier
                                                                                                                            # instance
                                                                                                                            _telegram_notifier = None

                                                                                                                            def get_telegram_notifier(
                                                                                                                            ) -> Optional[TelegramNotifier]:
                                                                                                                                """Get global Telegram notifier instance"""
                                                                                                                                global _telegram_notifier
                                                                                                                                if _telegram_notifier is None:
                                                                                                                                    try:
                                                                                                                                        _telegram_notifier = TelegramNotifier()
                                                                                                                                        except Exception as e:
                                                                                                                                            logger.warning(
                                                                                                                                                f"Telegram notifier not available: {e}")
                                                                                                                                            return None
                                                                                                                                            return _telegram_notifier

                                                                                                                                            # Convenience
                                                                                                                                            # functions

                                                                                                                                            def notify_trial_start(
                                                                                                                                                    symbol: str, model: str, parameters: Dict):
                                                                                                                                                """Notify trial start (convenience function)"""
                                                                                                                                                notifier = get_telegram_notifier()
                                                                                                                                                if notifier:
                                                                                                                                                    return notifier.send_trial_start(
                                                                                                                                                        symbol, model, parameters)
                                                                                                                                                    return False

                                                                                                                                                    def notify_trial_complete(
                                                                                                                                                            symbol: str, model: str, results: Dict):
                                                                                                                                                        """Notify trial completion (convenience function)"""
                                                                                                                                                        notifier = get_telegram_notifier()
                                                                                                                                                        if notifier:
                                                                                                                                                            return notifier.send_trial_complete(
                                                                                                                                                                symbol, model, results)
                                                                                                                                                            return False

                                                                                                                                                            def notify_error(
                                                                                                                                                                    error_type: str, error_message: str, context: Optional[Dict] = None):
                                                                                                                                                                """Notify error (convenience function)"""
                                                                                                                                                                notifier = get_telegram_notifier()
                                                                                                                                                                if notifier:
                                                                                                                                                                    return notifier.send_error(
                                                                                                                                                                        error_type, error_message, context)
                                                                                                                                                                    return False

                                                                                                                                                                    def send_notification(
                                                                                                                                                                            message: str) -> bool:
                                                                                                                                                                        """Send notification via Telegram"""
                                                                                                                                                                        try:
                                                                                                                                                                            notifier = TelegramNotifier()
                                                                                                                                                                            msg = TelegramMessage(
                                                                                                                                                                                type=NotificationType.INFO, title="mlTrainer Notification", message=message)
                                                                                                                                                                            return notifier.send_message(
                                                                                                                                                                                msg)
                                                                                                                                                                            except Exception as e:
                                                                                                                                                                                logger.error(
                                                                                                                                                                                    f"Failed to send notification: {e}")
                                                                                                                                                                                return False

                                                                                                                                                                                def test_telegram_connection():
                                                                                                                                                                                    """Test Telegram bot connection"""
                                                                                                                                                                                    try:
                                                                                                                                                                                        notifier = TelegramNotifier()
                                                                                                                                                                                        msg = TelegramMessage(
                                                                                                                                                                                            type=NotificationType.INFO, title="Connection Test", message="Test message from mlTrainer"
                                                                                                                                                                                        )
                                                                                                                                                                                        success = notifier.send_message(
                                                                                                                                                                                            msg)
                                                                                                                                                                                        if success:
                                                                                                                                                                                            logger.info(
                                                                                                                                                                                                "Telegram connection test successful")
                                                                                                                                                                                            else:
                                                                                                                                                                                                logger.error(
                                                                                                                                                                                                    "Telegram connection test failed")
                                                                                                                                                                                                return success
                                                                                                                                                                                                except Exception as e:
                                                                                                                                                                                                    logger.error(
                                                                                                                                                                                                        f"Telegram connection test error: {e}")
                                                                                                                                                                                                    return False

                                                                                                                                                                                                    if __name__ == "__main__":
                                                                                                                                                                                                        test_telegram_connection()
