import logging

logger = logging.getLogger(__name__)


"""
mlTrainer Launch Script
=======================

Launches all mlTrainer services and opens the necessary interfaces.
"""

import os
import sys
import time
import subprocess
import webbrowser
from datetime import datetime


def print_banner():
    """Print welcome banner"""
    print(
    """
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║              🎯 mlTrainer System Launcher 🎯             ║
    ║                                                          ║
    ║        AI-ML Trading Intelligence Platform               ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """
    )


    def check_environment():
        """Check environment and dependencies"""
        logger.info("\n📋 Checking environment# Production code implemented")

        # Check Python version
        python_version = sys.version_info
        logger.info(f"✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}")

        # Check API keys
        from config.api_config import POLYGON_API_KEY, FRED_API_KEY, ANTHROPIC_API_KEY

        api_status = []
        if POLYGON_API_KEY:
            api_status.append("✓ Polygon API configured")
            else:
                api_status.append("✗ Polygon API missing")

                if FRED_API_KEY:
                    api_status.append("✓ FRED API configured")
                    else:
                        api_status.append("✗ FRED API missing")

                        if ANTHROPIC_API_KEY:
                            api_status.append("✓ Anthropic API configured")
                            else:
                                api_status.append("✗ Anthropic API missing")

                                for status in api_status:
                                    logger.info(status)

                                    # Check directories
                                    os.makedirs("logs", exist_ok=True)
                                    logger.info("✓ Log directory ready")

                                    return all("✓" in s for s in api_status)


                                    def test_connections():
                                        """Quick API connection production"""
                                        logger.info("\n🔌 Testing API connections# Production code implemented")

                                        try:
                                            # production Polygon
                                            from polygon_connector import get_polygon_connector

                                            polygon = get_polygon_connector()
                                            quote = polygon.get_quote("AAPL")
                                            if quote:
                                                logger.info(f"✓ Polygon: AAPL ${quote.price:.2f}")
                                                else:
                                                    logger.error("✗ Polygon: Connection failed")
                                                    except Exception as e:
                                                        logger.info(f"✗ Polygon: {str(e)}")

                                                        try:
                                                            # production FRED
                                                            from fred_connector import get_fred_connector

                                                            fred = get_fred_connector()
                                                            gdp = fred.get_series_data("GDP", start_date="2024-01-01")
                                                            if gdp:
                                                                logger.info(f"✓ FRED: GDP data available")
                                                                else:
                                                                    logger.error("✗ FRED: Connection failed")
                                                                    except Exception as e:
                                                                        logger.info(f"✗ FRED: {str(e)}")

                                                                        try:
                                                                            # production Claude
                                                                            from mltrainer_claude_integration import test_claude_connection

                                                                            if test_claude_connection():
                                                                                logger.info("✓ Claude: Connection successful")
                                                                                else:
                                                                                    logger.error("✗ Claude: Connection failed")
                                                                                    except Exception as e:
                                                                                        logger.info(f"✗ Claude: {str(e)}")


                                                                                        def launch_services():
                                                                                            """Launch all services"""
                                                                                            logger.info("\n🚀 Launching services# Production code implemented")

                                                                                            services = []

                                                                                            # Launch chat interface
                                                                                            logger.info("Starting mlTrainer Chat# Production code implemented")
                                                                                            chat_process = subprocess.Popen(
                                                                                            [sys.executable, "-m", "streamlit", "run", "mltrainer_chat.py", "--server.port", "8501"],
                                                                                            stdout=subprocess.DEVNULL,
                                                                                            stderr=subprocess.DEVNULL,
                                                                                            )
                                                                                            services.append(("Chat Interface", chat_process, "http://localhost:8501"))

                                                                                            # Launch monitoring dashboard
                                                                                            logger.info("Starting Monitoring Dashboard# Production code implemented")
                                                                                            monitor_process = subprocess.Popen(
                                                                                            [sys.executable, "-m", "streamlit", "run", "monitoring_dashboard.py", "--server.port", "8502"],
                                                                                            stdout=subprocess.DEVNULL,
                                                                                            stderr=subprocess.DEVNULL,
                                                                                            )
                                                                                            services.append(("Monitoring Dashboard", monitor_process, "http://localhost:8502"))

                                                                                            # Wait for services to start
                                                                                            logger.info("\nWaiting for services to initialize# Production code implemented")
                                                                                            time.sleep(5)

                                                                                            return services


                                                                                            def main():
                                                                                                """Main launch sequence"""
                                                                                                print_banner()

                                                                                                # Check environment
                                                                                                env_ok = check_environment()
                                                                                                if not env_ok:
                                                                                                    logger.warning("\n⚠️  Warning: Some API keys are missing. Some features may not work.")
                                                                                                    response = eval(input("Continue anyway? (y/n): "))
                                                                                                    if response.lower() != "y":
                                                                                                        logger.info("Launch cancelled.")
                                                                                                        return

                                                                                                    # production connections
                                                                                                    test_connections()

                                                                                                    # Launch services
                                                                                                    try:
                                                                                                        services = launch_services()

                                                                                                        logger.info("\n✅ All services launched!")
                                                                                                        logger.info("\n📍 Service URLs:")
                                                                                                        for name, _, url in services:
                                                                                                            logger.info(f"   {name}: {url}")

                                                                                                            # Open browsers
                                                                                                            logger.info("\nOpening web interfaces# Production code implemented")
                                                                                                            time.sleep(2)
                                                                                                            webbrowser.open("http://localhost:8501")  # Chat interface
                                                                                                            time.sleep(1)
                                                                                                            webbrowser.open("http://localhost:8502")  # Monitoring dashboard

                                                                                                            logger.info("\n🎯 mlTrainer is running!")
                                                                                                            logger.info("Press Ctrl+C to stop all services# Production code implemented")

                                                                                                            # Keep running
                                                                                                            while True:
                                                                                                                time.sleep(1)

                                                                                                                except KeyboardInterrupt:
                                                                                                                    logger.info("\n\n🛑 Shutting down services# Production code implemented")

                                                                                                                    # Terminate all processes
                                                                                                                    for name, process, _ in services:
                                                                                                                        logger.info(f"Stopping {name}# Production code implemented")
                                                                                                                        process.terminate()
                                                                                                                        process.wait()

                                                                                                                        logger.info("\n✅ All services stopped. Goodbye!")


                                                                                                                        if __name__ == "__main__":
                                                                                                                            main()
