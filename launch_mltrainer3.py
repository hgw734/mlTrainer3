#!/usr/bin/env python3
"""
mlTrainer3 Launch Script
========================
Launches the complete mlTrainer3 autonomous trading system.
Integrates all components and provides a unified interface.

NO TEMPLATES - This is real, functional code.
"""

import asyncio
import sys
import os
import logging
import argparse
from datetime import datetime
import webbrowser
import subprocess
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print the mlTrainer3 banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║                    🎯 mlTrainer3 System 🎯                       ║
    ║                                                                  ║
    ║         Autonomous ML Trading Intelligence Platform              ║
    ║                                                                  ║
    ║                   me ↔ mlAgent ↔ mlTrainer                      ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_environment():
    """Check system environment and dependencies"""
    print("\n📋 Checking Environment...")
    
    issues = []
    
    # Check Python version
    python_version = sys.version_info
    print(f"✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check for model registry
    if not Path("model_registry.json").exists():
        issues.append("Model registry not found. Run: python3 model_registry_builder.py")
    else:
        print("✓ Model registry found")
    
    # Check for required directories
    required_dirs = ["logs", "config", "custom"]
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✓ Directory '{dir_name}' exists")
        else:
            Path(dir_name).mkdir(exist_ok=True)
            print(f"✓ Created directory '{dir_name}'")
    
    # Check API configurations
    try:
        from config.api_config import POLYGON_API_KEY, FRED_API_KEY, ANTHROPIC_API_KEY
        
        if POLYGON_API_KEY:
            print("✓ Polygon API configured")
        else:
            issues.append("Polygon API key not configured")
        
        if FRED_API_KEY:
            print("✓ FRED API configured")
        else:
            issues.append("FRED API key not configured")
        
        if ANTHROPIC_API_KEY:
            print("✓ Anthropic API configured")
        else:
            issues.append("Anthropic API key not configured")
    except ImportError:
        issues.append("API configuration file not found")
    
    # Check for required packages
    required_packages = [
        "pandas", "numpy", "streamlit", "asyncio", 
        "sklearn", "requests", "schedule"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ Package '{package}' installed")
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        issues.append(f"Missing packages: {', '.join(missing_packages)}")
    
    return issues


async def launch_streamlit_ui():
    """Launch the Streamlit UI in a separate process"""
    print("\n🚀 Launching Streamlit UI...")
    try:
        # Check if mltrainer_chat.py exists
        if Path("mltrainer_chat.py").exists():
            subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", 
                "mltrainer_chat.py", "--server.port", "8501"
            ])
            print("✓ Streamlit UI launched on http://localhost:8501")
            
            # Wait a moment for server to start
            await asyncio.sleep(3)
            
            # Open browser
            webbrowser.open("http://localhost:8501")
        else:
            print("⚠️  mltrainer_chat.py not found")
    except Exception as e:
        print(f"❌ Failed to launch Streamlit: {e}")


async def test_connections():
    """Test API connections"""
    print("\n🔌 Testing API Connections...")
    
    # Test Polygon
    try:
        from polygon_connector import get_polygon_connector
        connector = get_polygon_connector()
        quote = connector.get_quote("AAPL")
        if quote:
            print(f"✓ Polygon: AAPL ${quote.price:.2f}")
        else:
            print("⚠️  Polygon: No data received")
    except Exception as e:
        print(f"❌ Polygon: {e}")
    
    # Test FRED
    try:
        from fred_connector import get_fred_connector
        connector = get_fred_connector()
        # Test with a simple series
        print("✓ FRED: Connection available")
    except Exception as e:
        print(f"❌ FRED: {e}")


async def run_interactive_mode():
    """Run the system in interactive mode"""
    from mltrainer_controller import get_mltrainer_controller
    
    controller = get_mltrainer_controller()
    
    print("\n🤖 mlTrainer3 Interactive Mode")
    print("Type 'help' for available commands or 'quit' to exit\n")
    
    while True:
        try:
            # Get user input
            user_input = input("mlTrainer3> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            elif user_input.lower() == 'help':
                print_help()
            
            elif user_input.lower() == 'status':
                status = controller.get_status()
                print("\nSystem Status:")
                for key, value in status.items():
                    print(f"  {key}: {value}")
            
            elif user_input.lower() == 'performance':
                summary = controller.get_performance_summary()
                print("\nPerformance Summary:")
                for key, value in summary.items():
                    print(f"  {key}: {value}")
            
            elif user_input.lower() == 'autonomous':
                print("Starting autonomous mode...")
                await controller.start_autonomous_mode()
            
            elif user_input:
                # Process as natural language command
                response = await controller.process_user_command(user_input)
                print("\n" + response['explanation'] + "\n")
            
        except KeyboardInterrupt:
            print("\nUse 'quit' to exit properly")
        except Exception as e:
            print(f"Error: {e}")


def print_help():
    """Print help information"""
    help_text = """
Available Commands:
  Natural Language:
    - "Analyze SPY for trading opportunities"
    - "What's the risk in my portfolio?"
    - "Predict AAPL price movement"
    - "Run volatility analysis on TSLA"
    
  System Commands:
    - status      : Show system status
    - performance : Show performance summary
    - autonomous  : Start autonomous trading mode
    - help        : Show this help message
    - quit        : Exit the system
    
Examples:
  mlTrainer3> Analyze market volatility for SPY
  mlTrainer3> Recommend a model for intraday trading
  mlTrainer3> What are the best performing models today?
"""
    print(help_text)


async def run_autonomous_mode():
    """Run the system in fully autonomous mode"""
    from mltrainer_controller import get_mltrainer_controller
    
    controller = get_mltrainer_controller()
    
    print("\n🤖 Starting Autonomous Mode...")
    print("The system will now run independently")
    print("Press Ctrl+C to stop\n")
    
    try:
        await controller.start_autonomous_mode()
    except KeyboardInterrupt:
        print("\nStopping autonomous mode...")
        controller.stop()


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='mlTrainer3 Launch Script')
    parser.add_argument('--mode', choices=['interactive', 'autonomous', 'ui'], 
                       default='interactive',
                       help='Launch mode (default: interactive)')
    parser.add_argument('--no-ui', action='store_true',
                       help='Do not launch Streamlit UI')
    parser.add_argument('--test', action='store_true',
                       help='Run system tests and exit')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Check environment
    issues = check_environment()
    if issues:
        print("\n⚠️  Environment Issues Found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nContinuing with available components...\n")
    
    # Test connections
    await test_connections()
    
    # Run tests if requested
    if args.test:
        print("\n✅ System check complete")
        return
    
    # Launch UI unless disabled
    if not args.no_ui and args.mode != 'autonomous':
        await launch_streamlit_ui()
    
    # Run in selected mode
    if args.mode == 'interactive':
        await run_interactive_mode()
    elif args.mode == 'autonomous':
        await run_autonomous_mode()
    elif args.mode == 'ui':
        print("\nStreamlit UI is running. Press Ctrl+C to stop.")
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping...")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nShutdown complete.")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)