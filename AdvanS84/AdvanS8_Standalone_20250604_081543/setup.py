#!/usr/bin/env python3
"""
AdvanS8 Standalone Setup Script
Installs dependencies and configures the system
"""

import subprocess
import sys
import os

def install_requirements():
    """Install Python requirements"""
    print("Installing Python dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def setup_environment():
    """Setup environment variables"""
    print("Setting up environment...")
    
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write("""# AdvanS8 Configuration
# Add your API keys here
POLYGON_API_KEY=your_polygon_api_key_here
FRED_API_KEY=your_fred_api_key_here
DATABASE_URL=your_database_url_here

# Optional: Twilio for SMS alerts
TWILIO_ACCOUNT_SID=your_twilio_sid_here
TWILIO_AUTH_TOKEN=your_twilio_token_here
TWILIO_PHONE_NUMBER=your_twilio_phone_here
""")
        print("Created .env file - Please add your API keys")

def create_launch_script():
    """Create launch script"""
    with open('run_advans8.py', 'w') as f:
        f.write("""#!/usr/bin/env python3
import subprocess
import sys

def main():
    print("Starting AdvanS8 Live Trading Dashboard...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "AdvanS8_Live_Trading_Dashboard.py", "--server.port", "5000"])

if __name__ == "__main__":
    main()
""")
    
    # Make executable on Unix systems
    try:
        os.chmod('run_advans8.py', 0o755)
    except:
        pass

def main():
    print("AdvanS8 Standalone Setup")
    print("=" * 40)
    
    install_requirements()
    setup_environment()
    create_launch_script()
    
    print("\nSetup complete!")
    print("\nNext steps:")
    print("1. Edit .env file with your API keys")
    print("2. Run: python run_advans8.py")
    print("3. Open browser to: http://localhost:5000")

if __name__ == "__main__":
    main()
