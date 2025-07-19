#!/usr/bin/env python3
"""
Dashboard Runner - Starts the Streamlit dashboard
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def main():
    """Run the Streamlit dashboard"""
    # Set environment variables for Streamlit
    os.environ['STREAMLIT_SERVER_PORT'] = '5000'
    os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    
    # Run Streamlit
    cmd = [
        sys.executable, '-m', 'streamlit', 'run', 
        'dashboard/streamlit_app.py',
        '--server.port=5000',
        '--server.address=0.0.0.0',
        '--server.headless=true'
    ]
    
    print("Starting Advanced AI Trading Dashboard...")
    print("Dashboard will be available at: http://0.0.0.0:5000")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        print(f"Error running dashboard: {e}")

if __name__ == "__main__":
    main()
