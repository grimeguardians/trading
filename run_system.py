#!/usr/bin/env python3
"""
AI Trading System Launcher
Starts both the API server and Streamlit dashboard
"""

import os
import sys
import time
import signal
import logging
import subprocess
import threading
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradingSystemLauncher:
    """Main launcher for the AI Trading System"""
    
    def __init__(self):
        self.processes = []
        self.running = False
        
    def start_api_server(self):
        """Start the FastAPI backend server"""
        try:
            logger.info("🚀 Starting API server on port 8000...")
            
            # Start API server
            api_process = subprocess.Popen([
                sys.executable, "-m", "uvicorn", "api_server:app",
                "--host", "0.0.0.0",
                "--port", "8000",
                "--reload"
            ])
            
            self.processes.append(("API Server", api_process))
            
            # Wait a moment for server to start
            time.sleep(3)
            logger.info("✅ API server started successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to start API server: {e}")
            
    def start_dashboard(self):
        """Start the Streamlit dashboard"""
        try:
            logger.info("🎛️ Starting Streamlit dashboard on port 5000...")
            
            # Start Streamlit dashboard
            dashboard_process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run",
                "dashboard/streamlit_app.py",
                "--server.port", "5000",
                "--server.address", "0.0.0.0",
                "--server.headless", "true"
            ])
            
            self.processes.append(("Dashboard", dashboard_process))
            
            # Wait a moment for dashboard to start
            time.sleep(5)
            logger.info("✅ Dashboard started successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to start dashboard: {e}")
            
    def start_system(self):
        """Start the complete trading system"""
        try:
            logger.info("🎯 Starting AI Trading System...")
            
            # Display startup banner
            self.display_banner()
            
            # Start API server first
            self.start_api_server()
            
            # Start dashboard
            self.start_dashboard()
            
            self.running = True
            logger.info("🌟 AI Trading System started successfully!")
            
            # Display access URLs
            self.display_urls()
            
            # Monitor processes
            self.monitor_processes()
            
        except Exception as e:
            logger.error(f"❌ Failed to start system: {e}")
            self.shutdown()
            
    def monitor_processes(self):
        """Monitor running processes"""
        logger.info("📊 Monitoring system processes...")
        
        while self.running:
            try:
                # Check if all processes are still running
                for name, process in self.processes:
                    if process.poll() is not None:
                        logger.error(f"❌ {name} has stopped unexpectedly")
                        self.shutdown()
                        return
                
                # Sleep for a short interval
                time.sleep(5)
                
            except KeyboardInterrupt:
                logger.info("📡 Received interrupt signal, shutting down...")
                self.shutdown()
                break
            except Exception as e:
                logger.error(f"Error monitoring processes: {e}")
                time.sleep(10)
                
    def shutdown(self):
        """Shutdown all processes"""
        logger.info("🛑 Shutting down AI Trading System...")
        
        self.running = False
        
        # Terminate all processes
        for name, process in self.processes:
            try:
                logger.info(f"🔴 Stopping {name}...")
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"⚠️ Force killing {name}...")
                    process.kill()
                    
            except Exception as e:
                logger.error(f"Error stopping {name}: {e}")
                
        logger.info("✅ System shutdown complete")
        
    def display_banner(self):
        """Display startup banner"""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║               🤖 AI TRADING SYSTEM v2.0 🤖                   ║
║                                                              ║
║  • Multi-Exchange Support (Alpaca, TD Ameritrade, Binance)  ║
║  • Digital Brain Integration with Advanced Knowledge Base   ║
║  • MCP Agent Communication Protocol                         ║
║  • Advanced Mathematical Models & Fibonacci Analysis        ║
║  • Real-time Conversational AI Interface                    ║
║  • Production-Ready Architecture                            ║
║                                                              ║
║              Starting system components...                   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)
        
    def display_urls(self):
        """Display access URLs"""
        urls = """
╔══════════════════════════════════════════════════════════════╗
║                        🌐 ACCESS URLS 🌐                     ║
║                                                              ║
║  📊 Trading Dashboard: http://localhost:5000                ║
║  🔌 API Server:        http://localhost:8000                ║
║  📖 API Docs:          http://localhost:8000/docs           ║
║  🩺 Health Check:      http://localhost:8000/health         ║
║                                                              ║
║                    Press Ctrl+C to stop                     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(urls)
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"📡 Received signal {signum}, shutting down...")
        self.shutdown()
        sys.exit(0)

def main():
    """Main entry point"""
    # Create launcher instance
    launcher = TradingSystemLauncher()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, launcher.signal_handler)
    signal.signal(signal.SIGTERM, launcher.signal_handler)
    
    try:
        # Start the system
        launcher.start_system()
        
    except KeyboardInterrupt:
        logger.info("🔴 Received KeyboardInterrupt, shutting down...")
        launcher.shutdown()
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        launcher.shutdown()
        sys.exit(1)

if __name__ == "__main__":
    main()