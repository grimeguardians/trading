import asyncio
import logging
from contextlib import asynccontextmanager
from main import TradingSystem

# Alternative entry point for direct system operation
async def run_trading_system():
    """Direct system runner for development and testing"""
    system = TradingSystem()
    
    try:
        await system.initialize()
        await system.start_services()
        
        # Start dashboard
        system.start_dashboard()
        
        # Keep running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logging.info("Received shutdown signal")
    except Exception as e:
        logging.error(f"System error: {e}")
    finally:
        await system.stop_services()

if __name__ == "__main__":
    asyncio.run(run_trading_system())
