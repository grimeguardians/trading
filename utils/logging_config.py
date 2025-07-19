"""
Logging configuration
"""
import logging
import sys
from datetime import datetime

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('trading_system.log')
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized")