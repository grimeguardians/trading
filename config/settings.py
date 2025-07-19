"""
Settings module for AI Trading System
"""
import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from functools import lru_cache

@dataclass
class Settings:
    """Application settings"""
    # Database
    database_url: str = os.getenv("DATABASE_URL", "postgresql://localhost/trading_db")
    
    # API Keys
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    
    # Exchange Keys
    alpaca_api_key: str = os.getenv("ALPACA_API_KEY", "")
    alpaca_secret_key: str = os.getenv("ALPACA_SECRET_KEY", "")
    alpaca_sandbox: bool = os.getenv("ALPACA_SANDBOX", "true").lower() == "true"
    
    binance_api_key: str = os.getenv("BINANCE_API_KEY", "")
    binance_secret_key: str = os.getenv("BINANCE_SECRET_KEY", "")
    
    # Trading Settings
    paper_trading: bool = os.getenv("PAPER_TRADING", "true").lower() == "true"
    max_position_size: float = float(os.getenv("MAX_POSITION_SIZE", "0.1"))
    risk_per_trade: float = float(os.getenv("RISK_PER_TRADE", "0.02"))
    
    # Application
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    port: int = int(os.getenv("PORT", "8000"))
    host: str = os.getenv("HOST", "0.0.0.0")

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()