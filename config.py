#!/usr/bin/env python3
"""
Unified Configuration Management for Trading Bot
Centralized settings with environment-specific overrides
"""

import os
from dataclasses import dataclass
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

@dataclass
class TradingConfig:
    """Centralized trading configuration"""
    
    # Environment
    environment: str = os.getenv('ENVIRONMENT', 'development')
    debug_mode: bool = os.getenv('DEBUG', 'true').lower() == 'true'
    
    # API Keys
    alpaca_api_key: str = os.getenv('ALPACA_API_KEY', '')
    alpaca_secret_key: str = os.getenv('ALPACA_SECRET_KEY', '')
    openai_api_key: str = os.getenv('OPENAI_API_KEY', '')
    
    # Trading Settings
    paper_trading: bool = True
    max_position_size: float = 1000.0
    default_quantity: int = 10
    risk_per_trade: float = 0.02  # 2% risk per trade
    
    # Asset Classes
    enabled_assets: List[str] = None
    symbols: Dict[str, List[str]] = None
    
    # Risk Management
    stop_loss_percentage: float = 0.05  # 5% stop loss
    take_profit_percentage: float = 0.10  # 10% take profit
    max_daily_loss: float = 0.05  # 5% max daily loss
    
    # Performance
    data_refresh_interval: int = 10  # seconds
    portfolio_check_interval: int = 60  # seconds
    max_concurrent_orders: int = 5
    
    # Logging
    log_level: str = os.getenv('LOG_LEVEL', 'INFO')
    log_file: str = 'trading_bot.log'
    
    def __post_init__(self):
        """Initialize default values after creation"""
        if self.enabled_assets is None:
            self.enabled_assets = ['stocks', 'etfs', 'crypto']
            
        if self.symbols is None:
            # Comprehensive trading symbols by asset class
            self.symbols = {
                'stocks': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX', 'CRM', 'PLTR'],
                'etfs': ['SPY', 'QQQ', 'VTI', 'ARKK', 'GLD', 'TLT', 'XLF', 'SOXL', 'TQQQ', 'SPXL'],
                'crypto': [
                    # Flagship crypto pairs available on Alpaca
                    'BTCUSD', 'ETHUSD', 'XRPUSD'
                ],
                'futures': ['ES', 'NQ', 'GC', 'CL', 'ZB', 'ZN', 'EUR', 'GBP'],
                'options': ['SPY_CALL', 'QQQ_CALL', 'AAPL_CALL', 'TSLA_CALL', 'SPY_PUT', 'QQQ_PUT']
            }
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment.lower() == 'production'
    
    @property
    def all_symbols(self) -> List[str]:
        """Get all symbols from enabled asset classes"""
        symbols = []
        for asset_class in self.enabled_assets:
            if asset_class in self.symbols:
                symbols.extend(self.symbols[asset_class])
        return symbols
    
    def get_symbols_for_asset(self, asset_class: str) -> List[str]:
        """Get symbols for specific asset class"""
        return self.symbols.get(asset_class, [])
    
    def update_config(self, **kwargs):
        """Dynamically update configuration"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def validate(self) -> bool:
        """Validate configuration"""
        if not self.alpaca_api_key or not self.alpaca_secret_key:
            print("❌ Missing Alpaca API credentials")
            return False
            
        if self.risk_per_trade > 0.1:  # More than 10%
            print("⚠️ Risk per trade is very high (>10%)")
            
        if self.stop_loss_percentage > 0.2:  # More than 20%
            print("⚠️ Stop loss percentage is very high (>20%)")
            
        return True

# Global configuration instance
config = TradingConfig()

# Environment-specific overrides
if config.environment == 'production':
    config.debug_mode = False
    config.log_level = 'WARNING'
    config.data_refresh_interval = 5
elif config.environment == 'testing':
    config.paper_trading = True
    config.max_position_size = 100.0
    config.symbols['stocks'] = ['AAPL', 'MSFT']  # Limited for testing

# Configuration validation
if not config.validate():
    print("⚠️ Configuration validation failed - check your settings")