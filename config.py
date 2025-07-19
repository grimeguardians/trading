"""
Configuration management for the Advanced AI Trading System
Following 12-factor app principles with environment-based configuration
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import timedelta


@dataclass
class ExchangeConfig:
    """Configuration for individual exchanges"""
    name: str
    enabled: bool
    api_key: str
    api_secret: str
    passphrase: Optional[str] = None
    sandbox: bool = True
    rate_limit: int = 100  # requests per minute
    supported_assets: List[str] = None


class Config:
    """Main configuration class with environment variable support"""
    
    # Application Settings
    DEBUG = os.environ.get("DEBUG", "false").lower() == "true"
    SECRET_KEY = os.environ.get("SECRET_KEY", "advanced-trading-system-secret")
    
    # Database Configuration
    DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///trading_system.db")
    
    # MCP Server Configuration
    MCP_HOST = os.environ.get("MCP_HOST", "127.0.0.1")
    MCP_PORT = int(os.environ.get("MCP_PORT", "9000"))
    
    # AI/ML Configuration
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
    
    # Trading Configuration
    PAPER_TRADING = os.environ.get("PAPER_TRADING", "true").lower() == "true"
    MAX_POSITION_SIZE = float(os.environ.get("MAX_POSITION_SIZE", "0.1"))  # 10% of portfolio
    RISK_PER_TRADE = float(os.environ.get("RISK_PER_TRADE", "0.02"))  # 2% risk per trade
    
    # Exchange API Keys and Configuration
    EXCHANGES = {
        "alpaca": ExchangeConfig(
            name="alpaca",
            enabled=os.environ.get("ALPACA_ENABLED", "true").lower() == "true",
            api_key=os.environ.get("ALPACA_API_KEY", ""),
            api_secret=os.environ.get("ALPACA_SECRET_KEY", ""),
            sandbox=os.environ.get("ALPACA_SANDBOX", "true").lower() == "true",
            supported_assets=["stocks", "crypto", "etf", "options"]
        ),
        "binance": ExchangeConfig(
            name="binance",
            enabled=os.environ.get("BINANCE_ENABLED", "false").lower() == "true",
            api_key=os.environ.get("BINANCE_API_KEY", ""),
            api_secret=os.environ.get("BINANCE_SECRET_KEY", ""),
            sandbox=os.environ.get("BINANCE_SANDBOX", "true").lower() == "true",
            supported_assets=["crypto", "futures"]
        ),
        "kucoin": ExchangeConfig(
            name="kucoin",
            enabled=os.environ.get("KUCOIN_ENABLED", "false").lower() == "true",
            api_key=os.environ.get("KUCOIN_API_KEY", ""),
            api_secret=os.environ.get("KUCOIN_SECRET_KEY", ""),
            passphrase=os.environ.get("KUCOIN_PASSPHRASE", ""),
            sandbox=os.environ.get("KUCOIN_SANDBOX", "true").lower() == "true",
            supported_assets=["crypto", "futures"]
        ),
        "td_ameritrade": ExchangeConfig(
            name="td_ameritrade",
            enabled=os.environ.get("TD_ENABLED", "false").lower() == "true",
            api_key=os.environ.get("TD_API_KEY", ""),
            api_secret=os.environ.get("TD_SECRET_KEY", ""),
            sandbox=os.environ.get("TD_SANDBOX", "true").lower() == "true",
            supported_assets=["stocks", "options", "futures"]
        )
    }
    
    # Strategy Configuration
    STRATEGIES = {
        "swing": {
            "enabled": os.environ.get("SWING_ENABLED", "true").lower() == "true",
            "timeframe": "1h",
            "max_positions": int(os.environ.get("SWING_MAX_POSITIONS", "5")),
            "min_profit_target": float(os.environ.get("SWING_MIN_PROFIT", "0.05"))
        },
        "scalping": {
            "enabled": os.environ.get("SCALPING_ENABLED", "false").lower() == "true",
            "timeframe": "1m",
            "max_positions": int(os.environ.get("SCALPING_MAX_POSITIONS", "3")),
            "min_profit_target": float(os.environ.get("SCALPING_MIN_PROFIT", "0.01"))
        },
        "options": {
            "enabled": os.environ.get("OPTIONS_ENABLED", "false").lower() == "true",
            "max_positions": int(os.environ.get("OPTIONS_MAX_POSITIONS", "10")),
            "min_profit_target": float(os.environ.get("OPTIONS_MIN_PROFIT", "0.20"))
        },
        "intraday": {
            "enabled": os.environ.get("INTRADAY_ENABLED", "true").lower() == "true",
            "timeframe": "15m",
            "max_positions": int(os.environ.get("INTRADAY_MAX_POSITIONS", "8")),
            "min_profit_target": float(os.environ.get("INTRADAY_MIN_PROFIT", "0.03"))
        }
    }
    
    # Technical Analysis Configuration
    FIBONACCI_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.618, 2.618]
    
    # Risk Management
    STOP_LOSS_PERCENTAGE = float(os.environ.get("STOP_LOSS_PCT", "0.02"))  # 2%
    TAKE_PROFIT_PERCENTAGE = float(os.environ.get("TAKE_PROFIT_PCT", "0.06"))  # 6%
    MAX_DRAWDOWN = float(os.environ.get("MAX_DRAWDOWN", "0.10"))  # 10%
    
    # Machine Learning Configuration
    ML_MODEL_UPDATE_INTERVAL = int(os.environ.get("ML_UPDATE_INTERVAL", "3600"))  # seconds
    ML_MIN_ACCURACY_THRESHOLD = float(os.environ.get("ML_MIN_ACCURACY", "0.60"))
    
    # Logging Configuration
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Web Interface Configuration
    DASHBOARD_PORT = int(os.environ.get("DASHBOARD_PORT", "5000"))
    API_PORT = int(os.environ.get("API_PORT", "8000"))
    
    # Performance Optimization
    CACHE_TTL = int(os.environ.get("CACHE_TTL", "300"))  # 5 minutes
    MAX_CACHE_SIZE = int(os.environ.get("MAX_CACHE_SIZE", "1000"))
    
    # Market Data Configuration
    DATA_SOURCES = {
        "primary": os.environ.get("PRIMARY_DATA_SOURCE", "alpaca"),
        "fallback": os.environ.get("FALLBACK_DATA_SOURCE", "yahoo"),
        "realtime": os.environ.get("REALTIME_DATA", "true").lower() == "true"
    }
    
    # Notification Configuration
    SLACK_WEBHOOK = os.environ.get("SLACK_WEBHOOK", "")
    EMAIL_ENABLED = os.environ.get("EMAIL_ENABLED", "false").lower() == "true"
    EMAIL_SMTP_SERVER = os.environ.get("EMAIL_SMTP_SERVER", "")
    EMAIL_SMTP_PORT = int(os.environ.get("EMAIL_SMTP_PORT", "587"))
    EMAIL_USERNAME = os.environ.get("EMAIL_USERNAME", "")
    EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "")
    
    # Backtesting Configuration
    BACKTEST_START_DATE = os.environ.get("BACKTEST_START_DATE", "2020-01-01")
    BACKTEST_END_DATE = os.environ.get("BACKTEST_END_DATE", "2024-12-31")
    BACKTEST_INITIAL_CAPITAL = float(os.environ.get("BACKTEST_CAPITAL", "100000"))
    
    @classmethod
    def get_enabled_exchanges(cls) -> List[str]:
        """Get list of enabled exchanges"""
        return [name for name, config in cls.EXCHANGES.items() if config.enabled]
    
    @classmethod
    def get_exchange_config(cls, exchange_name: str) -> Optional[ExchangeConfig]:
        """Get configuration for specific exchange"""
        return cls.EXCHANGES.get(exchange_name)
    
    @classmethod
    def get_enabled_strategies(cls) -> List[str]:
        """Get list of enabled strategies"""
        return [name for name, config in cls.STRATEGIES.items() if config.get("enabled", False)]
    
    @classmethod
    def validate_configuration(cls) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Check required AI API keys
        if not cls.OPENAI_API_KEY and not cls.ANTHROPIC_API_KEY:
            issues.append("At least one AI API key (OpenAI or Anthropic) is required")
        
        # Check exchange configurations
        enabled_exchanges = cls.get_enabled_exchanges()
        if not enabled_exchanges:
            issues.append("At least one exchange must be enabled")
        
        for exchange_name in enabled_exchanges:
            config = cls.get_exchange_config(exchange_name)
            if not config.api_key:
                issues.append(f"API key missing for {exchange_name}")
            if not config.api_secret:
                issues.append(f"API secret missing for {exchange_name}")
        
        # Check strategy configurations
        enabled_strategies = cls.get_enabled_strategies()
        if not enabled_strategies:
            issues.append("At least one strategy must be enabled")
        
        # Check risk management parameters
        if cls.STOP_LOSS_PERCENTAGE <= 0 or cls.STOP_LOSS_PERCENTAGE >= 1:
            issues.append("Stop loss percentage must be between 0 and 1")
        
        if cls.TAKE_PROFIT_PERCENTAGE <= 0:
            issues.append("Take profit percentage must be positive")
        
        if cls.MAX_DRAWDOWN <= 0 or cls.MAX_DRAWDOWN >= 1:
            issues.append("Max drawdown must be between 0 and 1")
        
        return issues
    
    @classmethod
    def print_configuration_summary(cls):
        """Print configuration summary for debugging"""
        print("\n🔧 Configuration Summary:")
        print(f"   Debug Mode: {cls.DEBUG}")
        print(f"   Paper Trading: {cls.PAPER_TRADING}")
        print(f"   Enabled Exchanges: {cls.get_enabled_exchanges()}")
        print(f"   Enabled Strategies: {cls.get_enabled_strategies()}")
        print(f"   Max Position Size: {cls.MAX_POSITION_SIZE * 100}%")
        print(f"   Risk Per Trade: {cls.RISK_PER_TRADE * 100}%")
        print(f"   Stop Loss: {cls.STOP_LOSS_PERCENTAGE * 100}%")
        print(f"   Take Profit: {cls.TAKE_PROFIT_PERCENTAGE * 100}%")
        print(f"   Max Drawdown: {cls.MAX_DRAWDOWN * 100}%")
        print(f"   Dashboard Port: {cls.DASHBOARD_PORT}")
        print(f"   API Port: {cls.API_PORT}")
        print()
