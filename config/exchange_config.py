"""
Exchange-specific configuration and validation
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
from enum import Enum

class ExchangeType(str, Enum):
    """Supported exchange types"""
    STOCKS = "stocks"
    CRYPTO = "crypto"
    FOREX = "forex"
    OPTIONS = "options"
    FUTURES = "futures"

class AssetClass(str, Enum):
    """Supported asset classes"""
    STOCK = "stock"
    CRYPTO = "crypto"
    ETF = "etf"
    OPTION = "option"
    FUTURE = "future"
    FOREX = "forex"

class ExchangeConfig(BaseModel):
    """Base exchange configuration"""
    name: str
    display_name: str
    exchange_type: List[ExchangeType]
    supported_assets: List[AssetClass]
    api_url: str
    sandbox_url: Optional[str] = None
    rate_limit: int = Field(default=100, description="Requests per minute")
    min_order_value: float = Field(default=1.0, description="Minimum order value in USD")
    max_order_value: float = Field(default=1000000.0, description="Maximum order value in USD")
    supported_order_types: List[str] = Field(default=["market", "limit", "stop", "stop_limit"])
    trading_hours: Dict[str, Any] = Field(default_factory=dict)
    commission_structure: Dict[str, float] = Field(default_factory=dict)
    margin_enabled: bool = Field(default=False)
    short_selling_enabled: bool = Field(default=False)
    
    class Config:
        use_enum_values = True

class AlpacaConfig(ExchangeConfig):
    """Alpaca Markets configuration"""
    name: str = "alpaca"
    display_name: str = "Alpaca Markets"
    exchange_type: List[ExchangeType] = [ExchangeType.STOCKS, ExchangeType.CRYPTO, ExchangeType.OPTIONS]
    supported_assets: List[AssetClass] = [AssetClass.STOCK, AssetClass.CRYPTO, AssetClass.ETF, AssetClass.OPTION]
    api_url: str = "https://api.alpaca.markets"
    sandbox_url: str = "https://paper-api.alpaca.markets"
    rate_limit: int = 200
    min_order_value: float = 1.0
    max_order_value: float = 1000000.0
    supported_order_types: List[str] = ["market", "limit", "stop", "stop_limit", "trailing_stop"]
    trading_hours: Dict[str, Any] = {
        "market_open": "09:30",
        "market_close": "16:00",
        "timezone": "US/Eastern",
        "extended_hours": True
    }
    commission_structure: Dict[str, float] = {
        "stocks": 0.0,
        "options": 0.65,
        "crypto": 0.0025
    }
    margin_enabled: bool = True
    short_selling_enabled: bool = True

class BinanceConfig(ExchangeConfig):
    """Binance configuration"""
    name: str = "binance"
    display_name: str = "Binance"
    exchange_type: List[ExchangeType] = [ExchangeType.CRYPTO, ExchangeType.FUTURES]
    supported_assets: List[AssetClass] = [AssetClass.CRYPTO, AssetClass.FUTURE]
    api_url: str = "https://api.binance.com"
    sandbox_url: str = "https://testnet.binance.vision"
    rate_limit: int = 1200
    min_order_value: float = 10.0
    max_order_value: float = 10000000.0
    supported_order_types: List[str] = ["market", "limit", "stop_market", "stop_limit", "take_profit", "take_profit_limit"]
    trading_hours: Dict[str, Any] = {
        "24_7": True,
        "timezone": "UTC"
    }
    commission_structure: Dict[str, float] = {
        "spot": 0.001,
        "futures": 0.0004
    }
    margin_enabled: bool = True
    short_selling_enabled: bool = True

class KuCoinConfig(ExchangeConfig):
    """KuCoin configuration"""
    name: str = "kucoin"
    display_name: str = "KuCoin"
    exchange_type: List[ExchangeType] = [ExchangeType.CRYPTO, ExchangeType.FUTURES]
    supported_assets: List[AssetClass] = [AssetClass.CRYPTO, AssetClass.FUTURE]
    api_url: str = "https://api.kucoin.com"
    sandbox_url: str = "https://openapi-sandbox.kucoin.com"
    rate_limit: int = 180
    min_order_value: float = 1.0
    max_order_value: float = 5000000.0
    supported_order_types: List[str] = ["market", "limit", "stop", "stop_limit"]
    trading_hours: Dict[str, Any] = {
        "24_7": True,
        "timezone": "UTC"
    }
    commission_structure: Dict[str, float] = {
        "spot": 0.001,
        "futures": 0.0006
    }
    margin_enabled: bool = True
    short_selling_enabled: bool = True

class TDAmeritadeConfig(ExchangeConfig):
    """TD Ameritrade configuration"""
    name: str = "td_ameritrade"
    display_name: str = "TD Ameritrade"
    exchange_type: List[ExchangeType] = [ExchangeType.STOCKS, ExchangeType.OPTIONS, ExchangeType.FUTURES]
    supported_assets: List[AssetClass] = [AssetClass.STOCK, AssetClass.ETF, AssetClass.OPTION, AssetClass.FUTURE]
    api_url: str = "https://api.tdameritrade.com"
    sandbox_url: str = "https://api.tdameritrade.com"  # TD Ameritrade uses same endpoint
    rate_limit: int = 120
    min_order_value: float = 1.0
    max_order_value: float = 2000000.0
    supported_order_types: List[str] = ["market", "limit", "stop", "stop_limit", "trailing_stop"]
    trading_hours: Dict[str, Any] = {
        "market_open": "09:30",
        "market_close": "16:00",
        "timezone": "US/Eastern",
        "extended_hours": True
    }
    commission_structure: Dict[str, float] = {
        "stocks": 0.0,
        "options": 0.65,
        "futures": 2.25
    }
    margin_enabled: bool = True
    short_selling_enabled: bool = True

class ExchangeConfigManager:
    """Manages exchange configurations"""
    
    def __init__(self):
        self.configs = {
            "alpaca": AlpacaConfig(),
            "binance": BinanceConfig(),
            "kucoin": KuCoinConfig(),
            "td_ameritrade": TDAmeritadeConfig()
        }
    
    def get_config(self, exchange_name: str) -> Optional[ExchangeConfig]:
        """Get configuration for an exchange"""
        return self.configs.get(exchange_name)
    
    def get_all_configs(self) -> Dict[str, ExchangeConfig]:
        """Get all exchange configurations"""
        return self.configs
    
    def get_exchanges_by_asset_class(self, asset_class: AssetClass) -> List[str]:
        """Get exchanges that support a specific asset class"""
        exchanges = []
        for name, config in self.configs.items():
            if asset_class in config.supported_assets:
                exchanges.append(name)
        return exchanges
    
    def get_exchanges_by_type(self, exchange_type: ExchangeType) -> List[str]:
        """Get exchanges that support a specific exchange type"""
        exchanges = []
        for name, config in self.configs.items():
            if exchange_type in config.exchange_type:
                exchanges.append(name)
        return exchanges
    
    def validate_order_params(self, exchange_name: str, order_value: float, order_type: str) -> bool:
        """Validate order parameters against exchange limits"""
        config = self.get_config(exchange_name)
        if not config:
            return False
        
        # Check order value limits
        if order_value < config.min_order_value or order_value > config.max_order_value:
            return False
        
        # Check order type support
        if order_type not in config.supported_order_types:
            return False
        
        return True
    
    def get_commission(self, exchange_name: str, asset_class: str) -> float:
        """Get commission for an exchange and asset class"""
        config = self.get_config(exchange_name)
        if not config:
            return 0.0
        
        return config.commission_structure.get(asset_class, 0.0)
    
    def is_market_open(self, exchange_name: str) -> bool:
        """Check if market is open for an exchange"""
        config = self.get_config(exchange_name)
        if not config:
            return False
        
        trading_hours = config.trading_hours
        
        # 24/7 markets
        if trading_hours.get("24_7", False):
            return True
        
        # Regular market hours (simplified for demo)
        # In production, this would check actual market hours
        return True
    
    def get_rate_limit(self, exchange_name: str) -> int:
        """Get rate limit for an exchange"""
        config = self.get_config(exchange_name)
        if not config:
            return 60  # Default rate limit
        
        return config.rate_limit
    
    def supports_margin(self, exchange_name: str) -> bool:
        """Check if exchange supports margin trading"""
        config = self.get_config(exchange_name)
        if not config:
            return False
        
        return config.margin_enabled
    
    def supports_short_selling(self, exchange_name: str) -> bool:
        """Check if exchange supports short selling"""
        config = self.get_config(exchange_name)
        if not config:
            return False
        
        return config.short_selling_enabled

# Global instance
exchange_config_manager = ExchangeConfigManager()
