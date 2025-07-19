"""
Exchange factory for creating exchange instances
"""

from typing import Dict, Optional
from .base_exchange import BaseExchange
from .alpaca_exchange import AlpacaExchange
from .binance_exchange import BinanceExchange
from .td_ameritrade_exchange import TDAmeritradeExchange
from config import ExchangeType, ExchangeConfig

class ExchangeFactory:
    """Factory for creating exchange instances"""
    
    _exchange_classes = {
        ExchangeType.ALPACA: AlpacaExchange,
        ExchangeType.BINANCE: BinanceExchange,
        ExchangeType.TD_AMERITRADE: TDAmeritradeExchange,
        # Add more exchanges as needed
    }
    
    @classmethod
    def create_exchange(cls, exchange_config: ExchangeConfig) -> BaseExchange:
        """Create exchange instance"""
        exchange_class = cls._exchange_classes.get(exchange_config.exchange_type)
        
        if not exchange_class:
            raise ValueError(f"Unsupported exchange type: {exchange_config.exchange_type}")
        
        return exchange_class(exchange_config)
    
    @classmethod
    def get_supported_exchanges(cls) -> Dict[ExchangeType, str]:
        """Get supported exchanges"""
        return {
            ExchangeType.ALPACA: "Alpaca Markets",
            ExchangeType.BINANCE: "Binance",
            ExchangeType.TD_AMERITRADE: "TD Ameritrade",
        }
    
    @classmethod
    def validate_exchange_config(cls, exchange_config: ExchangeConfig) -> bool:
        """Validate exchange configuration"""
        if exchange_config.exchange_type not in cls._exchange_classes:
            return False
        
        if not exchange_config.api_key or not exchange_config.api_secret:
            return False
        
        return True
