"""
Base exchange interface for multi-exchange support
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"

class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """Order statuses"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class MarketDataPoint:
    """Market data point"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
@dataclass
class OrderRequest:
    """Order request"""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"
    
@dataclass
class OrderResponse:
    """Order response"""
    order_id: str
    exchange_order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float]
    status: OrderStatus
    filled_quantity: float = 0
    filled_price: Optional[float] = None
    created_at: datetime = None
    updated_at: datetime = None
    
@dataclass
class Position:
    """Position information"""
    symbol: str
    quantity: float
    average_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    
@dataclass
class AccountInfo:
    """Account information"""
    account_id: str
    cash: float
    portfolio_value: float
    buying_power: float
    day_trading_buying_power: float
    
class BaseExchange(ABC):
    """Base exchange interface"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get("api_key")
        self.api_secret = config.get("api_secret")
        self.base_url = config.get("base_url")
        self.sandbox = config.get("sandbox", True)
        self.rate_limit = config.get("rate_limit", 200)
        self.connected = False
        
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to exchange"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from exchange"""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> AccountInfo:
        """Get account information"""
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get all positions"""
        pass
    
    @abstractmethod
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol"""
        pass
    
    @abstractmethod
    async def place_order(self, order_request: OrderRequest) -> OrderResponse:
        """Place an order"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        pass
    
    @abstractmethod
    async def get_order(self, order_id: str) -> Optional[OrderResponse]:
        """Get order status"""
        pass
    
    @abstractmethod
    async def get_orders(self, symbol: Optional[str] = None, status: Optional[OrderStatus] = None) -> List[OrderResponse]:
        """Get orders"""
        pass
    
    @abstractmethod
    async def get_market_data(self, symbol: str, timeframe: str = "1m", limit: int = 100) -> List[MarketDataPoint]:
        """Get market data"""
        pass
    
    @abstractmethod
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get ticker data"""
        pass
    
    @abstractmethod
    async def get_supported_symbols(self) -> List[str]:
        """Get supported symbols"""
        pass
    
    @abstractmethod
    async def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol is supported"""
        pass
    
    @abstractmethod
    async def get_fees(self) -> Dict[str, Any]:
        """Get trading fees"""
        pass
    
    def normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol format for this exchange"""
        return symbol.upper()
    
    def denormalize_symbol(self, symbol: str) -> str:
        """Convert from exchange format to standard format"""
        return symbol.upper()
    
    async def health_check(self) -> bool:
        """Check if exchange is healthy"""
        try:
            await self.get_account_info()
            return True
        except Exception:
            return False
    
    def get_exchange_info(self) -> Dict[str, Any]:
        """Get exchange information"""
        return {
            "name": self.__class__.__name__,
            "base_url": self.base_url,
            "sandbox": self.sandbox,
            "rate_limit": self.rate_limit,
            "connected": self.connected
        }
