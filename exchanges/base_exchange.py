"""
Base Exchange class for unified exchange interface
Provides common functionality for all exchange implementations
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import json

from config import Config


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class BaseExchange(ABC):
    """
    Abstract base class for exchange implementations
    Provides unified interface for all exchanges
    """
    
    def __init__(self, config: Config, exchange_name: str):
        self.config = config
        self.exchange_name = exchange_name
        self.logger = logging.getLogger(f"Exchange.{exchange_name}")
        
        # Exchange configuration
        self.exchange_config = config.get_exchange_config(exchange_name)
        if not self.exchange_config:
            raise ValueError(f"Configuration not found for exchange: {exchange_name}")
        
        # Connection state
        self.connected = False
        self.authenticated = False
        self.sandbox_mode = self.exchange_config.sandbox
        
        # Rate limiting
        self.rate_limit = self.exchange_config.rate_limit
        self.request_count = 0
        self.last_request_time = datetime.utcnow()
        
        # Supported features
        self.supported_assets = self.exchange_config.supported_assets
        self.supported_order_types = [OrderType.MARKET, OrderType.LIMIT]
        
        # Trading limits
        self.min_order_size = {}
        self.max_order_size = {}
        self.price_precision = {}
        self.quantity_precision = {}
        
        # Market data cache
        self.market_data_cache = {}
        self.ticker_cache = {}
        
        # Order tracking
        self.open_orders = {}
        self.order_history = []
        
        # Performance metrics
        self.metrics = {
            "orders_placed": 0,
            "orders_filled": 0,
            "orders_cancelled": 0,
            "api_calls": 0,
            "errors": 0,
            "uptime": datetime.utcnow()
        }
        
        self.logger.info(f"🔗 Exchange {exchange_name} initialized")
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to exchange"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from exchange"""
        pass
    
    @abstractmethod
    async def authenticate(self) -> bool:
        """Authenticate with exchange"""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> Dict:
        """Get account information"""
        pass
    
    @abstractmethod
    async def get_balance(self, asset: str = None) -> Dict:
        """Get account balance"""
        pass
    
    @abstractmethod
    async def get_ticker(self, symbol: str) -> Dict:
        """Get ticker data for symbol"""
        pass
    
    @abstractmethod
    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict:
        """Get orderbook data"""
        pass
    
    @abstractmethod
    async def get_klines(self, symbol: str, interval: str, limit: int = 500) -> List[Dict]:
        """Get candlestick data"""
        pass
    
    @abstractmethod
    async def place_order(self, symbol: str, side: OrderSide, order_type: OrderType, 
                         quantity: float, price: float = None, **kwargs) -> Dict:
        """Place order"""
        pass
    
    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> Dict:
        """Cancel order"""
        pass
    
    @abstractmethod
    async def get_order_status(self, symbol: str, order_id: str) -> Dict:
        """Get order status"""
        pass
    
    @abstractmethod
    async def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """Get open orders"""
        pass
    
    @abstractmethod
    async def get_order_history(self, symbol: str = None, limit: int = 100) -> List[Dict]:
        """Get order history"""
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Dict]:
        """Get current positions"""
        pass
    
    # Common utility methods
    
    async def is_connected(self) -> bool:
        """Check if connected to exchange"""
        return self.connected
    
    async def is_authenticated(self) -> bool:
        """Check if authenticated"""
        return self.authenticated
    
    def get_supported_assets(self) -> List[str]:
        """Get list of supported asset types"""
        return self.supported_assets
    
    def get_supported_order_types(self) -> List[OrderType]:
        """Get list of supported order types"""
        return self.supported_order_types
    
    async def validate_symbol(self, symbol: str) -> bool:
        """Validate trading symbol"""
        try:
            # Try to get ticker to validate symbol
            ticker = await self.get_ticker(symbol)
            return ticker is not None
        except Exception as e:
            self.logger.error(f"❌ Symbol validation failed for {symbol}: {e}")
            return False
    
    async def validate_order(self, symbol: str, side: OrderSide, order_type: OrderType, 
                           quantity: float, price: float = None) -> Tuple[bool, str]:
        """Validate order parameters"""
        try:
            # Check if symbol is valid
            if not await self.validate_symbol(symbol):
                return False, f"Invalid symbol: {symbol}"
            
            # Check order type support
            if order_type not in self.supported_order_types:
                return False, f"Unsupported order type: {order_type}"
            
            # Check quantity limits
            if symbol in self.min_order_size:
                if quantity < self.min_order_size[symbol]:
                    return False, f"Quantity below minimum: {quantity} < {self.min_order_size[symbol]}"
            
            if symbol in self.max_order_size:
                if quantity > self.max_order_size[symbol]:
                    return False, f"Quantity above maximum: {quantity} > {self.max_order_size[symbol]}"
            
            # Check price for limit orders
            if order_type == OrderType.LIMIT and price is None:
                return False, "Price required for limit orders"
            
            # Check price precision
            if price is not None and symbol in self.price_precision:
                precision = self.price_precision[symbol]
                if round(price, precision) != price:
                    return False, f"Price precision error: {price} (max precision: {precision})"
            
            # Check quantity precision
            if symbol in self.quantity_precision:
                precision = self.quantity_precision[symbol]
                if round(quantity, precision) != quantity:
                    return False, f"Quantity precision error: {quantity} (max precision: {precision})"
            
            return True, "Order validation passed"
            
        except Exception as e:
            self.logger.error(f"❌ Order validation error: {e}")
            return False, f"Validation error: {e}"
    
    async def calculate_order_value(self, symbol: str, quantity: float, price: float = None) -> float:
        """Calculate order value"""
        try:
            if price is None:
                # Use current market price
                ticker = await self.get_ticker(symbol)
                price = ticker.get('price', 0)
            
            return quantity * price
            
        except Exception as e:
            self.logger.error(f"❌ Order value calculation error: {e}")
            return 0.0
    
    async def get_trading_fees(self, symbol: str, order_type: OrderType) -> Dict:
        """Get trading fees for symbol"""
        # Default fee structure - override in specific exchanges
        return {
            "maker_fee": 0.001,  # 0.1%
            "taker_fee": 0.001,  # 0.1%
            "fee_currency": "USD"
        }
    
    async def check_rate_limit(self) -> bool:
        """Check if request is within rate limits"""
        current_time = datetime.utcnow()
        time_diff = (current_time - self.last_request_time).total_seconds()
        
        if time_diff < 60:  # Within 1 minute
            if self.request_count >= self.rate_limit:
                return False
        else:
            # Reset counter after 1 minute
            self.request_count = 0
            self.last_request_time = current_time
        
        return True
    
    async def increment_request_count(self):
        """Increment API request counter"""
        self.request_count += 1
        self.metrics["api_calls"] += 1
    
    def standardize_symbol(self, symbol: str) -> str:
        """Standardize symbol format for the exchange"""
        # Default implementation - override in specific exchanges
        return symbol.upper()
    
    def parse_order_response(self, response: Dict) -> Dict:
        """Parse order response into standard format"""
        # Default implementation - override in specific exchanges
        return {
            "order_id": response.get("id", ""),
            "symbol": response.get("symbol", ""),
            "side": response.get("side", ""),
            "type": response.get("type", ""),
            "quantity": response.get("quantity", 0.0),
            "price": response.get("price", 0.0),
            "status": response.get("status", ""),
            "filled_quantity": response.get("filled_quantity", 0.0),
            "filled_price": response.get("filled_price", 0.0),
            "timestamp": response.get("timestamp", datetime.utcnow().isoformat()),
            "fees": response.get("fees", {}),
            "raw_response": response
        }
    
    def parse_ticker_response(self, response: Dict) -> Dict:
        """Parse ticker response into standard format"""
        return {
            "symbol": response.get("symbol", ""),
            "price": response.get("price", 0.0),
            "bid": response.get("bid", 0.0),
            "ask": response.get("ask", 0.0),
            "volume": response.get("volume", 0.0),
            "high": response.get("high", 0.0),
            "low": response.get("low", 0.0),
            "change": response.get("change", 0.0),
            "change_percent": response.get("change_percent", 0.0),
            "timestamp": response.get("timestamp", datetime.utcnow().isoformat()),
            "raw_response": response
        }
    
    def parse_balance_response(self, response: Dict) -> Dict:
        """Parse balance response into standard format"""
        return {
            "asset": response.get("asset", ""),
            "available": response.get("available", 0.0),
            "locked": response.get("locked", 0.0),
            "total": response.get("total", 0.0),
            "raw_response": response
        }
    
    async def get_health_status(self) -> Dict:
        """Get exchange health status"""
        return {
            "exchange": self.exchange_name,
            "connected": self.connected,
            "authenticated": self.authenticated,
            "sandbox_mode": self.sandbox_mode,
            "supported_assets": self.supported_assets,
            "metrics": self.metrics,
            "uptime_seconds": (datetime.utcnow() - self.metrics["uptime"]).total_seconds()
        }
    
    def get_metrics(self) -> Dict:
        """Get exchange metrics"""
        return {
            **self.metrics,
            "open_orders_count": len(self.open_orders),
            "request_count": self.request_count,
            "connected": self.connected,
            "authenticated": self.authenticated
        }
    
    async def start_heartbeat(self):
        """Start heartbeat to keep connection alive"""
        while self.connected:
            try:
                # Send heartbeat or ping
                await self.ping()
                await asyncio.sleep(30)  # 30 second heartbeat
            except Exception as e:
                self.logger.error(f"❌ Heartbeat error: {e}")
                await asyncio.sleep(5)
    
    async def ping(self) -> bool:
        """Ping exchange to keep connection alive"""
        try:
            # Default implementation - override in specific exchanges
            return True
        except Exception as e:
            self.logger.error(f"❌ Ping error: {e}")
            return False
    
    def __str__(self):
        return f"<Exchange {self.exchange_name}>"
    
    def __repr__(self):
        return f"<Exchange {self.exchange_name} connected={self.connected}>"
