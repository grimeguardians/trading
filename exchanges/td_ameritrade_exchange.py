"""
TD Ameritrade Exchange implementation
Stocks, options, and futures trading
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import aiohttp
import json
from urllib.parse import urlencode

from exchanges.base_exchange import BaseExchange, OrderType, OrderSide, OrderStatus
from config import Config


class TDAmeritradeExchange(BaseExchange):
    """
    TD Ameritrade exchange implementation for stocks, options, and futures
    """
    
    def __init__(self, config: Config, exchange_name: str = "td_ameritrade"):
        super().__init__(config, exchange_name)
        
        # TD Ameritrade-specific configuration
        self.api_key = self.exchange_config.api_key
        self.api_secret = self.exchange_config.api_secret
        self.access_token = None
        self.refresh_token = None
        
        # API endpoints
        self.base_url = "https://api.tdameritrade.com/v1"
        self.auth_url = "https://auth.tdameritrade.com"
        
        # Session for HTTP requests
        self.session = None
        
        # TD Ameritrade-specific settings
        self.supported_order_types = [OrderType.MARKET, OrderType.LIMIT, OrderType.STOP, OrderType.STOP_LIMIT]
        self.supported_timeframes = {
            "1min": "1",
            "5min": "5",
            "15min": "15",
            "30min": "30",
            "1h": "60",
            "1d": "daily"
        }
        
        # Rate limits
        self.rate_limit = 120  # requests per minute
        
        self.logger.info(f"🔗 TD Ameritrade Exchange initialized")
    
    async def connect(self) -> bool:
        """Connect to TD Ameritrade API"""
        try:
            if not self.api_key:
                self.logger.error("❌ TD Ameritrade API key not configured")
                return False
            
            # Initialize HTTP session
            self.session = aiohttp.ClientSession()
            
            # For demo purposes, assume we have a valid access token
            # In production, this would implement OAuth flow
            self.access_token = "demo_access_token"
            
            self.connected = True
            self.logger.info("✅ Connected to TD Ameritrade")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ TD Ameritrade connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from TD Ameritrade"""
        try:
            if self.session:
                await self.session.close()
            self.connected = False
            self.authenticated = False
            self.session = None
            self.logger.info("🔌 Disconnected from TD Ameritrade")
        except Exception as e:
            self.logger.error(f"❌ TD Ameritrade disconnection error: {e}")
    
    async def authenticate(self) -> bool:
        """Authenticate with TD Ameritrade"""
        try:
            if not self.session:
                return False
            
            # In production, this would validate the access token
            # For demo, assume authentication is successful
            self.authenticated = True
            self.logger.info("✅ TD Ameritrade authenticated")
            return True
                
        except Exception as e:
            self.logger.error(f"❌ TD Ameritrade authentication failed: {e}")
            return False
    
    async def _make_request(self, method: str, endpoint: str, params: Dict = None, data: Dict = None) -> Dict:
        """Make HTTP request to TD Ameritrade API"""
        try:
            if not await self.check_rate_limit():
                raise Exception("Rate limit exceeded")
            
            url = f"{self.base_url}/{endpoint}"
            
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            
            # Add API key to params
            if params is None:
                params = {}
            params["apikey"] = self.api_key
            
            async with self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                headers=headers
            ) as response:
                await self.increment_request_count()
                
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    self.logger.error(f"❌ TD Ameritrade API error: {response.status} - {error_text}")
                    return {}
                    
        except Exception as e:
            self.logger.error(f"❌ TD Ameritrade request error: {e}")
            self.metrics["errors"] += 1
            return {}
    
    async def get_account_info(self) -> Dict:
        """Get account information"""
        try:
            # For demo purposes, return mock data
            # In production, this would call the accounts endpoint
            return {
                "account_id": "demo_account",
                "status": "ACTIVE",
                "currency": "USD",
                "buying_power": 50000.0,
                "cash": 25000.0,
                "portfolio_value": 75000.0,
                "equity": 75000.0,
                "day_trade_count": 0,
                "pattern_day_trader": False
            }
            
        except Exception as e:
            self.logger.error(f"❌ Account info error: {e}")
            self.metrics["errors"] += 1
            return {}
    
    async def get_balance(self, asset: str = None) -> Dict:
        """Get account balance"""
        try:
            # For demo purposes, return mock data
            if asset:
                return self.parse_balance_response({
                    "asset": asset.upper(),
                    "available": 100.0,
                    "locked": 0.0,
                    "total": 100.0
                })
            else:
                return self.parse_balance_response({
                    "asset": "USD",
                    "available": 25000.0,
                    "locked": 0.0,
                    "total": 25000.0
                })
                
        except Exception as e:
            self.logger.error(f"❌ Balance error: {e}")
            self.metrics["errors"] += 1
            return {}
    
    async def get_ticker(self, symbol: str) -> Dict:
        """Get ticker data for symbol"""
        try:
            # For demo purposes, return mock data
            # In production, this would call the quotes endpoint
            return self.parse_ticker_response({
                "symbol": symbol,
                "price": 150.0,
                "bid": 149.95,
                "ask": 150.05,
                "volume": 1000000.0,
                "high": 155.0,
                "low": 145.0,
                "change": 2.5,
                "change_percent": 1.69,
                "timestamp": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"❌ Ticker error for {symbol}: {e}")
            self.metrics["errors"] += 1
            return {}
    
    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict:
        """Get orderbook data"""
        try:
            # TD Ameritrade doesn't provide full orderbook
            # Return Level 1 quote data
            return {
                "symbol": symbol,
                "bids": [[149.95, 1000]],
                "asks": [[150.05, 1000]],
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Orderbook error for {symbol}: {e}")
            self.metrics["errors"] += 1
            return {}
    
    async def get_klines(self, symbol: str, interval: str, limit: int = 500) -> List[Dict]:
        """Get candlestick data"""
        try:
            # For demo purposes, return mock data
            # In production, this would call the price history endpoint
            klines = []
            base_price = 150.0
            
            for i in range(min(limit, 100)):
                timestamp = datetime.utcnow() - timedelta(minutes=i)
                price_variation = (i % 10 - 5) * 0.5
                
                klines.append({
                    "timestamp": timestamp.isoformat(),
                    "open": base_price + price_variation,
                    "high": base_price + price_variation + 1.0,
                    "low": base_price + price_variation - 1.0,
                    "close": base_price + price_variation + 0.5,
                    "volume": 10000.0 + (i * 100)
                })
            
            return klines
            
        except Exception as e:
            self.logger.error(f"❌ Klines error for {symbol}: {e}")
            self.metrics["errors"] += 1
            return []
    
    async def place_order(self, symbol: str, side: OrderSide, order_type: OrderType, 
                         quantity: float, price: float = None, **kwargs) -> Dict:
        """Place order"""
        try:
            # Validate order
            is_valid, error_msg = await self.validate_order(symbol, side, order_type, quantity, price)
            if not is_valid:
                self.logger.error(f"❌ Order validation failed: {error_msg}")
                return {}
            
            # For demo purposes, return mock order
            # In production, this would call the orders endpoint
            order_id = f"td_order_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            order_data = {
                "id": order_id,
                "symbol": symbol,
                "side": side.value,
                "type": order_type.value,
                "quantity": quantity,
                "price": price,
                "status": "FILLED",
                "filled_quantity": quantity,
                "filled_price": price or 150.0,
                "timestamp": datetime.utcnow().isoformat(),
                "fees": {"fee": 0.0, "currency": "USD"}
            }
            
            # Store order
            self.open_orders[order_id] = order_data
            self.metrics["orders_placed"] += 1
            
            return self.parse_order_response(order_data)
                
        except Exception as e:
            self.logger.error(f"❌ Order placement error: {e}")
            self.metrics["errors"] += 1
            return {}
    
    async def cancel_order(self, symbol: str, order_id: str) -> Dict:
        """Cancel order"""
        try:
            # For demo purposes, return success
            # In production, this would call the cancel order endpoint
            
            # Remove from open orders
            if order_id in self.open_orders:
                del self.open_orders[order_id]
            
            self.metrics["orders_cancelled"] += 1
            
            return {
                "order_id": order_id,
                "symbol": symbol,
                "status": "cancelled",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Order cancellation error: {e}")
            self.metrics["errors"] += 1
            return {}
    
    async def get_order_status(self, symbol: str, order_id: str) -> Dict:
        """Get order status"""
        try:
            # For demo purposes, return mock data
            # In production, this would call the order status endpoint
            
            if order_id in self.open_orders:
                order = self.open_orders[order_id]
                return self.parse_order_response(order)
            else:
                return {}
                
        except Exception as e:
            self.logger.error(f"❌ Order status error: {e}")
            self.metrics["errors"] += 1
            return {}
    
    async def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """Get open orders"""
        try:
            # For demo purposes, return mock data
            # In production, this would call the orders endpoint
            
            open_orders = []
            for order_id, order_data in self.open_orders.items():
                if symbol is None or order_data.get("symbol") == symbol:
                    if order_data.get("status") in ["OPEN", "PARTIALLY_FILLED"]:
                        open_orders.append(self.parse_order_response(order_data))
            
            return open_orders
            
        except Exception as e:
            self.logger.error(f"❌ Open orders error: {e}")
            self.metrics["errors"] += 1
            return []
    
    async def get_order_history(self, symbol: str = None, limit: int = 100) -> List[Dict]:
        """Get order history"""
        try:
            # For demo purposes, return mock data
            # In production, this would call the orders endpoint with appropriate filters
            
            order_history = []
            for order_id, order_data in list(self.open_orders.items())[:limit]:
                if symbol is None or order_data.get("symbol") == symbol:
                    order_history.append(self.parse_order_response(order_data))
            
            return order_history
            
        except Exception as e:
            self.logger.error(f"❌ Order history error: {e}")
            self.metrics["errors"] += 1
            return []
    
    async def get_positions(self) -> List[Dict]:
        """Get current positions"""
        try:
            # For demo purposes, return mock data
            # In production, this would call the positions endpoint
            
            positions = [
                {
                    "symbol": "AAPL",
                    "quantity": 100.0,
                    "side": "long",
                    "entry_price": 145.0,
                    "current_price": 150.0,
                    "market_value": 15000.0,
                    "cost_basis": 14500.0,
                    "unrealized_pnl": 500.0,
                    "unrealized_pnl_pct": 3.45,
                    "timestamp": datetime.utcnow().isoformat()
                }
            ]
            
            return positions
            
        except Exception as e:
            self.logger.error(f"❌ Positions error: {e}")
            self.metrics["errors"] += 1
            return []
    
    async def get_trading_fees(self, symbol: str, order_type: OrderType) -> Dict:
        """Get trading fees for symbol"""
        try:
            # TD Ameritrade fee structure
            return {
                "maker_fee": 0.0,    # Commission-free for stocks
                "taker_fee": 0.0,    # Commission-free for stocks
                "fee_currency": "USD",
                "options_fee": 0.65  # Per contract for options
            }
            
        except Exception as e:
            self.logger.error(f"❌ Trading fees error: {e}")
            return {"maker_fee": 0.0, "taker_fee": 0.0, "fee_currency": "USD"}
    
    def standardize_symbol(self, symbol: str) -> str:
        """Standardize symbol format for TD Ameritrade"""
        # TD Ameritrade uses simple uppercase format
        return symbol.upper()
    
    async def ping(self) -> bool:
        """Ping TD Ameritrade to keep connection alive"""
        try:
            # For demo purposes, return True
            # In production, this would make a lightweight API call
            return True
        except Exception as e:
            self.logger.error(f"❌ TD Ameritrade ping error: {e}")
            return False
    
    def parse_order_response(self, response: Dict) -> Dict:
        """Parse order response into standard format"""
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
            "exchange": "td_ameritrade",
            "raw_response": response
        }
