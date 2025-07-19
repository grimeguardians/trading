"""
Alpaca Markets exchange implementation
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import aiohttp
import json

from .base_exchange import (
    BaseExchange, OrderRequest, OrderResponse, Position, AccountInfo,
    MarketDataPoint, OrderType, OrderSide, OrderStatus
)
from config import ExchangeConfig
from utils.logger import setup_logger

logger = setup_logger("alpaca_exchange")

class AlpacaExchange(BaseExchange):
    """Alpaca Markets exchange implementation"""
    
    def __init__(self, config: ExchangeConfig):
        super().__init__(config.__dict__)
        self.session = None
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json"
        }
        
        # Alpaca-specific URLs
        self.data_url = "https://data.alpaca.markets" if not self.sandbox else "https://paper-api.alpaca.markets"
        
    async def connect(self) -> bool:
        """Connect to Alpaca"""
        try:
            self.session = aiohttp.ClientSession(headers=self.headers)
            
            # Test connection
            account_info = await self.get_account_info()
            if account_info:
                self.connected = True
                logger.info("Connected to Alpaca Markets")
                return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            self.connected = False
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Alpaca"""
        if self.session:
            await self.session.close()
            self.session = None
        self.connected = False
        logger.info("Disconnected from Alpaca Markets")
        return True
    
    async def get_account_info(self) -> AccountInfo:
        """Get account information"""
        try:
            async with self.session.get(f"{self.base_url}/v2/account") as response:
                if response.status == 200:
                    data = await response.json()
                    return AccountInfo(
                        account_id=data["id"],
                        cash=float(data["cash"]),
                        portfolio_value=float(data["portfolio_value"]),
                        buying_power=float(data["buying_power"]),
                        day_trading_buying_power=float(data["day_trade_buying_power"])
                    )
                else:
                    raise Exception(f"Failed to get account info: {response.status}")
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            raise
    
    async def get_positions(self) -> List[Position]:
        """Get all positions"""
        try:
            async with self.session.get(f"{self.base_url}/v2/positions") as response:
                if response.status == 200:
                    data = await response.json()
                    positions = []
                    for pos in data:
                        positions.append(Position(
                            symbol=pos["symbol"],
                            quantity=float(pos["qty"]),
                            average_price=float(pos["avg_cost"]),
                            current_price=float(pos["current_price"]) if pos["current_price"] else 0.0,
                            unrealized_pnl=float(pos["unrealized_pnl"]) if pos["unrealized_pnl"] else 0.0,
                            realized_pnl=float(pos["realized_pnl"]) if pos["realized_pnl"] else 0.0
                        ))
                    return positions
                else:
                    raise Exception(f"Failed to get positions: {response.status}")
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            raise
    
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol"""
        try:
            async with self.session.get(f"{self.base_url}/v2/positions/{symbol}") as response:
                if response.status == 200:
                    pos = await response.json()
                    return Position(
                        symbol=pos["symbol"],
                        quantity=float(pos["qty"]),
                        average_price=float(pos["avg_cost"]),
                        current_price=float(pos["current_price"]) if pos["current_price"] else 0.0,
                        unrealized_pnl=float(pos["unrealized_pnl"]) if pos["unrealized_pnl"] else 0.0,
                        realized_pnl=float(pos["realized_pnl"]) if pos["realized_pnl"] else 0.0
                    )
                elif response.status == 404:
                    return None
                else:
                    raise Exception(f"Failed to get position: {response.status}")
        except Exception as e:
            logger.error(f"Error getting position for {symbol}: {e}")
            raise
    
    async def place_order(self, order_request: OrderRequest) -> OrderResponse:
        """Place an order"""
        try:
            # Convert order request to Alpaca format
            order_data = {
                "symbol": order_request.symbol,
                "qty": order_request.quantity,
                "side": order_request.side.value,
                "type": self._convert_order_type(order_request.order_type),
                "time_in_force": order_request.time_in_force
            }
            
            if order_request.price is not None:
                order_data["limit_price"] = order_request.price
            
            if order_request.stop_price is not None:
                order_data["stop_price"] = order_request.stop_price
            
            async with self.session.post(
                f"{self.base_url}/v2/orders",
                json=order_data
            ) as response:
                if response.status == 201:
                    data = await response.json()
                    return OrderResponse(
                        order_id=data["id"],
                        exchange_order_id=data["id"],
                        symbol=data["symbol"],
                        side=OrderSide(data["side"]),
                        order_type=self._convert_from_alpaca_order_type(data["type"]),
                        quantity=float(data["qty"]),
                        price=float(data["limit_price"]) if data.get("limit_price") else None,
                        status=self._convert_order_status(data["status"]),
                        filled_quantity=float(data["filled_qty"]),
                        filled_price=float(data["filled_avg_price"]) if data.get("filled_avg_price") else None,
                        created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")),
                        updated_at=datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00"))
                    )
                else:
                    error_data = await response.json()
                    raise Exception(f"Failed to place order: {error_data}")
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            async with self.session.delete(f"{self.base_url}/v2/orders/{order_id}") as response:
                return response.status == 204
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def get_order(self, order_id: str) -> Optional[OrderResponse]:
        """Get order status"""
        try:
            async with self.session.get(f"{self.base_url}/v2/orders/{order_id}") as response:
                if response.status == 200:
                    data = await response.json()
                    return OrderResponse(
                        order_id=data["id"],
                        exchange_order_id=data["id"],
                        symbol=data["symbol"],
                        side=OrderSide(data["side"]),
                        order_type=self._convert_from_alpaca_order_type(data["type"]),
                        quantity=float(data["qty"]),
                        price=float(data["limit_price"]) if data.get("limit_price") else None,
                        status=self._convert_order_status(data["status"]),
                        filled_quantity=float(data["filled_qty"]),
                        filled_price=float(data["filled_avg_price"]) if data.get("filled_avg_price") else None,
                        created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")),
                        updated_at=datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00"))
                    )
                elif response.status == 404:
                    return None
                else:
                    raise Exception(f"Failed to get order: {response.status}")
        except Exception as e:
            logger.error(f"Error getting order {order_id}: {e}")
            raise
    
    async def get_orders(self, symbol: Optional[str] = None, status: Optional[OrderStatus] = None) -> List[OrderResponse]:
        """Get orders"""
        try:
            params = {}
            if status:
                params["status"] = self._convert_to_alpaca_status(status)
            
            url = f"{self.base_url}/v2/orders"
            if params:
                url += "?" + "&".join([f"{k}={v}" for k, v in params.items()])
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    orders = []
                    for order in data:
                        if symbol is None or order["symbol"] == symbol:
                            orders.append(OrderResponse(
                                order_id=order["id"],
                                exchange_order_id=order["id"],
                                symbol=order["symbol"],
                                side=OrderSide(order["side"]),
                                order_type=self._convert_from_alpaca_order_type(order["type"]),
                                quantity=float(order["qty"]),
                                price=float(order["limit_price"]) if order.get("limit_price") else None,
                                status=self._convert_order_status(order["status"]),
                                filled_quantity=float(order["filled_qty"]),
                                filled_price=float(order["filled_avg_price"]) if order.get("filled_avg_price") else None,
                                created_at=datetime.fromisoformat(order["created_at"].replace("Z", "+00:00")),
                                updated_at=datetime.fromisoformat(order["updated_at"].replace("Z", "+00:00"))
                            ))
                    return orders
                else:
                    raise Exception(f"Failed to get orders: {response.status}")
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            raise
    
    async def get_market_data(self, symbol: str, timeframe: str = "1m", limit: int = 100) -> List[MarketDataPoint]:
        """Get market data"""
        try:
            # Convert timeframe to Alpaca format
            alpaca_timeframe = self._convert_timeframe(timeframe)
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1)  # Get last day of data
            
            params = {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "timeframe": alpaca_timeframe,
                "limit": limit
            }
            
            url = f"{self.data_url}/v2/stocks/{symbol}/bars"
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    bars = []
                    if "bars" in data:
                        for bar in data["bars"]:
                            bars.append(MarketDataPoint(
                                symbol=symbol,
                                timestamp=datetime.fromisoformat(bar["t"].replace("Z", "+00:00")),
                                open=float(bar["o"]),
                                high=float(bar["h"]),
                                low=float(bar["l"]),
                                close=float(bar["c"]),
                                volume=float(bar["v"])
                            ))
                    return bars
                else:
                    raise Exception(f"Failed to get market data: {response.status}")
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            raise
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get ticker data"""
        try:
            async with self.session.get(f"{self.data_url}/v2/stocks/{symbol}/trades/latest") as response:
                if response.status == 200:
                    data = await response.json()
                    trade = data["trade"]
                    return {
                        "symbol": symbol,
                        "price": float(trade["p"]),
                        "size": float(trade["s"]),
                        "timestamp": datetime.fromisoformat(trade["t"].replace("Z", "+00:00"))
                    }
                else:
                    raise Exception(f"Failed to get ticker: {response.status}")
        except Exception as e:
            logger.error(f"Error getting ticker for {symbol}: {e}")
            raise
    
    async def get_supported_symbols(self) -> List[str]:
        """Get supported symbols"""
        try:
            async with self.session.get(f"{self.base_url}/v2/assets") as response:
                if response.status == 200:
                    data = await response.json()
                    return [asset["symbol"] for asset in data if asset["tradable"]]
                else:
                    raise Exception(f"Failed to get assets: {response.status}")
        except Exception as e:
            logger.error(f"Error getting supported symbols: {e}")
            raise
    
    async def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol is supported"""
        try:
            async with self.session.get(f"{self.base_url}/v2/assets/{symbol}") as response:
                if response.status == 200:
                    data = await response.json()
                    return data["tradable"]
                else:
                    return False
        except Exception as e:
            logger.error(f"Error validating symbol {symbol}: {e}")
            return False
    
    async def get_fees(self) -> Dict[str, Any]:
        """Get trading fees"""
        # Alpaca doesn't charge commission for stocks
        return {
            "stock_commission": 0.0,
            "option_commission": 0.65,  # Per contract
            "crypto_commission": 0.0025  # 0.25%
        }
    
    def _convert_order_type(self, order_type: OrderType) -> str:
        """Convert order type to Alpaca format"""
        mapping = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP_LOSS: "stop",
            OrderType.STOP_LIMIT: "stop_limit"
        }
        return mapping.get(order_type, "market")
    
    def _convert_from_alpaca_order_type(self, alpaca_type: str) -> OrderType:
        """Convert from Alpaca order type"""
        mapping = {
            "market": OrderType.MARKET,
            "limit": OrderType.LIMIT,
            "stop": OrderType.STOP_LOSS,
            "stop_limit": OrderType.STOP_LIMIT
        }
        return mapping.get(alpaca_type, OrderType.MARKET)
    
    def _convert_order_status(self, alpaca_status: str) -> OrderStatus:
        """Convert Alpaca order status"""
        mapping = {
            "new": OrderStatus.PENDING,
            "accepted": OrderStatus.PENDING,
            "pending_new": OrderStatus.PENDING,
            "filled": OrderStatus.FILLED,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
            "canceled": OrderStatus.CANCELLED,
            "cancelled": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED,
            "expired": OrderStatus.CANCELLED
        }
        return mapping.get(alpaca_status, OrderStatus.PENDING)
    
    def _convert_to_alpaca_status(self, status: OrderStatus) -> str:
        """Convert to Alpaca status"""
        mapping = {
            OrderStatus.PENDING: "open",
            OrderStatus.FILLED: "filled",
            OrderStatus.PARTIALLY_FILLED: "partially_filled",
            OrderStatus.CANCELLED: "cancelled",
            OrderStatus.REJECTED: "rejected"
        }
        return mapping.get(status, "open")
    
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert timeframe to Alpaca format"""
        mapping = {
            "1m": "1Min",
            "5m": "5Min",
            "15m": "15Min",
            "30m": "30Min",
            "1h": "1Hour",
            "1d": "1Day"
        }
        return mapping.get(timeframe, "1Min")
