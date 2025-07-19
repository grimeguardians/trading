"""
Binance exchange implementation
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import aiohttp
import hashlib
import hmac
import time
import json

from .base_exchange import (
    BaseExchange, OrderRequest, OrderResponse, Position, AccountInfo,
    MarketDataPoint, OrderType, OrderSide, OrderStatus
)
from config import ExchangeConfig
from utils.logger import setup_logger

logger = setup_logger("binance_exchange")

class BinanceExchange(BaseExchange):
    """Binance exchange implementation"""
    
    def __init__(self, config: ExchangeConfig):
        super().__init__(config.__dict__)
        self.session = None
        self.recv_window = 5000
        
    async def connect(self) -> bool:
        """Connect to Binance"""
        try:
            self.session = aiohttp.ClientSession()
            
            # Test connection
            account_info = await self.get_account_info()
            if account_info:
                self.connected = True
                logger.info("Connected to Binance")
                return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Binance: {e}")
            self.connected = False
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Binance"""
        if self.session:
            await self.session.close()
            self.session = None
        self.connected = False
        logger.info("Disconnected from Binance")
        return True
    
    def _generate_signature(self, query_string: str) -> str:
        """Generate signature for Binance API"""
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _get_timestamp(self) -> int:
        """Get current timestamp"""
        return int(time.time() * 1000)
    
    async def _make_request(self, method: str, endpoint: str, params: Dict[str, Any] = None, signed: bool = False) -> Dict[str, Any]:
        """Make API request to Binance"""
        url = f"{self.base_url}{endpoint}"
        
        if params is None:
            params = {}
        
        if signed:
            params['timestamp'] = self._get_timestamp()
            params['recvWindow'] = self.recv_window
            
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            signature = self._generate_signature(query_string)
            params['signature'] = signature
            
            headers = {
                'X-MBX-APIKEY': self.api_key,
                'Content-Type': 'application/json'
            }
        else:
            headers = {'Content-Type': 'application/json'}
        
        try:
            if method == 'GET':
                async with self.session.get(url, params=params, headers=headers) as response:
                    return await response.json()
            elif method == 'POST':
                async with self.session.post(url, params=params, headers=headers) as response:
                    return await response.json()
            elif method == 'DELETE':
                async with self.session.delete(url, params=params, headers=headers) as response:
                    return await response.json()
        except Exception as e:
            logger.error(f"Error making request to {url}: {e}")
            raise
    
    async def get_account_info(self) -> AccountInfo:
        """Get account information"""
        try:
            data = await self._make_request('GET', '/api/v3/account', signed=True)
            
            # Calculate total balance
            total_balance = 0.0
            cash = 0.0
            
            for balance in data['balances']:
                if balance['asset'] == 'USDT':
                    cash = float(balance['free'])
                total_balance += float(balance['free']) + float(balance['locked'])
            
            return AccountInfo(
                account_id=str(data.get('accountType', 'SPOT')),
                cash=cash,
                portfolio_value=total_balance,
                buying_power=cash,
                day_trading_buying_power=cash
            )
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            raise
    
    async def get_positions(self) -> List[Position]:
        """Get all positions"""
        try:
            data = await self._make_request('GET', '/api/v3/account', signed=True)
            positions = []
            
            for balance in data['balances']:
                free = float(balance['free'])
                locked = float(balance['locked'])
                total = free + locked
                
                if total > 0:
                    # For spot trading, we consider any non-zero balance as a position
                    # Get current price for the asset
                    try:
                        if balance['asset'] != 'USDT':
                            ticker = await self.get_ticker(f"{balance['asset']}USDT")
                            current_price = ticker['price']
                        else:
                            current_price = 1.0
                        
                        positions.append(Position(
                            symbol=balance['asset'],
                            quantity=total,
                            average_price=current_price,  # Binance doesn't provide avg cost for spot
                            current_price=current_price,
                            unrealized_pnl=0.0,  # Not available for spot
                            realized_pnl=0.0
                        ))
                    except:
                        # Skip if unable to get price
                        pass
            
            return positions
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            raise
    
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol"""
        positions = await self.get_positions()
        for pos in positions:
            if pos.symbol == symbol:
                return pos
        return None
    
    async def place_order(self, order_request: OrderRequest) -> OrderResponse:
        """Place an order"""
        try:
            params = {
                'symbol': order_request.symbol,
                'side': order_request.side.value.upper(),
                'type': self._convert_order_type(order_request.order_type),
                'quantity': order_request.quantity,
                'timeInForce': order_request.time_in_force
            }
            
            if order_request.price is not None:
                params['price'] = order_request.price
            
            if order_request.stop_price is not None:
                params['stopPrice'] = order_request.stop_price
            
            data = await self._make_request('POST', '/api/v3/order', params=params, signed=True)
            
            return OrderResponse(
                order_id=str(data['orderId']),
                exchange_order_id=str(data['orderId']),
                symbol=data['symbol'],
                side=OrderSide(data['side'].lower()),
                order_type=self._convert_from_binance_order_type(data['type']),
                quantity=float(data['origQty']),
                price=float(data['price']) if data.get('price') else None,
                status=self._convert_order_status(data['status']),
                filled_quantity=float(data['executedQty']),
                filled_price=float(data['cummulativeQuoteQty']) / float(data['executedQty']) if float(data['executedQty']) > 0 else None,
                created_at=datetime.fromtimestamp(data['transactTime'] / 1000),
                updated_at=datetime.fromtimestamp(data['transactTime'] / 1000)
            )
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            # Need symbol to cancel order in Binance
            # This is a limitation - we'd need to store symbol with order_id
            logger.warning("Cancel order requires symbol in Binance - implementation incomplete")
            return False
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def get_order(self, order_id: str) -> Optional[OrderResponse]:
        """Get order status"""
        try:
            # Need symbol to get order in Binance
            # This is a limitation - we'd need to store symbol with order_id
            logger.warning("Get order requires symbol in Binance - implementation incomplete")
            return None
        except Exception as e:
            logger.error(f"Error getting order {order_id}: {e}")
            return None
    
    async def get_orders(self, symbol: Optional[str] = None, status: Optional[OrderStatus] = None) -> List[OrderResponse]:
        """Get orders"""
        try:
            if not symbol:
                # Binance requires symbol for order queries
                return []
            
            params = {'symbol': symbol}
            data = await self._make_request('GET', '/api/v3/allOrders', params=params, signed=True)
            
            orders = []
            for order in data:
                order_status = self._convert_order_status(order['status'])
                
                if status is None or order_status == status:
                    orders.append(OrderResponse(
                        order_id=str(order['orderId']),
                        exchange_order_id=str(order['orderId']),
                        symbol=order['symbol'],
                        side=OrderSide(order['side'].lower()),
                        order_type=self._convert_from_binance_order_type(order['type']),
                        quantity=float(order['origQty']),
                        price=float(order['price']) if order.get('price') else None,
                        status=order_status,
                        filled_quantity=float(order['executedQty']),
                        filled_price=float(order['cummulativeQuoteQty']) / float(order['executedQty']) if float(order['executedQty']) > 0 else None,
                        created_at=datetime.fromtimestamp(order['time'] / 1000),
                        updated_at=datetime.fromtimestamp(order['updateTime'] / 1000)
                    ))
            
            return orders
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            raise
    
    async def get_market_data(self, symbol: str, timeframe: str = "1m", limit: int = 100) -> List[MarketDataPoint]:
        """Get market data"""
        try:
            params = {
                'symbol': symbol,
                'interval': self._convert_timeframe(timeframe),
                'limit': limit
            }
            
            data = await self._make_request('GET', '/api/v3/klines', params=params)
            
            bars = []
            for kline in data:
                bars.append(MarketDataPoint(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(kline[0] / 1000),
                    open=float(kline[1]),
                    high=float(kline[2]),
                    low=float(kline[3]),
                    close=float(kline[4]),
                    volume=float(kline[5])
                ))
            
            return bars
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            raise
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get ticker data"""
        try:
            data = await self._make_request('GET', '/api/v3/ticker/price', params={'symbol': symbol})
            
            return {
                'symbol': symbol,
                'price': float(data['price']),
                'timestamp': datetime.utcnow()
            }
        except Exception as e:
            logger.error(f"Error getting ticker for {symbol}: {e}")
            raise
    
    async def get_supported_symbols(self) -> List[str]:
        """Get supported symbols"""
        try:
            data = await self._make_request('GET', '/api/v3/exchangeInfo')
            
            symbols = []
            for symbol_info in data['symbols']:
                if symbol_info['status'] == 'TRADING':
                    symbols.append(symbol_info['symbol'])
            
            return symbols
        except Exception as e:
            logger.error(f"Error getting supported symbols: {e}")
            raise
    
    async def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol is supported"""
        try:
            data = await self._make_request('GET', '/api/v3/exchangeInfo')
            
            for symbol_info in data['symbols']:
                if symbol_info['symbol'] == symbol and symbol_info['status'] == 'TRADING':
                    return True
            
            return False
        except Exception as e:
            logger.error(f"Error validating symbol {symbol}: {e}")
            return False
    
    async def get_fees(self) -> Dict[str, Any]:
        """Get trading fees"""
        try:
            data = await self._make_request('GET', '/api/v3/account', signed=True)
            
            return {
                'maker_fee': float(data['makerCommission']) / 10000,  # Convert from basis points
                'taker_fee': float(data['takerCommission']) / 10000,
                'buyer_fee': float(data['buyerCommission']) / 10000,
                'seller_fee': float(data['sellerCommission']) / 10000
            }
        except Exception as e:
            logger.error(f"Error getting fees: {e}")
            raise
    
    def _convert_order_type(self, order_type: OrderType) -> str:
        """Convert order type to Binance format"""
        mapping = {
            OrderType.MARKET: 'MARKET',
            OrderType.LIMIT: 'LIMIT',
            OrderType.STOP_LOSS: 'STOP_LOSS',
            OrderType.STOP_LIMIT: 'STOP_LOSS_LIMIT'
        }
        return mapping.get(order_type, 'MARKET')
    
    def _convert_from_binance_order_type(self, binance_type: str) -> OrderType:
        """Convert from Binance order type"""
        mapping = {
            'MARKET': OrderType.MARKET,
            'LIMIT': OrderType.LIMIT,
            'STOP_LOSS': OrderType.STOP_LOSS,
            'STOP_LOSS_LIMIT': OrderType.STOP_LIMIT
        }
        return mapping.get(binance_type, OrderType.MARKET)
    
    def _convert_order_status(self, binance_status: str) -> OrderStatus:
        """Convert Binance order status"""
        mapping = {
            'NEW': OrderStatus.PENDING,
            'PARTIALLY_FILLED': OrderStatus.PARTIALLY_FILLED,
            'FILLED': OrderStatus.FILLED,
            'CANCELED': OrderStatus.CANCELLED,
            'REJECTED': OrderStatus.REJECTED,
            'EXPIRED': OrderStatus.CANCELLED
        }
        return mapping.get(binance_status, OrderStatus.PENDING)
    
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert timeframe to Binance format"""
        mapping = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '1d': '1d'
        }
        return mapping.get(timeframe, '1m')
