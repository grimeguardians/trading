"""
Binance exchange adapter
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import aiohttp
import json
import hmac
import hashlib
import time
from urllib.parse import urlencode

from .base_exchange import BaseExchange, TradeOrder, Position, Balance, Ticker, OrderBook, OHLCV

logger = logging.getLogger(__name__)

class BinanceAdapter(BaseExchange):
    """Binance exchange adapter"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.session = None
        
        # Binance API endpoints
        self.base_url = "https://api.binance.com"
        self.endpoints = {
            'ticker': '/api/v3/ticker/24hr',
            'depth': '/api/v3/depth',
            'klines': '/api/v3/klines',
            'account': '/api/v3/account',
            'order': '/api/v3/order',
            'allOrders': '/api/v3/allOrders',
            'exchangeInfo': '/api/v3/exchangeInfo',
            'time': '/api/v3/time'
        }
        
        # Timeframe mapping
        self.timeframes = {
            '1min': '1m',
            '5min': '5m',
            '15min': '15m',
            '30min': '30m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d',
            '1w': '1w',
            '1M': '1M'
        }
        
        # Server time offset
        self.time_offset = 0
    
    async def connect(self) -> bool:
        """Connect to Binance API"""
        try:
            self.session = aiohttp.ClientSession()
            
            # Get server time to calculate offset
            server_time = await self._get_server_time()
            if server_time:
                self.time_offset = server_time - int(time.time() * 1000)
                
            # Test connection
            if self.api_key and self.secret_key:
                account_info = await self._signed_request('GET', self.endpoints['account'])
                if account_info:
                    self.logger.info(f"Connected to Binance - Account: {account_info.get('accountType', 'Unknown')}")
                    self.is_connected = True
                    return True
            else:
                # Test with public endpoint
                ticker_data = await self._public_request('GET', self.endpoints['ticker'])
                if ticker_data:
                    self.logger.info("Connected to Binance (public API only)")
                    self.is_connected = True
                    return True
                    
            return False
            
        except Exception as e:
            self.logger.error(f"Error connecting to Binance: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Binance API"""
        try:
            if self.session:
                await self.session.close()
                self.session = None
            self.is_connected = False
            self.logger.info("Disconnected from Binance")
            return True
        except Exception as e:
            self.logger.error(f"Error disconnecting from Binance: {e}")
            return False
    
    async def get_ticker(self, symbol: str) -> Optional[Ticker]:
        """Get ticker data for a symbol"""
        try:
            if not await self.check_rate_limit():
                return None
            
            # Format symbol for Binance (remove any slashes)
            binance_symbol = symbol.replace('/', '').upper()
            
            params = {'symbol': binance_symbol}
            ticker_data = await self._public_request('GET', self.endpoints['ticker'], params)
            
            if ticker_data:
                return Ticker(
                    symbol=symbol,
                    price=float(ticker_data['lastPrice']),
                    volume=float(ticker_data['volume']),
                    change_24h=float(ticker_data['priceChange']),
                    change_percentage_24h=float(ticker_data['priceChangePercent']),
                    high_24h=float(ticker_data['highPrice']),
                    low_24h=float(ticker_data['lowPrice']),
                    timestamp=datetime.now()
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting ticker for {symbol}: {e}")
            return None
    
    async def get_order_book(self, symbol: str, limit: int = 100) -> Optional[OrderBook]:
        """Get order book data for a symbol"""
        try:
            if not await self.check_rate_limit():
                return None
            
            binance_symbol = symbol.replace('/', '').upper()
            params = {'symbol': binance_symbol, 'limit': min(limit, 5000)}
            
            depth_data = await self._public_request('GET', self.endpoints['depth'], params)
            
            if depth_data:
                return OrderBook(
                    symbol=symbol,
                    bids=[(float(bid[0]), float(bid[1])) for bid in depth_data['bids']],
                    asks=[(float(ask[0]), float(ask[1])) for ask in depth_data['asks']],
                    timestamp=datetime.now()
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting order book for {symbol}: {e}")
            return None
    
    async def get_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 500) -> List[OHLCV]:
        """Get OHLCV data for a symbol"""
        try:
            if not await self.check_rate_limit():
                return []
            
            binance_symbol = symbol.replace('/', '').upper()
            binance_timeframe = self.timeframes.get(timeframe, '1h')
            
            params = {
                'symbol': binance_symbol,
                'interval': binance_timeframe,
                'limit': min(limit, 1000)
            }
            
            klines_data = await self._public_request('GET', self.endpoints['klines'], params)
            
            if klines_data:
                ohlcv_data = []
                for kline in klines_data:
                    ohlcv_data.append(OHLCV(
                        symbol=symbol,
                        timestamp=datetime.fromtimestamp(kline[0] / 1000),
                        open=float(kline[1]),
                        high=float(kline[2]),
                        low=float(kline[3]),
                        close=float(kline[4]),
                        volume=float(kline[5])
                    ))
                
                return ohlcv_data
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting OHLCV for {symbol}: {e}")
            return []
    
    async def get_balances(self) -> Dict[str, Balance]:
        """Get account balances"""
        try:
            if not await self.check_rate_limit():
                return {}
            
            if not self.api_key or not self.secret_key:
                return {}
            
            account_data = await self._signed_request('GET', self.endpoints['account'])
            
            if account_data:
                balances = {}
                for balance_data in account_data['balances']:
                    asset = balance_data['asset']
                    free = float(balance_data['free'])
                    locked = float(balance_data['locked'])
                    
                    if free > 0 or locked > 0:
                        balances[asset] = Balance(
                            asset=asset,
                            free=free,
                            locked=locked,
                            total=free + locked
                        )
                
                return balances
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting balances: {e}")
            return {}
    
    async def get_positions(self) -> Dict[str, Position]:
        """Get open positions (for futures)"""
        try:
            # Binance spot doesn't have positions in the traditional sense
            # This would be implemented for futures trading
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return {}
    
    async def place_order(self, order: TradeOrder) -> Optional[TradeOrder]:
        """Place a trade order"""
        try:
            if not await self.check_rate_limit():
                return None
            
            # If paper trading, simulate the order
            if self.paper_trading:
                return await self.simulate_order_fill(order)
            
            if not self.api_key or not self.secret_key:
                return None
            
            binance_symbol = order.symbol.replace('/', '').upper()
            
            # Prepare order parameters
            params = {
                'symbol': binance_symbol,
                'side': order.side.upper(),
                'type': order.type.upper(),
                'quantity': str(order.quantity),
                'timestamp': self._get_timestamp()
            }
            
            if order.type == 'LIMIT':
                params['price'] = str(order.price)
                params['timeInForce'] = order.time_in_force
            elif order.type == 'STOP_LOSS':
                params['stopPrice'] = str(order.stop_price)
            elif order.type == 'STOP_LOSS_LIMIT':
                params['price'] = str(order.price)
                params['stopPrice'] = str(order.stop_price)
                params['timeInForce'] = order.time_in_force
            
            response_data = await self._signed_request('POST', self.endpoints['order'], params)
            
            if response_data:
                order.order_id = str(response_data['orderId'])
                order.status = response_data['status'].lower()
                order.created_at = datetime.now()
                order.updated_at = datetime.now()
                
                if response_data.get('executedQty'):
                    order.filled_quantity = float(response_data['executedQty'])
                
                return order
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            if not await self.check_rate_limit():
                return False
            
            if not self.api_key or not self.secret_key:
                return False
            
            # Note: This requires symbol, which we don't have in this method
            # In a real implementation, you'd need to track order_id -> symbol mapping
            return False
            
        except Exception as e:
            self.logger.error(f"Error canceling order {order_id}: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> Optional[TradeOrder]:
        """Get order status"""
        try:
            if not await self.check_rate_limit():
                return None
            
            if not self.api_key or not self.secret_key:
                return None
            
            # Note: This requires symbol, which we don't have in this method
            # In a real implementation, you'd need to track order_id -> symbol mapping
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting order status for {order_id}: {e}")
            return None
    
    async def get_trade_history(self, symbol: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get trade history"""
        try:
            if not await self.check_rate_limit():
                return []
            
            if not self.api_key or not self.secret_key:
                return []
            
            if not symbol:
                return []
            
            binance_symbol = symbol.replace('/', '').upper()
            params = {
                'symbol': binance_symbol,
                'limit': min(limit, 1000),
                'timestamp': self._get_timestamp()
            }
            
            orders_data = await self._signed_request('GET', self.endpoints['allOrders'], params)
            
            if orders_data:
                trades = []
                for order_data in orders_data:
                    if order_data['status'] == 'FILLED':
                        trades.append({
                            'order_id': str(order_data['orderId']),
                            'symbol': symbol,
                            'side': order_data['side'].lower(),
                            'quantity': float(order_data['executedQty']),
                            'price': float(order_data['price']),
                            'timestamp': datetime.fromtimestamp(order_data['time'] / 1000)
                        })
                
                return trades
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting trade history: {e}")
            return []
    
    # Helper methods
    async def _get_server_time(self) -> Optional[int]:
        """Get server time"""
        try:
            response = await self._public_request('GET', self.endpoints['time'])
            if response:
                return response['serverTime']
            return None
        except Exception as e:
            self.logger.error(f"Error getting server time: {e}")
            return None
    
    def _get_timestamp(self) -> int:
        """Get current timestamp with offset"""
        return int(time.time() * 1000) + self.time_offset
    
    def _generate_signature(self, query_string: str) -> str:
        """Generate signature for authenticated requests"""
        return hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    async def _public_request(self, method: str, endpoint: str, params: dict = None) -> Optional[dict]:
        """Make public API request"""
        try:
            url = f"{self.base_url}{endpoint}"
            
            if method == 'GET':
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        self.logger.error(f"Public request failed: {response.status}")
                        return None
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in public request: {e}")
            return None
    
    async def _signed_request(self, method: str, endpoint: str, params: dict = None) -> Optional[dict]:
        """Make signed API request"""
        try:
            if not self.api_key or not self.secret_key:
                return None
            
            if params is None:
                params = {}
            
            # Add timestamp if not present
            if 'timestamp' not in params:
                params['timestamp'] = self._get_timestamp()
            
            # Create query string
            query_string = urlencode(params)
            
            # Generate signature
            signature = self._generate_signature(query_string)
            params['signature'] = signature
            
            # Prepare headers
            headers = {
                'X-MBX-APIKEY': self.api_key,
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            url = f"{self.base_url}{endpoint}"
            
            if method == 'GET':
                async with self.session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        response_text = await response.text()
                        self.logger.error(f"Signed request failed: {response.status} - {response_text}")
                        return None
            
            elif method == 'POST':
                async with self.session.post(url, data=params, headers=headers) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        response_text = await response.text()
                        self.logger.error(f"Signed request failed: {response.status} - {response_text}")
                        return None
            
            elif method == 'DELETE':
                async with self.session.delete(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        response_text = await response.text()
                        self.logger.error(f"Signed request failed: {response.status} - {response_text}")
                        return None
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in signed request: {e}")
            return None
    
    async def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols"""
        try:
            if not await self.check_rate_limit():
                return []
            
            exchange_info = await self._public_request('GET', self.endpoints['exchangeInfo'])
            
            if exchange_info:
                symbols = []
                for symbol_info in exchange_info['symbols']:
                    if symbol_info['status'] == 'TRADING':
                        # Convert BTCUSDT to BTC/USDT format
                        base_asset = symbol_info['baseAsset']
                        quote_asset = symbol_info['quoteAsset']
                        symbols.append(f"{base_asset}/{quote_asset}")
                
                return symbols
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting supported symbols: {e}")
            return []
