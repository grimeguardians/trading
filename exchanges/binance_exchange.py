"""
Binance Exchange implementation
Crypto and futures trading
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import ccxt.async_support as ccxt
import pandas as pd

from exchanges.base_exchange import BaseExchange, OrderType, OrderSide, OrderStatus
from config import Config


class BinanceExchange(BaseExchange):
    """
    Binance exchange implementation for crypto and futures trading
    """
    
    def __init__(self, config: Config, exchange_name: str = "binance"):
        super().__init__(config, exchange_name)
        
        # Binance-specific configuration
        self.api_key = self.exchange_config.api_key
        self.api_secret = self.exchange_config.api_secret
        
        # Initialize Binance client
        self.client = None
        
        # Binance-specific settings
        self.supported_order_types = [OrderType.MARKET, OrderType.LIMIT, OrderType.STOP, OrderType.STOP_LIMIT]
        self.supported_timeframes = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d"
        }
        
        # Rate limits
        self.rate_limit = 1200  # requests per minute
        
        self.logger.info(f"🔗 Binance Exchange initialized ({'Testnet' if self.sandbox_mode else 'Live'} mode)")
    
    async def connect(self) -> bool:
        """Connect to Binance API"""
        try:
            if not self.api_key or not self.api_secret:
                self.logger.error("❌ Binance API credentials not configured")
                return False
            
            # Initialize CCXT client
            self.client = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'sandbox': self.sandbox_mode,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot'  # spot, future, delivery, option
                }
            })
            
            # Test connection
            await self.client.load_markets()
            self.connected = True
            self.logger.info("✅ Connected to Binance")
            
            # Load trading limits
            await self._load_trading_limits()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Binance connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Binance"""
        try:
            if self.client:
                await self.client.close()
            self.connected = False
            self.authenticated = False
            self.client = None
            self.logger.info("🔌 Disconnected from Binance")
        except Exception as e:
            self.logger.error(f"❌ Binance disconnection error: {e}")
    
    async def authenticate(self) -> bool:
        """Authenticate with Binance"""
        try:
            if not self.client:
                return False
            
            # Get account info to verify authentication
            account = await self.client.fetch_balance()
            if account:
                self.authenticated = True
                self.logger.info("✅ Binance authenticated")
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Binance authentication failed: {e}")
            return False
    
    async def get_account_info(self) -> Dict:
        """Get account information"""
        try:
            if not await self.check_rate_limit():
                raise Exception("Rate limit exceeded")
            
            account = await self.client.fetch_balance()
            await self.increment_request_count()
            
            return {
                "account_id": "binance_account",
                "status": "ACTIVE",
                "currency": "USDT",
                "total_balance": account.get('total', {}),
                "free_balance": account.get('free', {}),
                "used_balance": account.get('used', {}),
                "info": account.get('info', {})
            }
            
        except Exception as e:
            self.logger.error(f"❌ Account info error: {e}")
            self.metrics["errors"] += 1
            return {}
    
    async def get_balance(self, asset: str = None) -> Dict:
        """Get account balance"""
        try:
            if not await self.check_rate_limit():
                raise Exception("Rate limit exceeded")
            
            balance = await self.client.fetch_balance()
            await self.increment_request_count()
            
            if asset:
                asset_upper = asset.upper()
                if asset_upper in balance['total']:
                    return self.parse_balance_response({
                        "asset": asset_upper,
                        "available": balance['free'].get(asset_upper, 0.0),
                        "locked": balance['used'].get(asset_upper, 0.0),
                        "total": balance['total'].get(asset_upper, 0.0)
                    })
                else:
                    return self.parse_balance_response({
                        "asset": asset_upper,
                        "available": 0.0,
                        "locked": 0.0,
                        "total": 0.0
                    })
            else:
                # Return USDT balance as default
                return self.parse_balance_response({
                    "asset": "USDT",
                    "available": balance['free'].get('USDT', 0.0),
                    "locked": balance['used'].get('USDT', 0.0),
                    "total": balance['total'].get('USDT', 0.0)
                })
                
        except Exception as e:
            self.logger.error(f"❌ Balance error: {e}")
            self.metrics["errors"] += 1
            return {}
    
    async def get_ticker(self, symbol: str) -> Dict:
        """Get ticker data for symbol"""
        try:
            if not await self.check_rate_limit():
                raise Exception("Rate limit exceeded")
            
            ticker = await self.client.fetch_ticker(symbol)
            await self.increment_request_count()
            
            return self.parse_ticker_response({
                "symbol": symbol,
                "price": ticker.get('last', 0.0),
                "bid": ticker.get('bid', 0.0),
                "ask": ticker.get('ask', 0.0),
                "volume": ticker.get('baseVolume', 0.0),
                "high": ticker.get('high', 0.0),
                "low": ticker.get('low', 0.0),
                "change": ticker.get('change', 0.0),
                "change_percent": ticker.get('percentage', 0.0),
                "timestamp": datetime.utcfromtimestamp(ticker.get('timestamp', 0) / 1000).isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"❌ Ticker error for {symbol}: {e}")
            self.metrics["errors"] += 1
            return {}
    
    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict:
        """Get orderbook data"""
        try:
            if not await self.check_rate_limit():
                raise Exception("Rate limit exceeded")
            
            orderbook = await self.client.fetch_order_book(symbol, limit)
            await self.increment_request_count()
            
            return {
                "symbol": symbol,
                "bids": orderbook.get('bids', []),
                "asks": orderbook.get('asks', []),
                "timestamp": datetime.utcfromtimestamp(orderbook.get('timestamp', 0) / 1000).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Orderbook error for {symbol}: {e}")
            self.metrics["errors"] += 1
            return {}
    
    async def get_klines(self, symbol: str, interval: str, limit: int = 500) -> List[Dict]:
        """Get candlestick data"""
        try:
            if not await self.check_rate_limit():
                raise Exception("Rate limit exceeded")
            
            # Map interval to Binance timeframe
            if interval not in self.supported_timeframes:
                self.logger.error(f"❌ Unsupported interval: {interval}")
                return []
            
            timeframe = self.supported_timeframes[interval]
            
            # Get OHLCV data
            ohlcv = await self.client.fetch_ohlcv(symbol, timeframe, limit=limit)
            await self.increment_request_count()
            
            klines = []
            for bar in ohlcv:
                klines.append({
                    "timestamp": datetime.utcfromtimestamp(bar[0] / 1000).isoformat(),
                    "open": float(bar[1]),
                    "high": float(bar[2]),
                    "low": float(bar[3]),
                    "close": float(bar[4]),
                    "volume": float(bar[5])
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
            if not await self.check_rate_limit():
                raise Exception("Rate limit exceeded")
            
            # Validate order
            is_valid, error_msg = await self.validate_order(symbol, side, order_type, quantity, price)
            if not is_valid:
                self.logger.error(f"❌ Order validation failed: {error_msg}")
                return {}
            
            # Convert enums to Binance format
            binance_side = "buy" if side == OrderSide.BUY else "sell"
            binance_type = self._convert_order_type(order_type)
            
            # Place order
            order_params = {
                'symbol': symbol,
                'type': binance_type,
                'side': binance_side,
                'amount': quantity,
                'params': {}
            }
            
            # Add price for limit orders
            if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
                order_params['price'] = price
            
            # Add stop price for stop orders
            if order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and 'stop_price' in kwargs:
                order_params['params']['stopPrice'] = kwargs['stop_price']
            
            order = await self.client.create_order(**order_params)
            await self.increment_request_count()
            
            if order:
                # Store order
                self.open_orders[order['id']] = order
                self.metrics["orders_placed"] += 1
                
                return self.parse_order_response({
                    "id": order['id'],
                    "symbol": order['symbol'],
                    "side": order['side'],
                    "type": order['type'],
                    "quantity": order['amount'],
                    "price": order.get('price', 0.0),
                    "status": order['status'],
                    "filled_quantity": order.get('filled', 0.0),
                    "filled_price": order.get('average', 0.0),
                    "timestamp": datetime.utcfromtimestamp(order.get('timestamp', 0) / 1000).isoformat(),
                    "fees": order.get('fee', {})
                })
            else:
                return {}
                
        except Exception as e:
            self.logger.error(f"❌ Order placement error: {e}")
            self.metrics["errors"] += 1
            return {}
    
    async def cancel_order(self, symbol: str, order_id: str) -> Dict:
        """Cancel order"""
        try:
            if not await self.check_rate_limit():
                raise Exception("Rate limit exceeded")
            
            # Cancel order
            result = await self.client.cancel_order(order_id, symbol)
            await self.increment_request_count()
            
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
            if not await self.check_rate_limit():
                raise Exception("Rate limit exceeded")
            
            order = await self.client.fetch_order(order_id, symbol)
            await self.increment_request_count()
            
            return self.parse_order_response({
                "id": order['id'],
                "symbol": order['symbol'],
                "side": order['side'],
                "type": order['type'],
                "quantity": order['amount'],
                "price": order.get('price', 0.0),
                "status": order['status'],
                "filled_quantity": order.get('filled', 0.0),
                "filled_price": order.get('average', 0.0),
                "timestamp": datetime.utcfromtimestamp(order.get('timestamp', 0) / 1000).isoformat(),
                "fees": order.get('fee', {})
            })
            
        except Exception as e:
            self.logger.error(f"❌ Order status error: {e}")
            self.metrics["errors"] += 1
            return {}
    
    async def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """Get open orders"""
        try:
            if not await self.check_rate_limit():
                raise Exception("Rate limit exceeded")
            
            if symbol:
                orders = await self.client.fetch_open_orders(symbol)
            else:
                orders = await self.client.fetch_open_orders()
            
            await self.increment_request_count()
            
            open_orders = []
            for order in orders:
                open_orders.append(self.parse_order_response({
                    "id": order['id'],
                    "symbol": order['symbol'],
                    "side": order['side'],
                    "type": order['type'],
                    "quantity": order['amount'],
                    "price": order.get('price', 0.0),
                    "status": order['status'],
                    "filled_quantity": order.get('filled', 0.0),
                    "filled_price": order.get('average', 0.0),
                    "timestamp": datetime.utcfromtimestamp(order.get('timestamp', 0) / 1000).isoformat(),
                    "fees": order.get('fee', {})
                }))
            
            return open_orders
            
        except Exception as e:
            self.logger.error(f"❌ Open orders error: {e}")
            self.metrics["errors"] += 1
            return []
    
    async def get_order_history(self, symbol: str = None, limit: int = 100) -> List[Dict]:
        """Get order history"""
        try:
            if not await self.check_rate_limit():
                raise Exception("Rate limit exceeded")
            
            if symbol:
                orders = await self.client.fetch_orders(symbol, limit=limit)
            else:
                # Get orders for all symbols (limited implementation)
                orders = []
                markets = await self.client.fetch_markets()
                for market in markets[:5]:  # Limit to first 5 markets
                    try:
                        symbol_orders = await self.client.fetch_orders(market['symbol'], limit=10)
                        orders.extend(symbol_orders)
                    except:
                        continue
            
            await self.increment_request_count()
            
            order_history = []
            for order in orders:
                order_history.append(self.parse_order_response({
                    "id": order['id'],
                    "symbol": order['symbol'],
                    "side": order['side'],
                    "type": order['type'],
                    "quantity": order['amount'],
                    "price": order.get('price', 0.0),
                    "status": order['status'],
                    "filled_quantity": order.get('filled', 0.0),
                    "filled_price": order.get('average', 0.0),
                    "timestamp": datetime.utcfromtimestamp(order.get('timestamp', 0) / 1000).isoformat(),
                    "fees": order.get('fee', {})
                }))
            
            return order_history
            
        except Exception as e:
            self.logger.error(f"❌ Order history error: {e}")
            self.metrics["errors"] += 1
            return []
    
    async def get_positions(self) -> List[Dict]:
        """Get current positions"""
        try:
            if not await self.check_rate_limit():
                raise Exception("Rate limit exceeded")
            
            balance = await self.client.fetch_balance()
            await self.increment_request_count()
            
            positions = []
            for asset, amount in balance['total'].items():
                if amount > 0:
                    positions.append({
                        "symbol": asset,
                        "quantity": amount,
                        "side": "long",
                        "entry_price": 0.0,  # Not available in spot trading
                        "current_price": 0.0,  # Would need to fetch ticker
                        "market_value": 0.0,
                        "cost_basis": 0.0,
                        "unrealized_pnl": 0.0,
                        "unrealized_pnl_pct": 0.0,
                        "timestamp": datetime.utcnow().isoformat()
                    })
            
            return positions
            
        except Exception as e:
            self.logger.error(f"❌ Positions error: {e}")
            self.metrics["errors"] += 1
            return []
    
    async def get_trading_fees(self, symbol: str, order_type: OrderType) -> Dict:
        """Get trading fees for symbol"""
        try:
            # Binance fee structure (VIP 0)
            return {
                "maker_fee": 0.001,  # 0.1%
                "taker_fee": 0.001,  # 0.1%
                "fee_currency": "BNB"  # Can be paid in BNB for discount
            }
            
        except Exception as e:
            self.logger.error(f"❌ Trading fees error: {e}")
            return {"maker_fee": 0.001, "taker_fee": 0.001, "fee_currency": "BNB"}
    
    def _convert_order_type(self, order_type: OrderType) -> str:
        """Convert OrderType enum to Binance format"""
        mapping = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP: "stop_market",
            OrderType.STOP_LIMIT: "stop_limit"
        }
        return mapping.get(order_type, "market")
    
    def standardize_symbol(self, symbol: str) -> str:
        """Standardize symbol format for Binance"""
        # Binance uses format like "BTCUSDT"
        return symbol.upper().replace("/", "")
    
    async def _load_trading_limits(self):
        """Load trading limits and precision"""
        try:
            markets = await self.client.fetch_markets()
            
            for market in markets:
                symbol = market['symbol']
                
                # Set limits from market info
                if 'limits' in market:
                    limits = market['limits']
                    
                    # Amount limits
                    if 'amount' in limits:
                        self.min_order_size[symbol] = limits['amount'].get('min', 0.0)
                        self.max_order_size[symbol] = limits['amount'].get('max', float('inf'))
                    
                    # Price limits
                    if 'price' in limits:
                        pass  # Could store price limits if needed
                
                # Set precision from market info
                if 'precision' in market:
                    precision = market['precision']
                    self.price_precision[symbol] = precision.get('price', 8)
                    self.quantity_precision[symbol] = precision.get('amount', 8)
                    
        except Exception as e:
            self.logger.error(f"❌ Trading limits loading error: {e}")
    
    async def ping(self) -> bool:
        """Ping Binance to keep connection alive"""
        try:
            await self.client.fetch_time()
            return True
        except Exception as e:
            self.logger.error(f"❌ Binance ping error: {e}")
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
            "exchange": "binance",
            "raw_response": response
        }
