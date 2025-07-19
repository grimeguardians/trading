"""
Alpaca Exchange implementation
Primary exchange for stocks, crypto, ETFs, options, and futures
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST, TimeFrame
from alpaca_trade_api.common import URL
import pandas as pd

from exchanges.base_exchange import BaseExchange, OrderType, OrderSide, OrderStatus
from config import Config


class AlpacaExchange(BaseExchange):
    """
    Alpaca Markets exchange implementation
    Supports stocks, crypto, ETFs, options, and futures trading
    """
    
    def __init__(self, config: Config, exchange_name: str = "alpaca"):
        super().__init__(config, exchange_name)
        
        # Alpaca-specific configuration
        self.api_key = self.exchange_config.api_key
        self.api_secret = self.exchange_config.api_secret
        self.base_url = URL.PAPER if self.sandbox_mode else URL.LIVE
        
        # Initialize Alpaca API
        self.api = None
        self.trading_client = None
        
        # Alpaca-specific settings
        self.supported_order_types = [OrderType.MARKET, OrderType.LIMIT, OrderType.STOP, OrderType.STOP_LIMIT]
        self.supported_timeframes = {
            "1min": TimeFrame.Minute,
            "5min": TimeFrame(5, "Min"),
            "15min": TimeFrame(15, "Min"),
            "1hour": TimeFrame.Hour,
            "1day": TimeFrame.Day
        }
        
        # Asset type mapping
        self.asset_classes = {
            "stocks": "us_equity",
            "crypto": "crypto",
            "etf": "us_equity",
            "options": "us_option",
            "futures": "commodity"
        }
        
        self.logger.info(f"🔗 Alpaca Exchange initialized ({'Paper' if self.sandbox_mode else 'Live'} mode)")
    
    async def connect(self) -> bool:
        """Connect to Alpaca API"""
        try:
            if not self.api_key or not self.api_secret:
                self.logger.error("❌ Alpaca API credentials not configured")
                return False
            
            # Initialize REST API client
            self.api = REST(
                key_id=self.api_key,
                secret_key=self.api_secret,
                base_url=self.base_url,
                api_version='v2'
            )
            
            # Test connection
            account = self.api.get_account()
            if account:
                self.connected = True
                self.logger.info(f"✅ Connected to Alpaca ({account.status})")
                
                # Get trading limits
                await self._load_trading_limits()
                
                return True
            else:
                self.logger.error("❌ Failed to get account information")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Alpaca connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Alpaca"""
        try:
            self.connected = False
            self.authenticated = False
            self.api = None
            self.logger.info("🔌 Disconnected from Alpaca")
        except Exception as e:
            self.logger.error(f"❌ Alpaca disconnection error: {e}")
    
    async def authenticate(self) -> bool:
        """Authenticate with Alpaca"""
        try:
            if not self.api:
                return False
            
            # Get account info to verify authentication
            account = self.api.get_account()
            if account:
                self.authenticated = True
                self.logger.info(f"✅ Alpaca authenticated - Account: {account.account_number}")
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Alpaca authentication failed: {e}")
            return False
    
    async def get_account_info(self) -> Dict:
        """Get account information"""
        try:
            if not await self.check_rate_limit():
                raise Exception("Rate limit exceeded")
            
            account = self.api.get_account()
            await self.increment_request_count()
            
            return {
                "account_id": account.account_number,
                "status": account.status,
                "currency": account.currency,
                "buying_power": float(account.buying_power),
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "equity": float(account.equity),
                "last_equity": float(account.last_equity),
                "multiplier": int(account.multiplier),
                "day_trade_count": int(account.day_trade_count),
                "pattern_day_trader": account.pattern_day_trader,
                "trading_blocked": account.trading_blocked,
                "transfers_blocked": account.transfers_blocked,
                "account_blocked": account.account_blocked,
                "created_at": account.created_at.isoformat() if account.created_at else None
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
            
            account = self.api.get_account()
            await self.increment_request_count()
            
            if asset:
                # Get specific asset balance
                positions = self.api.list_positions()
                for position in positions:
                    if position.symbol == asset.upper():
                        return self.parse_balance_response({
                            "asset": asset.upper(),
                            "available": float(position.qty),
                            "locked": 0.0,
                            "total": float(position.qty),
                            "market_value": float(position.market_value),
                            "cost_basis": float(position.cost_basis),
                            "unrealized_pl": float(position.unrealized_pl)
                        })
                
                # Asset not found in positions
                return self.parse_balance_response({
                    "asset": asset.upper(),
                    "available": 0.0,
                    "locked": 0.0,
                    "total": 0.0
                })
            else:
                # Get account balance
                return self.parse_balance_response({
                    "asset": "USD",
                    "available": float(account.cash),
                    "locked": float(account.buying_power) - float(account.cash),
                    "total": float(account.buying_power)
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
            
            # Get latest quote
            quote = self.api.get_latest_quote(symbol)
            await self.increment_request_count()
            
            if quote:
                return self.parse_ticker_response({
                    "symbol": symbol,
                    "price": float(quote.bid_price + quote.ask_price) / 2,
                    "bid": float(quote.bid_price),
                    "ask": float(quote.ask_price),
                    "volume": 0.0,  # Not available in quote
                    "high": 0.0,    # Not available in quote
                    "low": 0.0,     # Not available in quote
                    "change": 0.0,  # Not available in quote
                    "change_percent": 0.0,  # Not available in quote
                    "timestamp": quote.timestamp.isoformat() if quote.timestamp else datetime.utcnow().isoformat()
                })
            else:
                return {}
                
        except Exception as e:
            self.logger.error(f"❌ Ticker error for {symbol}: {e}")
            self.metrics["errors"] += 1
            return {}
    
    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict:
        """Get orderbook data"""
        try:
            if not await self.check_rate_limit():
                raise Exception("Rate limit exceeded")
            
            # Alpaca doesn't provide full orderbook, use latest quote
            quote = self.api.get_latest_quote(symbol)
            await self.increment_request_count()
            
            if quote:
                return {
                    "symbol": symbol,
                    "bids": [[float(quote.bid_price), float(quote.bid_size)]],
                    "asks": [[float(quote.ask_price), float(quote.ask_size)]],
                    "timestamp": quote.timestamp.isoformat() if quote.timestamp else datetime.utcnow().isoformat()
                }
            else:
                return {}
                
        except Exception as e:
            self.logger.error(f"❌ Orderbook error for {symbol}: {e}")
            self.metrics["errors"] += 1
            return {}
    
    async def get_klines(self, symbol: str, interval: str, limit: int = 500) -> List[Dict]:
        """Get candlestick data"""
        try:
            if not await self.check_rate_limit():
                raise Exception("Rate limit exceeded")
            
            # Map interval to Alpaca timeframe
            if interval not in self.supported_timeframes:
                self.logger.error(f"❌ Unsupported interval: {interval}")
                return []
            
            timeframe = self.supported_timeframes[interval]
            
            # Get bars
            end_time = datetime.now()
            start_time = end_time - timedelta(days=30)  # Last 30 days
            
            bars = self.api.get_bars(
                symbol,
                timeframe,
                start=start_time.isoformat(),
                end=end_time.isoformat(),
                limit=limit
            )
            
            await self.increment_request_count()
            
            klines = []
            for bar in bars:
                klines.append({
                    "timestamp": bar.timestamp.isoformat(),
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume)
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
            
            # Convert enums to Alpaca format
            alpaca_side = "buy" if side == OrderSide.BUY else "sell"
            alpaca_type = self._convert_order_type(order_type)
            
            # Prepare order parameters
            order_params = {
                "symbol": symbol,
                "qty": quantity,
                "side": alpaca_side,
                "type": alpaca_type,
                "time_in_force": kwargs.get("time_in_force", "day")
            }
            
            # Add price for limit orders
            if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and price:
                order_params["limit_price"] = price
            
            # Add stop price for stop orders
            if order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and "stop_price" in kwargs:
                order_params["stop_price"] = kwargs["stop_price"]
            
            # Submit order
            order = self.api.submit_order(**order_params)
            await self.increment_request_count()
            
            if order:
                # Store order
                self.open_orders[order.id] = order
                self.metrics["orders_placed"] += 1
                
                return self.parse_order_response({
                    "id": order.id,
                    "symbol": order.symbol,
                    "side": order.side,
                    "type": order.order_type,
                    "quantity": float(order.qty),
                    "price": float(order.limit_price) if order.limit_price else None,
                    "status": order.status,
                    "filled_quantity": float(order.filled_qty) if order.filled_qty else 0.0,
                    "filled_price": float(order.filled_avg_price) if order.filled_avg_price else 0.0,
                    "timestamp": order.created_at.isoformat() if order.created_at else datetime.utcnow().isoformat(),
                    "fees": {}
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
            self.api.cancel_order(order_id)
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
            
            order = self.api.get_order(order_id)
            await self.increment_request_count()
            
            if order:
                return self.parse_order_response({
                    "id": order.id,
                    "symbol": order.symbol,
                    "side": order.side,
                    "type": order.order_type,
                    "quantity": float(order.qty),
                    "price": float(order.limit_price) if order.limit_price else None,
                    "status": order.status,
                    "filled_quantity": float(order.filled_qty) if order.filled_qty else 0.0,
                    "filled_price": float(order.filled_avg_price) if order.filled_avg_price else 0.0,
                    "timestamp": order.created_at.isoformat() if order.created_at else datetime.utcnow().isoformat(),
                    "fees": {}
                })
            else:
                return {}
                
        except Exception as e:
            self.logger.error(f"❌ Order status error: {e}")
            self.metrics["errors"] += 1
            return {}
    
    async def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """Get open orders"""
        try:
            if not await self.check_rate_limit():
                raise Exception("Rate limit exceeded")
            
            orders = self.api.list_orders(status='open')
            await self.increment_request_count()
            
            open_orders = []
            for order in orders:
                if symbol is None or order.symbol == symbol:
                    open_orders.append(self.parse_order_response({
                        "id": order.id,
                        "symbol": order.symbol,
                        "side": order.side,
                        "type": order.order_type,
                        "quantity": float(order.qty),
                        "price": float(order.limit_price) if order.limit_price else None,
                        "status": order.status,
                        "filled_quantity": float(order.filled_qty) if order.filled_qty else 0.0,
                        "filled_price": float(order.filled_avg_price) if order.filled_avg_price else 0.0,
                        "timestamp": order.created_at.isoformat() if order.created_at else datetime.utcnow().isoformat(),
                        "fees": {}
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
            
            orders = self.api.list_orders(status='closed', limit=limit)
            await self.increment_request_count()
            
            order_history = []
            for order in orders:
                if symbol is None or order.symbol == symbol:
                    order_history.append(self.parse_order_response({
                        "id": order.id,
                        "symbol": order.symbol,
                        "side": order.side,
                        "type": order.order_type,
                        "quantity": float(order.qty),
                        "price": float(order.limit_price) if order.limit_price else None,
                        "status": order.status,
                        "filled_quantity": float(order.filled_qty) if order.filled_qty else 0.0,
                        "filled_price": float(order.filled_avg_price) if order.filled_avg_price else 0.0,
                        "timestamp": order.created_at.isoformat() if order.created_at else datetime.utcnow().isoformat(),
                        "fees": {}
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
            
            positions = self.api.list_positions()
            await self.increment_request_count()
            
            position_list = []
            for position in positions:
                position_list.append({
                    "symbol": position.symbol,
                    "quantity": float(position.qty),
                    "side": "long" if float(position.qty) > 0 else "short",
                    "entry_price": float(position.avg_entry_price),
                    "current_price": float(position.current_price) if position.current_price else 0.0,
                    "market_value": float(position.market_value),
                    "cost_basis": float(position.cost_basis),
                    "unrealized_pnl": float(position.unrealized_pl),
                    "unrealized_pnl_pct": float(position.unrealized_plpc),
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            return position_list
            
        except Exception as e:
            self.logger.error(f"❌ Positions error: {e}")
            self.metrics["errors"] += 1
            return []
    
    async def get_trading_fees(self, symbol: str, order_type: OrderType) -> Dict:
        """Get trading fees for symbol"""
        try:
            # Alpaca commission structure
            return {
                "maker_fee": 0.0,    # Commission-free
                "taker_fee": 0.0,    # Commission-free
                "fee_currency": "USD"
            }
            
        except Exception as e:
            self.logger.error(f"❌ Trading fees error: {e}")
            return {"maker_fee": 0.0, "taker_fee": 0.0, "fee_currency": "USD"}
    
    def _convert_order_type(self, order_type: OrderType) -> str:
        """Convert OrderType enum to Alpaca format"""
        mapping = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP: "stop",
            OrderType.STOP_LIMIT: "stop_limit"
        }
        return mapping.get(order_type, "market")
    
    def standardize_symbol(self, symbol: str) -> str:
        """Standardize symbol format for Alpaca"""
        # Alpaca uses simple uppercase format
        return symbol.upper()
    
    async def _load_trading_limits(self):
        """Load trading limits and precision"""
        try:
            # Get assets information
            assets = self.api.list_assets()
            
            for asset in assets:
                if asset.tradable:
                    symbol = asset.symbol
                    
                    # Set minimum order size (Alpaca minimum is usually 1 share)
                    self.min_order_size[symbol] = 1.0
                    
                    # Set maximum order size (based on account buying power)
                    account = self.api.get_account()
                    self.max_order_size[symbol] = float(account.buying_power) / 10  # Conservative limit
                    
                    # Set precision
                    self.price_precision[symbol] = 2  # Default 2 decimal places
                    self.quantity_precision[symbol] = 0  # Whole shares
                    
        except Exception as e:
            self.logger.error(f"❌ Trading limits loading error: {e}")
    
    async def ping(self) -> bool:
        """Ping Alpaca to keep connection alive"""
        try:
            account = self.api.get_account()
            return account is not None
        except Exception as e:
            self.logger.error(f"❌ Alpaca ping error: {e}")
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
            "exchange": "alpaca",
            "raw_response": response
        }
