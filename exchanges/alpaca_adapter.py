"""
Alpaca Markets exchange adapter
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import aiohttp
import json
from dataclasses import dataclass

from .base_exchange import BaseExchange, TradeOrder, Position, Balance, Ticker, OrderBook, OHLCV

logger = logging.getLogger(__name__)

class AlpacaAdapter(BaseExchange):
    """Alpaca Markets exchange adapter"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.session = None
        self.headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.secret_key,
            'Content-Type': 'application/json'
        }
        
        # Alpaca API endpoints
        self.endpoints = {
            'account': f'{self.base_url}/v2/account',
            'positions': f'{self.base_url}/v2/positions',
            'orders': f'{self.base_url}/v2/orders',
            'assets': f'{self.base_url}/v2/assets',
            'portfolio': f'{self.base_url}/v2/portfolio',
            'bars': f'{self.base_url}/v2/stocks/bars',
            'trades': f'{self.base_url}/v2/stocks/trades',
            'quotes': f'{self.base_url}/v2/stocks/quotes',
            'crypto_bars': f'{self.base_url}/v1beta3/crypto/bars',
            'crypto_quotes': f'{self.base_url}/v1beta3/crypto/quotes',
            'options_bars': f'{self.base_url}/v1beta1/options/bars',
            'options_quotes': f'{self.base_url}/v1beta1/options/quotes'
        }
        
        # Supported timeframes
        self.timeframes = {
            '1min': '1Min',
            '5min': '5Min',
            '15min': '15Min',
            '30min': '30Min',
            '1h': '1Hour',
            '4h': '4Hour',
            '1d': '1Day',
            '1w': '1Week',
            '1M': '1Month'
        }
        
        # Asset type mapping
        self.asset_types = {
            'stocks': 'us_equity',
            'crypto': 'crypto',
            'options': 'us_option',
            'etfs': 'us_equity'
        }
    
    async def connect(self) -> bool:
        """Connect to Alpaca API"""
        try:
            self.session = aiohttp.ClientSession(headers=self.headers)
            
            # Test connection by getting account info
            async with self.session.get(self.endpoints['account']) as response:
                if response.status == 200:
                    account_data = await response.json()
                    self.logger.info(f"Connected to Alpaca - Account: {account_data.get('id')}")
                    self.is_connected = True
                    return True
                else:
                    self.logger.error(f"Failed to connect to Alpaca: {response.status}")
                    return False
        except Exception as e:
            self.logger.error(f"Error connecting to Alpaca: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Alpaca API"""
        try:
            if self.session:
                await self.session.close()
                self.session = None
            self.is_connected = False
            self.logger.info("Disconnected from Alpaca")
            return True
        except Exception as e:
            self.logger.error(f"Error disconnecting from Alpaca: {e}")
            return False
    
    async def get_ticker(self, symbol: str) -> Optional[Ticker]:
        """Get ticker data for a symbol"""
        try:
            if not await self.check_rate_limit():
                return None
            
            # Determine asset type
            asset_type = await self._get_asset_type(symbol)
            
            if asset_type == 'crypto':
                endpoint = self.endpoints['crypto_quotes']
                params = {'symbols': symbol, 'asof': datetime.now().isoformat()}
            else:
                endpoint = self.endpoints['quotes']
                params = {'symbols': symbol, 'asof': datetime.now().isoformat()}
            
            async with self.session.get(endpoint, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if asset_type == 'crypto':
                        quote_data = data.get('quotes', {}).get(symbol, {})
                    else:
                        quote_data = data.get('quotes', {}).get(symbol, {})
                    
                    if quote_data:
                        # Get 24h change data
                        bars_data = await self._get_24h_change(symbol, asset_type)
                        
                        return Ticker(
                            symbol=symbol,
                            price=float(quote_data.get('bp', 0)),  # Best bid price
                            volume=float(quote_data.get('bs', 0)),  # Best bid size
                            change_24h=bars_data.get('change_24h', 0),
                            change_percentage_24h=bars_data.get('change_percentage_24h', 0),
                            high_24h=bars_data.get('high_24h', 0),
                            low_24h=bars_data.get('low_24h', 0),
                            timestamp=datetime.now()
                        )
                else:
                    self.logger.error(f"Failed to get ticker for {symbol}: {response.status}")
                    return None
        except Exception as e:
            self.logger.error(f"Error getting ticker for {symbol}: {e}")
            return None
    
    async def get_order_book(self, symbol: str, limit: int = 100) -> Optional[OrderBook]:
        """Get order book data for a symbol"""
        try:
            if not await self.check_rate_limit():
                return None
            
            # Alpaca doesn't provide full order book, only best bid/ask
            ticker = await self.get_ticker(symbol)
            if not ticker:
                return None
            
            # Simulate order book with best bid/ask
            return OrderBook(
                symbol=symbol,
                bids=[(ticker.price - 0.01, 100)],  # Simulated
                asks=[(ticker.price + 0.01, 100)],  # Simulated
                timestamp=datetime.now()
            )
        except Exception as e:
            self.logger.error(f"Error getting order book for {symbol}: {e}")
            return None
    
    async def get_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 500) -> List[OHLCV]:
        """Get OHLCV data for a symbol"""
        try:
            if not await self.check_rate_limit():
                return []
            
            # Convert timeframe
            alpaca_timeframe = self.timeframes.get(timeframe, '1Hour')
            
            # Determine asset type and endpoint
            asset_type = await self._get_asset_type(symbol)
            
            if asset_type == 'crypto':
                endpoint = self.endpoints['crypto_bars']
            else:
                endpoint = self.endpoints['bars']
            
            # Calculate start time
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=limit)
            
            params = {
                'symbols': symbol,
                'timeframe': alpaca_timeframe,
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'limit': limit
            }
            
            async with self.session.get(endpoint, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    bars = data.get('bars', {}).get(symbol, [])
                    
                    ohlcv_data = []
                    for bar in bars:
                        ohlcv_data.append(OHLCV(
                            symbol=symbol,
                            timestamp=datetime.fromisoformat(bar['t'].replace('Z', '+00:00')),
                            open=float(bar['o']),
                            high=float(bar['h']),
                            low=float(bar['l']),
                            close=float(bar['c']),
                            volume=float(bar['v'])
                        ))
                    
                    return ohlcv_data
                else:
                    self.logger.error(f"Failed to get OHLCV for {symbol}: {response.status}")
                    return []
        except Exception as e:
            self.logger.error(f"Error getting OHLCV for {symbol}: {e}")
            return []
    
    async def get_balances(self) -> Dict[str, Balance]:
        """Get account balances"""
        try:
            if not await self.check_rate_limit():
                return {}
            
            async with self.session.get(self.endpoints['account']) as response:
                if response.status == 200:
                    account_data = await response.json()
                    
                    # Get portfolio info
                    async with self.session.get(self.endpoints['portfolio']) as portfolio_response:
                        if portfolio_response.status == 200:
                            portfolio_data = await portfolio_response.json()
                            
                            return {
                                'USD': Balance(
                                    asset='USD',
                                    free=float(account_data.get('buying_power', 0)),
                                    locked=float(account_data.get('equity', 0)) - float(account_data.get('buying_power', 0)),
                                    total=float(account_data.get('equity', 0))
                                )
                            }
                else:
                    self.logger.error(f"Failed to get balances: {response.status}")
                    return {}
        except Exception as e:
            self.logger.error(f"Error getting balances: {e}")
            return {}
    
    async def get_positions(self) -> Dict[str, Position]:
        """Get open positions"""
        try:
            if not await self.check_rate_limit():
                return {}
            
            async with self.session.get(self.endpoints['positions']) as response:
                if response.status == 200:
                    positions_data = await response.json()
                    positions = {}
                    
                    for pos_data in positions_data:
                        symbol = pos_data['symbol']
                        
                        positions[symbol] = Position(
                            symbol=symbol,
                            side='long' if float(pos_data['qty']) > 0 else 'short',
                            quantity=abs(float(pos_data['qty'])),
                            entry_price=float(pos_data['avg_entry_price']),
                            current_price=float(pos_data['market_value']) / float(pos_data['qty']),
                            unrealized_pnl=float(pos_data['unrealized_pl']),
                            realized_pnl=float(pos_data.get('realized_pl', 0)),
                            timestamp=datetime.now()
                        )
                    
                    return positions
                else:
                    self.logger.error(f"Failed to get positions: {response.status}")
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
            
            # Prepare order data
            order_data = {
                'symbol': order.symbol,
                'qty': str(order.quantity),
                'side': order.side,
                'type': order.type,
                'time_in_force': order.time_in_force
            }
            
            if order.type == 'limit':
                order_data['limit_price'] = str(order.price)
            elif order.type == 'stop':
                order_data['stop_price'] = str(order.stop_price)
            elif order.type == 'stop_limit':
                order_data['limit_price'] = str(order.price)
                order_data['stop_price'] = str(order.stop_price)
            
            async with self.session.post(self.endpoints['orders'], json=order_data) as response:
                if response.status == 201:
                    response_data = await response.json()
                    
                    order.order_id = response_data['id']
                    order.status = response_data['status']
                    order.created_at = datetime.fromisoformat(response_data['created_at'].replace('Z', '+00:00'))
                    order.updated_at = datetime.fromisoformat(response_data['updated_at'].replace('Z', '+00:00'))
                    
                    if response_data.get('filled_qty'):
                        order.filled_quantity = float(response_data['filled_qty'])
                    if response_data.get('filled_avg_price'):
                        order.filled_price = float(response_data['filled_avg_price'])
                    
                    return order
                else:
                    self.logger.error(f"Failed to place order: {response.status}")
                    return None
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            if not await self.check_rate_limit():
                return False
            
            async with self.session.delete(f"{self.endpoints['orders']}/{order_id}") as response:
                if response.status == 204:
                    return True
                else:
                    self.logger.error(f"Failed to cancel order {order_id}: {response.status}")
                    return False
        except Exception as e:
            self.logger.error(f"Error canceling order {order_id}: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> Optional[TradeOrder]:
        """Get order status"""
        try:
            if not await self.check_rate_limit():
                return None
            
            async with self.session.get(f"{self.endpoints['orders']}/{order_id}") as response:
                if response.status == 200:
                    order_data = await response.json()
                    
                    order = TradeOrder(
                        order_id=order_data['id'],
                        symbol=order_data['symbol'],
                        side=order_data['side'],
                        type=order_data['order_type'],
                        quantity=float(order_data['qty']),
                        price=float(order_data.get('limit_price', 0)) if order_data.get('limit_price') else None,
                        stop_price=float(order_data.get('stop_price', 0)) if order_data.get('stop_price') else None,
                        time_in_force=order_data['time_in_force'],
                        status=order_data['status'],
                        filled_quantity=float(order_data.get('filled_qty', 0)),
                        filled_price=float(order_data.get('filled_avg_price', 0)) if order_data.get('filled_avg_price') else None,
                        created_at=datetime.fromisoformat(order_data['created_at'].replace('Z', '+00:00')),
                        updated_at=datetime.fromisoformat(order_data['updated_at'].replace('Z', '+00:00'))
                    )
                    
                    return order
                else:
                    self.logger.error(f"Failed to get order status for {order_id}: {response.status}")
                    return None
        except Exception as e:
            self.logger.error(f"Error getting order status for {order_id}: {e}")
            return None
    
    async def get_trade_history(self, symbol: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get trade history"""
        try:
            if not await self.check_rate_limit():
                return []
            
            params = {'status': 'all', 'limit': limit}
            if symbol:
                params['symbols'] = symbol
            
            async with self.session.get(self.endpoints['orders'], params=params) as response:
                if response.status == 200:
                    orders_data = await response.json()
                    
                    trades = []
                    for order_data in orders_data:
                        if order_data['status'] == 'filled':
                            trades.append({
                                'order_id': order_data['id'],
                                'symbol': order_data['symbol'],
                                'side': order_data['side'],
                                'quantity': float(order_data['filled_qty']),
                                'price': float(order_data['filled_avg_price']),
                                'timestamp': datetime.fromisoformat(order_data['updated_at'].replace('Z', '+00:00'))
                            })
                    
                    return trades
                else:
                    self.logger.error(f"Failed to get trade history: {response.status}")
                    return []
        except Exception as e:
            self.logger.error(f"Error getting trade history: {e}")
            return []
    
    async def _get_asset_type(self, symbol: str) -> str:
        """Determine asset type for symbol"""
        # Simple heuristic - in real implementation, you'd query the assets endpoint
        if symbol.endswith('USD') or symbol.endswith('BTC') or symbol.endswith('ETH'):
            return 'crypto'
        else:
            return 'stocks'
    
    async def _get_24h_change(self, symbol: str, asset_type: str) -> Dict[str, float]:
        """Get 24h change data for a symbol"""
        try:
            # Get 24h bars
            if asset_type == 'crypto':
                endpoint = self.endpoints['crypto_bars']
            else:
                endpoint = self.endpoints['bars']
            
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=24)
            
            params = {
                'symbols': symbol,
                'timeframe': '1Hour',
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'limit': 24
            }
            
            async with self.session.get(endpoint, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    bars = data.get('bars', {}).get(symbol, [])
                    
                    if len(bars) >= 2:
                        first_bar = bars[0]
                        last_bar = bars[-1]
                        
                        open_price = float(first_bar['o'])
                        close_price = float(last_bar['c'])
                        high_24h = max(float(bar['h']) for bar in bars)
                        low_24h = min(float(bar['l']) for bar in bars)
                        
                        change_24h = close_price - open_price
                        change_percentage_24h = (change_24h / open_price) * 100 if open_price > 0 else 0
                        
                        return {
                            'change_24h': change_24h,
                            'change_percentage_24h': change_percentage_24h,
                            'high_24h': high_24h,
                            'low_24h': low_24h
                        }
            
            return {'change_24h': 0, 'change_percentage_24h': 0, 'high_24h': 0, 'low_24h': 0}
        except Exception as e:
            self.logger.error(f"Error getting 24h change for {symbol}: {e}")
            return {'change_24h': 0, 'change_percentage_24h': 0, 'high_24h': 0, 'low_24h': 0}
    
    async def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols"""
        try:
            if not await self.check_rate_limit():
                return []
            
            async with self.session.get(self.endpoints['assets']) as response:
                if response.status == 200:
                    assets_data = await response.json()
                    
                    symbols = []
                    for asset in assets_data:
                        if asset['tradable'] and asset['status'] == 'active':
                            symbols.append(asset['symbol'])
                    
                    return symbols
                else:
                    self.logger.error(f"Failed to get supported symbols: {response.status}")
                    return []
        except Exception as e:
            self.logger.error(f"Error getting supported symbols: {e}")
            return []
