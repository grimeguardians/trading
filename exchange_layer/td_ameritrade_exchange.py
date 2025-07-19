"""
TD Ameritrade exchange implementation
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

logger = setup_logger("td_ameritrade_exchange")

class TDAmeritradeExchange(BaseExchange):
    """TD Ameritrade exchange implementation"""
    
    def __init__(self, config: ExchangeConfig):
        super().__init__(config.__dict__)
        self.session = None
        self.access_token = None
        self.refresh_token = None
        self.token_expires = None
        
    async def connect(self) -> bool:
        """Connect to TD Ameritrade"""
        try:
            self.session = aiohttp.ClientSession()
            
            # Get access token
            await self._get_access_token()
            
            # Test connection
            account_info = await self.get_account_info()
            if account_info:
                self.connected = True
                logger.info("Connected to TD Ameritrade")
                return True
            
        except Exception as e:
            logger.error(f"Failed to connect to TD Ameritrade: {e}")
            self.connected = False
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from TD Ameritrade"""
        if self.session:
            await self.session.close()
            self.session = None
        self.connected = False
        logger.info("Disconnected from TD Ameritrade")
        return True
    
    async def _get_access_token(self):
        """Get access token for TD Ameritrade API"""
        try:
            # TD Ameritrade OAuth flow would go here
            # For now, assume token is provided in config
            self.access_token = self.config.get("access_token", "")
            
            if not self.access_token:
                raise Exception("Access token not provided in configuration")
            
            logger.info("Access token obtained")
            
        except Exception as e:
            logger.error(f"Error getting access token: {e}")
            raise
    
    async def _make_request(self, method: str, endpoint: str, params: Dict[str, Any] = None, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make API request to TD Ameritrade"""
        url = f"{self.base_url}{endpoint}"
        
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        
        try:
            if method == 'GET':
                async with self.session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        raise Exception(f"Request failed with status {response.status}")
            elif method == 'POST':
                async with self.session.post(url, json=data, headers=headers) as response:
                    if response.status in [200, 201]:
                        return await response.json()
                    else:
                        raise Exception(f"Request failed with status {response.status}")
            elif method == 'DELETE':
                async with self.session.delete(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        raise Exception(f"Request failed with status {response.status}")
        except Exception as e:
            logger.error(f"Error making request to {url}: {e}")
            raise
    
    async def get_account_info(self) -> AccountInfo:
        """Get account information"""
        try:
            data = await self._make_request('GET', '/v1/accounts')
            
            if data and len(data) > 0:
                account = data[0]
                securities_account = account['securitiesAccount']
                
                return AccountInfo(
                    account_id=account['securitiesAccount']['accountId'],
                    cash=float(securities_account['currentBalances']['cashBalance']),
                    portfolio_value=float(securities_account['currentBalances']['liquidationValue']),
                    buying_power=float(securities_account['currentBalances']['buyingPower']),
                    day_trading_buying_power=float(securities_account['currentBalances']['dayTradingBuyingPower'])
                )
            else:
                raise Exception("No accounts found")
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            raise
    
    async def get_positions(self) -> List[Position]:
        """Get all positions"""
        try:
            data = await self._make_request('GET', '/v1/accounts', params={'fields': 'positions'})
            
            positions = []
            if data and len(data) > 0:
                account = data[0]
                securities_account = account['securitiesAccount']
                
                for pos in securities_account.get('positions', []):
                    instrument = pos['instrument']
                    positions.append(Position(
                        symbol=instrument['symbol'],
                        quantity=float(pos['longQuantity']) - float(pos['shortQuantity']),
                        average_price=float(pos['averagePrice']),
                        current_price=float(pos['marketValue']) / float(pos['longQuantity']) if float(pos['longQuantity']) > 0 else 0,
                        unrealized_pnl=float(pos['currentDayProfitLoss']),
                        realized_pnl=0.0  # Not directly available
                    ))
            
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
            # TD Ameritrade order format
            order_data = {
                "orderType": self._convert_order_type(order_request.order_type),
                "session": "NORMAL",
                "duration": "DAY",
                "orderStrategyType": "SINGLE",
                "orderLegCollection": [
                    {
                        "instruction": order_request.side.value.upper(),
                        "quantity": order_request.quantity,
                        "instrument": {
                            "symbol": order_request.symbol,
                            "assetType": "EQUITY"
                        }
                    }
                ]
            }
            
            if order_request.price is not None:
                order_data["price"] = order_request.price
            
            if order_request.stop_price is not None:
                order_data["stopPrice"] = order_request.stop_price
            
            # Get account ID
            accounts = await self._make_request('GET', '/v1/accounts')
            account_id = accounts[0]['securitiesAccount']['accountId']
            
            # Place order
            response = await self._make_request('POST', f'/v1/accounts/{account_id}/orders', data=order_data)
            
            # TD Ameritrade returns order ID in Location header, but we'll simulate for now
            order_id = f"td_{int(datetime.now().timestamp())}"
            
            return OrderResponse(
                order_id=order_id,
                exchange_order_id=order_id,
                symbol=order_request.symbol,
                side=order_request.side,
                order_type=order_request.order_type,
                quantity=order_request.quantity,
                price=order_request.price,
                status=OrderStatus.PENDING,
                filled_quantity=0.0,
                filled_price=None,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            # Get account ID
            accounts = await self._make_request('GET', '/v1/accounts')
            account_id = accounts[0]['securitiesAccount']['accountId']
            
            await self._make_request('DELETE', f'/v1/accounts/{account_id}/orders/{order_id}')
            return True
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def get_order(self, order_id: str) -> Optional[OrderResponse]:
        """Get order status"""
        try:
            # Get account ID
            accounts = await self._make_request('GET', '/v1/accounts')
            account_id = accounts[0]['securitiesAccount']['accountId']
            
            data = await self._make_request('GET', f'/v1/accounts/{account_id}/orders/{order_id}')
            
            if data:
                order_leg = data['orderLegCollection'][0]
                
                return OrderResponse(
                    order_id=str(data['orderId']),
                    exchange_order_id=str(data['orderId']),
                    symbol=order_leg['instrument']['symbol'],
                    side=OrderSide(order_leg['instruction'].lower()),
                    order_type=self._convert_from_td_order_type(data['orderType']),
                    quantity=float(order_leg['quantity']),
                    price=float(data.get('price', 0)),
                    status=self._convert_order_status(data['status']),
                    filled_quantity=float(data.get('filledQuantity', 0)),
                    filled_price=float(data.get('averageFillPrice', 0)) if data.get('averageFillPrice') else None,
                    created_at=datetime.fromisoformat(data['enteredTime'].replace('Z', '+00:00')),
                    updated_at=datetime.fromisoformat(data['enteredTime'].replace('Z', '+00:00'))
                )
            else:
                return None
        except Exception as e:
            logger.error(f"Error getting order {order_id}: {e}")
            return None
    
    async def get_orders(self, symbol: Optional[str] = None, status: Optional[OrderStatus] = None) -> List[OrderResponse]:
        """Get orders"""
        try:
            # Get account ID
            accounts = await self._make_request('GET', '/v1/accounts')
            account_id = accounts[0]['securitiesAccount']['accountId']
            
            # Get orders from last 7 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            params = {
                'fromEnteredTime': start_date.strftime('%Y-%m-%d'),
                'toEnteredTime': end_date.strftime('%Y-%m-%d')
            }
            
            data = await self._make_request('GET', f'/v1/accounts/{account_id}/orders', params=params)
            
            orders = []
            for order in data:
                order_leg = order['orderLegCollection'][0]
                order_status = self._convert_order_status(order['status'])
                order_symbol = order_leg['instrument']['symbol']
                
                if (symbol is None or order_symbol == symbol) and (status is None or order_status == status):
                    orders.append(OrderResponse(
                        order_id=str(order['orderId']),
                        exchange_order_id=str(order['orderId']),
                        symbol=order_symbol,
                        side=OrderSide(order_leg['instruction'].lower()),
                        order_type=self._convert_from_td_order_type(order['orderType']),
                        quantity=float(order_leg['quantity']),
                        price=float(order.get('price', 0)),
                        status=order_status,
                        filled_quantity=float(order.get('filledQuantity', 0)),
                        filled_price=float(order.get('averageFillPrice', 0)) if order.get('averageFillPrice') else None,
                        created_at=datetime.fromisoformat(order['enteredTime'].replace('Z', '+00:00')),
                        updated_at=datetime.fromisoformat(order['enteredTime'].replace('Z', '+00:00'))
                    ))
            
            return orders
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            raise
    
    async def get_market_data(self, symbol: str, timeframe: str = "1m", limit: int = 100) -> List[MarketDataPoint]:
        """Get market data"""
        try:
            # TD Ameritrade uses different endpoint for historical data
            period_type = "day"
            frequency_type = "minute"
            frequency = self._convert_timeframe(timeframe)
            
            params = {
                'periodType': period_type,
                'frequencyType': frequency_type,
                'frequency': frequency,
                'needExtendedHoursData': 'false'
            }
            
            data = await self._make_request('GET', f'/v1/marketdata/{symbol}/pricehistory', params=params)
            
            bars = []
            if 'candles' in data:
                for candle in data['candles']:
                    bars.append(MarketDataPoint(
                        symbol=symbol,
                        timestamp=datetime.fromtimestamp(candle['datetime'] / 1000),
                        open=float(candle['open']),
                        high=float(candle['high']),
                        low=float(candle['low']),
                        close=float(candle['close']),
                        volume=float(candle['volume'])
                    ))
            
            return bars[-limit:]  # Return last 'limit' bars
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            raise
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get ticker data"""
        try:
            data = await self._make_request('GET', f'/v1/marketdata/{symbol}/quotes')
            
            if symbol in data:
                quote = data[symbol]
                return {
                    'symbol': symbol,
                    'price': float(quote['lastPrice']),
                    'bid': float(quote['bidPrice']),
                    'ask': float(quote['askPrice']),
                    'volume': float(quote['totalVolume']),
                    'timestamp': datetime.fromtimestamp(quote['quoteTimeInLong'] / 1000)
                }
            else:
                raise Exception(f"No data for symbol {symbol}")
        except Exception as e:
            logger.error(f"Error getting ticker for {symbol}: {e}")
            raise
    
    async def get_supported_symbols(self) -> List[str]:
        """Get supported symbols"""
        try:
            # TD Ameritrade doesn't have a direct endpoint for all symbols
            # Would need to search or get from instruments endpoint
            # For now, return a sample list
            return ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "SPY", "QQQ"]
        except Exception as e:
            logger.error(f"Error getting supported symbols: {e}")
            raise
    
    async def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol is supported"""
        try:
            # Try to get quote for symbol
            await self.get_ticker(symbol)
            return True
        except Exception:
            return False
    
    async def get_fees(self) -> Dict[str, Any]:
        """Get trading fees"""
        # TD Ameritrade fee structure
        return {
            'stock_commission': 0.0,  # Commission-free stock trades
            'option_commission': 0.65,  # Per contract
            'option_base_fee': 0.0,
            'futures_commission': 2.25  # Per contract
        }
    
    def _convert_order_type(self, order_type: OrderType) -> str:
        """Convert order type to TD Ameritrade format"""
        mapping = {
            OrderType.MARKET: 'MARKET',
            OrderType.LIMIT: 'LIMIT',
            OrderType.STOP_LOSS: 'STOP',
            OrderType.STOP_LIMIT: 'STOP_LIMIT'
        }
        return mapping.get(order_type, 'MARKET')
    
    def _convert_from_td_order_type(self, td_type: str) -> OrderType:
        """Convert from TD Ameritrade order type"""
        mapping = {
            'MARKET': OrderType.MARKET,
            'LIMIT': OrderType.LIMIT,
            'STOP': OrderType.STOP_LOSS,
            'STOP_LIMIT': OrderType.STOP_LIMIT
        }
        return mapping.get(td_type, OrderType.MARKET)
    
    def _convert_order_status(self, td_status: str) -> OrderStatus:
        """Convert TD Ameritrade order status"""
        mapping = {
            'AWAITING_PARENT_ORDER': OrderStatus.PENDING,
            'AWAITING_CONDITION': OrderStatus.PENDING,
            'AWAITING_MANUAL_REVIEW': OrderStatus.PENDING,
            'ACCEPTED': OrderStatus.PENDING,
            'AWAITING_UR_OUT': OrderStatus.PENDING,
            'PENDING_ACTIVATION': OrderStatus.PENDING,
            'QUEUED': OrderStatus.PENDING,
            'WORKING': OrderStatus.PENDING,
            'REJECTED': OrderStatus.REJECTED,
            'PENDING_CANCEL': OrderStatus.PENDING,
            'CANCELED': OrderStatus.CANCELLED,
            'PENDING_REPLACE': OrderStatus.PENDING,
            'REPLACED': OrderStatus.PENDING,
            'FILLED': OrderStatus.FILLED,
            'EXPIRED': OrderStatus.CANCELLED
        }
        return mapping.get(td_status, OrderStatus.PENDING)
    
    def _convert_timeframe(self, timeframe: str) -> int:
        """Convert timeframe to TD Ameritrade format"""
        mapping = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '1d': 1440
        }
        return mapping.get(timeframe, 1)
