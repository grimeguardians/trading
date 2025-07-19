"""
Alpaca Markets Integration
Primary exchange integration with comprehensive trading capabilities
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from config import ExchangeConfig
from core.exchange_abstraction import ExchangeConnector, OrderResult, MarketData

class AlpacaConnector(ExchangeConnector):
    """
    Alpaca Markets integration for stocks, crypto, ETFs, options, and futures
    """
    
    def __init__(self, config: ExchangeConfig):
        super().__init__(config)
        self.trading_client = None
        self.market_data_client = None
        self.crypto_client = None
        self.options_client = None
        
    async def initialize(self) -> bool:
        """Initialize Alpaca connection"""
        try:
            # Import Alpaca SDK
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
            from alpaca.data.live import StockDataStream, CryptoDataStream
            from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
            
            # Store enums for later use
            self.OrderSide = OrderSide
            self.OrderType = OrderType
            self.TimeInForce = TimeInForce
            
            # Initialize trading client
            self.trading_client = TradingClient(
                api_key=self.config.api_key,
                secret_key=self.config.secret_key,
                paper=self.config.sandbox
            )
            
            # Initialize market data clients
            self.market_data_client = StockHistoricalDataClient(
                api_key=self.config.api_key,
                secret_key=self.config.secret_key
            )
            
            self.crypto_client = CryptoHistoricalDataClient(
                api_key=self.config.api_key,
                secret_key=self.config.secret_key
            )
            
            # Test connection
            account = self.trading_client.get_account()
            self.logger.info(f"✅ Alpaca connected - Account: {account.id}")
            self.logger.info(f"💰 Buying Power: ${float(account.buying_power):,.2f}")
            self.logger.info(f"📊 Portfolio Value: ${float(account.portfolio_value):,.2f}")
            
            self.connected = True
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Alpaca initialization failed: {e}")
            return False
            
    async def place_order(self, symbol: str, side: str, quantity: float,
                         order_type: str = "market", price: float = None,
                         stop_loss: float = None, take_profit: float = None) -> OrderResult:
        """Place order on Alpaca"""
        try:
            if not self._check_rate_limit("place_order"):
                raise Exception("Rate limit exceeded")
                
            from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
            from alpaca.trading.requests import StopLossRequest, TakeProfitRequest
            
            # Convert side to Alpaca enum
            alpaca_side = self.OrderSide.BUY if side.lower() == "buy" else self.OrderSide.SELL
            
            # Prepare order request
            order_data = {
                "symbol": symbol,
                "qty": quantity,
                "side": alpaca_side,
                "time_in_force": self.TimeInForce.DAY
            }
            
            # Add stop loss and take profit if provided
            if stop_loss:
                order_data["stop_loss"] = StopLossRequest(stop_price=stop_loss)
                
            if take_profit:
                order_data["take_profit"] = TakeProfitRequest(limit_price=take_profit)
                
            # Create order request based on type
            if order_type.lower() == "market":
                order_request = MarketOrderRequest(**order_data)
            elif order_type.lower() == "limit":
                if price is None:
                    raise ValueError("Price required for limit order")
                order_data["limit_price"] = price
                order_request = LimitOrderRequest(**order_data)
            else:
                raise ValueError(f"Unsupported order type: {order_type}")
                
            # Submit order
            order = self.trading_client.submit_order(order_request)
            
            # Convert to standardized result
            result = OrderResult(
                order_id=str(order.id),
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=float(order.limit_price) if order.limit_price else 0.0,
                status=order.status.value,
                filled_price=float(order.filled_avg_price) if order.filled_avg_price else None,
                filled_quantity=float(order.filled_qty) if order.filled_qty else None,
                timestamp=order.submitted_at,
                exchange=self.config.name
            )
            
            self.logger.info(f"📈 Alpaca order placed: {symbol} {side} {quantity} - ID: {order.id}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Alpaca order failed: {e}")
            raise
            
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order on Alpaca"""
        try:
            self.trading_client.cancel_order_by_id(order_id)
            self.logger.info(f"❌ Alpaca order cancelled: {order_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Alpaca cancel order failed: {e}")
            return False
            
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions from Alpaca"""
        try:
            positions = self.trading_client.get_all_positions()
            
            result = []
            for position in positions:
                result.append({
                    "symbol": position.symbol,
                    "quantity": float(position.qty),
                    "side": "long" if float(position.qty) > 0 else "short",
                    "market_value": float(position.market_value),
                    "cost_basis": float(position.cost_basis),
                    "unrealized_pnl": float(position.unrealized_pnl),
                    "unrealized_pnl_percent": float(position.unrealized_plpc),
                    "current_price": float(position.current_price),
                    "exchange": self.config.name
                })
                
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Alpaca get positions failed: {e}")
            return []
            
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information from Alpaca"""
        try:
            account = self.trading_client.get_account()
            
            return {
                "id": account.id,
                "buying_power": float(account.buying_power),
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "long_market_value": float(account.long_market_value),
                "short_market_value": float(account.short_market_value),
                "equity": float(account.equity),
                "last_equity": float(account.last_equity),
                "multiplier": float(account.multiplier),
                "currency": account.currency,
                "status": account.status.value,
                "pattern_day_trader": account.pattern_day_trader,
                "day_trade_count": account.day_trade_count,
                "daytrade_buying_power": float(account.daytrade_buying_power),
                "regt_buying_power": float(account.regt_buying_power),
                "exchange": self.config.name
            }
            
        except Exception as e:
            self.logger.error(f"❌ Alpaca account info failed: {e}")
            return {}
            
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from Alpaca"""
        try:
            from alpaca.data.requests import StockLatestQuoteRequest
            
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quote = self.market_data_client.get_stock_latest_quote(request)
            
            if symbol in quote:
                return float(quote[symbol].ask_price)
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Alpaca price fetch failed for {symbol}: {e}")
            return None
            
    async def get_historical_data(self, symbol: str, timeframe: str,
                                 limit: int = 100) -> Optional[pd.DataFrame]:
        """Get historical data from Alpaca"""
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            
            # Convert timeframe to Alpaca format
            timeframe_mapping = {
                "1m": TimeFrame.Minute,
                "5m": TimeFrame(5, "Minute"),
                "15m": TimeFrame(15, "Minute"),
                "30m": TimeFrame(30, "Minute"),
                "1h": TimeFrame.Hour,
                "1d": TimeFrame.Day,
                "1w": TimeFrame.Week,
                "1M": TimeFrame.Month
            }
            
            alpaca_timeframe = timeframe_mapping.get(timeframe, TimeFrame.Day)
            
            # Calculate start time based on limit
            end_time = datetime.now()
            if timeframe == "1d":
                start_time = end_time - timedelta(days=limit)
            elif timeframe == "1h":
                start_time = end_time - timedelta(hours=limit)
            elif timeframe == "1m":
                start_time = end_time - timedelta(minutes=limit)
            else:
                start_time = end_time - timedelta(days=limit)
                
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=alpaca_timeframe,
                start=start_time,
                end=end_time
            )
            
            bars = self.market_data_client.get_stock_bars(request)
            
            if symbol in bars:
                data = []
                for bar in bars[symbol]:
                    data.append({
                        "timestamp": bar.timestamp,
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": int(bar.volume)
                    })
                    
                df = pd.DataFrame(data)
                return df.tail(limit)  # Return only requested number of bars
                
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Alpaca historical data failed for {symbol}: {e}")
            return None
            
    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get real-time market data from Alpaca"""
        try:
            from alpaca.data.requests import StockLatestQuoteRequest
            
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quote = self.market_data_client.get_stock_latest_quote(request)
            
            if symbol in quote:
                quote_data = quote[symbol]
                return MarketData(
                    symbol=symbol,
                    price=float(quote_data.ask_price),
                    bid=float(quote_data.bid_price),
                    ask=float(quote_data.ask_price),
                    volume=int(quote_data.ask_size + quote_data.bid_size),
                    timestamp=quote_data.timestamp,
                    exchange=self.config.name
                )
                
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Alpaca market data failed for {symbol}: {e}")
            return None
            
    async def get_crypto_data(self, symbol: str) -> Optional[MarketData]:
        """Get cryptocurrency data from Alpaca"""
        try:
            from alpaca.data.requests import CryptoLatestQuoteRequest
            
            request = CryptoLatestQuoteRequest(symbol_or_symbols=symbol)
            quote = self.crypto_client.get_crypto_latest_quote(request)
            
            if symbol in quote:
                quote_data = quote[symbol]
                return MarketData(
                    symbol=symbol,
                    price=float(quote_data.ask_price),
                    bid=float(quote_data.bid_price),
                    ask=float(quote_data.ask_price),
                    volume=int(quote_data.ask_size + quote_data.bid_size),
                    timestamp=quote_data.timestamp,
                    exchange=self.config.name
                )
                
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Alpaca crypto data failed for {symbol}: {e}")
            return None
            
    async def place_crypto_order(self, symbol: str, side: str, quantity: float,
                               order_type: str = "market") -> OrderResult:
        """Place cryptocurrency order on Alpaca"""
        try:
            from alpaca.trading.requests import MarketOrderRequest
            
            # Convert side to Alpaca enum
            alpaca_side = self.OrderSide.BUY if side.lower() == "buy" else self.OrderSide.SELL
            
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=alpaca_side,
                time_in_force=self.TimeInForce.DAY
            )
            
            order = self.trading_client.submit_order(order_request)
            
            result = OrderResult(
                order_id=str(order.id),
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=float(order.limit_price) if order.limit_price else 0.0,
                status=order.status.value,
                filled_price=float(order.filled_avg_price) if order.filled_avg_price else None,
                filled_quantity=float(order.filled_qty) if order.filled_qty else None,
                timestamp=order.submitted_at,
                exchange=self.config.name
            )
            
            self.logger.info(f"₿ Alpaca crypto order placed: {symbol} {side} {quantity}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Alpaca crypto order failed: {e}")
            raise
            
    async def get_options_chain(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get options chain from Alpaca (placeholder for future implementation)"""
        try:
            # This would require options data feed from Alpaca
            # For now, return placeholder
            self.logger.info(f"📋 Options chain requested for {symbol} (not implemented)")
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Options chain failed for {symbol}: {e}")
            return None
            
    async def place_options_order(self, symbol: str, side: str, quantity: int,
                                option_type: str, strike: float, expiration: str) -> OrderResult:
        """Place options order on Alpaca (placeholder for future implementation)"""
        try:
            # This would require options trading capability
            self.logger.info(f"📋 Options order requested: {symbol} {option_type} {strike} {expiration}")
            
            # Placeholder result
            result = OrderResult(
                order_id=f"opt_{datetime.now().timestamp()}",
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=strike,
                status="submitted",
                timestamp=datetime.now(),
                exchange=self.config.name
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Options order failed: {e}")
            raise
            
    async def get_portfolio_history(self, period: str = "1D") -> Optional[pd.DataFrame]:
        """Get portfolio performance history"""
        try:
            from alpaca.trading.requests import GetPortfolioHistoryRequest
            
            request = GetPortfolioHistoryRequest(
                period=period,
                timeframe="1Min"
            )
            
            history = self.trading_client.get_portfolio_history(request)
            
            if history:
                data = []
                for i, timestamp in enumerate(history.timestamp):
                    data.append({
                        "timestamp": datetime.fromtimestamp(timestamp),
                        "equity": history.equity[i],
                        "profit_loss": history.profit_loss[i],
                        "profit_loss_pct": history.profit_loss_pct[i] if history.profit_loss_pct else None,
                        "base_value": history.base_value
                    })
                    
                return pd.DataFrame(data)
                
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Portfolio history failed: {e}")
            return None
            
    async def get_watchlist(self, watchlist_name: str = "Primary") -> List[str]:
        """Get watchlist symbols"""
        try:
            watchlists = self.trading_client.get_watchlists()
            
            for watchlist in watchlists:
                if watchlist.name == watchlist_name:
                    return [asset.symbol for asset in watchlist.assets]
                    
            return []
            
        except Exception as e:
            self.logger.error(f"❌ Get watchlist failed: {e}")
            return []
            
    async def health_check(self) -> bool:
        """Check Alpaca connection health"""
        try:
            account = self.trading_client.get_account()
            return account is not None
            
        except Exception as e:
            self.logger.error(f"❌ Alpaca health check failed: {e}")
            return False
            
    def get_market_hours(self) -> Dict[str, Any]:
        """Get market hours information"""
        try:
            clock = self.trading_client.get_clock()
            return {
                "timestamp": clock.timestamp,
                "is_open": clock.is_open,
                "next_open": clock.next_open,
                "next_close": clock.next_close,
                "timezone": "America/New_York"
            }
            
        except Exception as e:
            self.logger.error(f"❌ Market hours failed: {e}")
            return {}
            
    async def get_news(self, symbols: List[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get market news from Alpaca"""
        try:
            from alpaca.data.requests import NewsRequest
            
            request = NewsRequest(
                symbols=symbols,
                limit=limit
            )
            
            news = self.market_data_client.get_news(request)
            
            result = []
            for article in news:
                result.append({
                    "id": article.id,
                    "headline": article.headline,
                    "summary": article.summary,
                    "author": article.author,
                    "created_at": article.created_at,
                    "updated_at": article.updated_at,
                    "url": article.url,
                    "symbols": article.symbols
                })
                
            return result
            
        except Exception as e:
            self.logger.error(f"❌ News fetch failed: {e}")
            return []
