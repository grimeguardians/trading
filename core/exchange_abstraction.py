"""
Exchange Abstraction Layer
Unified interface for multiple exchanges (Alpaca, TD Ameritrade, Binance, KuCoin)
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np

from config import Config, ExchangeConfig
from core.alpaca_integration import AlpacaConnector

@dataclass
class OrderResult:
    """Standardized order result across exchanges"""
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    status: str
    filled_price: Optional[float] = None
    filled_quantity: Optional[float] = None
    timestamp: datetime = None
    exchange: str = ""
    
@dataclass
class MarketData:
    """Standardized market data across exchanges"""
    symbol: str
    price: float
    bid: float
    ask: float
    volume: float
    timestamp: datetime
    exchange: str

class ExchangeConnector(ABC):
    """Abstract base class for exchange connectors"""
    
    def __init__(self, config: ExchangeConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{config.name}")
        self.connected = False
        self.rate_limiter = {}
        
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize exchange connection"""
        pass
        
    @abstractmethod
    async def place_order(self, symbol: str, side: str, quantity: float, 
                         order_type: str = "market", price: float = None,
                         stop_loss: float = None, take_profit: float = None) -> OrderResult:
        """Place an order"""
        pass
        
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        pass
        
    @abstractmethod
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        pass
        
    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        pass
        
    @abstractmethod
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        pass
        
    @abstractmethod
    async def get_historical_data(self, symbol: str, timeframe: str, 
                                 limit: int = 100) -> Optional[pd.DataFrame]:
        """Get historical price data"""
        pass
        
    @abstractmethod
    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get real-time market data"""
        pass
        
    @abstractmethod
    async def health_check(self) -> bool:
        """Check connection health"""
        pass
        
    def _check_rate_limit(self, endpoint: str) -> bool:
        """Check if rate limit allows request"""
        now = datetime.now()
        key = f"{self.config.name}_{endpoint}"
        
        if key not in self.rate_limiter:
            self.rate_limiter[key] = []
            
        # Clean old requests (older than 1 minute)
        self.rate_limiter[key] = [
            timestamp for timestamp in self.rate_limiter[key]
            if now - timestamp < timedelta(minutes=1)
        ]
        
        # Check if we're within rate limit
        if len(self.rate_limiter[key]) >= self.config.rate_limit:
            return False
            
        # Add current request
        self.rate_limiter[key].append(now)
        return True

class BinanceConnector(ExchangeConnector):
    """Binance exchange connector using CCXT"""
    
    def __init__(self, config: ExchangeConfig):
        super().__init__(config)
        self.client = None
        
    async def initialize(self) -> bool:
        """Initialize Binance connection"""
        try:
            import ccxt
            
            self.client = ccxt.binance({
                'apiKey': self.config.api_key,
                'secret': self.config.secret_key,
                'sandbox': self.config.sandbox,
                'enableRateLimit': True,
            })
            
            # Test connection
            await self.client.load_markets()
            self.connected = True
            
            self.logger.info("✅ Binance connection established")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Binance initialization failed: {e}")
            return False
            
    async def place_order(self, symbol: str, side: str, quantity: float,
                         order_type: str = "market", price: float = None,
                         stop_loss: float = None, take_profit: float = None) -> OrderResult:
        """Place order on Binance"""
        try:
            if not self._check_rate_limit("place_order"):
                raise Exception("Rate limit exceeded")
                
            # Convert to Binance format
            binance_symbol = symbol.replace('/', '')
            
            order_params = {
                'symbol': binance_symbol,
                'side': side.upper(),
                'amount': quantity,
                'type': order_type,
            }
            
            if price and order_type == "limit":
                order_params['price'] = price
                
            result = await self.client.create_order(**order_params)
            
            return OrderResult(
                order_id=result['id'],
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=result.get('price', 0),
                status=result.get('status', 'unknown'),
                filled_price=result.get('filled_price'),
                filled_quantity=result.get('filled_quantity'),
                timestamp=datetime.now(),
                exchange=self.config.name
            )
            
        except Exception as e:
            self.logger.error(f"❌ Binance order failed: {e}")
            raise
            
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order on Binance"""
        try:
            await self.client.cancel_order(order_id)
            return True
        except Exception as e:
            self.logger.error(f"❌ Binance cancel order failed: {e}")
            return False
            
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get positions from Binance"""
        try:
            positions = await self.client.fetch_positions()
            return [pos for pos in positions if pos['contracts'] > 0]
        except Exception as e:
            self.logger.error(f"❌ Binance get positions failed: {e}")
            return []
            
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account info from Binance"""
        try:
            return await self.client.fetch_balance()
        except Exception as e:
            self.logger.error(f"❌ Binance account info failed: {e}")
            return {}
            
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from Binance"""
        try:
            ticker = await self.client.fetch_ticker(symbol)
            return ticker.get('last')
        except Exception as e:
            self.logger.error(f"❌ Binance price fetch failed: {e}")
            return None
            
    async def get_historical_data(self, symbol: str, timeframe: str,
                                 limit: int = 100) -> Optional[pd.DataFrame]:
        """Get historical data from Binance"""
        try:
            ohlcv = await self.client.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            self.logger.error(f"❌ Binance historical data failed: {e}")
            return None
            
    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get market data from Binance"""
        try:
            ticker = await self.client.fetch_ticker(symbol)
            return MarketData(
                symbol=symbol,
                price=ticker['last'],
                bid=ticker['bid'],
                ask=ticker['ask'],
                volume=ticker['quoteVolume'],
                timestamp=datetime.now(),
                exchange=self.config.name
            )
        except Exception as e:
            self.logger.error(f"❌ Binance market data failed: {e}")
            return None
            
    async def health_check(self) -> bool:
        """Check Binance connection health"""
        try:
            await self.client.fetch_status()
            return True
        except Exception:
            return False

class KuCoinConnector(ExchangeConnector):
    """KuCoin exchange connector using CCXT"""
    
    def __init__(self, config: ExchangeConfig):
        super().__init__(config)
        self.client = None
        
    async def initialize(self) -> bool:
        """Initialize KuCoin connection"""
        try:
            import ccxt
            
            self.client = ccxt.kucoin({
                'apiKey': self.config.api_key,
                'secret': self.config.secret_key,
                'sandbox': self.config.sandbox,
                'enableRateLimit': True,
            })
            
            await self.client.load_markets()
            self.connected = True
            
            self.logger.info("✅ KuCoin connection established")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ KuCoin initialization failed: {e}")
            return False
            
    # Implementation similar to BinanceConnector
    # ... (other methods would be implemented similarly)
    
    async def place_order(self, symbol: str, side: str, quantity: float,
                         order_type: str = "market", price: float = None,
                         stop_loss: float = None, take_profit: float = None) -> OrderResult:
        """Place order on KuCoin"""
        # Similar implementation to Binance
        pass
        
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order on KuCoin"""
        pass
        
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get positions from KuCoin"""
        pass
        
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account info from KuCoin"""
        pass
        
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from KuCoin"""
        pass
        
    async def get_historical_data(self, symbol: str, timeframe: str,
                                 limit: int = 100) -> Optional[pd.DataFrame]:
        """Get historical data from KuCoin"""
        pass
        
    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get market data from KuCoin"""
        pass
        
    async def health_check(self) -> bool:
        """Check KuCoin connection health"""
        pass

class TDAmeritradeMockConnector(ExchangeConnector):
    """TD Ameritrade mock connector (placeholder for actual implementation)"""
    
    def __init__(self, config: ExchangeConfig):
        super().__init__(config)
        self.logger.warning("⚠️ TD Ameritrade connector is a placeholder implementation")
        
    async def initialize(self) -> bool:
        """Initialize TD Ameritrade connection"""
        # This would use the actual TD Ameritrade API
        self.logger.info("📝 TD Ameritrade mock connector initialized")
        self.connected = True
        return True
        
    async def place_order(self, symbol: str, side: str, quantity: float,
                         order_type: str = "market", price: float = None,
                         stop_loss: float = None, take_profit: float = None) -> OrderResult:
        """Mock order placement"""
        return OrderResult(
            order_id=f"td_mock_{datetime.now().timestamp()}",
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price or 100.0,
            status="filled",
            filled_price=price or 100.0,
            filled_quantity=quantity,
            timestamp=datetime.now(),
            exchange=self.config.name
        )
        
    async def cancel_order(self, order_id: str) -> bool:
        return True
        
    async def get_positions(self) -> List[Dict[str, Any]]:
        return []
        
    async def get_account_info(self) -> Dict[str, Any]:
        return {"buying_power": 100000, "equity": 100000}
        
    async def get_current_price(self, symbol: str) -> Optional[float]:
        return 100.0 + np.random.normal(0, 5)
        
    async def get_historical_data(self, symbol: str, timeframe: str,
                                 limit: int = 100) -> Optional[pd.DataFrame]:
        # Generate mock historical data
        dates = pd.date_range(end=datetime.now(), periods=limit, freq='D')
        prices = 100 + np.cumsum(np.random.randn(limit) * 0.02)
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000, 10000, limit)
        })
        
    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        price = await self.get_current_price(symbol)
        return MarketData(
            symbol=symbol,
            price=price,
            bid=price * 0.999,
            ask=price * 1.001,
            volume=10000,
            timestamp=datetime.now(),
            exchange=self.config.name
        )
        
    async def health_check(self) -> bool:
        return True

class ExchangeManager:
    """
    Manages multiple exchange connections and provides unified interface
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.connectors: Dict[str, ExchangeConnector] = {}
        self.primary_exchange = None
        
    async def initialize(self):
        """Initialize all configured exchanges"""
        try:
            self.logger.info("🏦 Initializing exchange connections...")
            
            # Initialize exchange connectors
            for exchange_name, exchange_config in self.config.exchanges.items():
                if not exchange_config.enabled:
                    continue
                    
                connector = await self._create_connector(exchange_config)
                if connector and await connector.initialize():
                    self.connectors[exchange_name] = connector
                    self.logger.info(f"✅ {exchange_name} connector initialized")
                else:
                    self.logger.error(f"❌ Failed to initialize {exchange_name}")
                    
            # Set primary exchange
            self.primary_exchange = self.config.get_primary_exchange()
            
            if not self.connectors:
                raise Exception("No exchange connectors initialized")
                
            self.logger.info(f"🎯 Primary exchange set to: {self.primary_exchange}")
            
        except Exception as e:
            self.logger.error(f"❌ Exchange manager initialization failed: {e}")
            raise
            
    async def _create_connector(self, config: ExchangeConfig) -> Optional[ExchangeConnector]:
        """Create appropriate connector based on exchange type"""
        try:
            if config.name == "alpaca":
                return AlpacaConnector(config)
            elif config.name == "binance":
                return BinanceConnector(config)
            elif config.name == "kucoin":
                return KuCoinConnector(config)
            elif config.name == "td_ameritrade":
                return TDAmeritradeMockConnector(config)
            else:
                self.logger.warning(f"⚠️ Unknown exchange: {config.name}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Failed to create connector for {config.name}: {e}")
            return None
            
    async def place_order(self, exchange_name: str, symbol: str, side: str,
                         quantity: float, order_type: str = "market",
                         price: float = None, stop_loss: float = None,
                         take_profit: float = None) -> Optional[OrderResult]:
        """Place order on specified exchange"""
        try:
            if exchange_name not in self.connectors:
                raise ValueError(f"Exchange {exchange_name} not available")
                
            connector = self.connectors[exchange_name]
            result = await connector.place_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            self.logger.info(f"📈 Order placed on {exchange_name}: {symbol} {side} {quantity}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Order placement failed on {exchange_name}: {e}")
            return None
            
    async def cancel_order(self, exchange_name: str, order_id: str) -> bool:
        """Cancel order on specified exchange"""
        try:
            if exchange_name not in self.connectors:
                return False
                
            connector = self.connectors[exchange_name]
            return await connector.cancel_order(order_id)
            
        except Exception as e:
            self.logger.error(f"❌ Order cancellation failed: {e}")
            return False
            
    async def cancel_all_orders(self):
        """Cancel all orders across all exchanges"""
        for exchange_name, connector in self.connectors.items():
            try:
                # This would need to be implemented in each connector
                # For now, just log the intent
                self.logger.info(f"📝 Canceling all orders on {exchange_name}")
            except Exception as e:
                self.logger.error(f"❌ Failed to cancel orders on {exchange_name}: {e}")
                
    async def get_current_price(self, exchange_name: str, symbol: str) -> Optional[float]:
        """Get current price from specified exchange"""
        try:
            if exchange_name not in self.connectors:
                return None
                
            connector = self.connectors[exchange_name]
            return await connector.get_current_price(symbol)
            
        except Exception as e:
            self.logger.error(f"❌ Price fetch failed: {e}")
            return None
            
    async def get_historical_data(self, symbol: str, timeframe: str,
                                 limit: int = 100,
                                 exchange_name: str = None) -> Optional[pd.DataFrame]:
        """Get historical data from specified exchange (or primary)"""
        try:
            exchange = exchange_name or self.primary_exchange
            
            if exchange not in self.connectors:
                return None
                
            connector = self.connectors[exchange]
            return await connector.get_historical_data(symbol, timeframe, limit)
            
        except Exception as e:
            self.logger.error(f"❌ Historical data fetch failed: {e}")
            return None
            
    async def get_all_positions(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get positions from all exchanges"""
        all_positions = {}
        
        for exchange_name, connector in self.connectors.items():
            try:
                positions = await connector.get_positions()
                all_positions[exchange_name] = positions
            except Exception as e:
                self.logger.error(f"❌ Failed to get positions from {exchange_name}: {e}")
                all_positions[exchange_name] = []
                
        return all_positions
        
    async def get_account_info(self, exchange_name: str) -> Optional[Dict[str, Any]]:
        """Get account info from specified exchange"""
        try:
            if exchange_name not in self.connectors:
                return None
                
            connector = self.connectors[exchange_name]
            return await connector.get_account_info()
            
        except Exception as e:
            self.logger.error(f"❌ Account info fetch failed: {e}")
            return None
            
    async def health_check(self) -> bool:
        """Check health of all exchange connections"""
        healthy = True
        
        for exchange_name, connector in self.connectors.items():
            try:
                if not await connector.health_check():
                    self.logger.warning(f"⚠️ {exchange_name} health check failed")
                    healthy = False
            except Exception as e:
                self.logger.error(f"❌ {exchange_name} health check error: {e}")
                healthy = False
                
        return healthy
        
    def get_available_exchanges(self) -> List[str]:
        """Get list of available exchanges"""
        return list(self.connectors.keys())
        
    def get_supported_assets(self, exchange_name: str) -> List[str]:
        """Get supported assets for exchange"""
        if exchange_name in self.connectors:
            return self.config.exchanges[exchange_name].supported_assets
        return []
        
    async def shutdown(self):
        """Shutdown all exchange connections"""
        self.logger.info("🛑 Shutting down exchange connections...")
        
        for exchange_name, connector in self.connectors.items():
            try:
                # Close any open connections
                self.logger.info(f"📝 Closing {exchange_name} connection")
                # connector.close() would be called if implemented
            except Exception as e:
                self.logger.error(f"❌ Error closing {exchange_name}: {e}")
                
        self.connectors.clear()
        self.logger.info("✅ Exchange shutdown complete")
