"""
Multi-Exchange Manager for the Advanced AI Trading System
Provides unified interface for Alpaca, Binance, TD Ameritrade, KuCoin, and other exchanges
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import ccxt
from abc import ABC, abstractmethod

from config import Config, ExchangeConfig
from exchanges.alpaca_exchange import AlpacaExchange
from exchanges.binance_exchange import BinanceExchange
from exchanges.td_ameritrade_exchange import TDAmeritradeExchange
from exchanges.kucoin_exchange import KuCoinExchange
from exchanges.base_exchange import BaseExchange
from utils.logger import get_logger

logger = get_logger(__name__)

class ExchangeStatus(Enum):
    """Exchange connection status"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"

@dataclass
class ExchangeInfo:
    """Information about an exchange"""
    name: str
    status: ExchangeStatus
    supported_assets: List[str]
    active_pairs: List[str]
    last_heartbeat: datetime
    error_message: Optional[str] = None

class ExchangeManager:
    """
    Manages connections and operations across multiple exchanges
    Provides unified interface following Freqtrade patterns
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.exchanges: Dict[str, BaseExchange] = {}
        self.exchange_info: Dict[str, ExchangeInfo] = {}
        self.data_feeds: Dict[str, asyncio.Task] = {}
        self.market_data_cache: Dict[str, pd.DataFrame] = {}
        self.price_cache: Dict[str, float] = {}
        self.heartbeat_tasks: Dict[str, asyncio.Task] = {}
        
        # Connection monitoring
        self.connection_monitor_task: Optional[asyncio.Task] = None
        self.monitoring_active = False
        
        logger.info("ExchangeManager initialized")
    
    async def initialize(self):
        """Initialize all configured exchanges"""
        try:
            logger.info("Initializing exchange connections...")
            
            # Initialize enabled exchanges
            for exchange_name, exchange_config in self.config.exchanges.items():
                if exchange_config.enabled:
                    await self._initialize_exchange(exchange_name, exchange_config)
            
            # Start connection monitoring
            await self._start_connection_monitoring()
            
            logger.info(f"✅ Initialized {len(self.exchanges)} exchanges")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize exchanges: {e}")
            raise
    
    async def _initialize_exchange(self, exchange_name: str, exchange_config: ExchangeConfig):
        """Initialize a specific exchange"""
        try:
            logger.info(f"Initializing {exchange_name}...")
            
            # Create exchange instance
            exchange_instance = self._create_exchange_instance(exchange_name, exchange_config)
            
            # Initialize the exchange
            await exchange_instance.initialize()
            
            # Test connection
            connection_test = await exchange_instance.test_connection()
            if not connection_test['success']:
                raise Exception(f"Connection test failed: {connection_test['error']}")
            
            # Store exchange
            self.exchanges[exchange_name] = exchange_instance
            
            # Update exchange info
            self.exchange_info[exchange_name] = ExchangeInfo(
                name=exchange_name,
                status=ExchangeStatus.CONNECTED,
                supported_assets=exchange_config.supported_assets,
                active_pairs=await exchange_instance.get_active_pairs(),
                last_heartbeat=datetime.utcnow()
            )
            
            # Start heartbeat monitoring
            self.heartbeat_tasks[exchange_name] = asyncio.create_task(
                self._heartbeat_monitor(exchange_name)
            )
            
            logger.info(f"✅ {exchange_name} initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize {exchange_name}: {e}")
            
            # Update exchange info with error
            self.exchange_info[exchange_name] = ExchangeInfo(
                name=exchange_name,
                status=ExchangeStatus.ERROR,
                supported_assets=[],
                active_pairs=[],
                last_heartbeat=datetime.utcnow(),
                error_message=str(e)
            )
    
    def _create_exchange_instance(self, exchange_name: str, exchange_config: ExchangeConfig) -> BaseExchange:
        """Create an exchange instance based on the exchange type"""
        if exchange_name == "alpaca":
            return AlpacaExchange(exchange_config)
        elif exchange_name == "binance":
            return BinanceExchange(exchange_config)
        elif exchange_name == "td_ameritrade":
            return TDAmeritradeExchange(exchange_config)
        elif exchange_name == "kucoin":
            return KuCoinExchange(exchange_config)
        else:
            raise ValueError(f"Unsupported exchange: {exchange_name}")
    
    async def _start_connection_monitoring(self):
        """Start connection monitoring for all exchanges"""
        self.monitoring_active = True
        self.connection_monitor_task = asyncio.create_task(self._monitor_connections())
    
    async def _monitor_connections(self):
        """Monitor exchange connections"""
        while self.monitoring_active:
            try:
                for exchange_name, exchange in self.exchanges.items():
                    # Check connection status
                    connection_status = await exchange.test_connection()
                    
                    current_info = self.exchange_info[exchange_name]
                    
                    if connection_status['success']:
                        current_info.status = ExchangeStatus.CONNECTED
                        current_info.last_heartbeat = datetime.utcnow()
                        current_info.error_message = None
                    else:
                        current_info.status = ExchangeStatus.ERROR
                        current_info.error_message = connection_status['error']
                        
                        # Attempt reconnection
                        logger.warning(f"Connection lost to {exchange_name}, attempting reconnection...")
                        await self._reconnect_exchange(exchange_name)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Error in connection monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _heartbeat_monitor(self, exchange_name: str):
        """Monitor heartbeat for a specific exchange"""
        while self.monitoring_active:
            try:
                exchange = self.exchanges[exchange_name]
                heartbeat = await exchange.heartbeat()
                
                if heartbeat['success']:
                    self.exchange_info[exchange_name].last_heartbeat = datetime.utcnow()
                else:
                    logger.warning(f"Heartbeat failed for {exchange_name}: {heartbeat['error']}")
                
                await asyncio.sleep(60)  # Heartbeat every minute
                
            except Exception as e:
                logger.error(f"❌ Heartbeat error for {exchange_name}: {e}")
                await asyncio.sleep(60)
    
    async def _reconnect_exchange(self, exchange_name: str):
        """Reconnect to an exchange"""
        try:
            logger.info(f"Reconnecting to {exchange_name}...")
            
            exchange_config = self.config.exchanges[exchange_name]
            exchange = self.exchanges[exchange_name]
            
            # Attempt reconnection
            await exchange.reconnect()
            
            # Test connection
            connection_test = await exchange.test_connection()
            if connection_test['success']:
                self.exchange_info[exchange_name].status = ExchangeStatus.CONNECTED
                self.exchange_info[exchange_name].error_message = None
                logger.info(f"✅ Reconnected to {exchange_name}")
            else:
                raise Exception(connection_test['error'])
                
        except Exception as e:
            logger.error(f"❌ Failed to reconnect to {exchange_name}: {e}")
            self.exchange_info[exchange_name].status = ExchangeStatus.ERROR
            self.exchange_info[exchange_name].error_message = str(e)
    
    async def get_available_exchanges(self) -> List[Dict[str, Any]]:
        """Get list of available exchanges"""
        exchanges = []
        
        for exchange_name, info in self.exchange_info.items():
            exchanges.append({
                'name': exchange_name,
                'status': info.status.value,
                'supported_assets': info.supported_assets,
                'active_pairs': info.active_pairs,
                'last_heartbeat': info.last_heartbeat.isoformat(),
                'error_message': info.error_message
            })
        
        return exchanges
    
    async def is_exchange_available(self, exchange_name: str) -> bool:
        """Check if an exchange is available"""
        if exchange_name not in self.exchanges:
            return False
        
        info = self.exchange_info.get(exchange_name)
        return info and info.status == ExchangeStatus.CONNECTED
    
    async def is_symbol_available(self, exchange_name: str, symbol: str) -> bool:
        """Check if a symbol is available on an exchange"""
        if not await self.is_exchange_available(exchange_name):
            return False
        
        try:
            exchange = self.exchanges[exchange_name]
            return await exchange.is_symbol_available(symbol)
        except Exception as e:
            logger.error(f"❌ Error checking symbol availability: {e}")
            return False
    
    async def get_current_price(self, exchange_name: str, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        cache_key = f"{exchange_name}:{symbol}"
        
        # Check cache first
        if cache_key in self.price_cache:
            return self.price_cache[cache_key]
        
        if not await self.is_exchange_available(exchange_name):
            return None
        
        try:
            exchange = self.exchanges[exchange_name]
            price = await exchange.get_current_price(symbol)
            
            # Cache the price
            self.price_cache[cache_key] = price
            
            return price
            
        except Exception as e:
            logger.error(f"❌ Error getting current price: {e}")
            return None
    
    async def get_historical_data(self, exchange_name: str, symbol: str, 
                                 timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """Get historical OHLCV data"""
        if not await self.is_exchange_available(exchange_name):
            return pd.DataFrame()
        
        try:
            exchange = self.exchanges[exchange_name]
            return await exchange.get_historical_data(symbol, timeframe, limit)
            
        except Exception as e:
            logger.error(f"❌ Error getting historical data: {e}")
            return pd.DataFrame()
    
    async def get_latest_ohlcv(self, exchange_name: str, symbol: str, 
                              timeframe: str = '1m') -> pd.DataFrame:
        """Get latest OHLCV data"""
        if not await self.is_exchange_available(exchange_name):
            return pd.DataFrame()
        
        try:
            exchange = self.exchanges[exchange_name]
            return await exchange.get_latest_ohlcv(symbol, timeframe)
            
        except Exception as e:
            logger.error(f"❌ Error getting latest OHLCV: {e}")
            return pd.DataFrame()
    
    async def place_order(self, exchange_name: str, symbol: str, order_type: str,
                         side: str, quantity: float, price: Optional[float] = None,
                         params: Optional[Dict] = None) -> Dict[str, Any]:
        """Place an order on an exchange"""
        if not await self.is_exchange_available(exchange_name):
            return {
                'success': False,
                'error': f'Exchange {exchange_name} is not available'
            }
        
        try:
            exchange = self.exchanges[exchange_name]
            return await exchange.place_order(symbol, order_type, side, quantity, price, params)
            
        except Exception as e:
            logger.error(f"❌ Error placing order: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def cancel_order(self, exchange_name: str, order_id: str, symbol: str) -> Dict[str, Any]:
        """Cancel an order"""
        if not await self.is_exchange_available(exchange_name):
            return {
                'success': False,
                'error': f'Exchange {exchange_name} is not available'
            }
        
        try:
            exchange = self.exchanges[exchange_name]
            return await exchange.cancel_order(order_id, symbol)
            
        except Exception as e:
            logger.error(f"❌ Error canceling order: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_order_status(self, exchange_name: str, order_id: str, symbol: str) -> Dict[str, Any]:
        """Get order status"""
        if not await self.is_exchange_available(exchange_name):
            return {
                'success': False,
                'error': f'Exchange {exchange_name} is not available'
            }
        
        try:
            exchange = self.exchanges[exchange_name]
            return await exchange.get_order_status(order_id, symbol)
            
        except Exception as e:
            logger.error(f"❌ Error getting order status: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_account_balance(self, exchange_name: str) -> Dict[str, Any]:
        """Get account balance"""
        if not await self.is_exchange_available(exchange_name):
            return {
                'success': False,
                'error': f'Exchange {exchange_name} is not available'
            }
        
        try:
            exchange = self.exchanges[exchange_name]
            return await exchange.get_account_balance()
            
        except Exception as e:
            logger.error(f"❌ Error getting account balance: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_open_positions(self, exchange_name: str) -> List[Dict[str, Any]]:
        """Get open positions"""
        if not await self.is_exchange_available(exchange_name):
            return []
        
        try:
            exchange = self.exchanges[exchange_name]
            return await exchange.get_open_positions()
            
        except Exception as e:
            logger.error(f"❌ Error getting open positions: {e}")
            return []
    
    async def start_data_feeds(self):
        """Start real-time data feeds for all exchanges"""
        for exchange_name, exchange in self.exchanges.items():
            if self.exchange_info[exchange_name].status == ExchangeStatus.CONNECTED:
                self.data_feeds[exchange_name] = asyncio.create_task(
                    self._start_data_feed(exchange_name)
                )
    
    async def _start_data_feed(self, exchange_name: str):
        """Start data feed for a specific exchange"""
        try:
            exchange = self.exchanges[exchange_name]
            
            async for data in exchange.get_real_time_data():
                # Process real-time data
                await self._process_real_time_data(exchange_name, data)
                
        except Exception as e:
            logger.error(f"❌ Error in data feed for {exchange_name}: {e}")
    
    async def _process_real_time_data(self, exchange_name: str, data: Dict[str, Any]):
        """Process real-time data from an exchange"""
        try:
            symbol = data.get('symbol')
            price = data.get('price')
            
            if symbol and price:
                cache_key = f"{exchange_name}:{symbol}"
                self.price_cache[cache_key] = price
                
                # Update market data cache
                if cache_key not in self.market_data_cache:
                    self.market_data_cache[cache_key] = pd.DataFrame()
                
                # Add new data point
                new_data = pd.DataFrame([data])
                self.market_data_cache[cache_key] = pd.concat([
                    self.market_data_cache[cache_key], new_data
                ]).tail(1000)  # Keep last 1000 data points
                
        except Exception as e:
            logger.error(f"❌ Error processing real-time data: {e}")
    
    async def stop(self):
        """Stop exchange manager"""
        try:
            logger.info("Stopping exchange manager...")
            
            # Stop monitoring
            self.monitoring_active = False
            
            # Cancel connection monitor
            if self.connection_monitor_task:
                self.connection_monitor_task.cancel()
            
            # Cancel heartbeat tasks
            for task in self.heartbeat_tasks.values():
                task.cancel()
            
            # Cancel data feeds
            for task in self.data_feeds.values():
                task.cancel()
            
            # Disconnect exchanges
            for exchange in self.exchanges.values():
                await exchange.disconnect()
            
            logger.info("✅ Exchange manager stopped")
            
        except Exception as e:
            logger.error(f"❌ Error stopping exchange manager: {e}")
    
    def is_connected(self) -> bool:
        """Check if any exchange is connected"""
        return any(
            info.status == ExchangeStatus.CONNECTED 
            for info in self.exchange_info.values()
        )
    
    async def get_connection_status(self) -> Dict[str, str]:
        """Get connection status for all exchanges"""
        return {
            name: info.status.value 
            for name, info in self.exchange_info.items()
        }
