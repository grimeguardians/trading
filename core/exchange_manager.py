"""
Multi-exchange management system
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from abc import ABC, abstractmethod

from config import Config, ExchangeType
from exchanges.alpaca_exchange import AlpacaExchange
from exchanges.binance_exchange import BinanceExchange
from exchanges.kucoin_exchange import KuCoinExchange
from exchanges.td_ameritrade_exchange import TDAmeritradeExchange
from exchanges.base_exchange import BaseExchange
from models import Exchange, SessionLocal

logger = logging.getLogger(__name__)

class ExchangeManager:
    """Manages multiple exchanges and provides unified interface"""
    
    def __init__(self, config: Config):
        self.config = config
        self.exchanges: Dict[ExchangeType, BaseExchange] = {}
        self.active_exchanges: Dict[ExchangeType, bool] = {}
        self.exchange_health: Dict[ExchangeType, Dict[str, Any]] = {}
        
        # Exchange classes mapping
        self.exchange_classes = {
            ExchangeType.ALPACA: AlpacaExchange,
            ExchangeType.BINANCE: BinanceExchange,
            ExchangeType.KUCOIN: KuCoinExchange,
            ExchangeType.TD_AMERITRADE: TDAmeritradeExchange
        }
        
        # Monitoring tasks
        self.monitoring_tasks = set()
        self.running = False
        
        logger.info("ExchangeManager initialized")
    
    async def initialize(self):
        """Initialize all configured exchanges"""
        try:
            # Initialize enabled exchanges
            for exchange_type, exchange_config in self.config.exchanges.items():
                if exchange_config.enabled:
                    await self.add_exchange(exchange_type, exchange_config)
            
            # Start monitoring tasks
            await self.start_monitoring()
            
            self.running = True
            logger.info("ExchangeManager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ExchangeManager: {e}")
            raise
    
    async def add_exchange(self, exchange_type: ExchangeType, exchange_config):
        """Add and initialize an exchange"""
        try:
            # Get exchange class
            exchange_class = self.exchange_classes.get(exchange_type)
            if not exchange_class:
                raise ValueError(f"Unknown exchange type: {exchange_type}")
            
            # Create exchange instance
            exchange = exchange_class(exchange_config)
            
            # Initialize exchange
            await exchange.initialize()
            
            # Store exchange
            self.exchanges[exchange_type] = exchange
            self.active_exchanges[exchange_type] = True
            
            # Update database
            await self.update_exchange_status(exchange_type, "connected")
            
            logger.info(f"Exchange {exchange_config.name} initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize exchange {exchange_type}: {e}")
            self.active_exchanges[exchange_type] = False
            await self.update_exchange_status(exchange_type, "disconnected")
    
    async def start_monitoring(self):
        """Start exchange monitoring tasks"""
        # Health check task
        task = asyncio.create_task(self.monitor_health())
        self.monitoring_tasks.add(task)
        task.add_done_callback(self.monitoring_tasks.discard)
        
        # Reconnection task
        task = asyncio.create_task(self.monitor_connections())
        self.monitoring_tasks.add(task)
        task.add_done_callback(self.monitoring_tasks.discard)
    
    async def monitor_health(self):
        """Monitor exchange health"""
        while self.running:
            try:
                for exchange_type, exchange in self.exchanges.items():
                    if self.active_exchanges.get(exchange_type, False):
                        health = await exchange.get_health_status()
                        self.exchange_health[exchange_type] = health
                        
                        if not health.get("healthy", False):
                            logger.warning(f"Exchange {exchange_type} is unhealthy: {health}")
                            await self.handle_unhealthy_exchange(exchange_type, health)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring exchange health: {e}")
                await asyncio.sleep(5)
    
    async def monitor_connections(self):
        """Monitor and reconnect disconnected exchanges"""
        while self.running:
            try:
                for exchange_type, exchange_config in self.config.exchanges.items():
                    if exchange_config.enabled and not self.active_exchanges.get(exchange_type, False):
                        logger.info(f"Attempting to reconnect to {exchange_type}")
                        await self.add_exchange(exchange_type, exchange_config)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error monitoring connections: {e}")
                await asyncio.sleep(10)
    
    async def handle_unhealthy_exchange(self, exchange_type: ExchangeType, health: Dict[str, Any]):
        """Handle unhealthy exchange"""
        try:
            # Mark as inactive
            self.active_exchanges[exchange_type] = False
            
            # Update status
            await self.update_exchange_status(exchange_type, "unhealthy")
            
            # Try to recover
            exchange = self.exchanges.get(exchange_type)
            if exchange:
                await exchange.recover()
                
                # Check if recovery was successful
                new_health = await exchange.get_health_status()
                if new_health.get("healthy", False):
                    self.active_exchanges[exchange_type] = True
                    await self.update_exchange_status(exchange_type, "connected")
                    logger.info(f"Exchange {exchange_type} recovered successfully")
                else:
                    logger.error(f"Failed to recover exchange {exchange_type}")
                    
        except Exception as e:
            logger.error(f"Error handling unhealthy exchange {exchange_type}: {e}")
    
    async def update_exchange_status(self, exchange_type: ExchangeType, status: str):
        """Update exchange status in database"""
        try:
            with SessionLocal() as db:
                exchange = db.query(Exchange).filter(Exchange.type == exchange_type.value).first()
                if exchange:
                    exchange.status = status
                    exchange.last_heartbeat = datetime.utcnow()
                    db.commit()
                    
        except Exception as e:
            logger.error(f"Error updating exchange status: {e}")
    
    async def get_market_data(self, symbol: str, timeframe: str, exchange_type: Optional[ExchangeType] = None) -> Optional[Dict[str, Any]]:
        """Get market data from exchange"""
        try:
            # Use specific exchange or find best available
            if exchange_type:
                exchange = self.exchanges.get(exchange_type)
                if exchange and self.active_exchanges.get(exchange_type, False):
                    return await exchange.get_market_data(symbol, timeframe)
            else:
                # Try exchanges in order of preference
                for exchange_type in [ExchangeType.ALPACA, ExchangeType.BINANCE, ExchangeType.KUCOIN, ExchangeType.TD_AMERITRADE]:
                    exchange = self.exchanges.get(exchange_type)
                    if exchange and self.active_exchanges.get(exchange_type, False):
                        try:
                            return await exchange.get_market_data(symbol, timeframe)
                        except Exception as e:
                            logger.warning(f"Failed to get market data from {exchange_type}: {e}")
                            continue
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return None
    
    async def place_order(self, symbol: str, side: str, quantity: float, order_type: str, price: Optional[float] = None, exchange_type: Optional[ExchangeType] = None) -> Optional[Dict[str, Any]]:
        """Place order on exchange"""
        try:
            # Use specific exchange or find best available
            if exchange_type:
                exchange = self.exchanges.get(exchange_type)
                if exchange and self.active_exchanges.get(exchange_type, False):
                    return await exchange.place_order(symbol, side, quantity, order_type, price)
            else:
                # Use primary exchange (Alpaca) by default
                exchange = self.exchanges.get(ExchangeType.ALPACA)
                if exchange and self.active_exchanges.get(ExchangeType.ALPACA, False):
                    return await exchange.place_order(symbol, side, quantity, order_type, price)
            
            return None
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    async def cancel_order(self, order_id: str, exchange_type: ExchangeType) -> bool:
        """Cancel order on exchange"""
        try:
            exchange = self.exchanges.get(exchange_type)
            if exchange and self.active_exchanges.get(exchange_type, False):
                return await exchange.cancel_order(order_id)
            
            return False
            
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
    
    async def get_positions(self, exchange_type: Optional[ExchangeType] = None) -> List[Dict[str, Any]]:
        """Get positions from exchange(s)"""
        try:
            positions = []
            
            if exchange_type:
                exchange = self.exchanges.get(exchange_type)
                if exchange and self.active_exchanges.get(exchange_type, False):
                    positions = await exchange.get_positions()
            else:
                # Get positions from all active exchanges
                for exchange_type, exchange in self.exchanges.items():
                    if self.active_exchanges.get(exchange_type, False):
                        try:
                            exchange_positions = await exchange.get_positions()
                            positions.extend(exchange_positions)
                        except Exception as e:
                            logger.warning(f"Failed to get positions from {exchange_type}: {e}")
            
            return positions
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    async def get_account_info(self, exchange_type: ExchangeType) -> Optional[Dict[str, Any]]:
        """Get account information from exchange"""
        try:
            exchange = self.exchanges.get(exchange_type)
            if exchange and self.active_exchanges.get(exchange_type, False):
                return await exchange.get_account_info()
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None
    
    async def get_order_status(self, order_id: str, exchange_type: ExchangeType) -> Optional[Dict[str, Any]]:
        """Get order status from exchange"""
        try:
            exchange = self.exchanges.get(exchange_type)
            if exchange and self.active_exchanges.get(exchange_type, False):
                return await exchange.get_order_status(order_id)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return None
    
    def get_available_exchanges(self) -> List[ExchangeType]:
        """Get list of available exchanges"""
        return [exchange_type for exchange_type, active in self.active_exchanges.items() if active]
    
    def get_exchange_info(self, exchange_type: ExchangeType) -> Optional[Dict[str, Any]]:
        """Get exchange information"""
        try:
            exchange = self.exchanges.get(exchange_type)
            if exchange:
                return {
                    "name": exchange.name,
                    "type": exchange_type.value,
                    "active": self.active_exchanges.get(exchange_type, False),
                    "supported_assets": exchange.supported_assets,
                    "health": self.exchange_health.get(exchange_type, {})
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting exchange info: {e}")
            return None
    
    def is_healthy(self) -> bool:
        """Check if exchange manager is healthy"""
        return (
            self.running and 
            len(self.exchanges) > 0 and 
            any(self.active_exchanges.values())
        )
    
    async def shutdown(self):
        """Shutdown exchange manager"""
        logger.info("Shutting down ExchangeManager...")
        
        self.running = False
        
        # Cancel monitoring tasks
        for task in self.monitoring_tasks:
            task.cancel()
        
        # Shutdown exchanges
        for exchange_type, exchange in self.exchanges.items():
            try:
                await exchange.shutdown()
                logger.info(f"Exchange {exchange_type} shutdown")
            except Exception as e:
                logger.error(f"Error shutting down exchange {exchange_type}: {e}")
        
        logger.info("ExchangeManager shutdown complete")
