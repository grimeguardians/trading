import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

from exchanges.base_exchange import BaseExchange
from exchanges.alpaca_exchange import AlpacaExchange
from exchanges.binance_exchange import BinanceExchange
from exchanges.td_ameritrade_exchange import TDAmeritradeExchange
from exchanges.kucoin_exchange import KuCoinExchange
from config import Config

@dataclass
class ExchangeStatus:
    """Exchange status information"""
    exchange_id: str
    connected: bool
    authenticated: bool
    last_heartbeat: datetime
    error_count: int
    supported_pairs: int
    active_orders: int

class ExchangeManager:
    """Centralized exchange management system"""
    
    def __init__(self):
        self.logger = logging.getLogger("ExchangeManager")
        self.exchanges: Dict[str, BaseExchange] = {}
        self.exchange_configs: Dict[str, Dict[str, Any]] = {}
        self.exchange_status: Dict[str, ExchangeStatus] = {}
        
        # Exchange registry
        self.exchange_registry = {
            'alpaca': AlpacaExchange,
            'binance': BinanceExchange,
            'td_ameritrade': TDAmeritradeExchange,
            'kucoin': KuCoinExchange
        }
        
        # Default configurations
        self._setup_default_configs()
        
        # Performance metrics
        self.total_orders_placed = 0
        self.total_trades_executed = 0
        self.exchange_performance = {}
        
        # Health check settings
        self.health_check_interval = 30  # seconds
        self.max_consecutive_failures = 3
        
    def _setup_default_configs(self):
        """Setup default exchange configurations"""
        self.exchange_configs = {
            'alpaca': {
                'api_key': Config.ALPACA_API_KEY,
                'secret_key': Config.ALPACA_SECRET_KEY,
                'base_url': Config.ALPACA_BASE_URL,
                'enabled': True,
                'priority': 1  # Primary exchange
            },
            'binance': {
                'api_key': Config.BINANCE_API_KEY,
                'secret_key': Config.BINANCE_SECRET_KEY,
                'testnet': Config.BINANCE_TESTNET,
                'enabled': True,
                'priority': 2
            },
            'td_ameritrade': {
                'api_key': Config.TD_AMERITRADE_API_KEY,
                'client_id': Config.TD_AMERITRADE_CLIENT_ID,
                'redirect_uri': Config.TD_AMERITRADE_REDIRECT_URI,
                'enabled': True,
                'priority': 3
            },
            'kucoin': {
                'api_key': Config.KUCOIN_API_KEY,
                'secret_key': Config.KUCOIN_SECRET_KEY,
                'passphrase': Config.KUCOIN_PASSPHRASE,
                'sandbox': Config.KUCOIN_SANDBOX,
                'enabled': True,
                'priority': 4
            }
        }
    
    async def initialize(self):
        """Initialize all configured exchanges"""
        try:
            self.logger.info("Initializing exchange manager...")
            
            # Initialize each exchange
            for exchange_id, config in self.exchange_configs.items():
                if config.get('enabled', False):
                    await self._initialize_exchange(exchange_id, config)
            
            # Start health monitoring
            asyncio.create_task(self._health_monitor())
            
            self.logger.info(f"Exchange manager initialized with {len(self.exchanges)} exchanges")
            
        except Exception as e:
            self.logger.error(f"Error initializing exchange manager: {e}")
            raise
    
    async def _initialize_exchange(self, exchange_id: str, config: Dict[str, Any]):
        """Initialize a single exchange"""
        try:
            if exchange_id not in self.exchange_registry:
                self.logger.error(f"Unknown exchange: {exchange_id}")
                return
            
            # Create exchange instance
            exchange_class = self.exchange_registry[exchange_id]
            exchange = exchange_class(config)
            
            # Initialize exchange
            if await exchange.initialize():
                self.exchanges[exchange_id] = exchange
                
                # Initialize status
                self.exchange_status[exchange_id] = ExchangeStatus(
                    exchange_id=exchange_id,
                    connected=exchange.is_connected(),
                    authenticated=exchange.is_authenticated(),
                    last_heartbeat=datetime.now(),
                    error_count=0,
                    supported_pairs=len(exchange.get_supported_pairs()),
                    active_orders=len(exchange.get_active_orders())
                )
                
                # Initialize performance tracking
                self.exchange_performance[exchange_id] = {
                    'orders_placed': 0,
                    'orders_filled': 0,
                    'orders_cancelled': 0,
                    'orders_failed': 0,
                    'avg_execution_time': 0.0,
                    'success_rate': 0.0
                }
                
                self.logger.info(f"Successfully initialized {exchange_id}")
            else:
                self.logger.error(f"Failed to initialize {exchange_id}")
                
        except Exception as e:
            self.logger.error(f"Error initializing {exchange_id}: {e}")
    
    async def _health_monitor(self):
        """Monitor exchange health"""
        while True:
            try:
                for exchange_id, exchange in self.exchanges.items():
                    await self._check_exchange_health(exchange_id, exchange)
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _check_exchange_health(self, exchange_id: str, exchange: BaseExchange):
        """Check health of a specific exchange"""
        try:
            health_result = await exchange.health_check()
            status = self.exchange_status[exchange_id]
            
            if health_result.get('healthy', False):
                status.connected = True
                status.authenticated = True
                status.last_heartbeat = datetime.now()
                status.error_count = 0
            else:
                status.error_count += 1
                
                if status.error_count >= self.max_consecutive_failures:
                    status.connected = False
                    self.logger.warning(f"Exchange {exchange_id} marked as unhealthy")
                    
                    # Attempt reconnection
                    await self._attempt_reconnection(exchange_id, exchange)
            
            # Update metrics
            status.supported_pairs = len(exchange.get_supported_pairs())
            status.active_orders = len(exchange.get_active_orders())
            
        except Exception as e:
            self.logger.error(f"Error checking health for {exchange_id}: {e}")
            self.exchange_status[exchange_id].error_count += 1
    
    async def _attempt_reconnection(self, exchange_id: str, exchange: BaseExchange):
        """Attempt to reconnect to an exchange"""
        try:
            self.logger.info(f"Attempting to reconnect to {exchange_id}")
            
            # Disconnect first
            await exchange.disconnect()
            
            # Reinitialize
            config = self.exchange_configs[exchange_id]
            if await exchange.initialize():
                self.exchange_status[exchange_id].connected = True
                self.exchange_status[exchange_id].error_count = 0
                self.logger.info(f"Successfully reconnected to {exchange_id}")
            else:
                self.logger.error(f"Failed to reconnect to {exchange_id}")
                
        except Exception as e:
            self.logger.error(f"Error reconnecting to {exchange_id}: {e}")
    
    def get_exchange(self, exchange_id: str) -> Optional[BaseExchange]:
        """Get exchange instance by ID"""
        return self.exchanges.get(exchange_id)
    
    def get_available_exchanges(self) -> List[str]:
        """Get list of available exchanges"""
        return [
            exchange_id for exchange_id, status in self.exchange_status.items()
            if status.connected
        ]
    
    def get_exchange_for_symbol(self, symbol: str) -> Optional[BaseExchange]:
        """Get best exchange for a symbol"""
        try:
            # Check which exchanges support the symbol
            supporting_exchanges = []
            
            for exchange_id, exchange in self.exchanges.items():
                if (self.exchange_status[exchange_id].connected and 
                    symbol in exchange.get_supported_pairs()):
                    supporting_exchanges.append((exchange_id, exchange))
            
            if not supporting_exchanges:
                return None
            
            # Sort by priority (lower number = higher priority)
            supporting_exchanges.sort(
                key=lambda x: self.exchange_configs[x[0]].get('priority', 999)
            )
            
            return supporting_exchanges[0][1]
            
        except Exception as e:
            self.logger.error(f"Error getting exchange for symbol {symbol}: {e}")
            return None
    
    async def place_order(self, exchange_id: str, symbol: str, side: str, 
                         order_type: str, quantity: float, price: Optional[float] = None,
                         stop_price: Optional[float] = None, time_in_force: str = 'GTC'):
        """Place order on specific exchange"""
        try:
            exchange = self.get_exchange(exchange_id)
            if not exchange:
                return {'success': False, 'error': f'Exchange {exchange_id} not available'}
            
            if not self.exchange_status[exchange_id].connected:
                return {'success': False, 'error': f'Exchange {exchange_id} not connected'}
            
            # Track order placement
            start_time = datetime.now()
            self.exchange_performance[exchange_id]['orders_placed'] += 1
            
            # Place order
            result = await exchange.place_order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                time_in_force=time_in_force
            )
            
            # Update performance metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_metrics(exchange_id, result.success, execution_time)
            
            return {
                'success': result.success,
                'order_id': result.order_id,
                'error': result.error_message,
                'execution_time': execution_time,
                'exchange': exchange_id
            }
            
        except Exception as e:
            self.logger.error(f"Error placing order on {exchange_id}: {e}")
            self.exchange_performance[exchange_id]['orders_failed'] += 1
            return {'success': False, 'error': str(e)}
    
    async def cancel_order(self, exchange_id: str, order_id: str):
        """Cancel order on specific exchange"""
        try:
            exchange = self.get_exchange(exchange_id)
            if not exchange:
                return {'success': False, 'error': f'Exchange {exchange_id} not available'}
            
            result = await exchange.cancel_order(order_id)
            
            if result:
                self.exchange_performance[exchange_id]['orders_cancelled'] += 1
                return {'success': True, 'exchange': exchange_id}
            else:
                return {'success': False, 'error': 'Failed to cancel order'}
                
        except Exception as e:
            self.logger.error(f"Error cancelling order on {exchange_id}: {e}")
            return {'success': False, 'error': str(e)}
    
    async def get_market_data(self, exchange_id: str, symbol: str):
        """Get market data from specific exchange"""
        try:
            exchange = self.get_exchange(exchange_id)
            if not exchange:
                return None
            
            return await exchange.get_market_data(symbol)
            
        except Exception as e:
            self.logger.error(f"Error getting market data from {exchange_id}: {e}")
            return None
    
    async def get_account_balance(self, exchange_id: str):
        """Get account balance from specific exchange"""
        try:
            exchange = self.get_exchange(exchange_id)
            if not exchange:
                return []
            
            return await exchange.get_account_balance()
            
        except Exception as e:
            self.logger.error(f"Error getting account balance from {exchange_id}: {e}")
            return []
    
    async def get_positions(self, exchange_id: str):
        """Get positions from specific exchange"""
        try:
            exchange = self.get_exchange(exchange_id)
            if not exchange:
                return []
            
            return await exchange.get_positions()
            
        except Exception as e:
            self.logger.error(f"Error getting positions from {exchange_id}: {e}")
            return []
    
    async def get_all_balances(self) -> Dict[str, List]:
        """Get balances from all exchanges"""
        all_balances = {}
        
        for exchange_id, exchange in self.exchanges.items():
            if self.exchange_status[exchange_id].connected:
                try:
                    balances = await exchange.get_account_balance()
                    all_balances[exchange_id] = balances
                except Exception as e:
                    self.logger.error(f"Error getting balances from {exchange_id}: {e}")
                    all_balances[exchange_id] = []
        
        return all_balances
    
    async def get_all_positions(self) -> Dict[str, List]:
        """Get positions from all exchanges"""
        all_positions = {}
        
        for exchange_id, exchange in self.exchanges.items():
            if self.exchange_status[exchange_id].connected:
                try:
                    positions = await exchange.get_positions()
                    all_positions[exchange_id] = positions
                except Exception as e:
                    self.logger.error(f"Error getting positions from {exchange_id}: {e}")
                    all_positions[exchange_id] = []
        
        return all_positions
    
    async def subscribe_to_market_data(self, exchange_id: str, symbols: List[str]):
        """Subscribe to market data on specific exchange"""
        try:
            exchange = self.get_exchange(exchange_id)
            if not exchange:
                return False
            
            return await exchange.subscribe_to_market_data(symbols)
            
        except Exception as e:
            self.logger.error(f"Error subscribing to market data on {exchange_id}: {e}")
            return False
    
    async def find_arbitrage_opportunities(self, symbol: str) -> List[Dict[str, Any]]:
        """Find arbitrage opportunities across exchanges"""
        try:
            opportunities = []
            prices = {}
            
            # Get prices from all exchanges
            for exchange_id, exchange in self.exchanges.items():
                if (self.exchange_status[exchange_id].connected and 
                    symbol in exchange.get_supported_pairs()):
                    try:
                        market_data = await exchange.get_market_data(symbol)
                        if market_data:
                            prices[exchange_id] = {
                                'bid': market_data.bid,
                                'ask': market_data.ask,
                                'last': market_data.last,
                                'timestamp': market_data.timestamp
                            }
                    except Exception as e:
                        self.logger.error(f"Error getting price from {exchange_id}: {e}")
                        continue
            
            # Find arbitrage opportunities
            if len(prices) >= 2:
                exchanges = list(prices.keys())
                for i in range(len(exchanges)):
                    for j in range(i + 1, len(exchanges)):
                        exchange1 = exchanges[i]
                        exchange2 = exchanges[j]
                        
                        price1 = prices[exchange1]
                        price2 = prices[exchange2]
                        
                        # Check if we can buy on exchange1 and sell on exchange2
                        if price1['ask'] < price2['bid']:
                            profit = price2['bid'] - price1['ask']
                            profit_pct = (profit / price1['ask']) * 100
                            
                            if profit_pct > 0.1:  # Minimum 0.1% profit
                                opportunities.append({
                                    'symbol': symbol,
                                    'buy_exchange': exchange1,
                                    'sell_exchange': exchange2,
                                    'buy_price': price1['ask'],
                                    'sell_price': price2['bid'],
                                    'profit': profit,
                                    'profit_percentage': profit_pct
                                })
                        
                        # Check reverse direction
                        if price2['ask'] < price1['bid']:
                            profit = price1['bid'] - price2['ask']
                            profit_pct = (profit / price2['ask']) * 100
                            
                            if profit_pct > 0.1:
                                opportunities.append({
                                    'symbol': symbol,
                                    'buy_exchange': exchange2,
                                    'sell_exchange': exchange1,
                                    'buy_price': price2['ask'],
                                    'sell_price': price1['bid'],
                                    'profit': profit,
                                    'profit_percentage': profit_pct
                                })
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error finding arbitrage opportunities: {e}")
            return []
    
    def _update_performance_metrics(self, exchange_id: str, success: bool, execution_time: float):
        """Update performance metrics for an exchange"""
        try:
            metrics = self.exchange_performance[exchange_id]
            
            if success:
                metrics['orders_filled'] += 1
            else:
                metrics['orders_failed'] += 1
            
            # Update average execution time
            total_orders = metrics['orders_placed']
            if total_orders > 0:
                current_avg = metrics['avg_execution_time']
                metrics['avg_execution_time'] = (current_avg * (total_orders - 1) + execution_time) / total_orders
            
            # Update success rate
            successful_orders = metrics['orders_filled']
            if total_orders > 0:
                metrics['success_rate'] = successful_orders / total_orders
                
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def get_exchange_status(self, exchange_id: str) -> Optional[ExchangeStatus]:
        """Get status of a specific exchange"""
        return self.exchange_status.get(exchange_id)
    
    def get_all_exchange_status(self) -> Dict[str, ExchangeStatus]:
        """Get status of all exchanges"""
        return self.exchange_status.copy()
    
    def get_performance_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for all exchanges"""
        return self.exchange_performance.copy()
    
    def get_supported_symbols(self) -> Dict[str, List[str]]:
        """Get supported symbols for each exchange"""
        symbols = {}
        
        for exchange_id, exchange in self.exchanges.items():
            if self.exchange_status[exchange_id].connected:
                symbols[exchange_id] = exchange.get_supported_pairs()
        
        return symbols
    
    def get_common_symbols(self) -> List[str]:
        """Get symbols supported by all exchanges"""
        if not self.exchanges:
            return []
        
        common_symbols = None
        
        for exchange_id, exchange in self.exchanges.items():
            if self.exchange_status[exchange_id].connected:
                exchange_symbols = set(exchange.get_supported_pairs())
                
                if common_symbols is None:
                    common_symbols = exchange_symbols
                else:
                    common_symbols = common_symbols.intersection(exchange_symbols)
        
        return list(common_symbols) if common_symbols else []
    
    async def shutdown(self):
        """Shutdown all exchanges"""
        try:
            self.logger.info("Shutting down exchange manager...")
            
            for exchange_id, exchange in self.exchanges.items():
                try:
                    await exchange.disconnect()
                    self.logger.info(f"Disconnected from {exchange_id}")
                except Exception as e:
                    self.logger.error(f"Error disconnecting from {exchange_id}: {e}")
            
            self.exchanges.clear()
            self.exchange_status.clear()
            
            self.logger.info("Exchange manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error shutting down exchange manager: {e}")
