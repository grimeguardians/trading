import asyncio
import logging
from typing import Dict, List, Optional, Any
from config import Config
from exchanges.base_exchange import BaseExchange
from exchanges.alpaca_exchange import AlpacaExchange
from exchanges.binance_exchange import BinanceExchange

class ExchangeFactory:
    """Factory for creating and managing exchange connections"""
    
    def __init__(self, config: Config):
        self.config = config
        self.exchanges: Dict[str, BaseExchange] = {}
        self.primary_exchange_name = config.primary_exchange.value
        self.logger = logging.getLogger("ExchangeFactory")
    
    async def initialize(self):
        """Initialize all configured exchanges"""
        try:
            self.logger.info("Initializing exchange connections...")
            
            # Initialize exchanges based on configuration
            for exchange_name, exchange_config in self.config.exchanges.items():
                if self.config.is_exchange_enabled(exchange_name):
                    exchange = await self._create_exchange(exchange_name, exchange_config)
                    
                    if exchange:
                        success = await exchange.initialize()
                        if success:
                            self.exchanges[exchange_name] = exchange
                            self.logger.info(f"Successfully initialized {exchange_name} exchange")
                        else:
                            self.logger.error(f"Failed to initialize {exchange_name} exchange")
                    else:
                        self.logger.error(f"Failed to create {exchange_name} exchange")
            
            if not self.exchanges:
                raise Exception("No exchanges were successfully initialized")
            
            self.logger.info(f"Initialized {len(self.exchanges)} exchanges: {list(self.exchanges.keys())}")
            
        except Exception as e:
            self.logger.error(f"Error initializing exchanges: {e}")
            raise
    
    async def _create_exchange(self, exchange_name: str, exchange_config) -> Optional[BaseExchange]:
        """Create an exchange instance"""
        try:
            # Convert exchange config to dict
            config_dict = {
                'name': exchange_config.name,
                'api_key': exchange_config.api_key,
                'api_secret': exchange_config.api_secret,
                'sandbox': exchange_config.sandbox,
                'base_url': exchange_config.base_url,
                'paper_trading': exchange_config.paper_trading,
                'rate_limits': exchange_config.rate_limits,
                'supported_assets': exchange_config.supported_assets
            }
            
            # Create exchange based on type
            if exchange_name == 'alpaca':
                return AlpacaExchange(config_dict)
            elif exchange_name == 'binance':
                return BinanceExchange(config_dict)
            elif exchange_name == 'kucoin':
                # KuCoin implementation would go here
                from exchanges.kucoin_exchange import KuCoinExchange
                return KuCoinExchange(config_dict)
            elif exchange_name == 'td_ameritrade':
                # TD Ameritrade implementation would go here
                from exchanges.td_ameritrade_exchange import TDAmeritadeExchange
                return TDAmeritadeExchange(config_dict)
            else:
                self.logger.error(f"Unknown exchange type: {exchange_name}")
                return None
                
        except ImportError as e:
            self.logger.warning(f"Exchange {exchange_name} not implemented yet: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error creating exchange {exchange_name}: {e}")
            return None
    
    def get_exchange(self, exchange_name: str) -> Optional[BaseExchange]:
        """Get an exchange by name"""
        return self.exchanges.get(exchange_name)
    
    def get_primary_exchange(self) -> Optional[BaseExchange]:
        """Get the primary exchange"""
        return self.exchanges.get(self.primary_exchange_name)
    
    def get_all_exchanges(self) -> Dict[str, BaseExchange]:
        """Get all initialized exchanges"""
        return self.exchanges.copy()
    
    def get_enabled_exchanges(self) -> List[str]:
        """Get list of enabled exchange names"""
        return list(self.exchanges.keys())
    
    async def get_exchange_for_symbol(self, symbol: str) -> Optional[BaseExchange]:
        """Get the best exchange for a specific symbol"""
        try:
            # Check each exchange to see if it supports the symbol
            for exchange_name, exchange in self.exchanges.items():
                if await exchange.validate_symbol(symbol):
                    return exchange
            
            # If no exchange supports the symbol, return primary
            return self.get_primary_exchange()
            
        except Exception as e:
            self.logger.error(f"Error finding exchange for symbol {symbol}: {e}")
            return self.get_primary_exchange()
    
    async def get_best_price(self, symbol: str) -> Dict[str, Any]:
        """Get best price across all exchanges"""
        try:
            prices = {}
            
            # Get prices from all exchanges
            for exchange_name, exchange in self.exchanges.items():
                if await exchange.validate_symbol(symbol):
                    price = await exchange.get_current_price(symbol)
                    if price:
                        prices[exchange_name] = price
            
            if not prices:
                return {}
            
            # Find best bid and ask
            best_bid = max(prices.values())
            best_ask = min(prices.values())
            
            best_bid_exchange = max(prices, key=prices.get)
            best_ask_exchange = min(prices, key=prices.get)
            
            return {
                'symbol': symbol,
                'best_bid': best_bid,
                'best_ask': best_ask,
                'best_bid_exchange': best_bid_exchange,
                'best_ask_exchange': best_ask_exchange,
                'spread': best_ask - best_bid,
                'all_prices': prices
            }
            
        except Exception as e:
            self.logger.error(f"Error getting best price for {symbol}: {e}")
            return {}
    
    async def execute_arbitrage_opportunity(self, symbol: str, quantity: float) -> Dict[str, Any]:
        """Execute arbitrage opportunity between exchanges"""
        try:
            # Get best prices
            price_data = await self.get_best_price(symbol)
            
            if not price_data or price_data.get('spread', 0) <= 0:
                return {'success': False, 'error': 'No arbitrage opportunity'}
            
            # Buy on exchange with lowest price
            buy_exchange = self.get_exchange(price_data['best_ask_exchange'])
            sell_exchange = self.get_exchange(price_data['best_bid_exchange'])
            
            if not buy_exchange or not sell_exchange:
                return {'success': False, 'error': 'Exchange not available'}
            
            # Execute buy order
            buy_order = await buy_exchange.place_order(
                symbol=symbol,
                side='buy',
                quantity=quantity,
                order_type='market'
            )
            
            if not buy_order.success:
                return {'success': False, 'error': f'Buy order failed: {buy_order.error}'}
            
            # Execute sell order
            sell_order = await sell_exchange.place_order(
                symbol=symbol,
                side='sell',
                quantity=quantity,
                order_type='market'
            )
            
            if not sell_order.success:
                # Try to cancel buy order
                await buy_exchange.cancel_order(buy_order.order_id)
                return {'success': False, 'error': f'Sell order failed: {sell_order.error}'}
            
            # Calculate profit
            buy_price = price_data['best_ask']
            sell_price = price_data['best_bid']
            gross_profit = (sell_price - buy_price) * quantity
            
            # Estimate fees
            buy_fees = await buy_exchange.get_trading_fees(symbol)
            sell_fees = await sell_exchange.get_trading_fees(symbol)
            
            total_fees = (buy_fees['taker_fee'] + sell_fees['taker_fee']) * quantity * buy_price
            net_profit = gross_profit - total_fees
            
            return {
                'success': True,
                'symbol': symbol,
                'quantity': quantity,
                'buy_exchange': price_data['best_ask_exchange'],
                'sell_exchange': price_data['best_bid_exchange'],
                'buy_price': buy_price,
                'sell_price': sell_price,
                'gross_profit': gross_profit,
                'total_fees': total_fees,
                'net_profit': net_profit,
                'buy_order_id': buy_order.order_id,
                'sell_order_id': sell_order.order_id
            }
            
        except Exception as e:
            self.logger.error(f"Error executing arbitrage: {e}")
            return {'success': False, 'error': str(e)}
    
    async def start_data_feeds(self):
        """Start real-time data feeds for all exchanges"""
        try:
            for exchange_name, exchange in self.exchanges.items():
                # Start data feed for each exchange
                asyncio.create_task(self._start_exchange_data_feed(exchange_name, exchange))
            
            self.logger.info("Started data feeds for all exchanges")
            
        except Exception as e:
            self.logger.error(f"Error starting data feeds: {e}")
    
    async def _start_exchange_data_feed(self, exchange_name: str, exchange: BaseExchange):
        """Start data feed for a specific exchange"""
        try:
            # Get popular symbols for the exchange
            symbols = await self._get_popular_symbols(exchange)
            
            if symbols:
                # Subscribe to price updates
                await exchange.subscribe_to_price_updates(
                    symbols,
                    lambda data: self._handle_price_update(exchange_name, data)
                )
                
                self.logger.info(f"Started data feed for {exchange_name} with {len(symbols)} symbols")
            
        except Exception as e:
            self.logger.error(f"Error starting data feed for {exchange_name}: {e}")
    
    async def _get_popular_symbols(self, exchange: BaseExchange) -> List[str]:
        """Get popular symbols for an exchange"""
        try:
            # Get list of symbols
            all_symbols = await exchange.get_symbols()
            
            # Define popular symbols based on exchange type
            if exchange.name == 'alpaca':
                popular = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'SPY', 'QQQ']
            elif exchange.name == 'binance':
                popular = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT', 'SOLUSDT', 'DOTUSDT']
            else:
                popular = all_symbols[:10]  # First 10 symbols
            
            # Filter to only include symbols that exist
            available_symbols = [symbol for symbol in popular if symbol in all_symbols]
            
            return available_symbols[:10]  # Limit to 10 symbols
            
        except Exception as e:
            self.logger.error(f"Error getting popular symbols: {e}")
            return []
    
    async def _handle_price_update(self, exchange_name: str, data: Dict[str, Any]):
        """Handle price update from exchange"""
        try:
            # Log price update
            self.logger.debug(f"Price update from {exchange_name}: {data}")
            
            # Here you could broadcast to WebSocket clients, update database, etc.
            
        except Exception as e:
            self.logger.error(f"Error handling price update: {e}")
    
    async def stop(self):
        """Stop all exchanges"""
        try:
            self.logger.info("Stopping all exchanges...")
            
            for exchange_name, exchange in self.exchanges.items():
                await exchange.disconnect()
                self.logger.info(f"Stopped {exchange_name} exchange")
            
            self.exchanges.clear()
            self.logger.info("All exchanges stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping exchanges: {e}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get status of all exchanges"""
        try:
            status = {
                'total_exchanges': len(self.exchanges),
                'primary_exchange': self.primary_exchange_name,
                'exchanges': {}
            }
            
            for exchange_name, exchange in self.exchanges.items():
                health = await exchange.healthcheck()
                status['exchanges'][exchange_name] = health
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}
    
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary across all exchanges"""
        try:
            summary = {
                'total_value': 0.0,
                'total_cash': 0.0,
                'total_positions': 0,
                'exchanges': {}
            }
            
            for exchange_name, exchange in self.exchanges.items():
                # Get balances
                balances = await exchange.get_balances()
                positions = await exchange.get_positions()
                
                exchange_summary = {
                    'cash': sum(b.available for b in balances if b.currency == 'USD'),
                    'positions': len(positions),
                    'position_value': sum(p.quantity * p.current_price for p in positions if p.current_price)
                }
                
                summary['exchanges'][exchange_name] = exchange_summary
                summary['total_cash'] += exchange_summary['cash']
                summary['total_positions'] += exchange_summary['positions']
                summary['total_value'] += exchange_summary['cash'] + exchange_summary['position_value']
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio summary: {e}")
            return {'error': str(e)}
