import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

# Safe import for numpy/pandas with fallback
try:
    import numpy as np
    import pandas as pd
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Create mock numpy/pandas for basic functionality
    class MockNumpy:
        @staticmethod
        def array(data):
            return data
        @staticmethod
        def mean(data):
            return sum(data) / len(data) if data else 0
        @staticmethod
        def std(data):
            if not data:
                return 0
            mean = sum(data) / len(data)
            return (sum((x - mean) ** 2 for x in data) / len(data)) ** 0.5
    
    class MockPandas:
        @staticmethod
        def DataFrame(data):
            return data
    
    np = MockNumpy()
    pd = MockPandas()

# Import Config from the root config.py file
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location("config_module", os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.py"))
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
Config = config_module.Config
# Simplified imports to avoid circular dependencies
# from core.knowledge_engine import KnowledgeEngine
# from core.risk_manager import RiskManager
# from core.fibonacci_analysis import FibonacciAnalysis
# from exchanges.exchange_factory import ExchangeFactory
# from strategies.base_strategy import BaseStrategy
# from models import Strategy, Position, Order, TradingSignal, get_session
# from utils.technical_indicators import TechnicalIndicators

class TradingMode(Enum):
    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"

@dataclass
class TradingSignal:
    symbol: str
    signal_type: str  # buy, sell, hold
    strength: float
    confidence: float
    source: str
    metadata: Dict[str, Any]
    timestamp: datetime

class TradingEngine:
    """Core trading engine with multi-exchange support"""
    
    def __init__(self, config: Config, knowledge_engine: KnowledgeEngine, exchange_factory: ExchangeFactory):
        self.config = config
        self.knowledge_engine = knowledge_engine
        self.exchange_factory = exchange_factory
        
        # Core components
        self.risk_manager = RiskManager(config)
        self.fibonacci_analysis = FibonacciAnalysis()
        self.technical_indicators = TechnicalIndicators()
        
        # Trading state
        self.active_strategies: Dict[str, BaseStrategy] = {}
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.signals: Dict[str, List[TradingSignal]] = {}
        
        # Performance tracking
        self.portfolio_value = config.trading_config['portfolio_size']
        self.cash_balance = config.trading_config['portfolio_size']
        self.pnl = 0.0
        self.max_drawdown = 0.0
        
        # Trading mode
        self.mode = TradingMode.PAPER if config.trading_config['paper_trading'] else TradingMode.LIVE
        
        # State
        self.is_running = False
        self.last_update = datetime.now()
        
        self.logger = logging.getLogger("TradingEngine")
        self.executor = ThreadPoolExecutor(max_workers=8)
    
    async def initialize(self):
        """Initialize trading engine"""
        try:
            self.logger.info("Initializing Trading Engine...")
            
            # Initialize risk manager
            await self.risk_manager.initialize()
            
            # Initialize technical indicators
            await self.technical_indicators.initialize()
            
            # Load active strategies
            await self._load_strategies()
            
            # Load existing positions
            await self._load_positions()
            
            # Setup signal processing
            await self._setup_signal_processing()
            
            self.logger.info("Trading Engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Trading Engine: {e}")
            raise
    
    async def start(self):
        """Start the trading engine"""
        try:
            self.logger.info("Starting Trading Engine...")
            
            self.is_running = True
            
            # Start main trading loop
            asyncio.create_task(self._main_trading_loop())
            
            # Start signal processing
            asyncio.create_task(self._process_signals())
            
            # Start portfolio monitoring
            asyncio.create_task(self._monitor_portfolio())
            
            # Start strategy execution
            for strategy in self.active_strategies.values():
                asyncio.create_task(strategy.run())
            
            self.logger.info("Trading Engine started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start Trading Engine: {e}")
            raise
    
    async def stop(self):
        """Stop the trading engine"""
        try:
            self.logger.info("Stopping Trading Engine...")
            
            self.is_running = False
            
            # Stop all strategies
            for strategy in self.active_strategies.values():
                await strategy.stop()
            
            # Close all positions if in live mode
            if self.mode == TradingMode.LIVE:
                await self._close_all_positions()
            
            # Save state
            await self._save_state()
            
            self.logger.info("Trading Engine stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping Trading Engine: {e}")
    
    async def _main_trading_loop(self):
        """Main trading loop"""
        while self.is_running:
            try:
                await asyncio.sleep(1)  # 1 second intervals
                
                # Update market data
                await self._update_market_data()
                
                # Process pending orders
                await self._process_orders()
                
                # Update positions
                await self._update_positions()
                
                # Check risk limits
                await self._check_risk_limits()
                
                # Update performance metrics
                await self._update_performance()
                
                self.last_update = datetime.now()
                
            except Exception as e:
                self.logger.error(f"Error in main trading loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _load_strategies(self):
        """Load active strategies from database"""
        try:
            session = get_session()
            
            # Get active strategies
            strategies = session.query(Strategy).filter(Strategy.is_active == True).all()
            
            for strategy_model in strategies:
                # Import strategy class dynamically
                strategy_class = await self._get_strategy_class(strategy_model.strategy_type)
                
                if strategy_class:
                    strategy = strategy_class(
                        config=self.config,
                        strategy_model=strategy_model,
                        exchange_factory=self.exchange_factory,
                        knowledge_engine=self.knowledge_engine,
                        risk_manager=self.risk_manager
                    )
                    
                    self.active_strategies[strategy_model.name] = strategy
                    
                    self.logger.info(f"Loaded strategy: {strategy_model.name} ({strategy_model.strategy_type})")
            
            session.close()
            
        except Exception as e:
            self.logger.error(f"Error loading strategies: {e}")
    
    async def _get_strategy_class(self, strategy_type: str):
        """Get strategy class by type"""
        try:
            if strategy_type == "swing":
                from strategies.swing_strategy import SwingStrategy
                return SwingStrategy
            elif strategy_type == "scalping":
                from strategies.scalping_strategy import ScalpingStrategy
                return ScalpingStrategy
            elif strategy_type == "options":
                from strategies.options_strategy import OptionsStrategy
                return OptionsStrategy
            elif strategy_type == "intraday":
                from strategies.intraday_strategy import IntradayStrategy
                return IntradayStrategy
            else:
                self.logger.error(f"Unknown strategy type: {strategy_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting strategy class: {e}")
            return None
    
    async def _load_positions(self):
        """Load existing positions from database"""
        try:
            session = get_session()
            
            # Get open positions
            positions = session.query(Position).filter(Position.status == "open").all()
            
            for position in positions:
                self.positions[f"{position.symbol}_{position.strategy_id}"] = position
                
                self.logger.info(f"Loaded position: {position.symbol} ({position.side})")
            
            session.close()
            
        except Exception as e:
            self.logger.error(f"Error loading positions: {e}")
    
    async def _setup_signal_processing(self):
        """Setup signal processing"""
        try:
            # Initialize signal queues
            self.signals = {
                'technical': [],
                'fundamental': [],
                'sentiment': [],
                'ml': []
            }
            
            self.logger.info("Signal processing setup complete")
            
        except Exception as e:
            self.logger.error(f"Error setting up signal processing: {e}")
    
    async def _process_signals(self):
        """Process trading signals"""
        while self.is_running:
            try:
                await asyncio.sleep(2)  # Process every 2 seconds
                
                # Process signals from all sources
                all_signals = []
                for signal_type, signals in self.signals.items():
                    all_signals.extend(signals)
                
                if all_signals:
                    # Group signals by symbol
                    symbol_signals = {}
                    for signal in all_signals:
                        if signal.symbol not in symbol_signals:
                            symbol_signals[signal.symbol] = []
                        symbol_signals[signal.symbol].append(signal)
                    
                    # Process each symbol's signals
                    for symbol, signals in symbol_signals.items():
                        await self._process_symbol_signals(symbol, signals)
                
                # Clear processed signals
                for signal_type in self.signals:
                    self.signals[signal_type] = []
                
            except Exception as e:
                self.logger.error(f"Error processing signals: {e}")
                await asyncio.sleep(5)
    
    async def _process_symbol_signals(self, symbol: str, signals: List[TradingSignal]):
        """Process signals for a specific symbol"""
        try:
            if not signals:
                return
            
            # Aggregate signals
            buy_signals = [s for s in signals if s.signal_type == 'buy']
            sell_signals = [s for s in signals if s.signal_type == 'sell']
            
            # Calculate weighted signal strength
            buy_strength = sum(s.strength * s.confidence for s in buy_signals) / len(buy_signals) if buy_signals else 0
            sell_strength = sum(s.strength * s.confidence for s in sell_signals) / len(sell_signals) if sell_signals else 0
            
            # Make trading decision
            if buy_strength > sell_strength and buy_strength > 0.7:
                await self._execute_buy_signal(symbol, buy_strength, buy_signals)
            elif sell_strength > buy_strength and sell_strength > 0.7:
                await self._execute_sell_signal(symbol, sell_strength, sell_signals)
            
        except Exception as e:
            self.logger.error(f"Error processing signals for {symbol}: {e}")
    
    async def _execute_buy_signal(self, symbol: str, strength: float, signals: List[TradingSignal]):
        """Execute buy signal"""
        try:
            # Check if we already have a position
            position_key = f"{symbol}_buy"
            if position_key in self.positions:
                return
            
            # Risk check
            if not await self.risk_manager.check_position_risk(symbol, "buy", strength):
                return
            
            # Calculate position size
            position_size = await self.risk_manager.calculate_position_size(symbol, strength)
            
            # Get current price
            exchange = self.exchange_factory.get_primary_exchange()
            current_price = await exchange.get_current_price(symbol)
            
            # Calculate Fibonacci levels for stop loss and take profit
            fib_levels = await self.fibonacci_analysis.calculate_retracement_levels(
                symbol, current_price, strength
            )
            
            # Create order
            order = await self._create_order(
                symbol=symbol,
                side="buy",
                quantity=position_size,
                price=current_price,
                stop_loss=fib_levels.get('stop_loss'),
                take_profit=fib_levels.get('take_profit'),
                metadata={
                    'signals': [s.metadata for s in signals],
                    'strength': strength,
                    'fibonacci_levels': fib_levels
                }
            )
            
            if order:
                self.logger.info(f"Executed buy signal for {symbol}: {position_size} shares at ${current_price}")
            
        except Exception as e:
            self.logger.error(f"Error executing buy signal for {symbol}: {e}")
    
    async def _execute_sell_signal(self, symbol: str, strength: float, signals: List[TradingSignal]):
        """Execute sell signal"""
        try:
            # Check if we have a position to sell
            position_key = f"{symbol}_buy"
            if position_key not in self.positions:
                return
            
            position = self.positions[position_key]
            
            # Get current price
            exchange = self.exchange_factory.get_primary_exchange()
            current_price = await exchange.get_current_price(symbol)
            
            # Create sell order
            order = await self._create_order(
                symbol=symbol,
                side="sell",
                quantity=position.quantity,
                price=current_price,
                metadata={
                    'signals': [s.metadata for s in signals],
                    'strength': strength,
                    'close_position': True
                }
            )
            
            if order:
                self.logger.info(f"Executed sell signal for {symbol}: {position.quantity} shares at ${current_price}")
            
        except Exception as e:
            self.logger.error(f"Error executing sell signal for {symbol}: {e}")
    
    async def _create_order(self, symbol: str, side: str, quantity: float, price: float, 
                          stop_loss: float = None, take_profit: float = None, 
                          metadata: Dict[str, Any] = None) -> Optional[Order]:
        """Create a new order"""
        try:
            # Get exchange
            exchange = self.exchange_factory.get_primary_exchange()
            
            # Create order in database
            session = get_session()
            
            order = Order(
                symbol=symbol,
                side=side,
                order_type="market",
                quantity=quantity,
                price=price,
                exchange=exchange.name,
                status="pending"
            )
            
            session.add(order)
            session.commit()
            
            # Submit to exchange
            if self.mode == TradingMode.LIVE:
                exchange_order = await exchange.place_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    order_type="market",
                    price=price
                )
                
                if exchange_order:
                    order.exchange_order_id = exchange_order.get('id')
                    order.status = "submitted"
                    session.commit()
            else:
                # Paper trading - simulate order
                order.status = "filled"
                order.filled_quantity = quantity
                order.filled_price = price
                order.filled_at = datetime.now()
                session.commit()
                
                # Create position
                await self._create_position(order, stop_loss, take_profit, metadata)
            
            session.close()
            
            self.orders[str(order.id)] = order
            
            return order
            
        except Exception as e:
            self.logger.error(f"Error creating order: {e}")
            return None
    
    async def _create_position(self, order: Order, stop_loss: float = None, 
                             take_profit: float = None, metadata: Dict[str, Any] = None):
        """Create a new position from filled order"""
        try:
            session = get_session()
            
            position = Position(
                symbol=order.symbol,
                side="long" if order.side == "buy" else "short",
                quantity=order.filled_quantity,
                entry_price=order.filled_price,
                current_price=order.filled_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                exchange=order.exchange,
                status="open"
            )
            
            session.add(position)
            session.commit()
            
            # Add to active positions
            position_key = f"{order.symbol}_{order.side}"
            self.positions[position_key] = position
            
            # Update cash balance
            if order.side == "buy":
                self.cash_balance -= order.filled_quantity * order.filled_price
            else:
                self.cash_balance += order.filled_quantity * order.filled_price
            
            session.close()
            
            self.logger.info(f"Created position: {order.symbol} {order.side} {order.filled_quantity} @ ${order.filled_price}")
            
        except Exception as e:
            self.logger.error(f"Error creating position: {e}")
    
    async def _update_market_data(self):
        """Update market data for all active symbols"""
        try:
            # Get all active symbols
            symbols = set()
            for position in self.positions.values():
                symbols.add(position.symbol)
            
            # Update market data
            for symbol in symbols:
                exchange = self.exchange_factory.get_primary_exchange()
                market_data = await exchange.get_market_data(symbol)
                
                if market_data:
                    # Update technical indicators
                    await self.technical_indicators.update_indicators(symbol, market_data)
                    
                    # Generate signals
                    signals = await self.technical_indicators.generate_signals(symbol)
                    
                    if signals:
                        self.signals['technical'].extend(signals)
            
        except Exception as e:
            self.logger.error(f"Error updating market data: {e}")
    
    async def _process_orders(self):
        """Process pending orders"""
        try:
            session = get_session()
            
            # Get pending orders
            pending_orders = session.query(Order).filter(Order.status == "pending").all()
            
            for order in pending_orders:
                if self.mode == TradingMode.LIVE:
                    # Check order status with exchange
                    exchange = self.exchange_factory.get_exchange(order.exchange)
                    order_status = await exchange.get_order_status(order.exchange_order_id)
                    
                    if order_status and order_status.get('status') == 'filled':
                        order.status = "filled"
                        order.filled_quantity = order_status.get('filled_quantity', order.quantity)
                        order.filled_price = order_status.get('filled_price', order.price)
                        order.filled_at = datetime.now()
                        
                        # Create position
                        await self._create_position(order)
                
                session.commit()
            
            session.close()
            
        except Exception as e:
            self.logger.error(f"Error processing orders: {e}")
    
    async def _update_positions(self):
        """Update all positions"""
        try:
            session = get_session()
            
            for position_key, position in self.positions.items():
                # Get current price
                exchange = self.exchange_factory.get_exchange(position.exchange)
                current_price = await exchange.get_current_price(position.symbol)
                
                if current_price:
                    position.current_price = current_price
                    
                    # Calculate P&L
                    if position.side == "long":
                        position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                    else:
                        position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
                    
                    # Check stop loss and take profit
                    if position.stop_loss and ((position.side == "long" and current_price <= position.stop_loss) or 
                                             (position.side == "short" and current_price >= position.stop_loss)):
                        await self._close_position(position, "stop_loss")
                    elif position.take_profit and ((position.side == "long" and current_price >= position.take_profit) or 
                                                  (position.side == "short" and current_price <= position.take_profit)):
                        await self._close_position(position, "take_profit")
                    
                    session.commit()
            
            session.close()
            
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
    
    async def _close_position(self, position: Position, reason: str):
        """Close a position"""
        try:
            # Create closing order
            side = "sell" if position.side == "long" else "buy"
            
            order = await self._create_order(
                symbol=position.symbol,
                side=side,
                quantity=position.quantity,
                price=position.current_price,
                metadata={
                    'close_position': True,
                    'reason': reason
                }
            )
            
            if order:
                # Update position
                position.status = "closed"
                position.closed_at = datetime.now()
                position.realized_pnl = position.unrealized_pnl
                
                # Update cash balance
                if side == "sell":
                    self.cash_balance += position.quantity * position.current_price
                else:
                    self.cash_balance -= position.quantity * position.current_price
                
                # Remove from active positions
                position_key = f"{position.symbol}_{position.side}"
                if position_key in self.positions:
                    del self.positions[position_key]
                
                self.logger.info(f"Closed position: {position.symbol} {position.side} {position.quantity} @ ${position.current_price} ({reason})")
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
    
    async def _close_all_positions(self):
        """Close all open positions"""
        try:
            positions_to_close = list(self.positions.values())
            
            for position in positions_to_close:
                await self._close_position(position, "system_shutdown")
            
            self.logger.info(f"Closed {len(positions_to_close)} positions")
            
        except Exception as e:
            self.logger.error(f"Error closing all positions: {e}")
    
    async def _check_risk_limits(self):
        """Check risk limits"""
        try:
            # Check portfolio risk
            portfolio_risk = await self.risk_manager.calculate_portfolio_risk(self.positions)
            
            if portfolio_risk > self.config.risk_config['max_portfolio_risk']:
                self.logger.warning(f"Portfolio risk exceeded: {portfolio_risk:.2%}")
                
                # Reduce positions
                await self._reduce_positions(portfolio_risk)
            
            # Check drawdown
            current_drawdown = await self._calculate_drawdown()
            
            if current_drawdown > self.config.risk_config['max_drawdown']:
                self.logger.warning(f"Max drawdown exceeded: {current_drawdown:.2%}")
                
                # Stop trading
                await self._stop_trading()
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
    
    async def _reduce_positions(self, current_risk: float):
        """Reduce positions to manage risk"""
        try:
            # Sort positions by risk contribution
            positions_by_risk = sorted(
                self.positions.values(),
                key=lambda p: abs(p.unrealized_pnl / p.quantity / p.entry_price),
                reverse=True
            )
            
            # Close highest risk positions first
            for position in positions_by_risk:
                await self._close_position(position, "risk_management")
                
                # Recalculate portfolio risk
                portfolio_risk = await self.risk_manager.calculate_portfolio_risk(self.positions)
                
                if portfolio_risk <= self.config.risk_config['max_portfolio_risk']:
                    break
            
        except Exception as e:
            self.logger.error(f"Error reducing positions: {e}")
    
    async def _stop_trading(self):
        """Stop trading due to risk limits"""
        try:
            self.logger.warning("Stopping trading due to risk limits")
            
            # Stop all strategies
            for strategy in self.active_strategies.values():
                await strategy.stop()
            
            # Close all positions
            await self._close_all_positions()
            
            # Set flag to stop trading
            self.is_running = False
            
        except Exception as e:
            self.logger.error(f"Error stopping trading: {e}")
    
    async def _calculate_drawdown(self) -> float:
        """Calculate current drawdown"""
        try:
            current_value = await self.get_portfolio_value()
            
            if not hasattr(self, 'peak_value'):
                self.peak_value = current_value
            
            self.peak_value = max(self.peak_value, current_value)
            
            drawdown = (self.peak_value - current_value) / self.peak_value if self.peak_value > 0 else 0
            
            return drawdown
            
        except Exception as e:
            self.logger.error(f"Error calculating drawdown: {e}")
            return 0.0
    
    async def _update_performance(self):
        """Update performance metrics"""
        try:
            # Calculate portfolio value
            portfolio_value = await self.get_portfolio_value()
            
            # Calculate P&L
            total_pnl = portfolio_value - self.config.trading_config['portfolio_size']
            
            # Update metrics
            self.portfolio_value = portfolio_value
            self.pnl = total_pnl
            self.max_drawdown = max(self.max_drawdown, await self._calculate_drawdown())
            
        except Exception as e:
            self.logger.error(f"Error updating performance: {e}")
    
    async def _monitor_portfolio(self):
        """Monitor portfolio performance"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Log portfolio status
                portfolio_value = await self.get_portfolio_value()
                position_count = len(self.positions)
                
                self.logger.info(f"Portfolio: ${portfolio_value:,.2f} | Positions: {position_count} | P&L: ${self.pnl:,.2f}")
                
            except Exception as e:
                self.logger.error(f"Error monitoring portfolio: {e}")
                await asyncio.sleep(60)
    
    async def _save_state(self):
        """Save trading engine state"""
        try:
            # Save to database
            session = get_session()
            
            # Update all positions
            for position in self.positions.values():
                session.merge(position)
            
            # Update all orders
            for order in self.orders.values():
                session.merge(order)
            
            session.commit()
            session.close()
            
            self.logger.info("Trading engine state saved")
            
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")
    
    async def get_portfolio_value(self) -> float:
        """Get current portfolio value"""
        try:
            total_value = self.cash_balance
            
            for position in self.positions.values():
                if position.current_price:
                    position_value = position.quantity * position.current_price
                    total_value += position_value
            
            return total_value
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio value: {e}")
            return 0.0
    
    async def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status"""
        try:
            portfolio_value = await self.get_portfolio_value()
            
            return {
                'portfolio_value': portfolio_value,
                'cash_balance': self.cash_balance,
                'pnl': self.pnl,
                'max_drawdown': self.max_drawdown,
                'position_count': len(self.positions),
                'active_strategies': len(self.active_strategies),
                'last_update': self.last_update.isoformat(),
                'is_running': self.is_running,
                'mode': self.mode.value
            }
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio status: {e}")
            return {}
    
    async def add_signal(self, signal: TradingSignal):
        """Add a trading signal"""
        try:
            signal_type = signal.source
            if signal_type not in self.signals:
                signal_type = 'technical'
            
            self.signals[signal_type].append(signal)
            
            self.logger.debug(f"Added signal: {signal.symbol} {signal.signal_type} {signal.strength:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error adding signal: {e}")
