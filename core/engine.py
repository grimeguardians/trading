"""
Core Trading Engine
Freqtrade-inspired trading engine with multi-exchange support
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np

from config import Config
from core.exchange_abstraction import ExchangeManager
from brain.digital_brain import DigitalBrain
from strategies.base_strategy import BaseStrategy
from strategies.swing_trading import SwingTradingStrategy
from strategies.scalping import ScalpingStrategy
from strategies.options_trading import OptionsStrategy
from strategies.intraday import IntradayStrategy
from risk.risk_manager import RiskManager
from risk.position_sizing import PositionSizer
from models.trading_models import Trade, Position, Signal
from utils.helpers import calculate_fibonacci_levels, technical_indicators

class TradingState(Enum):
    """Trading engine states"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"

@dataclass
class EngineMetrics:
    """Trading engine performance metrics"""
    total_trades: int = 0
    successful_trades: int = 0
    failed_trades: int = 0
    total_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    avg_trade_duration: float = 0.0
    risk_adjusted_return: float = 0.0
    active_positions: int = 0
    
class TradingEngine:
    """
    Main trading engine with Freqtrade-inspired architecture
    Supports multiple exchanges and strategies
    """
    
    def __init__(self, config: Config, exchange_manager: ExchangeManager, digital_brain: DigitalBrain):
        self.config = config
        self.exchange_manager = exchange_manager
        self.digital_brain = digital_brain
        self.logger = logging.getLogger(__name__)
        
        # Engine state
        self.state = TradingState.INITIALIZING
        self.start_time: Optional[datetime] = None
        self.last_heartbeat = datetime.now()
        
        # Strategy management
        self.strategies: Dict[str, BaseStrategy] = {}
        self.active_strategies: List[str] = []
        
        # Risk management
        self.risk_manager = RiskManager(config)
        self.position_sizer = PositionSizer(config)
        
        # Trading data
        self.active_positions: Dict[str, Position] = {}
        self.trade_history: List[Trade] = []
        self.pending_orders: Dict[str, Any] = {}
        
        # Performance tracking
        self.metrics = EngineMetrics()
        self.portfolio_value_history: List[Tuple[datetime, float]] = []
        
        # Signal processing
        self.signal_queue: asyncio.Queue = asyncio.Queue()
        self.last_signal_time: Dict[str, datetime] = {}
        
        # Mathematical models
        self.fibonacci_cache: Dict[str, Dict[str, float]] = {}
        self.technical_indicators_cache: Dict[str, Dict[str, float]] = {}
        
    async def initialize(self):
        """Initialize the trading engine"""
        try:
            self.logger.info("⚙️ Initializing trading engine...")
            
            # Initialize strategies
            await self._initialize_strategies()
            
            # Initialize risk management
            await self.risk_manager.initialize()
            
            # Initialize position sizer
            await self.position_sizer.initialize()
            
            # Load historical data for analysis
            await self._load_historical_data()
            
            # Initialize mathematical models
            await self._initialize_mathematical_models()
            
            self.state = TradingState.STOPPED
            self.logger.info("✅ Trading engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Engine initialization failed: {e}")
            self.state = TradingState.ERROR
            raise
            
    async def _initialize_strategies(self):
        """Initialize all trading strategies"""
        strategy_classes = {
            "swing_trading": SwingTradingStrategy,
            "scalping": ScalpingStrategy,
            "options_trading": OptionsStrategy,
            "intraday": IntradayStrategy
        }
        
        for strategy_name, strategy_class in strategy_classes.items():
            try:
                strategy = strategy_class(
                    config=self.config,
                    exchange_manager=self.exchange_manager,
                    digital_brain=self.digital_brain
                )
                await strategy.initialize()
                self.strategies[strategy_name] = strategy
                self.logger.info(f"✅ Initialized strategy: {strategy_name}")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize strategy {strategy_name}: {e}")
                
    async def _load_historical_data(self):
        """Load historical data for analysis"""
        # This would typically load from database or exchange APIs
        # For now, we'll simulate some basic historical data loading
        self.logger.info("📊 Loading historical data...")
        
        # Load data for active symbols
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "SPY"]  # Example symbols
        
        for symbol in symbols:
            try:
                # In a real implementation, this would fetch actual historical data
                historical_data = await self.exchange_manager.get_historical_data(
                    symbol=symbol,
                    timeframe="1d",
                    limit=200
                )
                
                if historical_data is not None:
                    self.logger.debug(f"📈 Loaded {len(historical_data)} data points for {symbol}")
                    
            except Exception as e:
                self.logger.error(f"❌ Failed to load data for {symbol}: {e}")
                
    async def _initialize_mathematical_models(self):
        """Initialize mathematical models and indicators"""
        self.logger.info("🔢 Initializing mathematical models...")
        
        # Initialize Fibonacci analysis
        self.fibonacci_levels = self.config.trading.fibonacci_levels
        
        # Initialize technical indicators
        self.technical_indicators = [
            "RSI", "MACD", "BB", "SMA", "EMA", "STOCH", "WILLIAMS_R"
        ]
        
        self.logger.info("✅ Mathematical models initialized")
        
    async def start(self):
        """Start the trading engine"""
        try:
            if self.state != TradingState.STOPPED:
                raise ValueError(f"Cannot start engine in state: {self.state}")
                
            self.logger.info("🚀 Starting trading engine...")
            
            self.state = TradingState.RUNNING
            self.start_time = datetime.now()
            
            # Start background tasks
            asyncio.create_task(self._signal_processor())
            asyncio.create_task(self._market_monitor())
            asyncio.create_task(self._risk_monitor())
            asyncio.create_task(self._performance_tracker())
            
            # Enable active strategies
            self.active_strategies = [self.config.trading.default_strategy]
            
            self.logger.info("✅ Trading engine started successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start engine: {e}")
            self.state = TradingState.ERROR
            raise
            
    async def _signal_processor(self):
        """Process trading signals from strategies"""
        while self.state == TradingState.RUNNING:
            try:
                # Get signal from queue (with timeout)
                signal = await asyncio.wait_for(self.signal_queue.get(), timeout=1.0)
                
                # Process the signal
                await self._process_signal(signal)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"❌ Signal processing error: {e}")
                await asyncio.sleep(1)
                
    async def _process_signal(self, signal: Signal):
        """Process a trading signal"""
        try:
            self.logger.info(f"📡 Processing signal: {signal.symbol} - {signal.action}")
            
            # Risk management check
            if not await self.risk_manager.validate_signal(signal):
                self.logger.warning(f"⚠️ Signal rejected by risk manager: {signal.symbol}")
                return
                
            # Position sizing
            position_size = await self.position_sizer.calculate_position_size(signal)
            
            # Execute trade
            success = await self._execute_trade(signal, position_size)
            
            if success:
                self.logger.info(f"✅ Signal executed successfully: {signal.symbol}")
                self.metrics.successful_trades += 1
            else:
                self.logger.warning(f"❌ Signal execution failed: {signal.symbol}")
                self.metrics.failed_trades += 1
                
            self.metrics.total_trades += 1
            
        except Exception as e:
            self.logger.error(f"❌ Signal processing error: {e}")
            
    async def _execute_trade(self, signal: Signal, position_size: float) -> bool:
        """Execute a trade based on signal"""
        try:
            # Determine exchange to use
            exchange_name = signal.exchange or self.config.get_primary_exchange()
            
            # Calculate Fibonacci levels for entry/exit optimization
            fib_levels = await self._calculate_fibonacci_levels(signal.symbol)
            
            # Prepare order parameters
            order_params = {
                "symbol": signal.symbol,
                "side": signal.action,
                "quantity": position_size,
                "order_type": signal.order_type or "market",
                "stop_loss": self._calculate_stop_loss(signal, fib_levels),
                "take_profit": self._calculate_take_profit(signal, fib_levels)
            }
            
            # Execute order through exchange manager
            order_result = await self.exchange_manager.place_order(
                exchange_name=exchange_name,
                **order_params
            )
            
            if order_result and order_result.get("status") == "filled":
                # Create position record
                position = Position(
                    symbol=signal.symbol,
                    exchange=exchange_name,
                    side=signal.action,
                    quantity=position_size,
                    entry_price=order_result.get("filled_price"),
                    stop_loss=order_params["stop_loss"],
                    take_profit=order_params["take_profit"],
                    timestamp=datetime.now()
                )
                
                self.active_positions[f"{signal.symbol}_{exchange_name}"] = position
                
                # Update metrics
                self.metrics.active_positions = len(self.active_positions)
                
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Trade execution error: {e}")
            return False
            
    async def _calculate_fibonacci_levels(self, symbol: str) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels"""
        try:
            # Check cache first
            if symbol in self.fibonacci_cache:
                cache_time = self.fibonacci_cache[symbol].get("timestamp", 0)
                if time.time() - cache_time < 300:  # 5 minute cache
                    return self.fibonacci_cache[symbol]
                    
            # Get recent price data
            historical_data = await self.exchange_manager.get_historical_data(
                symbol=symbol,
                timeframe="1h",
                limit=100
            )
            
            if historical_data is None or len(historical_data) < 50:
                return {}
                
            # Calculate Fibonacci levels
            fib_levels = calculate_fibonacci_levels(
                historical_data,
                self.fibonacci_levels
            )
            
            # Cache the results
            fib_levels["timestamp"] = time.time()
            self.fibonacci_cache[symbol] = fib_levels
            
            return fib_levels
            
        except Exception as e:
            self.logger.error(f"❌ Fibonacci calculation error for {symbol}: {e}")
            return {}
            
    def _calculate_stop_loss(self, signal: Signal, fib_levels: Dict[str, float]) -> float:
        """Calculate stop loss using Fibonacci levels"""
        try:
            current_price = signal.price
            stop_loss_percent = self.config.trading.stop_loss_percent
            
            # Base stop loss
            if signal.action == "buy":
                base_stop = current_price * (1 - stop_loss_percent)
            else:
                base_stop = current_price * (1 + stop_loss_percent)
                
            # Adjust using Fibonacci levels if available
            if fib_levels and "support" in fib_levels:
                if signal.action == "buy":
                    # Use Fibonacci support as stop loss if it's better
                    fib_stop = fib_levels["support"]
                    if fib_stop < current_price and fib_stop > base_stop:
                        return fib_stop
                else:
                    # Use Fibonacci resistance as stop loss for short positions
                    fib_stop = fib_levels.get("resistance", base_stop)
                    if fib_stop > current_price and fib_stop < base_stop:
                        return fib_stop
                        
            return base_stop
            
        except Exception as e:
            self.logger.error(f"❌ Stop loss calculation error: {e}")
            return signal.price * 0.95  # Default 5% stop loss
            
    def _calculate_take_profit(self, signal: Signal, fib_levels: Dict[str, float]) -> float:
        """Calculate take profit using Fibonacci levels"""
        try:
            current_price = signal.price
            take_profit_percent = self.config.trading.take_profit_percent
            
            # Base take profit
            if signal.action == "buy":
                base_take_profit = current_price * (1 + take_profit_percent)
            else:
                base_take_profit = current_price * (1 - take_profit_percent)
                
            # Adjust using Fibonacci levels if available
            if fib_levels and "resistance" in fib_levels:
                if signal.action == "buy":
                    # Use Fibonacci resistance as take profit
                    fib_take_profit = fib_levels["resistance"]
                    if fib_take_profit > current_price:
                        return fib_take_profit
                else:
                    # Use Fibonacci support as take profit for short positions
                    fib_take_profit = fib_levels.get("support", base_take_profit)
                    if fib_take_profit < current_price:
                        return fib_take_profit
                        
            return base_take_profit
            
        except Exception as e:
            self.logger.error(f"❌ Take profit calculation error: {e}")
            return signal.price * 1.1  # Default 10% take profit
            
    async def _market_monitor(self):
        """Monitor market conditions and active positions"""
        while self.state == TradingState.RUNNING:
            try:
                # Monitor active positions
                for position_key, position in list(self.active_positions.items()):
                    await self._monitor_position(position_key, position)
                    
                # Generate new signals from active strategies
                for strategy_name in self.active_strategies:
                    if strategy_name in self.strategies:
                        strategy = self.strategies[strategy_name]
                        signals = await strategy.generate_signals()
                        
                        for signal in signals:
                            await self.signal_queue.put(signal)
                            
                # Update heartbeat
                self.last_heartbeat = datetime.now()
                
                # Sleep for 5 seconds before next iteration
                await asyncio.sleep(5)
                
            except Exception as e:
                self.logger.error(f"❌ Market monitoring error: {e}")
                await asyncio.sleep(10)
                
    async def _monitor_position(self, position_key: str, position: Position):
        """Monitor a single position for stop loss/take profit"""
        try:
            # Get current price
            current_price = await self.exchange_manager.get_current_price(
                position.exchange,
                position.symbol
            )
            
            if current_price is None:
                return
                
            # Check stop loss
            if position.side == "buy" and current_price <= position.stop_loss:
                await self._close_position(position_key, position, "stop_loss")
                return
            elif position.side == "sell" and current_price >= position.stop_loss:
                await self._close_position(position_key, position, "stop_loss")
                return
                
            # Check take profit
            if position.side == "buy" and current_price >= position.take_profit:
                await self._close_position(position_key, position, "take_profit")
                return
            elif position.side == "sell" and current_price <= position.take_profit:
                await self._close_position(position_key, position, "take_profit")
                return
                
            # Update unrealized PnL
            if position.side == "buy":
                unrealized_pnl = (current_price - position.entry_price) * position.quantity
            else:
                unrealized_pnl = (position.entry_price - current_price) * position.quantity
                
            position.unrealized_pnl = unrealized_pnl
            
        except Exception as e:
            self.logger.error(f"❌ Position monitoring error for {position_key}: {e}")
            
    async def _close_position(self, position_key: str, position: Position, reason: str):
        """Close a position"""
        try:
            # Place closing order
            close_side = "sell" if position.side == "buy" else "buy"
            
            order_result = await self.exchange_manager.place_order(
                exchange_name=position.exchange,
                symbol=position.symbol,
                side=close_side,
                quantity=position.quantity,
                order_type="market"
            )
            
            if order_result and order_result.get("status") == "filled":
                # Calculate realized PnL
                exit_price = order_result.get("filled_price")
                if position.side == "buy":
                    realized_pnl = (exit_price - position.entry_price) * position.quantity
                else:
                    realized_pnl = (position.entry_price - exit_price) * position.quantity
                    
                # Create trade record
                trade = Trade(
                    symbol=position.symbol,
                    exchange=position.exchange,
                    side=position.side,
                    quantity=position.quantity,
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    realized_pnl=realized_pnl,
                    entry_time=position.timestamp,
                    exit_time=datetime.now(),
                    exit_reason=reason
                )
                
                self.trade_history.append(trade)
                
                # Remove from active positions
                del self.active_positions[position_key]
                
                # Update metrics
                self.metrics.active_positions = len(self.active_positions)
                self.metrics.total_pnl += realized_pnl
                
                self.logger.info(f"✅ Position closed: {position.symbol} - {reason} - PnL: {realized_pnl:.2f}")
                
        except Exception as e:
            self.logger.error(f"❌ Position closing error: {e}")
            
    async def _risk_monitor(self):
        """Monitor risk metrics and portfolio health"""
        while self.state == TradingState.RUNNING:
            try:
                # Update portfolio metrics
                await self._update_portfolio_metrics()
                
                # Check risk limits
                if await self.risk_manager.check_risk_limits(self.active_positions):
                    self.logger.warning("⚠️ Risk limits breached - taking protective action")
                    
                # Sleep for 30 seconds
                await asyncio.sleep(30)
                
            except Exception as e:
                self.logger.error(f"❌ Risk monitoring error: {e}")
                await asyncio.sleep(60)
                
    async def _update_portfolio_metrics(self):
        """Update portfolio performance metrics"""
        try:
            # Calculate total portfolio value
            total_value = 0.0
            total_unrealized_pnl = 0.0
            
            for position in self.active_positions.values():
                current_price = await self.exchange_manager.get_current_price(
                    position.exchange,
                    position.symbol
                )
                
                if current_price:
                    if position.side == "buy":
                        position_value = current_price * position.quantity
                        unrealized_pnl = (current_price - position.entry_price) * position.quantity
                    else:
                        position_value = position.entry_price * position.quantity
                        unrealized_pnl = (position.entry_price - current_price) * position.quantity
                        
                    total_value += position_value
                    total_unrealized_pnl += unrealized_pnl
                    
            # Update metrics
            self.metrics.unrealized_pnl = total_unrealized_pnl
            
            # Record portfolio value for history
            self.portfolio_value_history.append((datetime.now(), total_value))
            
            # Calculate win rate
            if self.trade_history:
                winning_trades = sum(1 for trade in self.trade_history if trade.realized_pnl > 0)
                self.metrics.win_rate = winning_trades / len(self.trade_history) * 100
                
            # Calculate average trade duration
            if self.trade_history:
                total_duration = sum(
                    (trade.exit_time - trade.entry_time).total_seconds()
                    for trade in self.trade_history
                )
                self.metrics.avg_trade_duration = total_duration / len(self.trade_history)
                
        except Exception as e:
            self.logger.error(f"❌ Portfolio metrics update error: {e}")
            
    async def _performance_tracker(self):
        """Track and log performance metrics"""
        while self.state == TradingState.RUNNING:
            try:
                # Log performance every 5 minutes
                self.logger.info(f"📊 Performance Update: "
                               f"Total Trades: {self.metrics.total_trades}, "
                               f"Win Rate: {self.metrics.win_rate:.1f}%, "
                               f"Total PnL: ${self.metrics.total_pnl:.2f}, "
                               f"Unrealized PnL: ${self.metrics.unrealized_pnl:.2f}, "
                               f"Active Positions: {self.metrics.active_positions}")
                               
                # Sleep for 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                self.logger.error(f"❌ Performance tracking error: {e}")
                await asyncio.sleep(300)
                
    async def add_signal(self, signal: Signal):
        """Add a signal to the processing queue"""
        await self.signal_queue.put(signal)
        
    def get_status(self) -> Dict[str, Any]:
        """Get current engine status"""
        return {
            "state": self.state.value,
            "uptime": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            "active_strategies": self.active_strategies,
            "metrics": self.metrics.__dict__,
            "active_positions": len(self.active_positions),
            "last_heartbeat": self.last_heartbeat.isoformat()
        }
        
    def is_healthy(self) -> bool:
        """Check if engine is healthy"""
        try:
            # Check if state is running
            if self.state != TradingState.RUNNING:
                return False
                
            # Check if heartbeat is recent (within 1 minute)
            if (datetime.now() - self.last_heartbeat).total_seconds() > 60:
                return False
                
            # Check if risk limits are not breached
            # This would involve more complex risk checks
            
            return True
            
        except Exception:
            return False
            
    async def pause(self):
        """Pause the trading engine"""
        self.state = TradingState.PAUSED
        self.logger.info("⏸️ Trading engine paused")
        
    async def resume(self):
        """Resume the trading engine"""
        self.state = TradingState.RUNNING
        self.logger.info("▶️ Trading engine resumed")
        
    async def shutdown(self):
        """Shutdown the trading engine"""
        self.logger.info("🛑 Shutting down trading engine...")
        
        self.state = TradingState.STOPPING
        
        # Close all active positions
        for position_key, position in list(self.active_positions.items()):
            await self._close_position(position_key, position, "shutdown")
            
        # Cancel pending orders
        await self.exchange_manager.cancel_all_orders()
        
        self.state = TradingState.STOPPED
        self.logger.info("✅ Trading engine shut down complete")
