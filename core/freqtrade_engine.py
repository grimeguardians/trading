"""
Enhanced Trading Engine inspired by Freqtrade architecture
Integrates with Digital Brain for advanced decision making
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from enum import Enum
import uuid
import json

from config import Config
from core.exchange_abstraction import ExchangeAbstraction
from core.mathematical_models import MathematicalModels
from core.risk_management import RiskManager
from knowledge.digital_brain import DigitalBrain
from data.market_data_pipeline import MarketDataPipeline
from models.trading_models import Trade, Position, Order, OrderType, OrderStatus

class TradingState(Enum):
    """Trading engine states"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"

class StrategyState(Enum):
    """Strategy execution states"""
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"

@dataclass
class StrategyConfig:
    """Strategy configuration"""
    strategy_id: str
    name: str
    strategy_type: str  # swing, scalping, options, intraday
    exchange: str
    symbols: List[str]
    parameters: Dict[str, Any]
    risk_limits: Dict[str, float]
    enabled: bool = True
    max_positions: int = 5
    position_size: float = 0.02  # 2% of portfolio per position

@dataclass
class TradingSignal:
    """Trading signal from Digital Brain"""
    signal_id: str
    symbol: str
    exchange: str
    signal_type: str  # buy, sell, hold
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    reasoning: str
    timestamp: datetime
    strategy_id: str
    fibonacci_levels: Optional[Dict[str, float]] = None
    technical_indicators: Optional[Dict[str, Any]] = None

class FreqtradeEngine:
    """
    Enhanced Trading Engine with Digital Brain integration
    Follows Freqtrade architecture patterns with AI enhancements
    """
    
    def __init__(self, config: Config, digital_brain: DigitalBrain, market_data_pipeline: MarketDataPipeline):
        self.config = config
        self.digital_brain = digital_brain
        self.market_data_pipeline = market_data_pipeline
        
        # Core components
        self.exchange_abstraction = ExchangeAbstraction(config)
        self.mathematical_models = MathematicalModels(config)
        self.risk_manager = RiskManager(config)
        
        # State management
        self.state = TradingState.STOPPED
        self.strategies: Dict[str, StrategyConfig] = {}
        self.strategy_states: Dict[str, StrategyState] = {}
        
        # Trading data
        self.active_positions: Dict[str, Position] = {}
        self.pending_orders: Dict[str, Order] = {}
        self.trade_history: List[Trade] = []
        self.signals_queue: List[TradingSignal] = []
        
        # Performance tracking
        self.performance_metrics: Dict[str, Any] = {}
        self.daily_pnl: float = 0.0
        self.total_pnl: float = 0.0
        self.max_drawdown: float = 0.0
        
        # Threading and async
        self.running = False
        self.main_loop_task: Optional[asyncio.Task] = None
        
        # Logging
        self.logger = logging.getLogger("FreqtradeEngine")
        
        # Initialize default strategies
        self._initialize_default_strategies()
    
    def _initialize_default_strategies(self):
        """Initialize default trading strategies"""
        default_strategies = [
            {
                "strategy_id": "swing_001",
                "name": "AI Swing Trading",
                "strategy_type": "swing",
                "exchange": self.config.exchanges.default_exchange,
                "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA"],
                "parameters": {
                    "timeframe": "1h",
                    "fibonacci_enabled": True,
                    "rsi_threshold": 70,
                    "macd_enabled": True
                },
                "risk_limits": {
                    "max_position_size": 0.05,
                    "stop_loss": 0.02,
                    "take_profit": 0.06
                }
            },
            {
                "strategy_id": "scalp_001",
                "name": "AI Scalping",
                "strategy_type": "scalping",
                "exchange": self.config.exchanges.default_exchange,
                "symbols": ["SPY", "QQQ", "IWM"],
                "parameters": {
                    "timeframe": "5m",
                    "volume_threshold": 1000000,
                    "spread_threshold": 0.01
                },
                "risk_limits": {
                    "max_position_size": 0.01,
                    "stop_loss": 0.005,
                    "take_profit": 0.015
                }
            },
            {
                "strategy_id": "crypto_001",
                "name": "Crypto Momentum",
                "strategy_type": "intraday",
                "exchange": "binance",
                "symbols": ["BTCUSDT", "ETHUSDT", "ADAUSDT"],
                "parameters": {
                    "timeframe": "15m",
                    "volatility_threshold": 0.02,
                    "momentum_period": 20
                },
                "risk_limits": {
                    "max_position_size": 0.03,
                    "stop_loss": 0.03,
                    "take_profit": 0.09
                }
            }
        ]
        
        for strategy_data in default_strategies:
            strategy = StrategyConfig(**strategy_data)
            self.strategies[strategy.strategy_id] = strategy
            self.strategy_states[strategy.strategy_id] = StrategyState.STOPPED
    
    async def initialize(self):
        """Initialize the trading engine"""
        try:
            self.logger.info("Initializing Freqtrade Engine...")
            
            # Initialize exchange connections
            await self.exchange_abstraction.initialize()
            
            # Initialize mathematical models
            await self.mathematical_models.initialize()
            
            # Initialize risk management
            await self.risk_manager.initialize()
            
            # Load existing positions and orders
            await self._load_existing_positions()
            
            # Initialize performance metrics
            await self._initialize_performance_metrics()
            
            self.logger.info("Freqtrade Engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize trading engine: {e}")
            self.state = TradingState.ERROR
            raise
    
    async def start(self):
        """Start the trading engine"""
        if self.state != TradingState.STOPPED:
            raise ValueError(f"Cannot start engine in state: {self.state}")
        
        try:
            self.state = TradingState.STARTING
            self.running = True
            
            # Start main trading loop
            self.main_loop_task = asyncio.create_task(self._main_trading_loop())
            
            # Start strategy execution loops
            for strategy_id in self.strategies.keys():
                if self.strategies[strategy_id].enabled:
                    self.strategy_states[strategy_id] = StrategyState.ACTIVE
            
            self.state = TradingState.RUNNING
            self.logger.info("Trading engine started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start trading engine: {e}")
            self.state = TradingState.ERROR
            raise
    
    async def stop(self):
        """Stop the trading engine"""
        self.logger.info("Stopping trading engine...")
        self.state = TradingState.STOPPING
        self.running = False
        
        # Stop all strategies
        for strategy_id in self.strategy_states.keys():
            self.strategy_states[strategy_id] = StrategyState.STOPPED
        
        # Cancel main loop
        if self.main_loop_task:
            self.main_loop_task.cancel()
            try:
                await self.main_loop_task
            except asyncio.CancelledError:
                pass
        
        # Close all positions if configured
        if self.config.trading.paper_trading:
            await self._close_all_positions()
        
        self.state = TradingState.STOPPED
        self.logger.info("Trading engine stopped")
    
    async def _main_trading_loop(self):
        """Main trading loop"""
        while self.running:
            try:
                # Process signals queue
                await self._process_signals()
                
                # Update positions and orders
                await self._update_positions()
                
                # Check risk limits
                await self._check_risk_limits()
                
                # Generate new signals
                await self._generate_signals()
                
                # Update performance metrics
                await self._update_performance_metrics()
                
                # Sleep for next iteration
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in main trading loop: {e}")
                if not self.config.debug:
                    await asyncio.sleep(5)
    
    async def _process_signals(self):
        """Process trading signals from queue"""
        while self.signals_queue:
            signal = self.signals_queue.pop(0)
            
            try:
                # Validate signal
                if not self._validate_signal(signal):
                    continue
                
                # Check risk management
                if not await self.risk_manager.check_signal_risk(signal):
                    self.logger.warning(f"Signal rejected by risk management: {signal.symbol}")
                    continue
                
                # Execute signal
                await self._execute_signal(signal)
                
            except Exception as e:
                self.logger.error(f"Error processing signal {signal.signal_id}: {e}")
    
    async def _execute_signal(self, signal: TradingSignal):
        """Execute a trading signal"""
        try:
            # Get exchange for signal
            exchange = self.exchange_abstraction.get_exchange(signal.exchange)
            
            # Calculate position size
            position_size = await self._calculate_position_size(signal)
            
            # Create order
            order = Order(
                order_id=str(uuid.uuid4()),
                symbol=signal.symbol,
                exchange=signal.exchange,
                order_type=OrderType.MARKET if signal.signal_type == "buy" else OrderType.MARKET,
                side="buy" if signal.signal_type == "buy" else "sell",
                quantity=position_size,
                price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                status=OrderStatus.PENDING,
                timestamp=datetime.now(),
                strategy_id=signal.strategy_id,
                reasoning=signal.reasoning
            )
            
            # Submit order to exchange
            result = await exchange.submit_order(order)
            
            if result.get("success"):
                order.status = OrderStatus.FILLED
                order.filled_price = result.get("filled_price", signal.entry_price)
                order.filled_quantity = result.get("filled_quantity", position_size)
                
                # Create position
                position = Position(
                    position_id=str(uuid.uuid4()),
                    symbol=signal.symbol,
                    exchange=signal.exchange,
                    side=order.side,
                    quantity=order.filled_quantity,
                    entry_price=order.filled_price,
                    current_price=order.filled_price,
                    unrealized_pnl=0.0,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    timestamp=datetime.now(),
                    strategy_id=signal.strategy_id
                )
                
                self.active_positions[position.position_id] = position
                
                # Create trade record
                trade = Trade(
                    trade_id=str(uuid.uuid4()),
                    symbol=signal.symbol,
                    exchange=signal.exchange,
                    side=order.side,
                    quantity=order.filled_quantity,
                    entry_price=order.filled_price,
                    exit_price=None,
                    pnl=0.0,
                    entry_time=datetime.now(),
                    exit_time=None,
                    strategy_id=signal.strategy_id,
                    reasoning=signal.reasoning
                )
                
                self.trade_history.append(trade)
                
                self.logger.info(f"Signal executed: {signal.symbol} {signal.signal_type} at {order.filled_price}")
                
            else:
                order.status = OrderStatus.REJECTED
                self.logger.error(f"Order rejected: {result.get('error', 'Unknown error')}")
            
            self.pending_orders[order.order_id] = order
            
        except Exception as e:
            self.logger.error(f"Error executing signal: {e}")
    
    async def _generate_signals(self):
        """Generate trading signals using Digital Brain"""
        try:
            for strategy_id, strategy in self.strategies.items():
                if self.strategy_states[strategy_id] != StrategyState.ACTIVE:
                    continue
                
                # Get market data for strategy symbols
                for symbol in strategy.symbols:
                    # Get current market data
                    market_data = await self.market_data_pipeline.get_symbol_data(
                        symbol, strategy.exchange, timeframe=strategy.parameters.get("timeframe", "1h")
                    )
                    
                    if market_data is None:
                        continue
                    
                    # Apply mathematical models
                    analysis = await self.mathematical_models.analyze_symbol(symbol, market_data)
                    
                    # Query Digital Brain for insights
                    brain_analysis = await self.digital_brain.analyze_trading_opportunity(
                        symbol=symbol,
                        market_data=market_data,
                        technical_analysis=analysis,
                        strategy_type=strategy.strategy_type
                    )
                    
                    # Generate signal if conditions are met
                    if brain_analysis.get("signal_strength", 0) > 0.7:
                        signal = TradingSignal(
                            signal_id=str(uuid.uuid4()),
                            symbol=symbol,
                            exchange=strategy.exchange,
                            signal_type=brain_analysis.get("signal_type", "hold"),
                            confidence=brain_analysis.get("signal_strength", 0),
                            entry_price=market_data.get("close", 0),
                            stop_loss=brain_analysis.get("stop_loss", 0),
                            take_profit=brain_analysis.get("take_profit", 0),
                            reasoning=brain_analysis.get("reasoning", ""),
                            timestamp=datetime.now(),
                            strategy_id=strategy_id,
                            fibonacci_levels=analysis.get("fibonacci_levels"),
                            technical_indicators=analysis.get("indicators")
                        )
                        
                        self.signals_queue.append(signal)
                        self.logger.info(f"Signal generated: {symbol} {signal.signal_type} (confidence: {signal.confidence:.2f})")
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
    
    async def _update_positions(self):
        """Update active positions with current market data"""
        try:
            for position_id, position in self.active_positions.items():
                # Get current price
                current_data = await self.market_data_pipeline.get_current_price(
                    position.symbol, position.exchange
                )
                
                if current_data:
                    position.current_price = current_data.get("price", position.current_price)
                    
                    # Calculate unrealized PnL
                    if position.side == "buy":
                        position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity
                    else:
                        position.unrealized_pnl = (position.entry_price - position.current_price) * position.quantity
                    
                    # Check stop loss and take profit
                    await self._check_position_exits(position)
            
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
    
    async def _check_position_exits(self, position: Position):
        """Check if position should be closed based on stop loss or take profit"""
        try:
            should_close = False
            close_reason = ""
            
            if position.side == "buy":
                # Check stop loss
                if position.current_price <= position.stop_loss:
                    should_close = True
                    close_reason = "Stop loss triggered"
                # Check take profit
                elif position.current_price >= position.take_profit:
                    should_close = True
                    close_reason = "Take profit triggered"
            else:
                # Check stop loss for short position
                if position.current_price >= position.stop_loss:
                    should_close = True
                    close_reason = "Stop loss triggered"
                # Check take profit for short position
                elif position.current_price <= position.take_profit:
                    should_close = True
                    close_reason = "Take profit triggered"
            
            if should_close:
                await self._close_position(position, close_reason)
            
        except Exception as e:
            self.logger.error(f"Error checking position exits: {e}")
    
    async def _close_position(self, position: Position, reason: str):
        """Close a position"""
        try:
            # Get exchange
            exchange = self.exchange_abstraction.get_exchange(position.exchange)
            
            # Create close order
            close_order = Order(
                order_id=str(uuid.uuid4()),
                symbol=position.symbol,
                exchange=position.exchange,
                order_type=OrderType.MARKET,
                side="sell" if position.side == "buy" else "buy",
                quantity=position.quantity,
                price=position.current_price,
                status=OrderStatus.PENDING,
                timestamp=datetime.now(),
                strategy_id=position.strategy_id,
                reasoning=reason
            )
            
            # Submit close order
            result = await exchange.submit_order(close_order)
            
            if result.get("success"):
                # Update trade record
                for trade in self.trade_history:
                    if (trade.symbol == position.symbol and 
                        trade.strategy_id == position.strategy_id and 
                        trade.exit_price is None):
                        trade.exit_price = result.get("filled_price", position.current_price)
                        trade.exit_time = datetime.now()
                        trade.pnl = position.unrealized_pnl
                        break
                
                # Remove position
                del self.active_positions[position.position_id]
                
                self.logger.info(f"Position closed: {position.symbol} PnL: {position.unrealized_pnl:.2f} ({reason})")
                
            else:
                self.logger.error(f"Failed to close position: {result.get('error', 'Unknown error')}")
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
    
    def _validate_signal(self, signal: TradingSignal) -> bool:
        """Validate a trading signal"""
        # Basic validation
        if not signal.symbol or not signal.exchange:
            return False
        
        if signal.confidence < 0.6:
            return False
        
        if signal.signal_type not in ["buy", "sell"]:
            return False
        
        # Check if strategy is active
        if signal.strategy_id not in self.strategies:
            return False
        
        if self.strategy_states[signal.strategy_id] != StrategyState.ACTIVE:
            return False
        
        return True
    
    async def _calculate_position_size(self, signal: TradingSignal) -> float:
        """Calculate position size based on risk management"""
        try:
            strategy = self.strategies[signal.strategy_id]
            
            # Get portfolio value
            portfolio_value = await self.exchange_abstraction.get_portfolio_value(signal.exchange)
            
            # Calculate position size based on risk
            max_position_value = portfolio_value * strategy.risk_limits.get("max_position_size", 0.02)
            
            # Calculate quantity based on entry price
            quantity = max_position_value / signal.entry_price
            
            return quantity
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    async def _check_risk_limits(self):
        """Check global risk limits"""
        try:
            # Calculate total PnL
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.active_positions.values())
            
            # Check daily loss limit
            if total_unrealized_pnl < -self.config.trading.max_daily_loss:
                self.logger.warning("Daily loss limit exceeded, pausing trading")
                await self._pause_all_strategies()
            
            # Check maximum drawdown
            if total_unrealized_pnl < -self.config.trading.max_drawdown:
                self.logger.error("Maximum drawdown exceeded, stopping all trading")
                await self._emergency_stop()
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
    
    async def _pause_all_strategies(self):
        """Pause all active strategies"""
        for strategy_id in self.strategy_states.keys():
            if self.strategy_states[strategy_id] == StrategyState.ACTIVE:
                self.strategy_states[strategy_id] = StrategyState.PAUSED
    
    async def _emergency_stop(self):
        """Emergency stop - close all positions and stop trading"""
        # Close all positions
        for position in list(self.active_positions.values()):
            await self._close_position(position, "Emergency stop")
        
        # Stop all strategies
        for strategy_id in self.strategy_states.keys():
            self.strategy_states[strategy_id] = StrategyState.STOPPED
        
        # Stop engine
        await self.stop()
    
    async def _load_existing_positions(self):
        """Load existing positions from exchanges"""
        # This would typically load from database or exchange APIs
        pass
    
    async def _initialize_performance_metrics(self):
        """Initialize performance tracking"""
        self.performance_metrics = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "total_pnl": 0.0
        }
    
    async def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            # Calculate metrics from trade history
            closed_trades = [t for t in self.trade_history if t.exit_price is not None]
            
            if not closed_trades:
                return
            
            winning_trades = [t for t in closed_trades if t.pnl > 0]
            losing_trades = [t for t in closed_trades if t.pnl < 0]
            
            self.performance_metrics.update({
                "total_trades": len(closed_trades),
                "winning_trades": len(winning_trades),
                "losing_trades": len(losing_trades),
                "win_rate": len(winning_trades) / len(closed_trades) if closed_trades else 0,
                "avg_win": np.mean([t.pnl for t in winning_trades]) if winning_trades else 0,
                "avg_loss": np.mean([t.pnl for t in losing_trades]) if losing_trades else 0,
                "total_pnl": sum(t.pnl for t in closed_trades)
            })
            
            # Calculate profit factor
            total_wins = sum(t.pnl for t in winning_trades)
            total_losses = abs(sum(t.pnl for t in losing_trades))
            
            if total_losses > 0:
                self.performance_metrics["profit_factor"] = total_wins / total_losses
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    async def _close_all_positions(self):
        """Close all active positions"""
        for position in list(self.active_positions.values()):
            await self._close_position(position, "Engine shutdown")
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.stop()
        await self.exchange_abstraction.cleanup()
    
    # Public API methods
    def get_status(self) -> Dict[str, Any]:
        """Get current engine status"""
        return {
            "state": self.state.value,
            "active_strategies": len([s for s in self.strategy_states.values() if s == StrategyState.ACTIVE]),
            "active_positions": len(self.active_positions),
            "pending_orders": len(self.pending_orders),
            "signals_queue": len(self.signals_queue),
            "performance": self.performance_metrics
        }
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get all active positions"""
        return [
            {
                "position_id": pos.position_id,
                "symbol": pos.symbol,
                "exchange": pos.exchange,
                "side": pos.side,
                "quantity": pos.quantity,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "unrealized_pnl": pos.unrealized_pnl,
                "strategy_id": pos.strategy_id
            }
            for pos in self.active_positions.values()
        ]
    
    def get_trades(self) -> List[Dict[str, Any]]:
        """Get trade history"""
        return [
            {
                "trade_id": trade.trade_id,
                "symbol": trade.symbol,
                "exchange": trade.exchange,
                "side": trade.side,
                "quantity": trade.quantity,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "pnl": trade.pnl,
                "entry_time": trade.entry_time.isoformat(),
                "exit_time": trade.exit_time.isoformat() if trade.exit_time else None,
                "strategy_id": trade.strategy_id
            }
            for trade in self.trade_history
        ]
    
    async def add_strategy(self, strategy_config: Dict[str, Any]) -> str:
        """Add a new trading strategy"""
        strategy = StrategyConfig(**strategy_config)
        self.strategies[strategy.strategy_id] = strategy
        self.strategy_states[strategy.strategy_id] = StrategyState.STOPPED
        
        if strategy.enabled and self.state == TradingState.RUNNING:
            self.strategy_states[strategy.strategy_id] = StrategyState.ACTIVE
        
        return strategy.strategy_id
    
    async def remove_strategy(self, strategy_id: str) -> bool:
        """Remove a trading strategy"""
        if strategy_id in self.strategies:
            self.strategy_states[strategy_id] = StrategyState.STOPPED
            del self.strategies[strategy_id]
            del self.strategy_states[strategy_id]
            return True
        return False
    
    async def update_strategy(self, strategy_id: str, updates: Dict[str, Any]) -> bool:
        """Update a trading strategy"""
        if strategy_id in self.strategies:
            strategy = self.strategies[strategy_id]
            for key, value in updates.items():
                if hasattr(strategy, key):
                    setattr(strategy, key, value)
            return True
        return False
