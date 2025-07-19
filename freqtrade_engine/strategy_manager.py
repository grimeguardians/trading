"""
Strategy Manager for Multi-Exchange Trading
Manages trading strategies across different exchanges with AI integration
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

from config import Config, StrategyConfig, StrategyType
from freqtrade_engine.exchange_manager import ExchangeManager
from strategies.base_strategy import BaseStrategy
from strategies.swing_trading import SwingTradingStrategy
from strategies.scalping import ScalpingStrategy
from strategies.options_trading import OptionsStrategy
from strategies.intraday import IntradayStrategy
from ai_engine.digital_brain import DigitalBrain
from mathematical_models.fibonacci import FibonacciCalculator
from mathematical_models.technical_indicators import TechnicalIndicators
from utils.logger import get_logger

logger = get_logger(__name__)

class StrategyStatus(Enum):
    """Strategy execution status"""
    INACTIVE = "inactive"
    STARTING = "starting"
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"

@dataclass
class RunningStrategy:
    """Information about a running strategy"""
    strategy_id: str
    strategy_name: str
    strategy_type: StrategyType
    exchange: str
    symbol: str
    status: StrategyStatus
    start_time: datetime
    last_signal_time: Optional[datetime] = None
    trades_executed: int = 0
    total_pnl: float = 0.0
    current_position: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class StrategyManager:
    """
    Manages trading strategies across multiple exchanges
    Integrates with Digital Brain for AI-powered decision making
    """
    
    def __init__(self, config: Config, exchange_manager: ExchangeManager, digital_brain: DigitalBrain):
        self.config = config
        self.exchange_manager = exchange_manager
        self.digital_brain = digital_brain
        
        # Strategy registry
        self.available_strategies: Dict[str, type] = {}
        self.running_strategies: Dict[str, RunningStrategy] = {}
        self.strategy_instances: Dict[str, BaseStrategy] = {}
        
        # Mathematical models
        self.fibonacci_calculator = FibonacciCalculator()
        self.technical_indicators = TechnicalIndicators()
        
        # Performance tracking
        self.strategy_performance: Dict[str, Dict[str, Any]] = {}
        
        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.monitoring_active = False
        
        # Signal generation
        self.signal_generation_tasks: Dict[str, asyncio.Task] = {}
        
        logger.info("StrategyManager initialized")
    
    async def initialize(self):
        """Initialize strategy manager"""
        try:
            logger.info("Initializing StrategyManager...")
            
            # Register available strategies
            self._register_strategies()
            
            # Load strategy configurations
            await self._load_strategy_configs()
            
            # Initialize performance tracking
            await self._initialize_performance_tracking()
            
            # Start monitoring
            await self._start_monitoring()
            
            logger.info("✅ StrategyManager initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize StrategyManager: {e}")
            raise
    
    def _register_strategies(self):
        """Register available trading strategies"""
        self.available_strategies = {
            StrategyType.SWING_TRADE.value: SwingTradingStrategy,
            StrategyType.SCALPING.value: ScalpingStrategy,
            StrategyType.OPTIONS.value: OptionsStrategy,
            StrategyType.INTRADAY.value: IntradayStrategy
        }
        
        logger.info(f"Registered {len(self.available_strategies)} strategies")
    
    async def _load_strategy_configs(self):
        """Load strategy configurations from config"""
        for strategy_name, strategy_config in self.config.strategies.items():
            if strategy_config.enabled:
                self.strategy_performance[strategy_name] = {
                    'total_signals': 0,
                    'successful_signals': 0,
                    'total_trades': 0,
                    'winning_trades': 0,
                    'total_pnl': 0.0,
                    'max_drawdown': 0.0,
                    'sharpe_ratio': 0.0,
                    'accuracy': 0.0,
                    'last_updated': datetime.utcnow()
                }
    
    async def _initialize_performance_tracking(self):
        """Initialize performance tracking"""
        # Load historical performance data from database
        # This would integrate with your database models
        pass
    
    async def _start_monitoring(self):
        """Start strategy monitoring"""
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitor_strategies())
    
    async def _monitor_strategies(self):
        """Monitor running strategies"""
        while self.monitoring_active:
            try:
                for strategy_id, running_strategy in self.running_strategies.items():
                    if running_strategy.status == StrategyStatus.ACTIVE:
                        # Check strategy health
                        await self._check_strategy_health(strategy_id)
                        
                        # Update performance metrics
                        await self._update_strategy_performance(strategy_id)
                        
                        # Check stop conditions
                        await self._check_stop_conditions(strategy_id)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Error in strategy monitoring: {e}")
                await asyncio.sleep(60)
    
    async def get_available_strategies(self) -> List[Dict[str, Any]]:
        """Get list of available strategies"""
        strategies = []
        
        for strategy_name, strategy_config in self.config.strategies.items():
            if strategy_config.enabled:
                strategies.append({
                    'name': strategy_name,
                    'display_name': strategy_config.name,
                    'category': strategy_config.category.value,
                    'description': self._get_strategy_description(strategy_name),
                    'max_risk_per_trade': strategy_config.max_risk_per_trade,
                    'min_confidence': strategy_config.min_confidence,
                    'fibonacci_enabled': strategy_config.fibonacci_enabled,
                    'technical_indicators': strategy_config.technical_indicators,
                    'supported_exchanges': self._get_supported_exchanges(strategy_name),
                    'performance': self.strategy_performance.get(strategy_name, {})
                })
        
        return strategies
    
    def _get_strategy_description(self, strategy_name: str) -> str:
        """Get strategy description"""
        descriptions = {
            'swing_trade': 'Medium-term trading strategy holding positions for days to weeks',
            'scalping': 'High-frequency trading strategy for quick profits on small price movements',
            'options': 'Options trading strategy using various options strategies',
            'intraday': 'Day trading strategy closing all positions before market close'
        }
        return descriptions.get(strategy_name, 'Trading strategy')
    
    def _get_supported_exchanges(self, strategy_name: str) -> List[str]:
        """Get supported exchanges for a strategy"""
        # This would be configured per strategy
        strategy_exchange_support = {
            'swing_trade': ['alpaca', 'binance', 'td_ameritrade', 'kucoin'],
            'scalping': ['binance', 'kucoin'],  # High-frequency needs fast execution
            'options': ['alpaca', 'td_ameritrade'],  # Options support
            'intraday': ['alpaca', 'binance', 'td_ameritrade', 'kucoin']
        }
        
        return strategy_exchange_support.get(strategy_name, [])
    
    async def start_strategy(self, strategy_name: str, exchange: str, symbol: str, 
                           params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Start a trading strategy"""
        try:
            logger.info(f"Starting strategy {strategy_name} on {exchange} for {symbol}")
            
            # Validate inputs
            if strategy_name not in self.available_strategies:
                raise ValueError(f"Strategy {strategy_name} not available")
            
            if not await self.exchange_manager.is_exchange_available(exchange):
                raise ValueError(f"Exchange {exchange} not available")
            
            if not await self.exchange_manager.is_symbol_available(exchange, symbol):
                raise ValueError(f"Symbol {symbol} not available on {exchange}")
            
            # Check if strategy is already running for this symbol
            existing_strategy = self._find_running_strategy(exchange, symbol)
            if existing_strategy:
                raise ValueError(f"Strategy already running for {symbol} on {exchange}")
            
            # Create strategy instance
            strategy_id = f"{strategy_name}_{exchange}_{symbol}_{datetime.utcnow().timestamp()}"
            strategy_config = self.config.strategies[strategy_name]
            
            strategy_class = self.available_strategies[strategy_name]
            strategy_instance = strategy_class(
                config=strategy_config,
                exchange_manager=self.exchange_manager,
                digital_brain=self.digital_brain,
                exchange=exchange,
                symbol=symbol,
                params=params or {}
            )
            
            # Initialize strategy
            await strategy_instance.initialize()
            
            # Create running strategy record
            running_strategy = RunningStrategy(
                strategy_id=strategy_id,
                strategy_name=strategy_name,
                strategy_type=strategy_config.category,
                exchange=exchange,
                symbol=symbol,
                status=StrategyStatus.STARTING,
                start_time=datetime.utcnow(),
                metadata=params or {}
            )
            
            # Store references
            self.running_strategies[strategy_id] = running_strategy
            self.strategy_instances[strategy_id] = strategy_instance
            
            # Start strategy execution
            await strategy_instance.start()
            
            # Start signal generation
            self.signal_generation_tasks[strategy_id] = asyncio.create_task(
                self._generate_signals(strategy_id)
            )
            
            # Update status
            running_strategy.status = StrategyStatus.ACTIVE
            
            logger.info(f"✅ Strategy {strategy_name} started successfully")
            
            return {
                'success': True,
                'strategy_id': strategy_id,
                'message': f'Strategy {strategy_name} started successfully',
                'details': {
                    'strategy_name': strategy_name,
                    'exchange': exchange,
                    'symbol': symbol,
                    'start_time': running_strategy.start_time.isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to start strategy {strategy_name}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def stop_strategy(self, strategy_id: str) -> Dict[str, Any]:
        """Stop a running strategy"""
        try:
            if strategy_id not in self.running_strategies:
                raise ValueError(f"Strategy {strategy_id} not found")
            
            running_strategy = self.running_strategies[strategy_id]
            strategy_instance = self.strategy_instances[strategy_id]
            
            logger.info(f"Stopping strategy {running_strategy.strategy_name}")
            
            # Update status
            running_strategy.status = StrategyStatus.STOPPING
            
            # Cancel signal generation
            if strategy_id in self.signal_generation_tasks:
                self.signal_generation_tasks[strategy_id].cancel()
                del self.signal_generation_tasks[strategy_id]
            
            # Stop strategy instance
            await strategy_instance.stop()
            
            # Clean up
            del self.running_strategies[strategy_id]
            del self.strategy_instances[strategy_id]
            
            logger.info(f"✅ Strategy {running_strategy.strategy_name} stopped successfully")
            
            return {
                'success': True,
                'message': f'Strategy {running_strategy.strategy_name} stopped successfully'
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to stop strategy {strategy_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def pause_strategy(self, strategy_id: str) -> Dict[str, Any]:
        """Pause a running strategy"""
        try:
            if strategy_id not in self.running_strategies:
                raise ValueError(f"Strategy {strategy_id} not found")
            
            running_strategy = self.running_strategies[strategy_id]
            strategy_instance = self.strategy_instances[strategy_id]
            
            # Pause strategy
            await strategy_instance.pause()
            running_strategy.status = StrategyStatus.PAUSED
            
            return {
                'success': True,
                'message': f'Strategy {running_strategy.strategy_name} paused successfully'
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to pause strategy {strategy_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def resume_strategy(self, strategy_id: str) -> Dict[str, Any]:
        """Resume a paused strategy"""
        try:
            if strategy_id not in self.running_strategies:
                raise ValueError(f"Strategy {strategy_id} not found")
            
            running_strategy = self.running_strategies[strategy_id]
            strategy_instance = self.strategy_instances[strategy_id]
            
            # Resume strategy
            await strategy_instance.resume()
            running_strategy.status = StrategyStatus.ACTIVE
            
            return {
                'success': True,
                'message': f'Strategy {running_strategy.strategy_name} resumed successfully'
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to resume strategy {strategy_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _generate_signals(self, strategy_id: str):
        """Generate trading signals for a strategy"""
        try:
            running_strategy = self.running_strategies[strategy_id]
            strategy_instance = self.strategy_instances[strategy_id]
            
            while running_strategy.status == StrategyStatus.ACTIVE:
                try:
                    # Generate signal
                    signal = await strategy_instance.generate_signal()
                    
                    if signal:
                        # Enhance signal with AI insights
                        enhanced_signal = await self._enhance_signal_with_ai(signal)
                        
                        # Add Fibonacci analysis if enabled
                        if self.config.strategies[running_strategy.strategy_name].fibonacci_enabled:
                            fibonacci_analysis = await self._add_fibonacci_analysis(enhanced_signal)
                            enhanced_signal.fibonacci_levels = fibonacci_analysis
                        
                        # Send signal to trading engine
                        await self._process_strategy_signal(strategy_id, enhanced_signal)
                        
                        # Update tracking
                        running_strategy.last_signal_time = datetime.utcnow()
                        self.strategy_performance[running_strategy.strategy_name]['total_signals'] += 1
                    
                    # Wait before next signal generation
                    await asyncio.sleep(strategy_instance.signal_interval)
                    
                except Exception as e:
                    logger.error(f"❌ Error generating signal for {strategy_id}: {e}")
                    await asyncio.sleep(30)
                    
        except asyncio.CancelledError:
            logger.info(f"Signal generation cancelled for {strategy_id}")
        except Exception as e:
            logger.error(f"❌ Error in signal generation for {strategy_id}: {e}")
            running_strategy.status = StrategyStatus.ERROR
    
    async def _enhance_signal_with_ai(self, signal):
        """Enhance trading signal with AI insights"""
        try:
            # Get AI analysis from Digital Brain
            ai_analysis = await self.digital_brain.analyze_trading_signal(signal)
            
            # Enhance signal with AI insights
            signal.ai_reasoning = ai_analysis.get('reasoning', '')
            signal.confidence = max(signal.confidence, ai_analysis.get('confidence', 0))
            
            # Add market context
            market_context = await self.digital_brain.get_market_context(signal.symbol)
            signal.metadata = signal.metadata or {}
            signal.metadata.update(market_context)
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error enhancing signal with AI: {e}")
            return signal
    
    async def _add_fibonacci_analysis(self, signal):
        """Add Fibonacci analysis to signal"""
        try:
            # Get historical data
            historical_data = await self.exchange_manager.get_historical_data(
                signal.exchange, signal.symbol, timeframe='1h', limit=100
            )
            
            if historical_data.empty:
                return {}
            
            # Calculate Fibonacci levels
            high = historical_data['high'].max()
            low = historical_data['low'].min()
            current_price = historical_data['close'].iloc[-1]
            
            fibonacci_levels = self.fibonacci_calculator.calculate_retracement_levels(
                high, low, self.config.FIBONACCI_LEVELS
            )
            
            # Add extension levels
            extension_levels = self.fibonacci_calculator.calculate_extension_levels(
                high, low, current_price
            )
            
            fibonacci_levels.update(extension_levels)
            
            return fibonacci_levels
            
        except Exception as e:
            logger.error(f"❌ Error adding Fibonacci analysis: {e}")
            return {}
    
    async def _process_strategy_signal(self, strategy_id: str, signal):
        """Process a strategy signal"""
        try:
            # Send signal to FreqtradeEngine for execution
            from freqtrade_engine.core import TradeSignal
            
            trade_signal = TradeSignal(
                signal_id=signal.signal_id,
                strategy_name=self.running_strategies[strategy_id].strategy_name,
                exchange=signal.exchange,
                symbol=signal.symbol,
                action=signal.action,
                confidence=signal.confidence,
                price=signal.price,
                quantity=signal.quantity,
                timestamp=signal.timestamp,
                fibonacci_levels=signal.fibonacci_levels,
                technical_indicators=signal.technical_indicators,
                risk_metrics=signal.risk_metrics,
                ai_reasoning=signal.ai_reasoning
            )
            
            # This would be sent to the FreqtradeEngine
            # await self.freqtrade_engine.process_trade_signal(trade_signal)
            
        except Exception as e:
            logger.error(f"❌ Error processing strategy signal: {e}")
    
    async def _check_strategy_health(self, strategy_id: str):
        """Check strategy health"""
        try:
            running_strategy = self.running_strategies[strategy_id]
            strategy_instance = self.strategy_instances[strategy_id]
            
            # Check if strategy is responsive
            health_check = await strategy_instance.health_check()
            
            if not health_check['healthy']:
                logger.warning(f"Strategy {strategy_id} health check failed: {health_check['error']}")
                running_strategy.status = StrategyStatus.ERROR
                
        except Exception as e:
            logger.error(f"❌ Error checking strategy health: {e}")
    
    async def _update_strategy_performance(self, strategy_id: str):
        """Update strategy performance metrics"""
        try:
            running_strategy = self.running_strategies[strategy_id]
            strategy_instance = self.strategy_instances[strategy_id]
            
            # Get performance metrics from strategy
            performance = await strategy_instance.get_performance_metrics()
            
            # Update running strategy
            running_strategy.trades_executed = performance.get('trades_executed', 0)
            running_strategy.total_pnl = performance.get('total_pnl', 0.0)
            
            # Update global performance tracking
            strategy_perf = self.strategy_performance[running_strategy.strategy_name]
            strategy_perf['total_trades'] = performance.get('total_trades', 0)
            strategy_perf['winning_trades'] = performance.get('winning_trades', 0)
            strategy_perf['total_pnl'] = performance.get('total_pnl', 0.0)
            strategy_perf['accuracy'] = performance.get('accuracy', 0.0)
            strategy_perf['last_updated'] = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"❌ Error updating strategy performance: {e}")
    
    async def _check_stop_conditions(self, strategy_id: str):
        """Check if strategy should be stopped"""
        try:
            running_strategy = self.running_strategies[strategy_id]
            strategy_config = self.config.strategies[running_strategy.strategy_name]
            
            # Check maximum daily loss
            if running_strategy.total_pnl < -strategy_config.max_daily_loss:
                logger.warning(f"Strategy {strategy_id} hit daily loss limit")
                await self.stop_strategy(strategy_id)
                return
            
            # Check if strategy has been running too long without signals
            if running_strategy.last_signal_time:
                time_since_signal = datetime.utcnow() - running_strategy.last_signal_time
                if time_since_signal > timedelta(hours=4):  # 4 hours without signals
                    logger.warning(f"Strategy {strategy_id} inactive for too long")
                    await self.pause_strategy(strategy_id)
                    
        except Exception as e:
            logger.error(f"❌ Error checking stop conditions: {e}")
    
    def _find_running_strategy(self, exchange: str, symbol: str) -> Optional[RunningStrategy]:
        """Find running strategy for exchange and symbol"""
        for running_strategy in self.running_strategies.values():
            if (running_strategy.exchange == exchange and 
                running_strategy.symbol == symbol and 
                running_strategy.status == StrategyStatus.ACTIVE):
                return running_strategy
        return None
    
    async def get_running_strategies(self) -> List[Dict[str, Any]]:
        """Get list of running strategies"""
        strategies = []
        
        for strategy_id, running_strategy in self.running_strategies.items():
            strategies.append({
                'strategy_id': strategy_id,
                'strategy_name': running_strategy.strategy_name,
                'strategy_type': running_strategy.strategy_type.value,
                'exchange': running_strategy.exchange,
                'symbol': running_strategy.symbol,
                'status': running_strategy.status.value,
                'start_time': running_strategy.start_time.isoformat(),
                'last_signal_time': running_strategy.last_signal_time.isoformat() if running_strategy.last_signal_time else None,
                'trades_executed': running_strategy.trades_executed,
                'total_pnl': running_strategy.total_pnl,
                'current_position': running_strategy.current_position,
                'metadata': running_strategy.metadata
            })
        
        return strategies
    
    async def get_strategy_performance(self, strategy_name: str) -> Dict[str, Any]:
        """Get performance metrics for a strategy"""
        return self.strategy_performance.get(strategy_name, {})
    
    async def stop(self):
        """Stop strategy manager"""
        try:
            logger.info("Stopping StrategyManager...")
            
            # Stop monitoring
            self.monitoring_active = False
            if self.monitoring_task:
                self.monitoring_task.cancel()
            
            # Stop all running strategies
            for strategy_id in list(self.running_strategies.keys()):
                await self.stop_strategy(strategy_id)
            
            # Cancel signal generation tasks
            for task in self.signal_generation_tasks.values():
                task.cancel()
            
            logger.info("✅ StrategyManager stopped successfully")
            
        except Exception as e:
            logger.error(f"❌ Error stopping StrategyManager: {e}")
