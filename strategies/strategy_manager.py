"""
Strategy Manager for coordinating multiple trading strategies
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field

from .base_strategy import BaseStrategy
from .swing_strategy import SwingStrategy
from .scalping_strategy import ScalpingStrategy
from .options_strategy import OptionsStrategy
from .intraday_strategy import IntradayStrategy
from ..exchanges.exchange_manager import ExchangeManager
from ..config import Config
from ..utils.logger import get_logger

@dataclass
class StrategyInstance:
    """Running strategy instance"""
    strategy_id: str
    strategy: BaseStrategy
    exchange_name: str
    symbol: str
    is_active: bool = True
    start_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)

class StrategyManager:
    """Manages multiple trading strategies across different exchanges"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger("StrategyManager")
        
        # Strategy registry
        self.strategy_classes = {
            'swing_trade': SwingStrategy,
            'scalping': ScalpingStrategy,
            'options': OptionsStrategy,
            'intraday': IntradayStrategy
        }
        
        # Active strategies
        self.active_strategies: Dict[str, StrategyInstance] = {}
        
        # Performance tracking
        self.strategy_metrics = {}
        
        # Strategy coordination
        self.max_strategies_per_exchange = config.MAX_POSITIONS
        self.max_total_strategies = config.MAX_POSITIONS * 2
        
    def get_available_strategies(self) -> List[Dict[str, Any]]:
        """Get list of available strategy types"""
        strategies = []
        
        for strategy_name, strategy_class in self.strategy_classes.items():
            strategy_config = self.config.get_strategy_config(strategy_name)
            if strategy_config and strategy_config.enabled:
                strategies.append({
                    'name': strategy_name,
                    'display_name': strategy_name.replace('_', ' ').title(),
                    'description': strategy_class.__doc__ or f"{strategy_name} trading strategy",
                    'enabled': True,
                    'risk_level': strategy_config.risk_level,
                    'max_position_size': strategy_config.max_position_size,
                    'category': self._get_strategy_category(strategy_name)
                })
        
        return strategies
    
    def _get_strategy_category(self, strategy_name: str) -> str:
        """Get strategy category for UI grouping"""
        category_map = {
            'swing_trade': 'Swing Trade',
            'scalping': 'Scalping',
            'options': 'Options',
            'intraday': 'Intraday'
        }
        return category_map.get(strategy_name, 'Other')
    
    async def create_strategy(self, strategy_config: Dict[str, Any]) -> BaseStrategy:
        """Create a new strategy instance"""
        try:
            strategy_type = strategy_config['strategy_type']
            exchange_name = strategy_config['exchange']
            symbol = strategy_config['symbol']
            
            # Validate strategy type
            if strategy_type not in self.strategy_classes:
                raise ValueError(f"Unknown strategy type: {strategy_type}")
            
            # Get strategy class
            strategy_class = self.strategy_classes[strategy_type]
            
            # Get exchange instance (this would come from ExchangeManager)
            # For now, we'll pass None and handle it in the strategy
            exchange = None
            
            # Create strategy instance
            strategy = strategy_class(strategy_config, exchange)
            
            # Generate unique strategy ID
            strategy_id = f"{strategy_type}_{exchange_name}_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Store strategy instance
            strategy_instance = StrategyInstance(
                strategy_id=strategy_id,
                strategy=strategy,
                exchange_name=exchange_name,
                symbol=symbol
            )
            
            self.active_strategies[strategy_id] = strategy_instance
            
            self.logger.info(f"Created strategy: {strategy_id}")
            return strategy
            
        except Exception as e:
            self.logger.error(f"Failed to create strategy: {e}")
            raise
    
    async def start_strategy(self, strategy_id: str) -> bool:
        """Start a specific strategy"""
        try:
            if strategy_id not in self.active_strategies:
                raise ValueError(f"Strategy {strategy_id} not found")
            
            strategy_instance = self.active_strategies[strategy_id]
            
            if strategy_instance.is_active:
                self.logger.warning(f"Strategy {strategy_id} is already active")
                return True
            
            # Start the strategy
            await strategy_instance.strategy.start()
            strategy_instance.is_active = True
            strategy_instance.last_update = datetime.now()
            
            self.logger.info(f"Started strategy: {strategy_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start strategy {strategy_id}: {e}")
            return False
    
    async def stop_strategy(self, strategy_id: str) -> bool:
        """Stop a specific strategy"""
        try:
            if strategy_id not in self.active_strategies:
                raise ValueError(f"Strategy {strategy_id} not found")
            
            strategy_instance = self.active_strategies[strategy_id]
            
            if not strategy_instance.is_active:
                self.logger.warning(f"Strategy {strategy_id} is already stopped")
                return True
            
            # Stop the strategy
            await strategy_instance.strategy.stop()
            strategy_instance.is_active = False
            strategy_instance.last_update = datetime.now()
            
            self.logger.info(f"Stopped strategy: {strategy_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop strategy {strategy_id}: {e}")
            return False
    
    async def stop_all_strategies(self):
        """Stop all active strategies"""
        self.logger.info("Stopping all strategies...")
        
        stop_tasks = []
        for strategy_id, strategy_instance in self.active_strategies.items():
            if strategy_instance.is_active:
                stop_tasks.append(self.stop_strategy(strategy_id))
        
        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        self.logger.info("All strategies stopped")
    
    async def remove_strategy(self, strategy_id: str) -> bool:
        """Remove a strategy completely"""
        try:
            if strategy_id not in self.active_strategies:
                raise ValueError(f"Strategy {strategy_id} not found")
            
            # Stop the strategy first
            await self.stop_strategy(strategy_id)
            
            # Remove from active strategies
            del self.active_strategies[strategy_id]
            
            # Clean up metrics
            if strategy_id in self.strategy_metrics:
                del self.strategy_metrics[strategy_id]
            
            self.logger.info(f"Removed strategy: {strategy_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove strategy {strategy_id}: {e}")
            return False
    
    def get_strategy_status(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific strategy"""
        if strategy_id not in self.active_strategies:
            return None
        
        strategy_instance = self.active_strategies[strategy_id]
        strategy_metrics = strategy_instance.strategy.get_metrics()
        
        return {
            'strategy_id': strategy_id,
            'strategy_type': strategy_instance.strategy.name,
            'exchange': strategy_instance.exchange_name,
            'symbol': strategy_instance.symbol,
            'is_active': strategy_instance.is_active,
            'start_time': strategy_instance.start_time,
            'last_update': strategy_instance.last_update,
            'metrics': strategy_metrics
        }
    
    def get_all_strategies_status(self) -> List[Dict[str, Any]]:
        """Get status of all strategies"""
        strategies_status = []
        
        for strategy_id in self.active_strategies:
            status = self.get_strategy_status(strategy_id)
            if status:
                strategies_status.append(status)
        
        return strategies_status
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get aggregated performance metrics"""
        try:
            total_strategies = len(self.active_strategies)
            active_strategies = sum(1 for s in self.active_strategies.values() if s.is_active)
            
            # Aggregate metrics from all strategies
            total_trades = 0
            total_profit = 0.0
            winning_trades = 0
            losing_trades = 0
            
            strategy_performance = {}
            
            for strategy_id, strategy_instance in self.active_strategies.items():
                metrics = strategy_instance.strategy.get_metrics()
                
                total_trades += metrics.get('total_trades', 0)
                total_profit += metrics.get('total_profit', 0.0)
                winning_trades += metrics.get('winning_trades', 0)
                losing_trades += metrics.get('losing_trades', 0)
                
                strategy_performance[strategy_id] = {
                    'name': strategy_instance.strategy.name,
                    'symbol': strategy_instance.symbol,
                    'exchange': strategy_instance.exchange_name,
                    'total_trades': metrics.get('total_trades', 0),
                    'total_profit': metrics.get('total_profit', 0.0),
                    'win_rate': metrics.get('win_rate', 0.0),
                    'is_active': strategy_instance.is_active
                }
            
            # Calculate overall metrics
            win_rate = (winning_trades / max(total_trades, 1)) * 100
            avg_profit_per_trade = total_profit / max(total_trades, 1)
            
            return {
                'total_strategies': total_strategies,
                'active_strategies': active_strategies,
                'total_trades': total_trades,
                'total_profit': total_profit,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'avg_profit_per_trade': avg_profit_per_trade,
                'strategy_performance': strategy_performance
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    async def rebalance_strategies(self):
        """Rebalance active strategies based on performance"""
        try:
            self.logger.info("Rebalancing strategies...")
            
            # Get performance metrics
            performance = await self.get_performance_metrics()
            
            # Identify underperforming strategies
            underperforming_strategies = []
            for strategy_id, perf in performance.get('strategy_performance', {}).items():
                if perf['total_trades'] > 10 and perf['win_rate'] < 30:  # Less than 30% win rate
                    underperforming_strategies.append(strategy_id)
            
            # Stop underperforming strategies
            for strategy_id in underperforming_strategies:
                self.logger.warning(f"Stopping underperforming strategy: {strategy_id}")
                await self.stop_strategy(strategy_id)
            
            # Strategy rebalancing logic could be more sophisticated
            # For now, we just stop poor performers
            
            self.logger.info(f"Rebalancing complete. Stopped {len(underperforming_strategies)} strategies")
            
        except Exception as e:
            self.logger.error(f"Error rebalancing strategies: {e}")
    
    async def update_strategy_config(self, strategy_id: str, new_config: Dict[str, Any]) -> bool:
        """Update configuration for a running strategy"""
        try:
            if strategy_id not in self.active_strategies:
                raise ValueError(f"Strategy {strategy_id} not found")
            
            strategy_instance = self.active_strategies[strategy_id]
            
            # Update strategy configuration
            await strategy_instance.strategy.update_configuration(new_config)
            strategy_instance.last_update = datetime.now()
            
            self.logger.info(f"Updated configuration for strategy: {strategy_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update strategy config {strategy_id}: {e}")
            return False
    
    def get_strategy_by_symbol(self, symbol: str) -> List[StrategyInstance]:
        """Get all strategies trading a specific symbol"""
        return [
            strategy for strategy in self.active_strategies.values()
            if strategy.symbol == symbol
        ]
    
    def get_strategy_by_exchange(self, exchange_name: str) -> List[StrategyInstance]:
        """Get all strategies on a specific exchange"""
        return [
            strategy for strategy in self.active_strategies.values()
            if strategy.exchange_name == exchange_name
        ]
    
    async def emergency_stop_all(self):
        """Emergency stop all strategies"""
        self.logger.critical("EMERGENCY STOP: Stopping all strategies immediately")
        
        # Stop all strategies as fast as possible
        stop_tasks = []
        for strategy_instance in self.active_strategies.values():
            if strategy_instance.is_active:
                stop_tasks.append(strategy_instance.strategy.stop())
        
        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        # Mark all as inactive
        for strategy_instance in self.active_strategies.values():
            strategy_instance.is_active = False
        
        self.logger.critical("Emergency stop complete")
    
    def get_strategy_recommendations(self, market_conditions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get strategy recommendations based on market conditions"""
        recommendations = []
        
        try:
            volatility = market_conditions.get('volatility', 0.2)
            trend = market_conditions.get('trend', 'neutral')
            volume = market_conditions.get('volume', 'normal')
            
            # Recommend strategies based on market conditions
            if volatility > 0.3:  # High volatility
                recommendations.append({
                    'strategy': 'scalping',
                    'confidence': 0.8,
                    'reason': 'High volatility favors scalping strategies'
                })
            
            if trend == 'bullish':
                recommendations.append({
                    'strategy': 'swing_trade',
                    'confidence': 0.7,
                    'reason': 'Bullish trend supports swing trading'
                })
            
            if trend == 'neutral' and volatility < 0.2:
                recommendations.append({
                    'strategy': 'options',
                    'confidence': 0.6,
                    'reason': 'Low volatility and neutral trend favor options strategies'
                })
            
            if volume == 'high':
                recommendations.append({
                    'strategy': 'intraday',
                    'confidence': 0.7,
                    'reason': 'High volume supports intraday trading'
                })
            
        except Exception as e:
            self.logger.error(f"Error generating strategy recommendations: {e}")
        
        return recommendations
