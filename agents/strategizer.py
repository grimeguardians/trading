"""
Strategizer Agent - Manages trading strategies and optimization
Coordinates strategy selection, backtesting, and optimization
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
import json

from agents.base_agent import BaseAgent
from strategies.swing_strategy import SwingStrategy
from strategies.scalping_strategy import ScalpingStrategy
from strategies.options_strategy import OptionsStrategy
from strategies.intraday_strategy import IntradayStrategy
from backtest.engine import BacktestEngine
from mcp_server import MessageType


@dataclass
class StrategyPerformance:
    """Strategy performance metrics"""
    strategy_name: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_trade_duration: float
    profit_factor: float
    last_updated: datetime


class Strategizer(BaseAgent):
    """
    Strategizer Agent for managing trading strategies
    Handles strategy selection, optimization, and performance monitoring
    """
    
    def __init__(self, mcp_server, knowledge_engine, config):
        super().__init__(
            agent_id="strategizer",
            agent_type="strategy_manager",
            mcp_server=mcp_server,
            knowledge_engine=knowledge_engine,
            config=config
        )
        
        # Initialize strategies
        self.strategies = {}
        self.strategy_performances = {}
        self.active_strategies = {}
        self._initialize_strategies()
        
        # Initialize backtest engine
        self.backtest_engine = BacktestEngine(config)
        
        # Strategy optimization settings
        self.optimization_schedule = {
            "daily": ["swing", "intraday"],
            "weekly": ["options"],
            "monthly": ["scalping"]
        }
        
        # Performance tracking
        self.strategy_signals = {}
        self.strategy_allocations = {}
        self.portfolio_performance = {}
        
        # Market regime detection
        self.current_market_regime = "normal"
        self.regime_strategies = {
            "bull_market": ["swing", "intraday"],
            "bear_market": ["options", "scalping"],
            "sideways": ["scalping", "options"],
            "high_volatility": ["scalping", "options"],
            "normal": ["swing", "intraday", "options"]
        }
        
        # Optimization parameters
        self.optimization_interval = 3600  # 1 hour
        self.performance_window = 30  # 30 days
        self.min_trades_for_optimization = 10
        
        # Strategy coordination
        self.strategy_coordination_enabled = True
        self.max_concurrent_strategies = 3
        
        self.logger.info("🎯 Strategizer Agent initialized with multi-strategy management")
    
    def _setup_capabilities(self):
        """Setup strategizer capabilities"""
        self.capabilities = [
            "strategy_management",
            "performance_optimization",
            "backtesting",
            "market_regime_detection",
            "strategy_coordination",
            "risk_adjusted_allocation",
            "adaptive_parameters",
            "strategy_selection"
        ]
    
    def _setup_message_handlers(self):
        """Setup message handlers"""
        self.register_message_handler("strategy_request", self._handle_strategy_request)
        self.register_message_handler("performance_request", self._handle_performance_request)
        self.register_message_handler("optimization_request", self._handle_optimization_request)
        self.register_message_handler("backtest_request", self._handle_backtest_request)
        self.register_message_handler("market_regime_update", self._handle_market_regime_update)
        self.register_message_handler("strategy_signal", self._handle_strategy_signal)
    
    def _initialize_strategies(self):
        """Initialize all available strategies"""
        try:
            # Initialize strategy instances
            self.strategies["swing"] = SwingStrategy(self.config)
            self.strategies["scalping"] = ScalpingStrategy(self.config)
            self.strategies["options"] = OptionsStrategy(self.config)
            self.strategies["intraday"] = IntradayStrategy(self.config)
            
            # Initialize performance tracking
            for strategy_name in self.strategies.keys():
                self.strategy_performances[strategy_name] = StrategyPerformance(
                    strategy_name=strategy_name,
                    total_return=0.0,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    win_rate=0.0,
                    total_trades=0,
                    avg_trade_duration=0.0,
                    profit_factor=0.0,
                    last_updated=datetime.utcnow()
                )
            
            # Activate enabled strategies
            for strategy_name, strategy_config in self.config.STRATEGIES.items():
                if strategy_config.get("enabled", False):
                    self.active_strategies[strategy_name] = self.strategies[strategy_name]
                    self.logger.info(f"✅ {strategy_name} strategy activated")
            
        except Exception as e:
            self.logger.error(f"❌ Strategy initialization failed: {e}")
            raise
    
    async def _agent_logic(self):
        """Main strategizer agent logic"""
        self.logger.info("🎯 Strategizer Agent started - managing trading strategies")
        
        # Start background tasks
        asyncio.create_task(self._strategy_optimizer())
        asyncio.create_task(self._performance_monitor())
        asyncio.create_task(self._market_regime_detector())
        asyncio.create_task(self._strategy_coordinator())
        
        while self.running:
            try:
                # Update strategy performances
                await self._update_strategy_performances()
                
                # Optimize strategy allocations
                await self._optimize_strategy_allocations()
                
                # Check for strategy signals
                await self._check_strategy_signals()
                
                # Update market regime
                await self._update_market_regime()
                
                # Coordinate strategies
                await self._coordinate_strategies()
                
                # Wait for next cycle
                await asyncio.sleep(300)  # 5 minute cycle
                
            except Exception as e:
                self.logger.error(f"❌ Strategizer agent error: {e}")
                await asyncio.sleep(60)
    
    async def _strategy_optimizer(self):
        """Optimize strategy parameters"""
        while self.running:
            try:
                for strategy_name, strategy in self.active_strategies.items():
                    # Check if optimization is needed
                    if await self._needs_optimization(strategy_name):
                        await self._optimize_strategy(strategy_name)
                
                await asyncio.sleep(self.optimization_interval)
                
            except Exception as e:
                self.logger.error(f"❌ Strategy optimizer error: {e}")
                await asyncio.sleep(self.optimization_interval)
    
    async def _needs_optimization(self, strategy_name: str) -> bool:
        """Check if strategy needs optimization"""
        try:
            performance = self.strategy_performances.get(strategy_name)
            
            if not performance:
                return True
            
            # Check if enough time has passed
            time_since_last_update = datetime.utcnow() - performance.last_updated
            if time_since_last_update.total_seconds() < self.optimization_interval:
                return False
            
            # Check if enough trades have been made
            if performance.total_trades < self.min_trades_for_optimization:
                return False
            
            # Check if performance is declining
            if performance.total_return < -0.05:  # 5% loss threshold
                return True
            
            # Check if Sharpe ratio is too low
            if performance.sharpe_ratio < 0.5:
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Optimization check error for {strategy_name}: {e}")
            return False
    
    async def _optimize_strategy(self, strategy_name: str):
        """Optimize individual strategy"""
        try:
            self.logger.info(f"🔧 Optimizing strategy: {strategy_name}")
            
            strategy = self.strategies[strategy_name]
            
            # Run backtest with current parameters
            current_performance = await self._run_strategy_backtest(strategy_name)
            
            # Get parameter ranges for optimization
            param_ranges = strategy.get_optimization_parameters()
            
            # Simple grid search optimization
            best_params = await self._grid_search_optimization(strategy_name, param_ranges)
            
            if best_params:
                # Update strategy parameters
                await strategy.update_parameters(best_params)
                
                # Update performance
                new_performance = await self._run_strategy_backtest(strategy_name)
                
                if new_performance and new_performance.total_return > current_performance.total_return:
                    self.logger.info(f"✅ Strategy {strategy_name} optimized successfully")
                    
                    # Update knowledge engine
                    await self.update_knowledge("add_node", {
                        "node_type": "strategy_optimization",
                        "strategy": strategy_name,
                        "old_performance": current_performance.__dict__,
                        "new_performance": new_performance.__dict__,
                        "optimized_parameters": best_params
                    })
                else:
                    self.logger.warning(f"⚠️ Strategy {strategy_name} optimization did not improve performance")
            
        except Exception as e:
            self.logger.error(f"❌ Strategy optimization error for {strategy_name}: {e}")
    
    async def _grid_search_optimization(self, strategy_name: str, param_ranges: Dict) -> Optional[Dict]:
        """Perform grid search optimization"""
        try:
            best_params = None
            best_performance = -float('inf')
            
            # Generate parameter combinations (simplified)
            param_combinations = self._generate_param_combinations(param_ranges)
            
            for params in param_combinations[:10]:  # Limit to 10 combinations for performance
                # Test parameters
                test_performance = await self._test_strategy_parameters(strategy_name, params)
                
                if test_performance and test_performance.total_return > best_performance:
                    best_performance = test_performance.total_return
                    best_params = params
            
            return best_params
            
        except Exception as e:
            self.logger.error(f"❌ Grid search optimization error: {e}")
            return None
    
    def _generate_param_combinations(self, param_ranges: Dict) -> List[Dict]:
        """Generate parameter combinations for testing"""
        try:
            combinations = []
            
            # Simplified parameter combination generation
            # In production, this would use more sophisticated optimization
            
            for param_name, (min_val, max_val, step) in param_ranges.items():
                if isinstance(min_val, (int, float)):
                    values = np.arange(min_val, max_val + step, step)
                    for value in values[:5]:  # Limit values
                        combinations.append({param_name: value})
            
            return combinations[:10]  # Limit combinations
            
        except Exception as e:
            self.logger.error(f"❌ Parameter combination generation error: {e}")
            return []
    
    async def _test_strategy_parameters(self, strategy_name: str, params: Dict) -> Optional[StrategyPerformance]:
        """Test strategy with specific parameters"""
        try:
            # Create temporary strategy instance with test parameters
            strategy = self.strategies[strategy_name]
            original_params = strategy.get_current_parameters()
            
            # Apply test parameters
            await strategy.update_parameters(params)
            
            # Run backtest
            performance = await self._run_strategy_backtest(strategy_name)
            
            # Restore original parameters
            await strategy.update_parameters(original_params)
            
            return performance
            
        except Exception as e:
            self.logger.error(f"❌ Strategy parameter testing error: {e}")
            return None
    
    async def _run_strategy_backtest(self, strategy_name: str) -> Optional[StrategyPerformance]:
        """Run backtest for strategy"""
        try:
            strategy = self.strategies[strategy_name]
            
            # Set backtest parameters
            start_date = datetime.utcnow() - timedelta(days=self.performance_window)
            end_date = datetime.utcnow()
            
            # Run backtest
            backtest_results = await self.backtest_engine.run_backtest(
                strategy=strategy,
                start_date=start_date,
                end_date=end_date,
                initial_capital=100000
            )
            
            if backtest_results:
                return StrategyPerformance(
                    strategy_name=strategy_name,
                    total_return=backtest_results.get("total_return", 0.0),
                    sharpe_ratio=backtest_results.get("sharpe_ratio", 0.0),
                    max_drawdown=backtest_results.get("max_drawdown", 0.0),
                    win_rate=backtest_results.get("win_rate", 0.0),
                    total_trades=backtest_results.get("total_trades", 0),
                    avg_trade_duration=backtest_results.get("avg_trade_duration", 0.0),
                    profit_factor=backtest_results.get("profit_factor", 0.0),
                    last_updated=datetime.utcnow()
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Strategy backtest error for {strategy_name}: {e}")
            return None
    
    async def _performance_monitor(self):
        """Monitor strategy performances"""
        while self.running:
            try:
                # Update all strategy performances
                for strategy_name in self.active_strategies.keys():
                    performance = await self._calculate_strategy_performance(strategy_name)
                    
                    if performance:
                        self.strategy_performances[strategy_name] = performance
                        
                        # Send performance update
                        await self.broadcast_message({
                            "type": "strategy_performance_update",
                            "strategy": strategy_name,
                            "performance": performance.__dict__
                        })
                
                # Generate performance report
                await self._generate_performance_report()
                
                await asyncio.sleep(600)  # Update every 10 minutes
                
            except Exception as e:
                self.logger.error(f"❌ Performance monitor error: {e}")
                await asyncio.sleep(600)
    
    async def _calculate_strategy_performance(self, strategy_name: str) -> Optional[StrategyPerformance]:
        """Calculate real-time strategy performance"""
        try:
            # Get strategy trades from knowledge engine
            trades = await self._get_strategy_trades(strategy_name)
            
            if not trades:
                return None
            
            # Calculate performance metrics
            returns = [trade.get("return", 0.0) for trade in trades]
            total_return = sum(returns)
            
            # Calculate Sharpe ratio
            if len(returns) > 1:
                returns_std = np.std(returns)
                sharpe_ratio = np.mean(returns) / returns_std if returns_std > 0 else 0.0
            else:
                sharpe_ratio = 0.0
            
            # Calculate max drawdown
            cumulative_returns = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = cumulative_returns - running_max
            max_drawdown = np.min(drawdown)
            
            # Calculate win rate
            winning_trades = sum(1 for r in returns if r > 0)
            win_rate = winning_trades / len(returns) if returns else 0.0
            
            # Calculate average trade duration
            durations = [trade.get("duration", 0.0) for trade in trades]
            avg_trade_duration = np.mean(durations) if durations else 0.0
            
            # Calculate profit factor
            gross_profit = sum(r for r in returns if r > 0)
            gross_loss = abs(sum(r for r in returns if r < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
            
            return StrategyPerformance(
                strategy_name=strategy_name,
                total_return=total_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                total_trades=len(trades),
                avg_trade_duration=avg_trade_duration,
                profit_factor=profit_factor,
                last_updated=datetime.utcnow()
            )
            
        except Exception as e:
            self.logger.error(f"❌ Performance calculation error for {strategy_name}: {e}")
            return None
    
    async def _get_strategy_trades(self, strategy_name: str) -> List[Dict]:
        """Get trades for specific strategy"""
        try:
            # Query knowledge engine for strategy trades
            trades = await self.query_knowledge(
                f"strategy_trades_{strategy_name}",
                "search"
            )
            
            # For now, return empty list
            # In production, this would query the actual trade database
            return []
            
        except Exception as e:
            self.logger.error(f"❌ Strategy trades query error: {e}")
            return []
    
    async def _generate_performance_report(self):
        """Generate comprehensive performance report"""
        try:
            report = {
                "timestamp": datetime.utcnow().isoformat(),
                "active_strategies": len(self.active_strategies),
                "total_strategies": len(self.strategies),
                "performances": {},
                "market_regime": self.current_market_regime,
                "recommendations": []
            }
            
            # Add performance data
            for strategy_name, performance in self.strategy_performances.items():
                report["performances"][strategy_name] = performance.__dict__
            
            # Add recommendations
            report["recommendations"] = await self._generate_recommendations()
            
            # Store in knowledge engine
            await self.update_knowledge("add_node", {
                "node_type": "performance_report",
                "report": report
            })
            
            # Broadcast report
            await self.broadcast_message({
                "type": "performance_report",
                "report": report
            })
            
        except Exception as e:
            self.logger.error(f"❌ Performance report generation error: {e}")
    
    async def _generate_recommendations(self) -> List[str]:
        """Generate strategy recommendations"""
        try:
            recommendations = []
            
            # Analyze performance and generate recommendations
            for strategy_name, performance in self.strategy_performances.items():
                if performance.total_return < -0.05:  # 5% loss
                    recommendations.append(f"Consider disabling {strategy_name} strategy due to poor performance")
                elif performance.sharpe_ratio > 2.0:
                    recommendations.append(f"Consider increasing allocation to {strategy_name} strategy")
                elif performance.win_rate < 0.3:
                    recommendations.append(f"Review {strategy_name} strategy parameters")
            
            # Market regime recommendations
            if self.current_market_regime == "high_volatility":
                recommendations.append("Consider using volatility-based strategies")
            elif self.current_market_regime == "bull_market":
                recommendations.append("Consider increasing long positions")
            elif self.current_market_regime == "bear_market":
                recommendations.append("Consider defensive strategies")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"❌ Recommendation generation error: {e}")
            return []
    
    async def _market_regime_detector(self):
        """Detect current market regime"""
        while self.running:
            try:
                # Request market analysis from market analyst
                await self.send_direct_message("market_analyst", {
                    "type": "market_regime_request"
                })
                
                await asyncio.sleep(1800)  # Check every 30 minutes
                
            except Exception as e:
                self.logger.error(f"❌ Market regime detector error: {e}")
                await asyncio.sleep(1800)
    
    async def _update_market_regime(self):
        """Update market regime and adjust strategies"""
        try:
            # Get current market regime (simplified)
            # In production, this would use more sophisticated analysis
            
            # Adjust active strategies based on regime
            if self.current_market_regime in self.regime_strategies:
                recommended_strategies = self.regime_strategies[self.current_market_regime]
                
                # Activate recommended strategies
                for strategy_name in recommended_strategies:
                    if strategy_name in self.strategies and strategy_name not in self.active_strategies:
                        self.active_strategies[strategy_name] = self.strategies[strategy_name]
                        self.logger.info(f"✅ Activated {strategy_name} for {self.current_market_regime} regime")
                
                # Deactivate non-recommended strategies
                strategies_to_deactivate = []
                for strategy_name in self.active_strategies:
                    if strategy_name not in recommended_strategies:
                        strategies_to_deactivate.append(strategy_name)
                
                for strategy_name in strategies_to_deactivate:
                    del self.active_strategies[strategy_name]
                    self.logger.info(f"🔄 Deactivated {strategy_name} for {self.current_market_regime} regime")
                    
        except Exception as e:
            self.logger.error(f"❌ Market regime update error: {e}")
    
    async def _strategy_coordinator(self):
        """Coordinate multiple strategies"""
        while self.running:
            try:
                if self.strategy_coordination_enabled:
                    await self._coordinate_strategies()
                
                await asyncio.sleep(300)  # Coordinate every 5 minutes
                
            except Exception as e:
                self.logger.error(f"❌ Strategy coordinator error: {e}")
                await asyncio.sleep(300)
    
    async def _coordinate_strategies(self):
        """Coordinate active strategies"""
        try:
            # Limit concurrent strategies
            if len(self.active_strategies) > self.max_concurrent_strategies:
                await self._reduce_active_strategies()
            
            # Balance strategy allocations
            await self._balance_strategy_allocations()
            
            # Check for strategy conflicts
            await self._check_strategy_conflicts()
            
        except Exception as e:
            self.logger.error(f"❌ Strategy coordination error: {e}")
    
    async def _reduce_active_strategies(self):
        """Reduce number of active strategies"""
        try:
            # Sort strategies by performance
            strategy_performance_pairs = [
                (name, self.strategy_performances[name].total_return)
                for name in self.active_strategies.keys()
                if name in self.strategy_performances
            ]
            
            # Sort by performance (descending)
            strategy_performance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Keep only top performing strategies
            strategies_to_keep = strategy_performance_pairs[:self.max_concurrent_strategies]
            
            # Deactivate poor performing strategies
            for strategy_name in list(self.active_strategies.keys()):
                if strategy_name not in [s[0] for s in strategies_to_keep]:
                    del self.active_strategies[strategy_name]
                    self.logger.info(f"🔄 Deactivated {strategy_name} due to poor performance")
                    
        except Exception as e:
            self.logger.error(f"❌ Strategy reduction error: {e}")
    
    async def _balance_strategy_allocations(self):
        """Balance capital allocation across strategies"""
        try:
            if not self.active_strategies:
                return
            
            # Calculate allocation weights based on performance
            total_performance = sum(
                max(0.01, self.strategy_performances[name].total_return)
                for name in self.active_strategies.keys()
                if name in self.strategy_performances
            )
            
            # Calculate allocations
            for strategy_name in self.active_strategies.keys():
                if strategy_name in self.strategy_performances:
                    performance = max(0.01, self.strategy_performances[strategy_name].total_return)
                    allocation = performance / total_performance
                    self.strategy_allocations[strategy_name] = allocation
                else:
                    self.strategy_allocations[strategy_name] = 1.0 / len(self.active_strategies)
            
            # Broadcast allocation update
            await self.broadcast_message({
                "type": "strategy_allocation_update",
                "allocations": self.strategy_allocations
            })
            
        except Exception as e:
            self.logger.error(f"❌ Strategy allocation balancing error: {e}")
    
    async def _check_strategy_conflicts(self):
        """Check for conflicts between strategies"""
        try:
            # Check for opposing signals
            conflicting_pairs = []
            
            for strategy1 in self.active_strategies.keys():
                for strategy2 in self.active_strategies.keys():
                    if strategy1 != strategy2:
                        conflict = await self._check_strategy_pair_conflict(strategy1, strategy2)
                        if conflict:
                            conflicting_pairs.append((strategy1, strategy2))
            
            # Handle conflicts
            if conflicting_pairs:
                await self._resolve_strategy_conflicts(conflicting_pairs)
                
        except Exception as e:
            self.logger.error(f"❌ Strategy conflict check error: {e}")
    
    async def _check_strategy_pair_conflict(self, strategy1: str, strategy2: str) -> bool:
        """Check if two strategies conflict"""
        try:
            # Get recent signals from both strategies
            signals1 = self.strategy_signals.get(strategy1, [])
            signals2 = self.strategy_signals.get(strategy2, [])
            
            # Check for opposing signals on same symbol
            for signal1 in signals1[-5:]:  # Last 5 signals
                for signal2 in signals2[-5:]:
                    if (signal1.get("symbol") == signal2.get("symbol") and
                        signal1.get("signal_type") != signal2.get("signal_type")):
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Strategy pair conflict check error: {e}")
            return False
    
    async def _resolve_strategy_conflicts(self, conflicting_pairs: List[Tuple[str, str]]):
        """Resolve strategy conflicts"""
        try:
            for strategy1, strategy2 in conflicting_pairs:
                # Choose better performing strategy
                perf1 = self.strategy_performances.get(strategy1)
                perf2 = self.strategy_performances.get(strategy2)
                
                if perf1 and perf2:
                    if perf1.total_return > perf2.total_return:
                        # Reduce allocation to strategy2
                        self.strategy_allocations[strategy2] *= 0.5
                    else:
                        # Reduce allocation to strategy1
                        self.strategy_allocations[strategy1] *= 0.5
                
                self.logger.info(f"⚠️ Resolved conflict between {strategy1} and {strategy2}")
                
        except Exception as e:
            self.logger.error(f"❌ Strategy conflict resolution error: {e}")
    
    async def _update_strategy_performances(self):
        """Update strategy performances"""
        try:
            for strategy_name in self.active_strategies.keys():
                performance = await self._calculate_strategy_performance(strategy_name)
                if performance:
                    self.strategy_performances[strategy_name] = performance
                    
        except Exception as e:
            self.logger.error(f"❌ Strategy performance update error: {e}")
    
    async def _optimize_strategy_allocations(self):
        """Optimize capital allocation across strategies"""
        try:
            # Use Modern Portfolio Theory for allocation
            allocations = await self._calculate_optimal_allocations()
            
            if allocations:
                self.strategy_allocations = allocations
                
                # Broadcast allocation update
                await self.broadcast_message({
                    "type": "optimal_allocation_update",
                    "allocations": allocations
                })
                
        except Exception as e:
            self.logger.error(f"❌ Strategy allocation optimization error: {e}")
    
    async def _calculate_optimal_allocations(self) -> Optional[Dict[str, float]]:
        """Calculate optimal strategy allocations"""
        try:
            if len(self.active_strategies) < 2:
                return None
            
            # Simplified allocation based on Sharpe ratios
            strategy_sharpes = {}
            for strategy_name in self.active_strategies.keys():
                if strategy_name in self.strategy_performances:
                    sharpe = self.strategy_performances[strategy_name].sharpe_ratio
                    strategy_sharpes[strategy_name] = max(0.01, sharpe)
            
            # Calculate allocations
            total_sharpe = sum(strategy_sharpes.values())
            allocations = {}
            
            for strategy_name, sharpe in strategy_sharpes.items():
                allocations[strategy_name] = sharpe / total_sharpe
            
            return allocations
            
        except Exception as e:
            self.logger.error(f"❌ Optimal allocation calculation error: {e}")
            return None
    
    async def _check_strategy_signals(self):
        """Check for new strategy signals"""
        try:
            for strategy_name, strategy in self.active_strategies.items():
                # Get signals from strategy
                signals = await strategy.get_signals()
                
                if signals:
                    # Store signals
                    if strategy_name not in self.strategy_signals:
                        self.strategy_signals[strategy_name] = []
                    
                    self.strategy_signals[strategy_name].extend(signals)
                    
                    # Keep signal history manageable
                    if len(self.strategy_signals[strategy_name]) > 100:
                        self.strategy_signals[strategy_name] = self.strategy_signals[strategy_name][-50:]
                    
                    # Forward signals to trader agent
                    for signal in signals:
                        await self.send_direct_message("trader_agent", {
                            "type": "strategy_signal",
                            "signal": signal,
                            "strategy": strategy_name,
                            "allocation": self.strategy_allocations.get(strategy_name, 1.0)
                        })
                        
        except Exception as e:
            self.logger.error(f"❌ Strategy signal check error: {e}")
    
    # Message handlers
    
    async def _handle_strategy_request(self, data: Dict):
        """Handle strategy request"""
        try:
            request_type = data.get("request_type", "status")
            
            if request_type == "status":
                # Send strategy status
                status = {
                    "active_strategies": list(self.active_strategies.keys()),
                    "total_strategies": len(self.strategies),
                    "market_regime": self.current_market_regime,
                    "allocations": self.strategy_allocations
                }
                
                await self.send_direct_message(data["source"], {
                    "type": "strategy_status_response",
                    "status": status
                })
                
        except Exception as e:
            self.logger.error(f"❌ Strategy request handling error: {e}")
    
    async def _handle_performance_request(self, data: Dict):
        """Handle performance request"""
        try:
            strategy_name = data.get("strategy_name")
            
            if strategy_name and strategy_name in self.strategy_performances:
                performance = self.strategy_performances[strategy_name]
                
                await self.send_direct_message(data["source"], {
                    "type": "performance_response",
                    "strategy": strategy_name,
                    "performance": performance.__dict__
                })
            else:
                # Send all performances
                performances = {
                    name: perf.__dict__ 
                    for name, perf in self.strategy_performances.items()
                }
                
                await self.send_direct_message(data["source"], {
                    "type": "performance_response",
                    "performances": performances
                })
                
        except Exception as e:
            self.logger.error(f"❌ Performance request handling error: {e}")
    
    async def _handle_optimization_request(self, data: Dict):
        """Handle optimization request"""
        try:
            strategy_name = data.get("strategy_name")
            
            if strategy_name and strategy_name in self.strategies:
                await self._optimize_strategy(strategy_name)
                
                await self.send_direct_message(data["source"], {
                    "type": "optimization_response",
                    "strategy": strategy_name,
                    "status": "completed"
                })
                
        except Exception as e:
            self.logger.error(f"❌ Optimization request handling error: {e}")
    
    async def _handle_backtest_request(self, data: Dict):
        """Handle backtest request"""
        try:
            strategy_name = data.get("strategy_name")
            
            if strategy_name and strategy_name in self.strategies:
                performance = await self._run_strategy_backtest(strategy_name)
                
                await self.send_direct_message(data["source"], {
                    "type": "backtest_response",
                    "strategy": strategy_name,
                    "performance": performance.__dict__ if performance else None
                })
                
        except Exception as e:
            self.logger.error(f"❌ Backtest request handling error: {e}")
    
    async def _handle_market_regime_update(self, data: Dict):
        """Handle market regime update"""
        try:
            new_regime = data.get("regime")
            
            if new_regime and new_regime != self.current_market_regime:
                self.current_market_regime = new_regime
                self.logger.info(f"📊 Market regime updated to: {new_regime}")
                
                # Adjust strategies for new regime
                await self._update_market_regime()
                
        except Exception as e:
            self.logger.error(f"❌ Market regime update handling error: {e}")
    
    async def _handle_strategy_signal(self, data: Dict):
        """Handle strategy signal"""
        try:
            signal = data.get("signal")
            strategy_name = data.get("strategy")
            
            if signal and strategy_name:
                # Store signal
                if strategy_name not in self.strategy_signals:
                    self.strategy_signals[strategy_name] = []
                
                self.strategy_signals[strategy_name].append(signal)
                
        except Exception as e:
            self.logger.error(f"❌ Strategy signal handling error: {e}")
    
    def get_strategizer_metrics(self) -> Dict:
        """Get strategizer metrics"""
        return {
            "active_strategies": len(self.active_strategies),
            "total_strategies": len(self.strategies),
            "market_regime": self.current_market_regime,
            "strategy_allocations": self.strategy_allocations,
            "avg_sharpe_ratio": np.mean([
                perf.sharpe_ratio for perf in self.strategy_performances.values()
            ]) if self.strategy_performances else 0.0,
            "coordination_enabled": self.strategy_coordination_enabled,
            "uptime": (datetime.utcnow() - self.metrics["uptime"]).total_seconds()
        }
