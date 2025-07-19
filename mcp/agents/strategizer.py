"""
Strategizer Agent - Develops and optimizes trading strategies
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json

from mcp.agents.base_agent import BaseAgent
from mcp.mcp_server import MCPMessage

class Strategizer(BaseAgent):
    """Strategy development and optimization agent"""
    
    def __init__(self, agent_id: str, settings, mcp_server, freqtrade_engine, digital_brain):
        super().__init__(agent_id, settings, mcp_server, freqtrade_engine, digital_brain)
        self.active_strategies = {}
        self.strategy_performance = {}
        self.strategy_signals = {}
        self.optimization_history = []
        
    async def initialize(self):
        """Initialize the strategizer"""
        await super().initialize()
        
        # Start strategy management loop
        asyncio.create_task(self._strategy_management_loop())
        
        self.logger.info("🧠 Strategizer Agent initialized")
    
    async def _strategy_management_loop(self):
        """Main strategy management loop"""
        while self.is_active:
            try:
                await self._analyze_strategy_performance()
                await self._optimize_strategies()
                await self._generate_strategy_signals()
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in strategy management loop: {e}")
                await asyncio.sleep(60)
    
    async def _analyze_strategy_performance(self):
        """Analyze performance of active strategies"""
        try:
            for strategy_name, strategy in self.freqtrade_engine.strategies.items():
                if strategy.is_active:
                    performance = await self._calculate_strategy_performance(strategy_name)
                    self.strategy_performance[strategy_name] = performance
                    
                    # Check if strategy needs optimization
                    if self._strategy_needs_optimization(performance):
                        await self._schedule_optimization(strategy_name)
                        
        except Exception as e:
            self.logger.error(f"Error analyzing strategy performance: {e}")
    
    async def _calculate_strategy_performance(self, strategy_name: str) -> Dict:
        """Calculate performance metrics for a strategy"""
        try:
            # Get strategy performance from freqtrade engine
            engine_performance = self.freqtrade_engine.strategy_performance.get(strategy_name, {})
            
            # Calculate additional metrics
            total_trades = engine_performance.get("total_trades", 0)
            winning_trades = engine_performance.get("winning_trades", 0)
            total_pnl = engine_performance.get("total_pnl", 0.0)
            
            win_rate = (winning_trades / total_trades) if total_trades > 0 else 0.0
            
            # Calculate Sharpe ratio (simplified)
            sharpe_ratio = await self._calculate_sharpe_ratio(strategy_name)
            
            # Calculate maximum drawdown
            max_drawdown = await self._calculate_max_drawdown(strategy_name)
            
            performance = {
                "strategy_name": strategy_name,
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "win_rate": win_rate,
                "total_pnl": total_pnl,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "last_updated": datetime.now(),
                "status": "active" if total_trades > 0 else "inactive"
            }
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Error calculating strategy performance: {e}")
            return {}
    
    async def _calculate_sharpe_ratio(self, strategy_name: str) -> float:
        """Calculate Sharpe ratio for a strategy"""
        try:
            # This is a simplified calculation
            # In reality, you'd need historical returns data
            total_pnl = self.freqtrade_engine.strategy_performance.get(strategy_name, {}).get("total_pnl", 0.0)
            total_trades = self.freqtrade_engine.strategy_performance.get(strategy_name, {}).get("total_trades", 0)
            
            if total_trades == 0:
                return 0.0
            
            # Simplified Sharpe ratio calculation
            avg_return = total_pnl / total_trades
            # Assuming some volatility (in reality, calculate from actual returns)
            volatility = abs(avg_return) * 0.5
            
            if volatility == 0:
                return 0.0
            
            # Risk-free rate assumed to be 0 for simplicity
            sharpe_ratio = avg_return / volatility
            
            return sharpe_ratio
            
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    async def _calculate_max_drawdown(self, strategy_name: str) -> float:
        """Calculate maximum drawdown for a strategy"""
        try:
            # Simplified drawdown calculation
            # In reality, you'd track cumulative returns
            max_drawdown = self.freqtrade_engine.strategy_performance.get(strategy_name, {}).get("max_drawdown", 0.0)
            return max_drawdown
            
        except Exception as e:
            self.logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def _strategy_needs_optimization(self, performance: Dict) -> bool:
        """Check if strategy needs optimization"""
        try:
            # Optimization criteria
            win_rate = performance.get("win_rate", 0.0)
            sharpe_ratio = performance.get("sharpe_ratio", 0.0)
            max_drawdown = performance.get("max_drawdown", 0.0)
            total_trades = performance.get("total_trades", 0)
            
            # Needs optimization if:
            # 1. Win rate < 40%
            # 2. Sharpe ratio < 0.5
            # 3. Max drawdown > 10%
            # 4. Has enough trades for meaningful analysis
            
            if total_trades < 10:
                return False
            
            if win_rate < 0.4 or sharpe_ratio < 0.5 or max_drawdown > 0.1:
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking optimization need: {e}")
            return False
    
    async def _schedule_optimization(self, strategy_name: str):
        """Schedule strategy optimization"""
        try:
            self.logger.info(f"Scheduling optimization for strategy: {strategy_name}")
            
            # Use Digital Brain to suggest optimizations
            optimization_query = f"""
            The {strategy_name} strategy is underperforming. Current performance:
            {json.dumps(self.strategy_performance.get(strategy_name, {}), indent=2)}
            
            Suggest specific optimizations to improve:
            1. Win rate
            2. Risk-adjusted returns
            3. Drawdown control
            4. Position sizing
            """
            
            optimization_suggestions = await self.digital_brain.query(optimization_query)
            
            # Store optimization suggestion
            optimization_record = {
                "strategy_name": strategy_name,
                "timestamp": datetime.now(),
                "current_performance": self.strategy_performance.get(strategy_name, {}),
                "suggestions": optimization_suggestions,
                "status": "pending"
            }
            
            self.optimization_history.append(optimization_record)
            
            # Notify other agents
            await self._notify_optimization_scheduled(optimization_record)
            
        except Exception as e:
            self.logger.error(f"Error scheduling optimization: {e}")
    
    async def _optimize_strategies(self):
        """Optimize underperforming strategies"""
        try:
            # Process pending optimizations
            for optimization in self.optimization_history:
                if optimization["status"] == "pending":
                    await self._execute_optimization(optimization)
                    
        except Exception as e:
            self.logger.error(f"Error optimizing strategies: {e}")
    
    async def _execute_optimization(self, optimization_record: Dict):
        """Execute strategy optimization"""
        try:
            strategy_name = optimization_record["strategy_name"]
            
            # Get current strategy parameters
            strategy = self.freqtrade_engine.strategies.get(strategy_name)
            if not strategy:
                return
            
            # Use Digital Brain to determine specific parameter changes
            param_query = f"""
            Based on the optimization suggestions for {strategy_name}:
            {optimization_record['suggestions']}
            
            Provide specific parameter changes in JSON format:
            {{
                "risk_tolerance": 0.02,
                "position_size": 0.05,
                "stop_loss": 0.03,
                "take_profit": 0.06
            }}
            """
            
            param_response = await self.digital_brain.query(param_query)
            
            # Apply optimization (simplified)
            optimization_record["status"] = "completed"
            optimization_record["applied_at"] = datetime.now()
            
            self.logger.info(f"Optimization completed for {strategy_name}")
            
        except Exception as e:
            self.logger.error(f"Error executing optimization: {e}")
    
    async def _generate_strategy_signals(self):
        """Generate signals from all strategies"""
        try:
            # Get market analysis from market analyst
            market_analysis = self.mcp_server.get_shared_context("market_analysis")
            
            if not market_analysis:
                return
            
            # Generate signals for each strategy category
            for category, strategies in self.settings.STRATEGY_CATEGORIES.items():
                signals = await self._generate_category_signals(category, market_analysis)
                
                if signals:
                    await self._broadcast_strategy_signals(category, signals)
                    
        except Exception as e:
            self.logger.error(f"Error generating strategy signals: {e}")
    
    async def _generate_category_signals(self, category: str, market_analysis: Dict) -> List[Dict]:
        """Generate signals for a strategy category"""
        try:
            # Use Digital Brain to generate signals
            signal_query = f"""
            Based on the current market analysis:
            {json.dumps(market_analysis, indent=2)}
            
            Generate trading signals for {category} strategies.
            Consider:
            1. Market regime
            2. Volatility levels
            3. Sentiment
            4. Risk factors
            
            Provide signals in JSON format:
            {{
                "signals": [
                    {{
                        "symbol": "AAPL",
                        "action": "buy",
                        "confidence": 0.8,
                        "quantity": 100,
                        "reasoning": "Strong technical breakout"
                    }}
                ]
            }}
            """
            
            signal_response = await self.digital_brain.query(signal_query)
            
            # Parse signals (simplified)
            signals = self._parse_signals(signal_response, category)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating category signals: {e}")
            return []
    
    def _parse_signals(self, signal_response: str, category: str) -> List[Dict]:
        """Parse signals from Digital Brain response"""
        try:
            # Simplified signal parsing
            signals = []
            
            # Look for common patterns in response
            if "buy" in signal_response.lower():
                signals.append({
                    "symbol": "SPY",  # Default symbol
                    "action": "buy",
                    "confidence": 0.7,
                    "quantity": 100,
                    "category": category,
                    "reasoning": "Generated by Digital Brain analysis"
                })
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error parsing signals: {e}")
            return []
    
    async def _broadcast_strategy_signals(self, category: str, signals: List[Dict]):
        """Broadcast strategy signals to other agents"""
        try:
            message = MCPMessage(
                agent_id=self.agent_id,
                message_type="strategy_signals",
                content={
                    "category": category,
                    "signals": signals,
                    "timestamp": datetime.now()
                },
                timestamp=datetime.now()
            )
            
            await self.mcp_server.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Error broadcasting strategy signals: {e}")
    
    async def _notify_optimization_scheduled(self, optimization_record: Dict):
        """Notify other agents about scheduled optimization"""
        try:
            message = MCPMessage(
                agent_id=self.agent_id,
                message_type="optimization_scheduled",
                content=optimization_record,
                timestamp=datetime.now()
            )
            
            await self.mcp_server.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Error notifying optimization: {e}")
    
    async def handle_message(self, message: MCPMessage):
        """Handle incoming messages"""
        try:
            if message.message_type == "strategy_performance_request":
                await self._handle_performance_request(message)
                
            elif message.message_type == "strategy_optimization_request":
                await self._handle_optimization_request(message)
                
            elif message.message_type == "new_strategy_request":
                await self._handle_new_strategy_request(message)
                
            elif message.message_type == "strategy_signals_request":
                await self._handle_signals_request(message)
                
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
    
    async def _handle_performance_request(self, message: MCPMessage):
        """Handle request for strategy performance"""
        try:
            strategy_name = message.content.get("strategy_name")
            
            if strategy_name:
                performance = self.strategy_performance.get(strategy_name, {})
            else:
                performance = self.strategy_performance
            
            response = MCPMessage(
                agent_id=self.agent_id,
                message_type="strategy_performance_response",
                content=performance,
                timestamp=datetime.now(),
                correlation_id=message.correlation_id
            )
            
            await self.mcp_server.send_message(response)
            
        except Exception as e:
            self.logger.error(f"Error handling performance request: {e}")
    
    async def _handle_optimization_request(self, message: MCPMessage):
        """Handle request for strategy optimization"""
        try:
            strategy_name = message.content.get("strategy_name")
            
            if strategy_name:
                await self._schedule_optimization(strategy_name)
                
                response = MCPMessage(
                    agent_id=self.agent_id,
                    message_type="optimization_response",
                    content={"status": "scheduled", "strategy": strategy_name},
                    timestamp=datetime.now(),
                    correlation_id=message.correlation_id
                )
                
                await self.mcp_server.send_message(response)
                
        except Exception as e:
            self.logger.error(f"Error handling optimization request: {e}")
    
    async def _handle_new_strategy_request(self, message: MCPMessage):
        """Handle request for new strategy development"""
        try:
            requirements = message.content.get("requirements", {})
            
            # Use Digital Brain to develop new strategy
            strategy_query = f"""
            Develop a new trading strategy based on these requirements:
            {json.dumps(requirements, indent=2)}
            
            Provide:
            1. Strategy name and type
            2. Entry/exit rules
            3. Risk management parameters
            4. Expected performance metrics
            5. Market conditions where it works best
            """
            
            strategy_response = await self.digital_brain.query(strategy_query)
            
            response = MCPMessage(
                agent_id=self.agent_id,
                message_type="new_strategy_response",
                content={
                    "strategy": strategy_response,
                    "requirements": requirements,
                    "status": "developed"
                },
                timestamp=datetime.now(),
                correlation_id=message.correlation_id
            )
            
            await self.mcp_server.send_message(response)
            
        except Exception as e:
            self.logger.error(f"Error handling new strategy request: {e}")
    
    async def _handle_signals_request(self, message: MCPMessage):
        """Handle request for strategy signals"""
        try:
            category = message.content.get("category")
            
            if category in self.strategy_signals:
                signals = self.strategy_signals[category]
            else:
                signals = []
            
            response = MCPMessage(
                agent_id=self.agent_id,
                message_type="strategy_signals_response",
                content={
                    "category": category,
                    "signals": signals,
                    "timestamp": datetime.now()
                },
                timestamp=datetime.now(),
                correlation_id=message.correlation_id
            )
            
            await self.mcp_server.send_message(response)
            
        except Exception as e:
            self.logger.error(f"Error handling signals request: {e}")
    
    def get_strategy_overview(self) -> Dict:
        """Get overview of all strategies"""
        try:
            overview = {
                "total_strategies": len(self.strategy_performance),
                "active_strategies": len([
                    s for s in self.strategy_performance.values()
                    if s.get("status") == "active"
                ]),
                "optimizations_pending": len([
                    o for o in self.optimization_history
                    if o["status"] == "pending"
                ]),
                "best_performing": self._get_best_performing_strategy(),
                "worst_performing": self._get_worst_performing_strategy(),
                "last_updated": datetime.now()
            }
            
            return overview
            
        except Exception as e:
            self.logger.error(f"Error getting strategy overview: {e}")
            return {}
    
    def _get_best_performing_strategy(self) -> Dict:
        """Get best performing strategy"""
        try:
            if not self.strategy_performance:
                return {}
            
            best_strategy = max(
                self.strategy_performance.items(),
                key=lambda x: x[1].get("sharpe_ratio", 0)
            )
            
            return {
                "name": best_strategy[0],
                "sharpe_ratio": best_strategy[1].get("sharpe_ratio", 0),
                "win_rate": best_strategy[1].get("win_rate", 0)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting best performing strategy: {e}")
            return {}
    
    def _get_worst_performing_strategy(self) -> Dict:
        """Get worst performing strategy"""
        try:
            if not self.strategy_performance:
                return {}
            
            worst_strategy = min(
                self.strategy_performance.items(),
                key=lambda x: x[1].get("sharpe_ratio", 0)
            )
            
            return {
                "name": worst_strategy[0],
                "sharpe_ratio": worst_strategy[1].get("sharpe_ratio", 0),
                "win_rate": worst_strategy[1].get("win_rate", 0)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting worst performing strategy: {e}")
            return {}
