"""
Strategizer Agent - Develops and optimizes trading strategies
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import json

from .base_agent import BaseAgent, AgentMessage, MessageType, AgentCapability
from strategies.fibonacci_strategy import FibonacciStrategy
from strategies.swing_strategy import SwingStrategy
from strategies.scalping_strategy import ScalpingStrategy
from analytics.fibonacci_calculator import FibonacciCalculator
from analytics.technical_indicators import TechnicalIndicators
from backtesting.backtest_engine import BacktestEngine

logger = logging.getLogger(__name__)

@dataclass
class StrategyRecommendation:
    """Strategy recommendation data structure"""
    strategy_name: str
    category: str
    confidence: float
    parameters: Dict[str, Any]
    expected_return: float
    risk_level: str
    timeframe: str
    market_conditions: List[str]
    reasoning: str

class StrategizerAgent(BaseAgent):
    """Strategizer agent for strategy development and optimization"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        self.fibonacci_calculator = FibonacciCalculator()
        self.technical_indicators = TechnicalIndicators()
        self.backtest_engine = BacktestEngine()
        
        # Strategy instances
        self.strategies = {
            "fibonacci": FibonacciStrategy,
            "swing": SwingStrategy,
            "scalping": ScalpingStrategy
        }
        
        # Strategy performance tracking
        self.strategy_performance: Dict[str, Dict[str, Any]] = {}
        self.active_strategies: Dict[str, Any] = {}
        
        # Market regime detection
        self.market_regimes = ["bull", "bear", "sideways", "volatile"]
        self.current_regime = "neutral"
        
        # Strategy optimization parameters
        self.optimization_window = config.get("optimization_window", 30)  # days
        self.min_backtest_period = config.get("min_backtest_period", 60)  # days
        self.performance_threshold = config.get("performance_threshold", 0.05)  # 5%
        
    def _initialize_capabilities(self) -> None:
        """Initialize strategizer capabilities"""
        self.add_capability(AgentCapability(
            name="strategy_development",
            description="Develop and customize trading strategies",
            input_types=["market_data", "risk_parameters", "objectives"],
            output_types=["strategy_config", "backtesting_results"],
            parameters={"available_strategies": list(self.strategies.keys())}
        ))
        
        self.add_capability(AgentCapability(
            name="strategy_optimization",
            description="Optimize strategy parameters for performance",
            input_types=["strategy_config", "performance_data", "constraints"],
            output_types=["optimized_parameters", "performance_metrics"],
            parameters={"optimization_methods": ["grid_search", "bayesian", "genetic"]}
        ))
        
        self.add_capability(AgentCapability(
            name="market_regime_detection",
            description="Detect current market regime and adapt strategies",
            input_types=["market_data", "indicators", "volatility"],
            output_types=["regime_classification", "regime_confidence"],
            parameters={"regimes": self.market_regimes}
        ))
        
        self.add_capability(AgentCapability(
            name="strategy_recommendations",
            description="Recommend optimal strategies based on market conditions",
            input_types=["market_analysis", "risk_profile", "objectives"],
            output_types=["strategy_recommendations", "allocation_weights"],
            parameters={"recommendation_factors": ["performance", "risk", "market_fit"]}
        ))
        
        self.add_capability(AgentCapability(
            name="portfolio_strategy_allocation",
            description="Allocate capital across multiple strategies",
            input_types=["strategies", "risk_budget", "correlations"],
            output_types=["allocation_weights", "risk_metrics"],
            parameters={"allocation_methods": ["mean_variance", "risk_parity", "equal_weight"]}
        ))
        
    async def initialize(self) -> bool:
        """Initialize strategizer agent"""
        try:
            self.logger.info("🧠 Initializing Strategizer Agent...")
            
            # Initialize analytics components
            await self.fibonacci_calculator.initialize()
            await self.technical_indicators.initialize()
            await self.backtest_engine.initialize()
            
            # Load existing strategy performance data
            await self._load_strategy_performance()
            
            # Initialize market regime detection
            await self._initialize_regime_detection()
            
            # Subscribe to relevant topics
            self.subscribe_to_topic("strategy_request")
            self.subscribe_to_topic("optimization_request")
            self.subscribe_to_topic("market_regime_update")
            self.subscribe_to_topic("performance_update")
            
            self.logger.info("✅ Strategizer Agent initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Strategizer Agent initialization failed: {e}")
            return False
            
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming messages"""
        try:
            content = message.content
            message_type = message.message_type
            
            if message_type == MessageType.QUERY:
                return await self._handle_query(message)
            elif message_type == MessageType.COMMAND:
                return await self._handle_command(message)
            elif message_type == MessageType.RESPONSE:
                self.handle_response(message)
                
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error processing message: {e}")
            return self.create_response(message, {"error": str(e)})
            
    async def _handle_query(self, message: AgentMessage) -> AgentMessage:
        """Handle query messages"""
        content = message.content
        query_type = content.get("type")
        
        if query_type == "strategy_recommendations":
            result = await self._get_strategy_recommendations(content)
        elif query_type == "market_regime":
            result = await self._detect_market_regime(content)
        elif query_type == "strategy_performance":
            result = await self._get_strategy_performance(content)
        elif query_type == "optimize_strategy":
            result = await self._optimize_strategy(content)
        elif query_type == "portfolio_allocation":
            result = await self._get_portfolio_allocation(content)
        elif query_type == "backtest_strategy":
            result = await self._backtest_strategy(content)
        else:
            result = {"error": f"Unknown query type: {query_type}"}
            
        return self.create_response(message, result)
        
    async def _handle_command(self, message: AgentMessage) -> AgentMessage:
        """Handle command messages"""
        content = message.content
        command = content.get("command")
        
        if command == "create_strategy":
            result = await self._create_strategy(content)
        elif command == "update_strategy":
            result = await self._update_strategy(content)
        elif command == "delete_strategy":
            result = await self._delete_strategy(content)
        elif command == "optimize_all_strategies":
            result = await self._optimize_all_strategies()
        elif command == "update_market_regime":
            result = await self._update_market_regime(content)
        else:
            result = {"error": f"Unknown command: {command}"}
            
        return self.create_response(message, result)
        
    async def _get_strategy_recommendations(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get strategy recommendations based on market conditions"""
        try:
            symbols = params.get("symbols", ["SPY"])
            risk_tolerance = params.get("risk_tolerance", "medium")
            investment_horizon = params.get("investment_horizon", "medium")
            market_conditions = params.get("market_conditions", {})
            
            # Get market analysis from market analyst
            market_analysis = await self.query_agent("market_analyst", {
                "type": "market_overview",
                "symbols": symbols
            })
            
            if not market_analysis:
                return {"error": "Could not get market analysis"}
                
            # Detect current market regime
            regime_analysis = await self._detect_market_regime({"symbols": symbols})
            current_regime = regime_analysis.get("regime", "neutral")
            
            # Generate recommendations for each strategy category
            recommendations = []
            
            # Fibonacci strategy recommendation
            fib_rec = await self._evaluate_fibonacci_strategy(
                symbols, current_regime, risk_tolerance, market_analysis
            )
            if fib_rec:
                recommendations.append(fib_rec)
                
            # Swing trading recommendation
            swing_rec = await self._evaluate_swing_strategy(
                symbols, current_regime, risk_tolerance, market_analysis
            )
            if swing_rec:
                recommendations.append(swing_rec)
                
            # Scalping recommendation
            scalp_rec = await self._evaluate_scalping_strategy(
                symbols, current_regime, risk_tolerance, market_analysis
            )
            if scalp_rec:
                recommendations.append(scalp_rec)
                
            # Options strategy recommendation
            options_rec = await self._evaluate_options_strategy(
                symbols, current_regime, risk_tolerance, market_analysis
            )
            if options_rec:
                recommendations.append(options_rec)
                
            # Sort recommendations by confidence
            recommendations.sort(key=lambda x: x.confidence, reverse=True)
            
            # Generate portfolio allocation
            allocation = await self._generate_portfolio_allocation(
                recommendations, risk_tolerance
            )
            
            return {
                "timestamp": datetime.now().isoformat(),
                "market_regime": current_regime,
                "recommendations": [
                    {
                        "strategy_name": rec.strategy_name,
                        "category": rec.category,
                        "confidence": rec.confidence,
                        "parameters": rec.parameters,
                        "expected_return": rec.expected_return,
                        "risk_level": rec.risk_level,
                        "timeframe": rec.timeframe,
                        "market_conditions": rec.market_conditions,
                        "reasoning": rec.reasoning
                    }
                    for rec in recommendations
                ],
                "portfolio_allocation": allocation,
                "risk_tolerance": risk_tolerance,
                "investment_horizon": investment_horizon
            }
            
        except Exception as e:
            self.logger.error(f"❌ Strategy recommendations error: {e}")
            return {"error": str(e)}
            
    async def _detect_market_regime(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Detect current market regime"""
        try:
            symbols = params.get("symbols", ["SPY"])
            lookback_days = params.get("lookback_days", 60)
            
            # Get market data for analysis
            market_data = await self.query_agent("market_analyst", {
                "type": "technical_analysis",
                "symbol": symbols[0],
                "indicators": ["trend", "volatility", "momentum"]
            })
            
            if not market_data:
                return {"error": "Could not get market data"}
                
            # Analyze regime indicators
            regime_indicators = await self._analyze_regime_indicators(market_data)
            
            # Classify regime
            regime_classification = await self._classify_market_regime(regime_indicators)
            
            # Calculate confidence
            confidence = await self._calculate_regime_confidence(regime_indicators)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "regime": regime_classification,
                "confidence": confidence,
                "indicators": regime_indicators,
                "symbols_analyzed": symbols,
                "lookback_days": lookback_days
            }
            
        except Exception as e:
            self.logger.error(f"❌ Market regime detection error: {e}")
            return {"error": str(e)}
            
    async def _get_strategy_performance(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get strategy performance metrics"""
        try:
            strategy_name = params.get("strategy_name")
            time_period = params.get("time_period", "30d")
            
            if strategy_name:
                # Get specific strategy performance
                if strategy_name in self.strategy_performance:
                    performance = self.strategy_performance[strategy_name]
                else:
                    return {"error": f"Strategy {strategy_name} not found"}
            else:
                # Get all strategies performance
                performance = self.strategy_performance
                
            # Calculate performance metrics
            metrics = await self._calculate_performance_metrics(performance, time_period)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "strategy_name": strategy_name,
                "time_period": time_period,
                "performance": performance,
                "metrics": metrics
            }
            
        except Exception as e:
            self.logger.error(f"❌ Strategy performance error: {e}")
            return {"error": str(e)}
            
    async def _optimize_strategy(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize strategy parameters"""
        try:
            strategy_name = params.get("strategy_name")
            optimization_method = params.get("method", "grid_search")
            constraints = params.get("constraints", {})
            
            if strategy_name not in self.strategies:
                return {"error": f"Strategy {strategy_name} not supported"}
                
            # Get strategy class
            strategy_class = self.strategies[strategy_name]
            
            # Define parameter space
            param_space = await self._define_parameter_space(strategy_name)
            
            # Run optimization
            optimization_results = await self._run_optimization(
                strategy_class, param_space, optimization_method, constraints
            )
            
            # Update strategy performance
            if optimization_results["success"]:
                self.strategy_performance[strategy_name] = optimization_results["performance"]
                
            return {
                "timestamp": datetime.now().isoformat(),
                "strategy_name": strategy_name,
                "optimization_method": optimization_method,
                "results": optimization_results
            }
            
        except Exception as e:
            self.logger.error(f"❌ Strategy optimization error: {e}")
            return {"error": str(e)}
            
    async def _get_portfolio_allocation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get optimal portfolio allocation across strategies"""
        try:
            strategies = params.get("strategies", list(self.strategies.keys()))
            risk_budget = params.get("risk_budget", 0.1)
            allocation_method = params.get("method", "mean_variance")
            
            # Get strategy performance data
            strategy_returns = {}
            strategy_risks = {}
            
            for strategy_name in strategies:
                if strategy_name in self.strategy_performance:
                    perf = self.strategy_performance[strategy_name]
                    strategy_returns[strategy_name] = perf.get("expected_return", 0.0)
                    strategy_risks[strategy_name] = perf.get("volatility", 0.1)
                    
            # Calculate optimal allocation
            allocation = await self._calculate_optimal_allocation(
                strategy_returns, strategy_risks, risk_budget, allocation_method
            )
            
            # Calculate portfolio metrics
            portfolio_metrics = await self._calculate_portfolio_metrics(
                allocation, strategy_returns, strategy_risks
            )
            
            return {
                "timestamp": datetime.now().isoformat(),
                "allocation_method": allocation_method,
                "allocation": allocation,
                "portfolio_metrics": portfolio_metrics,
                "risk_budget": risk_budget
            }
            
        except Exception as e:
            self.logger.error(f"❌ Portfolio allocation error: {e}")
            return {"error": str(e)}
            
    async def _backtest_strategy(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Backtest a strategy"""
        try:
            strategy_name = params.get("strategy_name")
            symbol = params.get("symbol", "SPY")
            start_date = params.get("start_date")
            end_date = params.get("end_date")
            initial_capital = params.get("initial_capital", 100000)
            
            if strategy_name not in self.strategies:
                return {"error": f"Strategy {strategy_name} not supported"}
                
            # Run backtest
            backtest_results = await self.backtest_engine.run_backtest(
                strategy_name=strategy_name,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital
            )
            
            return {
                "timestamp": datetime.now().isoformat(),
                "strategy_name": strategy_name,
                "symbol": symbol,
                "backtest_results": backtest_results
            }
            
        except Exception as e:
            self.logger.error(f"❌ Backtest error: {e}")
            return {"error": str(e)}
            
    async def _evaluate_fibonacci_strategy(self, symbols: List[str], regime: str, 
                                         risk_tolerance: str, market_analysis: Dict[str, Any]) -> Optional[StrategyRecommendation]:
        """Evaluate Fibonacci strategy for current conditions"""
        try:
            # Check if market conditions favor Fibonacci analysis
            trend_strength = market_analysis.get("technical_indicators", {}).get("trend", {}).get("trend", "neutral")
            volatility = market_analysis.get("technical_indicators", {}).get("volatility", {}).get("volatility_percentile", 0.5)
            
            # Fibonacci works best in trending markets with moderate volatility
            if trend_strength in ["bullish", "bearish"] and 0.3 <= volatility <= 0.7:
                confidence = 0.8
                expected_return = 0.12
                risk_level = "medium"
            elif trend_strength == "neutral" and volatility > 0.7:
                confidence = 0.4
                expected_return = 0.06
                risk_level = "high"
            else:
                return None
                
            # Adjust for risk tolerance
            if risk_tolerance == "low" and risk_level == "high":
                confidence *= 0.7
                
            parameters = {
                "swing_lookback": 50,
                "min_swing_size": 0.02,
                "confluence_threshold": 2,
                "volume_confirmation": True,
                "trend_confirmation": True
            }
            
            return StrategyRecommendation(
                strategy_name="fibonacci",
                category="Swing Trade",
                confidence=confidence,
                parameters=parameters,
                expected_return=expected_return,
                risk_level=risk_level,
                timeframe="4h-1d",
                market_conditions=["trending", "moderate_volatility"],
                reasoning=f"Fibonacci retracements work well in {trend_strength} trending markets with moderate volatility"
            )
            
        except Exception as e:
            self.logger.error(f"❌ Fibonacci evaluation error: {e}")
            return None
            
    async def _evaluate_swing_strategy(self, symbols: List[str], regime: str, 
                                     risk_tolerance: str, market_analysis: Dict[str, Any]) -> Optional[StrategyRecommendation]:
        """Evaluate swing trading strategy"""
        try:
            # Swing trading works best in trending markets
            trend_strength = market_analysis.get("technical_indicators", {}).get("trend", {}).get("trend", "neutral")
            momentum = market_analysis.get("technical_indicators", {}).get("momentum", {}).get("rsi", 50)
            
            if trend_strength in ["bullish", "bearish"]:
                confidence = 0.75
                expected_return = 0.15
                risk_level = "medium"
            elif trend_strength == "neutral":
                confidence = 0.5
                expected_return = 0.08
                risk_level = "medium"
            else:
                return None
                
            parameters = {
                "trend_period": 50,
                "momentum_period": 14,
                "volume_threshold": 1.5,
                "breakout_confirmation": True,
                "mean_reversion_enabled": True
            }
            
            return StrategyRecommendation(
                strategy_name="swing",
                category="Swing Trade",
                confidence=confidence,
                parameters=parameters,
                expected_return=expected_return,
                risk_level=risk_level,
                timeframe="1h-4h",
                market_conditions=["trending", "momentum"],
                reasoning=f"Swing trading suitable for {trend_strength} market with momentum indicators"
            )
            
        except Exception as e:
            self.logger.error(f"❌ Swing evaluation error: {e}")
            return None
            
    async def _evaluate_scalping_strategy(self, symbols: List[str], regime: str, 
                                        risk_tolerance: str, market_analysis: Dict[str, Any]) -> Optional[StrategyRecommendation]:
        """Evaluate scalping strategy"""
        try:
            # Scalping works best in high volatility, high volume conditions
            volatility = market_analysis.get("technical_indicators", {}).get("volatility", {}).get("volatility_percentile", 0.5)
            volume = market_analysis.get("technical_indicators", {}).get("volume", {}).get("volume_trend", "normal")
            
            if volatility > 0.6 and volume == "high":
                confidence = 0.7
                expected_return = 0.08
                risk_level = "high"
            elif volatility > 0.4:
                confidence = 0.5
                expected_return = 0.05
                risk_level = "medium"
            else:
                return None
                
            # Scalping not suitable for low risk tolerance
            if risk_tolerance == "low":
                return None
                
            parameters = {
                "fast_ema_period": 5,
                "slow_ema_period": 13,
                "rsi_period": 7,
                "volume_spike_threshold": 2.0,
                "quick_profit_target": 0.003,
                "tight_stop_loss": 0.002
            }
            
            return StrategyRecommendation(
                strategy_name="scalping",
                category="Scalping",
                confidence=confidence,
                parameters=parameters,
                expected_return=expected_return,
                risk_level=risk_level,
                timeframe="1m-5m",
                market_conditions=["high_volatility", "high_volume"],
                reasoning=f"Scalping suitable for high volatility environment with {volume} volume"
            )
            
        except Exception as e:
            self.logger.error(f"❌ Scalping evaluation error: {e}")
            return None
            
    async def _evaluate_options_strategy(self, symbols: List[str], regime: str, 
                                       risk_tolerance: str, market_analysis: Dict[str, Any]) -> Optional[StrategyRecommendation]:
        """Evaluate options strategy"""
        try:
            # Options strategies based on volatility and trend
            volatility = market_analysis.get("technical_indicators", {}).get("volatility", {}).get("volatility_percentile", 0.5)
            trend = market_analysis.get("technical_indicators", {}).get("trend", {}).get("trend", "neutral")
            
            if volatility > 0.7:
                # High volatility - volatility selling strategies
                strategy_type = "iron_condor"
                confidence = 0.65
                expected_return = 0.10
                risk_level = "medium"
            elif volatility < 0.3 and trend in ["bullish", "bearish"]:
                # Low volatility trending - directional strategies
                strategy_type = "covered_call" if trend == "bullish" else "protective_put"
                confidence = 0.6
                expected_return = 0.08
                risk_level = "low"
            else:
                return None
                
            parameters = {
                "strategy_type": strategy_type,
                "dte_target": 30,
                "delta_target": 0.3,
                "profit_target": 0.5,
                "max_loss": 0.2
            }
            
            return StrategyRecommendation(
                strategy_name="options",
                category="Options",
                confidence=confidence,
                parameters=parameters,
                expected_return=expected_return,
                risk_level=risk_level,
                timeframe="1w-1m",
                market_conditions=[f"volatility_{volatility:.1f}", trend],
                reasoning=f"Options strategy {strategy_type} suitable for current volatility and trend"
            )
            
        except Exception as e:
            self.logger.error(f"❌ Options evaluation error: {e}")
            return None
            
    async def _generate_portfolio_allocation(self, recommendations: List[StrategyRecommendation], 
                                           risk_tolerance: str) -> Dict[str, float]:
        """Generate portfolio allocation weights"""
        try:
            if not recommendations:
                return {}
                
            # Calculate allocation based on confidence and risk tolerance
            total_confidence = sum(rec.confidence for rec in recommendations)
            allocation = {}
            
            for rec in recommendations:
                # Base allocation on confidence
                base_weight = rec.confidence / total_confidence
                
                # Adjust for risk tolerance
                if risk_tolerance == "low":
                    if rec.risk_level == "high":
                        base_weight *= 0.5
                    elif rec.risk_level == "low":
                        base_weight *= 1.2
                elif risk_tolerance == "high":
                    if rec.risk_level == "high":
                        base_weight *= 1.2
                    elif rec.risk_level == "low":
                        base_weight *= 0.8
                        
                allocation[rec.strategy_name] = base_weight
                
            # Normalize to sum to 1
            total_weight = sum(allocation.values())
            if total_weight > 0:
                allocation = {k: v / total_weight for k, v in allocation.items()}
                
            return allocation
            
        except Exception as e:
            self.logger.error(f"❌ Portfolio allocation error: {e}")
            return {}
            
    async def _analyze_regime_indicators(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze indicators for regime detection"""
        try:
            indicators = {}
            
            # Trend indicators
            trend_data = market_data.get("technical_indicators", {}).get("trend", {})
            indicators["trend_strength"] = trend_data.get("trend", "neutral")
            indicators["sma_20_50_ratio"] = trend_data.get("price_vs_sma_20", 0)
            
            # Volatility indicators
            vol_data = market_data.get("technical_indicators", {}).get("volatility", {})
            indicators["volatility_percentile"] = vol_data.get("volatility_percentile", 0.5)
            indicators["bb_width"] = vol_data.get("bb_width", 0.1)
            
            # Momentum indicators
            momentum_data = market_data.get("technical_indicators", {}).get("momentum", {})
            indicators["rsi"] = momentum_data.get("rsi", 50)
            indicators["macd_histogram"] = momentum_data.get("macd_histogram", 0)
            
            # Volume indicators
            volume_data = market_data.get("technical_indicators", {}).get("volume", {})
            indicators["volume_trend"] = volume_data.get("volume_trend", "normal")
            indicators["volume_ratio"] = volume_data.get("volume_ratio", 1.0)
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"❌ Regime indicators error: {e}")
            return {}
            
    async def _classify_market_regime(self, indicators: Dict[str, Any]) -> str:
        """Classify market regime based on indicators"""
        try:
            # Bull market indicators
            if (indicators.get("trend_strength") == "bullish" and
                indicators.get("rsi", 50) > 45 and
                indicators.get("volatility_percentile", 0.5) < 0.7):
                return "bull"
                
            # Bear market indicators
            elif (indicators.get("trend_strength") == "bearish" and
                  indicators.get("rsi", 50) < 55 and
                  indicators.get("volatility_percentile", 0.5) < 0.7):
                return "bear"
                
            # Volatile market indicators
            elif indicators.get("volatility_percentile", 0.5) > 0.8:
                return "volatile"
                
            # Sideways market
            elif indicators.get("trend_strength") == "neutral":
                return "sideways"
                
            else:
                return "neutral"
                
        except Exception as e:
            self.logger.error(f"❌ Regime classification error: {e}")
            return "neutral"
            
    async def _calculate_regime_confidence(self, indicators: Dict[str, Any]) -> float:
        """Calculate confidence in regime classification"""
        try:
            confidence = 0.5  # Base confidence
            
            # Trend strength contribution
            if indicators.get("trend_strength") in ["bullish", "bearish"]:
                confidence += 0.2
                
            # Volatility consistency
            vol_percentile = indicators.get("volatility_percentile", 0.5)
            if 0.3 <= vol_percentile <= 0.7:
                confidence += 0.15
                
            # Volume confirmation
            if indicators.get("volume_trend") == "high":
                confidence += 0.1
                
            # Momentum alignment
            rsi = indicators.get("rsi", 50)
            if (rsi > 60 and indicators.get("trend_strength") == "bullish") or \
               (rsi < 40 and indicators.get("trend_strength") == "bearish"):
                confidence += 0.15
                
            return min(confidence, 1.0)
            
        except Exception as e:
            self.logger.error(f"❌ Regime confidence error: {e}")
            return 0.5
            
    async def _define_parameter_space(self, strategy_name: str) -> Dict[str, List[Any]]:
        """Define parameter space for optimization"""
        try:
            if strategy_name == "fibonacci":
                return {
                    "swing_lookback": [30, 40, 50, 60, 70],
                    "min_swing_size": [0.015, 0.02, 0.025, 0.03],
                    "confluence_threshold": [1, 2, 3],
                    "volume_confirmation": [True, False],
                    "trend_confirmation": [True, False]
                }
            elif strategy_name == "swing":
                return {
                    "trend_period": [20, 30, 50, 70],
                    "momentum_period": [10, 14, 20],
                    "volume_threshold": [1.2, 1.5, 2.0],
                    "breakout_confirmation": [True, False],
                    "mean_reversion_enabled": [True, False]
                }
            elif strategy_name == "scalping":
                return {
                    "fast_ema_period": [3, 5, 8],
                    "slow_ema_period": [10, 13, 21],
                    "rsi_period": [5, 7, 10],
                    "volume_spike_threshold": [1.5, 2.0, 2.5],
                    "quick_profit_target": [0.002, 0.003, 0.005],
                    "tight_stop_loss": [0.001, 0.002, 0.003]
                }
            else:
                return {}
                
        except Exception as e:
            self.logger.error(f"❌ Parameter space definition error: {e}")
            return {}
            
    async def _run_optimization(self, strategy_class, param_space: Dict[str, List[Any]], 
                              method: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Run strategy optimization"""
        try:
            # This is a simplified optimization - in production, use more sophisticated methods
            best_params = None
            best_performance = -float('inf')
            
            # Grid search implementation
            if method == "grid_search":
                import itertools
                
                param_names = list(param_space.keys())
                param_values = list(param_space.values())
                
                for param_combo in itertools.product(*param_values):
                    params = dict(zip(param_names, param_combo))
                    
                    # Create strategy instance
                    strategy = strategy_class({"parameters": params})
                    
                    # Simulate performance (in production, use backtesting)
                    performance = await self._simulate_strategy_performance(strategy)
                    
                    if performance > best_performance:
                        best_performance = performance
                        best_params = params
                        
            return {
                "success": True,
                "best_parameters": best_params,
                "best_performance": best_performance,
                "optimization_method": method,
                "performance": {
                    "expected_return": best_performance,
                    "volatility": 0.1,  # Simplified
                    "sharpe_ratio": best_performance / 0.1
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Optimization error: {e}")
            return {"success": False, "error": str(e)}
            
    async def _simulate_strategy_performance(self, strategy) -> float:
        """Simulate strategy performance (simplified)"""
        try:
            # This is a placeholder - in production, use proper backtesting
            import random
            
            # Simulate based on strategy type and parameters
            base_return = random.uniform(0.02, 0.15)
            
            # Adjust based on strategy parameters
            if hasattr(strategy, 'parameters'):
                params = strategy.parameters
                
                # Reward conservative parameters
                if params.get("stop_loss_pct", 0.02) < 0.03:
                    base_return *= 1.1
                    
                # Reward volume confirmation
                if params.get("volume_confirmation", False):
                    base_return *= 1.05
                    
            return base_return
            
        except Exception as e:
            self.logger.error(f"❌ Performance simulation error: {e}")
            return 0.05
            
    async def _calculate_optimal_allocation(self, returns: Dict[str, float], risks: Dict[str, float], 
                                          risk_budget: float, method: str) -> Dict[str, float]:
        """Calculate optimal portfolio allocation"""
        try:
            strategies = list(returns.keys())
            n_strategies = len(strategies)
            
            if n_strategies == 0:
                return {}
                
            if method == "equal_weight":
                # Equal weight allocation
                weight = 1.0 / n_strategies
                return {strategy: weight for strategy in strategies}
                
            elif method == "risk_parity":
                # Risk parity allocation
                risk_contributions = {strategy: 1.0 / risks[strategy] for strategy in strategies}
                total_risk_contribution = sum(risk_contributions.values())
                
                return {
                    strategy: contrib / total_risk_contribution 
                    for strategy, contrib in risk_contributions.items()
                }
                
            elif method == "mean_variance":
                # Simplified mean-variance optimization
                # Weight by return/risk ratio
                ratios = {strategy: returns[strategy] / risks[strategy] for strategy in strategies}
                total_ratio = sum(ratios.values())
                
                return {
                    strategy: ratio / total_ratio 
                    for strategy, ratio in ratios.items()
                }
                
            else:
                # Default to equal weight
                weight = 1.0 / n_strategies
                return {strategy: weight for strategy in strategies}
                
        except Exception as e:
            self.logger.error(f"❌ Allocation calculation error: {e}")
            return {}
            
    async def _calculate_portfolio_metrics(self, allocation: Dict[str, float], 
                                         returns: Dict[str, float], risks: Dict[str, float]) -> Dict[str, float]:
        """Calculate portfolio-level metrics"""
        try:
            if not allocation:
                return {}
                
            # Portfolio return
            portfolio_return = sum(allocation[strategy] * returns[strategy] 
                                 for strategy in allocation.keys())
            
            # Portfolio risk (simplified - assuming zero correlation)
            portfolio_risk = sum(allocation[strategy]**2 * risks[strategy]**2 
                               for strategy in allocation.keys())**0.5
            
            # Sharpe ratio
            sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
            
            return {
                "expected_return": portfolio_return,
                "volatility": portfolio_risk,
                "sharpe_ratio": sharpe_ratio,
                "strategies_count": len(allocation)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Portfolio metrics error: {e}")
            return {}
            
    async def _calculate_performance_metrics(self, performance: Dict[str, Any], 
                                           time_period: str) -> Dict[str, Any]:
        """Calculate performance metrics"""
        try:
            metrics = {}
            
            for strategy_name, perf_data in performance.items():
                if isinstance(perf_data, dict):
                    metrics[strategy_name] = {
                        "total_return": perf_data.get("total_return", 0.0),
                        "volatility": perf_data.get("volatility", 0.0),
                        "sharpe_ratio": perf_data.get("sharpe_ratio", 0.0),
                        "max_drawdown": perf_data.get("max_drawdown", 0.0),
                        "win_rate": perf_data.get("win_rate", 0.0)
                    }
                    
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Performance metrics error: {e}")
            return {}
            
    async def _load_strategy_performance(self) -> None:
        """Load existing strategy performance data"""
        try:
            # In production, load from database
            # For now, initialize with default values
            self.strategy_performance = {
                "fibonacci": {
                    "total_return": 0.12,
                    "volatility": 0.15,
                    "sharpe_ratio": 0.8,
                    "max_drawdown": 0.08,
                    "win_rate": 0.65
                },
                "swing": {
                    "total_return": 0.15,
                    "volatility": 0.18,
                    "sharpe_ratio": 0.83,
                    "max_drawdown": 0.10,
                    "win_rate": 0.62
                },
                "scalping": {
                    "total_return": 0.08,
                    "volatility": 0.12,
                    "sharpe_ratio": 0.67,
                    "max_drawdown": 0.06,
                    "win_rate": 0.58
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Load performance error: {e}")
            
    async def _initialize_regime_detection(self) -> None:
        """Initialize market regime detection"""
        try:
            # Initialize with neutral regime
            self.current_regime = "neutral"
            
        except Exception as e:
            self.logger.error(f"❌ Regime detection init error: {e}")
            
    async def periodic_task(self) -> None:
        """Periodic strategy optimization and regime detection"""
        try:
            # Update market regime
            regime_update = await self._detect_market_regime({"symbols": ["SPY"]})
            new_regime = regime_update.get("regime", "neutral")
            
            if new_regime != self.current_regime:
                self.current_regime = new_regime
                
                # Send regime change notification
                notification = self.create_notification(
                    receiver="system",
                    content={
                        "type": "market_regime_change",
                        "old_regime": self.current_regime,
                        "new_regime": new_regime,
                        "confidence": regime_update.get("confidence", 0.5)
                    }
                )
                await self.send_message(notification)
                
            # Periodic strategy optimization
            await self._optimize_all_strategies()
            
            await self.log_activity("periodic_optimization", {
                "current_regime": self.current_regime,
                "strategies_optimized": len(self.strategies)
            })
            
        except Exception as e:
            self.logger.error(f"❌ Periodic task error: {e}")
            
    async def _optimize_all_strategies(self) -> Dict[str, Any]:
        """Optimize all strategies"""
        try:
            results = {}
            
            for strategy_name in self.strategies.keys():
                optimization_result = await self._optimize_strategy({
                    "strategy_name": strategy_name,
                    "method": "grid_search",
                    "constraints": {}
                })
                
                results[strategy_name] = optimization_result
                
            return {
                "success": True,
                "results": results,
                "strategies_optimized": len(results)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Optimize all strategies error: {e}")
            return {"success": False, "error": str(e)}
            
    def get_status_info(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "agent_type": "strategizer",
            "status": self.status.value,
            "available_strategies": list(self.strategies.keys()),
            "current_regime": self.current_regime,
            "active_strategies": len(self.active_strategies),
            "last_optimization": self.metrics.last_activity.isoformat()
        }
