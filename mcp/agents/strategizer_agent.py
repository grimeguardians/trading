"""
Strategizer Agent - Develops and optimizes trading strategies
"""

import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from .base_agent import BaseAgent
from ...config import Config
from ...strategies.base_strategy import StrategyMetrics

class OptimizationMethod(Enum):
    GENETIC_ALGORITHM = "genetic_algorithm"
    GRID_SEARCH = "grid_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    RANDOM_SEARCH = "random_search"

@dataclass
class StrategyOptimization:
    """Strategy optimization result"""
    strategy_id: str
    strategy_type: str
    parameters: Dict[str, Any]
    performance_score: float
    metrics: Dict[str, float]
    confidence: float
    optimization_method: OptimizationMethod
    timestamp: datetime

class StrategizerAgent(BaseAgent):
    """Agent responsible for strategy development and optimization"""
    
    def __init__(self, config: Config):
        super().__init__(config, "StrategizerAgent")
        
        # Strategy optimization parameters
        self.optimization_interval = 3600  # 1 hour
        self.lookback_period = 30  # Days
        self.min_trades_for_optimization = 20
        self.performance_threshold = 0.6
        
        # Optimization methods
        self.optimization_methods = {
            OptimizationMethod.GENETIC_ALGORITHM: self.genetic_algorithm_optimization,
            OptimizationMethod.GRID_SEARCH: self.grid_search_optimization,
            OptimizationMethod.BAYESIAN_OPTIMIZATION: self.bayesian_optimization,
            OptimizationMethod.RANDOM_SEARCH: self.random_search_optimization
        }
        
        # Strategy templates
        self.strategy_templates = {
            'swing_trade': {
                'parameters': {
                    'rsi_period': {'min': 10, 'max': 20, 'default': 14},
                    'ma_short': {'min': 10, 'max': 30, 'default': 20},
                    'ma_long': {'min': 30, 'max': 100, 'default': 50},
                    'stop_loss_pct': {'min': 0.02, 'max': 0.1, 'default': 0.05},
                    'take_profit_pct': {'min': 0.05, 'max': 0.2, 'default': 0.1}
                }
            },
            'scalping': {
                'parameters': {
                    'ema_fast': {'min': 3, 'max': 8, 'default': 5},
                    'ema_slow': {'min': 8, 'max': 15, 'default': 10},
                    'rsi_period': {'min': 3, 'max': 10, 'default': 5},
                    'target_profit_ticks': {'min': 1, 'max': 5, 'default': 2},
                    'stop_loss_ticks': {'min': 1, 'max': 3, 'default': 1}
                }
            },
            'intraday': {
                'parameters': {
                    'momentum_period': {'min': 5, 'max': 15, 'default': 10},
                    'rsi_period': {'min': 5, 'max': 15, 'default': 9},
                    'vwap_deviation_threshold': {'min': 0.001, 'max': 0.005, 'default': 0.002},
                    'stop_loss_pct': {'min': 0.01, 'max': 0.05, 'default': 0.02},
                    'take_profit_pct': {'min': 0.02, 'max': 0.08, 'default': 0.04}
                }
            }
        }
        
        # Strategy tracking
        self.active_strategies = {}
        self.strategy_performance = {}
        self.optimization_history = []
        
        # Market regime detection
        self.market_regimes = ['bullish', 'bearish', 'neutral', 'volatile']
        self.current_market_regime = 'neutral'
        
        # Performance metrics
        self.optimization_metrics = {
            'strategies_optimized': 0,
            'successful_optimizations': 0,
            'average_improvement': 0.0,
            'best_strategy_performance': 0.0
        }
        
    async def initialize(self):
        """Initialize strategizer agent components"""
        try:
            # Initialize optimization engines
            await self.initialize_optimization_engines()
            
            # Start strategy monitoring task
            monitor_task = asyncio.create_task(self._monitor_strategies())
            self.tasks.append(monitor_task)
            
            # Start optimization task
            optimization_task = asyncio.create_task(self._run_optimizations())
            self.tasks.append(optimization_task)
            
            # Start market regime detection task
            regime_task = asyncio.create_task(self._detect_market_regime())
            self.tasks.append(regime_task)
            
            self.logger.info("Strategizer agent initialized successfully")
            
        except Exception as e:
            await self.handle_error(e, "initialization")
            raise
    
    async def process(self):
        """Main strategizer processing"""
        try:
            # Update strategy performance
            await self.update_strategy_performance()
            
            # Analyze strategy effectiveness
            await self.analyze_strategy_effectiveness()
            
            # Generate strategy recommendations
            await self.generate_strategy_recommendations()
            
            # Adapt strategies to market conditions
            await self.adapt_strategies_to_market()
            
        except Exception as e:
            await self.handle_error(e, "processing")
    
    async def handle_message(self, message):
        """Handle incoming messages"""
        try:
            message_type = message.type
            content = message.content
            
            if message_type.value == "strategy_performance":
                await self.process_strategy_performance(content)
            elif message_type.value == "optimization_request":
                await self.process_optimization_request(content)
            elif message_type.value == "market_analysis":
                await self.process_market_analysis(content)
            elif message_type.value == "strategy_update":
                await self.process_strategy_update(content)
            else:
                self.logger.debug(f"Unhandled message type: {message_type}")
                
        except Exception as e:
            await self.handle_error(e, "message handling")
    
    async def cleanup(self):
        """Cleanup strategizer resources"""
        try:
            # Clear strategy tracking
            self.active_strategies.clear()
            self.strategy_performance.clear()
            self.optimization_history.clear()
            
            self.logger.info("Strategizer agent cleanup completed")
            
        except Exception as e:
            await self.handle_error(e, "cleanup")
    
    def get_processing_interval(self) -> float:
        """Get processing interval"""
        return 300.0  # Process every 5 minutes
    
    async def initialize_optimization_engines(self):
        """Initialize optimization engines"""
        try:
            # Initialize genetic algorithm parameters
            self.ga_params = {
                'population_size': 20,
                'generations': 10,
                'mutation_rate': 0.1,
                'crossover_rate': 0.7,
                'elite_size': 2
            }
            
            # Initialize grid search parameters
            self.grid_search_params = {
                'max_combinations': 100,
                'parallel_evaluations': 4
            }
            
            # Initialize Bayesian optimization parameters
            self.bayesian_params = {
                'initial_samples': 5,
                'max_iterations': 20,
                'acquisition_function': 'expected_improvement'
            }
            
            await self.log_activity("Optimization engines initialized")
            
        except Exception as e:
            await self.handle_error(e, "optimization engine initialization")
    
    async def _monitor_strategies(self):
        """Monitor active strategies"""
        while self.is_running:
            try:
                # Check strategy health
                await self.check_strategy_health()
                
                # Update strategy metrics
                await self.update_strategy_metrics()
                
                # Detect underperforming strategies
                await self.detect_underperforming_strategies()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                await self.handle_error(e, "strategy monitoring")
                await asyncio.sleep(60)
    
    async def _run_optimizations(self):
        """Run strategy optimizations"""
        while self.is_running:
            try:
                # Check if optimization is needed
                strategies_to_optimize = await self.identify_optimization_candidates()
                
                if strategies_to_optimize:
                    for strategy_id in strategies_to_optimize:
                        await self.optimize_strategy(strategy_id)
                
                await asyncio.sleep(self.optimization_interval)
                
            except Exception as e:
                await self.handle_error(e, "optimization loop")
                await asyncio.sleep(self.optimization_interval)
    
    async def _detect_market_regime(self):
        """Detect current market regime"""
        while self.is_running:
            try:
                # This would analyze market data to determine regime
                # For now, we'll use a simplified approach
                
                # Get market analysis from market analyst
                market_data = await self.get_market_analysis()
                
                if market_data:
                    regime = await self.classify_market_regime(market_data)
                    
                    if regime != self.current_market_regime:
                        self.current_market_regime = regime
                        await self.log_activity(f"Market regime changed to: {regime}")
                        
                        # Adapt strategies to new regime
                        await self.adapt_strategies_to_regime(regime)
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                await self.handle_error(e, "market regime detection")
                await asyncio.sleep(300)
    
    async def classify_market_regime(self, market_data: Dict[str, Any]) -> str:
        """Classify market regime based on market data"""
        try:
            # Analyze market conditions
            market_conditions = market_data.get('market_conditions', {})
            
            if not market_conditions:
                return 'neutral'
            
            # Count bullish vs bearish trends
            bullish_count = sum(1 for cond in market_conditions.values() if cond.get('trend') == 'bullish')
            bearish_count = sum(1 for cond in market_conditions.values() if cond.get('trend') == 'bearish')
            total_count = len(market_conditions)
            
            # Calculate average volatility
            volatilities = [cond.get('volatility', 0.2) for cond in market_conditions.values()]
            avg_volatility = sum(volatilities) / len(volatilities) if volatilities else 0.2
            
            # Classify regime
            if avg_volatility > 0.3:
                return 'volatile'
            elif bullish_count / total_count > 0.6:
                return 'bullish'
            elif bearish_count / total_count > 0.6:
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception as e:
            await self.handle_error(e, "market regime classification")
            return 'neutral'
    
    async def get_market_analysis(self) -> Optional[Dict[str, Any]]:
        """Get market analysis from market analyst"""
        try:
            # This would typically request data from market analyst
            # For now, return None - would be implemented with actual MCP integration
            return None
            
        except Exception as e:
            await self.handle_error(e, "market analysis retrieval")
            return None
    
    async def check_strategy_health(self):
        """Check health of active strategies"""
        try:
            unhealthy_strategies = []
            
            for strategy_id, strategy_data in self.active_strategies.items():
                performance = self.strategy_performance.get(strategy_id, {})
                
                # Check win rate
                win_rate = performance.get('win_rate', 0)
                if win_rate < 40:  # Less than 40% win rate
                    unhealthy_strategies.append({
                        'strategy_id': strategy_id,
                        'issue': 'low_win_rate',
                        'value': win_rate
                    })
                
                # Check drawdown
                max_drawdown = performance.get('max_drawdown', 0)
                if max_drawdown > 0.1:  # More than 10% drawdown
                    unhealthy_strategies.append({
                        'strategy_id': strategy_id,
                        'issue': 'high_drawdown',
                        'value': max_drawdown
                    })
                
                # Check trade frequency
                total_trades = performance.get('total_trades', 0)
                if total_trades == 0:  # No trades
                    unhealthy_strategies.append({
                        'strategy_id': strategy_id,
                        'issue': 'no_trades',
                        'value': 0
                    })
            
            if unhealthy_strategies:
                await self.handle_unhealthy_strategies(unhealthy_strategies)
                
        except Exception as e:
            await self.handle_error(e, "strategy health check")
    
    async def handle_unhealthy_strategies(self, unhealthy_strategies: List[Dict[str, Any]]):
        """Handle unhealthy strategies"""
        try:
            for strategy_issue in unhealthy_strategies:
                strategy_id = strategy_issue['strategy_id']
                issue = strategy_issue['issue']
                
                if issue == 'low_win_rate':
                    # Recommend optimization
                    await self.recommend_strategy_optimization(strategy_id, 'low_win_rate')
                elif issue == 'high_drawdown':
                    # Recommend risk reduction
                    await self.recommend_risk_reduction(strategy_id)
                elif issue == 'no_trades':
                    # Check strategy parameters
                    await self.check_strategy_parameters(strategy_id)
                
                await self.log_activity(f"Handled unhealthy strategy: {strategy_id} - {issue}")
                
        except Exception as e:
            await self.handle_error(e, "unhealthy strategy handling")
    
    async def update_strategy_performance(self):
        """Update strategy performance metrics"""
        try:
            # This would get performance data from strategy manager
            # For now, we'll simulate performance updates
            
            for strategy_id in self.active_strategies:
                # Get performance metrics
                performance = await self.get_strategy_performance(strategy_id)
                
                if performance:
                    self.strategy_performance[strategy_id] = performance
                    
                    # Calculate performance score
                    performance_score = self.calculate_performance_score(performance)
                    self.strategy_performance[strategy_id]['performance_score'] = performance_score
            
            await self.log_activity(f"Updated performance for {len(self.active_strategies)} strategies")
            
        except Exception as e:
            await self.handle_error(e, "strategy performance update")
    
    async def get_strategy_performance(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Get performance metrics for a strategy"""
        try:
            # This would query the strategy manager for performance data
            # For now, return simulated data
            
            import random
            
            return {
                'total_trades': random.randint(0, 100),
                'winning_trades': random.randint(0, 60),
                'losing_trades': random.randint(0, 40),
                'win_rate': random.uniform(30, 70),
                'total_profit': random.uniform(-1000, 2000),
                'max_drawdown': random.uniform(0, 0.15),
                'sharpe_ratio': random.uniform(-1, 2),
                'avg_trade_duration': random.uniform(300, 3600)
            }
            
        except Exception as e:
            await self.handle_error(e, "strategy performance retrieval")
            return None
    
    def calculate_performance_score(self, performance: Dict[str, Any]) -> float:
        """Calculate overall performance score"""
        try:
            # Weighted scoring system
            weights = {
                'win_rate': 0.3,
                'profit_factor': 0.25,
                'sharpe_ratio': 0.2,
                'max_drawdown': 0.15,
                'trade_frequency': 0.1
            }
            
            # Normalize metrics
            win_rate_score = min(performance.get('win_rate', 0) / 100, 1.0)
            
            profit_factor = abs(performance.get('total_profit', 0)) / 1000
            profit_factor_score = min(profit_factor, 1.0)
            
            sharpe_ratio = performance.get('sharpe_ratio', 0)
            sharpe_score = min(max(sharpe_ratio + 1, 0) / 3, 1.0)
            
            drawdown = performance.get('max_drawdown', 0)
            drawdown_score = max(1 - drawdown * 5, 0)
            
            total_trades = performance.get('total_trades', 0)
            frequency_score = min(total_trades / 50, 1.0)
            
            # Calculate weighted score
            score = (
                weights['win_rate'] * win_rate_score +
                weights['profit_factor'] * profit_factor_score +
                weights['sharpe_ratio'] * sharpe_score +
                weights['max_drawdown'] * drawdown_score +
                weights['trade_frequency'] * frequency_score
            )
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error calculating performance score: {e}")
            return 0.0
    
    async def identify_optimization_candidates(self) -> List[str]:
        """Identify strategies that need optimization"""
        try:
            candidates = []
            
            for strategy_id, performance in self.strategy_performance.items():
                performance_score = performance.get('performance_score', 0)
                total_trades = performance.get('total_trades', 0)
                
                # Check if strategy meets optimization criteria
                if (performance_score < self.performance_threshold and 
                    total_trades >= self.min_trades_for_optimization):
                    candidates.append(strategy_id)
            
            return candidates
            
        except Exception as e:
            await self.handle_error(e, "optimization candidate identification")
            return []
    
    async def optimize_strategy(self, strategy_id: str):
        """Optimize a specific strategy"""
        try:
            strategy_data = self.active_strategies.get(strategy_id)
            if not strategy_data:
                return
            
            strategy_type = strategy_data.get('strategy_type', 'swing_trade')
            
            # Get strategy template
            template = self.strategy_templates.get(strategy_type)
            if not template:
                return
            
            # Choose optimization method
            optimization_method = self.choose_optimization_method(strategy_type)
            
            # Run optimization
            optimization_func = self.optimization_methods[optimization_method]
            result = await optimization_func(strategy_id, template)
            
            if result:
                # Store optimization result
                optimization = StrategyOptimization(
                    strategy_id=strategy_id,
                    strategy_type=strategy_type,
                    parameters=result['parameters'],
                    performance_score=result['performance_score'],
                    metrics=result['metrics'],
                    confidence=result['confidence'],
                    optimization_method=optimization_method,
                    timestamp=datetime.now()
                )
                
                self.optimization_history.append(optimization)
                
                # Send optimization result
                await self.send_optimization_result(optimization)
                
                # Update metrics
                self.optimization_metrics['strategies_optimized'] += 1
                if result['performance_score'] > self.performance_threshold:
                    self.optimization_metrics['successful_optimizations'] += 1
                
                await self.log_activity(f"Optimized strategy: {strategy_id} using {optimization_method.value}")
            
        except Exception as e:
            await self.handle_error(e, "strategy optimization")
    
    def choose_optimization_method(self, strategy_type: str) -> OptimizationMethod:
        """Choose appropriate optimization method"""
        try:
            # Choose based on strategy type and complexity
            if strategy_type == 'scalping':
                return OptimizationMethod.GRID_SEARCH  # Fast execution needed
            elif strategy_type == 'options':
                return OptimizationMethod.BAYESIAN_OPTIMIZATION  # Complex parameter space
            else:
                return OptimizationMethod.GENETIC_ALGORITHM  # General purpose
                
        except Exception as e:
            return OptimizationMethod.RANDOM_SEARCH
    
    async def genetic_algorithm_optimization(self, strategy_id: str, template: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Optimize strategy using genetic algorithm"""
        try:
            parameters = template['parameters']
            
            # Initialize population
            population = self.initialize_population(parameters, self.ga_params['population_size'])
            
            best_fitness = -float('inf')
            best_individual = None
            
            for generation in range(self.ga_params['generations']):
                # Evaluate fitness
                fitness_scores = []
                for individual in population:
                    fitness = await self.evaluate_strategy_fitness(strategy_id, individual)
                    fitness_scores.append(fitness)
                
                # Find best individual
                max_fitness_idx = np.argmax(fitness_scores)
                if fitness_scores[max_fitness_idx] > best_fitness:
                    best_fitness = fitness_scores[max_fitness_idx]
                    best_individual = population[max_fitness_idx]
                
                # Selection
                selected = self.tournament_selection(population, fitness_scores)
                
                # Crossover and mutation
                new_population = []
                for i in range(0, len(selected), 2):
                    parent1, parent2 = selected[i], selected[i + 1] if i + 1 < len(selected) else selected[i]
                    
                    if np.random.random() < self.ga_params['crossover_rate']:
                        child1, child2 = self.crossover(parent1, parent2, parameters)
                    else:
                        child1, child2 = parent1.copy(), parent2.copy()
                    
                    if np.random.random() < self.ga_params['mutation_rate']:
                        child1 = self.mutate(child1, parameters)
                    if np.random.random() < self.ga_params['mutation_rate']:
                        child2 = self.mutate(child2, parameters)
                    
                    new_population.extend([child1, child2])
                
                # Keep elite
                elite_indices = np.argsort(fitness_scores)[-self.ga_params['elite_size']:]
                elite = [population[i] for i in elite_indices]
                
                population = elite + new_population[:self.ga_params['population_size'] - len(elite)]
            
            if best_individual:
                # Evaluate final performance
                final_metrics = await self.evaluate_strategy_comprehensive(strategy_id, best_individual)
                
                return {
                    'parameters': best_individual,
                    'performance_score': best_fitness,
                    'metrics': final_metrics,
                    'confidence': 0.8
                }
            
            return None
            
        except Exception as e:
            await self.handle_error(e, "genetic algorithm optimization")
            return None
    
    def initialize_population(self, parameters: Dict[str, Any], size: int) -> List[Dict[str, Any]]:
        """Initialize population for genetic algorithm"""
        try:
            population = []
            
            for _ in range(size):
                individual = {}
                for param_name, param_config in parameters.items():
                    min_val = param_config['min']
                    max_val = param_config['max']
                    
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        individual[param_name] = np.random.randint(min_val, max_val + 1)
                    else:
                        individual[param_name] = np.random.uniform(min_val, max_val)
                
                population.append(individual)
            
            return population
            
        except Exception as e:
            self.logger.error(f"Error initializing population: {e}")
            return []
    
    async def evaluate_strategy_fitness(self, strategy_id: str, parameters: Dict[str, Any]) -> float:
        """Evaluate fitness of strategy parameters"""
        try:
            # This would run backtest with parameters
            # For now, return random fitness
            import random
            return random.uniform(0, 1)
            
        except Exception as e:
            await self.handle_error(e, "strategy fitness evaluation")
            return 0.0
    
    def tournament_selection(self, population: List[Dict[str, Any]], fitness_scores: List[float]) -> List[Dict[str, Any]]:
        """Tournament selection for genetic algorithm"""
        try:
            tournament_size = 3
            selected = []
            
            for _ in range(len(population)):
                tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                selected.append(population[winner_idx].copy())
            
            return selected
            
        except Exception as e:
            self.logger.error(f"Error in tournament selection: {e}")
            return population
    
    def crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any], parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Crossover operation for genetic algorithm"""
        try:
            child1, child2 = parent1.copy(), parent2.copy()
            
            for param_name in parameters:
                if np.random.random() < 0.5:
                    child1[param_name], child2[param_name] = child2[param_name], child1[param_name]
            
            return child1, child2
            
        except Exception as e:
            self.logger.error(f"Error in crossover: {e}")
            return parent1, parent2
    
    def mutate(self, individual: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Mutation operation for genetic algorithm"""
        try:
            mutated = individual.copy()
            
            for param_name, param_config in parameters.items():
                if np.random.random() < 0.1:  # 10% mutation rate per parameter
                    min_val = param_config['min']
                    max_val = param_config['max']
                    
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        mutated[param_name] = np.random.randint(min_val, max_val + 1)
                    else:
                        mutated[param_name] = np.random.uniform(min_val, max_val)
            
            return mutated
            
        except Exception as e:
            self.logger.error(f"Error in mutation: {e}")
            return individual
    
    async def grid_search_optimization(self, strategy_id: str, template: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Optimize strategy using grid search"""
        try:
            parameters = template['parameters']
            
            # Generate parameter grid
            param_grid = self.generate_parameter_grid(parameters)
            
            # Limit combinations
            if len(param_grid) > self.grid_search_params['max_combinations']:
                param_grid = np.random.choice(param_grid, self.grid_search_params['max_combinations'], replace=False)
            
            best_score = -float('inf')
            best_params = None
            
            # Evaluate each combination
            for params in param_grid:
                score = await self.evaluate_strategy_fitness(strategy_id, params)
                
                if score > best_score:
                    best_score = score
                    best_params = params
            
            if best_params:
                final_metrics = await self.evaluate_strategy_comprehensive(strategy_id, best_params)
                
                return {
                    'parameters': best_params,
                    'performance_score': best_score,
                    'metrics': final_metrics,
                    'confidence': 0.9
                }
            
            return None
            
        except Exception as e:
            await self.handle_error(e, "grid search optimization")
            return None
    
    def generate_parameter_grid(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate parameter grid for grid search"""
        try:
            grid = []
            
            # Create parameter combinations
            param_values = {}
            for param_name, param_config in parameters.items():
                min_val = param_config['min']
                max_val = param_config['max']
                
                if isinstance(min_val, int) and isinstance(max_val, int):
                    param_values[param_name] = list(range(min_val, max_val + 1))
                else:
                    param_values[param_name] = [
                        min_val + i * (max_val - min_val) / 4 for i in range(5)
                    ]
            
            # Generate combinations
            import itertools
            
            keys = list(param_values.keys())
            values = list(param_values.values())
            
            for combination in itertools.product(*values):
                grid.append(dict(zip(keys, combination)))
            
            return grid
            
        except Exception as e:
            self.logger.error(f"Error generating parameter grid: {e}")
            return []
    
    async def bayesian_optimization(self, strategy_id: str, template: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Optimize strategy using Bayesian optimization"""
        try:
            # This is a simplified Bayesian optimization
            # In practice, you'd use libraries like scikit-optimize
            
            parameters = template['parameters']
            
            # Sample initial points
            initial_points = self.initialize_population(parameters, self.bayesian_params['initial_samples'])
            
            best_score = -float('inf')
            best_params = None
            
            # Evaluate initial points
            for params in initial_points:
                score = await self.evaluate_strategy_fitness(strategy_id, params)
                
                if score > best_score:
                    best_score = score
                    best_params = params
            
            # Simplified acquisition function (random sampling)
            for _ in range(self.bayesian_params['max_iterations']):
                # Generate candidate
                candidate = {}
                for param_name, param_config in parameters.items():
                    min_val = param_config['min']
                    max_val = param_config['max']
                    
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        candidate[param_name] = np.random.randint(min_val, max_val + 1)
                    else:
                        candidate[param_name] = np.random.uniform(min_val, max_val)
                
                # Evaluate candidate
                score = await self.evaluate_strategy_fitness(strategy_id, candidate)
                
                if score > best_score:
                    best_score = score
                    best_params = candidate
            
            if best_params:
                final_metrics = await self.evaluate_strategy_comprehensive(strategy_id, best_params)
                
                return {
                    'parameters': best_params,
                    'performance_score': best_score,
                    'metrics': final_metrics,
                    'confidence': 0.75
                }
            
            return None
            
        except Exception as e:
            await self.handle_error(e, "Bayesian optimization")
            return None
    
    async def random_search_optimization(self, strategy_id: str, template: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Optimize strategy using random search"""
        try:
            parameters = template['parameters']
            
            best_score = -float('inf')
            best_params = None
            
            # Random search
            for _ in range(50):  # 50 random evaluations
                params = {}
                for param_name, param_config in parameters.items():
                    min_val = param_config['min']
                    max_val = param_config['max']
                    
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        params[param_name] = np.random.randint(min_val, max_val + 1)
                    else:
                        params[param_name] = np.random.uniform(min_val, max_val)
                
                score = await self.evaluate_strategy_fitness(strategy_id, params)
                
                if score > best_score:
                    best_score = score
                    best_params = params
            
            if best_params:
                final_metrics = await self.evaluate_strategy_comprehensive(strategy_id, best_params)
                
                return {
                    'parameters': best_params,
                    'performance_score': best_score,
                    'metrics': final_metrics,
                    'confidence': 0.6
                }
            
            return None
            
        except Exception as e:
            await self.handle_error(e, "random search optimization")
            return None
    
    async def evaluate_strategy_comprehensive(self, strategy_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive strategy evaluation"""
        try:
            # This would run detailed backtests
            # For now, return simulated metrics
            
            import random
            
            return {
                'total_return': random.uniform(-0.1, 0.3),
                'volatility': random.uniform(0.1, 0.4),
                'sharpe_ratio': random.uniform(-0.5, 2.0),
                'max_drawdown': random.uniform(0.02, 0.15),
                'win_rate': random.uniform(0.4, 0.7),
                'profit_factor': random.uniform(0.8, 2.5),
                'total_trades': random.randint(20, 200)
            }
            
        except Exception as e:
            await self.handle_error(e, "comprehensive strategy evaluation")
            return {}
    
    async def send_optimization_result(self, optimization: StrategyOptimization):
        """Send optimization result to other agents"""
        try:
            message = {
                'type': 'strategy_optimization',
                'optimization': {
                    'strategy_id': optimization.strategy_id,
                    'strategy_type': optimization.strategy_type,
                    'parameters': optimization.parameters,
                    'performance_score': optimization.performance_score,
                    'metrics': optimization.metrics,
                    'confidence': optimization.confidence,
                    'optimization_method': optimization.optimization_method.value,
                    'timestamp': optimization.timestamp
                }
            }
            
            await self.send_message_to_mcp(
                "strategy_update",
                "broadcast",
                message,
                priority=2
            )
            
        except Exception as e:
            await self.handle_error(e, "optimization result sending")
    
    async def adapt_strategies_to_market(self):
        """Adapt strategies to current market conditions"""
        try:
            adaptations = []
            
            for strategy_id, strategy_data in self.active_strategies.items():
                strategy_type = strategy_data.get('strategy_type', 'swing_trade')
                
                # Get regime-specific recommendations
                recommendations = self.get_regime_recommendations(strategy_type, self.current_market_regime)
                
                if recommendations:
                    adaptations.append({
                        'strategy_id': strategy_id,
                        'strategy_type': strategy_type,
                        'recommendations': recommendations,
                        'market_regime': self.current_market_regime
                    })
            
            if adaptations:
                await self.send_adaptation_recommendations(adaptations)
                
        except Exception as e:
            await self.handle_error(e, "strategy adaptation")
    
    def get_regime_recommendations(self, strategy_type: str, market_regime: str) -> List[Dict[str, Any]]:
        """Get strategy recommendations for market regime"""
        try:
            recommendations = []
            
            if market_regime == 'bullish':
                if strategy_type == 'swing_trade':
                    recommendations.append({
                        'parameter': 'take_profit_pct',
                        'adjustment': 'increase',
                        'reason': 'Bull market - extend profit targets'
                    })
                elif strategy_type == 'scalping':
                    recommendations.append({
                        'parameter': 'target_profit_ticks',
                        'adjustment': 'increase',
                        'reason': 'Bull market - higher profit targets'
                    })
            
            elif market_regime == 'bearish':
                if strategy_type == 'swing_trade':
                    recommendations.append({
                        'parameter': 'stop_loss_pct',
                        'adjustment': 'decrease',
                        'reason': 'Bear market - tighter stop losses'
                    })
                elif strategy_type == 'scalping':
                    recommendations.append({
                        'parameter': 'stop_loss_ticks',
                        'adjustment': 'decrease',
                        'reason': 'Bear market - tighter stops'
                    })
            
            elif market_regime == 'volatile':
                recommendations.append({
                    'parameter': 'position_size',
                    'adjustment': 'decrease',
                    'reason': 'High volatility - reduce position sizes'
                })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error getting regime recommendations: {e}")
            return []
    
    async def send_adaptation_recommendations(self, adaptations: List[Dict[str, Any]]):
        """Send adaptation recommendations"""
        try:
            message = {
                'type': 'strategy_adaptations',
                'adaptations': adaptations,
                'market_regime': self.current_market_regime,
                'timestamp': datetime.now()
            }
            
            await self.send_message_to_mcp(
                "strategy_update",
                "broadcast",
                message,
                priority=2
            )
            
            await self.log_activity(f"Sent {len(adaptations)} adaptation recommendations")
            
        except Exception as e:
            await self.handle_error(e, "adaptation recommendation sending")
    
    async def process_strategy_performance(self, performance_data: Dict[str, Any]):
        """Process strategy performance data"""
        try:
            strategy_id = performance_data.get('strategy_id')
            if strategy_id:
                self.strategy_performance[strategy_id] = performance_data
                
                # Update active strategies
                if strategy_id not in self.active_strategies:
                    self.active_strategies[strategy_id] = {
                        'strategy_id': strategy_id,
                        'strategy_type': performance_data.get('strategy_type', 'unknown'),
                        'last_update': datetime.now()
                    }
                
                await self.log_activity(f"Updated performance for strategy: {strategy_id}")
            
        except Exception as e:
            await self.handle_error(e, "strategy performance processing")
    
    async def process_optimization_request(self, request: Dict[str, Any]):
        """Process optimization request"""
        try:
            strategy_id = request.get('strategy_id')
            if strategy_id:
                await self.optimize_strategy(strategy_id)
                
        except Exception as e:
            await self.handle_error(e, "optimization request processing")
    
    async def process_market_analysis(self, analysis: Dict[str, Any]):
        """Process market analysis data"""
        try:
            # Update market regime based on analysis
            if 'market_conditions' in analysis:
                new_regime = await self.classify_market_regime(analysis)
                
                if new_regime != self.current_market_regime:
                    self.current_market_regime = new_regime
                    await self.adapt_strategies_to_regime(new_regime)
                    
                    await self.log_activity(f"Market regime updated to: {new_regime}")
            
        except Exception as e:
            await self.handle_error(e, "market analysis processing")
    
    async def adapt_strategies_to_regime(self, regime: str):
        """Adapt strategies to new market regime"""
        try:
            adaptations = []
            
            for strategy_id, strategy_data in self.active_strategies.items():
                strategy_type = strategy_data.get('strategy_type', 'swing_trade')
                recommendations = self.get_regime_recommendations(strategy_type, regime)
                
                if recommendations:
                    adaptations.append({
                        'strategy_id': strategy_id,
                        'recommendations': recommendations
                    })
            
            if adaptations:
                await self.send_adaptation_recommendations(adaptations)
                
        except Exception as e:
            await self.handle_error(e, "regime adaptation")
    
    async def process_strategy_update(self, update: Dict[str, Any]):
        """Process strategy update"""
        try:
            strategy_id = update.get('strategy_id')
            if strategy_id and strategy_id in self.active_strategies:
                self.active_strategies[strategy_id].update(update)
                await self.log_activity(f"Updated strategy: {strategy_id}")
                
        except Exception as e:
            await self.handle_error(e, "strategy update processing")
    
    def get_status(self) -> Dict[str, Any]:
        """Get strategizer agent status"""
        base_status = super().get_status()
        
        base_status.update({
            'active_strategies': len(self.active_strategies),
            'optimization_history': len(self.optimization_history),
            'current_market_regime': self.current_market_regime,
            'optimization_metrics': self.optimization_metrics,
            'recent_optimizations': len([o for o in self.optimization_history 
                                       if (datetime.now() - o.timestamp).days <= 7])
        })
        
        return base_status
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get optimization summary"""
        try:
            recent_optimizations = [
                o for o in self.optimization_history
                if (datetime.now() - o.timestamp).days <= 7
            ]
            
            return {
                'total_optimizations': len(self.optimization_history),
                'recent_optimizations': len(recent_optimizations),
                'success_rate': (self.optimization_metrics['successful_optimizations'] / 
                               max(self.optimization_metrics['strategies_optimized'], 1)) * 100,
                'average_improvement': self.optimization_metrics['average_improvement'],
                'best_performance': self.optimization_metrics['best_strategy_performance'],
                'current_market_regime': self.current_market_regime,
                'active_strategies': len(self.active_strategies)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting optimization summary: {e}")
            return {}
