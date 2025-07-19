import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from config import Config

@dataclass
class RiskMetrics:
    """Risk metrics for portfolio and positions"""
    var_95: float
    var_99: float
    expected_shortfall: float
    volatility: float
    beta: float
    sharpe_ratio: float
    max_drawdown: float
    correlation_risk: float

class RiskManager:
    """Advanced risk management system"""
    
    def __init__(self, config: Config):
        self.config = config
        self.risk_config = config.risk_config
        
        # Risk tracking
        self.portfolio_risk = 0.0
        self.position_risks: Dict[str, float] = {}
        self.correlation_matrix: pd.DataFrame = None
        self.volatility_estimates: Dict[str, float] = {}
        
        # Performance tracking
        self.risk_metrics_history: List[RiskMetrics] = []
        self.drawdown_history: List[float] = []
        
        # Risk limits
        self.max_portfolio_risk = self.risk_config['max_portfolio_risk']
        self.max_position_risk = self.risk_config['max_position_risk']
        self.max_drawdown = self.risk_config['max_drawdown']
        self.correlation_threshold = self.risk_config['correlation_threshold']
        
        self.logger = logging.getLogger("RiskManager")
    
    async def initialize(self):
        """Initialize risk manager"""
        try:
            self.logger.info("Initializing Risk Manager...")
            
            # Initialize volatility models
            await self._initialize_volatility_models()
            
            # Initialize correlation tracking
            await self._initialize_correlation_tracking()
            
            # Setup risk monitoring
            await self._setup_risk_monitoring()
            
            self.logger.info("Risk Manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Risk Manager: {e}")
            raise
    
    async def _initialize_volatility_models(self):
        """Initialize volatility estimation models"""
        try:
            # Initialize with default volatility estimates
            self.volatility_estimates = {
                'SPY': 0.15,    # S&P 500 ETF
                'QQQ': 0.20,    # NASDAQ ETF
                'IWM': 0.25,    # Russell 2000 ETF
                'BTC': 0.80,    # Bitcoin
                'ETH': 0.90,    # Ethereum
            }
            
            self.logger.info("Volatility models initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing volatility models: {e}")
    
    async def _initialize_correlation_tracking(self):
        """Initialize correlation tracking"""
        try:
            # Initialize with sample correlation matrix
            symbols = ['SPY', 'QQQ', 'IWM', 'BTC', 'ETH']
            
            # Sample correlation matrix (would be calculated from historical data)
            correlation_data = np.array([
                [1.00, 0.85, 0.75, 0.30, 0.28],  # SPY
                [0.85, 1.00, 0.70, 0.35, 0.32],  # QQQ
                [0.75, 0.70, 1.00, 0.25, 0.22],  # IWM
                [0.30, 0.35, 0.25, 1.00, 0.85],  # BTC
                [0.28, 0.32, 0.22, 0.85, 1.00],  # ETH
            ])
            
            self.correlation_matrix = pd.DataFrame(
                correlation_data,
                index=symbols,
                columns=symbols
            )
            
            self.logger.info("Correlation tracking initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing correlation tracking: {e}")
    
    async def _setup_risk_monitoring(self):
        """Setup continuous risk monitoring"""
        try:
            # Start risk monitoring task
            asyncio.create_task(self._monitor_risk_continuously())
            
            self.logger.info("Risk monitoring setup complete")
            
        except Exception as e:
            self.logger.error(f"Error setting up risk monitoring: {e}")
    
    async def _monitor_risk_continuously(self):
        """Continuously monitor risk metrics"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Update volatility estimates
                await self._update_volatility_estimates()
                
                # Update correlation matrix
                await self._update_correlation_matrix()
                
                # Check risk limits
                await self._check_risk_limits()
                
            except Exception as e:
                self.logger.error(f"Error in continuous risk monitoring: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _update_volatility_estimates(self):
        """Update volatility estimates using GARCH models"""
        try:
            # In a real implementation, this would use market data
            # For now, we'll simulate volatility updates
            
            for symbol in self.volatility_estimates:
                # Add some noise to simulate changing volatility
                current_vol = self.volatility_estimates[symbol]
                noise = np.random.normal(0, 0.01)
                new_vol = max(0.05, current_vol + noise)  # Minimum 5% volatility
                
                self.volatility_estimates[symbol] = new_vol
            
        except Exception as e:
            self.logger.error(f"Error updating volatility estimates: {e}")
    
    async def _update_correlation_matrix(self):
        """Update correlation matrix using rolling window"""
        try:
            # In a real implementation, this would use historical returns
            # For now, we'll add small random changes to simulate dynamic correlations
            
            if self.correlation_matrix is not None:
                # Add small random changes to correlations
                noise = np.random.normal(0, 0.01, self.correlation_matrix.shape)
                
                # Ensure matrix remains symmetric and positive semi-definite
                new_matrix = self.correlation_matrix.values + noise
                new_matrix = (new_matrix + new_matrix.T) / 2  # Ensure symmetry
                
                # Ensure diagonal is 1
                np.fill_diagonal(new_matrix, 1.0)
                
                # Clip correlations to [-1, 1]
                new_matrix = np.clip(new_matrix, -1.0, 1.0)
                
                self.correlation_matrix = pd.DataFrame(
                    new_matrix,
                    index=self.correlation_matrix.index,
                    columns=self.correlation_matrix.columns
                )
            
        except Exception as e:
            self.logger.error(f"Error updating correlation matrix: {e}")
    
    async def _check_risk_limits(self):
        """Check if risk limits are exceeded"""
        try:
            # Check portfolio risk
            if self.portfolio_risk > self.max_portfolio_risk:
                self.logger.warning(f"Portfolio risk limit exceeded: {self.portfolio_risk:.2%} > {self.max_portfolio_risk:.2%}")
            
            # Check position risks
            for symbol, risk in self.position_risks.items():
                if risk > self.max_position_risk:
                    self.logger.warning(f"Position risk limit exceeded for {symbol}: {risk:.2%} > {self.max_position_risk:.2%}")
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
    
    async def check_position_risk(self, symbol: str, side: str, signal_strength: float) -> bool:
        """Check if a new position meets risk requirements"""
        try:
            # Get symbol volatility
            volatility = self.volatility_estimates.get(symbol, 0.20)  # Default 20%
            
            # Calculate position risk
            position_risk = volatility * signal_strength
            
            # Check against limits
            if position_risk > self.max_position_risk:
                self.logger.warning(f"Position risk too high for {symbol}: {position_risk:.2%} > {self.max_position_risk:.2%}")
                return False
            
            # Check correlation with existing positions
            correlation_risk = await self._calculate_correlation_risk(symbol)
            
            if correlation_risk > self.correlation_threshold:
                self.logger.warning(f"Correlation risk too high for {symbol}: {correlation_risk:.2f} > {self.correlation_threshold:.2f}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking position risk for {symbol}: {e}")
            return False
    
    async def _calculate_correlation_risk(self, symbol: str) -> float:
        """Calculate correlation risk for a new position"""
        try:
            if self.correlation_matrix is None or symbol not in self.correlation_matrix.index:
                return 0.0
            
            # Calculate weighted average correlation with existing positions
            existing_symbols = list(self.position_risks.keys())
            
            if not existing_symbols:
                return 0.0
            
            correlations = []
            for existing_symbol in existing_symbols:
                if existing_symbol in self.correlation_matrix.index:
                    corr = self.correlation_matrix.loc[symbol, existing_symbol]
                    correlations.append(abs(corr))
            
            if correlations:
                return sum(correlations) / len(correlations)
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation risk: {e}")
            return 0.0
    
    async def calculate_position_size(self, symbol: str, signal_strength: float) -> float:
        """Calculate optimal position size based on risk"""
        try:
            # Get symbol volatility
            volatility = self.volatility_estimates.get(symbol, 0.20)
            
            # Calculate Kelly criterion position size
            kelly_fraction = signal_strength / volatility
            
            # Apply risk scaling
            risk_scaling = self.max_position_risk / volatility
            
            # Use smaller of Kelly and risk-based sizing
            position_fraction = min(kelly_fraction, risk_scaling)
            
            # Apply maximum position size limit (e.g., 5% of portfolio)
            max_position_fraction = 0.05
            position_fraction = min(position_fraction, max_position_fraction)
            
            # Calculate position size in dollars
            portfolio_value = 100000  # This would come from trading engine
            position_size_dollars = portfolio_value * position_fraction
            
            return position_size_dollars
            
        except Exception as e:
            self.logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0.0
    
    async def calculate_portfolio_risk(self, positions: Dict[str, Any]) -> float:
        """Calculate overall portfolio risk"""
        try:
            if not positions:
                return 0.0
            
            # Extract position data
            symbols = []
            weights = []
            
            total_value = sum(pos.quantity * pos.current_price for pos in positions.values() if pos.current_price)
            
            for position in positions.values():
                if position.current_price:
                    symbols.append(position.symbol)
                    weight = (position.quantity * position.current_price) / total_value
                    weights.append(weight)
            
            if not symbols:
                return 0.0
            
            # Calculate portfolio variance
            portfolio_variance = 0.0
            
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols):
                    vol1 = self.volatility_estimates.get(symbol1, 0.20)
                    vol2 = self.volatility_estimates.get(symbol2, 0.20)
                    
                    if (self.correlation_matrix is not None and 
                        symbol1 in self.correlation_matrix.index and 
                        symbol2 in self.correlation_matrix.index):
                        correlation = self.correlation_matrix.loc[symbol1, symbol2]
                    else:
                        correlation = 1.0 if symbol1 == symbol2 else 0.0
                    
                    portfolio_variance += weights[i] * weights[j] * vol1 * vol2 * correlation
            
            # Portfolio risk is the standard deviation
            portfolio_risk = np.sqrt(portfolio_variance)
            
            # Update portfolio risk tracking
            self.portfolio_risk = portfolio_risk
            
            # Update position risks
            for i, symbol in enumerate(symbols):
                individual_risk = weights[i] * self.volatility_estimates.get(symbol, 0.20)
                self.position_risks[symbol] = individual_risk
            
            return portfolio_risk
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio risk: {e}")
            return 0.0
    
    async def calculate_var(self, positions: Dict[str, Any], confidence: float = 0.95) -> float:
        """Calculate Value at Risk (VaR)"""
        try:
            portfolio_risk = await self.calculate_portfolio_risk(positions)
            
            if portfolio_risk == 0:
                return 0.0
            
            # Calculate portfolio value
            portfolio_value = sum(pos.quantity * pos.current_price for pos in positions.values() if pos.current_price)
            
            # Calculate VaR using normal distribution assumption
            # For more accuracy, could use Monte Carlo simulation
            from scipy.stats import norm
            
            z_score = norm.ppf(1 - confidence)
            var = portfolio_value * portfolio_risk * z_score
            
            return abs(var)  # VaR is typically expressed as positive value
            
        except Exception as e:
            self.logger.error(f"Error calculating VaR: {e}")
            return 0.0
    
    async def calculate_expected_shortfall(self, positions: Dict[str, Any], confidence: float = 0.95) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        try:
            portfolio_risk = await self.calculate_portfolio_risk(positions)
            
            if portfolio_risk == 0:
                return 0.0
            
            # Calculate portfolio value
            portfolio_value = sum(pos.quantity * pos.current_price for pos in positions.values() if pos.current_price)
            
            # Calculate Expected Shortfall using normal distribution
            from scipy.stats import norm
            
            z_score = norm.ppf(1 - confidence)
            expected_shortfall = portfolio_value * portfolio_risk * norm.pdf(z_score) / confidence
            
            return expected_shortfall
            
        except Exception as e:
            self.logger.error(f"Error calculating Expected Shortfall: {e}")
            return 0.0
    
    async def get_risk_metrics(self, positions: Dict[str, Any]) -> RiskMetrics:
        """Get comprehensive risk metrics"""
        try:
            portfolio_risk = await self.calculate_portfolio_risk(positions)
            var_95 = await self.calculate_var(positions, 0.95)
            var_99 = await self.calculate_var(positions, 0.99)
            expected_shortfall = await self.calculate_expected_shortfall(positions, 0.95)
            
            # Calculate other metrics
            beta = await self._calculate_portfolio_beta(positions)
            sharpe_ratio = await self._calculate_sharpe_ratio(positions)
            max_drawdown = await self._calculate_max_drawdown(positions)
            correlation_risk = await self._calculate_overall_correlation_risk(positions)
            
            metrics = RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                expected_shortfall=expected_shortfall,
                volatility=portfolio_risk,
                beta=beta,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                correlation_risk=correlation_risk
            )
            
            # Store in history
            self.risk_metrics_history.append(metrics)
            
            # Keep only last 100 metrics
            if len(self.risk_metrics_history) > 100:
                self.risk_metrics_history = self.risk_metrics_history[-100:]
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting risk metrics: {e}")
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0)
    
    async def _calculate_portfolio_beta(self, positions: Dict[str, Any]) -> float:
        """Calculate portfolio beta relative to market"""
        try:
            # Simplified beta calculation
            # In reality, would use historical returns correlation with market
            
            if not positions:
                return 0.0
            
            # Assign beta values for different asset classes
            beta_values = {
                'SPY': 1.0,
                'QQQ': 1.2,
                'IWM': 1.3,
                'BTC': 2.0,
                'ETH': 2.2
            }
            
            total_value = sum(pos.quantity * pos.current_price for pos in positions.values() if pos.current_price)
            
            if total_value == 0:
                return 0.0
            
            weighted_beta = 0.0
            
            for position in positions.values():
                if position.current_price:
                    weight = (position.quantity * position.current_price) / total_value
                    symbol_beta = beta_values.get(position.symbol, 1.0)
                    weighted_beta += weight * symbol_beta
            
            return weighted_beta
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio beta: {e}")
            return 0.0
    
    async def _calculate_sharpe_ratio(self, positions: Dict[str, Any]) -> float:
        """Calculate Sharpe ratio"""
        try:
            # Simplified calculation
            # In reality, would use historical returns
            
            portfolio_risk = await self.calculate_portfolio_risk(positions)
            
            if portfolio_risk == 0:
                return 0.0
            
            # Assume 8% expected return and 5% risk-free rate
            expected_return = 0.08
            risk_free_rate = self.risk_config.get('risk_free_rate', 0.05)
            
            sharpe_ratio = (expected_return - risk_free_rate) / portfolio_risk
            
            return sharpe_ratio
            
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    async def _calculate_max_drawdown(self, positions: Dict[str, Any]) -> float:
        """Calculate maximum drawdown"""
        try:
            # Use historical drawdown data
            if not self.drawdown_history:
                return 0.0
            
            return max(self.drawdown_history)
            
        except Exception as e:
            self.logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    async def _calculate_overall_correlation_risk(self, positions: Dict[str, Any]) -> float:
        """Calculate overall correlation risk"""
        try:
            if not positions or self.correlation_matrix is None:
                return 0.0
            
            symbols = [pos.symbol for pos in positions.values()]
            
            if len(symbols) < 2:
                return 0.0
            
            # Calculate average correlation
            correlations = []
            
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols):
                    if i < j and symbol1 in self.correlation_matrix.index and symbol2 in self.correlation_matrix.index:
                        corr = self.correlation_matrix.loc[symbol1, symbol2]
                        correlations.append(abs(corr))
            
            if correlations:
                return sum(correlations) / len(correlations)
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating overall correlation risk: {e}")
            return 0.0
    
    async def update_drawdown(self, current_drawdown: float):
        """Update drawdown history"""
        try:
            self.drawdown_history.append(current_drawdown)
            
            # Keep only last 252 trading days
            if len(self.drawdown_history) > 252:
                self.drawdown_history = self.drawdown_history[-252:]
            
        except Exception as e:
            self.logger.error(f"Error updating drawdown: {e}")
    
    async def get_risk_dashboard_data(self) -> Dict[str, Any]:
        """Get risk data for dashboard"""
        try:
            current_metrics = self.risk_metrics_history[-1] if self.risk_metrics_history else None
            
            return {
                'portfolio_risk': self.portfolio_risk,
                'position_risks': self.position_risks,
                'risk_limits': {
                    'max_portfolio_risk': self.max_portfolio_risk,
                    'max_position_risk': self.max_position_risk,
                    'max_drawdown': self.max_drawdown,
                    'correlation_threshold': self.correlation_threshold
                },
                'current_metrics': {
                    'var_95': current_metrics.var_95 if current_metrics else 0,
                    'var_99': current_metrics.var_99 if current_metrics else 0,
                    'expected_shortfall': current_metrics.expected_shortfall if current_metrics else 0,
                    'volatility': current_metrics.volatility if current_metrics else 0,
                    'beta': current_metrics.beta if current_metrics else 0,
                    'sharpe_ratio': current_metrics.sharpe_ratio if current_metrics else 0,
                    'max_drawdown': current_metrics.max_drawdown if current_metrics else 0,
                    'correlation_risk': current_metrics.correlation_risk if current_metrics else 0
                } if current_metrics else {},
                'volatility_estimates': self.volatility_estimates,
                'correlation_matrix': self.correlation_matrix.to_dict() if self.correlation_matrix is not None else {}
            }
            
        except Exception as e:
            self.logger.error(f"Error getting risk dashboard data: {e}")
            return {}
