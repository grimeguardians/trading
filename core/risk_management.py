"""
Advanced Risk Management System
Implements comprehensive risk controls, portfolio risk assessment, and dynamic risk adjustments
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from config import Config
from models.trading_models import Position, Trade, Order
from core.mathematical_models import TradingSignal

class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class RiskMetrics:
    """Risk assessment metrics"""
    # Portfolio risk metrics
    total_exposure: float
    portfolio_var: float  # Value at Risk
    portfolio_volatility: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    
    # Position risk metrics
    position_concentration: float
    sector_concentration: float
    correlation_risk: float
    
    # Market risk metrics
    market_beta: float
    market_correlation: float
    
    # Risk levels
    overall_risk_level: RiskLevel
    risk_score: float  # 0-100
    
    # Limits and constraints
    position_limit_utilization: float
    daily_loss_utilization: float
    drawdown_utilization: float
    
    timestamp: datetime

@dataclass
class RiskAlert:
    """Risk alert notification"""
    alert_id: str
    alert_type: str
    severity: str  # info, warning, error, critical
    message: str
    affected_positions: List[str]
    recommended_actions: List[str]
    timestamp: datetime

class RiskManager:
    """Advanced risk management system"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("RiskManager")
        
        # Risk limits and parameters
        self.max_position_size = config.trading.max_position_size
        self.max_daily_loss = config.trading.max_daily_loss
        self.max_drawdown = config.trading.max_drawdown
        self.max_correlation = 0.7  # Maximum correlation between positions
        self.max_sector_concentration = 0.3  # Maximum 30% in any sector
        
        # Risk tracking
        self.risk_metrics_history: List[RiskMetrics] = []
        self.risk_alerts: List[RiskAlert] = []
        self.position_risk_cache: Dict[str, float] = {}
        
        # Portfolio tracking
        self.portfolio_value_history: List[float] = []
        self.daily_pnl_history: List[float] = []
        self.peak_value = 0.0
        
        # Risk assessment parameters
        self.lookback_period = 30  # Days
        self.var_confidence = 0.95  # 95% VaR
        self.stress_test_scenarios = self._initialize_stress_scenarios()
        
    async def initialize(self):
        """Initialize risk management system"""
        try:
            self.logger.info("Initializing Risk Management System...")
            
            # Load historical data if available
            await self._load_historical_data()
            
            # Initialize risk monitoring
            await self._initialize_risk_monitoring()
            
            self.logger.info("Risk Management System initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize risk management: {e}")
            raise
    
    async def check_signal_risk(self, signal: TradingSignal) -> bool:
        """
        Check if a trading signal passes risk management criteria
        
        Args:
            signal: Trading signal to evaluate
        
        Returns:
            bool: True if signal passes risk checks
        """
        try:
            # Check position size limits
            if not self._check_position_size_limit(signal):
                self.logger.warning(f"Signal rejected: Position size limit exceeded for {signal.symbol}")
                return False
            
            # Check portfolio concentration
            if not self._check_concentration_limit(signal):
                self.logger.warning(f"Signal rejected: Concentration limit exceeded for {signal.symbol}")
                return False
            
            # Check correlation risk
            if not await self._check_correlation_risk(signal):
                self.logger.warning(f"Signal rejected: Correlation risk too high for {signal.symbol}")
                return False
            
            # Check daily loss limit
            if not self._check_daily_loss_limit(signal):
                self.logger.warning(f"Signal rejected: Daily loss limit would be exceeded")
                return False
            
            # Check volatility risk
            if not self._check_volatility_risk(signal):
                self.logger.warning(f"Signal rejected: Volatility risk too high for {signal.symbol}")
                return False
            
            # Check market conditions
            if not await self._check_market_conditions(signal):
                self.logger.warning(f"Signal rejected: Unfavorable market conditions")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking signal risk: {e}")
            return False
    
    async def assess_portfolio_risk(self, positions: List[Position]) -> RiskMetrics:
        """
        Assess comprehensive portfolio risk
        
        Args:
            positions: List of current positions
        
        Returns:
            RiskMetrics: Comprehensive risk assessment
        """
        try:
            # Calculate portfolio value
            portfolio_value = sum(pos.quantity * pos.current_price for pos in positions)
            
            # Calculate total exposure
            total_exposure = sum(abs(pos.quantity * pos.current_price) for pos in positions)
            
            # Calculate portfolio volatility
            portfolio_volatility = await self._calculate_portfolio_volatility(positions)
            
            # Calculate Value at Risk
            portfolio_var = await self._calculate_portfolio_var(positions)
            
            # Calculate maximum drawdown
            max_drawdown = self._calculate_max_drawdown()
            
            # Calculate Sharpe ratio
            sharpe_ratio = self._calculate_sharpe_ratio()
            
            # Calculate Sortino ratio
            sortino_ratio = self._calculate_sortino_ratio()
            
            # Calculate concentration metrics
            position_concentration = self._calculate_position_concentration(positions)
            sector_concentration = self._calculate_sector_concentration(positions)
            
            # Calculate correlation risk
            correlation_risk = await self._calculate_correlation_risk(positions)
            
            # Calculate market risk metrics
            market_beta = await self._calculate_market_beta(positions)
            market_correlation = await self._calculate_market_correlation(positions)
            
            # Calculate overall risk level
            overall_risk_level, risk_score = self._calculate_overall_risk_level(
                portfolio_var, portfolio_volatility, max_drawdown, 
                position_concentration, correlation_risk
            )
            
            # Calculate limit utilization
            position_limit_utilization = total_exposure / (portfolio_value * self.max_position_size) if portfolio_value > 0 else 0
            daily_loss_utilization = abs(self.daily_pnl_history[-1] if self.daily_pnl_history else 0) / (portfolio_value * self.max_daily_loss) if portfolio_value > 0 else 0
            drawdown_utilization = abs(max_drawdown) / self.max_drawdown if self.max_drawdown > 0 else 0
            
            risk_metrics = RiskMetrics(
                total_exposure=total_exposure,
                portfolio_var=portfolio_var,
                portfolio_volatility=portfolio_volatility,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                position_concentration=position_concentration,
                sector_concentration=sector_concentration,
                correlation_risk=correlation_risk,
                market_beta=market_beta,
                market_correlation=market_correlation,
                overall_risk_level=overall_risk_level,
                risk_score=risk_score,
                position_limit_utilization=position_limit_utilization,
                daily_loss_utilization=daily_loss_utilization,
                drawdown_utilization=drawdown_utilization,
                timestamp=datetime.now()
            )
            
            # Store in history
            self.risk_metrics_history.append(risk_metrics)
            
            # Generate alerts if necessary
            await self._generate_risk_alerts(risk_metrics)
            
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f"Error assessing portfolio risk: {e}")
            # Return default risk metrics
            return RiskMetrics(
                total_exposure=0.0,
                portfolio_var=0.0,
                portfolio_volatility=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                position_concentration=0.0,
                sector_concentration=0.0,
                correlation_risk=0.0,
                market_beta=0.0,
                market_correlation=0.0,
                overall_risk_level=RiskLevel.LOW,
                risk_score=0.0,
                position_limit_utilization=0.0,
                daily_loss_utilization=0.0,
                drawdown_utilization=0.0,
                timestamp=datetime.now()
            )
    
    async def calculate_position_sizing(self, signal: TradingSignal, portfolio_value: float) -> float:
        """
        Calculate optimal position size based on risk management
        
        Args:
            signal: Trading signal
            portfolio_value: Current portfolio value
        
        Returns:
            float: Recommended position size
        """
        try:
            # Base position size from configuration
            base_size = portfolio_value * self.max_position_size
            
            # Adjust for signal confidence
            confidence_multiplier = min(signal.confidence, 1.0)
            
            # Adjust for volatility
            volatility_adjustment = await self._calculate_volatility_adjustment(signal.symbol)
            
            # Adjust for correlation
            correlation_adjustment = await self._calculate_correlation_adjustment(signal.symbol)
            
            # Calculate Kelly criterion sizing
            kelly_size = self._calculate_kelly_sizing(signal)
            
            # Take the minimum of all sizing methods
            position_size = min(
                base_size * confidence_multiplier,
                base_size * volatility_adjustment,
                base_size * correlation_adjustment,
                kelly_size
            )
            
            # Ensure position size is within limits
            position_size = max(0, min(position_size, base_size))
            
            # Convert to quantity
            if signal.entry_price > 0:
                quantity = position_size / signal.entry_price
            else:
                quantity = 0
            
            return quantity
            
        except Exception as e:
            self.logger.error(f"Error calculating position sizing: {e}")
            return 0.0
    
    async def update_risk_metrics(self, portfolio_value: float, daily_pnl: float):
        """Update risk tracking metrics"""
        try:
            # Update portfolio value history
            self.portfolio_value_history.append(portfolio_value)
            
            # Update daily PnL history
            self.daily_pnl_history.append(daily_pnl)
            
            # Update peak value
            if portfolio_value > self.peak_value:
                self.peak_value = portfolio_value
            
            # Trim history to lookback period
            if len(self.portfolio_value_history) > self.lookback_period:
                self.portfolio_value_history = self.portfolio_value_history[-self.lookback_period:]
            
            if len(self.daily_pnl_history) > self.lookback_period:
                self.daily_pnl_history = self.daily_pnl_history[-self.lookback_period:]
            
        except Exception as e:
            self.logger.error(f"Error updating risk metrics: {e}")
    
    def get_risk_alerts(self, severity: Optional[str] = None) -> List[RiskAlert]:
        """Get risk alerts, optionally filtered by severity"""
        if severity:
            return [alert for alert in self.risk_alerts if alert.severity == severity]
        return self.risk_alerts
    
    def clear_risk_alerts(self):
        """Clear all risk alerts"""
        self.risk_alerts.clear()
    
    # Private helper methods
    def _check_position_size_limit(self, signal: TradingSignal) -> bool:
        """Check if position size is within limits"""
        # This would check against current portfolio value
        # Simplified implementation
        return True
    
    def _check_concentration_limit(self, signal: TradingSignal) -> bool:
        """Check if position would exceed concentration limits"""
        # This would check sector/symbol concentration
        # Simplified implementation
        return True
    
    async def _check_correlation_risk(self, signal: TradingSignal) -> bool:
        """Check if position would increase correlation risk"""
        # This would calculate correlation with existing positions
        # Simplified implementation
        return True
    
    def _check_daily_loss_limit(self, signal: TradingSignal) -> bool:
        """Check if position would exceed daily loss limit"""
        if not self.daily_pnl_history:
            return True
        
        current_daily_pnl = self.daily_pnl_history[-1]
        return current_daily_pnl > -self.max_daily_loss
    
    def _check_volatility_risk(self, signal: TradingSignal) -> bool:
        """Check if position volatility is acceptable"""
        # This would check historical volatility
        # Simplified implementation
        return True
    
    async def _check_market_conditions(self, signal: TradingSignal) -> bool:
        """Check if market conditions are favorable"""
        # This would check market volatility, trends, etc.
        # Simplified implementation
        return True
    
    async def _calculate_portfolio_volatility(self, positions: List[Position]) -> float:
        """Calculate portfolio volatility"""
        try:
            if not positions:
                return 0.0
            
            # Simplified calculation
            individual_volatilities = []
            for pos in positions:
                # Would calculate actual volatility from historical data
                vol = 0.2  # 20% annualized volatility as default
                individual_volatilities.append(vol)
            
            # Portfolio volatility (simplified)
            return np.mean(individual_volatilities)
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio volatility: {e}")
            return 0.0
    
    async def _calculate_portfolio_var(self, positions: List[Position]) -> float:
        """Calculate portfolio Value at Risk"""
        try:
            if not positions or not self.daily_pnl_history:
                return 0.0
            
            # Calculate VaR from historical PnL
            pnl_series = np.array(self.daily_pnl_history)
            var_value = np.percentile(pnl_series, (1 - self.var_confidence) * 100)
            
            return abs(var_value)
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio VaR: {e}")
            return 0.0
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        try:
            if not self.portfolio_value_history:
                return 0.0
            
            values = np.array(self.portfolio_value_history)
            running_max = np.maximum.accumulate(values)
            drawdown = (values - running_max) / running_max
            
            return np.min(drawdown)
            
        except Exception as e:
            self.logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        try:
            if not self.daily_pnl_history:
                return 0.0
            
            returns = np.array(self.daily_pnl_history)
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0.0
            
            # Assuming risk-free rate of 2% annually (0.02/252 daily)
            risk_free_rate = 0.02 / 252
            
            sharpe = (mean_return - risk_free_rate) / std_return
            return sharpe * np.sqrt(252)  # Annualized
            
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def _calculate_sortino_ratio(self) -> float:
        """Calculate Sortino ratio"""
        try:
            if not self.daily_pnl_history:
                return 0.0
            
            returns = np.array(self.daily_pnl_history)
            mean_return = np.mean(returns)
            
            # Downside deviation
            negative_returns = returns[returns < 0]
            if len(negative_returns) == 0:
                return float('inf')
            
            downside_deviation = np.std(negative_returns)
            
            if downside_deviation == 0:
                return 0.0
            
            # Assuming risk-free rate of 2% annually
            risk_free_rate = 0.02 / 252
            
            sortino = (mean_return - risk_free_rate) / downside_deviation
            return sortino * np.sqrt(252)  # Annualized
            
        except Exception as e:
            self.logger.error(f"Error calculating Sortino ratio: {e}")
            return 0.0
    
    def _calculate_position_concentration(self, positions: List[Position]) -> float:
        """Calculate position concentration risk"""
        try:
            if not positions:
                return 0.0
            
            # Calculate position values
            position_values = [abs(pos.quantity * pos.current_price) for pos in positions]
            total_value = sum(position_values)
            
            if total_value == 0:
                return 0.0
            
            # Calculate concentration (largest position as percentage of portfolio)
            max_position = max(position_values)
            concentration = max_position / total_value
            
            return concentration
            
        except Exception as e:
            self.logger.error(f"Error calculating position concentration: {e}")
            return 0.0
    
    def _calculate_sector_concentration(self, positions: List[Position]) -> float:
        """Calculate sector concentration risk"""
        try:
            if not positions:
                return 0.0
            
            # This would map symbols to sectors
            # Simplified implementation
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating sector concentration: {e}")
            return 0.0
    
    async def _calculate_correlation_risk(self, positions: List[Position]) -> float:
        """Calculate correlation risk"""
        try:
            if len(positions) < 2:
                return 0.0
            
            # This would calculate correlation matrix of positions
            # Simplified implementation
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation risk: {e}")
            return 0.0
    
    async def _calculate_market_beta(self, positions: List[Position]) -> float:
        """Calculate portfolio beta"""
        try:
            if not positions:
                return 0.0
            
            # This would calculate beta relative to market benchmark
            # Simplified implementation
            return 1.0
            
        except Exception as e:
            self.logger.error(f"Error calculating market beta: {e}")
            return 0.0
    
    async def _calculate_market_correlation(self, positions: List[Position]) -> float:
        """Calculate market correlation"""
        try:
            if not positions:
                return 0.0
            
            # This would calculate correlation with market benchmark
            # Simplified implementation
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating market correlation: {e}")
            return 0.0
    
    def _calculate_overall_risk_level(self, var: float, volatility: float, drawdown: float, 
                                    concentration: float, correlation: float) -> Tuple[RiskLevel, float]:
        """Calculate overall risk level and score"""
        try:
            # Risk scoring (0-100)
            risk_score = 0.0
            
            # VaR contribution (0-25)
            var_score = min(25, (var / (self.max_daily_loss * 100)) * 25)
            
            # Volatility contribution (0-25)
            vol_score = min(25, (volatility / 0.5) * 25)
            
            # Drawdown contribution (0-25)
            dd_score = min(25, (abs(drawdown) / self.max_drawdown) * 25)
            
            # Concentration contribution (0-25)
            conc_score = min(25, (concentration / 0.5) * 25)
            
            risk_score = var_score + vol_score + dd_score + conc_score
            
            # Determine risk level
            if risk_score < 25:
                risk_level = RiskLevel.LOW
            elif risk_score < 50:
                risk_level = RiskLevel.MEDIUM
            elif risk_score < 75:
                risk_level = RiskLevel.HIGH
            else:
                risk_level = RiskLevel.EXTREME
            
            return risk_level, risk_score
            
        except Exception as e:
            self.logger.error(f"Error calculating overall risk level: {e}")
            return RiskLevel.LOW, 0.0
    
    async def _generate_risk_alerts(self, risk_metrics: RiskMetrics):
        """Generate risk alerts based on metrics"""
        try:
            alerts = []
            
            # Check for high risk score
            if risk_metrics.risk_score > 75:
                alerts.append(RiskAlert(
                    alert_id=f"risk_alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    alert_type="high_risk",
                    severity="critical",
                    message=f"Portfolio risk score is {risk_metrics.risk_score:.1f}/100 (EXTREME)",
                    affected_positions=[],
                    recommended_actions=["Reduce position sizes", "Close high-risk positions", "Increase diversification"],
                    timestamp=datetime.now()
                ))
            
            # Check for high drawdown
            if risk_metrics.drawdown_utilization > 0.8:
                alerts.append(RiskAlert(
                    alert_id=f"drawdown_alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    alert_type="high_drawdown",
                    severity="warning",
                    message=f"Drawdown utilization is {risk_metrics.drawdown_utilization:.1%}",
                    affected_positions=[],
                    recommended_actions=["Review stop losses", "Consider reducing leverage"],
                    timestamp=datetime.now()
                ))
            
            # Check for high concentration
            if risk_metrics.position_concentration > 0.4:
                alerts.append(RiskAlert(
                    alert_id=f"concentration_alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    alert_type="high_concentration",
                    severity="warning",
                    message=f"Position concentration is {risk_metrics.position_concentration:.1%}",
                    affected_positions=[],
                    recommended_actions=["Diversify holdings", "Reduce largest positions"],
                    timestamp=datetime.now()
                ))
            
            # Add alerts to list
            self.risk_alerts.extend(alerts)
            
            # Limit alert history
            if len(self.risk_alerts) > 100:
                self.risk_alerts = self.risk_alerts[-100:]
            
        except Exception as e:
            self.logger.error(f"Error generating risk alerts: {e}")
    
    async def _calculate_volatility_adjustment(self, symbol: str) -> float:
        """Calculate volatility adjustment for position sizing"""
        try:
            # This would calculate based on historical volatility
            # Simplified implementation
            return 1.0
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility adjustment: {e}")
            return 1.0
    
    async def _calculate_correlation_adjustment(self, symbol: str) -> float:
        """Calculate correlation adjustment for position sizing"""
        try:
            # This would calculate based on correlation with existing positions
            # Simplified implementation
            return 1.0
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation adjustment: {e}")
            return 1.0
    
    def _calculate_kelly_sizing(self, signal: TradingSignal) -> float:
        """Calculate Kelly criterion position sizing"""
        try:
            # Kelly formula: f = (bp - q) / b
            # where b = odds, p = probability of win, q = probability of loss
            
            # Simplified implementation based on signal confidence
            win_probability = signal.confidence
            loss_probability = 1 - win_probability
            
            # Estimate odds from stop loss and take profit
            if signal.stop_loss > 0 and signal.take_profit > signal.entry_price:
                odds = (signal.take_profit - signal.entry_price) / (signal.entry_price - signal.stop_loss)
            else:
                odds = 2.0  # Default 2:1 risk-reward
            
            # Kelly fraction
            kelly_fraction = (odds * win_probability - loss_probability) / odds
            
            # Limit Kelly fraction to reasonable values
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Max 25%
            
            # Convert to position size (would need portfolio value)
            return kelly_fraction * 100000  # Placeholder
            
        except Exception as e:
            self.logger.error(f"Error calculating Kelly sizing: {e}")
            return 0.0
    
    def _initialize_stress_scenarios(self) -> List[Dict[str, Any]]:
        """Initialize stress test scenarios"""
        return [
            {
                "name": "Market Crash",
                "description": "20% market decline",
                "market_shock": -0.20,
                "volatility_shock": 2.0
            },
            {
                "name": "Volatility Spike",
                "description": "VIX spike to 40",
                "market_shock": -0.10,
                "volatility_shock": 3.0
            },
            {
                "name": "Interest Rate Shock",
                "description": "200bp rate increase",
                "market_shock": -0.15,
                "volatility_shock": 1.5
            }
        ]
    
    async def _load_historical_data(self):
        """Load historical risk data"""
        # This would load from database
        pass
    
    async def _initialize_risk_monitoring(self):
        """Initialize risk monitoring"""
        # This would set up monitoring tasks
        pass
