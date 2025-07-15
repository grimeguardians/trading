"""Streamlined risk management agent"""

import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any

from .base_agent import BaseAgent
from ..data_models import TradingSignal, TradeOrder, Position, OrderType, RiskAlert
from ..utils.stop_loss_manager import StopLossManager

class RiskManagerAgent(BaseAgent):
    """Streamlined risk management agent"""

    def __init__(self):
        super().__init__("RiskManager")
        self.portfolio_value = 100000.0
        self.max_position_size = 0.1  # 10% max per position
        self.max_portfolio_risk = 0.02  # 2% max portfolio risk
        self.position_limits = 10
        self.stop_loss_manager = StopLossManager()
        self.risk_metrics = {
            'max_drawdown': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'var_95': 0.0
        }
        self.portfolio_history = []
        self.risk_alerts = []

    def process(self, signal: TradingSignal, positions: Dict[str, Position]) -> Optional[TradeOrder]:
        """Process trading signal and generate order if risk allows"""
        try:
            # Update risk metrics
            self._update_risk_metrics(positions)

            # Validate signal against risk parameters
            if not self._validate_signal(signal, positions):
                return None

            # Calculate position size
            position_size = self._calculate_position_size(signal, positions)
            if position_size <= 0:
                return None

            # Create order with risk management
            order_id = f"ORDER_{signal.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            order = TradeOrder(
                order_id=order_id,
                symbol=signal.symbol,
                action=signal.action,
                quantity=position_size,
                price=100.0,  # Simplified - would use real market price
                order_type=OrderType.MARKET,
                timestamp=datetime.now(),
                stop_loss_price=signal.stop_loss_price,
                take_profit_price=signal.take_profit_price
            )

            self.logger.info(f"Risk-approved order created: {order.order_id}")
            return order

        except Exception as e:
            self.logger.error(f"Error in risk processing: {e}")
            return None

    def process_market_update(self, market_data) -> List[TradeOrder]:
        """Process market updates for stop-loss triggers"""
        try:
            # Check for stop-loss triggers
            triggered_orders = []
            triggered_orders.extend(self.stop_loss_manager.check_stop_triggers(market_data))
            triggered_orders.extend(self.stop_loss_manager.update_trailing_stops(market_data))

            if triggered_orders:
                self.logger.info(f"Stop-loss triggered {len(triggered_orders)} orders for {market_data.symbol}")

            return triggered_orders

        except Exception as e:
            self.logger.error(f"Error processing market update: {e}")
            return []

    def create_stop_loss_orders(self, position: Position) -> List[TradeOrder]:
        """Create stop-loss orders for a position"""
        orders = []
        try:
            if position.stop_loss_price:
                stop_order = self.stop_loss_manager.create_stop_loss_order(position, position.stop_loss_price)
                orders.append(stop_order)

            if position.trailing_stop_price:
                trail_distance = abs(position.current_price - position.trailing_stop_price)
                self.stop_loss_manager.create_trailing_stop(position, trail_distance)

        except Exception as e:
            self.logger.error(f"Error creating stop-loss orders: {e}")

        return orders

    def _validate_signal(self, signal: TradingSignal, positions: Dict[str, Position]) -> bool:
        """Validate signal against risk parameters"""
        # Check confidence threshold
        if signal.confidence < 0.3:
            return False

        # Check maximum positions
        active_positions = len([p for p in positions.values() if p.quantity > 0])
        if active_positions >= self.position_limits and signal.action == 'BUY':
            return False

        # Check drawdown limit
        if self.risk_metrics['max_drawdown'] > 0.2:  # 20% max drawdown
            return False

        # Require stop-loss for new positions
        if signal.action == 'BUY' and not signal.stop_loss_price:
            self.logger.warning(f"No stop-loss provided for {signal.symbol}")
            return False

        return True

    def _calculate_position_size(self, signal: TradingSignal, positions: Dict[str, Position]) -> int:
        """Calculate appropriate position size"""
        # Base position sizing
        base_position_value = self.portfolio_value * self.max_position_size

        # Risk adjustment based on confidence
        confidence_adjustment = 0.5 + (signal.confidence * 0.5)

        # Stop-loss distance adjustment
        if signal.stop_loss_price:
            estimated_price = 100.0  # Simplified
            stop_distance = abs(estimated_price - signal.stop_loss_price) / estimated_price
            risk_adjustment = max(0.3, 1 - stop_distance * 5)
        else:
            risk_adjustment = 0.5

        # Portfolio heat adjustment
        active_positions = len([p for p in positions.values() if p.quantity > 0])
        heat_adjustment = max(0.6, 1 - (active_positions / self.position_limits) * 0.4)

        # Calculate final size
        adjusted_value = (base_position_value * confidence_adjustment * 
                         risk_adjustment * heat_adjustment)

        estimated_price = 100.0  # Simplified
        return max(int(adjusted_value / estimated_price), 1)

    def _update_risk_metrics(self, positions: Dict[str, Position]):
        """Update risk metrics"""
        try:
            # Calculate current portfolio value
            current_value = self.portfolio_value
            for pos in positions.values():
                if pos.quantity != 0:
                    current_value += pos.unrealized_pnl

            # Track portfolio history
            self.portfolio_history.append(current_value)
            if len(self.portfolio_history) > 200:
                self.portfolio_history = self.portfolio_history[-200:]

            # Calculate risk metrics
            if len(self.portfolio_history) >= 10:
                returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
                
                self.risk_metrics['volatility'] = np.std(returns) * np.sqrt(252)
                self.risk_metrics['var_95'] = np.percentile(returns, 5) * current_value

                # Drawdown calculation
                peak = np.maximum.accumulate(self.portfolio_history)
                drawdown = (self.portfolio_history - peak) / peak
                self.risk_metrics['max_drawdown'] = abs(np.min(drawdown))

                # Sharpe ratio
                risk_free_rate = 0.02 / 252
                excess_returns = returns - risk_free_rate
                if np.std(excess_returns) > 0:
                    self.risk_metrics['sharpe_ratio'] = (np.mean(excess_returns) / 
                                                        np.std(excess_returns) * np.sqrt(252))

        except Exception as e:
            self.logger.error(f"Error updating risk metrics: {e}")

    def get_risk_report(self) -> Dict[str, Any]:
        """Get current risk report"""
        return {
            'risk_metrics': self.risk_metrics.copy(),
            'portfolio_value': self.portfolio_value,
            'max_position_size': self.max_position_size,
            'position_limits': self.position_limits,
            'active_stops': len(self.stop_loss_manager.active_stops),
            'trailing_stops': len(self.stop_loss_manager.trailing_stops),
            'risk_alerts': len(self.risk_alerts)
        }