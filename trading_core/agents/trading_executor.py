"""Streamlined trading execution agent"""

import random
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

from .base_agent import BaseAgent
from ..data_models import TradeOrder, Position, MarketData, OrderStatus

class TradingExecutorAgent(BaseAgent):
    """Streamlined trading execution agent"""

    def __init__(self):
        super().__init__("TradingExecutor")
        self.positions = {}
        self.executed_orders = []
        self.cash_balance = 100000.0
        self.order_history = []

    def process(self, order: TradeOrder) -> bool:
        """Execute a trade order"""
        try:
            # Simulate market price (in real system, get from market data)
            market_price = random.uniform(95, 105)

            success = self._execute_trade(order, market_price)

            if success:
                order.status = OrderStatus.FILLED
                order.price = market_price
                order.avg_fill_price = market_price
                order.filled_quantity = order.quantity
                self.executed_orders.append(order)
                self.order_history.append(order)

                # Update position with stop-loss info
                if order.symbol in self.positions:
                    position = self.positions[order.symbol]
                    if order.stop_loss_price:
                        position.stop_loss_price = order.stop_loss_price
                    if order.take_profit_price:
                        position.take_profit_price = order.take_profit_price

                self.logger.info(f"Order executed: {order.order_id} at ${market_price:.2f}")
            else:
                order.status = OrderStatus.REJECTED
                self.logger.warning(f"Order rejected: {order.order_id}")

            return success

        except Exception as e:
            self.logger.error(f"Error executing order: {e}")
            return False

    def _execute_trade(self, order: TradeOrder, price: float) -> bool:
        """Execute the actual trade"""
        trade_value = order.quantity * price

        if order.action == 'BUY':
            if self.cash_balance >= trade_value:
                self.cash_balance -= trade_value
                self._update_position(order.symbol, order.quantity, price, order)
                return True
            else:
                self.logger.warning(f"Insufficient cash for order {order.order_id}")
                return False

        elif order.action == 'SELL':
            if order.symbol in self.positions and self.positions[order.symbol].quantity >= order.quantity:
                self.cash_balance += trade_value
                self._update_position(order.symbol, -order.quantity, price, order)
                return True
            else:
                self.logger.warning(f"Insufficient position for sell order {order.order_id}")
                return False

        return False

    def _update_position(self, symbol: str, quantity: int, price: float, order: TradeOrder):
        """Update position with new trade"""
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=0,
                avg_price=0.0,
                current_price=price,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                entry_timestamp=datetime.now()
            )

        position = self.positions[symbol]

        if quantity > 0:  # Buy
            total_cost = (position.quantity * position.avg_price) + (quantity * price)
            position.quantity += quantity
            position.avg_price = total_cost / position.quantity if position.quantity > 0 else 0

            # Set stop-loss and take-profit from order
            if order.stop_loss_price:
                position.stop_loss_price = order.stop_loss_price
            if order.take_profit_price:
                position.take_profit_price = order.take_profit_price

        else:  # Sell
            # Calculate realized P&L
            realized_pnl = abs(quantity) * (price - position.avg_price)
            position.realized_pnl += realized_pnl
            position.quantity += quantity  # quantity is negative for sells

            # Clear stop-loss if position is closed
            if position.quantity == 0:
                position.stop_loss_price = None
                position.take_profit_price = None

        position.current_price = price
        position.unrealized_pnl = (price - position.avg_price) * position.quantity

        # Update max price for trailing stops
        if position.quantity > 0:
            position.max_price_since_entry = max(position.max_price_since_entry, price)

    def update_positions(self, market_data: MarketData):
        """Update position prices and P&L"""
        if market_data.symbol in self.positions:
            position = self.positions[market_data.symbol]
            position.current_price = market_data.price
            position.unrealized_pnl = (market_data.price - position.avg_price) * position.quantity

            # Update max price for trailing stops
            if position.quantity > 0:
                position.max_price_since_entry = max(position.max_price_since_entry, market_data.price)

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        total_portfolio_value = self.cash_balance
        total_unrealized_pnl = 0.0
        total_realized_pnl = 0.0
        active_positions = []
        stop_loss_coverage = 0

        for position in self.positions.values():
            if position.quantity != 0:
                position_value = position.quantity * position.current_price
                total_portfolio_value += position_value
                total_unrealized_pnl += position.unrealized_pnl
                total_realized_pnl += position.realized_pnl

                if position.quantity > 0:
                    active_positions.append(position)
                    if position.stop_loss_price:
                        stop_loss_coverage += 1

        # Performance metrics
        total_return = ((total_portfolio_value - 100000) / 100000) * 100
        profitable_trades = len([o for o in self.executed_orders if self._is_profitable_trade(o)])
        total_trades = len(self.executed_orders)
        win_rate = (profitable_trades / max(total_trades, 1)) * 100

        return {
            'cash_balance': self.cash_balance,
            'total_portfolio_value': total_portfolio_value,
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_realized_pnl': total_realized_pnl,
            'total_return_pct': total_return,
            'positions': dict(self.positions),
            'active_positions_count': len(active_positions),
            'executed_orders_count': len(self.executed_orders),
            'win_rate': win_rate,
            'cash_allocation_pct': (self.cash_balance / total_portfolio_value) * 100,
            'stop_loss_coverage': stop_loss_coverage,
            'stop_loss_coverage_pct': (stop_loss_coverage / max(len(active_positions), 1)) * 100,
            'total_trades': total_trades
        }

    def _is_profitable_trade(self, order: TradeOrder) -> bool:
        """Check if a trade was profitable"""
        if order.symbol in self.positions:
            position = self.positions[order.symbol]
            return (position.unrealized_pnl + position.realized_pnl) > 0
        return False