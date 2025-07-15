"""
Live Trading Executor with Alpaca Integration
Executes real trades through Alpaca paper trading account
"""

import logging
from datetime import datetime
from typing import Dict, List, Any

from .base_agent import BaseAgent
from ..data_models import TradeOrder, Position, MarketData, OrderStatus
from ..brokers.alpaca_broker import AlpacaBroker

class LiveTradingExecutorAgent(BaseAgent):
    """Live trading execution agent with real Alpaca integration"""

    def __init__(self, use_live_broker: bool = True):
        super().__init__("LiveTradingExecutor")
        
        # Initialize broker
        self.use_live_broker = use_live_broker
        self.alpaca_broker = AlpacaBroker() if use_live_broker else None
        
        # Local tracking (for simulation fallback)
        self.local_positions = {}
        self.local_cash = 100000.0
        self.executed_orders = []
        self.order_history = []
        
        # Performance tracking
        self.trade_count = 0
        self.successful_trades = 0
        
        self.logger.info(f"Live Trading Executor initialized (Live Broker: {use_live_broker})")

    def process(self, order: TradeOrder) -> bool:
        """Execute a trade order through Alpaca or simulation"""
        try:
            self.trade_count += 1
            
            if self.use_live_broker and self.alpaca_broker and self.alpaca_broker.is_connected:
                # Execute through Alpaca
                success = self._execute_live_order(order)
            else:
                # Fallback to simulation
                success = self._execute_simulated_order(order)
            
            if success:
                self.successful_trades += 1
                self.executed_orders.append(order)
                self.order_history.append(order)
                
                # Log execution details
                mode = "LIVE" if (self.use_live_broker and self.alpaca_broker and self.alpaca_broker.is_connected) else "SIMULATED"
                self.logger.info(f"✅ {mode} ORDER: {order.action} {order.quantity} {order.symbol} @ ${order.price:.2f}")
                
                if order.stop_loss_price:
                    self.logger.info(f"   Stop Loss: ${order.stop_loss_price:.2f}")
                if order.take_profit_price:
                    self.logger.info(f"   Take Profit: ${order.take_profit_price:.2f}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error executing order: {e}")
            order.status = OrderStatus.REJECTED
            return False

    def _execute_live_order(self, order: TradeOrder) -> bool:
        """Execute order through live Alpaca broker"""
        try:
            # Place order through Alpaca
            success = self.alpaca_broker.place_order(order)
            
            if success:
                order.status = OrderStatus.FILLED
                order.avg_fill_price = order.price  # Market orders typically fill at market price
                order.filled_quantity = order.quantity
                
                self.logger.info(f"🟢 LIVE TRADE EXECUTED: {order.symbol}")
                
                # Get updated account info
                account_info = self.alpaca_broker.get_account_info()
                if account_info:
                    self.logger.info(f"💰 Portfolio Value: ${account_info.get('portfolio_value', 0):,.2f}")
                    self.logger.info(f"💵 Buying Power: ${account_info.get('buying_power', 0):,.2f}")
                
                return True
            else:
                order.status = OrderStatus.REJECTED
                self.logger.warning(f"🔴 LIVE TRADE REJECTED: {order.symbol}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in live order execution: {e}")
            order.status = OrderStatus.REJECTED
            return False

    def _execute_simulated_order(self, order: TradeOrder) -> bool:
        """Execute order in simulation mode"""
        try:
            trade_value = order.quantity * order.price
            
            if order.action == 'BUY':
                if self.local_cash >= trade_value:
                    self.local_cash -= trade_value
                    self._update_local_position(order.symbol, order.quantity, order.price, order)
                    order.status = OrderStatus.FILLED
                    return True
                else:
                    self.logger.warning(f"Insufficient cash for simulated order: ${trade_value:.2f} > ${self.local_cash:.2f}")
                    return False
                    
            elif order.action == 'SELL':
                if order.symbol in self.local_positions and self.local_positions[order.symbol].quantity >= order.quantity:
                    self.local_cash += trade_value
                    self._update_local_position(order.symbol, -order.quantity, order.price, order)
                    order.status = OrderStatus.FILLED
                    return True
                else:
                    self.logger.warning(f"Insufficient position for simulated sell: {order.quantity} > {self.local_positions.get(order.symbol, Position('', 0, 0, 0, 0, 0)).quantity}")
                    return False
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in simulated order execution: {e}")
            return False

    def _update_local_position(self, symbol: str, quantity: int, price: float, order: TradeOrder):
        """Update local position tracking"""
        if symbol not in self.local_positions:
            self.local_positions[symbol] = Position(
                symbol=symbol,
                quantity=0,
                avg_price=0.0,
                current_price=price,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                entry_timestamp=datetime.now()
            )

        position = self.local_positions[symbol]

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

    def get_positions(self) -> Dict[str, Position]:
        """Get current positions from Alpaca or local simulation"""
        if self.use_live_broker and self.alpaca_broker and self.alpaca_broker.is_connected:
            # Get live positions from Alpaca
            try:
                live_positions = self.alpaca_broker.get_positions()
                self.logger.debug(f"Retrieved {len(live_positions)} live positions from Alpaca")
                return live_positions
            except Exception as e:
                self.logger.error(f"Error fetching live positions: {e}")
                return self.local_positions
        else:
            # Return simulated positions
            return self.local_positions

    def update_positions(self, market_data: MarketData):
        """Update position prices and P&L"""
        if self.use_live_broker and self.alpaca_broker and self.alpaca_broker.is_connected:
            # Live positions are updated by Alpaca automatically
            # We just need to refresh our local cache if needed
            pass
        else:
            # Update simulated positions
            if market_data.symbol in self.local_positions:
                position = self.local_positions[market_data.symbol]
                position.current_price = market_data.price
                position.unrealized_pnl = (market_data.price - position.avg_price) * position.quantity

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        if self.use_live_broker and self.alpaca_broker and self.alpaca_broker.is_connected:
            return self._get_live_portfolio_summary()
        else:
            return self._get_simulated_portfolio_summary()

    def _get_live_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary from live Alpaca account"""
        try:
            # Get account info
            account_info = self.alpaca_broker.get_account_info()
            positions = self.get_positions()
            
            # Calculate additional metrics
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in positions.values())
            active_positions = [pos for pos in positions.values() if pos.quantity > 0]
            
            # Calculate total return
            portfolio_value = account_info.get('portfolio_value', 100000)
            initial_value = 100000  # Assuming starting value
            total_return = ((portfolio_value - initial_value) / initial_value) * 100
            
            return {
                'cash_balance': account_info.get('cash', 0),
                'total_portfolio_value': portfolio_value,
                'buying_power': account_info.get('buying_power', 0),
                'total_unrealized_pnl': total_unrealized_pnl,
                'total_realized_pnl': 0.0,  # Would need to calculate from order history
                'total_return_pct': total_return,
                'positions': dict(positions),
                'active_positions_count': len(active_positions),
                'executed_orders_count': len(self.executed_orders),
                'trade_success_rate': (self.successful_trades / max(self.trade_count, 1)) * 100,
                'broker_status': 'LIVE_ALPACA',
                'day_trade_count': account_info.get('day_trade_count', 0),
                'pattern_day_trader': account_info.get('pattern_day_trader', False),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting live portfolio summary: {e}")
            return self._get_simulated_portfolio_summary()

    def _get_simulated_portfolio_summary(self) -> Dict[str, Any]:
        """Get simulated portfolio summary"""
        positions = self.local_positions
        total_portfolio_value = self.local_cash
        total_unrealized_pnl = 0.0
        active_positions = []

        for position in positions.values():
            if position.quantity != 0:
                position_value = position.quantity * position.current_price
                total_portfolio_value += position_value
                total_unrealized_pnl += position.unrealized_pnl

                if position.quantity > 0:
                    active_positions.append(position)

        # Calculate total return
        initial_value = 100000
        total_return = ((total_portfolio_value - initial_value) / initial_value) * 100

        return {
            'cash_balance': self.local_cash,
            'total_portfolio_value': total_portfolio_value,
            'buying_power': self.local_cash,
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_realized_pnl': sum(pos.realized_pnl for pos in positions.values()),
            'total_return_pct': total_return,
            'positions': dict(positions),
            'active_positions_count': len(active_positions),
            'executed_orders_count': len(self.executed_orders),
            'trade_success_rate': (self.successful_trades / max(self.trade_count, 1)) * 100,
            'broker_status': 'SIMULATION',
            'last_updated': datetime.now().isoformat()
        }

    def get_broker_status(self) -> Dict[str, Any]:
        """Get broker connection and status information"""
        if self.alpaca_broker:
            alpaca_status = self.alpaca_broker.get_connection_status()
            alpaca_status['trade_count'] = self.trade_count
            alpaca_status['success_rate'] = (self.successful_trades / max(self.trade_count, 1)) * 100
            return alpaca_status
        else:
            return {
                'broker': 'Simulation',
                'connected': True,
                'paper_trading': True,
                'trade_count': self.trade_count,
                'success_rate': (self.successful_trades / max(self.trade_count, 1)) * 100
            }

    def get_order_history(self) -> List[Dict[str, Any]]:
        """Get order history from Alpaca or local tracking"""
        if self.use_live_broker and self.alpaca_broker and self.alpaca_broker.is_connected:
            try:
                return self.alpaca_broker.get_order_history()
            except Exception as e:
                self.logger.error(f"Error fetching order history: {e}")
        
        # Return local order history
        return [
            {
                'id': order.order_id,
                'symbol': order.symbol,
                'side': order.action,
                'quantity': order.quantity,
                'status': order.status.value,
                'filled_qty': order.filled_quantity,
                'avg_price': order.avg_fill_price,
                'created_at': order.timestamp.isoformat()
            }
            for order in self.order_history[-10:]  # Last 10 orders
        ]