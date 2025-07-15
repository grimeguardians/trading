"""Stop-loss management utilities"""

from datetime import datetime
from typing import Dict, List

from ..data_models import TradeOrder, Position, OrderType, MarketData

class StopLossManager:
    """Manages stop-loss and trailing stop orders"""

    def __init__(self):
        self.active_stops = {}  # order_id -> stop_loss_config
        self.trailing_stops = {}  # position_symbol -> trailing_config

    def create_stop_loss_order(self, position: Position, stop_loss_price: float, 
                              order_type: OrderType = OrderType.STOP_LOSS) -> TradeOrder:
        """Create a stop-loss order for a position"""
        order_id = f"SL_{position.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Determine action (opposite of position)
        action = 'SELL' if position.quantity > 0 else 'BUY'

        stop_order = TradeOrder(
            order_id=order_id,
            symbol=position.symbol,
            action=action,
            quantity=abs(position.quantity),
            price=stop_loss_price,
            order_type=order_type,
            timestamp=datetime.now(),
            stop_loss_price=stop_loss_price
        )

        self.active_stops[order_id] = {
            'order': stop_order,
            'position_symbol': position.symbol,
            'trigger_price': stop_loss_price
        }

        return stop_order

    def create_trailing_stop(self, position: Position, trail_distance: float) -> None:
        """Create a trailing stop for a position"""
        if position.quantity > 0:  # Long position
            initial_stop = position.current_price - trail_distance
        else:  # Short position
            initial_stop = position.current_price + trail_distance

        self.trailing_stops[position.symbol] = {
            'trail_distance': trail_distance,
            'current_stop_price': initial_stop,
            'highest_price': position.current_price if position.quantity > 0 else None,
            'lowest_price': position.current_price if position.quantity < 0 else None,
            'position_quantity': position.quantity
        }

    def check_stop_triggers(self, market_data: MarketData) -> List[TradeOrder]:
        """Check if any stop-loss orders should be triggered"""
        triggered_orders = []
        orders_to_remove = []

        for order_id, stop_config in self.active_stops.items():
            if stop_config['position_symbol'] == market_data.symbol:
                stop_order = stop_config['order']
                trigger_price = stop_config['trigger_price']

                should_trigger = False

                if stop_order.action == 'SELL' and market_data.price <= trigger_price:
                    should_trigger = True
                elif stop_order.action == 'BUY' and market_data.price >= trigger_price:
                    should_trigger = True

                if should_trigger:
                    # Convert to market order
                    market_order = TradeOrder(
                        order_id=f"MKT_{order_id}",
                        symbol=stop_order.symbol,
                        action=stop_order.action,
                        quantity=stop_order.quantity,
                        price=market_data.price,
                        order_type=OrderType.MARKET,
                        timestamp=datetime.now(),
                        parent_order_id=order_id
                    )
                    triggered_orders.append(market_order)
                    orders_to_remove.append(order_id)

        # Remove triggered stops
        for order_id in orders_to_remove:
            del self.active_stops[order_id]

        return triggered_orders

    def update_trailing_stops(self, market_data: MarketData) -> List[TradeOrder]:
        """Update trailing stops based on new market data"""
        triggered_orders = []

        if market_data.symbol in self.trailing_stops:
            trail_config = self.trailing_stops[market_data.symbol]
            current_price = market_data.price

            if trail_config['position_quantity'] > 0:  # Long position
                # Update highest price
                if current_price > trail_config['highest_price']:
                    trail_config['highest_price'] = current_price
                    # Update stop price
                    new_stop = current_price - trail_config['trail_distance']
                    if new_stop > trail_config['current_stop_price']:
                        trail_config['current_stop_price'] = new_stop

                # Check if stop is triggered
                if current_price <= trail_config['current_stop_price']:
                    order = self._create_triggered_order(market_data.symbol, trail_config, current_price)
                    triggered_orders.append(order)
                    del self.trailing_stops[market_data.symbol]

            else:  # Short position
                # Update lowest price
                if current_price < trail_config['lowest_price']:
                    trail_config['lowest_price'] = current_price
                    # Update stop price
                    new_stop = current_price + trail_config['trail_distance']
                    if new_stop < trail_config['current_stop_price']:
                        trail_config['current_stop_price'] = new_stop

                # Check if stop is triggered
                if current_price >= trail_config['current_stop_price']:
                    order = self._create_triggered_order(market_data.symbol, trail_config, current_price)
                    triggered_orders.append(order)
                    del self.trailing_stops[market_data.symbol]

        return triggered_orders

    def _create_triggered_order(self, symbol: str, trail_config: Dict, current_price: float) -> TradeOrder:
        """Create order when trailing stop is triggered"""
        action = 'SELL' if trail_config['position_quantity'] > 0 else 'BUY'

        return TradeOrder(
            order_id=f"TS_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            symbol=symbol,
            action=action,
            quantity=abs(trail_config['position_quantity']),
            price=current_price,
            order_type=OrderType.MARKET,
            timestamp=datetime.now()
        )