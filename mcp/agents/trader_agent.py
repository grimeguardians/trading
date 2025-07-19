"""
Trader Agent for MCP System
Handles trade execution, order management, and position tracking
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import websockets
import uuid
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

class OrderType(str, Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class OrderSide(str, Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(str, Enum):
    """Order status"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class Order:
    """Trading order"""
    order_id: str
    symbol: str
    exchange: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Position:
    """Trading position"""
    symbol: str
    exchange: str
    quantity: float
    average_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    market_value: float
    side: str  # "long" or "short"
    opened_at: datetime
    updated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class TradeExecution:
    """Trade execution result"""
    execution_id: str
    order_id: str
    symbol: str
    exchange: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    commission: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class TraderAgent:
    """Trader Agent for order execution and position management"""
    
    def __init__(self, mcp_server_url: str = "ws://localhost:9000"):
        self.agent_id = "trader_agent_001"
        self.agent_type = "trader"
        self.name = "Trader Agent"
        self.description = "Advanced trading execution and position management"
        self.capabilities = [
            "order_execution",
            "position_management",
            "risk_monitoring",
            "portfolio_tracking",
            "order_routing",
            "execution_optimization"
        ]
        
        self.mcp_server_url = mcp_server_url
        self.websocket = None
        self.running = False
        
        # Trading state
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.executions: List[TradeExecution] = []
        
        # Trading parameters
        self.max_position_size = 1000000  # $1M max position
        self.max_daily_loss = 50000  # $50K max daily loss
        self.max_orders_per_minute = 60
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.win_rate = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        
        # Risk management
        self.position_limits = {}
        self.risk_checks_enabled = True
        self.emergency_stop_enabled = False
        
        # Order routing
        self.exchange_preferences = {
            "stocks": "alpaca",
            "crypto": "binance",
            "options": "td_ameritrade"
        }
        
        # Setup logging
        self.logger = logging.getLogger(f"Trader_{self.agent_id}")
        
        # Performance metrics
        self.metrics = {
            "orders_executed": 0,
            "execution_time": 0.0,
            "fill_rate": 0.0,
            "slippage": 0.0,
            "commission_paid": 0.0
        }
    
    async def start(self):
        """Start the trader agent"""
        try:
            self.logger.info("Starting Trader Agent...")
            
            # Connect to MCP server
            self.websocket = await websockets.connect(self.mcp_server_url)
            
            # Initialize with server
            await self._send_initialize_message()
            
            self.running = True
            
            # Start background tasks
            asyncio.create_task(self._listen_for_messages())
            asyncio.create_task(self._monitor_positions())
            asyncio.create_task(self._process_pending_orders())
            asyncio.create_task(self._send_heartbeat())
            
            self.logger.info("Trader Agent started successfully")
            
        except Exception as e:
            self.logger.error(f"Error starting Trader Agent: {e}")
            raise
    
    async def stop(self):
        """Stop the trader agent"""
        self.logger.info("Stopping Trader Agent...")
        
        self.running = False
        
        # Cancel all pending orders
        await self._cancel_all_orders()
        
        if self.websocket:
            await self.websocket.close()
        
        self.logger.info("Trader Agent stopped")
    
    async def _send_initialize_message(self):
        """Send initialization message to MCP server"""
        message = {
            "message_id": str(uuid.uuid4()),
            "message_type": "initialize",
            "sender": self.agent_id,
            "payload": {
                "agent_type": self.agent_type,
                "name": self.name,
                "description": self.description,
                "capabilities": self.capabilities
            }
        }
        
        await self.websocket.send(json.dumps(message))
    
    async def _listen_for_messages(self):
        """Listen for messages from MCP server"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                await self._handle_message(data)
        except websockets.exceptions.ConnectionClosed:
            self.logger.info("Connection to MCP server closed")
            self.running = False
        except Exception as e:
            self.logger.error(f"Error listening for messages: {e}")
    
    async def _handle_message(self, data: Dict[str, Any]):
        """Handle incoming messages"""
        message_type = data.get("message_type")
        
        if message_type == "ready":
            self.logger.info("Received ready message from server")
        elif message_type == "request":
            await self._handle_request(data)
        elif message_type == "notification":
            await self._handle_notification(data)
        elif message_type == "error":
            self.logger.error(f"Error from server: {data.get('payload', {}).get('error')}")
        else:
            self.logger.warning(f"Unknown message type: {message_type}")
    
    async def _handle_request(self, data: Dict[str, Any]):
        """Handle request messages"""
        request_type = data.get("payload", {}).get("request_type")
        
        if request_type == "execute_trade":
            await self._handle_execute_trade_request(data)
        elif request_type == "cancel_order":
            await self._handle_cancel_order_request(data)
        elif request_type == "get_positions":
            await self._handle_get_positions_request(data)
        elif request_type == "get_orders":
            await self._handle_get_orders_request(data)
        elif request_type == "modify_order":
            await self._handle_modify_order_request(data)
        else:
            await self._send_error_response(data, f"Unknown request type: {request_type}")
    
    async def _handle_notification(self, data: Dict[str, Any]):
        """Handle notification messages"""
        notification_type = data.get("payload", {}).get("notification_type")
        
        if notification_type == "market_update":
            await self._handle_market_update(data)
        elif notification_type == "risk_alert":
            await self._handle_risk_alert(data)
        elif notification_type == "position_update":
            await self._handle_position_update(data)
    
    async def _handle_execute_trade_request(self, data: Dict[str, Any]):
        """Handle trade execution request"""
        payload = data.get("payload", {})
        
        try:
            # Extract trade parameters
            symbol = payload.get("symbol")
            exchange = payload.get("exchange")
            side = OrderSide(payload.get("side"))
            order_type = OrderType(payload.get("order_type", "market"))
            quantity = float(payload.get("quantity", 0))
            price = payload.get("price")
            stop_price = payload.get("stop_price")
            
            # Validate trade parameters
            if not symbol or not exchange or quantity <= 0:
                await self._send_error_response(data, "Invalid trade parameters")
                return
            
            # Risk checks
            if self.risk_checks_enabled:
                risk_check = await self._perform_risk_check(symbol, exchange, side, quantity, price)
                if not risk_check["approved"]:
                    await self._send_error_response(data, f"Risk check failed: {risk_check['reason']}")
                    return
            
            # Create order
            order = Order(
                order_id=str(uuid.uuid4()),
                symbol=symbol,
                exchange=exchange,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                metadata=payload.get("metadata", {})
            )
            
            # Execute order
            execution_result = await self._execute_order(order)
            
            # Send response
            response = {
                "message_id": str(uuid.uuid4()),
                "message_type": "response",
                "sender": self.agent_id,
                "recipient": data["sender"],
                "payload": {
                    "request_type": "execute_trade",
                    "order_id": order.order_id,
                    "execution_result": execution_result
                },
                "correlation_id": data["message_id"]
            }
            
            await self.websocket.send(json.dumps(response))
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            await self._send_error_response(data, str(e))
    
    async def _execute_order(self, order: Order) -> Dict[str, Any]:
        """Execute a trading order"""
        try:
            self.orders[order.order_id] = order
            
            # Route to appropriate exchange
            exchange_client = await self._get_exchange_client(order.exchange)
            
            # Execute the order (simulated for now)
            if order.order_type == OrderType.MARKET:
                execution = await self._execute_market_order(order)
            elif order.order_type == OrderType.LIMIT:
                execution = await self._execute_limit_order(order)
            elif order.order_type == OrderType.STOP:
                execution = await self._execute_stop_order(order)
            else:
                raise ValueError(f"Unsupported order type: {order.order_type}")
            
            # Update order status
            order.status = OrderStatus.FILLED
            order.filled_quantity = execution.quantity
            order.average_price = execution.price
            order.updated_at = datetime.utcnow()
            
            # Update positions
            await self._update_position(execution)
            
            # Record execution
            self.executions.append(execution)
            
            # Update metrics
            self.metrics["orders_executed"] += 1
            self.metrics["commission_paid"] += execution.commission
            
            # Send notification
            await self._send_execution_notification(execution)
            
            return {
                "status": "filled",
                "execution_id": execution.execution_id,
                "quantity": execution.quantity,
                "price": execution.price,
                "commission": execution.commission,
                "timestamp": execution.timestamp.isoformat()
            }
            
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.updated_at = datetime.utcnow()
            self.logger.error(f"Order execution failed: {e}")
            raise
    
    async def _execute_market_order(self, order: Order) -> TradeExecution:
        """Execute a market order"""
        # Simulate market execution
        current_price = await self._get_current_price(order.symbol, order.exchange)
        
        # Apply slippage
        slippage = np.random.uniform(0.001, 0.005)  # 0.1% to 0.5% slippage
        if order.side == OrderSide.BUY:
            execution_price = current_price * (1 + slippage)
        else:
            execution_price = current_price * (1 - slippage)
        
        # Calculate commission
        commission = self._calculate_commission(order.exchange, order.quantity, execution_price)
        
        return TradeExecution(
            execution_id=str(uuid.uuid4()),
            order_id=order.order_id,
            symbol=order.symbol,
            exchange=order.exchange,
            side=order.side,
            quantity=order.quantity,
            price=execution_price,
            timestamp=datetime.utcnow(),
            commission=commission
        )
    
    async def _execute_limit_order(self, order: Order) -> TradeExecution:
        """Execute a limit order"""
        # Simulate limit order execution
        current_price = await self._get_current_price(order.symbol, order.exchange)
        
        # Check if limit price is executable
        if order.side == OrderSide.BUY and current_price <= order.price:
            execution_price = min(order.price, current_price)
        elif order.side == OrderSide.SELL and current_price >= order.price:
            execution_price = max(order.price, current_price)
        else:
            # Order not executable at current price
            order.status = OrderStatus.PENDING
            raise ValueError("Limit order not executable at current price")
        
        commission = self._calculate_commission(order.exchange, order.quantity, execution_price)
        
        return TradeExecution(
            execution_id=str(uuid.uuid4()),
            order_id=order.order_id,
            symbol=order.symbol,
            exchange=order.exchange,
            side=order.side,
            quantity=order.quantity,
            price=execution_price,
            timestamp=datetime.utcnow(),
            commission=commission
        )
    
    async def _execute_stop_order(self, order: Order) -> TradeExecution:
        """Execute a stop order"""
        # Simulate stop order execution
        current_price = await self._get_current_price(order.symbol, order.exchange)
        
        # Check if stop price is triggered
        if order.side == OrderSide.BUY and current_price >= order.stop_price:
            execution_price = current_price
        elif order.side == OrderSide.SELL and current_price <= order.stop_price:
            execution_price = current_price
        else:
            # Stop not triggered
            order.status = OrderStatus.PENDING
            raise ValueError("Stop order not triggered")
        
        commission = self._calculate_commission(order.exchange, order.quantity, execution_price)
        
        return TradeExecution(
            execution_id=str(uuid.uuid4()),
            order_id=order.order_id,
            symbol=order.symbol,
            exchange=order.exchange,
            side=order.side,
            quantity=order.quantity,
            price=execution_price,
            timestamp=datetime.utcnow(),
            commission=commission
        )
    
    async def _get_current_price(self, symbol: str, exchange: str) -> float:
        """Get current price for a symbol"""
        # Request current price from market data
        request_message = {
            "message_id": str(uuid.uuid4()),
            "message_type": "request",
            "sender": self.agent_id,
            "payload": {
                "request_type": "market_data",
                "symbol": symbol,
                "exchange": exchange
            }
        }
        
        await self.websocket.send(json.dumps(request_message))
        
        # For simulation, return a random price
        return np.random.uniform(100, 200)
    
    def _calculate_commission(self, exchange: str, quantity: float, price: float) -> float:
        """Calculate commission for a trade"""
        commission_rates = {
            "alpaca": 0.0,  # Commission-free
            "binance": 0.001,  # 0.1%
            "kucoin": 0.001,  # 0.1%
            "td_ameritrade": 0.0  # Commission-free stocks
        }
        
        rate = commission_rates.get(exchange, 0.001)
        return quantity * price * rate
    
    async def _update_position(self, execution: TradeExecution):
        """Update position based on execution"""
        position_key = f"{execution.symbol}_{execution.exchange}"
        
        if position_key in self.positions:
            position = self.positions[position_key]
            
            if execution.side == OrderSide.BUY:
                # Adding to position
                total_cost = (position.quantity * position.average_price) + (execution.quantity * execution.price)
                position.quantity += execution.quantity
                position.average_price = total_cost / position.quantity
                position.side = "long"
            else:
                # Reducing position
                if position.quantity >= execution.quantity:
                    position.quantity -= execution.quantity
                    if position.quantity == 0:
                        del self.positions[position_key]
                        return
                else:
                    # Reversing position
                    remaining_quantity = execution.quantity - position.quantity
                    position.quantity = remaining_quantity
                    position.average_price = execution.price
                    position.side = "short"
            
            position.updated_at = datetime.utcnow()
            
        else:
            # New position
            self.positions[position_key] = Position(
                symbol=execution.symbol,
                exchange=execution.exchange,
                quantity=execution.quantity if execution.side == OrderSide.BUY else -execution.quantity,
                average_price=execution.price,
                current_price=execution.price,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                market_value=execution.quantity * execution.price,
                side="long" if execution.side == OrderSide.BUY else "short",
                opened_at=datetime.utcnow()
            )
    
    async def _perform_risk_check(self, symbol: str, exchange: str, side: OrderSide, 
                                 quantity: float, price: Optional[float]) -> Dict[str, Any]:
        """Perform risk check for a trade"""
        # Calculate position value
        if price is None:
            price = await self._get_current_price(symbol, exchange)
        
        position_value = quantity * price
        
        # Check position size limits
        if position_value > self.max_position_size:
            return {
                "approved": False,
                "reason": f"Position size ${position_value:,.2f} exceeds limit ${self.max_position_size:,.2f}"
            }
        
        # Check daily loss limits
        if self.daily_pnl < -self.max_daily_loss:
            return {
                "approved": False,
                "reason": f"Daily loss limit reached: ${self.daily_pnl:,.2f}"
            }
        
        # Check emergency stop
        if self.emergency_stop_enabled:
            return {
                "approved": False,
                "reason": "Emergency stop is enabled"
            }
        
        return {"approved": True, "reason": "Risk check passed"}
    
    async def _get_exchange_client(self, exchange: str):
        """Get exchange client for order execution"""
        # Return mock client for simulation
        return {"exchange": exchange, "status": "connected"}
    
    async def _send_execution_notification(self, execution: TradeExecution):
        """Send execution notification"""
        notification = {
            "message_id": str(uuid.uuid4()),
            "message_type": "notification",
            "sender": self.agent_id,
            "payload": {
                "notification_type": "trade_executed",
                "execution_id": execution.execution_id,
                "symbol": execution.symbol,
                "exchange": execution.exchange,
                "side": execution.side.value,
                "quantity": execution.quantity,
                "price": execution.price,
                "timestamp": execution.timestamp.isoformat()
            }
        }
        
        await self.websocket.send(json.dumps(notification))
    
    async def _monitor_positions(self):
        """Monitor positions and update P&L"""
        while self.running:
            try:
                for position in self.positions.values():
                    # Get current price
                    current_price = await self._get_current_price(position.symbol, position.exchange)
                    position.current_price = current_price
                    
                    # Calculate unrealized P&L
                    if position.side == "long":
                        position.unrealized_pnl = position.quantity * (current_price - position.average_price)
                    else:
                        position.unrealized_pnl = position.quantity * (position.average_price - current_price)
                    
                    position.market_value = position.quantity * current_price
                    position.updated_at = datetime.utcnow()
                
                # Update total P&L
                self.daily_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                self.logger.error(f"Error monitoring positions: {e}")
                await asyncio.sleep(60)
    
    async def _process_pending_orders(self):
        """Process pending orders"""
        while self.running:
            try:
                pending_orders = [order for order in self.orders.values() 
                                if order.status == OrderStatus.PENDING]
                
                for order in pending_orders:
                    try:
                        # Try to execute pending order
                        if order.order_type == OrderType.LIMIT:
                            execution = await self._execute_limit_order(order)
                            await self._update_position(execution)
                            self.executions.append(execution)
                            order.status = OrderStatus.FILLED
                            
                        elif order.order_type == OrderType.STOP:
                            execution = await self._execute_stop_order(order)
                            await self._update_position(execution)
                            self.executions.append(execution)
                            order.status = OrderStatus.FILLED
                            
                    except ValueError:
                        # Order not executable yet
                        continue
                    except Exception as e:
                        self.logger.error(f"Error processing pending order {order.order_id}: {e}")
                        order.status = OrderStatus.REJECTED
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error processing pending orders: {e}")
                await asyncio.sleep(10)
    
    async def _cancel_all_orders(self):
        """Cancel all pending orders"""
        for order in self.orders.values():
            if order.status == OrderStatus.PENDING:
                order.status = OrderStatus.CANCELLED
                order.updated_at = datetime.utcnow()
    
    async def _handle_get_positions_request(self, data: Dict[str, Any]):
        """Handle get positions request"""
        positions_data = {}
        for key, position in self.positions.items():
            positions_data[key] = {
                "symbol": position.symbol,
                "exchange": position.exchange,
                "quantity": position.quantity,
                "average_price": position.average_price,
                "current_price": position.current_price,
                "unrealized_pnl": position.unrealized_pnl,
                "market_value": position.market_value,
                "side": position.side,
                "opened_at": position.opened_at.isoformat()
            }
        
        response = {
            "message_id": str(uuid.uuid4()),
            "message_type": "response",
            "sender": self.agent_id,
            "recipient": data["sender"],
            "payload": {
                "request_type": "get_positions",
                "positions": positions_data
            },
            "correlation_id": data["message_id"]
        }
        
        await self.websocket.send(json.dumps(response))
    
    async def _send_error_response(self, original_message: Dict[str, Any], error: str):
        """Send error response"""
        response = {
            "message_id": str(uuid.uuid4()),
            "message_type": "error",
            "sender": self.agent_id,
            "recipient": original_message["sender"],
            "payload": {"error": error},
            "correlation_id": original_message["message_id"]
        }
        
        await self.websocket.send(json.dumps(response))
    
    async def _send_heartbeat(self):
        """Send heartbeat to MCP server"""
        while self.running:
            try:
                heartbeat = {
                    "message_id": str(uuid.uuid4()),
                    "message_type": "heartbeat",
                    "sender": self.agent_id,
                    "payload": {
                        "status": "online",
                        "metrics": self.metrics,
                        "positions_count": len(self.positions),
                        "pending_orders": len([o for o in self.orders.values() if o.status == OrderStatus.PENDING]),
                        "daily_pnl": self.daily_pnl
                    }
                }
                
                await self.websocket.send(json.dumps(heartbeat))
                await asyncio.sleep(30)  # Send every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error sending heartbeat: {e}")
                await asyncio.sleep(30)
    
    async def _handle_market_update(self, data: Dict[str, Any]):
        """Handle market update notification"""
        # Update position values based on market data
        payload = data.get("payload", {})
        symbol = payload.get("symbol")
        price = payload.get("price")
        
        if symbol and price:
            for position in self.positions.values():
                if position.symbol == symbol:
                    position.current_price = price
                    position.updated_at = datetime.utcnow()
    
    async def _handle_risk_alert(self, data: Dict[str, Any]):
        """Handle risk alert notification"""
        payload = data.get("payload", {})
        alert_type = payload.get("alert_type")
        
        if alert_type == "emergency_stop":
            self.emergency_stop_enabled = True
            self.logger.warning("Emergency stop activated")
            
            # Cancel all pending orders
            await self._cancel_all_orders()
    
    async def _handle_position_update(self, data: Dict[str, Any]):
        """Handle position update notification"""
        # Process position updates from other systems
        payload = data.get("payload", {})
        symbol = payload.get("symbol")
        
        if symbol:
            # Refresh position data
            for position in self.positions.values():
                if position.symbol == symbol:
                    position.updated_at = datetime.utcnow()

if __name__ == "__main__":
    # Run the agent
    agent = TraderAgent()
    
    async def main():
        await agent.start()
        
        # Keep running
        while agent.running:
            await asyncio.sleep(1)
    
    asyncio.run(main())
