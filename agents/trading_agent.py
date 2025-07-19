"""
Trading Agent - Executes trades and manages positions
Part of the multi-agent system coordinated by MCP server
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import websockets
from enum import Enum

from config import Config
from agents.mcp_server import MCPMessage, MessageType, AgentRole

class TradeAction(Enum):
    """Trade action types"""
    BUY = "buy"
    SELL = "sell"
    CLOSE = "close"
    MODIFY = "modify"

@dataclass
class TradeDecision:
    """Trade decision from analysis"""
    symbol: str
    action: TradeAction
    quantity: float
    price: float
    stop_loss: float
    take_profit: float
    confidence: float
    reasoning: str
    risk_score: float
    expected_return: float
    max_risk: float

class TradingAgent:
    """
    Trading Agent responsible for:
    - Executing trades based on signals
    - Managing positions
    - Risk monitoring
    - Order management
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.agent_id = f"trading_agent_{uuid.uuid4().hex[:8]}"
        self.logger = logging.getLogger(f"TradingAgent_{self.agent_id}")
        
        # MCP connection
        self.mcp_host = config.mcp.server_host
        self.mcp_port = config.mcp.server_port
        self.websocket = None
        self.is_connected = False
        
        # Agent state
        self.is_active = False
        self.last_heartbeat = datetime.now()
        
        # Trading state
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        self.pending_orders: Dict[str, Dict[str, Any]] = {}
        self.trade_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.performance_metrics = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0
        }
        
        # Trading parameters
        self.max_position_size = config.trading.max_position_size
        self.max_concurrent_trades = config.trading.max_concurrent_trades
        self.default_stop_loss = config.trading.default_stop_loss
        self.default_take_profit = config.trading.default_take_profit
        
        # Risk management
        self.risk_limits = {
            "max_daily_loss": config.trading.max_daily_loss,
            "max_drawdown": config.trading.max_drawdown,
            "max_position_risk": 0.02,  # 2% per position
            "max_correlation": 0.7
        }
        
        # Message handlers
        self.message_handlers = {
            "trading_signal": self._handle_trading_signal,
            "risk_alert": self._handle_risk_alert,
            "market_update": self._handle_market_update,
            "position_update": self._handle_position_update,
            "order_update": self._handle_order_update,
            "system_command": self._handle_system_command
        }
        
        # Capabilities
        self.capabilities = [
            "execute_trades",
            "manage_positions",
            "risk_monitoring",
            "order_management",
            "performance_tracking"
        ]
    
    async def start(self):
        """Start the trading agent"""
        try:
            self.logger.info(f"Starting Trading Agent: {self.agent_id}")
            
            # Connect to MCP server
            await self._connect_to_mcp()
            
            # Register with MCP server
            await self._register_with_mcp()
            
            # Start main agent loop
            self.is_active = True
            asyncio.create_task(self._agent_loop())
            asyncio.create_task(self._heartbeat_loop())
            
            self.logger.info("Trading Agent started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start Trading Agent: {e}")
            raise
    
    async def stop(self):
        """Stop the trading agent"""
        try:
            self.logger.info("Stopping Trading Agent...")
            
            self.is_active = False
            
            # Close all positions if configured
            if self.config.trading.paper_trading:
                await self._close_all_positions()
            
            # Unregister from MCP
            await self._unregister_from_mcp()
            
            # Close websocket
            if self.websocket:
                await self.websocket.close()
            
            self.logger.info("Trading Agent stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping Trading Agent: {e}")
    
    async def _connect_to_mcp(self):
        """Connect to MCP server"""
        try:
            uri = f"ws://{self.mcp_host}:{self.mcp_port}"
            self.websocket = await websockets.connect(uri)
            self.is_connected = True
            
            # Start message listener
            asyncio.create_task(self._message_listener())
            
        except Exception as e:
            self.logger.error(f"Failed to connect to MCP server: {e}")
            raise
    
    async def _register_with_mcp(self):
        """Register with MCP server"""
        try:
            registration_message = MCPMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.REGISTER,
                sender_id=self.agent_id,
                data={
                    "role": AgentRole.TRADING_AGENT.value,
                    "capabilities": self.capabilities,
                    "metadata": {
                        "max_concurrent_trades": self.max_concurrent_trades,
                        "risk_limits": self.risk_limits
                    }
                }
            )
            
            await self._send_message(registration_message)
            
        except Exception as e:
            self.logger.error(f"Failed to register with MCP: {e}")
            raise
    
    async def _unregister_from_mcp(self):
        """Unregister from MCP server"""
        try:
            unregistration_message = MCPMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.UNREGISTER,
                sender_id=self.agent_id
            )
            
            await self._send_message(unregistration_message)
            
        except Exception as e:
            self.logger.error(f"Failed to unregister from MCP: {e}")
    
    async def _agent_loop(self):
        """Main agent processing loop"""
        while self.is_active:
            try:
                # Update positions
                await self._update_positions()
                
                # Check pending orders
                await self._check_pending_orders()
                
                # Monitor risk
                await self._monitor_risk()
                
                # Update performance metrics
                await self._update_performance_metrics()
                
                # Sleep before next iteration
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in agent loop: {e}")
                await asyncio.sleep(5)
    
    async def _heartbeat_loop(self):
        """Send heartbeat to MCP server"""
        while self.is_active:
            try:
                heartbeat_message = MCPMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.HEARTBEAT,
                    sender_id=self.agent_id,
                    data={"timestamp": datetime.now().isoformat()}
                )
                
                await self._send_message(heartbeat_message)
                
                await asyncio.sleep(self.config.mcp.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(30)
    
    async def _message_listener(self):
        """Listen for messages from MCP server"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    
                    # Handle different message types
                    if data.get("message_type") == "broadcast":
                        await self._handle_broadcast_message(data)
                    elif data.get("message_type") == "response":
                        await self._handle_response_message(data)
                    elif data.get("message_type") == "request":
                        await self._handle_request_message(data)
                    
                except json.JSONDecodeError:
                    self.logger.error("Invalid JSON received from MCP server")
                except Exception as e:
                    self.logger.error(f"Error processing message: {e}")
        
        except websockets.exceptions.ConnectionClosed:
            self.logger.warning("Connection to MCP server closed")
            self.is_connected = False
        except Exception as e:
            self.logger.error(f"Error in message listener: {e}")
    
    async def _handle_broadcast_message(self, data: Dict[str, Any]):
        """Handle broadcast messages from MCP server"""
        try:
            message_data = data.get("data", {})
            message_type = message_data.get("type")
            
            handler = self.message_handlers.get(message_type)
            if handler:
                await handler(message_data)
            else:
                self.logger.debug(f"No handler for message type: {message_type}")
        
        except Exception as e:
            self.logger.error(f"Error handling broadcast message: {e}")
    
    async def _handle_response_message(self, data: Dict[str, Any]):
        """Handle response messages from MCP server"""
        try:
            # Handle registration confirmation
            if data.get("data", {}).get("status") == "registered":
                self.logger.info("Successfully registered with MCP server")
        
        except Exception as e:
            self.logger.error(f"Error handling response message: {e}")
    
    async def _handle_request_message(self, data: Dict[str, Any]):
        """Handle request messages from MCP server"""
        try:
            # Handle requests for agent status, positions, etc.
            request_type = data.get("data", {}).get("request_type")
            
            if request_type == "get_positions":
                response = await self._get_positions_response()
            elif request_type == "get_performance":
                response = await self._get_performance_response()
            elif request_type == "get_status":
                response = await self._get_status_response()
            else:
                response = {"error": f"Unknown request type: {request_type}"}
            
            # Send response
            response_message = MCPMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.RESPONSE,
                sender_id=self.agent_id,
                recipient_id=data.get("sender_id"),
                data=response,
                correlation_id=data.get("message_id")
            )
            
            await self._send_message(response_message)
        
        except Exception as e:
            self.logger.error(f"Error handling request message: {e}")
    
    async def _handle_trading_signal(self, data: Dict[str, Any]):
        """Handle trading signal from market analyst"""
        try:
            signal = data.get("signal", {})
            
            self.logger.info(f"Received trading signal: {signal.get('symbol')} {signal.get('action')}")
            
            # Validate signal
            if not self._validate_signal(signal):
                self.logger.warning(f"Invalid signal received: {signal}")
                return
            
            # Check if we can execute this trade
            if not await self._can_execute_trade(signal):
                self.logger.info(f"Cannot execute trade for {signal.get('symbol')}: risk/capacity limits")
                return
            
            # Create trade decision
            trade_decision = TradeDecision(
                symbol=signal.get("symbol"),
                action=TradeAction(signal.get("action")),
                quantity=signal.get("quantity", 0),
                price=signal.get("price", 0),
                stop_loss=signal.get("stop_loss", 0),
                take_profit=signal.get("take_profit", 0),
                confidence=signal.get("confidence", 0),
                reasoning=signal.get("reasoning", ""),
                risk_score=signal.get("risk_score", 0),
                expected_return=signal.get("expected_return", 0),
                max_risk=signal.get("max_risk", 0)
            )
            
            # Execute trade
            await self._execute_trade(trade_decision)
            
        except Exception as e:
            self.logger.error(f"Error handling trading signal: {e}")
    
    async def _handle_risk_alert(self, data: Dict[str, Any]):
        """Handle risk alert from risk manager"""
        try:
            alert = data.get("alert", {})
            severity = alert.get("severity", "info")
            
            self.logger.warning(f"Risk alert: {alert.get('message')}")
            
            # Take action based on severity
            if severity == "critical":
                await self._handle_critical_risk_alert(alert)
            elif severity == "warning":
                await self._handle_warning_risk_alert(alert)
            
        except Exception as e:
            self.logger.error(f"Error handling risk alert: {e}")
    
    async def _handle_market_update(self, data: Dict[str, Any]):
        """Handle market update from market analyst"""
        try:
            update = data.get("update", {})
            
            # Update position values based on market prices
            for symbol, position in self.active_positions.items():
                if symbol in update:
                    position["current_price"] = update[symbol].get("price", position["current_price"])
                    position["unrealized_pnl"] = self._calculate_unrealized_pnl(position)
            
        except Exception as e:
            self.logger.error(f"Error handling market update: {e}")
    
    async def _handle_position_update(self, data: Dict[str, Any]):
        """Handle position update from exchange"""
        try:
            update = data.get("update", {})
            symbol = update.get("symbol")
            
            if symbol in self.active_positions:
                self.active_positions[symbol].update(update)
                
        except Exception as e:
            self.logger.error(f"Error handling position update: {e}")
    
    async def _handle_order_update(self, data: Dict[str, Any]):
        """Handle order update from exchange"""
        try:
            update = data.get("update", {})
            order_id = update.get("order_id")
            
            if order_id in self.pending_orders:
                self.pending_orders[order_id].update(update)
                
                # If order is filled, update position
                if update.get("status") == "filled":
                    await self._handle_order_filled(order_id, update)
                
        except Exception as e:
            self.logger.error(f"Error handling order update: {e}")
    
    async def _handle_system_command(self, data: Dict[str, Any]):
        """Handle system command"""
        try:
            command = data.get("command")
            
            if command == "stop_trading":
                await self._stop_trading()
            elif command == "close_all_positions":
                await self._close_all_positions()
            elif command == "pause_trading":
                await self._pause_trading()
            elif command == "resume_trading":
                await self._resume_trading()
            
        except Exception as e:
            self.logger.error(f"Error handling system command: {e}")
    
    async def _execute_trade(self, decision: TradeDecision):
        """Execute a trade decision"""
        try:
            # Calculate position size
            position_size = await self._calculate_position_size(decision)
            
            # Create order
            order = {
                "order_id": str(uuid.uuid4()),
                "symbol": decision.symbol,
                "action": decision.action.value,
                "quantity": position_size,
                "price": decision.price,
                "stop_loss": decision.stop_loss,
                "take_profit": decision.take_profit,
                "timestamp": datetime.now(),
                "status": "pending",
                "reasoning": decision.reasoning,
                "confidence": decision.confidence
            }
            
            # Add to pending orders
            self.pending_orders[order["order_id"]] = order
            
            # Send order to exchange (via MCP)
            await self._send_order_to_exchange(order)
            
            self.logger.info(f"Trade executed: {decision.symbol} {decision.action.value} {position_size}")
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
    
    async def _send_order_to_exchange(self, order: Dict[str, Any]):
        """Send order to exchange via MCP"""
        try:
            request_message = MCPMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.REQUEST,
                sender_id=self.agent_id,
                data={
                    "request_type": "execute_order",
                    "order": order
                }
            )
            
            await self._send_message(request_message)
            
        except Exception as e:
            self.logger.error(f"Error sending order to exchange: {e}")
    
    async def _calculate_position_size(self, decision: TradeDecision) -> float:
        """Calculate position size based on risk management"""
        try:
            # Base position size
            base_size = decision.quantity
            
            # Adjust for confidence
            confidence_adjustment = min(decision.confidence, 1.0)
            
            # Adjust for risk
            risk_adjustment = max(0.1, 1.0 - decision.risk_score)
            
            # Calculate final size
            position_size = base_size * confidence_adjustment * risk_adjustment
            
            # Ensure within limits
            max_size = self.max_position_size
            position_size = min(position_size, max_size)
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    async def _update_positions(self):
        """Update active positions"""
        try:
            # Update unrealized PnL for all positions
            for symbol, position in self.active_positions.items():
                position["unrealized_pnl"] = self._calculate_unrealized_pnl(position)
                
                # Check for stop loss/take profit
                await self._check_position_exits(position)
            
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
    
    async def _check_pending_orders(self):
        """Check status of pending orders"""
        try:
            for order_id, order in list(self.pending_orders.items()):
                # Check if order is too old
                if datetime.now() - order["timestamp"] > timedelta(minutes=5):
                    self.logger.warning(f"Order {order_id} is stale, canceling")
                    await self._cancel_order(order_id)
            
        except Exception as e:
            self.logger.error(f"Error checking pending orders: {e}")
    
    async def _monitor_risk(self):
        """Monitor risk levels"""
        try:
            # Calculate current risk metrics
            total_exposure = sum(
                abs(pos["quantity"] * pos["current_price"]) 
                for pos in self.active_positions.values()
            )
            
            # Check if we're within risk limits
            if total_exposure > self.risk_limits["max_daily_loss"]:
                await self._handle_risk_limit_breach("daily_loss")
            
        except Exception as e:
            self.logger.error(f"Error monitoring risk: {e}")
    
    async def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            # Calculate metrics from trade history
            closed_trades = [t for t in self.trade_history if t.get("status") == "closed"]
            
            if closed_trades:
                winning_trades = [t for t in closed_trades if t["pnl"] > 0]
                losing_trades = [t for t in closed_trades if t["pnl"] < 0]
                
                self.performance_metrics.update({
                    "total_trades": len(closed_trades),
                    "winning_trades": len(winning_trades),
                    "losing_trades": len(losing_trades),
                    "win_rate": len(winning_trades) / len(closed_trades) if closed_trades else 0,
                    "total_pnl": sum(t["pnl"] for t in closed_trades),
                    "avg_win": sum(t["pnl"] for t in winning_trades) / len(winning_trades) if winning_trades else 0,
                    "avg_loss": sum(t["pnl"] for t in losing_trades) / len(losing_trades) if losing_trades else 0
                })
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def _validate_signal(self, signal: Dict[str, Any]) -> bool:
        """Validate trading signal"""
        required_fields = ["symbol", "action", "quantity", "price"]
        
        for field in required_fields:
            if field not in signal:
                return False
        
        if signal["action"] not in ["buy", "sell"]:
            return False
        
        if signal["quantity"] <= 0 or signal["price"] <= 0:
            return False
        
        return True
    
    async def _can_execute_trade(self, signal: Dict[str, Any]) -> bool:
        """Check if we can execute this trade"""
        try:
            # Check if we have capacity
            if len(self.active_positions) >= self.max_concurrent_trades:
                return False
            
            # Check if we already have a position in this symbol
            if signal["symbol"] in self.active_positions:
                return False
            
            # Check risk limits
            # This would include more sophisticated risk checks
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking trade capacity: {e}")
            return False
    
    def _calculate_unrealized_pnl(self, position: Dict[str, Any]) -> float:
        """Calculate unrealized PnL for a position"""
        try:
            entry_price = position["entry_price"]
            current_price = position["current_price"]
            quantity = position["quantity"]
            side = position["side"]
            
            if side == "buy":
                return (current_price - entry_price) * quantity
            else:
                return (entry_price - current_price) * quantity
                
        except Exception as e:
            self.logger.error(f"Error calculating unrealized PnL: {e}")
            return 0.0
    
    async def _check_position_exits(self, position: Dict[str, Any]):
        """Check if position should be closed"""
        try:
            current_price = position["current_price"]
            stop_loss = position.get("stop_loss", 0)
            take_profit = position.get("take_profit", 0)
            side = position["side"]
            
            should_close = False
            close_reason = ""
            
            if side == "buy":
                if current_price <= stop_loss:
                    should_close = True
                    close_reason = "stop_loss"
                elif current_price >= take_profit:
                    should_close = True
                    close_reason = "take_profit"
            else:
                if current_price >= stop_loss:
                    should_close = True
                    close_reason = "stop_loss"
                elif current_price <= take_profit:
                    should_close = True
                    close_reason = "take_profit"
            
            if should_close:
                await self._close_position(position, close_reason)
                
        except Exception as e:
            self.logger.error(f"Error checking position exits: {e}")
    
    async def _close_position(self, position: Dict[str, Any], reason: str):
        """Close a position"""
        try:
            # Create closing order
            close_order = {
                "order_id": str(uuid.uuid4()),
                "symbol": position["symbol"],
                "action": "sell" if position["side"] == "buy" else "buy",
                "quantity": position["quantity"],
                "price": position["current_price"],
                "timestamp": datetime.now(),
                "status": "pending",
                "reasoning": f"Position close: {reason}"
            }
            
            # Send close order
            await self._send_order_to_exchange(close_order)
            
            # Remove from active positions
            if position["symbol"] in self.active_positions:
                del self.active_positions[position["symbol"]]
            
            # Add to trade history
            trade_record = {
                "symbol": position["symbol"],
                "side": position["side"],
                "quantity": position["quantity"],
                "entry_price": position["entry_price"],
                "exit_price": position["current_price"],
                "pnl": position["unrealized_pnl"],
                "entry_time": position["timestamp"],
                "exit_time": datetime.now(),
                "close_reason": reason,
                "status": "closed"
            }
            
            self.trade_history.append(trade_record)
            
            self.logger.info(f"Position closed: {position['symbol']} ({reason}) PnL: {position['unrealized_pnl']:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
    
    async def _handle_order_filled(self, order_id: str, update: Dict[str, Any]):
        """Handle filled order"""
        try:
            order = self.pending_orders[order_id]
            
            # Create position if it's an opening trade
            if order["action"] in ["buy", "sell"]:
                position = {
                    "symbol": order["symbol"],
                    "side": order["action"],
                    "quantity": update.get("filled_quantity", order["quantity"]),
                    "entry_price": update.get("filled_price", order["price"]),
                    "current_price": update.get("filled_price", order["price"]),
                    "stop_loss": order.get("stop_loss", 0),
                    "take_profit": order.get("take_profit", 0),
                    "timestamp": datetime.now(),
                    "unrealized_pnl": 0.0
                }
                
                self.active_positions[order["symbol"]] = position
                
                self.logger.info(f"Position opened: {order['symbol']} {order['action']} {position['quantity']}")
            
            # Remove from pending orders
            del self.pending_orders[order_id]
            
        except Exception as e:
            self.logger.error(f"Error handling filled order: {e}")
    
    async def _cancel_order(self, order_id: str):
        """Cancel an order"""
        try:
            # Send cancel request to exchange
            request_message = MCPMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.REQUEST,
                sender_id=self.agent_id,
                data={
                    "request_type": "cancel_order",
                    "order_id": order_id
                }
            )
            
            await self._send_message(request_message)
            
            # Remove from pending orders
            if order_id in self.pending_orders:
                del self.pending_orders[order_id]
            
        except Exception as e:
            self.logger.error(f"Error canceling order: {e}")
    
    async def _handle_critical_risk_alert(self, alert: Dict[str, Any]):
        """Handle critical risk alert"""
        try:
            # Close all positions immediately
            await self._close_all_positions()
            
            # Stop trading
            await self._stop_trading()
            
            self.logger.critical(f"Critical risk alert handled: {alert.get('message')}")
            
        except Exception as e:
            self.logger.error(f"Error handling critical risk alert: {e}")
    
    async def _handle_warning_risk_alert(self, alert: Dict[str, Any]):
        """Handle warning risk alert"""
        try:
            # Reduce position sizes
            for position in self.active_positions.values():
                if position["unrealized_pnl"] < 0:
                    # Consider closing losing positions
                    await self._close_position(position, "risk_warning")
            
        except Exception as e:
            self.logger.error(f"Error handling warning risk alert: {e}")
    
    async def _handle_risk_limit_breach(self, limit_type: str):
        """Handle risk limit breach"""
        try:
            self.logger.warning(f"Risk limit breach: {limit_type}")
            
            # Take appropriate action
            if limit_type == "daily_loss":
                await self._stop_trading()
            elif limit_type == "max_drawdown":
                await self._close_all_positions()
            
        except Exception as e:
            self.logger.error(f"Error handling risk limit breach: {e}")
    
    async def _close_all_positions(self):
        """Close all active positions"""
        try:
            positions_to_close = list(self.active_positions.values())
            
            for position in positions_to_close:
                await self._close_position(position, "close_all")
            
            self.logger.info("All positions closed")
            
        except Exception as e:
            self.logger.error(f"Error closing all positions: {e}")
    
    async def _stop_trading(self):
        """Stop trading temporarily"""
        try:
            self.is_active = False
            self.logger.info("Trading stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping trading: {e}")
    
    async def _pause_trading(self):
        """Pause trading"""
        try:
            self.is_active = False
            self.logger.info("Trading paused")
            
        except Exception as e:
            self.logger.error(f"Error pausing trading: {e}")
    
    async def _resume_trading(self):
        """Resume trading"""
        try:
            self.is_active = True
            self.logger.info("Trading resumed")
            
        except Exception as e:
            self.logger.error(f"Error resuming trading: {e}")
    
    async def _send_message(self, message: MCPMessage):
        """Send message to MCP server"""
        try:
            if self.websocket and self.is_connected:
                data = {
                    "message_id": message.message_id,
                    "message_type": message.message_type.value,
                    "sender_id": message.sender_id,
                    "recipient_id": message.recipient_id,
                    "data": message.data,
                    "timestamp": message.timestamp.isoformat(),
                    "correlation_id": message.correlation_id,
                    "priority": message.priority
                }
                
                await self.websocket.send(json.dumps(data))
                
        except Exception as e:
            self.logger.error(f"Error sending message: {e}")
    
    # Response handlers
    async def _get_positions_response(self) -> Dict[str, Any]:
        """Get positions response"""
        return {
            "positions": [
                {
                    "symbol": pos["symbol"],
                    "side": pos["side"],
                    "quantity": pos["quantity"],
                    "entry_price": pos["entry_price"],
                    "current_price": pos["current_price"],
                    "unrealized_pnl": pos["unrealized_pnl"],
                    "timestamp": pos["timestamp"].isoformat()
                }
                for pos in self.active_positions.values()
            ]
        }
    
    async def _get_performance_response(self) -> Dict[str, Any]:
        """Get performance response"""
        return {
            "performance": self.performance_metrics
        }
    
    async def _get_status_response(self) -> Dict[str, Any]:
        """Get status response"""
        return {
            "status": {
                "is_active": self.is_active,
                "is_connected": self.is_connected,
                "active_positions": len(self.active_positions),
                "pending_orders": len(self.pending_orders),
                "last_heartbeat": self.last_heartbeat.isoformat()
            }
        }
