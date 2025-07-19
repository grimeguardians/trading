"""
Model Context Protocol (MCP) Server for Multi-Agent Coordination
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
import websockets
from websockets.server import WebSocketServerProtocol

class MessageType(str, Enum):
    """MCP message types"""
    INITIALIZE = "initialize"
    READY = "ready"
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    SHUTDOWN = "shutdown"

class AgentType(str, Enum):
    """Agent types in the system"""
    MARKET_ANALYST = "market_analyst"
    TRADER = "trader"
    NEWS_AGENT = "news_agent"
    STRATEGIZER = "strategizer"
    RISK_MANAGER = "risk_manager"
    PORTFOLIO_MANAGER = "portfolio_manager"

@dataclass
class MCPMessage:
    """MCP message structure"""
    message_id: str
    message_type: MessageType
    sender: str
    recipient: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None
    priority: int = 0  # 0 = highest, 9 = lowest

@dataclass
class AgentInfo:
    """Agent information"""
    agent_id: str
    agent_type: AgentType
    name: str
    description: str
    capabilities: List[str]
    status: str = "offline"
    last_heartbeat: Optional[datetime] = None
    websocket: Optional[WebSocketServerProtocol] = None

class MCPServer:
    """Model Context Protocol Server for agent coordination"""
    
    def __init__(self, digital_brain=None, exchange_manager=None, strategy_manager=None, 
                 portfolio_manager=None, risk_manager=None, port: int = 9000):
        self.port = port
        self.digital_brain = digital_brain
        self.exchange_manager = exchange_manager
        self.strategy_manager = strategy_manager
        self.portfolio_manager = portfolio_manager
        self.risk_manager = risk_manager
        
        # Server state
        self.server = None
        self.running = False
        self.agents: Dict[str, AgentInfo] = {}
        self.message_queue = asyncio.Queue()
        self.response_handlers: Dict[str, Callable] = {}
        
        # Message routing
        self.message_handlers: Dict[MessageType, Callable] = {
            MessageType.INITIALIZE: self._handle_initialize,
            MessageType.REQUEST: self._handle_request,
            MessageType.NOTIFICATION: self._handle_notification,
            MessageType.HEARTBEAT: self._handle_heartbeat,
            MessageType.SHUTDOWN: self._handle_shutdown
        }
        
        # Setup logging
        self.logger = logging.getLogger("MCPServer")
        
        # Performance metrics
        self.metrics = {
            "messages_processed": 0,
            "agents_connected": 0,
            "errors": 0,
            "average_response_time": 0.0
        }
    
    async def start(self):
        """Start the MCP server"""
        try:
            self.logger.info(f"Starting MCP server on port {self.port}")
            
            # Start the WebSocket server
            self.server = await websockets.serve(
                self._handle_websocket,
                "0.0.0.0",
                self.port,
                ping_interval=30,
                ping_timeout=10
            )
            
            self.running = True
            
            # Start background tasks
            asyncio.create_task(self._process_messages())
            asyncio.create_task(self._monitor_agents())
            asyncio.create_task(self._initialize_core_agents())
            
            self.logger.info("MCP server started successfully")
            
        except Exception as e:
            self.logger.error(f"Error starting MCP server: {e}")
            raise
    
    async def stop(self):
        """Stop the MCP server"""
        self.logger.info("Stopping MCP server")
        
        self.running = False
        
        # Notify all agents of shutdown
        shutdown_message = MCPMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.SHUTDOWN,
            sender="server",
            payload={"reason": "Server shutdown"}
        )
        
        await self._broadcast_message(shutdown_message)
        
        # Close all connections
        for agent in self.agents.values():
            if agent.websocket:
                await agent.websocket.close()
        
        # Stop the server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        self.logger.info("MCP server stopped")
    
    async def _handle_websocket(self, websocket: WebSocketServerProtocol, path: str):
        """Handle WebSocket connections"""
        self.logger.info(f"New WebSocket connection from {websocket.remote_address}")
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    mcp_message = MCPMessage(
                        message_id=data.get("message_id", str(uuid.uuid4())),
                        message_type=MessageType(data.get("message_type")),
                        sender=data.get("sender"),
                        recipient=data.get("recipient"),
                        payload=data.get("payload", {}),
                        correlation_id=data.get("correlation_id"),
                        priority=data.get("priority", 0)
                    )
                    
                    # Store websocket reference for sender
                    if mcp_message.sender in self.agents:
                        self.agents[mcp_message.sender].websocket = websocket
                    
                    await self.message_queue.put(mcp_message)
                    
                except json.JSONDecodeError as e:
                    self.logger.error(f"Invalid JSON message: {e}")
                    await self._send_error(websocket, "Invalid JSON format")
                except Exception as e:
                    self.logger.error(f"Error processing message: {e}")
                    await self._send_error(websocket, str(e))
        
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"WebSocket connection closed: {websocket.remote_address}")
        except Exception as e:
            self.logger.error(f"WebSocket error: {e}")
        finally:
            # Clean up agent references
            for agent in self.agents.values():
                if agent.websocket == websocket:
                    agent.websocket = None
                    agent.status = "offline"
    
    async def _process_messages(self):
        """Process messages from the queue"""
        while self.running:
            try:
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                
                start_time = datetime.utcnow()
                
                # Route message to appropriate handler
                handler = self.message_handlers.get(message.message_type)
                if handler:
                    await handler(message)
                else:
                    self.logger.warning(f"No handler for message type: {message.message_type}")
                
                # Update metrics
                self.metrics["messages_processed"] += 1
                response_time = (datetime.utcnow() - start_time).total_seconds()
                self.metrics["average_response_time"] = (
                    self.metrics["average_response_time"] * 0.9 + response_time * 0.1
                )
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")
                self.metrics["errors"] += 1
    
    async def _handle_initialize(self, message: MCPMessage):
        """Handle agent initialization"""
        payload = message.payload
        
        agent_info = AgentInfo(
            agent_id=message.sender,
            agent_type=AgentType(payload.get("agent_type")),
            name=payload.get("name", message.sender),
            description=payload.get("description", ""),
            capabilities=payload.get("capabilities", []),
            status="online",
            last_heartbeat=datetime.utcnow()
        )
        
        self.agents[message.sender] = agent_info
        self.metrics["agents_connected"] += 1
        
        # Send ready response
        response = MCPMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.READY,
            sender="server",
            recipient=message.sender,
            payload={
                "status": "ready",
                "server_capabilities": ["message_routing", "agent_discovery", "load_balancing"]
            },
            correlation_id=message.message_id
        )
        
        await self._send_message(response)
        
        self.logger.info(f"Agent initialized: {agent_info.name} ({agent_info.agent_type})")
    
    async def _handle_request(self, message: MCPMessage):
        """Handle request messages"""
        request_type = message.payload.get("request_type")
        
        if request_type == "agent_discovery":
            await self._handle_agent_discovery(message)
        elif request_type == "market_data":
            await self._handle_market_data_request(message)
        elif request_type == "execute_trade":
            await self._handle_trade_execution(message)
        elif request_type == "risk_check":
            await self._handle_risk_check(message)
        elif request_type == "strategy_signal":
            await self._handle_strategy_signal(message)
        else:
            # Route to appropriate agent
            await self._route_message(message)
    
    async def _handle_notification(self, message: MCPMessage):
        """Handle notification messages"""
        notification_type = message.payload.get("notification_type")
        
        if notification_type == "market_update":
            await self._broadcast_market_update(message)
        elif notification_type == "trade_executed":
            await self._notify_trade_execution(message)
        elif notification_type == "risk_alert":
            await self._handle_risk_alert(message)
        else:
            # Route to specific recipient or broadcast
            if message.recipient:
                await self._route_message(message)
            else:
                await self._broadcast_message(message)
    
    async def _handle_heartbeat(self, message: MCPMessage):
        """Handle heartbeat messages"""
        if message.sender in self.agents:
            self.agents[message.sender].last_heartbeat = datetime.utcnow()
            self.agents[message.sender].status = "online"
    
    async def _handle_shutdown(self, message: MCPMessage):
        """Handle shutdown messages"""
        if message.sender in self.agents:
            self.agents[message.sender].status = "offline"
            del self.agents[message.sender]
            self.metrics["agents_connected"] -= 1
    
    async def _handle_agent_discovery(self, message: MCPMessage):
        """Handle agent discovery requests"""
        requested_type = message.payload.get("agent_type")
        
        agents = []
        for agent in self.agents.values():
            if not requested_type or agent.agent_type == requested_type:
                agents.append({
                    "agent_id": agent.agent_id,
                    "agent_type": agent.agent_type,
                    "name": agent.name,
                    "description": agent.description,
                    "capabilities": agent.capabilities,
                    "status": agent.status
                })
        
        response = MCPMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.RESPONSE,
            sender="server",
            recipient=message.sender,
            payload={
                "request_type": "agent_discovery",
                "agents": agents
            },
            correlation_id=message.message_id
        )
        
        await self._send_message(response)
    
    async def _handle_market_data_request(self, message: MCPMessage):
        """Handle market data requests"""
        symbol = message.payload.get("symbol")
        exchange = message.payload.get("exchange")
        
        if self.exchange_manager:
            try:
                market_data = await self.exchange_manager.get_market_data(symbol, exchange)
                
                response = MCPMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.RESPONSE,
                    sender="server",
                    recipient=message.sender,
                    payload={
                        "request_type": "market_data",
                        "symbol": symbol,
                        "exchange": exchange,
                        "data": market_data
                    },
                    correlation_id=message.message_id
                )
                
                await self._send_message(response)
                
            except Exception as e:
                await self._send_error_response(message, f"Market data error: {str(e)}")
    
    async def _handle_trade_execution(self, message: MCPMessage):
        """Handle trade execution requests"""
        if self.portfolio_manager:
            try:
                trade_params = message.payload
                result = await self.portfolio_manager.execute_trade(trade_params)
                
                response = MCPMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.RESPONSE,
                    sender="server",
                    recipient=message.sender,
                    payload={
                        "request_type": "execute_trade",
                        "result": result
                    },
                    correlation_id=message.message_id
                )
                
                await self._send_message(response)
                
            except Exception as e:
                await self._send_error_response(message, f"Trade execution error: {str(e)}")
    
    async def _handle_risk_check(self, message: MCPMessage):
        """Handle risk check requests"""
        if self.risk_manager:
            try:
                trade_params = message.payload
                risk_check = await self.risk_manager.check_trade_risk(trade_params)
                
                response = MCPMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.RESPONSE,
                    sender="server",
                    recipient=message.sender,
                    payload={
                        "request_type": "risk_check",
                        "risk_check": risk_check
                    },
                    correlation_id=message.message_id
                )
                
                await self._send_message(response)
                
            except Exception as e:
                await self._send_error_response(message, f"Risk check error: {str(e)}")
    
    async def _handle_strategy_signal(self, message: MCPMessage):
        """Handle strategy signal requests"""
        if self.strategy_manager:
            try:
                signal_params = message.payload
                signals = await self.strategy_manager.generate_signals(signal_params)
                
                response = MCPMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.RESPONSE,
                    sender="server",
                    recipient=message.sender,
                    payload={
                        "request_type": "strategy_signal",
                        "signals": signals
                    },
                    correlation_id=message.message_id
                )
                
                await self._send_message(response)
                
            except Exception as e:
                await self._send_error_response(message, f"Strategy signal error: {str(e)}")
    
    async def _route_message(self, message: MCPMessage):
        """Route message to appropriate agent"""
        if message.recipient and message.recipient in self.agents:
            await self._send_message(message)
        else:
            # Find appropriate agent based on message type and content
            target_agent = await self._find_target_agent(message)
            if target_agent:
                message.recipient = target_agent
                await self._send_message(message)
            else:
                await self._send_error_response(message, "No suitable agent found")
    
    async def _find_target_agent(self, message: MCPMessage) -> Optional[str]:
        """Find the most suitable agent for a message"""
        request_type = message.payload.get("request_type")
        
        # Simple routing logic (can be made more sophisticated)
        if request_type in ["market_analysis", "sentiment_analysis"]:
            return self._find_agent_by_type(AgentType.MARKET_ANALYST)
        elif request_type in ["trade_execution", "order_management"]:
            return self._find_agent_by_type(AgentType.TRADER)
        elif request_type in ["news_analysis", "event_processing"]:
            return self._find_agent_by_type(AgentType.NEWS_AGENT)
        elif request_type in ["strategy_generation", "signal_analysis"]:
            return self._find_agent_by_type(AgentType.STRATEGIZER)
        
        return None
    
    def _find_agent_by_type(self, agent_type: AgentType) -> Optional[str]:
        """Find an agent by type"""
        for agent in self.agents.values():
            if agent.agent_type == agent_type and agent.status == "online":
                return agent.agent_id
        return None
    
    async def _send_message(self, message: MCPMessage):
        """Send message to specific agent"""
        if message.recipient in self.agents:
            agent = self.agents[message.recipient]
            if agent.websocket:
                try:
                    message_data = {
                        "message_id": message.message_id,
                        "message_type": message.message_type.value,
                        "sender": message.sender,
                        "recipient": message.recipient,
                        "payload": message.payload,
                        "timestamp": message.timestamp.isoformat(),
                        "correlation_id": message.correlation_id,
                        "priority": message.priority
                    }
                    
                    await agent.websocket.send(json.dumps(message_data))
                    
                except Exception as e:
                    self.logger.error(f"Error sending message to {message.recipient}: {e}")
                    agent.status = "offline"
    
    async def _broadcast_message(self, message: MCPMessage):
        """Broadcast message to all agents"""
        for agent in self.agents.values():
            if agent.websocket and agent.status == "online":
                message.recipient = agent.agent_id
                await self._send_message(message)
    
    async def _send_error(self, websocket: WebSocketServerProtocol, error_message: str):
        """Send error message to websocket"""
        error_data = {
            "message_id": str(uuid.uuid4()),
            "message_type": MessageType.ERROR.value,
            "sender": "server",
            "payload": {"error": error_message},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            await websocket.send(json.dumps(error_data))
        except Exception as e:
            self.logger.error(f"Error sending error message: {e}")
    
    async def _send_error_response(self, original_message: MCPMessage, error_message: str):
        """Send error response to original message"""
        error_response = MCPMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.ERROR,
            sender="server",
            recipient=original_message.sender,
            payload={"error": error_message},
            correlation_id=original_message.message_id
        )
        
        await self._send_message(error_response)
    
    async def _monitor_agents(self):
        """Monitor agent health and connectivity"""
        while self.running:
            try:
                current_time = datetime.utcnow()
                
                # Check for stale agents
                stale_agents = []
                for agent_id, agent in self.agents.items():
                    if agent.last_heartbeat:
                        time_since_heartbeat = (current_time - agent.last_heartbeat).total_seconds()
                        if time_since_heartbeat > 60:  # 1 minute timeout
                            stale_agents.append(agent_id)
                
                # Mark stale agents as offline
                for agent_id in stale_agents:
                    self.agents[agent_id].status = "offline"
                    self.logger.warning(f"Agent {agent_id} marked as offline (no heartbeat)")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error monitoring agents: {e}")
                await asyncio.sleep(30)
    
    async def _initialize_core_agents(self):
        """Initialize core system agents"""
        # This would typically spawn the core agents
        # For now, we'll just log that they should be started
        self.logger.info("Core agents should be initialized externally")
    
    async def _broadcast_market_update(self, message: MCPMessage):
        """Broadcast market update to relevant agents"""
        # Route to market analysts and traders
        target_types = [AgentType.MARKET_ANALYST, AgentType.TRADER, AgentType.STRATEGIZER]
        
        for agent in self.agents.values():
            if agent.agent_type in target_types and agent.status == "online":
                message.recipient = agent.agent_id
                await self._send_message(message)
    
    async def _notify_trade_execution(self, message: MCPMessage):
        """Notify relevant agents of trade execution"""
        # Route to portfolio manager and risk manager
        target_types = [AgentType.PORTFOLIO_MANAGER, AgentType.RISK_MANAGER]
        
        for agent in self.agents.values():
            if agent.agent_type in target_types and agent.status == "online":
                message.recipient = agent.agent_id
                await self._send_message(message)
    
    async def _handle_risk_alert(self, message: MCPMessage):
        """Handle risk alert notifications"""
        # Broadcast to all agents - risk alerts are critical
        await self._broadcast_message(message)
    
    def is_running(self) -> bool:
        """Check if server is running"""
        return self.running
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get server metrics"""
        return {
            **self.metrics,
            "agents_online": len([a for a in self.agents.values() if a.status == "online"]),
            "agents_total": len(self.agents),
            "uptime": datetime.utcnow().isoformat()
        }
    
    def get_agents(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            agent_id: {
                "agent_type": agent.agent_type.value,
                "name": agent.name,
                "description": agent.description,
                "capabilities": agent.capabilities,
                "status": agent.status,
                "last_heartbeat": agent.last_heartbeat.isoformat() if agent.last_heartbeat else None
            }
            for agent_id, agent in self.agents.items()
        }
