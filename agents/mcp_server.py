"""
MCP (Model Context Protocol) Server for Agent Communication
Coordinates multi-agent interactions and manages shared context
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
import websockets
from enum import Enum
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

from config import Config
from knowledge.digital_brain import DigitalBrain
from core.freqtrade_engine import FreqtradeEngine

class MessageType(Enum):
    """MCP message types"""
    REGISTER = "register"
    UNREGISTER = "unregister"
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    HEARTBEAT = "heartbeat"
    ERROR = "error"

class AgentRole(Enum):
    """Agent roles in the system"""
    TRADING_AGENT = "trading_agent"
    MARKET_ANALYST = "market_analyst"
    NEWS_AGENT = "news_agent"
    STRATEGIZER = "strategizer"
    RISK_MANAGER = "risk_manager"
    CHAT_INTERFACE = "chat_interface"

@dataclass
class MCPMessage:
    """MCP protocol message"""
    message_id: str
    message_type: MessageType
    sender_id: str
    recipient_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    priority: int = 0  # 0 = normal, 1 = high, 2 = critical

@dataclass
class AgentRegistration:
    """Agent registration information"""
    agent_id: str
    agent_role: AgentRole
    capabilities: List[str]
    websocket: Any
    last_heartbeat: datetime
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SharedContext:
    """Shared context across agents"""
    market_data: Dict[str, Any] = field(default_factory=dict)
    trading_signals: List[Dict[str, Any]] = field(default_factory=list)
    risk_metrics: Dict[str, Any] = field(default_factory=dict)
    news_sentiment: Dict[str, Any] = field(default_factory=dict)
    portfolio_status: Dict[str, Any] = field(default_factory=dict)
    system_alerts: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

class MCPServer:
    """MCP Server for agent coordination"""
    
    def __init__(self, config: Config, digital_brain: DigitalBrain, trading_engine: FreqtradeEngine):
        self.config = config
        self.digital_brain = digital_brain
        self.trading_engine = trading_engine
        
        # Server configuration
        self.host = config.mcp.server_host
        self.port = config.mcp.server_port
        self.max_connections = config.mcp.max_connections
        self.heartbeat_interval = config.mcp.heartbeat_interval
        
        # Agent management
        self.registered_agents: Dict[str, AgentRegistration] = {}
        self.agent_roles: Dict[AgentRole, List[str]] = {role: [] for role in AgentRole}
        
        # Message handling
        self.message_queue = asyncio.Queue()
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.request_handlers: Dict[str, Callable] = {}
        
        # Context management
        self.shared_context = SharedContext()
        self.context_lock = asyncio.Lock()
        
        # Server state
        self.is_running = False
        self.server = None
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Logging
        self.logger = logging.getLogger("MCPServer")
        
        # Initialize message handlers
        self._setup_message_handlers()
        self._setup_request_handlers()
        
        # Performance metrics
        self.metrics = {
            "messages_processed": 0,
            "agents_connected": 0,
            "context_updates": 0,
            "errors": 0
        }
    
    async def initialize(self):
        """Initialize the MCP server"""
        try:
            self.logger.info("Initializing MCP Server...")
            
            # Set up message handlers
            self._setup_message_handlers()
            
            # Initialize shared context
            await self._initialize_shared_context()
            
            self.logger.info("MCP Server initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MCP server: {e}")
            raise
    
    async def start(self):
        """Start the MCP server"""
        try:
            self.logger.info(f"Starting MCP Server on {self.host}:{self.port}")
            
            # Start WebSocket server
            self.server = await websockets.serve(
                self._handle_client_connection,
                self.host,
                self.port,
                max_size=1048576,  # 1MB max message size
                ping_interval=self.heartbeat_interval,
                ping_timeout=self.heartbeat_interval * 2
            )
            
            self.is_running = True
            
            # Start background tasks
            asyncio.create_task(self._message_processor())
            asyncio.create_task(self._heartbeat_monitor())
            asyncio.create_task(self._context_updater())
            
            self.logger.info("MCP Server started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start MCP server: {e}")
            raise
    
    async def stop(self):
        """Stop the MCP server"""
        try:
            self.logger.info("Stopping MCP Server...")
            
            self.is_running = False
            
            # Close all agent connections
            for agent_id in list(self.registered_agents.keys()):
                await self._unregister_agent(agent_id)
            
            # Close server
            if self.server:
                self.server.close()
                await self.server.wait_closed()
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            self.logger.info("MCP Server stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping MCP server: {e}")
    
    async def _handle_client_connection(self, websocket, path):
        """Handle incoming client connections"""
        client_id = str(uuid.uuid4())
        
        try:
            self.logger.info(f"New client connection: {client_id}")
            
            async for message in websocket:
                try:
                    # Parse message
                    data = json.loads(message)
                    mcp_message = MCPMessage(
                        message_id=data.get("message_id", str(uuid.uuid4())),
                        message_type=MessageType(data.get("message_type")),
                        sender_id=data.get("sender_id", client_id),
                        recipient_id=data.get("recipient_id"),
                        data=data.get("data", {}),
                        correlation_id=data.get("correlation_id"),
                        priority=data.get("priority", 0)
                    )
                    
                    # Add to message queue
                    await self.message_queue.put((mcp_message, websocket))
                    
                except json.JSONDecodeError:
                    self.logger.error(f"Invalid JSON received from {client_id}")
                    await self._send_error(websocket, "Invalid JSON format")
                except Exception as e:
                    self.logger.error(f"Error processing message from {client_id}: {e}")
                    await self._send_error(websocket, f"Message processing error: {e}")
        
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Client {client_id} disconnected")
        except Exception as e:
            self.logger.error(f"Error handling client {client_id}: {e}")
        finally:
            # Clean up agent registration if exists
            if client_id in self.registered_agents:
                await self._unregister_agent(client_id)
    
    async def _message_processor(self):
        """Process incoming messages"""
        while self.is_running:
            try:
                # Get message from queue
                message, websocket = await asyncio.wait_for(
                    self.message_queue.get(), 
                    timeout=1.0
                )
                
                # Process message
                await self._process_message(message, websocket)
                
                # Update metrics
                self.metrics["messages_processed"] += 1
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error in message processor: {e}")
                self.metrics["errors"] += 1
    
    async def _process_message(self, message: MCPMessage, websocket):
        """Process a single message"""
        try:
            handler = self.message_handlers.get(message.message_type)
            if handler:
                await handler(message, websocket)
            else:
                self.logger.warning(f"No handler for message type: {message.message_type}")
                await self._send_error(websocket, f"Unknown message type: {message.message_type}")
        
        except Exception as e:
            self.logger.error(f"Error processing message {message.message_id}: {e}")
            await self._send_error(websocket, f"Processing error: {e}")
    
    async def _handle_register(self, message: MCPMessage, websocket):
        """Handle agent registration"""
        try:
            agent_id = message.sender_id
            agent_role = AgentRole(message.data.get("role"))
            capabilities = message.data.get("capabilities", [])
            metadata = message.data.get("metadata", {})
            
            # Register agent
            registration = AgentRegistration(
                agent_id=agent_id,
                agent_role=agent_role,
                capabilities=capabilities,
                websocket=websocket,
                last_heartbeat=datetime.now(),
                metadata=metadata
            )
            
            self.registered_agents[agent_id] = registration
            self.agent_roles[agent_role].append(agent_id)
            
            self.logger.info(f"Agent registered: {agent_id} ({agent_role.value})")
            
            # Send registration confirmation
            response = MCPMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.RESPONSE,
                sender_id="mcp_server",
                recipient_id=agent_id,
                data={
                    "status": "registered",
                    "agent_id": agent_id,
                    "shared_context": asdict(self.shared_context)
                },
                correlation_id=message.message_id
            )
            
            await self._send_message(websocket, response)
            
            # Update metrics
            self.metrics["agents_connected"] += 1
            
        except Exception as e:
            self.logger.error(f"Error handling registration: {e}")
            await self._send_error(websocket, f"Registration error: {e}")
    
    async def _handle_unregister(self, message: MCPMessage, websocket):
        """Handle agent unregistration"""
        try:
            agent_id = message.sender_id
            await self._unregister_agent(agent_id)
            
            # Send confirmation
            response = MCPMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.RESPONSE,
                sender_id="mcp_server",
                recipient_id=agent_id,
                data={"status": "unregistered"},
                correlation_id=message.message_id
            )
            
            await self._send_message(websocket, response)
            
        except Exception as e:
            self.logger.error(f"Error handling unregistration: {e}")
            await self._send_error(websocket, f"Unregistration error: {e}")
    
    async def _handle_request(self, message: MCPMessage, websocket):
        """Handle agent requests"""
        try:
            request_type = message.data.get("request_type")
            
            handler = self.request_handlers.get(request_type)
            if handler:
                response_data = await handler(message.data, message.sender_id)
                
                response = MCPMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.RESPONSE,
                    sender_id="mcp_server",
                    recipient_id=message.sender_id,
                    data=response_data,
                    correlation_id=message.message_id
                )
                
                await self._send_message(websocket, response)
            else:
                await self._send_error(websocket, f"Unknown request type: {request_type}")
        
        except Exception as e:
            self.logger.error(f"Error handling request: {e}")
            await self._send_error(websocket, f"Request error: {e}")
    
    async def _handle_broadcast(self, message: MCPMessage, websocket):
        """Handle broadcast messages"""
        try:
            # Broadcast to all relevant agents
            target_role = message.data.get("target_role")
            
            if target_role:
                # Broadcast to specific role
                role_enum = AgentRole(target_role)
                target_agents = self.agent_roles.get(role_enum, [])
            else:
                # Broadcast to all agents
                target_agents = list(self.registered_agents.keys())
            
            # Send to target agents
            for agent_id in target_agents:
                if agent_id != message.sender_id:  # Don't send back to sender
                    agent = self.registered_agents.get(agent_id)
                    if agent and agent.is_active:
                        broadcast_message = MCPMessage(
                            message_id=str(uuid.uuid4()),
                            message_type=MessageType.BROADCAST,
                            sender_id=message.sender_id,
                            recipient_id=agent_id,
                            data=message.data,
                            correlation_id=message.message_id
                        )
                        
                        await self._send_message(agent.websocket, broadcast_message)
        
        except Exception as e:
            self.logger.error(f"Error handling broadcast: {e}")
            await self._send_error(websocket, f"Broadcast error: {e}")
    
    async def _handle_heartbeat(self, message: MCPMessage, websocket):
        """Handle heartbeat messages"""
        try:
            agent_id = message.sender_id
            
            if agent_id in self.registered_agents:
                self.registered_agents[agent_id].last_heartbeat = datetime.now()
                
                # Send heartbeat response
                response = MCPMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.HEARTBEAT,
                    sender_id="mcp_server",
                    recipient_id=agent_id,
                    data={"status": "alive"},
                    correlation_id=message.message_id
                )
                
                await self._send_message(websocket, response)
        
        except Exception as e:
            self.logger.error(f"Error handling heartbeat: {e}")
    
    async def _send_message(self, websocket, message: MCPMessage):
        """Send message to websocket"""
        try:
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
            
            await websocket.send(json.dumps(data))
            
        except Exception as e:
            self.logger.error(f"Error sending message: {e}")
    
    async def _send_error(self, websocket, error_message: str):
        """Send error message"""
        try:
            error_msg = MCPMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.ERROR,
                sender_id="mcp_server",
                data={"error": error_message}
            )
            
            await self._send_message(websocket, error_msg)
            
        except Exception as e:
            self.logger.error(f"Error sending error message: {e}")
    
    async def _unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        try:
            if agent_id in self.registered_agents:
                agent = self.registered_agents[agent_id]
                
                # Remove from role mapping
                if agent.agent_role in self.agent_roles:
                    if agent_id in self.agent_roles[agent.agent_role]:
                        self.agent_roles[agent.agent_role].remove(agent_id)
                
                # Remove from registered agents
                del self.registered_agents[agent_id]
                
                self.logger.info(f"Agent unregistered: {agent_id}")
                
                # Update metrics
                self.metrics["agents_connected"] -= 1
        
        except Exception as e:
            self.logger.error(f"Error unregistering agent {agent_id}: {e}")
    
    async def _heartbeat_monitor(self):
        """Monitor agent heartbeats"""
        while self.is_running:
            try:
                current_time = datetime.now()
                timeout_threshold = current_time - timedelta(seconds=self.heartbeat_interval * 3)
                
                # Check for timed out agents
                timed_out_agents = []
                for agent_id, agent in self.registered_agents.items():
                    if agent.last_heartbeat < timeout_threshold:
                        timed_out_agents.append(agent_id)
                        agent.is_active = False
                
                # Unregister timed out agents
                for agent_id in timed_out_agents:
                    self.logger.warning(f"Agent {agent_id} timed out")
                    await self._unregister_agent(agent_id)
                
                # Wait before next check
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"Error in heartbeat monitor: {e}")
                await asyncio.sleep(5)
    
    async def _context_updater(self):
        """Update shared context periodically"""
        while self.is_running:
            try:
                await self._update_shared_context()
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in context updater: {e}")
                await asyncio.sleep(30)
    
    async def _update_shared_context(self):
        """Update shared context with latest data"""
        try:
            async with self.context_lock:
                # Update market data
                # This would fetch from market data pipeline
                
                # Update trading signals
                # This would fetch from trading engine
                
                # Update risk metrics
                # This would fetch from risk manager
                
                # Update portfolio status
                if self.trading_engine:
                    status = self.trading_engine.get_status()
                    self.shared_context.portfolio_status = status
                
                # Update timestamp
                self.shared_context.timestamp = datetime.now()
                
                # Update metrics
                self.metrics["context_updates"] += 1
                
                # Broadcast context update to interested agents
                await self._broadcast_context_update()
        
        except Exception as e:
            self.logger.error(f"Error updating shared context: {e}")
    
    async def _broadcast_context_update(self):
        """Broadcast context update to all agents"""
        try:
            update_message = MCPMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.BROADCAST,
                sender_id="mcp_server",
                data={
                    "type": "context_update",
                    "context": asdict(self.shared_context)
                }
            )
            
            # Send to all active agents
            for agent_id, agent in self.registered_agents.items():
                if agent.is_active:
                    await self._send_message(agent.websocket, update_message)
        
        except Exception as e:
            self.logger.error(f"Error broadcasting context update: {e}")
    
    def _setup_message_handlers(self):
        """Setup message handlers"""
        self.message_handlers = {
            MessageType.REGISTER: self._handle_register,
            MessageType.UNREGISTER: self._handle_unregister,
            MessageType.REQUEST: self._handle_request,
            MessageType.BROADCAST: self._handle_broadcast,
            MessageType.HEARTBEAT: self._handle_heartbeat
        }
    
    def _setup_request_handlers(self):
        """Setup request handlers"""
        self.request_handlers = {
            "get_market_data": self._handle_get_market_data,
            "get_trading_signals": self._handle_get_trading_signals,
            "get_risk_metrics": self._handle_get_risk_metrics,
            "get_portfolio_status": self._handle_get_portfolio_status,
            "query_digital_brain": self._handle_query_digital_brain,
            "execute_trade": self._handle_execute_trade,
            "get_agent_list": self._handle_get_agent_list
        }
    
    # Request handlers
    async def _handle_get_market_data(self, data: Dict[str, Any], sender_id: str) -> Dict[str, Any]:
        """Handle market data requests"""
        try:
            symbol = data.get("symbol")
            timeframe = data.get("timeframe", "1h")
            
            # This would fetch from market data pipeline
            return {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "data": []  # Would contain actual market data
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _handle_get_trading_signals(self, data: Dict[str, Any], sender_id: str) -> Dict[str, Any]:
        """Handle trading signals requests"""
        try:
            # This would fetch from trading engine
            return {
                "success": True,
                "signals": []  # Would contain actual signals
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _handle_get_risk_metrics(self, data: Dict[str, Any], sender_id: str) -> Dict[str, Any]:
        """Handle risk metrics requests"""
        try:
            # This would fetch from risk manager
            return {
                "success": True,
                "metrics": {}  # Would contain actual metrics
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _handle_get_portfolio_status(self, data: Dict[str, Any], sender_id: str) -> Dict[str, Any]:
        """Handle portfolio status requests"""
        try:
            if self.trading_engine:
                status = self.trading_engine.get_status()
                return {
                    "success": True,
                    "status": status
                }
            else:
                return {"success": False, "error": "Trading engine not available"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _handle_query_digital_brain(self, data: Dict[str, Any], sender_id: str) -> Dict[str, Any]:
        """Handle digital brain queries"""
        try:
            query = data.get("query")
            context = data.get("context", {})
            
            if self.digital_brain:
                response = await self.digital_brain.query(query, context)
                return {
                    "success": True,
                    "response": response
                }
            else:
                return {"success": False, "error": "Digital brain not available"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _handle_execute_trade(self, data: Dict[str, Any], sender_id: str) -> Dict[str, Any]:
        """Handle trade execution requests"""
        try:
            # This would execute trade through trading engine
            return {
                "success": True,
                "trade_id": str(uuid.uuid4())
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _handle_get_agent_list(self, data: Dict[str, Any], sender_id: str) -> Dict[str, Any]:
        """Handle agent list requests"""
        try:
            agents = []
            for agent_id, agent in self.registered_agents.items():
                agents.append({
                    "agent_id": agent_id,
                    "role": agent.agent_role.value,
                    "capabilities": agent.capabilities,
                    "is_active": agent.is_active,
                    "last_heartbeat": agent.last_heartbeat.isoformat()
                })
            
            return {
                "success": True,
                "agents": agents
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _initialize_shared_context(self):
        """Initialize shared context"""
        try:
            # Initialize with default values
            self.shared_context = SharedContext()
            
            # Load any persistent context data
            # This would load from database or file
            
        except Exception as e:
            self.logger.error(f"Error initializing shared context: {e}")
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.stop()
    
    # Public API methods
    def get_metrics(self) -> Dict[str, Any]:
        """Get server metrics"""
        return {
            **self.metrics,
            "active_agents": len([a for a in self.registered_agents.values() if a.is_active]),
            "total_registered": len(self.registered_agents),
            "agents_by_role": {role.value: len(agents) for role, agents in self.agent_roles.items()}
        }
    
    def get_agent_status(self) -> List[Dict[str, Any]]:
        """Get status of all agents"""
        return [
            {
                "agent_id": agent_id,
                "role": agent.agent_role.value,
                "capabilities": agent.capabilities,
                "is_active": agent.is_active,
                "last_heartbeat": agent.last_heartbeat.isoformat(),
                "metadata": agent.metadata
            }
            for agent_id, agent in self.registered_agents.items()
        ]
    
    async def send_system_alert(self, alert_type: str, message: str, severity: str = "info"):
        """Send system alert to all agents"""
        try:
            alert = {
                "type": "system_alert",
                "alert_type": alert_type,
                "message": message,
                "severity": severity,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add to shared context
            async with self.context_lock:
                self.shared_context.system_alerts.append(alert)
                
                # Limit alerts history
                if len(self.shared_context.system_alerts) > 100:
                    self.shared_context.system_alerts = self.shared_context.system_alerts[-100:]
            
            # Broadcast to all agents
            broadcast_message = MCPMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.BROADCAST,
                sender_id="mcp_server",
                data=alert
            )
            
            for agent_id, agent in self.registered_agents.items():
                if agent.is_active:
                    await self._send_message(agent.websocket, broadcast_message)
        
        except Exception as e:
            self.logger.error(f"Error sending system alert: {e}")
