"""
Model Context Protocol (MCP) Server for Agent Communication
Handles inter-agent messaging and coordination
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import websockets
from websockets.server import WebSocketServerProtocol
import traceback


class MessageType(Enum):
    """MCP message types"""
    AGENT_REGISTER = "agent_register"
    AGENT_HEARTBEAT = "agent_heartbeat"
    MARKET_DATA = "market_data"
    TRADING_SIGNAL = "trading_signal"
    ORDER_REQUEST = "order_request"
    ORDER_UPDATE = "order_update"
    KNOWLEDGE_QUERY = "knowledge_query"
    KNOWLEDGE_UPDATE = "knowledge_update"
    BROADCAST = "broadcast"
    DIRECT_MESSAGE = "direct_message"
    SYSTEM_EVENT = "system_event"


@dataclass
class MCPMessage:
    """MCP message structure"""
    id: str
    type: MessageType
    source: str
    target: Optional[str] = None
    timestamp: datetime = None
    data: Dict[str, Any] = None
    correlation_id: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.data is None:
            self.data = {}


@dataclass
class AgentInfo:
    """Agent registration information"""
    agent_id: str
    agent_type: str
    websocket: WebSocketServerProtocol
    capabilities: List[str]
    last_heartbeat: datetime
    status: str = "connected"


class MCPServer:
    """
    Model Context Protocol Server for multi-agent coordination
    Handles message routing, agent registration, and knowledge sharing
    """
    
    def __init__(self, host: str = "127.0.0.1", port: int = 9000, knowledge_engine=None):
        self.host = host
        self.port = port
        self.knowledge_engine = knowledge_engine
        self.logger = logging.getLogger("MCPServer")
        
        # Agent management
        self.agents: Dict[str, AgentInfo] = {}
        self.message_handlers: Dict[MessageType, List[Callable]] = {}
        self.message_history: List[MCPMessage] = []
        
        # Server state
        self.server = None
        self.running = False
        self.heartbeat_interval = 30  # seconds
        self.cleanup_interval = 300  # seconds
        
        # Message routing
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.broadcast_subscribers: Dict[str, List[str]] = {}
        
        # Performance metrics
        self.metrics = {
            "messages_processed": 0,
            "agents_connected": 0,
            "errors": 0,
            "uptime": datetime.utcnow()
        }
        
        # Initialize default handlers
        self._setup_default_handlers()
    
    def _setup_default_handlers(self):
        """Setup default message handlers"""
        self.register_handler(MessageType.AGENT_REGISTER, self._handle_agent_register)
        self.register_handler(MessageType.AGENT_HEARTBEAT, self._handle_agent_heartbeat)
        self.register_handler(MessageType.KNOWLEDGE_QUERY, self._handle_knowledge_query)
        self.register_handler(MessageType.KNOWLEDGE_UPDATE, self._handle_knowledge_update)
        self.register_handler(MessageType.BROADCAST, self._handle_broadcast)
        self.register_handler(MessageType.DIRECT_MESSAGE, self._handle_direct_message)
    
    def register_handler(self, message_type: MessageType, handler: Callable):
        """Register a message handler"""
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        self.message_handlers[message_type].append(handler)
        self.logger.debug(f"Registered handler for {message_type.value}")
    
    async def start(self):
        """Start the MCP server"""
        try:
            self.logger.info(f"🚀 Starting MCP Server on {self.host}:{self.port}")
            
            # Start the WebSocket server
            self.server = await websockets.serve(
                self._handle_client,
                self.host,
                self.port,
                ping_interval=20,
                ping_timeout=10,
                max_size=10 * 1024 * 1024  # 10MB max message size
            )
            
            self.running = True
            
            # Start background tasks
            asyncio.create_task(self._message_processor())
            asyncio.create_task(self._heartbeat_monitor())
            asyncio.create_task(self._cleanup_task())
            
            self.logger.info(f"✅ MCP Server started successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start MCP Server: {e}")
            raise
    
    async def stop(self):
        """Stop the MCP server"""
        try:
            self.logger.info("🔄 Stopping MCP Server...")
            
            self.running = False
            
            # Disconnect all agents
            for agent_id, agent_info in self.agents.items():
                try:
                    await agent_info.websocket.close()
                except:
                    pass
            
            # Close server
            if self.server:
                self.server.close()
                await self.server.wait_closed()
            
            self.logger.info("✅ MCP Server stopped successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error stopping MCP Server: {e}")
    
    async def _handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new client connections"""
        client_id = str(uuid.uuid4())
        self.logger.info(f"📱 New client connected: {client_id}")
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    mcp_message = MCPMessage(
                        id=data.get("id", str(uuid.uuid4())),
                        type=MessageType(data["type"]),
                        source=data["source"],
                        target=data.get("target"),
                        timestamp=datetime.fromisoformat(data.get("timestamp", datetime.utcnow().isoformat())),
                        data=data.get("data", {}),
                        correlation_id=data.get("correlation_id")
                    )
                    
                    # Add to message queue for processing
                    await self.message_queue.put(mcp_message)
                    
                except json.JSONDecodeError:
                    self.logger.error(f"Invalid JSON received from {client_id}")
                    await self._send_error(websocket, "Invalid JSON format")
                except ValueError as e:
                    self.logger.error(f"Invalid message type from {client_id}: {e}")
                    await self._send_error(websocket, f"Invalid message type: {e}")
                except Exception as e:
                    self.logger.error(f"Error processing message from {client_id}: {e}")
                    await self._send_error(websocket, f"Processing error: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"📱 Client disconnected: {client_id}")
        except Exception as e:
            self.logger.error(f"❌ Client handler error: {e}")
        finally:
            # Clean up agent registration
            agent_to_remove = None
            for agent_id, agent_info in self.agents.items():
                if agent_info.websocket == websocket:
                    agent_to_remove = agent_id
                    break
            
            if agent_to_remove:
                del self.agents[agent_to_remove]
                self.metrics["agents_connected"] -= 1
                self.logger.info(f"🔌 Agent unregistered: {agent_to_remove}")
    
    async def _message_processor(self):
        """Process messages from the queue"""
        while self.running:
            try:
                # Get message from queue with timeout
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                
                # Process the message
                await self._process_message(message)
                
                # Update metrics
                self.metrics["messages_processed"] += 1
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"❌ Message processor error: {e}")
                self.metrics["errors"] += 1
    
    async def _process_message(self, message: MCPMessage):
        """Process a single message"""
        try:
            # Add to history
            self.message_history.append(message)
            
            # Keep history size manageable
            if len(self.message_history) > 1000:
                self.message_history = self.message_history[-500:]
            
            # Call registered handlers
            if message.type in self.message_handlers:
                for handler in self.message_handlers[message.type]:
                    try:
                        await handler(message)
                    except Exception as e:
                        self.logger.error(f"❌ Handler error for {message.type.value}: {e}")
            
            self.logger.debug(f"📨 Processed message: {message.type.value} from {message.source}")
            
        except Exception as e:
            self.logger.error(f"❌ Error processing message: {e}")
            self.metrics["errors"] += 1
    
    async def _handle_agent_register(self, message: MCPMessage):
        """Handle agent registration"""
        try:
            agent_id = message.data.get("agent_id")
            agent_type = message.data.get("agent_type")
            capabilities = message.data.get("capabilities", [])
            
            if not agent_id or not agent_type:
                self.logger.error("Invalid agent registration data")
                return
            
            # Find the websocket for this agent
            websocket = None
            for agent_info in self.agents.values():
                if agent_info.websocket and hasattr(agent_info.websocket, 'remote_address'):
                    # This is a simplified check - in real implementation you'd need better tracking
                    websocket = agent_info.websocket
                    break
            
            if not websocket:
                self.logger.error(f"Could not find websocket for agent {agent_id}")
                return
            
            # Register the agent
            self.agents[agent_id] = AgentInfo(
                agent_id=agent_id,
                agent_type=agent_type,
                websocket=websocket,
                capabilities=capabilities,
                last_heartbeat=datetime.utcnow()
            )
            
            self.metrics["agents_connected"] += 1
            self.logger.info(f"✅ Agent registered: {agent_id} ({agent_type})")
            
            # Send confirmation
            await self._send_message(websocket, {
                "type": "registration_confirmed",
                "agent_id": agent_id,
                "server_time": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"❌ Agent registration error: {e}")
    
    async def _handle_agent_heartbeat(self, message: MCPMessage):
        """Handle agent heartbeat"""
        agent_id = message.source
        if agent_id in self.agents:
            self.agents[agent_id].last_heartbeat = datetime.utcnow()
            self.agents[agent_id].status = "connected"
    
    async def _handle_knowledge_query(self, message: MCPMessage):
        """Handle knowledge graph queries"""
        try:
            if not self.knowledge_engine:
                await self._send_error_to_agent(message.source, "Knowledge engine not available")
                return
            
            query = message.data.get("query")
            query_type = message.data.get("query_type", "search")
            
            if query_type == "search":
                results = self.knowledge_engine.search_nodes(query)
            elif query_type == "pattern":
                results = self.knowledge_engine.find_patterns(query)
            else:
                results = []
            
            # Send results back to requesting agent
            await self._send_to_agent(message.source, {
                "type": "knowledge_query_result",
                "correlation_id": message.correlation_id,
                "results": results,
                "query": query
            })
            
        except Exception as e:
            self.logger.error(f"❌ Knowledge query error: {e}")
            await self._send_error_to_agent(message.source, f"Knowledge query failed: {e}")
    
    async def _handle_knowledge_update(self, message: MCPMessage):
        """Handle knowledge graph updates"""
        try:
            if not self.knowledge_engine:
                return
            
            update_type = message.data.get("update_type")
            data = message.data.get("data")
            
            if update_type == "add_node":
                self.knowledge_engine.add_node(data)
            elif update_type == "add_edge":
                self.knowledge_engine.add_edge(data)
            elif update_type == "update_node":
                self.knowledge_engine.update_node(data)
            
            self.logger.debug(f"📚 Knowledge updated: {update_type}")
            
        except Exception as e:
            self.logger.error(f"❌ Knowledge update error: {e}")
    
    async def _handle_broadcast(self, message: MCPMessage):
        """Handle broadcast messages"""
        try:
            # Send to all connected agents except sender
            for agent_id, agent_info in self.agents.items():
                if agent_id != message.source:
                    await self._send_message(agent_info.websocket, {
                        "type": "broadcast_message",
                        "source": message.source,
                        "data": message.data,
                        "timestamp": message.timestamp.isoformat()
                    })
            
            self.logger.debug(f"📢 Broadcast sent from {message.source}")
            
        except Exception as e:
            self.logger.error(f"❌ Broadcast error: {e}")
    
    async def _handle_direct_message(self, message: MCPMessage):
        """Handle direct messages between agents"""
        try:
            target_agent = message.target
            if target_agent and target_agent in self.agents:
                await self._send_message(self.agents[target_agent].websocket, {
                    "type": "direct_message",
                    "source": message.source,
                    "data": message.data,
                    "timestamp": message.timestamp.isoformat()
                })
                
                self.logger.debug(f"📧 Direct message: {message.source} -> {target_agent}")
            else:
                await self._send_error_to_agent(message.source, f"Target agent not found: {target_agent}")
                
        except Exception as e:
            self.logger.error(f"❌ Direct message error: {e}")
    
    async def _send_message(self, websocket: WebSocketServerProtocol, data: Dict):
        """Send message to websocket"""
        try:
            await websocket.send(json.dumps(data))
        except Exception as e:
            self.logger.error(f"❌ Error sending message: {e}")
    
    async def _send_error(self, websocket: WebSocketServerProtocol, error_message: str):
        """Send error message to websocket"""
        await self._send_message(websocket, {
            "type": "error",
            "message": error_message,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def _send_to_agent(self, agent_id: str, data: Dict):
        """Send message to specific agent"""
        if agent_id in self.agents:
            await self._send_message(self.agents[agent_id].websocket, data)
    
    async def _send_error_to_agent(self, agent_id: str, error_message: str):
        """Send error message to specific agent"""
        if agent_id in self.agents:
            await self._send_error(self.agents[agent_id].websocket, error_message)
    
    async def _heartbeat_monitor(self):
        """Monitor agent heartbeats"""
        while self.running:
            try:
                current_time = datetime.utcnow()
                expired_agents = []
                
                for agent_id, agent_info in self.agents.items():
                    time_since_heartbeat = (current_time - agent_info.last_heartbeat).total_seconds()
                    
                    if time_since_heartbeat > self.heartbeat_interval * 2:
                        expired_agents.append(agent_id)
                        agent_info.status = "disconnected"
                
                # Remove expired agents
                for agent_id in expired_agents:
                    del self.agents[agent_id]
                    self.metrics["agents_connected"] -= 1
                    self.logger.warning(f"⚠️ Agent removed due to missed heartbeat: {agent_id}")
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"❌ Heartbeat monitor error: {e}")
                await asyncio.sleep(5)
    
    async def _cleanup_task(self):
        """Periodic cleanup task"""
        while self.running:
            try:
                # Clean up old message history
                if len(self.message_history) > 500:
                    self.message_history = self.message_history[-200:]
                
                # Log metrics
                self.logger.debug(f"📊 MCP Metrics: {self.metrics}")
                
                await asyncio.sleep(self.cleanup_interval)
                
            except Exception as e:
                self.logger.error(f"❌ Cleanup task error: {e}")
                await asyncio.sleep(60)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get server metrics"""
        return {
            **self.metrics,
            "connected_agents": len(self.agents),
            "message_queue_size": self.message_queue.qsize(),
            "uptime_seconds": (datetime.utcnow() - self.metrics["uptime"]).total_seconds()
        }
    
    def get_agent_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all agents"""
        status = {}
        for agent_id, agent_info in self.agents.items():
            status[agent_id] = {
                "type": agent_info.agent_type,
                "status": agent_info.status,
                "capabilities": agent_info.capabilities,
                "last_heartbeat": agent_info.last_heartbeat.isoformat(),
                "uptime": (datetime.utcnow() - agent_info.last_heartbeat).total_seconds()
            }
        return status
