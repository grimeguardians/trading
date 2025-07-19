import asyncio
import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import websockets
import uuid
from dataclasses import dataclass, asdict
from enum import Enum

from config import Config
from agents.base_agent import AgentMessage

class MCPMessageType(Enum):
    REGISTER = "register"
    UNREGISTER = "unregister"
    MESSAGE = "message"
    BROADCAST = "broadcast"
    PING = "ping"
    PONG = "pong"
    ERROR = "error"

@dataclass
class MCPMessage:
    """MCP protocol message structure"""
    id: str
    type: MCPMessageType
    sender: str
    receiver: Optional[str]
    content: Dict[str, Any]
    timestamp: datetime
    correlation_id: Optional[str] = None

class MCPServer:
    """Model Context Protocol server for agent communication"""
    
    def __init__(self, config: Config):
        self.config = config
        self.mcp_config = config.mcp_config
        
        # Server configuration
        self.host = self.mcp_config['host']
        self.port = self.mcp_config['port']
        self.max_agents = self.mcp_config['max_agents']
        self.message_queue_size = self.mcp_config['message_queue_size']
        self.agent_timeout = self.mcp_config['agent_timeout']
        
        # Connected agents
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.websocket_connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        
        # Message queues
        self.message_queues: Dict[str, asyncio.Queue] = {}
        self.broadcast_queue: asyncio.Queue = asyncio.Queue()
        
        # Server state
        self.is_running = False
        self.server = None
        
        # Statistics
        self.stats = {
            'messages_processed': 0,
            'agents_connected': 0,
            'broadcasts_sent': 0,
            'errors': 0,
            'uptime': 0
        }
        
        self.logger = logging.getLogger("MCPServer")
    
    async def start(self):
        """Start the MCP server"""
        try:
            self.logger.info(f"Starting MCP server on {self.host}:{self.port}")
            
            # Start WebSocket server
            self.server = await websockets.serve(
                self.handle_connection,
                self.host,
                self.port,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )
            
            self.is_running = True
            
            # Start background tasks
            asyncio.create_task(self._process_broadcasts())
            asyncio.create_task(self._monitor_agents())
            asyncio.create_task(self._cleanup_stale_connections())
            
            self.logger.info("MCP server started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start MCP server: {e}")
            raise
    
    async def stop(self):
        """Stop the MCP server"""
        try:
            self.logger.info("Stopping MCP server...")
            
            self.is_running = False
            
            # Close all connections
            for agent_id, websocket in self.websocket_connections.items():
                await websocket.close()
                self.logger.info(f"Closed connection for agent {agent_id}")
            
            # Close server
            if self.server:
                self.server.close()
                await self.server.wait_closed()
            
            self.logger.info("MCP server stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping MCP server: {e}")
    
    async def handle_connection(self, websocket, path):
        """Handle new WebSocket connection"""
        agent_id = None
        
        try:
            self.logger.info(f"New connection from {websocket.remote_address}")
            
            # Wait for registration message
            registration_message = await asyncio.wait_for(
                websocket.recv(),
                timeout=30.0
            )
            
            message_data = json.loads(registration_message)
            
            if message_data.get('type') != MCPMessageType.REGISTER.value:
                await websocket.close(code=1008, reason="Expected registration message")
                return
            
            agent_id = message_data.get('sender')
            
            if not agent_id:
                await websocket.close(code=1008, reason="Agent ID required")
                return
            
            # Check if agent already connected
            if agent_id in self.agents:
                await websocket.close(code=1008, reason="Agent already connected")
                return
            
            # Check connection limit
            if len(self.agents) >= self.max_agents:
                await websocket.close(code=1008, reason="Maximum agents reached")
                return
            
            # Register agent
            await self._register_agent(agent_id, websocket, message_data.get('content', {}))
            
            # Send registration confirmation
            await self._send_message(websocket, MCPMessage(
                id=str(uuid.uuid4()),
                type=MCPMessageType.REGISTER,
                sender="server",
                receiver=agent_id,
                content={"status": "registered", "agent_id": agent_id},
                timestamp=datetime.now()
            ))
            
            # Handle messages
            await self._handle_agent_messages(agent_id, websocket)
            
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Connection closed for agent {agent_id}")
        except asyncio.TimeoutError:
            self.logger.warning(f"Registration timeout for connection from {websocket.remote_address}")
        except Exception as e:
            self.logger.error(f"Error handling connection: {e}")
        finally:
            if agent_id:
                await self._unregister_agent(agent_id)
    
    async def _register_agent(self, agent_id: str, websocket: websockets.WebSocketServerProtocol, 
                            agent_info: Dict[str, Any]):
        """Register a new agent"""
        try:
            self.agents[agent_id] = {
                'agent_id': agent_id,
                'websocket': websocket,
                'info': agent_info,
                'registered_at': datetime.now(),
                'last_ping': datetime.now(),
                'message_count': 0,
                'status': 'active'
            }
            
            self.websocket_connections[agent_id] = websocket
            
            # Create message queue for agent
            self.message_queues[agent_id] = asyncio.Queue(maxsize=self.message_queue_size)
            
            self.stats['agents_connected'] += 1
            
            self.logger.info(f"Agent {agent_id} registered successfully")
            
        except Exception as e:
            self.logger.error(f"Error registering agent {agent_id}: {e}")
            raise
    
    async def _unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        try:
            if agent_id in self.agents:
                del self.agents[agent_id]
                self.logger.info(f"Agent {agent_id} unregistered")
            
            if agent_id in self.websocket_connections:
                del self.websocket_connections[agent_id]
            
            if agent_id in self.message_queues:
                del self.message_queues[agent_id]
            
        except Exception as e:
            self.logger.error(f"Error unregistering agent {agent_id}: {e}")
    
    async def _handle_agent_messages(self, agent_id: str, websocket: websockets.WebSocketServerProtocol):
        """Handle messages from an agent"""
        try:
            while self.is_running:
                try:
                    # Wait for message
                    raw_message = await asyncio.wait_for(
                        websocket.recv(),
                        timeout=self.agent_timeout
                    )
                    
                    # Parse message
                    message_data = json.loads(raw_message)
                    
                    # Create MCP message
                    mcp_message = MCPMessage(
                        id=message_data.get('id', str(uuid.uuid4())),
                        type=MCPMessageType(message_data.get('type')),
                        sender=agent_id,
                        receiver=message_data.get('receiver'),
                        content=message_data.get('content', {}),
                        timestamp=datetime.now(),
                        correlation_id=message_data.get('correlation_id')
                    )
                    
                    # Process message
                    await self._process_message(mcp_message)
                    
                    # Update agent stats
                    self.agents[agent_id]['message_count'] += 1
                    self.agents[agent_id]['last_ping'] = datetime.now()
                    
                except asyncio.TimeoutError:
                    # Send ping
                    await self._send_ping(agent_id, websocket)
                    
                except websockets.exceptions.ConnectionClosed:
                    break
                    
                except json.JSONDecodeError:
                    self.logger.error(f"Invalid JSON from agent {agent_id}")
                    await self._send_error(websocket, "Invalid JSON format")
                    
                except Exception as e:
                    self.logger.error(f"Error processing message from agent {agent_id}: {e}")
                    await self._send_error(websocket, str(e))
                    
        except Exception as e:
            self.logger.error(f"Error handling messages for agent {agent_id}: {e}")
    
    async def _process_message(self, message: MCPMessage):
        """Process an MCP message"""
        try:
            self.stats['messages_processed'] += 1
            
            if message.type == MCPMessageType.MESSAGE:
                # Route message to specific agent
                if message.receiver and message.receiver in self.agents:
                    await self._route_message(message)
                else:
                    await self._send_error(
                        self.websocket_connections[message.sender],
                        f"Agent {message.receiver} not found"
                    )
            
            elif message.type == MCPMessageType.BROADCAST:
                # Add to broadcast queue
                await self.broadcast_queue.put(message)
                
            elif message.type == MCPMessageType.PING:
                # Send pong
                await self._send_pong(message.sender)
                
            elif message.type == MCPMessageType.PONG:
                # Update last ping time
                if message.sender in self.agents:
                    self.agents[message.sender]['last_ping'] = datetime.now()
                    
            else:
                self.logger.warning(f"Unknown message type: {message.type}")
                
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            self.stats['errors'] += 1
    
    async def _route_message(self, message: MCPMessage):
        """Route message to target agent"""
        try:
            if message.receiver not in self.message_queues:
                self.logger.error(f"Message queue not found for agent {message.receiver}")
                return
            
            # Add to agent's message queue
            try:
                self.message_queues[message.receiver].put_nowait(message)
                
                # Send message to agent
                websocket = self.websocket_connections[message.receiver]
                await self._send_message(websocket, message)
                
            except asyncio.QueueFull:
                self.logger.warning(f"Message queue full for agent {message.receiver}")
                await self._send_error(
                    self.websocket_connections[message.sender],
                    f"Message queue full for {message.receiver}"
                )
                
        except Exception as e:
            self.logger.error(f"Error routing message: {e}")
    
    async def _send_message(self, websocket: websockets.WebSocketServerProtocol, message: MCPMessage):
        """Send message to WebSocket"""
        try:
            message_data = asdict(message)
            message_data['type'] = message.type.value
            message_data['timestamp'] = message.timestamp.isoformat()
            
            await websocket.send(json.dumps(message_data))
            
        except Exception as e:
            self.logger.error(f"Error sending message: {e}")
    
    async def _send_error(self, websocket: websockets.WebSocketServerProtocol, error_message: str):
        """Send error message"""
        try:
            error_msg = MCPMessage(
                id=str(uuid.uuid4()),
                type=MCPMessageType.ERROR,
                sender="server",
                receiver=None,
                content={"error": error_message},
                timestamp=datetime.now()
            )
            
            await self._send_message(websocket, error_msg)
            
        except Exception as e:
            self.logger.error(f"Error sending error message: {e}")
    
    async def _send_ping(self, agent_id: str, websocket: websockets.WebSocketServerProtocol):
        """Send ping to agent"""
        try:
            ping_msg = MCPMessage(
                id=str(uuid.uuid4()),
                type=MCPMessageType.PING,
                sender="server",
                receiver=agent_id,
                content={"timestamp": datetime.now().isoformat()},
                timestamp=datetime.now()
            )
            
            await self._send_message(websocket, ping_msg)
            
        except Exception as e:
            self.logger.error(f"Error sending ping: {e}")
    
    async def _send_pong(self, agent_id: str):
        """Send pong to agent"""
        try:
            if agent_id not in self.websocket_connections:
                return
            
            websocket = self.websocket_connections[agent_id]
            
            pong_msg = MCPMessage(
                id=str(uuid.uuid4()),
                type=MCPMessageType.PONG,
                sender="server",
                receiver=agent_id,
                content={"timestamp": datetime.now().isoformat()},
                timestamp=datetime.now()
            )
            
            await self._send_message(websocket, pong_msg)
            
        except Exception as e:
            self.logger.error(f"Error sending pong: {e}")
    
    async def _process_broadcasts(self):
        """Process broadcast messages"""
        while self.is_running:
            try:
                # Get broadcast message
                broadcast_message = await asyncio.wait_for(
                    self.broadcast_queue.get(),
                    timeout=1.0
                )
                
                # Send to all connected agents except sender
                for agent_id, websocket in self.websocket_connections.items():
                    if agent_id != broadcast_message.sender:
                        try:
                            # Create copy with correct receiver
                            broadcast_copy = MCPMessage(
                                id=str(uuid.uuid4()),
                                type=MCPMessageType.MESSAGE,
                                sender=broadcast_message.sender,
                                receiver=agent_id,
                                content=broadcast_message.content,
                                timestamp=datetime.now(),
                                correlation_id=broadcast_message.id
                            )
                            
                            await self._send_message(websocket, broadcast_copy)
                            
                        except Exception as e:
                            self.logger.error(f"Error sending broadcast to {agent_id}: {e}")
                
                self.stats['broadcasts_sent'] += 1
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error processing broadcasts: {e}")
                await asyncio.sleep(1)
    
    async def _monitor_agents(self):
        """Monitor agent health"""
        while self.is_running:
            try:
                current_time = datetime.now()
                stale_agents = []
                
                for agent_id, agent_info in self.agents.items():
                    last_ping = agent_info['last_ping']
                    time_since_ping = (current_time - last_ping).total_seconds()
                    
                    if time_since_ping > self.agent_timeout * 2:
                        stale_agents.append(agent_id)
                        agent_info['status'] = 'stale'
                    elif time_since_ping > self.agent_timeout:
                        agent_info['status'] = 'inactive'
                    else:
                        agent_info['status'] = 'active'
                
                # Remove stale agents
                for agent_id in stale_agents:
                    self.logger.warning(f"Removing stale agent: {agent_id}")
                    await self._unregister_agent(agent_id)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error monitoring agents: {e}")
                await asyncio.sleep(30)
    
    async def _cleanup_stale_connections(self):
        """Clean up stale connections"""
        while self.is_running:
            try:
                # Clean up closed connections
                closed_agents = []
                
                for agent_id, websocket in self.websocket_connections.items():
                    if websocket.closed:
                        closed_agents.append(agent_id)
                
                for agent_id in closed_agents:
                    await self._unregister_agent(agent_id)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error cleaning up connections: {e}")
                await asyncio.sleep(60)
    
    # Public API methods
    async def send_message_to_agent(self, sender_id: str, receiver_id: str, 
                                   message_type: str, content: Dict[str, Any]) -> bool:
        """Send message to specific agent"""
        try:
            if receiver_id not in self.agents:
                return False
            
            message = MCPMessage(
                id=str(uuid.uuid4()),
                type=MCPMessageType.MESSAGE,
                sender=sender_id,
                receiver=receiver_id,
                content={
                    'message_type': message_type,
                    'content': content
                },
                timestamp=datetime.now()
            )
            
            await self._route_message(message)
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending message to agent: {e}")
            return False
    
    async def broadcast_message(self, sender_id: str, message_type: str, 
                              content: Dict[str, Any]) -> bool:
        """Broadcast message to all agents"""
        try:
            broadcast_message = MCPMessage(
                id=str(uuid.uuid4()),
                type=MCPMessageType.BROADCAST,
                sender=sender_id,
                receiver=None,
                content={
                    'message_type': message_type,
                    'content': content
                },
                timestamp=datetime.now()
            )
            
            await self.broadcast_queue.put(broadcast_message)
            return True
            
        except Exception as e:
            self.logger.error(f"Error broadcasting message: {e}")
            return False
    
    def get_connected_agents(self) -> List[Dict[str, Any]]:
        """Get list of connected agents"""
        return [
            {
                'agent_id': agent_id,
                'info': agent_info['info'],
                'status': agent_info['status'],
                'registered_at': agent_info['registered_at'].isoformat(),
                'last_ping': agent_info['last_ping'].isoformat(),
                'message_count': agent_info['message_count']
            }
            for agent_id, agent_info in self.agents.items()
        ]
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get server statistics"""
        return {
            'is_running': self.is_running,
            'connected_agents': len(self.agents),
            'stats': self.stats,
            'config': {
                'host': self.host,
                'port': self.port,
                'max_agents': self.max_agents,
                'agent_timeout': self.agent_timeout
            }
        }
