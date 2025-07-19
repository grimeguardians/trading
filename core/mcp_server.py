"""
Model Context Protocol (MCP) Server for Multi-Agent Coordination
Enhanced for AI Trading Agent with Digital Brain integration
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import uuid
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import websockets
import sqlite3
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentRole(Enum):
    """Predefined agent roles in the trading system"""
    MARKET_ANALYST = "market_analyst"
    SENTIMENT_ANALYZER = "sentiment_analyzer"
    RISK_MANAGER = "risk_manager"
    STRATEGY_COORDINATOR = "strategy_coordinator"
    EXECUTION_MANAGER = "execution_manager"
    PATTERN_RECOGNIZER = "pattern_recognizer"
    NEWS_PROCESSOR = "news_processor"
    FIBONACCI_ANALYZER = "fibonacci_analyzer"
    DIGITAL_BRAIN = "digital_brain"

class MessageType(Enum):
    """Message types for inter-agent communication"""
    QUERY = "query"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    COMMAND = "command"
    DATA_SYNC = "data_sync"
    KNOWLEDGE_UPDATE = "knowledge_update"

@dataclass
class MCPMessage:
    """Message structure for MCP communication"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.QUERY
    sender: str = ""
    recipient: str = ""
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 1  # 1=low, 2=medium, 3=high, 4=critical
    requires_response: bool = False
    conversation_id: Optional[str] = None

@dataclass
class Agent:
    """Agent representation in the MCP system"""
    id: str
    role: AgentRole
    capabilities: List[str]
    status: str = "active"
    last_heartbeat: datetime = field(default_factory=datetime.now)
    message_queue: List[MCPMessage] = field(default_factory=list)
    knowledge_context: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)

class MCPServer:
    """
    Model Context Protocol Server for coordinating multiple AI agents
    Inspired by 12-factor app principles with stateless processes
    """
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.agents: Dict[str, Agent] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.knowledge_graph = {}
        self.conversation_contexts: Dict[str, Dict[str, Any]] = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.running = False
        
        # Initialize SQLite for message persistence
        self.db_path = Path("mcp_messages.db")
        self._init_database()
        
        # Register default message handlers
        self._register_default_handlers()
    
    def _init_database(self):
        """Initialize SQLite database for message persistence"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    sender TEXT NOT NULL,
                    recipient TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    priority INTEGER NOT NULL,
                    conversation_id TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_states (
                    agent_id TEXT PRIMARY KEY,
                    role TEXT NOT NULL,
                    status TEXT NOT NULL,
                    last_heartbeat TIMESTAMP NOT NULL,
                    knowledge_context TEXT,
                    performance_metrics TEXT
                )
            """)
    
    def _register_default_handlers(self):
        """Register default message handlers"""
        self.message_handlers.update({
            MessageType.QUERY.value: self._handle_query,
            MessageType.RESPONSE.value: self._handle_response,
            MessageType.NOTIFICATION.value: self._handle_notification,
            MessageType.COMMAND.value: self._handle_command,
            MessageType.DATA_SYNC.value: self._handle_data_sync,
            MessageType.KNOWLEDGE_UPDATE.value: self._handle_knowledge_update
        })
    
    async def start_server(self):
        """Start the MCP server"""
        self.running = True
        logger.info(f"Starting MCP Server on {self.host}:{self.port}")
        
        # Start background tasks
        asyncio.create_task(self._heartbeat_monitor())
        asyncio.create_task(self._message_processor())
        
        # Start WebSocket server
        start_server = websockets.serve(
            self._handle_websocket_connection,
            self.host,
            self.port
        )
        
        await start_server
        logger.info("MCP Server started successfully")
    
    async def _handle_websocket_connection(self, websocket, path):
        """Handle WebSocket connections from agents"""
        agent_id = None
        try:
            async for message in websocket:
                data = json.loads(message)
                
                if data.get("type") == "register":
                    agent_id = await self._register_agent(data, websocket)
                elif data.get("type") == "message":
                    await self._process_message(data)
                elif data.get("type") == "heartbeat":
                    await self._update_heartbeat(data.get("agent_id"))
                    
        except websockets.exceptions.ConnectionClosed:
            if agent_id:
                await self._deregister_agent(agent_id)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
    
    async def _register_agent(self, data: Dict[str, Any], websocket) -> str:
        """Register a new agent with the MCP server"""
        agent_id = data.get("agent_id", str(uuid.uuid4()))
        role = AgentRole(data.get("role", "market_analyst"))
        capabilities = data.get("capabilities", [])
        
        agent = Agent(
            id=agent_id,
            role=role,
            capabilities=capabilities
        )
        
        self.agents[agent_id] = agent
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO agent_states 
                (agent_id, role, status, last_heartbeat, knowledge_context, performance_metrics)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                agent_id,
                role.value,
                "active",
                datetime.now(),
                json.dumps(agent.knowledge_context),
                json.dumps(agent.performance_metrics)
            ))
        
        logger.info(f"Agent registered: {agent_id} ({role.value})")
        
        # Send welcome message
        welcome_msg = MCPMessage(
            type=MessageType.NOTIFICATION,
            sender="mcp_server",
            recipient=agent_id,
            content={"message": "Welcome to MCP Server", "agents": list(self.agents.keys())}
        )
        
        await self._send_message(welcome_msg)
        return agent_id
    
    async def _deregister_agent(self, agent_id: str):
        """Deregister an agent"""
        if agent_id in self.agents:
            self.agents[agent_id].status = "inactive"
            logger.info(f"Agent deregistered: {agent_id}")
    
    async def _process_message(self, data: Dict[str, Any]):
        """Process incoming messages from agents"""
        message = MCPMessage(
            id=data.get("id", str(uuid.uuid4())),
            type=MessageType(data.get("type", "query")),
            sender=data.get("sender", ""),
            recipient=data.get("recipient", ""),
            content=data.get("content", {}),
            priority=data.get("priority", 1),
            requires_response=data.get("requires_response", False),
            conversation_id=data.get("conversation_id")
        )
        
        # Store message in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO messages 
                (id, type, sender, recipient, content, timestamp, priority, conversation_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                message.id,
                message.type.value,
                message.sender,
                message.recipient,
                json.dumps(message.content),
                message.timestamp,
                message.priority,
                message.conversation_id
            ))
        
        # Route message to appropriate handler
        handler = self.message_handlers.get(message.type.value)
        if handler:
            await handler(message)
        else:
            logger.warning(f"No handler for message type: {message.type.value}")
    
    async def _handle_query(self, message: MCPMessage):
        """Handle query messages between agents"""
        recipient = message.recipient
        
        if recipient == "broadcast":
            # Broadcast to all agents
            for agent_id, agent in self.agents.items():
                if agent_id != message.sender and agent.status == "active":
                    agent.message_queue.append(message)
        elif recipient in self.agents:
            # Direct message to specific agent
            self.agents[recipient].message_queue.append(message)
        else:
            # Unknown recipient, send error back
            error_msg = MCPMessage(
                type=MessageType.RESPONSE,
                sender="mcp_server",
                recipient=message.sender,
                content={"error": f"Unknown recipient: {recipient}"},
                conversation_id=message.conversation_id
            )
            await self._send_message(error_msg)
    
    async def _handle_response(self, message: MCPMessage):
        """Handle response messages"""
        if message.recipient in self.agents:
            self.agents[message.recipient].message_queue.append(message)
    
    async def _handle_notification(self, message: MCPMessage):
        """Handle notification messages"""
        # Log important notifications
        logger.info(f"Notification from {message.sender}: {message.content}")
        
        # Update knowledge graph if relevant
        if message.type == MessageType.KNOWLEDGE_UPDATE:
            self._update_knowledge_graph(message.content)
    
    async def _handle_command(self, message: MCPMessage):
        """Handle command messages"""
        command = message.content.get("command")
        
        if command == "shutdown":
            logger.info("Received shutdown command")
            self.running = False
        elif command == "list_agents":
            response = MCPMessage(
                type=MessageType.RESPONSE,
                sender="mcp_server",
                recipient=message.sender,
                content={"agents": [{"id": a.id, "role": a.role.value, "status": a.status} 
                                  for a in self.agents.values()]},
                conversation_id=message.conversation_id
            )
            await self._send_message(response)
    
    async def _handle_data_sync(self, message: MCPMessage):
        """Handle data synchronization between agents"""
        sync_type = message.content.get("sync_type")
        
        if sync_type == "market_data":
            # Broadcast market data to relevant agents
            for agent_id, agent in self.agents.items():
                if agent.role in [AgentRole.MARKET_ANALYST, AgentRole.STRATEGY_COORDINATOR]:
                    agent.message_queue.append(message)
        elif sync_type == "risk_metrics":
            # Send to risk manager
            risk_agents = [a for a in self.agents.values() if a.role == AgentRole.RISK_MANAGER]
            for agent in risk_agents:
                agent.message_queue.append(message)
    
    async def _handle_knowledge_update(self, message: MCPMessage):
        """Handle knowledge graph updates from Digital Brain"""
        knowledge_data = message.content.get("knowledge_data", {})
        self._update_knowledge_graph(knowledge_data)
        
        # Notify relevant agents about knowledge update
        for agent_id, agent in self.agents.items():
            if "knowledge_processing" in agent.capabilities:
                update_msg = MCPMessage(
                    type=MessageType.NOTIFICATION,
                    sender="mcp_server",
                    recipient=agent_id,
                    content={"message": "Knowledge graph updated", "data": knowledge_data}
                )
                agent.message_queue.append(update_msg)
    
    def _update_knowledge_graph(self, knowledge_data: Dict[str, Any]):
        """Update the central knowledge graph"""
        # This would integrate with your existing knowledge_engine.py
        for key, value in knowledge_data.items():
            self.knowledge_graph[key] = value
        
        logger.info(f"Knowledge graph updated with {len(knowledge_data)} items")
    
    async def _send_message(self, message: MCPMessage):
        """Send message to specific agent or broadcast"""
        # This would be implemented with actual WebSocket sending
        logger.info(f"Sending message: {message.sender} -> {message.recipient}")
    
    async def _heartbeat_monitor(self):
        """Monitor agent heartbeats and remove inactive agents"""
        while self.running:
            current_time = datetime.now()
            inactive_agents = []
            
            for agent_id, agent in self.agents.items():
                if (current_time - agent.last_heartbeat).seconds > 300:  # 5 minutes timeout
                    inactive_agents.append(agent_id)
            
            for agent_id in inactive_agents:
                await self._deregister_agent(agent_id)
            
            await asyncio.sleep(60)  # Check every minute
    
    async def _message_processor(self):
        """Process queued messages for agents"""
        while self.running:
            for agent_id, agent in self.agents.items():
                if agent.message_queue and agent.status == "active":
                    # Process messages in priority order
                    agent.message_queue.sort(key=lambda x: x.priority, reverse=True)
                    
                    # Process up to 10 messages per cycle
                    for _ in range(min(10, len(agent.message_queue))):
                        message = agent.message_queue.pop(0)
                        await self._deliver_message(agent_id, message)
            
            await asyncio.sleep(1)  # Process every second
    
    async def _deliver_message(self, agent_id: str, message: MCPMessage):
        """Deliver message to specific agent"""
        # This would implement actual message delivery via WebSocket
        logger.info(f"Delivering message to {agent_id}: {message.content}")
    
    async def _update_heartbeat(self, agent_id: str):
        """Update agent heartbeat"""
        if agent_id in self.agents:
            self.agents[agent_id].last_heartbeat = datetime.now()
    
    async def shutdown(self):
        """Gracefully shutdown the MCP server"""
        logger.info("Shutting down MCP Server")
        self.running = False
        self.executor.shutdown(wait=True)

# Factory function for creating MCP server
def create_mcp_server(host: str = "localhost", port: int = 8765) -> MCPServer:
    """Create and configure MCP server instance"""
    return MCPServer(host, port)

if __name__ == "__main__":
    # Example usage
    server = create_mcp_server()
    asyncio.run(server.start_server())