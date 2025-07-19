"""
Base Agent class for the Advanced AI Trading System
Provides common functionality for all specialized agents
"""

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
import websockets
from websockets.client import WebSocketClientProtocol
import threading
import signal

from config import Config
from mcp_server import MCPMessage, MessageType


class BaseAgent(ABC):
    """
    Abstract base class for all trading agents
    Provides MCP communication, lifecycle management, and common utilities
    """
    
    def __init__(self, agent_id: str, agent_type: str, mcp_server, knowledge_engine, config: Config):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.mcp_server = mcp_server
        self.knowledge_engine = knowledge_engine
        self.config = config
        self.logger = logging.getLogger(f"Agent.{agent_id}")
        
        # Agent state
        self.running = False
        self.websocket: Optional[WebSocketClientProtocol] = None
        self.capabilities: List[str] = []
        self.message_handlers: Dict[str, Callable] = {}
        
        # Communication
        self.mcp_host = config.MCP_HOST
        self.mcp_port = config.MCP_PORT
        self.heartbeat_interval = 30  # seconds
        self.reconnect_delay = 5  # seconds
        self.max_reconnect_attempts = 10
        
        # Performance tracking
        self.metrics = {
            "messages_sent": 0,
            "messages_received": 0,
            "errors": 0,
            "uptime": datetime.utcnow()
        }
        
        # Initialize agent-specific setup
        self._setup_capabilities()
        self._setup_message_handlers()
        
        self.logger.info(f"🤖 Agent {agent_id} ({agent_type}) initialized")
    
    @abstractmethod
    def _setup_capabilities(self):
        """Setup agent capabilities - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def _setup_message_handlers(self):
        """Setup message handlers - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    async def _agent_logic(self):
        """Main agent logic - must be implemented by subclasses"""
        pass
    
    def register_message_handler(self, message_type: str, handler: Callable):
        """Register a message handler"""
        self.message_handlers[message_type] = handler
        self.logger.debug(f"Registered handler for {message_type}")
    
    async def start(self):
        """Start the agent"""
        try:
            self.logger.info(f"🚀 Starting agent {self.agent_id}")
            
            # Connect to MCP server
            await self._connect_to_mcp()
            
            # Register with MCP server
            await self._register_with_mcp()
            
            # Start agent logic
            self.running = True
            asyncio.create_task(self._agent_logic())
            asyncio.create_task(self._heartbeat_sender())
            asyncio.create_task(self._message_listener())
            
            self.logger.info(f"✅ Agent {self.agent_id} started successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start agent {self.agent_id}: {e}")
            raise
    
    async def stop(self):
        """Stop the agent"""
        try:
            self.logger.info(f"🔄 Stopping agent {self.agent_id}")
            
            self.running = False
            
            # Close websocket connection
            if self.websocket:
                await self.websocket.close()
            
            self.logger.info(f"✅ Agent {self.agent_id} stopped successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error stopping agent {self.agent_id}: {e}")
    
    async def _connect_to_mcp(self):
        """Connect to MCP server with retry logic"""
        for attempt in range(self.max_reconnect_attempts):
            try:
                uri = f"ws://{self.mcp_host}:{self.mcp_port}"
                self.websocket = await websockets.connect(uri)
                self.logger.info(f"✅ Connected to MCP server: {uri}")
                return
                
            except Exception as e:
                self.logger.warning(f"⚠️ Connection attempt {attempt + 1} failed: {e}")
                if attempt < self.max_reconnect_attempts - 1:
                    await asyncio.sleep(self.reconnect_delay)
                else:
                    raise Exception(f"Failed to connect to MCP server after {self.max_reconnect_attempts} attempts")
    
    async def _register_with_mcp(self):
        """Register with MCP server"""
        try:
            registration_message = {
                "id": str(uuid.uuid4()),
                "type": MessageType.AGENT_REGISTER.value,
                "source": self.agent_id,
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "agent_id": self.agent_id,
                    "agent_type": self.agent_type,
                    "capabilities": self.capabilities
                }
            }
            
            await self.websocket.send(json.dumps(registration_message))
            self.logger.info(f"📝 Registered with MCP server")
            
        except Exception as e:
            self.logger.error(f"❌ Registration failed: {e}")
            raise
    
    async def _heartbeat_sender(self):
        """Send periodic heartbeats to MCP server"""
        while self.running:
            try:
                heartbeat_message = {
                    "id": str(uuid.uuid4()),
                    "type": MessageType.AGENT_HEARTBEAT.value,
                    "source": self.agent_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": {
                        "status": "active",
                        "metrics": self.metrics
                    }
                }
                
                await self.websocket.send(json.dumps(heartbeat_message))
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"❌ Heartbeat error: {e}")
                await asyncio.sleep(5)
    
    async def _message_listener(self):
        """Listen for messages from MCP server"""
        while self.running:
            try:
                message = await self.websocket.recv()
                data = json.loads(message)
                
                # Handle the message
                await self._handle_message(data)
                
                self.metrics["messages_received"] += 1
                
            except websockets.exceptions.ConnectionClosed:
                self.logger.warning("🔌 Connection to MCP server lost")
                if self.running:
                    await self._reconnect()
            except Exception as e:
                self.logger.error(f"❌ Message listener error: {e}")
                self.metrics["errors"] += 1
                await asyncio.sleep(1)
    
    async def _handle_message(self, data: Dict):
        """Handle incoming message"""
        try:
            message_type = data.get("type")
            
            if message_type in self.message_handlers:
                await self.message_handlers[message_type](data)
            else:
                self.logger.debug(f"📨 Unhandled message type: {message_type}")
                
        except Exception as e:
            self.logger.error(f"❌ Message handling error: {e}")
    
    async def _reconnect(self):
        """Reconnect to MCP server"""
        try:
            self.logger.info("🔄 Attempting to reconnect to MCP server...")
            await self._connect_to_mcp()
            await self._register_with_mcp()
            self.logger.info("✅ Reconnected to MCP server")
            
        except Exception as e:
            self.logger.error(f"❌ Reconnection failed: {e}")
            await asyncio.sleep(self.reconnect_delay)
    
    async def send_message(self, message_type: MessageType, data: Dict, target: Optional[str] = None):
        """Send message via MCP"""
        try:
            message = {
                "id": str(uuid.uuid4()),
                "type": message_type.value,
                "source": self.agent_id,
                "target": target,
                "timestamp": datetime.utcnow().isoformat(),
                "data": data
            }
            
            await self.websocket.send(json.dumps(message))
            self.metrics["messages_sent"] += 1
            
            self.logger.debug(f"📤 Sent message: {message_type.value}")
            
        except Exception as e:
            self.logger.error(f"❌ Error sending message: {e}")
            self.metrics["errors"] += 1
    
    async def broadcast_message(self, data: Dict):
        """Broadcast message to all agents"""
        await self.send_message(MessageType.BROADCAST, data)
    
    async def send_direct_message(self, target_agent: str, data: Dict):
        """Send direct message to specific agent"""
        await self.send_message(MessageType.DIRECT_MESSAGE, data, target=target_agent)
    
    async def query_knowledge(self, query: str, query_type: str = "search") -> List[Dict]:
        """Query the knowledge engine"""
        try:
            correlation_id = str(uuid.uuid4())
            
            await self.send_message(MessageType.KNOWLEDGE_QUERY, {
                "query": query,
                "query_type": query_type,
                "correlation_id": correlation_id
            })
            
            # Wait for response (simplified - in real implementation would use async response handling)
            # For now, return empty list
            return []
            
        except Exception as e:
            self.logger.error(f"❌ Knowledge query error: {e}")
            return []
    
    async def update_knowledge(self, update_type: str, data: Dict):
        """Update the knowledge engine"""
        try:
            await self.send_message(MessageType.KNOWLEDGE_UPDATE, {
                "update_type": update_type,
                "data": data
            })
            
        except Exception as e:
            self.logger.error(f"❌ Knowledge update error: {e}")
    
    async def send_trading_signal(self, signal: Dict):
        """Send trading signal"""
        await self.send_message(MessageType.TRADING_SIGNAL, signal)
    
    async def send_order_request(self, order: Dict):
        """Send order request"""
        await self.send_message(MessageType.ORDER_REQUEST, order)
    
    async def send_system_event(self, event_type: str, severity: str, message: str, metadata: Dict = None):
        """Send system event"""
        await self.send_message(MessageType.SYSTEM_EVENT, {
            "event_type": event_type,
            "severity": severity,
            "message": message,
            "metadata": metadata or {}
        })
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics"""
        return {
            **self.metrics,
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "uptime_seconds": (datetime.utcnow() - self.metrics["uptime"]).total_seconds(),
            "running": self.running
        }
    
    def log_performance(self, operation: str, duration: float, success: bool = True):
        """Log performance metrics"""
        self.logger.debug(f"⚡ {operation}: {duration:.3f}s {'✅' if success else '❌'}")
    
    async def wait_for_shutdown(self):
        """Wait for shutdown signal"""
        while self.running:
            await asyncio.sleep(1)
