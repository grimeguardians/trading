"""
Agent Communication Module for MCP
Handles inter-agent communication and message routing
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

from agents.base_agent import BaseAgent, AgentMessage, MessageType

logger = logging.getLogger(__name__)

class CommunicationProtocol(str, Enum):
    DIRECT = "direct"
    BROADCAST = "broadcast"
    PUBSUB = "pubsub"
    REQUEST_RESPONSE = "request_response"

@dataclass
class MessageRoute:
    """Message routing information"""
    source: str
    destination: str
    protocol: CommunicationProtocol
    priority: int = 0
    ttl: int = 300  # Time to live in seconds
    retry_count: int = 3

@dataclass
class CommunicationMetrics:
    """Communication metrics"""
    messages_sent: int = 0
    messages_received: int = 0
    messages_failed: int = 0
    average_latency: float = 0.0
    success_rate: float = 1.0
    last_activity: datetime = None

class AgentCommunicationManager:
    """Manages communication between agents"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.topic_subscriptions: Dict[str, List[str]] = {}  # topic -> agent_ids
        self.message_handlers: Dict[str, Callable] = {}
        self.metrics: Dict[str, CommunicationMetrics] = {}
        self.is_running = False
        self.logger = logging.getLogger("AgentCommunication")
        
        # Message routing
        self.routing_table: Dict[str, List[MessageRoute]] = {}
        self.message_history: List[AgentMessage] = []
        self.max_history_size = 1000
        
        # Performance monitoring
        self.pending_messages: Dict[str, Dict[str, Any]] = {}
        self.failed_messages: List[Dict[str, Any]] = []
        
    async def start(self):
        """Start the communication manager"""
        try:
            self.is_running = True
            
            # Start message processing loop
            asyncio.create_task(self._process_message_queue())
            
            # Start metrics collection
            asyncio.create_task(self._collect_metrics())
            
            # Start cleanup task
            asyncio.create_task(self._cleanup_old_data())
            
            self.logger.info("🔄 Agent Communication Manager started")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start communication manager: {e}")
            raise
            
    async def stop(self):
        """Stop the communication manager"""
        try:
            self.is_running = False
            
            # Clear message queue
            while not self.message_queue.empty():
                try:
                    self.message_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                    
            self.logger.info("🛑 Agent Communication Manager stopped")
            
        except Exception as e:
            self.logger.error(f"❌ Error stopping communication manager: {e}")
            
    def register_agent(self, agent: BaseAgent):
        """Register an agent for communication"""
        try:
            agent_id = agent.agent_id
            self.agents[agent_id] = agent
            self.metrics[agent_id] = CommunicationMetrics()
            
            # Subscribe to agent's topics
            for topic in agent.subscribed_topics:
                self.subscribe_to_topic(agent_id, topic)
                
            self.logger.info(f"📋 Registered agent: {agent_id}")
            
        except Exception as e:
            self.logger.error(f"❌ Agent registration error: {e}")
            
    def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        try:
            if agent_id in self.agents:
                # Unsubscribe from all topics
                for topic, subscribers in self.topic_subscriptions.items():
                    if agent_id in subscribers:
                        subscribers.remove(agent_id)
                        
                # Remove agent
                del self.agents[agent_id]
                del self.metrics[agent_id]
                
                self.logger.info(f"📋 Unregistered agent: {agent_id}")
                
        except Exception as e:
            self.logger.error(f"❌ Agent unregistration error: {e}")
            
    def subscribe_to_topic(self, agent_id: str, topic: str):
        """Subscribe agent to a topic"""
        try:
            if topic not in self.topic_subscriptions:
                self.topic_subscriptions[topic] = []
                
            if agent_id not in self.topic_subscriptions[topic]:
                self.topic_subscriptions[topic].append(agent_id)
                
            self.logger.debug(f"🔔 Agent {agent_id} subscribed to topic: {topic}")
            
        except Exception as e:
            self.logger.error(f"❌ Topic subscription error: {e}")
            
    def unsubscribe_from_topic(self, agent_id: str, topic: str):
        """Unsubscribe agent from a topic"""
        try:
            if topic in self.topic_subscriptions:
                if agent_id in self.topic_subscriptions[topic]:
                    self.topic_subscriptions[topic].remove(agent_id)
                    
            self.logger.debug(f"🔕 Agent {agent_id} unsubscribed from topic: {topic}")
            
        except Exception as e:
            self.logger.error(f"❌ Topic unsubscription error: {e}")
            
    async def send_message(self, message: AgentMessage, 
                          protocol: CommunicationProtocol = CommunicationProtocol.DIRECT):
        """Send message to agent(s)"""
        try:
            # Add message to queue
            await self.message_queue.put({
                "message": message,
                "protocol": protocol,
                "timestamp": datetime.now(),
                "retry_count": 0
            })
            
            # Update metrics
            if message.sender in self.metrics:
                self.metrics[message.sender].messages_sent += 1
                
            self.logger.debug(f"📤 Queued message: {message.sender} -> {message.receiver}")
            
        except Exception as e:
            self.logger.error(f"❌ Send message error: {e}")
            
    async def broadcast_message(self, message: AgentMessage, topic: str = None):
        """Broadcast message to all agents or topic subscribers"""
        try:
            if topic:
                # Broadcast to topic subscribers
                subscribers = self.topic_subscriptions.get(topic, [])
                for agent_id in subscribers:
                    if agent_id != message.sender:  # Don't send to self
                        broadcast_message = AgentMessage(
                            id=str(uuid.uuid4()),
                            sender=message.sender,
                            receiver=agent_id,
                            message_type=message.message_type,
                            content=message.content,
                            timestamp=datetime.now(),
                            correlation_id=message.id
                        )
                        await self.send_message(broadcast_message, CommunicationProtocol.BROADCAST)
            else:
                # Broadcast to all agents
                for agent_id in self.agents.keys():
                    if agent_id != message.sender:  # Don't send to self
                        broadcast_message = AgentMessage(
                            id=str(uuid.uuid4()),
                            sender=message.sender,
                            receiver=agent_id,
                            message_type=message.message_type,
                            content=message.content,
                            timestamp=datetime.now(),
                            correlation_id=message.id
                        )
                        await self.send_message(broadcast_message, CommunicationProtocol.BROADCAST)
                        
            self.logger.debug(f"📡 Broadcasted message from {message.sender} to topic: {topic}")
            
        except Exception as e:
            self.logger.error(f"❌ Broadcast message error: {e}")
            
    async def request_response(self, sender_id: str, receiver_id: str, 
                             request_content: Dict[str, Any], timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        """Send request and wait for response"""
        try:
            # Create request message
            request_message = AgentMessage(
                id=str(uuid.uuid4()),
                sender=sender_id,
                receiver=receiver_id,
                message_type=MessageType.QUERY,
                content=request_content,
                timestamp=datetime.now(),
                expires_at=datetime.now() + timedelta(seconds=timeout)
            )
            
            # Create response future
            response_future = asyncio.Future()
            
            # Store pending request
            self.pending_messages[request_message.id] = {
                "future": response_future,
                "timestamp": datetime.now(),
                "timeout": timeout
            }
            
            # Send request
            await self.send_message(request_message, CommunicationProtocol.REQUEST_RESPONSE)
            
            # Wait for response
            try:
                response = await asyncio.wait_for(response_future, timeout=timeout)
                return response
            except asyncio.TimeoutError:
                self.logger.warning(f"⏰ Request timeout: {sender_id} -> {receiver_id}")
                return None
            finally:
                # Clean up pending request
                if request_message.id in self.pending_messages:
                    del self.pending_messages[request_message.id]
                    
        except Exception as e:
            self.logger.error(f"❌ Request-response error: {e}")
            return None
            
    async def _process_message_queue(self):
        """Process message queue"""
        while self.is_running:
            try:
                # Get message from queue
                message_data = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                
                # Process message
                await self._route_message(message_data)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"❌ Message queue processing error: {e}")
                
    async def _route_message(self, message_data: Dict[str, Any]):
        """Route message to appropriate agent(s)"""
        try:
            message = message_data["message"]
            protocol = message_data["protocol"]
            retry_count = message_data["retry_count"]
            
            # Check if message has expired
            if message.expires_at and datetime.now() > message.expires_at:
                self.logger.warning(f"⏰ Message expired: {message.id}")
                return
                
            # Route based on protocol
            if protocol == CommunicationProtocol.DIRECT:
                await self._route_direct_message(message)
            elif protocol == CommunicationProtocol.BROADCAST:
                await self._route_broadcast_message(message)
            elif protocol == CommunicationProtocol.REQUEST_RESPONSE:
                await self._route_request_response(message)
                
            # Add to history
            self.message_history.append(message)
            if len(self.message_history) > self.max_history_size:
                self.message_history.pop(0)
                
        except Exception as e:
            self.logger.error(f"❌ Message routing error: {e}")
            
            # Retry logic
            if retry_count < 3:
                message_data["retry_count"] = retry_count + 1
                await asyncio.sleep(1)  # Wait before retry
                await self.message_queue.put(message_data)
            else:
                self.failed_messages.append({
                    "message": message,
                    "error": str(e),
                    "timestamp": datetime.now()
                })
                
    async def _route_direct_message(self, message: AgentMessage):
        """Route direct message to specific agent"""
        try:
            receiver_id = message.receiver
            
            if receiver_id in self.agents:
                start_time = datetime.now()
                
                # Deliver message
                agent = self.agents[receiver_id]
                await agent.receive_message(message)
                
                # Update metrics
                latency = (datetime.now() - start_time).total_seconds()
                self._update_metrics(receiver_id, latency, success=True)
                
                self.logger.debug(f"📬 Delivered message: {message.sender} -> {receiver_id}")
                
            else:
                self.logger.warning(f"⚠️ Agent not found: {receiver_id}")
                self._update_metrics(message.sender, 0, success=False)
                
        except Exception as e:
            self.logger.error(f"❌ Direct message routing error: {e}")
            self._update_metrics(message.receiver, 0, success=False)
            
    async def _route_broadcast_message(self, message: AgentMessage):
        """Route broadcast message"""
        try:
            # This is handled by the broadcast_message method
            await self._route_direct_message(message)
            
        except Exception as e:
            self.logger.error(f"❌ Broadcast message routing error: {e}")
            
    async def _route_request_response(self, message: AgentMessage):
        """Route request-response message"""
        try:
            if message.message_type == MessageType.RESPONSE:
                # Handle response
                correlation_id = message.correlation_id
                
                if correlation_id in self.pending_messages:
                    pending_request = self.pending_messages[correlation_id]
                    future = pending_request["future"]
                    
                    if not future.done():
                        future.set_result(message.content)
                        
                    del self.pending_messages[correlation_id]
                    
            else:
                # Handle request - route normally and expect response
                await self._route_direct_message(message)
                
        except Exception as e:
            self.logger.error(f"❌ Request-response routing error: {e}")
            
    def _update_metrics(self, agent_id: str, latency: float, success: bool):
        """Update communication metrics"""
        try:
            if agent_id in self.metrics:
                metrics = self.metrics[agent_id]
                
                if success:
                    metrics.messages_received += 1
                    
                    # Update average latency
                    if metrics.messages_received == 1:
                        metrics.average_latency = latency
                    else:
                        metrics.average_latency = (
                            metrics.average_latency * 0.9 + latency * 0.1
                        )
                else:
                    metrics.messages_failed += 1
                    
                # Update success rate
                total_messages = metrics.messages_received + metrics.messages_failed
                if total_messages > 0:
                    metrics.success_rate = metrics.messages_received / total_messages
                    
                metrics.last_activity = datetime.now()
                
        except Exception as e:
            self.logger.error(f"❌ Metrics update error: {e}")
            
    async def _collect_metrics(self):
        """Collect and log communication metrics"""
        while self.is_running:
            try:
                # Log metrics every 5 minutes
                await asyncio.sleep(300)
                
                total_sent = sum(m.messages_sent for m in self.metrics.values())
                total_received = sum(m.messages_received for m in self.metrics.values())
                total_failed = sum(m.messages_failed for m in self.metrics.values())
                
                self.logger.info(f"📊 Communication metrics - Sent: {total_sent}, Received: {total_received}, Failed: {total_failed}")
                
                # Log per-agent metrics
                for agent_id, metrics in self.metrics.items():
                    self.logger.debug(f"📊 Agent {agent_id} - Success rate: {metrics.success_rate:.2f}, Avg latency: {metrics.average_latency:.3f}s")
                    
            except Exception as e:
                self.logger.error(f"❌ Metrics collection error: {e}")
                
    async def _cleanup_old_data(self):
        """Clean up old data"""
        while self.is_running:
            try:
                # Clean up every hour
                await asyncio.sleep(3600)
                
                current_time = datetime.now()
                
                # Clean up expired pending messages
                expired_messages = []
                for msg_id, pending_data in self.pending_messages.items():
                    if (current_time - pending_data["timestamp"]).total_seconds() > pending_data["timeout"]:
                        expired_messages.append(msg_id)
                        
                for msg_id in expired_messages:
                    del self.pending_messages[msg_id]
                    
                # Clean up old failed messages (keep last 100)
                if len(self.failed_messages) > 100:
                    self.failed_messages = self.failed_messages[-100:]
                    
                self.logger.debug(f"🧹 Cleaned up {len(expired_messages)} expired messages")
                
            except Exception as e:
                self.logger.error(f"❌ Cleanup error: {e}")
                
    def get_agent_metrics(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get metrics for specific agent"""
        try:
            if agent_id in self.metrics:
                return asdict(self.metrics[agent_id])
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Get agent metrics error: {e}")
            return None
            
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all communication metrics"""
        try:
            return {
                "agent_metrics": {
                    agent_id: asdict(metrics)
                    for agent_id, metrics in self.metrics.items()
                },
                "system_metrics": {
                    "total_agents": len(self.agents),
                    "total_topics": len(self.topic_subscriptions),
                    "pending_messages": len(self.pending_messages),
                    "failed_messages": len(self.failed_messages),
                    "message_history_size": len(self.message_history)
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Get all metrics error: {e}")
            return {}
            
    def get_topic_subscribers(self, topic: str) -> List[str]:
        """Get list of subscribers for a topic"""
        return self.topic_subscriptions.get(topic, [])
        
    def get_agent_topics(self, agent_id: str) -> List[str]:
        """Get list of topics an agent is subscribed to"""
        topics = []
        for topic, subscribers in self.topic_subscriptions.items():
            if agent_id in subscribers:
                topics.append(topic)
        return topics
        
    def get_message_history(self, agent_id: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get message history"""
        try:
            history = self.message_history
            
            if agent_id:
                history = [
                    msg for msg in history 
                    if msg.sender == agent_id or msg.receiver == agent_id
                ]
                
            # Return last 'limit' messages
            history = history[-limit:]
            
            return [
                {
                    "id": msg.id,
                    "sender": msg.sender,
                    "receiver": msg.receiver,
                    "message_type": msg.message_type.value,
                    "timestamp": msg.timestamp.isoformat(),
                    "content": msg.content
                }
                for msg in history
            ]
            
        except Exception as e:
            self.logger.error(f"❌ Get message history error: {e}")
            return []
            
    def get_failed_messages(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get failed messages"""
        try:
            return self.failed_messages[-limit:]
            
        except Exception as e:
            self.logger.error(f"❌ Get failed messages error: {e}")
            return []
            
    def get_communication_status(self) -> Dict[str, Any]:
        """Get overall communication status"""
        try:
            total_sent = sum(m.messages_sent for m in self.metrics.values())
            total_received = sum(m.messages_received for m in self.metrics.values())
            total_failed = sum(m.messages_failed for m in self.metrics.values())
            
            success_rate = total_received / (total_received + total_failed) if (total_received + total_failed) > 0 else 1.0
            
            return {
                "is_running": self.is_running,
                "total_agents": len(self.agents),
                "total_topics": len(self.topic_subscriptions),
                "messages_sent": total_sent,
                "messages_received": total_received,
                "messages_failed": total_failed,
                "success_rate": success_rate,
                "pending_messages": len(self.pending_messages),
                "queue_size": self.message_queue.qsize()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Get communication status error: {e}")
            return {"is_running": False, "error": str(e)}
