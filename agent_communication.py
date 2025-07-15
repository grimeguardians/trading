
import logging
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import json
from collections import deque
import uuid

class MessageType(Enum):
    MARKET_DATA = "market_data"
    TRADING_SIGNAL = "trading_signal"
    RISK_ALERT = "risk_alert"
    PATTERN_RECOGNITION = "pattern_recognition"
    PORTFOLIO_UPDATE = "portfolio_update"
    SYSTEM_STATUS = "system_status"
    COMMAND = "command"
    RESPONSE = "response"

class MessagePriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Message:
    """Inter-agent communication message"""
    message_id: str
    sender: str
    recipient: str
    message_type: MessageType
    priority: MessagePriority
    payload: Any
    timestamp: datetime
    expires_at: Optional[datetime] = None
    requires_response: bool = False
    correlation_id: Optional[str] = None

@dataclass
class AgentCapability:
    """Describes what an agent can do"""
    agent_name: str
    capabilities: List[str]
    message_types_handled: List[MessageType]
    max_concurrent_messages: int = 10
    average_response_time_ms: float = 100.0

class MessageBus:
    """High-performance message bus for agent communication"""
    
    def __init__(self):
        self.subscribers: Dict[MessageType, List[Callable]] = {}
        self.agent_queues: Dict[str, deque] = {}
        self.agent_capabilities: Dict[str, AgentCapability] = {}
        self.message_history: deque = deque(maxlen=1000)
        self.performance_metrics: Dict[str, Any] = {}
        self.logger = logging.getLogger("MessageBus")
        self._running = False
        self._worker_thread = None
        
    def start(self):
        """Start the message bus"""
        self._running = True
        self._worker_thread = threading.Thread(target=self._message_worker, daemon=True)
        self._worker_thread.start()
        self.logger.info("Message bus started")
    
    def stop(self):
        """Stop the message bus"""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=1.0)
        self.logger.info("Message bus stopped")
    
    def register_agent(self, capability: AgentCapability):
        """Register an agent with its capabilities"""
        self.agent_capabilities[capability.agent_name] = capability
        self.agent_queues[capability.agent_name] = deque(maxlen=100)
        
        self.logger.info(f"Registered agent {capability.agent_name} with capabilities: {capability.capabilities}")
    
    def subscribe(self, message_type: MessageType, handler: Callable, agent_name: str):
        """Subscribe to specific message types"""
        if message_type not in self.subscribers:
            self.subscribers[message_type] = []
        
        self.subscribers[message_type].append({
            'handler': handler,
            'agent': agent_name
        })
        
        self.logger.debug(f"Agent {agent_name} subscribed to {message_type}")
    
    def publish(self, message: Message) -> bool:
        """Publish a message to the bus"""
        try:
            # Add to history
            self.message_history.append(message)
            
            # Route to specific recipient if specified
            if message.recipient and message.recipient in self.agent_queues:
                self.agent_queues[message.recipient].append(message)
                return True
            
            # Broadcast to subscribers
            subscribers = self.subscribers.get(message.message_type, [])
            if not subscribers:
                self.logger.warning(f"No subscribers for message type {message.message_type}")
                return False
            
            for subscriber in subscribers:
                try:
                    self.agent_queues[subscriber['agent']].append(message)
                except Exception as e:
                    self.logger.error(f"Error delivering message to {subscriber['agent']}: {e}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error publishing message: {e}")
            return False
    
    def send_request(self, sender: str, recipient: str, message_type: MessageType, 
                    payload: Any, timeout_seconds: float = 5.0) -> Optional[Any]:
        """Send a request and wait for response"""
        correlation_id = str(uuid.uuid4())
        
        request = Message(
            message_id=str(uuid.uuid4()),
            sender=sender,
            recipient=recipient,
            message_type=message_type,
            priority=MessagePriority.NORMAL,
            payload=payload,
            timestamp=datetime.now(),
            requires_response=True,
            correlation_id=correlation_id
        )
        
        if self.publish(request):
            # Wait for response
            start_time = time.time()
            while time.time() - start_time < timeout_seconds:
                # Check for response in sender's queue
                if sender in self.agent_queues:
                    queue = self.agent_queues[sender]
                    for i, msg in enumerate(queue):
                        if (msg.correlation_id == correlation_id and 
                            msg.message_type == MessageType.RESPONSE):
                            # Remove and return response
                            del queue[i]
                            return msg.payload
                
                time.sleep(0.01)  # 10ms polling interval
            
            self.logger.warning(f"Request timeout: {sender} -> {recipient}")
        
        return None
    
    def get_messages(self, agent_name: str, max_messages: int = 10) -> List[Message]:
        """Get pending messages for an agent"""
        if agent_name not in self.agent_queues:
            return []
        
        messages = []
        queue = self.agent_queues[agent_name]
        
        for _ in range(min(max_messages, len(queue))):
            if queue:
                messages.append(queue.popleft())
        
        return messages
    
    def send_response(self, original_message: Message, response_payload: Any, sender: str):
        """Send a response to a request"""
        if not original_message.requires_response:
            return
        
        response = Message(
            message_id=str(uuid.uuid4()),
            sender=sender,
            recipient=original_message.sender,
            message_type=MessageType.RESPONSE,
            priority=original_message.priority,
            payload=response_payload,
            timestamp=datetime.now(),
            correlation_id=original_message.correlation_id
        )
        
        self.publish(response)
    
    def _message_worker(self):
        """Background worker for message processing"""
        while self._running:
            try:
                # Clean up expired messages
                current_time = datetime.now()
                for queue in self.agent_queues.values():
                    expired_indices = []
                    for i, msg in enumerate(queue):
                        if msg.expires_at and msg.expires_at < current_time:
                            expired_indices.append(i)
                    
                    # Remove expired messages (reverse order to maintain indices)
                    for i in reversed(expired_indices):
                        del queue[i]
                
                time.sleep(0.1)  # 100ms cleanup cycle
                
            except Exception as e:
                self.logger.error(f"Error in message worker: {e}")
                time.sleep(1.0)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get message bus performance metrics"""
        total_messages = len(self.message_history)
        queue_sizes = {agent: len(queue) for agent, queue in self.agent_queues.items()}
        
        return {
            'total_messages_processed': total_messages,
            'active_agents': len(self.agent_capabilities),
            'queue_sizes': queue_sizes,
            'average_queue_size': sum(queue_sizes.values()) / max(len(queue_sizes), 1),
            'message_types_distribution': self._get_message_type_distribution()
        }
    
    def _get_message_type_distribution(self) -> Dict[str, int]:
        """Get distribution of message types"""
        distribution = {}
        for msg in self.message_history:
            msg_type = msg.message_type.value
            distribution[msg_type] = distribution.get(msg_type, 0) + 1
        return distribution

class AgentInterface(ABC):
    """Base interface for agents using the communication layer"""
    
    def __init__(self, agent_name: str, message_bus: MessageBus):
        self.agent_name = agent_name
        self.message_bus = message_bus
        self.logger = logging.getLogger(f"Agent-{agent_name}")
        self.is_active = False
        
    @abstractmethod
    def get_capabilities(self) -> AgentCapability:
        """Return agent capabilities"""
        pass
    
    @abstractmethod
    def handle_message(self, message: Message) -> Optional[Any]:
        """Handle incoming messages"""
        pass
    
    def start(self):
        """Start the agent"""
        self.is_active = True
        capabilities = self.get_capabilities()
        self.message_bus.register_agent(capabilities)
        
        # Subscribe to relevant message types
        for msg_type in capabilities.message_types_handled:
            self.message_bus.subscribe(msg_type, self.handle_message, self.agent_name)
        
        self.logger.info(f"Agent {self.agent_name} started")
    
    def stop(self):
        """Stop the agent"""
        self.is_active = False
        self.logger.info(f"Agent {self.agent_name} stopped")
    
    def send_message(self, recipient: str, message_type: MessageType, payload: Any, 
                    priority: MessagePriority = MessagePriority.NORMAL) -> bool:
        """Send a message to another agent"""
        message = Message(
            message_id=str(uuid.uuid4()),
            sender=self.agent_name,
            recipient=recipient,
            message_type=message_type,
            priority=priority,
            payload=payload,
            timestamp=datetime.now()
        )
        
        return self.message_bus.publish(message)
    
    def broadcast_message(self, message_type: MessageType, payload: Any, 
                         priority: MessagePriority = MessagePriority.NORMAL) -> bool:
        """Broadcast a message to all subscribers"""
        message = Message(
            message_id=str(uuid.uuid4()),
            sender=self.agent_name,
            recipient="",  # Empty for broadcast
            message_type=message_type,
            priority=priority,
            payload=payload,
            timestamp=datetime.now()
        )
        
        return self.message_bus.publish(message)
    
    def request_data(self, recipient: str, request_type: MessageType, 
                    request_payload: Any, timeout: float = 5.0) -> Optional[Any]:
        """Request data from another agent"""
        return self.message_bus.send_request(
            self.agent_name, recipient, request_type, request_payload, timeout
        )
    
    def process_messages(self):
        """Process pending messages"""
        if not self.is_active:
            return
        
        messages = self.message_bus.get_messages(self.agent_name)
        
        for message in messages:
            try:
                response = self.handle_message(message)
                
                # Send response if required
                if message.requires_response and response is not None:
                    self.message_bus.send_response(message, response, self.agent_name)
                    
            except Exception as e:
                self.logger.error(f"Error handling message {message.message_id}: {e}")
