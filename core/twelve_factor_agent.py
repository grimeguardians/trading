"""
12-Factor Agent Implementation for AI Trading System
Stateless, configurable, and scalable agent architecture
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
from enum import Enum
import asyncio
import uuid
from pathlib import Path
import signal
import sys

# Configure logging from environment
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format=os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)
logger = logging.getLogger(__name__)

class AgentState(Enum):
    """Agent lifecycle states"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PROCESSING = "processing"
    IDLE = "idle"
    ERROR = "error"
    SHUTDOWN = "shutdown"

class ProcessType(Enum):
    """Process types following 12-factor methodology"""
    WEB = "web"           # Web server for API endpoints
    WORKER = "worker"     # Background job processing
    SCHEDULER = "scheduler"  # Cron-like scheduled tasks
    STREAM = "stream"     # Real-time data stream processing

@dataclass
class AgentConfig:
    """Configuration for 12-factor agent"""
    # Factor 1: Codebase - One codebase tracked in version control
    version: str = "1.0.0"
    
    # Factor 2: Dependencies - Explicitly declare and isolate dependencies
    dependencies: List[str] = field(default_factory=list)
    
    # Factor 3: Config - Store config in environment variables
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Factor 4: Backing services - Treat backing services as attached resources
    backing_services: Dict[str, str] = field(default_factory=dict)
    
    # Factor 5: Build, release, run - Strictly separate build and run stages
    build_info: Dict[str, Any] = field(default_factory=dict)
    
    # Factor 6: Processes - Execute app as stateless processes
    process_type: ProcessType = ProcessType.WORKER
    
    # Factor 7: Port binding - Export services via port binding
    port: Optional[int] = None
    
    # Factor 8: Concurrency - Scale out via process model
    concurrency: int = 1
    
    # Factor 9: Disposability - Maximize robustness with fast startup and graceful shutdown
    startup_timeout: int = 30
    shutdown_timeout: int = 30
    
    # Factor 10: Dev/prod parity - Keep development, staging, and production as similar as possible
    environment: str = "development"
    
    # Factor 11: Logs - Treat logs as event streams
    log_destination: str = "stdout"
    
    # Factor 12: Admin processes - Run admin/management tasks as one-off processes
    admin_commands: Dict[str, Callable] = field(default_factory=dict)

class TwelveFactorAgent(ABC):
    """
    Abstract base class for 12-factor compliant agents
    """
    
    def __init__(self, agent_id: str, config: AgentConfig):
        self.agent_id = agent_id
        self.config = config
        self.state = AgentState.INITIALIZING
        self.startup_time = datetime.now()
        self.last_heartbeat = datetime.now()
        self.metrics = {
            'messages_processed': 0,
            'errors_count': 0,
            'uptime_seconds': 0,
            'memory_usage_mb': 0
        }
        
        # Factor 3: Configuration from environment
        self._load_config_from_env()
        
        # Factor 11: Structured logging
        self.logger = self._setup_logging()
        
        # Factor 9: Graceful shutdown handling
        self._setup_signal_handlers()
        
        # Factor 4: Initialize backing services
        self._initialize_backing_services()
        
        self.logger.info(f"Agent {self.agent_id} initialized with config: {self.config.environment}")
    
    def _load_config_from_env(self):
        """Load configuration from environment variables (Factor 3)"""
        env_config = {
            'database_url': os.getenv('DATABASE_URL'),
            'mcp_server_host': os.getenv('MCP_SERVER_HOST', 'localhost'),
            'mcp_server_port': int(os.getenv('MCP_SERVER_PORT', '8765')),
            'alpaca_api_key': os.getenv('ALPACA_API_KEY'),
            'alpaca_secret': os.getenv('ALPACA_SECRET'),
            'alpaca_base_url': os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets'),
            'redis_url': os.getenv('REDIS_URL'),
            'log_level': os.getenv('LOG_LEVEL', 'INFO'),
            'max_memory_mb': int(os.getenv('MAX_MEMORY_MB', '512')),
            'max_processing_time': int(os.getenv('MAX_PROCESSING_TIME', '300')),
            'heartbeat_interval': int(os.getenv('HEARTBEAT_INTERVAL', '30')),
            'environment': os.getenv('ENVIRONMENT', 'development'),
            'port': int(os.getenv('PORT', '8000')) if os.getenv('PORT') else None
        }
        
        # Merge with existing config
        self.config.config.update(env_config)
        
        # Update other config fields
        self.config.environment = env_config['environment']
        self.config.port = env_config['port']
    
    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging (Factor 11)"""
        logger = logging.getLogger(f"{self.__class__.__name__}_{self.agent_id}")
        
        # Configure log destination
        if self.config.log_destination == "stdout":
            handler = logging.StreamHandler(sys.stdout)
        else:
            handler = logging.FileHandler(self.config.log_destination)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers (Factor 9)"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, starting graceful shutdown")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    def _initialize_backing_services(self):
        """Initialize connections to backing services (Factor 4)"""
        # This would initialize database connections, message queues, etc.
        self.backing_services = {}
        
        for service_name, service_url in self.config.backing_services.items():
            self.logger.info(f"Connecting to backing service: {service_name}")
            # Initialize service connection here
            self.backing_services[service_name] = service_url
    
    @abstractmethod
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single message (to be implemented by subclasses)"""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Return health status of the agent"""
        pass
    
    async def start(self):
        """Start the agent (Factor 9: Fast startup)"""
        self.logger.info(f"Starting agent {self.agent_id}")
        self.state = AgentState.ACTIVE
        
        # Start background tasks
        asyncio.create_task(self._heartbeat_loop())
        asyncio.create_task(self._metrics_collector())
        
        # Start main processing loop
        await self._main_loop()
    
    async def _main_loop(self):
        """Main agent processing loop"""
        while self.state not in [AgentState.SHUTDOWN, AgentState.ERROR]:
            try:
                # Process any pending messages
                await self._process_pending_messages()
                
                # Update state
                if self.state == AgentState.PROCESSING:
                    self.state = AgentState.IDLE
                
                # Brief pause to prevent busy loop
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                self.metrics['errors_count'] += 1
                self.state = AgentState.ERROR
                await asyncio.sleep(1)  # Error recovery pause
    
    async def _process_pending_messages(self):
        """Process messages from message queue"""
        # This would integrate with your MCP server
        # For now, it's a placeholder for the message processing logic
        pass
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats (Factor 9: Disposability)"""
        while self.state != AgentState.SHUTDOWN:
            try:
                self.last_heartbeat = datetime.now()
                
                # Send heartbeat to MCP server
                await self._send_heartbeat()
                
                # Update uptime metric
                self.metrics['uptime_seconds'] = (
                    datetime.now() - self.startup_time
                ).total_seconds()
                
                await asyncio.sleep(self.config.config['heartbeat_interval'])
                
            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(5)  # Retry after 5 seconds
    
    async def _send_heartbeat(self):
        """Send heartbeat to MCP server"""
        heartbeat_data = {
            'agent_id': self.agent_id,
            'timestamp': self.last_heartbeat.isoformat(),
            'state': self.state.value,
            'metrics': self.metrics
        }
        
        # This would send to MCP server
        self.logger.debug(f"Heartbeat: {heartbeat_data}")
    
    async def _metrics_collector(self):
        """Collect and report metrics"""
        while self.state != AgentState.SHUTDOWN:
            try:
                # Collect system metrics
                import psutil
                process = psutil.Process(os.getpid())
                self.metrics['memory_usage_mb'] = process.memory_info().rss / 1024 / 1024
                
                # Log metrics periodically
                if self.metrics['uptime_seconds'] % 300 == 0:  # Every 5 minutes
                    self.logger.info(f"Agent metrics: {self.metrics}")
                
                await asyncio.sleep(10)  # Collect every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(30)
    
    async def shutdown(self):
        """Graceful shutdown (Factor 9)"""
        self.logger.info(f"Shutting down agent {self.agent_id}")
        self.state = AgentState.SHUTDOWN
        
        # Close backing service connections
        for service_name in self.backing_services:
            self.logger.info(f"Closing connection to {service_name}")
            # Close service connection here
        
        # Final metrics log
        self.logger.info(f"Final metrics: {self.metrics}")
        
        # Exit process
        sys.exit(0)
    
    def run_admin_command(self, command: str, args: List[str] = None) -> Any:
        """Run administrative command (Factor 12)"""
        if command in self.config.admin_commands:
            return self.config.admin_commands[command](args or [])
        else:
            raise ValueError(f"Unknown admin command: {command}")


class MarketAnalystAgent(TwelveFactorAgent):
    """Market analysis agent implementing 12-factor principles"""
    
    def __init__(self, agent_id: str = None):
        config = AgentConfig(
            process_type=ProcessType.WORKER,
            dependencies=['pandas', 'numpy', 'yfinance', 'alpaca-trade-api'],
            backing_services={
                'database': 'postgresql://user:pass@localhost/trading',
                'mcp_server': 'ws://localhost:8765',
                'market_data': 'https://paper-api.alpaca.markets'
            }
        )
        
        super().__init__(agent_id or f"market_analyst_{uuid.uuid4().hex[:8]}", config)
    
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process market analysis requests"""
        self.state = AgentState.PROCESSING
        self.metrics['messages_processed'] += 1
        
        try:
            message_type = message.get('type')
            
            if message_type == 'analyze_symbol':
                return await self._analyze_symbol(message.get('symbol'))
            elif message_type == 'market_sentiment':
                return await self._get_market_sentiment()
            elif message_type == 'fibonacci_analysis':
                return await self._fibonacci_analysis(message.get('symbol'))
            else:
                return {'error': f'Unknown message type: {message_type}'}
                
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            self.metrics['errors_count'] += 1
            return {'error': str(e)}
    
    async def _analyze_symbol(self, symbol: str) -> Dict[str, Any]:
        """Analyze a specific symbol"""
        # This would integrate with your existing analysis logic
        return {
            'symbol': symbol,
            'analysis': 'placeholder_analysis',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _get_market_sentiment(self) -> Dict[str, Any]:
        """Get overall market sentiment"""
        return {
            'sentiment': 'neutral',
            'confidence': 0.75,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _fibonacci_analysis(self, symbol: str) -> Dict[str, Any]:
        """Perform Fibonacci retracement analysis"""
        # This would integrate with your mathematical models
        return {
            'symbol': symbol,
            'fibonacci_levels': [0.236, 0.382, 0.5, 0.618, 0.786],
            'current_level': 0.5,
            'timestamp': datetime.now().isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Return health status"""
        return {
            'agent_id': self.agent_id,
            'state': self.state.value,
            'uptime_seconds': self.metrics['uptime_seconds'],
            'memory_usage_mb': self.metrics['memory_usage_mb'],
            'messages_processed': self.metrics['messages_processed'],
            'errors_count': self.metrics['errors_count'],
            'healthy': self.state == AgentState.ACTIVE
        }


# Factory functions for creating different agent types
def create_market_analyst_agent(agent_id: str = None) -> MarketAnalystAgent:
    """Create a market analyst agent"""
    return MarketAnalystAgent(agent_id)

def create_risk_manager_agent(agent_id: str = None) -> TwelveFactorAgent:
    """Create a risk manager agent"""
    # This would be implemented similar to MarketAnalystAgent
    pass

def create_strategy_coordinator_agent(agent_id: str = None) -> TwelveFactorAgent:
    """Create a strategy coordinator agent"""
    # This would be implemented similar to MarketAnalystAgent
    pass

if __name__ == "__main__":
    # Example usage
    agent = create_market_analyst_agent()
    asyncio.run(agent.start())