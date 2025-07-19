"""
Base Agent class for MCP agents
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime

class BaseAgent(ABC):
    """Base class for all MCP agents"""
    
    def __init__(self, agent_id: str, settings, mcp_server, freqtrade_engine, digital_brain):
        self.agent_id = agent_id
        self.settings = settings
        self.mcp_server = mcp_server
        self.freqtrade_engine = freqtrade_engine
        self.digital_brain = digital_brain
        
        # Agent state
        self.is_active = False
        self.last_activity = datetime.now()
        self.error_count = 0
        
        # Logging
        self.logger = logging.getLogger(f"Agent_{agent_id}")
        
    async def initialize(self):
        """Initialize the agent"""
        self.is_active = True
        self.last_activity = datetime.now()
        self.logger.info(f"Agent {self.agent_id} initialized")
    
    @abstractmethod
    async def handle_message(self, message):
        """Handle incoming messages"""
        pass
    
    async def shutdown(self):
        """Shutdown the agent"""
        self.is_active = False
        self.logger.info(f"Agent {self.agent_id} shutdown")
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()
    
    def increment_error_count(self):
        """Increment error count"""
        self.error_count += 1
    
    def get_status(self) -> Dict:
        """Get agent status"""
        return {
            "agent_id": self.agent_id,
            "is_active": self.is_active,
            "last_activity": self.last_activity.isoformat(),
            "error_count": self.error_count
        }
