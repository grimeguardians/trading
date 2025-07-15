"""Base agent class for all trading agents"""

import logging
from abc import ABC, abstractmethod
from typing import Any

class BaseAgent(ABC):
    """Base class for all trading agents"""

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)
        self.is_active = False

    @abstractmethod
    def process(self, data: Any) -> Any:
        """Main processing method for the agent"""
        pass

    def start(self):
        """Start the agent"""
        self.is_active = True
        self.logger.info(f"{self.name} started")

    def stop(self):
        """Stop the agent"""
        self.is_active = False
        self.logger.info(f"{self.name} stopped")