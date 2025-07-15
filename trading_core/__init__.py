"""
Trading Core - Streamlined Multi-Agent Trading System
Optimized for performance and maintainability while preserving Digital Brain functionality.
"""

__version__ = "2.0.0"
__author__ = "Trading Agent Suite"

from .agents.market_analyst import MarketAnalystAgent
from .agents.risk_manager import RiskManagerAgent  
from .agents.trading_executor import TradingExecutorAgent
from .agents.coordinator import CoordinatorAgent

from .strategies.technical_analysis import TechnicalIndicators
from .strategies.ml_engine import MLPredictionEngine
from .strategies.sentiment_analyzer import SentimentAnalyzer

__all__ = [
    'MarketAnalystAgent',
    'RiskManagerAgent', 
    'TradingExecutorAgent',
    'CoordinatorAgent',
    'TechnicalIndicators',
    'MLPredictionEngine',
    'SentimentAnalyzer'
]