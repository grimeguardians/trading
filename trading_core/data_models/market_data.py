"""Core data models for market data and trading signals"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional
from enum import Enum

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LIMIT = "STOP_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT"
    TRAILING_STOP = "TRAILING_STOP"

class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"

@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    price: float
    volume: int
    timestamp: datetime
    bid: float
    ask: float
    high_24h: float = 0.0
    low_24h: float = 0.0

@dataclass
class TradingSignal:
    """Trading signal structure"""
    symbol: str
    action: str  # 'BUY' or 'SELL'
    confidence: float
    reason: str
    timestamp: datetime
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    signal_strength: str = "MEDIUM"  # WEAK, MEDIUM, STRONG