"""Trading-specific data models"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional
from .market_data import OrderType, OrderStatus

@dataclass
class TradeOrder:
    """Enhanced trade order structure with stop-loss support"""
    order_id: str
    symbol: str
    action: str
    quantity: int
    price: float
    order_type: OrderType
    timestamp: datetime
    status: OrderStatus = OrderStatus.PENDING
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    trailing_stop_distance: Optional[float] = None
    parent_order_id: Optional[str] = None
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    expiry_time: Optional[datetime] = None

@dataclass
class Position:
    """Enhanced position tracking structure"""
    symbol: str
    quantity: int
    avg_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    trailing_stop_price: Optional[float] = None
    max_price_since_entry: float = 0.0
    entry_timestamp: datetime = field(default_factory=datetime.now)
    risk_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class TechnicalAnalysis:
    """Technical analysis results"""
    symbol: str
    sma_20: float
    sma_50: float
    ema_12: float
    ema_26: float
    rsi: float
    macd: float
    macd_signal: float
    macd_histogram: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    volume_sma: float
    atr: float
    fibonacci_levels: Dict[str, float]
    stochastic_k: float
    stochastic_d: float
    williams_r: float
    timestamp: datetime
    support_level: float = 0.0
    resistance_level: float = 0.0
    volatility: float = 0.0

@dataclass
class MLPrediction:
    """Machine learning prediction results"""
    symbol: str
    prediction: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    feature_importance: Dict[str, float]
    model_accuracy: float
    timestamp: datetime
    price_target: Optional[float] = None
    risk_score: float = 0.0

@dataclass
class SentimentData:
    """Sentiment analysis data"""
    symbol: str
    sentiment_score: float  # -1 to +1
    sentiment_label: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    news_count: int
    social_mentions: int
    timestamp: datetime
    sentiment_trend: str = "STABLE"  # IMPROVING, DECLINING, STABLE

@dataclass
class PortfolioOptimization:
    """Portfolio optimization results"""
    optimal_weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    risk_metrics: Dict[str, float]
    rebalance_suggestions: Dict[str, float]
    timestamp: datetime

@dataclass
class RiskAlert:
    """Risk management alerts"""
    alert_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    message: str
    symbol: Optional[str]
    timestamp: datetime
    action_required: bool = False