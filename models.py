"""
Database models for the Advanced AI Trading System
SQLAlchemy models with relationships and indexes for optimal performance
"""

import os
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from enum import Enum
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, JSON, ForeignKey, Index
from sqlalchemy.orm import relationship, DeclarativeBase
from sqlalchemy.ext.hybrid import hybrid_property
from flask_sqlalchemy import SQLAlchemy
import json


class Base(DeclarativeBase):
    pass


db = SQLAlchemy(model_class=Base)


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class StrategyType(Enum):
    SWING = "swing"
    SCALPING = "scalping"
    OPTIONS = "options"
    INTRADAY = "intraday"


class Exchange(db.Model):
    """Exchange configuration and status"""
    __tablename__ = 'exchanges'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)
    display_name = Column(String(100), nullable=False)
    enabled = Column(Boolean, default=True)
    api_key_configured = Column(Boolean, default=False)
    sandbox_mode = Column(Boolean, default=True)
    supported_assets = Column(JSON, default=list)
    rate_limit = Column(Integer, default=100)
    last_heartbeat = Column(DateTime, default=datetime.utcnow)
    status = Column(String(20), default="disconnected")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    orders = relationship("Order", back_populates="exchange")
    positions = relationship("Position", back_populates="exchange")
    market_data = relationship("MarketData", back_populates="exchange")
    
    def __repr__(self):
        return f"<Exchange {self.name}>"


class Strategy(db.Model):
    """Trading strategy configuration"""
    __tablename__ = 'strategies'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    strategy_type = Column(String(20), nullable=False)
    enabled = Column(Boolean, default=True)
    exchange_id = Column(Integer, ForeignKey('exchanges.id'), nullable=False)
    parameters = Column(JSON, default=dict)
    max_positions = Column(Integer, default=5)
    min_profit_target = Column(Float, default=0.05)
    stop_loss_pct = Column(Float, default=0.02)
    take_profit_pct = Column(Float, default=0.06)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    exchange = relationship("Exchange")
    positions = relationship("Position", back_populates="strategy")
    performance = relationship("StrategyPerformance", back_populates="strategy")
    
    def __repr__(self):
        return f"<Strategy {self.name}>"


class Position(db.Model):
    """Trading position tracking"""
    __tablename__ = 'positions'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    exchange_id = Column(Integer, ForeignKey('exchanges.id'), nullable=False)
    strategy_id = Column(Integer, ForeignKey('strategies.id'), nullable=False)
    side = Column(String(10), nullable=False)  # long/short
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float, default=0.0)
    stop_loss_price = Column(Float)
    take_profit_price = Column(Float)
    unrealized_pnl = Column(Float, default=0.0)
    realized_pnl = Column(Float, default=0.0)
    status = Column(String(20), default="open")  # open/closed
    opened_at = Column(DateTime, default=datetime.utcnow)
    closed_at = Column(DateTime)
    metadata = Column(JSON, default=dict)
    
    # Relationships
    exchange = relationship("Exchange", back_populates="positions")
    strategy = relationship("Strategy", back_populates="positions")
    orders = relationship("Order", back_populates="position")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_position_symbol', 'symbol'),
        Index('idx_position_status', 'status'),
        Index('idx_position_exchange', 'exchange_id'),
        Index('idx_position_strategy', 'strategy_id'),
    )
    
    @hybrid_property
    def profit_loss_pct(self):
        """Calculate profit/loss percentage"""
        if self.entry_price == 0:
            return 0.0
        return ((self.current_price - self.entry_price) / self.entry_price) * 100
    
    def __repr__(self):
        return f"<Position {self.symbol} {self.side} {self.quantity}>"


class Order(db.Model):
    """Order tracking and management"""
    __tablename__ = 'orders'
    
    id = Column(Integer, primary_key=True)
    exchange_order_id = Column(String(100), nullable=False)
    symbol = Column(String(20), nullable=False)
    exchange_id = Column(Integer, ForeignKey('exchanges.id'), nullable=False)
    position_id = Column(Integer, ForeignKey('positions.id'), nullable=True)
    order_type = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)
    quantity = Column(Float, nullable=False)
    price = Column(Float)
    filled_quantity = Column(Float, default=0.0)
    filled_price = Column(Float, default=0.0)
    status = Column(String(20), default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    filled_at = Column(DateTime)
    metadata = Column(JSON, default=dict)
    
    # Relationships
    exchange = relationship("Exchange", back_populates="orders")
    position = relationship("Position", back_populates="orders")
    
    # Indexes
    __table_args__ = (
        Index('idx_order_symbol', 'symbol'),
        Index('idx_order_status', 'status'),
        Index('idx_order_exchange', 'exchange_id'),
        Index('idx_order_exchange_id', 'exchange_order_id'),
    )
    
    def __repr__(self):
        return f"<Order {self.symbol} {self.side} {self.quantity}>"


class MarketData(db.Model):
    """Real-time and historical market data"""
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    exchange_id = Column(Integer, ForeignKey('exchanges.id'), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    timeframe = Column(String(10), nullable=False)  # 1m, 5m, 1h, 1d
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    exchange = relationship("Exchange", back_populates="market_data")
    
    # Indexes for time-series queries
    __table_args__ = (
        Index('idx_market_data_symbol_time', 'symbol', 'timestamp'),
        Index('idx_market_data_exchange_time', 'exchange_id', 'timestamp'),
        Index('idx_market_data_timeframe', 'timeframe'),
    )
    
    def __repr__(self):
        return f"<MarketData {self.symbol} {self.timestamp}>"


class StrategyPerformance(db.Model):
    """Strategy performance metrics"""
    __tablename__ = 'strategy_performance'
    
    id = Column(Integer, primary_key=True)
    strategy_id = Column(Integer, ForeignKey('strategies.id'), nullable=False)
    date = Column(DateTime, nullable=False)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    total_pnl = Column(Float, default=0.0)
    total_return_pct = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    win_rate = Column(Float, default=0.0)
    avg_win = Column(Float, default=0.0)
    avg_loss = Column(Float, default=0.0)
    profit_factor = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    strategy = relationship("Strategy", back_populates="performance")
    
    # Indexes
    __table_args__ = (
        Index('idx_performance_strategy_date', 'strategy_id', 'date'),
    )
    
    def __repr__(self):
        return f"<StrategyPerformance {self.strategy_id} {self.date}>"


class TradingSignal(db.Model):
    """AI-generated trading signals"""
    __tablename__ = 'trading_signals'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    signal_type = Column(String(20), nullable=False)  # buy/sell/hold
    confidence = Column(Float, nullable=False)
    source = Column(String(50), nullable=False)  # agent name
    reasoning = Column(Text)
    price_target = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    timeframe = Column(String(10), nullable=False)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    processed = Column(Boolean, default=False)
    metadata = Column(JSON, default=dict)
    
    # Indexes
    __table_args__ = (
        Index('idx_signal_symbol_time', 'symbol', 'created_at'),
        Index('idx_signal_processed', 'processed'),
        Index('idx_signal_expires', 'expires_at'),
    )
    
    def __repr__(self):
        return f"<TradingSignal {self.symbol} {self.signal_type}>"


class KnowledgeNode(db.Model):
    """Knowledge graph nodes for AI brain"""
    __tablename__ = 'knowledge_nodes'
    
    id = Column(Integer, primary_key=True)
    node_id = Column(String(100), unique=True, nullable=False)
    node_type = Column(String(50), nullable=False)
    attributes = Column(JSON, default=dict)
    confidence = Column(Float, default=1.0)
    source = Column(String(100), default="system")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    source_edges = relationship("KnowledgeEdge", foreign_keys="KnowledgeEdge.source_node_id", back_populates="source_node")
    target_edges = relationship("KnowledgeEdge", foreign_keys="KnowledgeEdge.target_node_id", back_populates="target_node")
    
    # Indexes
    __table_args__ = (
        Index('idx_knowledge_node_type', 'node_type'),
        Index('idx_knowledge_node_confidence', 'confidence'),
    )
    
    def __repr__(self):
        return f"<KnowledgeNode {self.node_id}>"


class KnowledgeEdge(db.Model):
    """Knowledge graph edges for AI brain"""
    __tablename__ = 'knowledge_edges'
    
    id = Column(Integer, primary_key=True)
    edge_id = Column(String(100), unique=True, nullable=False)
    source_node_id = Column(Integer, ForeignKey('knowledge_nodes.id'), nullable=False)
    target_node_id = Column(Integer, ForeignKey('knowledge_nodes.id'), nullable=False)
    relationship_type = Column(String(50), nullable=False)
    strength = Column(Float, nullable=False)
    confidence = Column(Float, default=1.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    source_node = relationship("KnowledgeNode", foreign_keys=[source_node_id], back_populates="source_edges")
    target_node = relationship("KnowledgeNode", foreign_keys=[target_node_id], back_populates="target_edges")
    
    # Indexes
    __table_args__ = (
        Index('idx_knowledge_edge_source', 'source_node_id'),
        Index('idx_knowledge_edge_target', 'target_node_id'),
        Index('idx_knowledge_edge_type', 'relationship_type'),
    )
    
    def __repr__(self):
        return f"<KnowledgeEdge {self.edge_id}>"


class BacktestResult(db.Model):
    """Backtesting results storage"""
    __tablename__ = 'backtest_results'
    
    id = Column(Integer, primary_key=True)
    strategy_id = Column(Integer, ForeignKey('strategies.id'), nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_capital = Column(Float, nullable=False)
    final_capital = Column(Float, nullable=False)
    total_return = Column(Float, nullable=False)
    max_drawdown = Column(Float, nullable=False)
    sharpe_ratio = Column(Float, nullable=False)
    total_trades = Column(Integer, nullable=False)
    win_rate = Column(Float, nullable=False)
    profit_factor = Column(Float, nullable=False)
    results_data = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    strategy = relationship("Strategy")
    
    # Indexes
    __table_args__ = (
        Index('idx_backtest_strategy', 'strategy_id'),
        Index('idx_backtest_date', 'start_date', 'end_date'),
    )
    
    def __repr__(self):
        return f"<BacktestResult {self.strategy_id} {self.total_return}%>"


class SystemEvent(db.Model):
    """System events and alerts"""
    __tablename__ = 'system_events'
    
    id = Column(Integer, primary_key=True)
    event_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)  # info/warning/error/critical
    message = Column(Text, nullable=False)
    source = Column(String(100), nullable=False)
    metadata = Column(JSON, default=dict)
    acknowledged = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_system_event_type', 'event_type'),
        Index('idx_system_event_severity', 'severity'),
        Index('idx_system_event_time', 'created_at'),
    )
    
    def __repr__(self):
        return f"<SystemEvent {self.event_type} {self.severity}>"
