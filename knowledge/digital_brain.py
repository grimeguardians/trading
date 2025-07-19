"""
Digital Brain - Advanced Knowledge Engine for Trading Intelligence
Integrates with existing knowledge_engine.py for enhanced pattern recognition and decision making
"""

import asyncio
import logging
import json
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import sqlite3
from enum import Enum
import re
from collections import defaultdict, Counter
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

from config import Config
from knowledge.pattern_recognition import PatternRecognition
from knowledge.document_processor import DocumentProcessor

# Import AI models based on configuration
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

class KnowledgeType(Enum):
    """Types of knowledge in the digital brain"""
    TECHNICAL_PATTERN = "technical_pattern"
    MARKET_CONCEPT = "market_concept"
    TRADING_RULE = "trading_rule"
    RISK_PRINCIPLE = "risk_principle"
    ECONOMIC_INDICATOR = "economic_indicator"
    COMPANY_FUNDAMENTAL = "company_fundamental"
    NEWS_EVENT = "news_event"
    SENTIMENT_SIGNAL = "sentiment_signal"
    CORRELATION_PATTERN = "correlation_pattern"
    VOLATILITY_REGIME = "volatility_regime"

class ConfidenceLevel(Enum):
    """Confidence levels for knowledge"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class KnowledgeNode:
    """Enhanced knowledge node with trading context"""
    node_id: str
    knowledge_type: KnowledgeType
    title: str
    description: str
    content: str
    confidence: float
    relevance_score: float
    tags: List[str]
    symbols: List[str]
    timeframe: str
    market_regime: str
    success_rate: float
    sample_size: int
    last_updated: datetime
    created_at: datetime
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embeddings: Optional[List[float]] = None

@dataclass
class KnowledgeQuery:
    """Knowledge query structure"""
    query_id: str
    query_text: str
    context: Dict[str, Any]
    symbols: List[str]
    timeframe: str
    market_conditions: Dict[str, Any]
    priority: int
    timestamp: datetime

@dataclass
class KnowledgeResponse:
    """Knowledge response structure"""
    query_id: str
    relevant_nodes: List[KnowledgeNode]
    insights: List[str]
    recommendations: List[str]
    confidence: float
    reasoning: str
    supporting_evidence: List[str]
    risk_factors: List[str]
    timestamp: datetime

@dataclass
class TradingInsight:
    """Trading insight from digital brain"""
    insight_id: str
    insight_type: str
    symbol: str
    timeframe: str
    title: str
    description: str
    actionable_advice: str
    confidence: float
    risk_level: str
    expected_outcome: str
    supporting_patterns: List[str]
    contradicting_factors: List[str]
    historical_success_rate: float
    market_conditions: Dict[str, Any]
    timestamp: datetime

class DigitalBrain:
    """
    Advanced Digital Brain for trading intelligence
    Integrates pattern recognition, document processing, and AI reasoning
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("DigitalBrain")
        
        # Initialize components
        self.pattern_recognition = PatternRecognition(config)
        self.document_processor = DocumentProcessor(config)
        
        # AI clients
        self.openai_client = None
        self.anthropic_client = None
        self._initialize_ai_clients()
        
        # Knowledge storage
        self.knowledge_graph = nx.DiGraph()
        self.knowledge_nodes: Dict[str, KnowledgeNode] = {}
        self.knowledge_index: Dict[str, List[str]] = defaultdict(list)
        
        # Text processing
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.document_vectors = None
        
        # Database connection
        self.db_path = config.knowledge.knowledge_db_path
        self.conn = None
        
        # Caching
        self.query_cache: Dict[str, KnowledgeResponse] = {}
        self.insight_cache: Dict[str, List[TradingInsight]] = {}
        
        # Performance metrics
        self.metrics = {
            "queries_processed": 0,
            "insights_generated": 0,
            "patterns_discovered": 0,
            "knowledge_nodes": 0,
            "accuracy_score": 0.0,
            "response_time": 0.0
        }
        
        # Knowledge categories
        self.knowledge_categories = {
            KnowledgeType.TECHNICAL_PATTERN: self._load_technical_patterns,
            KnowledgeType.MARKET_CONCEPT: self._load_market_concepts,
            KnowledgeType.TRADING_RULE: self._load_trading_rules,
            KnowledgeType.RISK_PRINCIPLE: self._load_risk_principles,
            KnowledgeType.ECONOMIC_INDICATOR: self._load_economic_indicators
        }
        
        # Market context
        self.current_market_regime = "normal"
        self.market_volatility = 0.0
        self.market_sentiment = 0.0
        
        # Real-time learning
        self.learning_enabled = True
        self.feedback_history: List[Dict[str, Any]] = []
        
    def _initialize_ai_clients(self):
        """Initialize AI clients"""
        try:
            if OPENAI_AVAILABLE and self.config.ai.openai_api_key:
                self.openai_client = OpenAI(
                    api_key=self.config.ai.openai_api_key
                )
            
            if ANTHROPIC_AVAILABLE and self.config.ai.anthropic_api_key:
                self.anthropic_client = Anthropic(
                    api_key=self.config.ai.anthropic_api_key
                )
        
        except Exception as e:
            self.logger.error(f"Error initializing AI clients: {e}")
    
    async def initialize(self):
        """Initialize the digital brain"""
        try:
            self.logger.info("Initializing Digital Brain...")
            
            # Initialize database
            await self._initialize_database()
            
            # Initialize components
            await self.pattern_recognition.initialize()
            await self.document_processor.initialize()
            
            # Load existing knowledge
            await self._load_knowledge_base()
            
            # Build knowledge graph
            await self._build_knowledge_graph()
            
            # Initialize text processing
            await self._initialize_text_processing()
            
            self.logger.info("Digital Brain initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Digital Brain: {e}")
            raise
    
    async def _initialize_database(self):
        """Initialize SQLite database for knowledge storage"""
        try:
            # Create database directory if it doesn't exist
            db_dir = Path(self.db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            
            # Connect to database
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            
            # Create tables
            cursor = self.conn.cursor()
            
            # Knowledge nodes table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge_nodes (
                    node_id TEXT PRIMARY KEY,
                    knowledge_type TEXT,
                    title TEXT,
                    description TEXT,
                    content TEXT,
                    confidence REAL,
                    relevance_score REAL,
                    tags TEXT,
                    symbols TEXT,
                    timeframe TEXT,
                    market_regime TEXT,
                    success_rate REAL,
                    sample_size INTEGER,
                    last_updated TIMESTAMP,
                    created_at TIMESTAMP,
                    source TEXT,
                    metadata TEXT
                )
            ''')
            
            # Knowledge relationships table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge_relationships (
                    id INTEGER PRIMARY KEY,
                    source_node TEXT,
                    target_node TEXT,
                    relationship_type TEXT,
                    strength REAL,
                    created_at TIMESTAMP,
                    FOREIGN KEY (source_node) REFERENCES knowledge_nodes (node_id),
                    FOREIGN KEY (target_node) REFERENCES knowledge_nodes (node_id)
                )
            ''')
            
            # Query history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS query_history (
                    query_id TEXT PRIMARY KEY,
                    query_text TEXT,
                    context TEXT,
                    response TEXT,
                    confidence REAL,
                    timestamp TIMESTAMP
                )
            ''')
            
            # Insights table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS insights (
                    insight_id TEXT PRIMARY KEY,
                    insight_type TEXT,
                    symbol TEXT,
                    timeframe TEXT,
                    title TEXT,
                    description TEXT,
                    actionable_advice TEXT,
                    confidence REAL,
                    risk_level TEXT,
                    expected_outcome TEXT,
                    supporting_patterns TEXT,
                    contradicting_factors TEXT,
                    historical_success_rate REAL,
                    market_conditions TEXT,
                    timestamp TIMESTAMP
                )
            ''')
            
            self.conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            raise
    
    async def _load_knowledge_base(self):
        """Load existing knowledge from database"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('SELECT * FROM knowledge_nodes')
            
            for row in cursor.fetchall():
                node = KnowledgeNode(
                    node_id=row['node_id'],
                    knowledge_type=KnowledgeType(row['knowledge_type']),
                    title=row['title'],
                    description=row['description'],
                    content=row['content'],
                    confidence=row['confidence'],
                    relevance_score=row['relevance_score'],
                    tags=json.loads(row['tags'] or '[]'),
                    symbols=json.loads(row['symbols'] or '[]'),
                    timeframe=row['timeframe'],
                    market_regime=row['market_regime'],
                    success_rate=row['success_rate'],
                    sample_size=row['sample_size'],
                    last_updated=datetime.fromisoformat(row['last_updated']),
                    created_at=datetime.fromisoformat(row['created_at']),
                    source=row['source'],
                    metadata=json.loads(row['metadata'] or '{}')
                )
                
                self.knowledge_nodes[node.node_id] = node
                self._index_knowledge_node(node)
            
            # Load base knowledge if database is empty
            if not self.knowledge_nodes:
                await self._load_base_knowledge()
            
            self.metrics["knowledge_nodes"] = len(self.knowledge_nodes)
            self.logger.info(f"Loaded {len(self.knowledge_nodes)} knowledge nodes")
            
        except Exception as e:
            self.logger.error(f"Error loading knowledge base: {e}")
    
    async def _load_base_knowledge(self):
        """Load base trading knowledge"""
        try:
            # Load knowledge from each category
            for knowledge_type, loader_func in self.knowledge_categories.items():
                await loader_func()
            
            # Save to database
            await self._save_knowledge_to_db()
            
        except Exception as e:
            self.logger.error(f"Error loading base knowledge: {e}")
    
    async def _load_technical_patterns(self):
        """Load technical pattern knowledge"""
        try:
            patterns = [
                {
                    "title": "Double Bottom Pattern",
                    "description": "A bullish reversal pattern that occurs after a downtrend",
                    "content": "The double bottom pattern is characterized by two consecutive troughs at approximately the same price level, separated by a peak. This pattern suggests that the downtrend is losing momentum and a reversal to the upside is likely. Volume should ideally decrease on the second bottom and increase on the breakout above the neckline.",
                    "tags": ["reversal", "bullish", "support", "volume"],
                    "success_rate": 0.72,
                    "sample_size": 1500,
                    "timeframe": "daily"
                },
                {
                    "title": "Head and Shoulders",
                    "description": "A bearish reversal pattern with three peaks",
                    "content": "The head and shoulders pattern consists of three peaks: a higher peak (head) between two lower peaks (shoulders). This pattern indicates a potential trend reversal from bullish to bearish. The pattern is confirmed when price breaks below the neckline connecting the two troughs.",
                    "tags": ["reversal", "bearish", "resistance", "volume"],
                    "success_rate": 0.68,
                    "sample_size": 1200,
                    "timeframe": "daily"
                },
                {
                    "title": "Fibonacci Retracement",
                    "description": "Key retracement levels based on Fibonacci ratios",
                    "content": "Fibonacci retracement levels (23.6%, 38.2%, 50%, 61.8%, 78.6%) are horizontal lines that indicate potential support and resistance levels. These levels are derived from the Fibonacci sequence and are widely watched by traders. The 61.8% level is particularly significant as it often acts as strong support or resistance.",
                    "tags": ["support", "resistance", "fibonacci", "retracement"],
                    "success_rate": 0.65,
                    "sample_size": 2000,
                    "timeframe": "multiple"
                },
                {
                    "title": "RSI Divergence",
                    "description": "Momentum divergence between price and RSI indicator",
                    "content": "RSI divergence occurs when price makes a new high or low but RSI fails to confirm it. Bullish divergence happens when price makes a lower low but RSI makes a higher low, suggesting upward momentum. Bearish divergence occurs when price makes a higher high but RSI makes a lower high, indicating weakening momentum.",
                    "tags": ["momentum", "divergence", "rsi", "reversal"],
                    "success_rate": 0.58,
                    "sample_size": 800,
                    "timeframe": "multiple"
                }
            ]
            
            for pattern in patterns:
                await self._add_knowledge_node(
                    knowledge_type=KnowledgeType.TECHNICAL_PATTERN,
                    title=pattern["title"],
                    description=pattern["description"],
                    content=pattern["content"],
                    tags=pattern["tags"],
                    success_rate=pattern["success_rate"],
                    sample_size=pattern["sample_size"],
                    timeframe=pattern["timeframe"],
                    confidence=0.8,
                    source="base_knowledge"
                )
            
        except Exception as e:
            self.logger.error(f"Error loading technical patterns: {e}")
    
    async def _load_market_concepts(self):
        """Load market concept knowledge"""
        try:
            concepts = [
                {
                    "title": "Market Regime",
                    "description": "Different market environments with distinct characteristics",
                    "content": "Market regimes are distinct periods characterized by different risk-return profiles, volatility levels, and correlation patterns. Common regimes include bull markets (sustained upward trends), bear markets (sustained downward trends), and sideways markets (range-bound trading). Understanding the current regime is crucial for strategy selection.",
                    "tags": ["regime", "market", "volatility", "correlation"],
                    "success_rate": 0.75,
                    "sample_size": 500
                },
                {
                    "title": "Volatility Clustering",
                    "description": "The tendency for volatility to cluster in time",
                    "content": "Volatility clustering refers to the empirical observation that periods of high volatility tend to be followed by periods of high volatility, and periods of low volatility tend to be followed by periods of low volatility. This phenomenon is important for risk management and option pricing.",
                    "tags": ["volatility", "clustering", "risk", "options"],
                    "success_rate": 0.82,
                    "sample_size": 1000
                },
                {
                    "title": "Mean Reversion",
                    "description": "The tendency of prices to return to their long-term average",
                    "content": "Mean reversion is the theory that prices and returns eventually move back toward their long-term average. This concept is fundamental to many trading strategies and suggests that extreme price movements are often followed by movements in the opposite direction.",
                    "tags": ["mean_reversion", "pricing", "strategy", "average"],
                    "success_rate": 0.62,
                    "sample_size": 1500
                }
            ]
            
            for concept in concepts:
                await self._add_knowledge_node(
                    knowledge_type=KnowledgeType.MARKET_CONCEPT,
                    title=concept["title"],
                    description=concept["description"],
                    content=concept["content"],
                    tags=concept["tags"],
                    success_rate=concept["success_rate"],
                    sample_size=concept["sample_size"],
                    confidence=0.9,
                    source="base_knowledge"
                )
            
        except Exception as e:
            self.logger.error(f"Error loading market concepts: {e}")
    
    async def _load_trading_rules(self):
        """Load trading rule knowledge"""
        try:
            rules = [
                {
                    "title": "Cut Losses Short, Let Profits Run",
                    "description": "Fundamental risk management principle",
                    "content": "This classic trading rule emphasizes the importance of limiting losses while allowing profitable trades to continue. By setting stop-losses to limit downside risk and using trailing stops or profit targets that exceed the stop-loss distance, traders can maintain a positive risk-reward ratio even with a lower win rate.",
                    "tags": ["risk_management", "stop_loss", "profit_target", "ratio"],
                    "success_rate": 0.85,
                    "sample_size": 2000
                },
                {
                    "title": "Trade with the Trend",
                    "description": "Align trades with the prevailing market direction",
                    "content": "Trading with the trend increases the probability of success by aligning positions with the underlying market momentum. Trend-following strategies typically have higher win rates and can capture significant moves. However, traders must be prepared for trend reversals and drawdowns during consolidation periods.",
                    "tags": ["trend", "momentum", "direction", "probability"],
                    "success_rate": 0.73,
                    "sample_size": 1800
                },
                {
                    "title": "Position Sizing",
                    "description": "Determine appropriate position size based on risk",
                    "content": "Position sizing is the process of determining how much capital to allocate to each trade based on the risk level and account size. Common methods include fixed percentage risk, Kelly Criterion, and volatility-based sizing. Proper position sizing is crucial for long-term survival and growth.",
                    "tags": ["position_sizing", "risk", "capital", "kelly"],
                    "success_rate": 0.78,
                    "sample_size": 1200
                }
            ]
            
            for rule in rules:
                await self._add_knowledge_node(
                    knowledge_type=KnowledgeType.TRADING_RULE,
                    title=rule["title"],
                    description=rule["description"],
                    content=rule["content"],
                    tags=rule["tags"],
                    success_rate=rule["success_rate"],
                    sample_size=rule["sample_size"],
                    confidence=0.9,
                    source="base_knowledge"
                )
            
        except Exception as e:
            self.logger.error(f"Error loading trading rules: {e}")
    
    async def _load_risk_principles(self):
        """Load risk management principles"""
        try:
            principles = [
                {
                    "title": "Diversification",
                    "description": "Spread risk across multiple assets and strategies",
                    "content": "Diversification reduces portfolio risk by spreading investments across different assets, sectors, strategies, and time frames. The key is to combine assets with low or negative correlations to achieve risk reduction without proportional return reduction. However, diversification provides limited protection during systemic market stress.",
                    "tags": ["diversification", "correlation", "risk_reduction", "portfolio"],
                    "success_rate": 0.80,
                    "sample_size": 1000
                },
                {
                    "title": "Value at Risk (VaR)",
                    "description": "Quantitative measure of potential losses",
                    "content": "Value at Risk estimates the maximum potential loss over a specific time period at a given confidence level. VaR helps traders understand their risk exposure and set appropriate position sizes. However, VaR has limitations, including the assumption of normal distributions and the inability to capture tail risks.",
                    "tags": ["var", "risk_measure", "quantitative", "loss"],
                    "success_rate": 0.70,
                    "sample_size": 800
                },
                {
                    "title": "Correlation Risk",
                    "description": "Risk of correlated assets moving together",
                    "content": "Correlation risk arises when supposedly diversified assets move in the same direction during market stress. This risk is particularly pronounced during financial crises when correlations tend to increase. Traders should monitor rolling correlations and adjust position sizes accordingly.",
                    "tags": ["correlation", "risk", "diversification", "crisis"],
                    "success_rate": 0.65,
                    "sample_size": 600
                }
            ]
            
            for principle in principles:
                await self._add_knowledge_node(
                    knowledge_type=KnowledgeType.RISK_PRINCIPLE,
                    title=principle["title"],
                    description=principle["description"],
                    content=principle["content"],
                    tags=principle["tags"],
                    success_rate=principle["success_rate"],
                    sample_size=principle["sample_size"],
                    confidence=0.85,
                    source="base_knowledge"
                )
            
        except Exception as e:
            self.logger.error(f"Error loading risk principles: {e}")
    
    async def _load_economic_indicators(self):
        """Load economic indicator knowledge"""
        try:
            indicators = [
                {
                    "title": "Federal Funds Rate",
                    "description": "Key interest rate set by the Federal Reserve",
                    "content": "The Federal Funds Rate is the target interest rate at which commercial banks lend to each other overnight. Changes in this rate affect borrowing costs throughout the economy and influence stock valuations, bond prices, and currency exchange rates. Rising rates typically strengthen the dollar and can pressure stock valuations.",
                    "tags": ["interest_rate", "fed", "monetary_policy", "economy"],
                    "success_rate": 0.75,
                    "sample_size": 400
                },
                {
                    "title": "Inflation Rate",
                    "description": "Measure of price level changes over time",
                    "content": "Inflation measures the rate at which the general level of prices for goods and services rises. Moderate inflation is generally positive for stocks and the economy, while high inflation can erode purchasing power and lead to tighter monetary policy. Deflation can signal economic weakness.",
                    "tags": ["inflation", "prices", "economy", "monetary_policy"],
                    "success_rate": 0.70,
                    "sample_size": 500
                },
                {
                    "title": "Unemployment Rate",
                    "description": "Percentage of labor force that is unemployed",
                    "content": "The unemployment rate is a key indicator of economic health. Low unemployment suggests a strong economy but may lead to wage inflation. High unemployment indicates economic weakness but may prompt stimulus measures. The relationship between unemployment and stock markets is complex and depends on economic context.",
                    "tags": ["unemployment", "labor", "economy", "stimulus"],
                    "success_rate": 0.68,
                    "sample_size": 450
                }
            ]
            
            for indicator in indicators:
                await self._add_knowledge_node(
                    knowledge_type=KnowledgeType.ECONOMIC_INDICATOR,
                    title=indicator["title"],
                    description=indicator["description"],
                    content=indicator["content"],
                    tags=indicator["tags"],
                    success_rate=indicator["success_rate"],
                    sample_size=indicator["sample_size"],
                    confidence=0.85,
                    source="base_knowledge"
                )
            
        except Exception as e:
            self.logger.error(f"Error loading economic indicators: {e}")
    
    async def _add_knowledge_node(self, knowledge_type: KnowledgeType, title: str, description: str, 
                                content: str, tags: List[str], success_rate: float = 0.0, 
                                sample_size: int = 0, symbols: List[str] = None, 
                                timeframe: str = "daily", market_regime: str = "normal",
                                confidence: float = 0.8, source: str = "user"):
        """Add a new knowledge node"""
        try:
            node_id = str(uuid.uuid4())
            
            node = KnowledgeNode(
                node_id=node_id,
                knowledge_type=knowledge_type,
                title=title,
                description=description,
                content=content,
                confidence=confidence,
                relevance_score=0.0,  # Will be calculated later
                tags=tags or [],
                symbols=symbols or [],
                timeframe=timeframe,
                market_regime=market_regime,
                success_rate=success_rate,
                sample_size=sample_size,
                last_updated=datetime.now(),
                created_at=datetime.now(),
                source=source
            )
            
            # Calculate relevance score
            node.relevance_score = self._calculate_relevance_score(node)
            
            # Store node
            self.knowledge_nodes[node_id] = node
            self._index_knowledge_node(node)
            
            # Add to knowledge graph
            self.knowledge_graph.add_node(node_id, **node.__dict__)
            
            return node_id
            
        except Exception as e:
            self.logger.error(f"Error adding knowledge node: {e}")
            return None
    
    def _index_knowledge_node(self, node: KnowledgeNode):
        """Index knowledge node for efficient retrieval"""
        try:
            # Index by type
            self.knowledge_index[f"type:{node.knowledge_type.value}"].append(node.node_id)
            
            # Index by tags
            for tag in node.tags:
                self.knowledge_index[f"tag:{tag}"].append(node.node_id)
            
            # Index by symbols
            for symbol in node.symbols:
                self.knowledge_index[f"symbol:{symbol}"].append(node.node_id)
            
            # Index by timeframe
            self.knowledge_index[f"timeframe:{node.timeframe}"].append(node.node_id)
            
            # Index by market regime
            self.knowledge_index[f"regime:{node.market_regime}"].append(node.node_id)
            
            # Index by confidence level
            confidence_level = self._get_confidence_level(node.confidence)
            self.knowledge_index[f"confidence:{confidence_level.value}"].append(node.node_id)
            
        except Exception as e:
            self.logger.error(f"Error indexing knowledge node: {e}")
    
    def _calculate_relevance_score(self, node: KnowledgeNode) -> float:
        """Calculate relevance score for a knowledge node"""
        try:
            score = 0.0
            
            # Base score from confidence
            score += node.confidence * 0.3
            
            # Score from success rate
            if node.success_rate > 0:
                score += node.success_rate * 0.3
            
            # Score from sample size (normalized)
            if node.sample_size > 0:
                sample_score = min(node.sample_size / 1000, 1.0)
                score += sample_score * 0.2
            
            # Score from recency
            days_old = (datetime.now() - node.created_at).days
            recency_score = max(0, 1 - days_old / 365)  # Decay over a year
            score += recency_score * 0.1
            
            # Score from source reliability
            source_scores = {
                "base_knowledge": 0.9,
                "expert_input": 0.8,
                "ai_generated": 0.6,
                "user_input": 0.5
            }
            score += source_scores.get(node.source, 0.5) * 0.1
            
            return min(score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating relevance score: {e}")
            return 0.5
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert confidence score to confidence level"""
        if confidence >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.6:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.4:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    async def _build_knowledge_graph(self):
        """Build knowledge graph with relationships"""
        try:
            # Add relationships between nodes
            for node_id, node in self.knowledge_nodes.items():
                # Find related nodes
                related_nodes = await self._find_related_nodes(node)
                
                for related_id, similarity in related_nodes[:5]:  # Top 5 related nodes
                    if similarity > 0.3:  # Minimum similarity threshold
                        self.knowledge_graph.add_edge(
                            node_id, related_id,
                            relationship_type="similarity",
                            strength=similarity
                        )
            
            self.logger.info(f"Built knowledge graph with {len(self.knowledge_graph.nodes)} nodes and {len(self.knowledge_graph.edges)} edges")
            
        except Exception as e:
            self.logger.error(f"Error building knowledge graph: {e}")
    
    async def _find_related_nodes(self, node: KnowledgeNode) -> List[Tuple[str, float]]:
        """Find related nodes using similarity metrics"""
        try:
            related_nodes = []
            
            for other_id, other_node in self.knowledge_nodes.items():
                if other_id == node.node_id:
                    continue
                
                similarity = self._calculate_node_similarity(node, other_node)
                if similarity > 0.1:  # Minimum similarity threshold
                    related_nodes.append((other_id, similarity))
            
            # Sort by similarity
            related_nodes.sort(key=lambda x: x[1], reverse=True)
            
            return related_nodes
            
        except Exception as e:
            self.logger.error(f"Error finding related nodes: {e}")
            return []
    
    def _calculate_node_similarity(self, node1: KnowledgeNode, node2: KnowledgeNode) -> float:
        """Calculate similarity between two nodes"""
        try:
            similarity = 0.0
            
            # Tag overlap
            common_tags = set(node1.tags) & set(node2.tags)
            if node1.tags and node2.tags:
                tag_similarity = len(common_tags) / len(set(node1.tags) | set(node2.tags))
                similarity += tag_similarity * 0.3
            
            # Symbol overlap
            common_symbols = set(node1.symbols) & set(node2.symbols)
            if node1.symbols and node2.symbols:
                symbol_similarity = len(common_symbols) / len(set(node1.symbols) | set(node2.symbols))
                similarity += symbol_similarity * 0.2
            
            # Knowledge type similarity
            if node1.knowledge_type == node2.knowledge_type:
                similarity += 0.2
            
            # Timeframe similarity
            if node1.timeframe == node2.timeframe:
                similarity += 0.1
            
            # Market regime similarity
            if node1.market_regime == node2.market_regime:
                similarity += 0.1
            
            # Text similarity (using TF-IDF)
            try:
                text1 = f"{node1.title} {node1.description} {node1.content}"
                text2 = f"{node2.title} {node2.description} {node2.content}"
                
                vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
                vectors = vectorizer.fit_transform([text1, text2])
                
                text_similarity = cosine_similarity(vectors)[0, 1]
                similarity += text_similarity * 0.1
                
            except:
                pass
            
            return min(similarity, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating node similarity: {e}")
            return 0.0
    
    async def _initialize_text_processing(self):
        """Initialize text processing for semantic search"""
        try:
            if not self.knowledge_nodes:
                return
            
            # Extract text from all nodes
            documents = []
            for node in self.knowledge_nodes.values():
                text = f"{node.title} {node.description} {node.content}"
                documents.append(text)
            
            # Fit vectorizer
            self.document_vectors = self.vectorizer.fit_transform(documents)
            
            self.logger.info("Text processing initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing text processing: {e}")
    
    async def _save_knowledge_to_db(self):
        """Save knowledge nodes to database"""
        try:
            cursor = self.conn.cursor()
            
            for node in self.knowledge_nodes.values():
                cursor.execute('''
                    INSERT OR REPLACE INTO knowledge_nodes 
                    (node_id, knowledge_type, title, description, content, confidence, 
                     relevance_score, tags, symbols, timeframe, market_regime, 
                     success_rate, sample_size, last_updated, created_at, source, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    node.node_id,
                    node.knowledge_type.value,
                    node.title,
                    node.description,
                    node.content,
                    node.confidence,
                    node.relevance_score,
                    json.dumps(node.tags),
                    json.dumps(node.symbols),
                    node.timeframe,
                    node.market_regime,
                    node.success_rate,
                    node.sample_size,
                    node.last_updated.isoformat(),
                    node.created_at.isoformat(),
                    node.source,
                    json.dumps(node.metadata)
                ))
            
            self.conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error saving knowledge to database: {e}")
    
    async def query(self, query_text: str, context: Dict[str, Any] = None) -> KnowledgeResponse:
        """Query the digital brain for knowledge"""
        try:
            start_time = datetime.now()
            
            # Create query object
            query_id = str(uuid.uuid4())
            query = KnowledgeQuery(
                query_id=query_id,
                query_text=query_text,
                context=context or {},
                symbols=context.get("symbols", []) if context else [],
                timeframe=context.get("timeframe", "daily") if context else "daily",
                market_conditions=context.get("market_conditions", {}) if context else {},
                priority=context.get("priority", 0) if context else 0,
                timestamp=datetime.now()
            )
            
            # Check cache first
            cache_key = self._generate_cache_key(query)
            if cache_key in self.query_cache:
                cached_response = self.query_cache[cache_key]
                if (datetime.now() - cached_response.timestamp).seconds < 300:  # 5 minutes
                    return cached_response
            
            # Find relevant knowledge nodes
            relevant_nodes = await self._find_relevant_nodes(query)
            
            # Generate insights and recommendations
            insights = await self._generate_insights(query, relevant_nodes)
            recommendations = await self._generate_recommendations(query, relevant_nodes)
            
            # Calculate confidence
            confidence = self._calculate_response_confidence(relevant_nodes)
            
            # Generate reasoning
            reasoning = await self._generate_reasoning(query, relevant_nodes, insights)
            
            # Find supporting evidence
            supporting_evidence = self._extract_supporting_evidence(relevant_nodes)
            
            # Identify risk factors
            risk_factors = self._identify_risk_factors(query, relevant_nodes)
            
            # Create response
            response = KnowledgeResponse(
                query_id=query_id,
                relevant_nodes=relevant_nodes,
                insights=insights,
                recommendations=recommendations,
                confidence=confidence,
                reasoning=reasoning,
                supporting_evidence=supporting_evidence,
                risk_factors=risk_factors,
                timestamp=datetime.now()
            )
            
            # Cache response
            self.query_cache[cache_key] = response
            
            # Update metrics
            self.metrics["queries_processed"] += 1
            self.metrics["response_time"] = (datetime.now() - start_time).total_seconds()
            
            # Save query to database
            await self._save_query_to_db(query, response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return KnowledgeResponse(
                query_id=query_id,
                relevant_nodes=[],
                insights=[],
                recommendations=[],
                confidence=0.0,
                reasoning="Error processing query",
                supporting_evidence=[],
                risk_factors=[],
                timestamp=datetime.now()
            )
    
    def _generate_cache_key(self, query: KnowledgeQuery) -> str:
        """Generate cache key for query"""
        key_parts = [
            query.query_text,
            str(sorted(query.symbols)),
            query.timeframe,
            str(query.market_conditions)
        ]
        
        import hashlib
        return hashlib.md5("".join(key_parts).encode()).hexdigest()
    
    async def _find_relevant_nodes(self, query: KnowledgeQuery) -> List[KnowledgeNode]:
        """Find relevant knowledge nodes for query"""
        try:
            relevant_nodes = []
            
            # Text-based search
            if self.document_vectors is not None:
                query_vector = self.vectorizer.transform([query.query_text])
                similarities = cosine_similarity(query_vector, self.document_vectors)[0]
                
                # Get top matches
                top_indices = similarities.argsort()[-10:][::-1]  # Top 10
                
                for idx in top_indices:
                    if similarities[idx] > 0.1:  # Minimum similarity threshold
                        node_id = list(self.knowledge_nodes.keys())[idx]
                        node = self.knowledge_nodes[node_id]
                        relevant_nodes.append(node)
            
            # Symbol-based search
            if query.symbols:
                for symbol in query.symbols:
                    symbol_nodes = self.knowledge_index.get(f"symbol:{symbol}", [])
                    for node_id in symbol_nodes:
                        node = self.knowledge_nodes[node_id]
                        if node not in relevant_nodes:
                            relevant_nodes.append(node)
            
            # Timeframe-based search
            timeframe_nodes = self.knowledge_index.get(f"timeframe:{query.timeframe}", [])
            for node_id in timeframe_nodes:
                node = self.knowledge_nodes[node_id]
                if node not in relevant_nodes:
                    relevant_nodes.append(node)
            
            # Market condition-based search
            if query.market_conditions:
                regime = query.market_conditions.get("regime", "normal")
                regime_nodes = self.knowledge_index.get(f"regime:{regime}", [])
                for node_id in regime_nodes:
                    node = self.knowledge_nodes[node_id]
                    if node not in relevant_nodes:
                        relevant_nodes.append(node)
            
            # Sort by relevance score
            relevant_nodes.sort(key=lambda x: x.relevance_score, reverse=True)
            
            return relevant_nodes[:20]  # Return top 20 most relevant
            
        except Exception as e:
            self.logger.error(f"Error finding relevant nodes: {e}")
            return []
    
    async def _generate_insights(self, query: KnowledgeQuery, relevant_nodes: List[KnowledgeNode]) -> List[str]:
        """Generate insights from relevant nodes"""
        try:
            insights = []
            
            # Extract key insights from nodes
            for node in relevant_nodes[:5]:  # Top 5 most relevant
                if node.knowledge_type == KnowledgeType.TECHNICAL_PATTERN:
                    insight = f"Technical pattern '{node.title}' has a {node.success_rate:.1%} success rate based on {node.sample_size} samples"
                    insights.append(insight)
                
                elif node.knowledge_type == KnowledgeType.TRADING_RULE:
                    insight = f"Trading rule: {node.title} - {node.description}"
                    insights.append(insight)
                
                elif node.knowledge_type == KnowledgeType.RISK_PRINCIPLE:
                    insight = f"Risk consideration: {node.title} - {node.description}"
                    insights.append(insight)
                
                elif node.knowledge_type == KnowledgeType.MARKET_CONCEPT:
                    insight = f"Market concept: {node.title} - {node.description}"
                    insights.append(insight)
            
            # Generate AI-powered insights if available
            if self.anthropic_client or self.openai_client:
                ai_insights = await self._generate_ai_insights(query, relevant_nodes)
                insights.extend(ai_insights)
            
            return insights[:10]  # Return top 10 insights
            
        except Exception as e:
            self.logger.error(f"Error generating insights: {e}")
            return []
    
    async def _generate_ai_insights(self, query: KnowledgeQuery, relevant_nodes: List[KnowledgeNode]) -> List[str]:
        """Generate AI-powered insights"""
        try:
            insights = []
            
            # Prepare context for AI
            context_text = f"Query: {query.query_text}\n\n"
            context_text += "Relevant knowledge:\n"
            
            for node in relevant_nodes[:3]:  # Top 3 most relevant
                context_text += f"- {node.title}: {node.description}\n"
            
            if query.symbols:
                context_text += f"\nSymbols: {', '.join(query.symbols)}\n"
            
            if query.market_conditions:
                context_text += f"Market conditions: {query.market_conditions}\n"
            
            # Generate insights using AI
            if self.anthropic_client:
                # The newest Anthropic model is "claude-sonnet-4-20250514", not "claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20241022" nor "claude-3-sonnet-20240229"
                response = self.anthropic_client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=500,
                    messages=[
                        {
                            "role": "user",
                            "content": f"""Based on the following trading knowledge and query, provide 3 key insights for traders:

{context_text}

Please provide specific, actionable insights that combine the relevant knowledge with the query context."""
                        }
                    ]
                )
                
                ai_response = response.content[0].text.strip()
                
                # Extract insights from AI response
                lines = ai_response.split('\n')
                for line in lines:
                    if line.strip() and (line.startswith('-') or line.startswith('•') or line[0].isdigit()):
                        insight = line.strip().lstrip('1234567890.-• ')
                        if insight:
                            insights.append(insight)
            
            elif self.openai_client:
                # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    max_tokens=500,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a trading expert. Provide specific, actionable insights based on the knowledge provided."
                        },
                        {
                            "role": "user",
                            "content": f"Based on this trading knowledge and query, provide 3 key insights:\n\n{context_text}"
                        }
                    ]
                )
                
                ai_response = response.choices[0].message.content.strip()
                
                # Extract insights from AI response
                lines = ai_response.split('\n')
                for line in lines:
                    if line.strip() and (line.startswith('-') or line.startswith('•') or line[0].isdigit()):
                        insight = line.strip().lstrip('1234567890.-• ')
                        if insight:
                            insights.append(insight)
            
            return insights[:3]  # Return top 3 AI insights
            
        except Exception as e:
            self.logger.error(f"Error generating AI insights: {e}")
            return []
    
    async def _generate_recommendations(self, query: KnowledgeQuery, relevant_nodes: List[KnowledgeNode]) -> List[str]:
        """Generate actionable recommendations"""
        try:
            recommendations = []
            
            # Generate recommendations based on node types
            for node in relevant_nodes[:3]:  # Top 3 most relevant
                if node.knowledge_type == KnowledgeType.TECHNICAL_PATTERN:
                    if node.success_rate > 0.7:
                        recommendations.append(f"Consider watching for {node.title} pattern with {node.success_rate:.1%} success rate")
                
                elif node.knowledge_type == KnowledgeType.TRADING_RULE:
                    recommendations.append(f"Apply trading rule: {node.title}")
                
                elif node.knowledge_type == KnowledgeType.RISK_PRINCIPLE:
                    recommendations.append(f"Risk management: {node.title}")
            
            # Add specific recommendations based on query context
            if query.symbols:
                recommendations.append(f"Monitor {', '.join(query.symbols[:3])} for relevant patterns and signals")
            
            if query.market_conditions:
                volatility = query.market_conditions.get("volatility", 0)
                if volatility > 0.3:
                    recommendations.append("Consider reducing position sizes due to high volatility")
                elif volatility < 0.1:
                    recommendations.append("Low volatility environment - consider volatility breakout strategies")
            
            return recommendations[:5]  # Return top 5 recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return []
    
    def _calculate_response_confidence(self, relevant_nodes: List[KnowledgeNode]) -> float:
        """Calculate confidence in response"""
        try:
            if not relevant_nodes:
                return 0.0
            
            # Base confidence from node relevance
            avg_relevance = sum(node.relevance_score for node in relevant_nodes) / len(relevant_nodes)
            
            # Adjust for number of relevant nodes
            node_count_factor = min(len(relevant_nodes) / 5, 1.0)
            
            # Adjust for consistency of confidence across nodes
            confidences = [node.confidence for node in relevant_nodes]
            consistency_factor = 1.0 - (np.std(confidences) if len(confidences) > 1 else 0.0)
            
            # Combined confidence
            confidence = avg_relevance * 0.5 + node_count_factor * 0.3 + consistency_factor * 0.2
            
            return min(confidence, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating response confidence: {e}")
            return 0.5
    
    async def _generate_reasoning(self, query: KnowledgeQuery, relevant_nodes: List[KnowledgeNode], insights: List[str]) -> str:
        """Generate reasoning for the response"""
        try:
            reasoning_parts = []
            
            # Explain relevance of top nodes
            for node in relevant_nodes[:3]:
                reasoning_parts.append(f"'{node.title}' is relevant because it has a {node.confidence:.1%} confidence rating and {node.success_rate:.1%} success rate")
            
            # Explain query context matching
            if query.symbols:
                reasoning_parts.append(f"Analysis focused on {', '.join(query.symbols)} based on query context")
            
            # Explain market conditions consideration
            if query.market_conditions:
                reasoning_parts.append(f"Considered current market conditions: {query.market_conditions}")
            
            reasoning = "Analysis based on: " + "; ".join(reasoning_parts)
            
            return reasoning
            
        except Exception as e:
            self.logger.error(f"Error generating reasoning: {e}")
            return "Analysis based on available knowledge base"
    
    def _extract_supporting_evidence(self, relevant_nodes: List[KnowledgeNode]) -> List[str]:
        """Extract supporting evidence from nodes"""
        try:
            evidence = []
            
            for node in relevant_nodes[:3]:
                if node.sample_size > 0:
                    evidence.append(f"{node.title}: {node.sample_size} historical samples with {node.success_rate:.1%} success rate")
                
                if node.source != "user_input":
                    evidence.append(f"{node.title}: From {node.source} with {node.confidence:.1%} confidence")
            
            return evidence
            
        except Exception as e:
            self.logger.error(f"Error extracting supporting evidence: {e}")
            return []
    
    def _identify_risk_factors(self, query: KnowledgeQuery, relevant_nodes: List[KnowledgeNode]) -> List[str]:
        """Identify risk factors from analysis"""
        try:
            risk_factors = []
            
            # Check for low success rates
            for node in relevant_nodes:
                if node.success_rate < 0.5 and node.success_rate > 0:
                    risk_factors.append(f"Low success rate for {node.title}: {node.success_rate:.1%}")
            
            # Check for high volatility
            if query.market_conditions:
                volatility = query.market_conditions.get("volatility", 0)
                if volatility > 0.3:
                    risk_factors.append(f"High market volatility: {volatility:.1%}")
            
            # Check for low confidence
            low_confidence_nodes = [node for node in relevant_nodes if node.confidence < 0.5]
            if low_confidence_nodes:
                risk_factors.append(f"Low confidence in {len(low_confidence_nodes)} relevant knowledge areas")
            
            return risk_factors
            
        except Exception as e:
            self.logger.error(f"Error identifying risk factors: {e}")
            return []
    
    async def _save_query_to_db(self, query: KnowledgeQuery, response: KnowledgeResponse):
        """Save query and response to database"""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
                INSERT INTO query_history 
                (query_id, query_text, context, response, confidence, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                query.query_id,
                query.query_text,
                json.dumps(query.context),
                json.dumps({
                    "insights": response.insights,
                    "recommendations": response.recommendations,
                    "reasoning": response.reasoning
                }),
                response.confidence,
                query.timestamp.isoformat()
            ))
            
            self.conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error saving query to database: {e}")
    
    async def analyze_trading_opportunity(self, symbol: str, market_data: Dict[str, Any], 
                                       technical_analysis: Dict[str, Any], 
                                       strategy_type: str) -> Dict[str, Any]:
        """Analyze trading opportunity using digital brain"""
        try:
            # Create query context
            context = {
                "symbols": [symbol],
                "strategy_type": strategy_type,
                "market_data": market_data,
                "technical_analysis": technical_analysis,
                "market_conditions": {
                    "volatility": technical_analysis.get("volatility", {}).get("historical_volatility", 0),
                    "regime": self.current_market_regime
                }
            }
            
            # Generate query text
            query_text = f"Analyze trading opportunity for {symbol} using {strategy_type} strategy"
            
            # Query digital brain
            response = await self.query(query_text, context)
            
            # Generate trading signals
            signal_strength = min(response.confidence, 1.0)
            
            # Determine signal type
            signal_type = "hold"
            if signal_strength > 0.7:
                # Use technical analysis to determine direction
                if technical_analysis.get("signals", {}).get("overall_signal") == "buy":
                    signal_type = "buy"
                elif technical_analysis.get("signals", {}).get("overall_signal") == "sell":
                    signal_type = "sell"
            
            # Calculate stop loss and take profit
            current_price = market_data.get("close", 0)
            volatility = technical_analysis.get("volatility", {}).get("historical_volatility", 0.02)
            
            if signal_type == "buy":
                stop_loss = current_price * (1 - volatility * 2)
                take_profit = current_price * (1 + volatility * 3)
            elif signal_type == "sell":
                stop_loss = current_price * (1 + volatility * 2)
                take_profit = current_price * (1 - volatility * 3)
            else:
                stop_loss = current_price
                take_profit = current_price
            
            return {
                "signal_strength": signal_strength,
                "signal_type": signal_type,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "reasoning": response.reasoning,
                "insights": response.insights,
                "recommendations": response.recommendations,
                "risk_factors": response.risk_factors,
                "confidence": response.confidence
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing trading opportunity: {e}")
            return {
                "signal_strength": 0.0,
                "signal_type": "hold",
                "stop_loss": 0.0,
                "take_profit": 0.0,
                "reasoning": "Error in analysis",
                "insights": [],
                "recommendations": [],
                "risk_factors": ["Analysis error"],
                "confidence": 0.0
            }
    
    async def learn_from_feedback(self, feedback: Dict[str, Any]):
        """Learn from trading feedback"""
        try:
            if not self.learning_enabled:
                return
            
            # Store feedback
            self.feedback_history.append(feedback)
            
            # Extract learning signals
            success = feedback.get("success", False)
            strategy_type = feedback.get("strategy_type", "")
            symbol = feedback.get("symbol", "")
            pattern = feedback.get("pattern", "")
            
            # Update success rates for relevant patterns
            if pattern:
                for node in self.knowledge_nodes.values():
                    if node.knowledge_type == KnowledgeType.TECHNICAL_PATTERN and pattern in node.title:
                        # Update success rate
                        old_success = node.success_rate
                        old_sample_size = node.sample_size
                        
                        new_sample_size = old_sample_size + 1
                        if success:
                            new_success = (old_success * old_sample_size + 1) / new_sample_size
                        else:
                            new_success = (old_success * old_sample_size) / new_sample_size
                        
                        node.success_rate = new_success
                        node.sample_size = new_sample_size
                        node.last_updated = datetime.now()
                        
                        # Update confidence based on performance
                        if new_success > 0.7:
                            node.confidence = min(node.confidence + 0.01, 1.0)
                        elif new_success < 0.3:
                            node.confidence = max(node.confidence - 0.01, 0.1)
                        
                        break
            
            # Save updated knowledge
            await self._save_knowledge_to_db()
            
            self.logger.info(f"Learned from feedback: {feedback}")
            
        except Exception as e:
            self.logger.error(f"Error learning from feedback: {e}")
    
    async def add_document(self, document_path: str, document_type: str = "pdf"):
        """Add document to knowledge base"""
        try:
            # Process document
            processed_content = await self.document_processor.process_document(
                document_path, document_type
            )
            
            if processed_content:
                # Extract knowledge from document
                knowledge_nodes = await self._extract_knowledge_from_document(processed_content)
                
                # Add nodes to knowledge base
                for node_data in knowledge_nodes:
                    await self._add_knowledge_node(**node_data)
                
                # Rebuild text processing
                await self._initialize_text_processing()
                
                # Save to database
                await self._save_knowledge_to_db()
                
                self.logger.info(f"Added document to knowledge base: {document_path}")
                
        except Exception as e:
            self.logger.error(f"Error adding document: {e}")
    
    async def _extract_knowledge_from_document(self, content: str) -> List[Dict[str, Any]]:
        """Extract knowledge from document content"""
        try:
            knowledge_nodes = []
            
            # Use AI to extract knowledge if available
            if self.anthropic_client:
                # The newest Anthropic model is "claude-sonnet-4-20250514", not "claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20241022" nor "claude-3-sonnet-20240229"
                response = self.anthropic_client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1000,
                    messages=[
                        {
                            "role": "user",
                            "content": f"""Extract key trading knowledge from this document content. 
                            Identify patterns, rules, concepts, and principles.
                            
                            Content: {content[:2000]}
                            
                            Please provide structured knowledge in the following format:
                            - Title: [Knowledge title]
                            - Type: [technical_pattern, trading_rule, market_concept, risk_principle]
                            - Description: [Brief description]
                            - Content: [Detailed content]
                            - Tags: [relevant, tags]
                            """
                        }
                    ]
                )
                
                ai_response = response.content[0].text.strip()
                
                # Parse AI response to extract knowledge
                knowledge_nodes = self._parse_ai_knowledge_extraction(ai_response)
            
            return knowledge_nodes
            
        except Exception as e:
            self.logger.error(f"Error extracting knowledge from document: {e}")
            return []
    
    def _parse_ai_knowledge_extraction(self, ai_response: str) -> List[Dict[str, Any]]:
        """Parse AI knowledge extraction response"""
        try:
            knowledge_nodes = []
            
            # Simple parsing logic
            sections = ai_response.split("- Title:")
            
            for section in sections[1:]:  # Skip first empty section
                try:
                    lines = section.strip().split("\n")
                    
                    title = lines[0].strip()
                    knowledge_type = KnowledgeType.MARKET_CONCEPT  # Default
                    description = ""
                    content = ""
                    tags = []
                    
                    for line in lines[1:]:
                        if line.startswith("- Type:"):
                            type_str = line.replace("- Type:", "").strip()
                            try:
                                knowledge_type = KnowledgeType(type_str)
                            except:
                                knowledge_type = KnowledgeType.MARKET_CONCEPT
                        elif line.startswith("- Description:"):
                            description = line.replace("- Description:", "").strip()
                        elif line.startswith("- Content:"):
                            content = line.replace("- Content:", "").strip()
                        elif line.startswith("- Tags:"):
                            tags_str = line.replace("- Tags:", "").strip()
                            tags = [tag.strip() for tag in tags_str.split(",")]
                    
                    if title and description:
                        knowledge_nodes.append({
                            "knowledge_type": knowledge_type,
                            "title": title,
                            "description": description,
                            "content": content or description,
                            "tags": tags,
                            "confidence": 0.7,
                            "source": "document"
                        })
                
                except Exception as e:
                    self.logger.error(f"Error parsing knowledge section: {e}")
                    continue
            
            return knowledge_nodes
            
        except Exception as e:
            self.logger.error(f"Error parsing AI knowledge extraction: {e}")
            return []
    
    def get_knowledge_summary(self) -> Dict[str, Any]:
        """Get summary of knowledge base"""
        try:
            summary = {
                "total_nodes": len(self.knowledge_nodes),
                "by_type": {},
                "by_confidence": {},
                "recent_updates": 0,
                "avg_success_rate": 0.0
            }
            
            # Count by type
            for node in self.knowledge_nodes.values():
                type_name = node.knowledge_type.value
                summary["by_type"][type_name] = summary["by_type"].get(type_name, 0) + 1
            
            # Count by confidence level
            for node in self.knowledge_nodes.values():
                confidence_level = self._get_confidence_level(node.confidence).value
                summary["by_confidence"][confidence_level] = summary["by_confidence"].get(confidence_level, 0) + 1
            
            # Count recent updates (last 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            summary["recent_updates"] = sum(
                1 for node in self.knowledge_nodes.values()
                if node.last_updated > cutoff_time
            )
            
            # Calculate average success rate
            success_rates = [node.success_rate for node in self.knowledge_nodes.values() if node.success_rate > 0]
            if success_rates:
                summary["avg_success_rate"] = sum(success_rates) / len(success_rates)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting knowledge summary: {e}")
            return {}
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.conn:
                self.conn.close()
            
            self.logger.info("Digital Brain cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    # Public API methods
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.metrics.copy()
    
    def get_knowledge_nodes(self, knowledge_type: KnowledgeType = None) -> List[KnowledgeNode]:
        """Get knowledge nodes by type"""
        if knowledge_type:
            return [node for node in self.knowledge_nodes.values() if node.knowledge_type == knowledge_type]
        return list(self.knowledge_nodes.values())
    
    def search_knowledge(self, search_term: str, limit: int = 10) -> List[KnowledgeNode]:
        """Search knowledge base"""
        try:
            results = []
            search_term_lower = search_term.lower()
            
            for node in self.knowledge_nodes.values():
                score = 0
                
                # Check title
                if search_term_lower in node.title.lower():
                    score += 3
                
                # Check description
                if search_term_lower in node.description.lower():
                    score += 2
                
                # Check tags
                for tag in node.tags:
                    if search_term_lower in tag.lower():
                        score += 1
                
                # Check content
                if search_term_lower in node.content.lower():
                    score += 1
                
                if score > 0:
                    results.append((node, score))
            
            # Sort by score and return top results
            results.sort(key=lambda x: x[1], reverse=True)
            return [result[0] for result in results[:limit]]
            
        except Exception as e:
            self.logger.error(f"Error searching knowledge: {e}")
            return []
