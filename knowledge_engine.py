import logging
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import pickle
import hashlib
import sqlite3
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import asyncio

# Enhanced Knowledge Graph Components
@dataclass
class KnowledgeNode:
    """Represents a node in the knowledge graph"""
    node_id: str
    node_type: str  # 'pattern', 'indicator', 'company', 'sector', 'event', 'strategy'
    attributes: Dict[str, Any]
    timestamp: datetime
    confidence: float = 1.0
    source: str = "system"
    exchange: str = "all"  # Multi-exchange support
    
@dataclass
class KnowledgeEdge:
    """Represents a relationship in the knowledge graph"""
    edge_id: str
    source_node: str
    target_node: str
    relationship_type: str  # 'correlates_with', 'predicts', 'influences', 'contains'
    strength: float  # -1.0 to 1.0
    attributes: Dict[str, Any]
    timestamp: datetime
    confidence: float = 1.0
    exchange: str = "all"  # Multi-exchange support

@dataclass
class MarketPattern:
    """Represents a learned market pattern"""
    pattern_id: str
    pattern_type: str  # 'technical', 'fundamental', 'sentiment', 'macro'
    conditions: Dict[str, Any]
    outcomes: Dict[str, Any]
    success_rate: float
    sample_size: int
    last_seen: datetime
    symbols: List[str]
    market_regime: str = "normal"
    exchange: str = "all"  # Multi-exchange support
    fibonacci_levels: Dict[str, float] = field(default_factory=dict)

@dataclass
class DocumentEntity:
    """Represents a processed document entity"""
    entity_id: str
    entity_type: str  # 'earnings_report', 'news_article', 'research_report', 'sec_filing'
    content: str
    metadata: Dict[str, Any]
    extracted_facts: List[Dict[str, Any]]
    sentiment_score: float
    relevance_score: float
    timestamp: datetime
    symbols: List[str]
    exchange: str = "all"  # Multi-exchange support

class KnowledgeEngine:
    """Enhanced knowledge engine with multi-exchange support and advanced analytics"""
    
    def __init__(self, db_path: str = "knowledge_graph.db"):
        self.db_path = db_path
        self.logger = logging.getLogger("KnowledgeEngine")
        
        # Initialize database
        self._init_database()
        
        # In-memory indexes for fast access
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: Dict[str, KnowledgeEdge] = {}
        self.patterns: Dict[str, MarketPattern] = {}
        self.documents: Dict[str, DocumentEntity] = {}
        
        # Advanced indexing
        self.node_index: Dict[str, Set[str]] = defaultdict(set)
        self.edge_index: Dict[str, Set[str]] = defaultdict(set)
        self.pattern_index: Dict[str, Set[str]] = defaultdict(set)
        self.symbol_index: Dict[str, Set[str]] = defaultdict(set)
        self.exchange_index: Dict[str, Set[str]] = defaultdict(set)
        self.temporal_index: Dict[str, List[str]] = defaultdict(list)
        
        # TF-IDF for semantic search
        self.tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        self.document_vectors = None
        self.document_ids = []
        
        # Multi-exchange correlation matrix
        self.correlation_matrix = {}
        
        # Load existing data
        self._load_from_database()
        
        # Performance metrics
        self.query_stats = defaultdict(lambda: {'count': 0, 'total_time': 0})
        
        self.logger.info("Knowledge Engine initialized with multi-exchange support")
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_nodes (
                    node_id TEXT PRIMARY KEY,
                    node_type TEXT,
                    attributes TEXT,
                    timestamp TEXT,
                    confidence REAL,
                    source TEXT,
                    exchange TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_edges (
                    edge_id TEXT PRIMARY KEY,
                    source_node TEXT,
                    target_node TEXT,
                    relationship_type TEXT,
                    strength REAL,
                    attributes TEXT,
                    timestamp TEXT,
                    confidence REAL,
                    exchange TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    pattern_type TEXT,
                    conditions TEXT,
                    outcomes TEXT,
                    success_rate REAL,
                    sample_size INTEGER,
                    last_seen TEXT,
                    symbols TEXT,
                    market_regime TEXT,
                    exchange TEXT,
                    fibonacci_levels TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    entity_id TEXT PRIMARY KEY,
                    entity_type TEXT,
                    content TEXT,
                    metadata TEXT,
                    extracted_facts TEXT,
                    sentiment_score REAL,
                    relevance_score REAL,
                    timestamp TEXT,
                    symbols TEXT,
                    exchange TEXT
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_node_type ON knowledge_nodes(node_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_exchange ON knowledge_nodes(exchange)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pattern_type ON market_patterns(pattern_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON knowledge_nodes(timestamp)")
            
            conn.commit()
            conn.close()
            
            self.logger.info("Database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            raise
    
    def _load_from_database(self):
        """Load existing data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Load nodes
            cursor.execute("SELECT * FROM knowledge_nodes")
            for row in cursor.fetchall():
                node = KnowledgeNode(
                    node_id=row[0],
                    node_type=row[1],
                    attributes=json.loads(row[2]),
                    timestamp=datetime.fromisoformat(row[3]),
                    confidence=row[4],
                    source=row[5],
                    exchange=row[6]
                )
                self.nodes[node.node_id] = node
                self._update_indexes_for_node(node)
            
            # Load edges
            cursor.execute("SELECT * FROM knowledge_edges")
            for row in cursor.fetchall():
                edge = KnowledgeEdge(
                    edge_id=row[0],
                    source_node=row[1],
                    target_node=row[2],
                    relationship_type=row[3],
                    strength=row[4],
                    attributes=json.loads(row[5]),
                    timestamp=datetime.fromisoformat(row[6]),
                    confidence=row[7],
                    exchange=row[8]
                )
                self.edges[edge.edge_id] = edge
                self._update_indexes_for_edge(edge)
            
            # Load patterns
            cursor.execute("SELECT * FROM market_patterns")
            for row in cursor.fetchall():
                pattern = MarketPattern(
                    pattern_id=row[0],
                    pattern_type=row[1],
                    conditions=json.loads(row[2]),
                    outcomes=json.loads(row[3]),
                    success_rate=row[4],
                    sample_size=row[5],
                    last_seen=datetime.fromisoformat(row[6]),
                    symbols=json.loads(row[7]),
                    market_regime=row[8],
                    exchange=row[9],
                    fibonacci_levels=json.loads(row[10]) if row[10] else {}
                )
                self.patterns[pattern.pattern_id] = pattern
                self._update_indexes_for_pattern(pattern)
            
            # Load documents
            cursor.execute("SELECT * FROM documents")
            for row in cursor.fetchall():
                document = DocumentEntity(
                    entity_id=row[0],
                    entity_type=row[1],
                    content=row[2],
                    metadata=json.loads(row[3]),
                    extracted_facts=json.loads(row[4]),
                    sentiment_score=row[5],
                    relevance_score=row[6],
                    timestamp=datetime.fromisoformat(row[7]),
                    symbols=json.loads(row[8]),
                    exchange=row[9]
                )
                self.documents[document.entity_id] = document
            
            conn.close()
            
            self.logger.info(f"Loaded {len(self.nodes)} nodes, {len(self.edges)} edges, "
                           f"{len(self.patterns)} patterns, {len(self.documents)} documents")
            
        except Exception as e:
            self.logger.error(f"Error loading from database: {e}")
    
    def _update_indexes_for_node(self, node: KnowledgeNode):
        """Update indexes when adding a node"""
        self.node_index[node.node_type].add(node.node_id)
        self.exchange_index[node.exchange].add(node.node_id)
        
        # Symbol indexing
        symbols = node.attributes.get('symbols', [])
        for symbol in symbols:
            self.symbol_index[symbol.upper()].add(node.node_id)
        
        # Temporal indexing
        date_key = node.timestamp.strftime("%Y-%m")
        self.temporal_index[date_key].append(node.node_id)
    
    def _update_indexes_for_edge(self, edge: KnowledgeEdge):
        """Update indexes when adding an edge"""
        self.edge_index[edge.relationship_type].add(edge.edge_id)
        self.exchange_index[edge.exchange].add(edge.edge_id)
    
    def _update_indexes_for_pattern(self, pattern: MarketPattern):
        """Update indexes when adding a pattern"""
        self.pattern_index[pattern.pattern_type].add(pattern.pattern_id)
        self.exchange_index[pattern.exchange].add(pattern.pattern_id)
        
        for symbol in pattern.symbols:
            self.symbol_index[symbol.upper()].add(pattern.pattern_id)
    
    def add_node(self, node: KnowledgeNode) -> bool:
        """Add a node to the knowledge graph"""
        try:
            # Add to memory
            self.nodes[node.node_id] = node
            self._update_indexes_for_node(node)
            
            # Persist to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO knowledge_nodes 
                (node_id, node_type, attributes, timestamp, confidence, source, exchange)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                node.node_id,
                node.node_type,
                json.dumps(node.attributes),
                node.timestamp.isoformat(),
                node.confidence,
                node.source,
                node.exchange
            ))
            conn.commit()
            conn.close()
            
            self.logger.debug(f"Added node: {node.node_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding node {node.node_id}: {e}")
            return False
    
    def add_edge(self, edge: KnowledgeEdge) -> bool:
        """Add an edge to the knowledge graph"""
        try:
            # Add to memory
            self.edges[edge.edge_id] = edge
            self._update_indexes_for_edge(edge)
            
            # Persist to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO knowledge_edges 
                (edge_id, source_node, target_node, relationship_type, strength, attributes, timestamp, confidence, exchange)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                edge.edge_id,
                edge.source_node,
                edge.target_node,
                edge.relationship_type,
                edge.strength,
                json.dumps(edge.attributes),
                edge.timestamp.isoformat(),
                edge.confidence,
                edge.exchange
            ))
            conn.commit()
            conn.close()
            
            self.logger.debug(f"Added edge: {edge.edge_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding edge {edge.edge_id}: {e}")
            return False
    
    def add_pattern(self, pattern: MarketPattern) -> bool:
        """Add a market pattern to the knowledge graph"""
        try:
            # Add to memory
            self.patterns[pattern.pattern_id] = pattern
            self._update_indexes_for_pattern(pattern)
            
            # Persist to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO market_patterns 
                (pattern_id, pattern_type, conditions, outcomes, success_rate, sample_size, 
                 last_seen, symbols, market_regime, exchange, fibonacci_levels)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pattern.pattern_id,
                pattern.pattern_type,
                json.dumps(pattern.conditions),
                json.dumps(pattern.outcomes),
                pattern.success_rate,
                pattern.sample_size,
                pattern.last_seen.isoformat(),
                json.dumps(pattern.symbols),
                pattern.market_regime,
                pattern.exchange,
                json.dumps(pattern.fibonacci_levels)
            ))
            conn.commit()
            conn.close()
            
            self.logger.debug(f"Added pattern: {pattern.pattern_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding pattern {pattern.pattern_id}: {e}")
            return False
    
    def add_document(self, document: DocumentEntity) -> bool:
        """Add a document to the knowledge graph"""
        try:
            # Add to memory
            self.documents[document.entity_id] = document
            
            # Persist to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO documents 
                (entity_id, entity_type, content, metadata, extracted_facts, 
                 sentiment_score, relevance_score, timestamp, symbols, exchange)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                document.entity_id,
                document.entity_type,
                document.content,
                json.dumps(document.metadata),
                json.dumps(document.extracted_facts),
                document.sentiment_score,
                document.relevance_score,
                document.timestamp.isoformat(),
                json.dumps(document.symbols),
                document.exchange
            ))
            conn.commit()
            conn.close()
            
            self.logger.debug(f"Added document: {document.entity_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding document {document.entity_id}: {e}")
            return False
    
    def query_patterns(self, 
                      pattern_type: Optional[str] = None,
                      symbols: Optional[List[str]] = None,
                      exchange: Optional[str] = None,
                      min_success_rate: float = 0.0,
                      min_sample_size: int = 1,
                      include_fibonacci: bool = False) -> List[MarketPattern]:
        """Query market patterns with advanced filtering"""
        start_time = time.time()
        
        try:
            results = []
            
            # Start with all patterns or filter by type
            if pattern_type:
                pattern_ids = self.pattern_index.get(pattern_type, set())
            else:
                pattern_ids = set(self.patterns.keys())
            
            # Filter by exchange
            if exchange:
                exchange_patterns = self.exchange_index.get(exchange, set())
                pattern_ids = pattern_ids.intersection(exchange_patterns)
            
            # Filter by symbols
            if symbols:
                symbol_patterns = set()
                for symbol in symbols:
                    symbol_patterns.update(self.symbol_index.get(symbol.upper(), set()))
                pattern_ids = pattern_ids.intersection(symbol_patterns)
            
            # Apply filters
            for pattern_id in pattern_ids:
                pattern = self.patterns.get(pattern_id)
                if pattern and pattern.success_rate >= min_success_rate and pattern.sample_size >= min_sample_size:
                    if include_fibonacci and pattern.fibonacci_levels:
                        results.append(pattern)
                    elif not include_fibonacci:
                        results.append(pattern)
            
            # Sort by success rate
            results.sort(key=lambda x: x.success_rate, reverse=True)
            
            # Update query stats
            query_time = time.time() - start_time
            self.query_stats['query_patterns']['count'] += 1
            self.query_stats['query_patterns']['total_time'] += query_time
            
            self.logger.debug(f"Pattern query returned {len(results)} results in {query_time:.3f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"Error querying patterns: {e}")
            return []
    
    def query_nodes(self, 
                   node_type: Optional[str] = None,
                   exchange: Optional[str] = None,
                   symbols: Optional[List[str]] = None,
                   min_confidence: float = 0.0,
                   limit: int = 100) -> List[KnowledgeNode]:
        """Query knowledge nodes with advanced filtering"""
        start_time = time.time()
        
        try:
            results = []
            
            # Start with all nodes or filter by type
            if node_type:
                node_ids = self.node_index.get(node_type, set())
            else:
                node_ids = set(self.nodes.keys())
            
            # Filter by exchange
            if exchange:
                exchange_nodes = self.exchange_index.get(exchange, set())
                node_ids = node_ids.intersection(exchange_nodes)
            
            # Filter by symbols
            if symbols:
                symbol_nodes = set()
                for symbol in symbols:
                    symbol_nodes.update(self.symbol_index.get(symbol.upper(), set()))
                node_ids = node_ids.intersection(symbol_nodes)
            
            # Apply confidence filter and collect results
            for node_id in node_ids:
                node = self.nodes.get(node_id)
                if node and node.confidence >= min_confidence:
                    results.append(node)
                    if len(results) >= limit:
                        break
            
            # Sort by confidence
            results.sort(key=lambda x: x.confidence, reverse=True)
            
            # Update query stats
            query_time = time.time() - start_time
            self.query_stats['query_nodes']['count'] += 1
            self.query_stats['query_nodes']['total_time'] += query_time
            
            self.logger.debug(f"Node query returned {len(results)} results in {query_time:.3f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"Error querying nodes: {e}")
            return []
    
    def get_related_nodes(self, node_id: str, relationship_types: Optional[List[str]] = None, max_depth: int = 2) -> List[KnowledgeNode]:
        """Get nodes related to a given node"""
        try:
            related_nodes = []
            visited = set()
            queue = [(node_id, 0)]  # (node_id, depth)
            
            while queue:
                current_node_id, depth = queue.pop(0)
                
                if current_node_id in visited or depth > max_depth:
                    continue
                
                visited.add(current_node_id)
                
                if depth > 0:  # Don't include the original node
                    node = self.nodes.get(current_node_id)
                    if node:
                        related_nodes.append(node)
                
                # Find connected nodes
                for edge in self.edges.values():
                    if edge.source_node == current_node_id:
                        if not relationship_types or edge.relationship_type in relationship_types:
                            queue.append((edge.target_node, depth + 1))
                    elif edge.target_node == current_node_id:
                        if not relationship_types or edge.relationship_type in relationship_types:
                            queue.append((edge.source_node, depth + 1))
            
            return related_nodes
            
        except Exception as e:
            self.logger.error(f"Error getting related nodes for {node_id}: {e}")
            return []
    
    def calculate_cross_exchange_correlation(self, symbol: str, exchanges: List[str]) -> Dict[str, float]:
        """Calculate correlation between the same symbol across different exchanges"""
        try:
            correlations = {}
            
            # Get price data for the symbol from different exchanges
            price_data = {}
            for exchange in exchanges:
                # Query market data for this symbol and exchange
                patterns = self.query_patterns(symbols=[symbol], exchange=exchange)
                if patterns:
                    # Extract price movements from patterns
                    price_data[exchange] = [p.outcomes.get('price_change', 0) for p in patterns]
            
            # Calculate correlations
            for i, exchange1 in enumerate(exchanges):
                for exchange2 in exchanges[i+1:]:
                    if exchange1 in price_data and exchange2 in price_data:
                        data1 = price_data[exchange1]
                        data2 = price_data[exchange2]
                        
                        if len(data1) > 1 and len(data2) > 1:
                            # Align data by length
                            min_len = min(len(data1), len(data2))
                            correlation = np.corrcoef(data1[:min_len], data2[:min_len])[0, 1]
                            correlations[f"{exchange1}_{exchange2}"] = correlation
            
            return correlations
            
        except Exception as e:
            self.logger.error(f"Error calculating cross-exchange correlation for {symbol}: {e}")
            return {}
    
    def get_fibonacci_analysis(self, symbol: str, exchange: str = "all") -> Dict[str, Any]:
        """Get Fibonacci analysis for a symbol"""
        try:
            patterns = self.query_patterns(symbols=[symbol], exchange=exchange, include_fibonacci=True)
            
            if not patterns:
                return {"error": "No Fibonacci patterns found"}
            
            # Aggregate Fibonacci levels
            fib_levels = defaultdict(list)
            for pattern in patterns:
                for level, value in pattern.fibonacci_levels.items():
                    fib_levels[level].append(value)
            
            # Calculate statistics
            fib_stats = {}
            for level, values in fib_levels.items():
                if values:
                    fib_stats[level] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'count': len(values)
                    }
            
            return {
                'symbol': symbol,
                'exchange': exchange,
                'fibonacci_levels': fib_stats,
                'pattern_count': len(patterns),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting Fibonacci analysis for {symbol}: {e}")
            return {"error": str(e)}
    
    def semantic_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Perform semantic search across documents"""
        try:
            if not self.documents:
                return []
            
            # Prepare document texts
            if self.document_vectors is None:
                self._build_document_vectors()
            
            # Transform query
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
            
            # Get top results
            top_indices = similarities.argsort()[-limit:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0:
                    doc_id = self.document_ids[idx]
                    document = self.documents[doc_id]
                    results.append({
                        'document_id': doc_id,
                        'document_type': document.entity_type,
                        'content': document.content[:500] + "..." if len(document.content) > 500 else document.content,
                        'similarity_score': similarities[idx],
                        'sentiment_score': document.sentiment_score,
                        'symbols': document.symbols,
                        'exchange': document.exchange,
                        'timestamp': document.timestamp.isoformat()
                    })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in semantic search: {e}")
            return []
    
    def _build_document_vectors(self):
        """Build TF-IDF vectors for documents"""
        try:
            if not self.documents:
                return
            
            document_texts = []
            self.document_ids = []
            
            for doc_id, document in self.documents.items():
                document_texts.append(document.content)
                self.document_ids.append(doc_id)
            
            self.document_vectors = self.tfidf_vectorizer.fit_transform(document_texts)
            
            self.logger.info(f"Built TF-IDF vectors for {len(document_texts)} documents")
            
        except Exception as e:
            self.logger.error(f"Error building document vectors: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'total_patterns': len(self.patterns),
            'total_documents': len(self.documents),
            'query_performance': {}
        }
        
        # Calculate average query times
        for query_type, metrics in self.query_stats.items():
            if metrics['count'] > 0:
                stats['query_performance'][query_type] = {
                    'avg_time': metrics['total_time'] / metrics['count'],
                    'total_queries': metrics['count']
                }
        
        return stats
    
    def export_knowledge_graph(self, filepath: str) -> bool:
        """Export knowledge graph to file"""
        try:
            data = {
                'nodes': {k: {
                    'node_id': v.node_id,
                    'node_type': v.node_type,
                    'attributes': v.attributes,
                    'timestamp': v.timestamp.isoformat(),
                    'confidence': v.confidence,
                    'source': v.source,
                    'exchange': v.exchange
                } for k, v in self.nodes.items()},
                'edges': {k: {
                    'edge_id': v.edge_id,
                    'source_node': v.source_node,
                    'target_node': v.target_node,
                    'relationship_type': v.relationship_type,
                    'strength': v.strength,
                    'attributes': v.attributes,
                    'timestamp': v.timestamp.isoformat(),
                    'confidence': v.confidence,
                    'exchange': v.exchange
                } for k, v in self.edges.items()},
                'patterns': {k: {
                    'pattern_id': v.pattern_id,
                    'pattern_type': v.pattern_type,
                    'conditions': v.conditions,
                    'outcomes': v.outcomes,
                    'success_rate': v.success_rate,
                    'sample_size': v.sample_size,
                    'last_seen': v.last_seen.isoformat(),
                    'symbols': v.symbols,
                    'market_regime': v.market_regime,
                    'exchange': v.exchange,
                    'fibonacci_levels': v.fibonacci_levels
                } for k, v in self.patterns.items()}
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"Knowledge graph exported to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting knowledge graph: {e}")
            return False
    
    async def process_market_event(self, event: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a market event and extract insights"""
        try:
            insights = []
            
            # Extract event details
            event_type = event.get('type', 'unknown')
            symbol = event.get('symbol', '')
            exchange = event.get('exchange', 'all')
            timestamp = datetime.fromisoformat(event.get('timestamp', datetime.now().isoformat()))
            
            # Find related patterns
            related_patterns = self.query_patterns(
                symbols=[symbol] if symbol else None,
                exchange=exchange
            )
            
            # Generate insights based on patterns
            for pattern in related_patterns[:5]:  # Top 5 patterns
                insight = {
                    'pattern_id': pattern.pattern_id,
                    'pattern_type': pattern.pattern_type,
                    'success_rate': pattern.success_rate,
                    'expected_outcome': pattern.outcomes,
                    'confidence': pattern.success_rate * 0.8,  # Adjust confidence
                    'fibonacci_levels': pattern.fibonacci_levels,
                    'recommendation': self._generate_recommendation(pattern, event)
                }
                insights.append(insight)
            
            # Find cross-exchange opportunities
            if symbol and exchange != 'all':
                other_exchanges = ['alpaca', 'binance', 'td_ameritrade', 'kucoin']
                if exchange in other_exchanges:
                    other_exchanges.remove(exchange)
                
                correlations = self.calculate_cross_exchange_correlation(symbol, [exchange] + other_exchanges)
                if correlations:
                    insights.append({
                        'type': 'cross_exchange_analysis',
                        'symbol': symbol,
                        'correlations': correlations,
                        'arbitrage_opportunities': self._detect_arbitrage_opportunities(correlations)
                    })
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error processing market event: {e}")
            return []
    
    def _generate_recommendation(self, pattern: MarketPattern, event: Dict[str, Any]) -> str:
        """Generate trading recommendation based on pattern and event"""
        try:
            if pattern.success_rate > 0.7:
                confidence = "High"
            elif pattern.success_rate > 0.5:
                confidence = "Medium"
            else:
                confidence = "Low"
            
            outcome = pattern.outcomes.get('direction', 'neutral')
            
            if outcome == 'bullish':
                action = "Consider buying"
            elif outcome == 'bearish':
                action = "Consider selling"
            else:
                action = "Hold position"
            
            return f"{action} - {confidence} confidence ({pattern.success_rate:.1%} success rate)"
            
        except Exception as e:
            self.logger.error(f"Error generating recommendation: {e}")
            return "No recommendation available"
    
    def _detect_arbitrage_opportunities(self, correlations: Dict[str, float]) -> List[Dict[str, Any]]:
        """Detect potential arbitrage opportunities"""
        opportunities = []
        
        try:
            for pair, correlation in correlations.items():
                if correlation < 0.8:  # Low correlation might indicate arbitrage opportunity
                    exchanges = pair.split('_')
                    opportunities.append({
                        'exchanges': exchanges,
                        'correlation': correlation,
                        'opportunity_score': 1.0 - correlation,
                        'type': 'statistical_arbitrage'
                    })
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error detecting arbitrage opportunities: {e}")
            return []
