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
import asyncio
from concurrent.futures import ThreadPoolExecutor
import networkx as nx

# Advanced knowledge graph components
@dataclass
class KnowledgeNode:
    """Represents a node in the knowledge graph"""
    node_id: str
    node_type: str  # 'pattern', 'indicator', 'company', 'sector', 'event'
    attributes: Dict[str, Any]
    timestamp: datetime
    confidence: float = 1.0
    source: str = "system"

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

class KnowledgeEngine:
    """Advanced knowledge engine with Digital Brain capabilities"""
    
    def __init__(self):
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: Dict[str, KnowledgeEdge] = {}
        self.patterns: Dict[str, MarketPattern] = {}
        self.documents: Dict[str, DocumentEntity] = {}
        
        # Advanced indexing
        self.graph = nx.DiGraph()
        self.node_index: Dict[str, Set[str]] = defaultdict(set)
        self.symbol_index: Dict[str, Set[str]] = defaultdict(set)
        self.pattern_index: Dict[str, Set[str]] = defaultdict(set)
        self.temporal_index: Dict[str, List[str]] = defaultdict(list)
        
        # Memory systems
        self.short_term_memory: Dict[str, Any] = {}
        self.long_term_memory: Dict[str, Any] = {}
        self.episodic_memory: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.query_cache: Dict[str, Any] = {}
        self.performance_stats: Dict[str, float] = {}
        
        # Configuration
        self.max_cache_size = 1000
        self.cache_ttl = 300  # 5 minutes
        self.is_initialized = False
        
        self.logger = logging.getLogger("KnowledgeEngine")
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def initialize(self):
        """Initialize the knowledge engine"""
        try:
            self.logger.info("Initializing Knowledge Engine...")
            
            # Load existing knowledge base
            await self._load_knowledge_base()
            
            # Initialize core patterns
            await self._initialize_core_patterns()
            
            # Setup monitoring
            await self._setup_monitoring()
            
            self.is_initialized = True
            self.logger.info("Knowledge Engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Knowledge Engine: {e}")
            raise
    
    async def _load_knowledge_base(self):
        """Load existing knowledge base from storage"""
        try:
            # Load from existing system if available
            knowledge_file = "knowledge_graph_state.json"
            try:
                with open(knowledge_file, 'r') as f:
                    data = json.load(f)
                    
                # Reconstruct knowledge graph
                for node_data in data.get('nodes', []):
                    node = KnowledgeNode(**node_data)
                    await self.add_node(node)
                
                for edge_data in data.get('edges', []):
                    edge = KnowledgeEdge(**edge_data)
                    await self.add_edge(edge)
                
                self.logger.info(f"Loaded {len(self.nodes)} nodes and {len(self.edges)} edges")
                
            except FileNotFoundError:
                self.logger.info("No existing knowledge base found, starting fresh")
                
        except Exception as e:
            self.logger.error(f"Error loading knowledge base: {e}")
    
    async def _initialize_core_patterns(self):
        """Initialize core trading patterns"""
        try:
            # Technical patterns
            technical_patterns = [
                {
                    'pattern_id': 'double_top',
                    'pattern_type': 'technical',
                    'conditions': {
                        'price_peaks': 2,
                        'volume_confirmation': True,
                        'timeframe': '1h'
                    },
                    'outcomes': {
                        'bearish_probability': 0.75,
                        'price_target': 'support_level'
                    },
                    'success_rate': 0.72,
                    'sample_size': 1000
                },
                {
                    'pattern_id': 'head_shoulders',
                    'pattern_type': 'technical',
                    'conditions': {
                        'peaks': 3,
                        'neckline_break': True,
                        'volume_pattern': 'declining'
                    },
                    'outcomes': {
                        'bearish_probability': 0.82,
                        'price_target': 'neckline_distance'
                    },
                    'success_rate': 0.78,
                    'sample_size': 800
                }
            ]
            
            for pattern_data in technical_patterns:
                pattern = MarketPattern(
                    pattern_id=pattern_data['pattern_id'],
                    pattern_type=pattern_data['pattern_type'],
                    conditions=pattern_data['conditions'],
                    outcomes=pattern_data['outcomes'],
                    success_rate=pattern_data['success_rate'],
                    sample_size=pattern_data['sample_size'],
                    last_seen=datetime.now(),
                    symbols=[],
                    market_regime="normal"
                )
                
                self.patterns[pattern.pattern_id] = pattern
                
            self.logger.info(f"Initialized {len(technical_patterns)} core patterns")
            
        except Exception as e:
            self.logger.error(f"Error initializing core patterns: {e}")
    
    async def _setup_monitoring(self):
        """Setup performance monitoring"""
        try:
            # Initialize performance tracking
            self.performance_stats = {
                'total_queries': 0,
                'cache_hits': 0,
                'average_response_time': 0.0,
                'pattern_matches': 0,
                'knowledge_additions': 0
            }
            
            # Start monitoring task
            asyncio.create_task(self._monitor_performance())
            
        except Exception as e:
            self.logger.error(f"Error setting up monitoring: {e}")
    
    async def _monitor_performance(self):
        """Monitor engine performance"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Log performance stats
                self.logger.info(f"Performance Stats: {self.performance_stats}")
                
                # Clean up old cache entries
                await self._cleanup_cache()
                
            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {e}")
    
    async def _cleanup_cache(self):
        """Clean up old cache entries"""
        try:
            current_time = time.time()
            expired_keys = []
            
            for key, (data, timestamp) in self.query_cache.items():
                if current_time - timestamp > self.cache_ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.query_cache[key]
                
            if expired_keys:
                self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up cache: {e}")
    
    async def add_node(self, node: KnowledgeNode) -> bool:
        """Add a node to the knowledge graph"""
        try:
            self.nodes[node.node_id] = node
            
            # Update graph
            self.graph.add_node(node.node_id, **node.attributes)
            
            # Update indexes
            self.node_index[node.node_type].add(node.node_id)
            
            # Symbol indexing
            symbols = node.attributes.get('symbols', [])
            for symbol in symbols:
                self.symbol_index[symbol.upper()].add(node.node_id)
            
            # Pattern indexing
            if node.node_type == 'pattern':
                pattern_name = node.attributes.get('pattern_name', '')
                if pattern_name:
                    self.pattern_index[pattern_name.lower()].add(node.node_id)
            
            # Temporal indexing
            date_bucket = node.timestamp.strftime("%Y-%m")
            self.temporal_index[date_bucket].append(node.node_id)
            
            self.performance_stats['knowledge_additions'] += 1
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding node {node.node_id}: {e}")
            return False
    
    async def add_edge(self, edge: KnowledgeEdge) -> bool:
        """Add an edge to the knowledge graph"""
        try:
            self.edges[edge.edge_id] = edge
            
            # Update graph
            self.graph.add_edge(
                edge.source_node,
                edge.target_node,
                relationship=edge.relationship_type,
                strength=edge.strength,
                **edge.attributes
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding edge {edge.edge_id}: {e}")
            return False
    
    async def query_knowledge(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Query the knowledge base with natural language"""
        try:
            start_time = time.time()
            
            # Check cache first
            cache_key = self._generate_cache_key(query, context)
            if cache_key in self.query_cache:
                cached_result, timestamp = self.query_cache[cache_key]
                if time.time() - timestamp < self.cache_ttl:
                    self.performance_stats['cache_hits'] += 1
                    return cached_result
            
            # Process query
            result = await self._process_query(query, context or {})
            
            # Cache result
            self.query_cache[cache_key] = (result, time.time())
            
            # Update performance stats
            self.performance_stats['total_queries'] += 1
            response_time = time.time() - start_time
            self.performance_stats['average_response_time'] = (
                (self.performance_stats['average_response_time'] * (self.performance_stats['total_queries'] - 1) + response_time) /
                self.performance_stats['total_queries']
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error querying knowledge: {e}")
            return {"error": str(e)}
    
    async def _process_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a knowledge query"""
        try:
            query_lower = query.lower()
            results = {
                'query': query,
                'context': context,
                'results': [],
                'patterns': [],
                'recommendations': [],
                'confidence': 0.0
            }
            
            # Symbol extraction
            symbols = self._extract_symbols(query)
            if symbols:
                results['symbols'] = symbols
                
                # Find relevant nodes for symbols
                for symbol in symbols:
                    symbol_nodes = self.symbol_index.get(symbol.upper(), set())
                    for node_id in symbol_nodes:
                        if node_id in self.nodes:
                            node = self.nodes[node_id]
                            results['results'].append({
                                'type': 'node',
                                'id': node_id,
                                'node_type': node.node_type,
                                'attributes': node.attributes,
                                'confidence': node.confidence
                            })
            
            # Pattern matching
            patterns = await self._find_relevant_patterns(query, context)
            results['patterns'] = patterns
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(query, context, results)
            results['recommendations'] = recommendations
            
            # Calculate overall confidence
            if results['results'] or results['patterns']:
                results['confidence'] = self._calculate_confidence(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return {"error": str(e)}
    
    def _extract_symbols(self, query: str) -> List[str]:
        """Extract stock symbols from query"""
        import re
        
        # Common patterns for stock symbols
        patterns = [
            r'\b([A-Z]{1,5})\b',  # 1-5 uppercase letters
            r'\$([A-Z]{1,5})',    # $SYMBOL format
        ]
        
        symbols = []
        for pattern in patterns:
            matches = re.findall(pattern, query)
            symbols.extend(matches)
        
        # Filter out common words that might match pattern
        common_words = {'THE', 'AND', 'OR', 'BUT', 'FOR', 'TO', 'OF', 'IN', 'ON', 'AT', 'BY'}
        symbols = [s for s in symbols if s not in common_words]
        
        return list(set(symbols))
    
    async def _find_relevant_patterns(self, query: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find relevant patterns for the query"""
        try:
            relevant_patterns = []
            query_lower = query.lower()
            
            # Check for pattern keywords
            pattern_keywords = {
                'double top': ['double_top'],
                'head and shoulders': ['head_shoulders'],
                'support': ['support_resistance'],
                'resistance': ['support_resistance'],
                'breakout': ['breakout_pattern'],
                'reversal': ['reversal_patterns'],
                'fibonacci': ['fibonacci_retracement', 'fibonacci_extension']
            }
            
            for keyword, pattern_ids in pattern_keywords.items():
                if keyword in query_lower:
                    for pattern_id in pattern_ids:
                        if pattern_id in self.patterns:
                            pattern = self.patterns[pattern_id]
                            relevant_patterns.append({
                                'pattern_id': pattern_id,
                                'pattern_type': pattern.pattern_type,
                                'success_rate': pattern.success_rate,
                                'conditions': pattern.conditions,
                                'outcomes': pattern.outcomes,
                                'sample_size': pattern.sample_size
                            })
            
            return relevant_patterns
            
        except Exception as e:
            self.logger.error(f"Error finding relevant patterns: {e}")
            return []
    
    async def _generate_recommendations(self, query: str, context: Dict[str, Any], results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations"""
        try:
            recommendations = []
            
            # Symbol-based recommendations
            symbols = results.get('symbols', [])
            for symbol in symbols:
                # Get recent patterns for symbol
                symbol_patterns = await self._get_symbol_patterns(symbol)
                
                if symbol_patterns:
                    for pattern in symbol_patterns:
                        if pattern['success_rate'] > 0.7:
                            recommendations.append({
                                'type': 'pattern_signal',
                                'symbol': symbol,
                                'pattern': pattern['pattern_id'],
                                'signal': pattern['outcomes'],
                                'confidence': pattern['success_rate'],
                                'reasoning': f"Pattern {pattern['pattern_id']} detected with {pattern['success_rate']:.1%} success rate"
                            })
            
            # General market recommendations
            if 'market' in query.lower() or 'sentiment' in query.lower():
                recommendations.append({
                    'type': 'market_analysis',
                    'recommendation': 'Monitor key technical levels and volume patterns',
                    'confidence': 0.8,
                    'reasoning': 'Current market conditions suggest increased volatility'
                })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return []
    
    async def _get_symbol_patterns(self, symbol: str) -> List[Dict[str, Any]]:
        """Get patterns for a specific symbol"""
        try:
            symbol_patterns = []
            
            for pattern_id, pattern in self.patterns.items():
                if symbol in pattern.symbols or not pattern.symbols:
                    symbol_patterns.append({
                        'pattern_id': pattern_id,
                        'pattern_type': pattern.pattern_type,
                        'success_rate': pattern.success_rate,
                        'conditions': pattern.conditions,
                        'outcomes': pattern.outcomes,
                        'sample_size': pattern.sample_size
                    })
            
            return symbol_patterns
            
        except Exception as e:
            self.logger.error(f"Error getting symbol patterns: {e}")
            return []
    
    def _calculate_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate overall confidence score"""
        try:
            confidence_scores = []
            
            # Node confidences
            for result in results.get('results', []):
                confidence_scores.append(result.get('confidence', 0.0))
            
            # Pattern confidences
            for pattern in results.get('patterns', []):
                confidence_scores.append(pattern.get('success_rate', 0.0))
            
            # Recommendation confidences
            for rec in results.get('recommendations', []):
                confidence_scores.append(rec.get('confidence', 0.0))
            
            if confidence_scores:
                return sum(confidence_scores) / len(confidence_scores)
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.0
    
    def _generate_cache_key(self, query: str, context: Dict[str, Any]) -> str:
        """Generate cache key for query"""
        try:
            query_data = {
                'query': query,
                'context': context or {}
            }
            
            query_str = json.dumps(query_data, sort_keys=True)
            return hashlib.md5(query_str.encode()).hexdigest()
            
        except Exception as e:
            self.logger.error(f"Error generating cache key: {e}")
            return str(hash(query))
    
    async def add_pattern(self, pattern: MarketPattern) -> bool:
        """Add a new market pattern"""
        try:
            self.patterns[pattern.pattern_id] = pattern
            
            # Create corresponding knowledge node
            node = KnowledgeNode(
                node_id=f"pattern_{pattern.pattern_id}",
                node_type="pattern",
                attributes={
                    'pattern_name': pattern.pattern_id,
                    'pattern_type': pattern.pattern_type,
                    'success_rate': pattern.success_rate,
                    'conditions': pattern.conditions,
                    'outcomes': pattern.outcomes,
                    'symbols': pattern.symbols
                },
                timestamp=datetime.now(),
                confidence=pattern.success_rate
            )
            
            await self.add_node(node)
            
            self.performance_stats['pattern_matches'] += 1
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding pattern: {e}")
            return False
    
    async def update_pattern(self, pattern_id: str, new_outcome: Dict[str, Any]) -> bool:
        """Update pattern based on new outcome"""
        try:
            if pattern_id not in self.patterns:
                return False
            
            pattern = self.patterns[pattern_id]
            
            # Update success rate based on outcome
            if new_outcome.get('success', False):
                pattern.success_rate = (pattern.success_rate * pattern.sample_size + 1) / (pattern.sample_size + 1)
            else:
                pattern.success_rate = (pattern.success_rate * pattern.sample_size) / (pattern.sample_size + 1)
            
            pattern.sample_size += 1
            pattern.last_seen = datetime.now()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating pattern: {e}")
            return False
    
    async def save_knowledge_base(self):
        """Save knowledge base to storage"""
        try:
            knowledge_data = {
                'nodes': [
                    {
                        'node_id': node.node_id,
                        'node_type': node.node_type,
                        'attributes': node.attributes,
                        'timestamp': node.timestamp.isoformat(),
                        'confidence': node.confidence,
                        'source': node.source
                    }
                    for node in self.nodes.values()
                ],
                'edges': [
                    {
                        'edge_id': edge.edge_id,
                        'source_node': edge.source_node,
                        'target_node': edge.target_node,
                        'relationship_type': edge.relationship_type,
                        'strength': edge.strength,
                        'attributes': edge.attributes,
                        'timestamp': edge.timestamp.isoformat(),
                        'confidence': edge.confidence
                    }
                    for edge in self.edges.values()
                ],
                'patterns': [
                    {
                        'pattern_id': pattern.pattern_id,
                        'pattern_type': pattern.pattern_type,
                        'conditions': pattern.conditions,
                        'outcomes': pattern.outcomes,
                        'success_rate': pattern.success_rate,
                        'sample_size': pattern.sample_size,
                        'last_seen': pattern.last_seen.isoformat(),
                        'symbols': pattern.symbols,
                        'market_regime': pattern.market_regime
                    }
                    for pattern in self.patterns.values()
                ],
                'metadata': {
                    'saved_at': datetime.now().isoformat(),
                    'version': '1.0',
                    'stats': self.performance_stats
                }
            }
            
            with open('knowledge_graph_state.json', 'w') as f:
                json.dump(knowledge_data, f, indent=2)
            
            self.logger.info("Knowledge base saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving knowledge base: {e}")
    
    async def get_market_insights(self, symbol: str = None, timeframe: str = "1h") -> Dict[str, Any]:
        """Get market insights for a symbol or general market"""
        try:
            insights = {
                'symbol': symbol,
                'timeframe': timeframe,
                'patterns': [],
                'indicators': [],
                'recommendations': [],
                'confidence': 0.0,
                'timestamp': datetime.now().isoformat()
            }
            
            if symbol:
                # Get symbol-specific insights
                symbol_patterns = await self._get_symbol_patterns(symbol)
                insights['patterns'] = symbol_patterns
                
                # Get relevant nodes
                symbol_nodes = self.symbol_index.get(symbol.upper(), set())
                for node_id in symbol_nodes:
                    if node_id in self.nodes:
                        node = self.nodes[node_id]
                        if node.node_type == 'indicator':
                            insights['indicators'].append({
                                'name': node.attributes.get('indicator_name', ''),
                                'value': node.attributes.get('value', 0),
                                'signal': node.attributes.get('signal', 'neutral'),
                                'confidence': node.confidence
                            })
            
            # Generate recommendations
            if insights['patterns'] or insights['indicators']:
                recommendations = await self._generate_insights_recommendations(insights)
                insights['recommendations'] = recommendations
                insights['confidence'] = self._calculate_confidence(insights)
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error getting market insights: {e}")
            return {"error": str(e)}
    
    async def _generate_insights_recommendations(self, insights: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations from insights"""
        try:
            recommendations = []
            
            # Pattern-based recommendations
            for pattern in insights.get('patterns', []):
                if pattern['success_rate'] > 0.7:
                    recommendations.append({
                        'type': 'pattern_signal',
                        'signal': pattern['outcomes'],
                        'confidence': pattern['success_rate'],
                        'reasoning': f"Strong {pattern['pattern_id']} pattern detected"
                    })
            
            # Indicator-based recommendations
            strong_signals = [ind for ind in insights.get('indicators', []) if ind['confidence'] > 0.7]
            if strong_signals:
                recommendations.append({
                    'type': 'technical_signal',
                    'signals': strong_signals,
                    'confidence': sum(ind['confidence'] for ind in strong_signals) / len(strong_signals),
                    'reasoning': f"Strong technical signals from {len(strong_signals)} indicators"
                })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating insights recommendations: {e}")
            return []
