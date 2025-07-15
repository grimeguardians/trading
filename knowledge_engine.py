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

# Knowledge Graph Components
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

class KnowledgeGraph:
    """Advanced knowledge graph for financial market intelligence"""

    def __init__(self):
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: Dict[str, KnowledgeEdge] = {}
        self.node_index: Dict[str, Set[str]] = defaultdict(set)  # type -> node_ids
        self.edge_index: Dict[str, Set[str]] = defaultdict(set)  # type -> edge_ids
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)  # node_id -> connected_node_ids
        self.reverse_adjacency: Dict[str, Set[str]] = defaultdict(set)

        # Enhanced indexing for performance
        self.pattern_index: Dict[str, Set[str]] = defaultdict(set)  # pattern_type -> node_ids
        self.symbol_index: Dict[str, Set[str]] = defaultdict(set)   # symbol -> node_ids
        self.confidence_index: Dict[str, List[str]] = {}            # confidence_bucket -> sorted node_ids
        self.attribute_index: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        
        # Advanced optimization indexes
        self.text_search_index: Dict[str, Set[str]] = defaultdict(set)  # keyword -> node_ids
        self.pattern_category_index: Dict[str, Set[str]] = defaultdict(set)  # category -> pattern_node_ids
        self.concept_hierarchy_index: Dict[str, Set[str]] = defaultdict(set)  # parent_concept -> child_nodes
        self.temporal_index: Dict[str, List[str]] = defaultdict(list)  # date_bucket -> node_ids (sorted by time)
        self.fuzzy_search_cache: Dict[str, List[Tuple[str, float]]] = {}  # query -> [(node_id, score)]
        self.query_performance_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Performance optimization settings
        self.enable_caching = True
        self.cache_ttl = 300  # 5 minutes
        self.max_cache_size = 1000

        self.logger = logging.getLogger("KnowledgeGraph")

    def add_node(self, node: KnowledgeNode) -> bool:
        """Add a node to the knowledge graph with optimized indexing"""
        try:
            self.nodes[node.node_id] = node
            self.node_index[node.node_type].add(node.node_id)

            # Enhanced pattern indexing
            if node.node_type == 'chart_pattern':
                pattern_name = node.attributes.get('pattern_name', '')
                if pattern_name:
                    self.pattern_index[pattern_name.lower()].add(node.node_id)
                    # Add pattern category indexing
                    category = node.attributes.get('category', 'general')
                    self.pattern_category_index[category].add(node.node_id)

            # Enhanced trading concept indexing
            if node.node_type == 'trading_concept':
                concept_name = node.attributes.get('concept_name', '')
                if concept_name:
                    self.concept_hierarchy_index[concept_name.lower()].add(node.node_id)

            # Optimized symbol indexing
            symbols = node.attributes.get('symbols', [])
            for symbol in symbols:
                self.symbol_index[symbol.upper()].add(node.node_id)

            # Enhanced confidence indexing with sorting
            confidence_bucket = round(node.confidence, 1)  # 0.1 precision
            if confidence_bucket not in self.confidence_index:
                self.confidence_index[confidence_bucket] = []
            self.confidence_index[confidence_bucket].append(node.node_id)
            # Keep confidence buckets sorted by confidence descending
            self.confidence_index[confidence_bucket].sort(
                key=lambda nid: self.nodes[nid].confidence, reverse=True
            )

            # Advanced text search indexing
            searchable_text = self._extract_searchable_text(node)
            for keyword in searchable_text:
                if len(keyword) > 2:  # Only index meaningful keywords
                    self.text_search_index[keyword.lower()].add(node.node_id)

            # Temporal indexing for time-based queries
            date_bucket = node.timestamp.strftime("%Y-%m")
            self.temporal_index[date_bucket].append(node.node_id)

            # Enhanced attribute indexing with fuzzy matching support
            for key, value in node.attributes.items():
                if isinstance(value, (str, int, float)):
                    self.attribute_index[key][str(value)].add(node.node_id)

            # Clear relevant caches
            if self.enable_caching:
                self._invalidate_related_caches(node)

            self.logger.debug(f"Added node with advanced indexing: {node.node_id} ({node.node_type})")
            return True
        except Exception as e:
            self.logger.error(f"Error adding node {node.node_id}: {e}")
            return False

    def _extract_searchable_text(self, node: KnowledgeNode) -> List[str]:
        """Extract keywords for text search indexing"""
        keywords = []
        
        # Extract from pattern/concept names
        if 'pattern_name' in node.attributes:
            keywords.extend(node.attributes['pattern_name'].split())
        if 'concept_name' in node.attributes:
            keywords.extend(node.attributes['concept_name'].split())
        
        # Extract from description
        if 'description' in node.attributes:
            keywords.extend(node.attributes['description'].split())
        
        # Extract from other text attributes
        for key, value in node.attributes.items():
            if isinstance(value, str) and len(value) < 100:  # Avoid indexing large text blocks
                keywords.extend(value.split())
        
        # Clean keywords
        cleaned = []
        for keyword in keywords:
            clean = ''.join(c for c in keyword.lower() if c.isalnum())
            if len(clean) > 2:
                cleaned.append(clean)
        
        return cleaned

    def _invalidate_related_caches(self, node: KnowledgeNode):
        """Invalidate caches that might be affected by new node"""
        # Simple cache invalidation - in production, could be more sophisticated
        self.fuzzy_search_cache.clear()

    def add_edge(self, edge: KnowledgeEdge) -> bool:
        """Add an edge to the knowledge graph"""
        try:
            if edge.source_node not in self.nodes or edge.target_node not in self.nodes:
                self.logger.warning(f"Cannot add edge: missing nodes {edge.source_node} or {edge.target_node}")
                return False

            self.edges[edge.edge_id] = edge
            self.edge_index[edge.relationship_type].add(edge.edge_id)
            self.adjacency[edge.source_node].add(edge.target_node)
            self.reverse_adjacency[edge.target_node].add(edge.source_node)
            self.logger.debug(f"Added edge: {edge.edge_id} ({edge.relationship_type})")
            return True
        except Exception as e:
            self.logger.error(f"Error adding edge {edge.edge_id}: {e}")
            return False

    def find_patterns(self, pattern_type: str, symbol: str = None, category: str = None) -> List[KnowledgeNode]:
        """Find patterns in the knowledge graph using optimized indexing with fuzzy search"""
        import time
        start_time = time.time()
        
        pattern_nodes = []
        pattern_node_ids = set()

        # Direct pattern index lookup
        direct_matches = self.pattern_index.get(pattern_type.lower(), set())
        pattern_node_ids.update(direct_matches)

        # Fuzzy pattern matching for partial queries
        if not pattern_node_ids or len(pattern_type) > 3:
            fuzzy_matches = self._fuzzy_pattern_search(pattern_type)
            pattern_node_ids.update(fuzzy_matches)

        # Category-based filtering if specified
        if category:
            category_nodes = self.pattern_category_index.get(category.lower(), set())
            if category_nodes:
                pattern_node_ids = pattern_node_ids.intersection(category_nodes)

        # Symbol-based filtering with case-insensitive matching
        if symbol:
            symbol_upper = symbol.upper()
            symbol_nodes = self.symbol_index.get(symbol_upper, set())
            if symbol_nodes:
                pattern_node_ids = pattern_node_ids.intersection(symbol_nodes)

        # Convert to nodes and apply final filtering
        for node_id in pattern_node_ids:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                if node.node_type in ['chart_pattern', 'pattern']:
                    pattern_nodes.append(node)

        # Sort by confidence and relevance
        pattern_nodes.sort(key=lambda x: (x.confidence, x.timestamp), reverse=True)

        # Performance tracking
        query_time = time.time() - start_time
        self.query_performance_stats['find_patterns'] = {
            'last_query_time': query_time,
            'last_result_count': len(pattern_nodes),
            'pattern_type': pattern_type
        }

        return pattern_nodes

    def _fuzzy_pattern_search(self, pattern_type: str) -> Set[str]:
        """Advanced fuzzy search for pattern matching"""
        cache_key = f"fuzzy_{pattern_type.lower()}"
        
        # Check cache first
        if self.enable_caching and cache_key in self.fuzzy_search_cache:
            return {result[0] for result in self.fuzzy_search_cache[cache_key]}

        matches = set()
        query_lower = pattern_type.lower()
        
        # Exact substring matches
        for indexed_pattern, node_ids in self.pattern_index.items():
            if query_lower in indexed_pattern or indexed_pattern in query_lower:
                matches.update(node_ids)
        
        # Keyword-based text search
        query_keywords = query_lower.split()
        for keyword in query_keywords:
            if keyword in self.text_search_index:
                matches.update(self.text_search_index[keyword])

        # Advanced pattern name similarity
        for node_id in self.node_index.get('chart_pattern', set()):
            node = self.nodes[node_id]
            pattern_name = node.attributes.get('pattern_name', '').lower()
            
            # Check for word overlap
            pattern_words = set(pattern_name.split())
            query_words = set(query_keywords)
            
            if pattern_words.intersection(query_words):
                matches.add(node_id)
                
            # Check for common trading pattern synonyms
            if self._check_pattern_synonyms(query_lower, pattern_name):
                matches.add(node_id)

        # Cache results
        if self.enable_caching:
            scored_results = [(node_id, 1.0) for node_id in matches]  # Basic scoring
            self.fuzzy_search_cache[cache_key] = scored_results
            
            # Limit cache size
            if len(self.fuzzy_search_cache) > self.max_cache_size:
                # Remove oldest entries
                oldest_key = min(self.fuzzy_search_cache.keys())
                del self.fuzzy_search_cache[oldest_key]

        return matches

    def _check_pattern_synonyms(self, query: str, pattern_name: str) -> bool:
        """Check for trading pattern synonyms and related terms"""
        synonyms = {
            'breakout': ['break out', 'breakthrough', 'escape', 'burst'],
            'breakdown': ['break down', 'collapse', 'fall'],
            'reversal': ['turn', 'flip', 'reverse', 'turnaround'],
            'continuation': ['continue', 'persist', 'maintain'],
            'triangle': ['triangular', 'wedge'],
            'flag': ['pennant'],
            'support': ['floor', 'base', 'bottom'],
            'resistance': ['ceiling', 'top', 'barrier'],
            'head and shoulders': ['h&s', 'head shoulders'],
            'double top': ['double peak', 'twin top'],
            'double bottom': ['double trough', 'twin bottom']
        }
        
        for base_term, synonym_list in synonyms.items():
            if base_term in query or base_term in pattern_name:
                for synonym in synonym_list:
                    if synonym in query or synonym in pattern_name:
                        return True
        
        return False

    def find_patterns_by_symbol(self, symbol: str, min_confidence: float = 0.5) -> List[KnowledgeNode]:
        """Fast symbol-based pattern lookup"""
        pattern_nodes = []
        symbol_node_ids = self.symbol_index.get(symbol, set())

        for node_id in symbol_node_ids:
            node = self.nodes[node_id]
            if node.node_type in ['chart_pattern', 'pattern'] and node.confidence >= min_confidence:
                pattern_nodes.append(node)

        return sorted(pattern_nodes, key=lambda x: x.confidence, reverse=True)

    def find_high_confidence_patterns(self, min_confidence: float = 0.8) -> List[KnowledgeNode]:
        """Fast high-confidence pattern lookup"""
        pattern_nodes = []

        for confidence_bucket, node_ids in self.confidence_index.items():
            if confidence_bucket >= min_confidence:
                for node_id in node_ids:
                    node = self.nodes[node_id]
                    if node.node_type in ['chart_pattern', 'pattern']:
                        pattern_nodes.append(node)

        return sorted(pattern_nodes, key=lambda x: x.confidence, reverse=True)

    def get_related_nodes(self, node_id: str, relationship_types: List[str] = None, 
                         max_depth: int = 2) -> List[Tuple[KnowledgeNode, float]]:
        """Get nodes related to a given node with relationship strength"""
        if node_id not in self.nodes:
            return []

        visited = set()
        related = []
        queue = [(node_id, 1.0, 0)]  # (node_id, strength, depth)

        while queue:
            current_id, strength, depth = queue.pop(0)
            if current_id in visited or depth > max_depth:
                continue

            visited.add(current_id)
            if current_id != node_id:
                related.append((self.nodes[current_id], strength))

            # Find connected nodes
            for connected_id in self.adjacency[current_id]:
                if connected_id not in visited:
                    # Find the edge to calculate relationship strength
                    edge_strength = 1.0
                    for edge_id in self.edge_index.get('correlates_with', set()):
                        edge = self.edges[edge_id]
                        if (edge.source_node == current_id and edge.target_node == connected_id):
                            if relationship_types is None or edge.relationship_type in relationship_types:
                                edge_strength = edge.strength
                                break

                    new_strength = strength * edge_strength
                    if new_strength > 0.1:  # Threshold for relevance
                        queue.append((connected_id, new_strength, depth + 1))

        return sorted(related, key=lambda x: x[1], reverse=True)

    def query_knowledge(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Optimized knowledge querying with advanced indexing and caching"""
        import time
        start_time = time.time()
        
        # Generate cache key for this query
        cache_key = self._generate_query_cache_key(query)
        
        # Check cache first
        if self.enable_caching and cache_key in self.fuzzy_search_cache:
            cached_results = self.fuzzy_search_cache[cache_key]
            return [{'type': 'node', 'id': result[0], 'node': self.nodes[result[0]], 'score': result[1]} 
                   for result in cached_results if result[0] in self.nodes]

        results = []
        
        # Optimized node querying using indexes
        candidate_nodes = self._get_candidate_nodes(query)
        
        for node_id in candidate_nodes:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                score = self._calculate_enhanced_query_match(node.attributes, query, node)
                if score > 0.3:  # Lower threshold for better recall
                    results.append({
                        'type': 'node',
                        'id': node_id,
                        'node': node,
                        'score': score
                    })

        # Enhanced edge querying (only if explicitly requested)
        if query.get('include_edges', False):
            for edge_id, edge in self.edges.items():
                score = self._calculate_query_match(edge.attributes, query)
                if score > 0.4:
                    results.append({
                        'type': 'edge',
                        'id': edge_id,
                        'edge': edge,
                        'score': score
                    })

        # Sort by score with confidence boost
        results.sort(key=lambda x: x['score'] + (x.get('node', x.get('edge', type('obj', (), {'confidence': 0})())).confidence * 0.1), reverse=True)

        # Cache results
        if self.enable_caching and results:
            cached_data = [(r['id'], r['score']) for r in results[:50]]  # Cache top 50
            self.fuzzy_search_cache[cache_key] = cached_data

        # Performance tracking
        query_time = time.time() - start_time
        self.query_performance_stats['query_knowledge'] = {
            'last_query_time': query_time,
            'last_result_count': len(results),
            'cache_hit': cache_key in self.fuzzy_search_cache
        }

        return results

    def _get_candidate_nodes(self, query: Dict[str, Any]) -> Set[str]:
        """Get candidate nodes using optimized index lookups"""
        candidates = set()
        
        # Pattern type filtering
        if 'pattern_type' in query:
            pattern_type = query['pattern_type']
            candidates.update(self.pattern_index.get(pattern_type.lower(), set()))
            # Add fuzzy matches
            candidates.update(self._fuzzy_pattern_search(pattern_type))
        
        # Query type filtering
        if 'query_type' in query:
            query_type = query['query_type']
            if query_type == 'pattern':
                candidates.update(self.node_index.get('chart_pattern', set()))
            elif query_type == 'concept':
                candidates.update(self.node_index.get('trading_concept', set()))
        
        # Symbol filtering
        if 'symbol' in query:
            symbol = query['symbol'].upper()
            symbol_nodes = self.symbol_index.get(symbol, set())
            if candidates:
                candidates = candidates.intersection(symbol_nodes)
            else:
                candidates.update(symbol_nodes)
        
        # Confidence filtering
        if 'min_confidence' in query:
            min_conf = query['min_confidence']
            high_conf_nodes = set()
            for conf_bucket, node_ids in self.confidence_index.items():
                if conf_bucket >= min_conf:
                    high_conf_nodes.update(node_ids)
            
            if candidates:
                candidates = candidates.intersection(high_conf_nodes)
            else:
                candidates.update(high_conf_nodes)
        
        # Text search
        if 'text_search' in query:
            text_query = query['text_search'].lower()
            text_matches = set()
            for keyword in text_query.split():
                if keyword in self.text_search_index:
                    text_matches.update(self.text_search_index[keyword])
            
            if candidates:
                candidates = candidates.intersection(text_matches)
            else:
                candidates.update(text_matches)
        
        # If no specific filters, return all nodes (with limit)
        if not candidates and not any(key in query for key in ['pattern_type', 'query_type', 'symbol', 'min_confidence', 'text_search']):
            all_nodes = set(self.nodes.keys())
            # Prioritize high-confidence nodes
            high_conf_nodes = []
            for conf_bucket in sorted(self.confidence_index.keys(), reverse=True):
                high_conf_nodes.extend(self.confidence_index[conf_bucket])
                if len(high_conf_nodes) >= 200:  # Limit for performance
                    break
            candidates = set(high_conf_nodes[:200])
        
        return candidates

    def _calculate_enhanced_query_match(self, attributes: Dict[str, Any], query: Dict[str, Any], node: KnowledgeNode) -> float:
        """Enhanced query matching with better scoring"""
        if not query:
            return 0.0

        score = 0.0
        max_possible_score = 0.0
        
        # Base matching from original method
        for key, value in query.items():
            if key in ['include_edges', 'min_confidence']:  # Skip meta-query parameters
                continue
                
            max_possible_score += 1.0
            
            if key in attributes:
                if isinstance(value, str) and isinstance(attributes[key], str):
                    # Enhanced string matching
                    if value.lower() == attributes[key].lower():
                        score += 1.0  # Exact match
                    elif value.lower() in attributes[key].lower() or attributes[key].lower() in value.lower():
                        score += 0.8  # Substring match
                    else:
                        # Check for synonym matches
                        if self._check_pattern_synonyms(value.lower(), attributes[key].lower()):
                            score += 0.6
                elif value == attributes[key]:
                    score += 1.0
                elif isinstance(value, (int, float)) and isinstance(attributes[key], (int, float)):
                    diff_ratio = abs(value - attributes[key]) / max(abs(value), abs(attributes[key]), 1)
                    if diff_ratio < 0.1:
                        score += 1.0 - diff_ratio
        
        # Confidence boost
        confidence_boost = node.confidence * 0.2
        
        # Node type relevance boost
        type_boost = 0.0
        if 'query_type' in query:
            if (query['query_type'] == 'pattern' and node.node_type == 'chart_pattern') or \
               (query['query_type'] == 'concept' and node.node_type == 'trading_concept'):
                type_boost = 0.3
        
        final_score = (score / max(max_possible_score, 1.0)) + confidence_boost + type_boost
        return min(final_score, 1.0)

    def _generate_query_cache_key(self, query: Dict[str, Any]) -> str:
        """Generate a cache key for the query"""
        import hashlib
        query_str = str(sorted(query.items()))
        return hashlib.md5(query_str.encode()).hexdigest()[:16]

    def _calculate_query_match(self, attributes: Dict[str, Any], query: Dict[str, Any]) -> float:
        """Calculate how well attributes match a query"""
        if not query:
            return 0.0

        matches = 0
        total = len(query)

        for key, value in query.items():
            if key in attributes:
                if isinstance(value, str) and isinstance(attributes[key], str):
                    if value.lower() in attributes[key].lower():
                        matches += 1
                elif value == attributes[key]:
                    matches += 1
                elif isinstance(value, (int, float)) and isinstance(attributes[key], (int, float)):
                    if abs(value - attributes[key]) / max(abs(value), abs(attributes[key]), 1) < 0.1:
                        matches += 1

        return matches / total

    def save_to_file(self, filepath: str) -> bool:
        """Save knowledge graph to file"""
        try:
            graph_data = {
                'nodes': {nid: {
                    'node_id': n.node_id,
                    'node_type': n.node_type,
                    'attributes': n.attributes,
                    'timestamp': n.timestamp.isoformat(),
                    'confidence': n.confidence,
                    'source': n.source
                } for nid, n in self.nodes.items()},
                'edges': {eid: {
                    'edge_id': e.edge_id,
                    'source_node': e.source_node,
                    'target_node': e.target_node,
                    'relationship_type': e.relationship_type,
                    'strength': e.strength,
                    'attributes': e.attributes,
                    'timestamp': e.timestamp.isoformat(),
                    'confidence': e.confidence
                } for eid, e in self.edges.items()}
            }

            with open(filepath, 'w') as f:
                json.dump(graph_data, f, indent=2)

            self.logger.info(f"Knowledge graph saved to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving knowledge graph: {e}")
            return False

    def load_from_file(self, filepath: str) -> bool:
        """Load knowledge graph from file"""
        try:
            with open(filepath, 'r') as f:
                graph_data = json.load(f)

            # Load nodes
            for node_data in graph_data.get('nodes', {}).values():
                node = KnowledgeNode(
                    node_id=node_data['node_id'],
                    node_type=node_data['node_type'],
                    attributes=node_data['attributes'],
                    timestamp=datetime.fromisoformat(node_data['timestamp']),
                    confidence=node_data['confidence'],
                    source=node_data['source']
                )
                self.add_node(node)

            # Load edges
            for edge_data in graph_data.get('edges', {}).values():
                edge = KnowledgeEdge(
                    edge_id=edge_data['edge_id'],
                    source_node=edge_data['source_node'],
                    target_node=edge_data['target_node'],
                    relationship_type=edge_data['relationship_type'],
                    strength=edge_data['strength'],
                    attributes=edge_data['attributes'],
                    timestamp=datetime.fromisoformat(edge_data['timestamp']),
                    confidence=edge_data['confidence']
                )
                self.add_edge(edge)

            self.logger.info(f"Knowledge graph loaded from {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading knowledge graph: {e}")
            return False

class PatternRecognitionEngine:
    """Advanced pattern recognition and learning system"""

    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.knowledge_graph = knowledge_graph
        self.patterns: Dict[str, MarketPattern] = {}
        self.pattern_templates = self._initialize_pattern_templates()
        self.logger = logging.getLogger("PatternRecognition")
        self.min_pattern_samples = 5
        self.confidence_threshold = 0.6
        self.brain = None  # Add brain reference

    def _initialize_pattern_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize pattern recognition templates"""
        return {
            'bullish_reversal': {
                'conditions': ['rsi_oversold', 'hammer_candle', 'volume_spike'],
                'target_outcome': 'price_increase',
                'min_confidence': 0.7
            },
            'bearish_reversal': {
                'conditions': ['rsi_overbought', 'shooting_star', 'volume_spike'],
                'target_outcome': 'price_decrease',
                'min_confidence': 0.7
            },
            'breakout_pattern': {
                'conditions': ['resistance_break', 'volume_confirmation', 'momentum_positive'],
                'target_outcome': 'continued_uptrend',
                'min_confidence': 0.65
            },
            'earnings_beat': {
                'conditions': ['eps_beat', 'revenue_beat', 'guidance_raise'],
                'target_outcome': 'post_earnings_drift',
                'min_confidence': 0.8
            },
            'sector_rotation': {
                'conditions': ['relative_strength', 'economic_indicator_change', 'institutional_flow'],
                'target_outcome': 'sector_outperformance',
                'min_confidence': 0.75
            }
        }

    def learn_pattern(self, symbol: str, market_data: Dict[str, Any], 
                     outcome: Dict[str, Any], pattern_type: str = None) -> bool:
        """Learn a new pattern from market data and outcomes"""
        try:
            # Extract features from market data
            features = self._extract_features(market_data)

            # Auto-detect pattern type if not provided
            if pattern_type is None:
                pattern_type = self._detect_pattern_type(features, outcome)

            pattern_id = f"{pattern_type}_{symbol}_{int(time.time())}"

            # Create or update pattern
            if pattern_id in self.patterns:
                pattern = self.patterns[pattern_id]
                pattern.sample_size += 1
                pattern.last_seen = datetime.now()
            else:
                pattern = MarketPattern(
                    pattern_id=pattern_id,
                    pattern_type=pattern_type,
                    conditions=features,
                    outcomes=outcome,
                    success_rate=1.0,
                    sample_size=1,
                    last_seen=datetime.now(),
                    symbols=[symbol]
                )
                self.patterns[pattern_id] = pattern

            # Update success rate based on outcome
            if outcome.get('successful', False):
                pattern.success_rate = (pattern.success_rate * (pattern.sample_size - 1) + 1.0) / pattern.sample_size
            else:
                pattern.success_rate = (pattern.success_rate * (pattern.sample_size - 1)) / pattern.sample_size

            # Add to knowledge graph if significant
            if pattern.sample_size >= self.min_pattern_samples and pattern.success_rate >= self.confidence_threshold:
                self._add_pattern_to_knowledge_graph(pattern)

            self.logger.info(f"Learned pattern {pattern_type} for {symbol} (success rate: {pattern.success_rate:.2f})")
            return True

        except Exception as e:
            self.logger.error(f"Error learning pattern: {e}")
            return False

    def recognize_patterns(self, symbol: str, current_data: Dict[str, Any]) -> List[Tuple[MarketPattern, float]]:
        """Recognize patterns in current market data"""
        recognized = []
        current_features = self._extract_features(current_data)

        for pattern in self.patterns.values():
            if symbol in pattern.symbols or not pattern.symbols:
                similarity = self._calculate_pattern_similarity(current_features, pattern.conditions)

                if similarity >= self.confidence_threshold:
                    confidence = similarity * pattern.success_rate * min(pattern.sample_size / 10, 1.0)
                    recognized.append((pattern, confidence))

        return sorted(recognized, key=lambda x: x[1], reverse=True)

    def _extract_features(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant features from market data"""
        features = {}

        # Technical indicators
        if 'rsi' in market_data:
            features['rsi_oversold'] = market_data['rsi'] < 30
            features['rsi_overbought'] = market_data['rsi'] > 70
            features['rsi_neutral'] = 30 <= market_data['rsi'] <= 70

        # Price action
        if 'price' in market_data and 'previous_price' in market_data:
            price_change = (market_data['price'] - market_data['previous_price']) / market_data['previous_price']
            features['price_increase'] = price_change > 0.02
            features['price_decrease'] = price_change < -0.02
            features['price_stable'] = abs(price_change) <= 0.02

        # Volume
        if 'volume' in market_data and 'avg_volume' in market_data:
            volume_ratio = market_data['volume'] / market_data['avg_volume']
            features['volume_spike'] = volume_ratio > 1.5
            features['volume_low'] = volume_ratio < 0.7
            features['volume_normal'] = 0.7 <= volume_ratio <= 1.5

        # Support/Resistance
        if all(k in market_data for k in ['price', 'support', 'resistance']):
            features['near_support'] = abs(market_data['price'] - market_data['support']) / market_data['price'] < 0.02
            features['near_resistance'] = abs(market_data['price'] - market_data['resistance']) / market_data['price'] < 0.02
            features['resistance_break'] = market_data['price'] > market_data['resistance'] * 1.01

        # Sentiment
        if 'sentiment' in market_data:
            features['sentiment_bullish'] = market_data['sentiment'] > 0.2
            features['sentiment_bearish'] = market_data['sentiment'] < -0.2
            features['sentiment_neutral'] = abs(market_data['sentiment']) <= 0.2

        return features

    def _detect_pattern_type(self, features: Dict[str, Any], outcome: Dict[str, Any]) -> str:
        """Auto-detect pattern type based on features and outcome"""
        # Simple heuristic-based pattern detection
        if features.get('rsi_oversold') and outcome.get('price_increase'):
            return 'bullish_reversal'
        elif features.get('rsi_overbought') and outcome.get('price_decrease'):
            return 'bearish_reversal'
        elif features.get('resistance_break') and outcome.get('continued_uptrend'):
            return 'breakout_pattern'
        elif features.get('volume_spike'):
            return 'volume_pattern'
        else:
            return 'general_pattern'

    def _calculate_pattern_similarity(self, current_features: Dict[str, Any], 
                                    pattern_conditions: Dict[str, Any]) -> float:
        """Calculate similarity between current features and pattern conditions"""
        if not pattern_conditions:
            return 0.0

        matches = 0
        total = len(pattern_conditions)

        for condition, expected_value in pattern_conditions.items():
            if condition in current_features:
                current_value = current_features[condition]
                if isinstance(expected_value, bool) and isinstance(current_value, bool):
                    if expected_value == current_value:
                        matches += 1
                elif isinstance(expected_value, (int, float)) and isinstance(current_value, (int, float)):
                    if abs(expected_value - current_value) / max(abs(expected_value), abs(current_value), 1) < 0.1:
                        matches += 1

        return matches / total

    def _add_pattern_to_knowledge_graph(self, pattern: MarketPattern):
        """Add significant patterns to knowledge graph"""
        node = KnowledgeNode(
            node_id=f"pattern_{pattern.pattern_id}",
            node_type="pattern",
            attributes={
                'pattern_type': pattern.pattern_type,
                'conditions': pattern.conditions,
                'outcomes': pattern.outcomes,
                'success_rate': pattern.success_rate,
                'sample_size': pattern.sample_size,
                'symbols': pattern.symbols,
                'market_regime': pattern.market_regime
            },
            timestamp=pattern.last_seen,
            confidence=pattern.success_rate
        )
        self.knowledge_graph.add_node(node)

class DocumentProcessor:
    """Advanced document processing and information extraction"""

    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.knowledge_graph = knowledge_graph
        self.processed_documents: Dict[str, DocumentEntity] = {}
        self.logger = logging.getLogger("DocumentProcessor")
        self.financial_keywords = self._initialize_financial_keywords()

    def _initialize_financial_keywords(self) -> Dict[str, List[str]]:
        """Initialize financial keyword dictionaries"""
        return {
            'positive_indicators': [
                'revenue growth', 'profit margin', 'earnings beat', 'guidance raise',
                'market share', 'strong performance', 'positive outlook', 'expansion',
                'innovation', 'competitive advantage', 'operational efficiency'
            ],
            'negative_indicators': [
                'revenue decline', 'margin compression', 'earnings miss', 'guidance cut',
                'market share loss', 'weak performance', 'negative outlook', 'restructuring',
                'regulatory issues', 'competitive pressure', 'operational challenges'
            ],
            'financial_metrics': [
                'eps', 'revenue', 'ebitda', 'free cash flow', 'debt-to-equity',
                'return on equity', 'gross margin', 'operating margin', 'p/e ratio'
            ],
            'market_events': [
                'earnings', 'merger', 'acquisition', 'ipo', 'stock split',
                'dividend', 'buyback', 'partnership', 'product launch'
            ]
        }

    def process_document(self, document_text: str, doc_type: str, 
                        metadata: Dict[str, Any]) -> Optional[DocumentEntity]:
        """Process a financial document and extract insights"""
        try:
            # Generate document ID
            doc_hash = hashlib.md5(document_text.encode()).hexdigest()
            entity_id = f"{doc_type}_{doc_hash[:12]}"

            # Check if already processed
            if entity_id in self.processed_documents:
                return self.processed_documents[entity_id]

            # Extract facts from document
            extracted_facts = self._extract_facts(document_text, doc_type)

            # Calculate sentiment
            sentiment_score = self._calculate_document_sentiment(document_text)

            # Calculate relevance score
            relevance_score = self._calculate_relevance(document_text, metadata)

            # Extract mentioned symbols
            symbols = self._extract_symbols(document_text, metadata)

            # Create document entity
            entity = DocumentEntity(
                entity_id=entity_id,
                entity_type=doc_type,
                content=document_text[:1000],  # Store first 1000 chars
                metadata=metadata,
                extracted_facts=extracted_facts,
                sentiment_score=sentiment_score,
                relevance_score=relevance_score,
                timestamp=datetime.now(),
                symbols=symbols
            )

            self.processed_documents[entity_id] = entity

            # Add to knowledge graph
            self._add_document_to_knowledge_graph(entity)

            self.logger.info(f"Processed document {entity_id} ({doc_type}) with {len(extracted_facts)} facts")
            return entity

        except Exception as e:
            self.logger.error(f"Error processing document: {e}")
            return None

    def _extract_facts(self, text: str, doc_type: str) -> List[Dict[str, Any]]:
        """Extract structured facts from document text"""
        facts = []
        text_lower = text.lower()

        # Extract financial metrics
        for metric in self.financial_keywords['financial_metrics']:
            if metric in text_lower:
                # Simple pattern matching for numbers near metrics
                import re
                pattern = f"{metric}.*?([0-9.,]+[%$]?)"
                matches = re.findall(pattern, text_lower)
                for match in matches:
                    facts.append({
                        'type': 'financial_metric',
                        'metric': metric,
                        'value': match,
                        'confidence': 0.8
                    })

        # Extract market events
        for event in self.financial_keywords['market_events']:
            if event in text_lower:
                facts.append({
                    'type': 'market_event',
                    'event': event,
                    'confidence': 0.7
                })

        # Extract positive/negative indicators
        for indicator in self.financial_keywords['positive_indicators']:
            if indicator in text_lower:
                facts.append({
                    'type': 'positive_indicator',
                    'indicator': indicator,
                    'confidence': 0.6
                })

        for indicator in self.financial_keywords['negative_indicators']:
            if indicator in text_lower:
                facts.append({
                    'type': 'negative_indicator',
                    'indicator': indicator,
                    'confidence': 0.6
                })

        return facts

    def _calculate_document_sentiment(self, text: str) -> float:
        """Calculate sentiment score for document"""
        text_lower = text.lower()
        positive_count = 0
        negative_count = 0

        for indicator in self.financial_keywords['positive_indicators']:
            positive_count += text_lower.count(indicator)

        for indicator in self.financial_keywords['negative_indicators']:
            negative_count += text_lower.count(indicator)

        total_indicators = positive_count + negative_count
        if total_indicators == 0:
            return 0.0

        return (positive_count - negative_count) / total_indicators

    def _calculate_relevance(self, text: str, metadata: Dict[str, Any]) -> float:
        """Calculate relevance score based on content and metadata"""
        relevance = 0.5  # Base relevance

        # Check for financial keywords
        text_lower = text.lower()
        keyword_count = 0
        for keyword_list in self.financial_keywords.values():
            for keyword in keyword_list:
                if keyword in text_lower:
                    keyword_count += 1

        # Boost relevance based on keyword density
        relevance += min(keyword_count / 100, 0.4)

        # Boost for recent documents
        if 'timestamp' in metadata:
            doc_time = metadata['timestamp']
            if isinstance(doc_time, str):
                doc_time = datetime.fromisoformat(doc_time)
            age_days = (datetime.now() - doc_time).days
            recency_boost = max(0, 0.1 - age_days / 100)
            relevance += recency_boost

        return min(relevance, 1.0)

    def _extract_symbols(self, text: str, metadata: Dict[str, Any]) -> List[str]:
        """Extract stock symbols from text and metadata"""
        symbols = []

        # Check metadata first
        if 'symbols' in metadata:
            symbols.extend(metadata['symbols'])

        # Simple pattern matching for stock symbols
        import re
        pattern = r'\b[A-Z]{1,5}\b'
        potential_symbols = re.findall(pattern, text)

        # Filter to likely stock symbols (basic heuristic)
        common_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'META', 'SPY', 'QQQ']
        for symbol in potential_symbols:
            if symbol in common_symbols or len(symbol) <= 5:
                if symbol not in symbols:
                    symbols.append(symbol)

        return symbols[:10]  # Limit to 10 symbols

    def _add_document_to_knowledge_graph(self, entity: DocumentEntity):
        """Add processed document to knowledge graph"""
        # Add document node
        doc_node = KnowledgeNode(
            node_id=f"doc_{entity.entity_id}",
            node_type="document",
            attributes={
                'doc_type': entity.entity_type,
                'sentiment_score': entity.sentiment_score,
                'relevance_score': entity.relevance_score,
                'fact_count': len(entity.extracted_facts),
                'symbols': entity.symbols
            },
            timestamp=entity.timestamp,
            confidence=entity.relevance_score
        )
        self.knowledge_graph.add_node(doc_node)

        # Create edges to symbol nodes
        for symbol in entity.symbols:
            symbol_node_id = f"symbol_{symbol}"

            # Create symbol node if it doesn't exist
            if symbol_node_id not in self.knowledge_graph.nodes:
                symbol_node = KnowledgeNode(
                    node_id=symbol_node_id,
                    node_type="symbol",
                    attributes={'symbol': symbol},
                    timestamp=datetime.now()
                )
                self.knowledge_graph.add_node(symbol_node)

            # Create edge between document and symbol
            edge = KnowledgeEdge(
                edge_id=f"doc_symbol_{entity.entity_id}_{symbol}",
                source_node=doc_node.node_id,
                target_node=symbol_node_id,
                relationship_type="mentions",
                strength=entity.relevance_score,
                attributes={'sentiment': entity.sentiment_score},
                timestamp=datetime.now()
            )
            self.knowledge_graph.add_edge(edge)

class DigitalBrain:
    """Main digital brain orchestrator that coordinates all knowledge components"""

    def __init__(self):
        self.knowledge_graph = KnowledgeGraph()
        self.pattern_engine = PatternRecognitionEngine(self.knowledge_graph)
        self.document_processor = DocumentProcessor(self.knowledge_graph)
        self.memory_consolidation_interval = 300  # 5 minutes
        self.last_consolidation = datetime.now()
        self.logger = logging.getLogger("DigitalBrain")

        # Initialize with some basic financial knowledge
        self._initialize_base_knowledge()

    def _initialize_base_knowledge(self):
        """Initialize the brain with basic financial knowledge"""
        try:
            # Add basic market concepts
            concepts = [
                ('support_resistance', 'concept', {'description': 'Price levels that act as barriers'}),
                ('technical_analysis', 'concept', {'description': 'Analysis based on price and volume data'}),
                ('fundamental_analysis', 'concept', {'description': 'Analysis based on company financials'}),
                ('market_sentiment', 'concept', {'description': 'Overall market mood and investor psychology'}),
                ('risk_management', 'concept', {'description': 'Strategies to limit potential losses'})
            ]

            for concept_id, node_type, attributes in concepts:
                node = KnowledgeNode(
                    node_id=concept_id,
                    node_type=node_type,
                    attributes=attributes,
                    timestamp=datetime.now()
                )
                self.knowledge_graph.add_node(node)

            # Add relationships between concepts
            relationships = [
                ('technical_analysis', 'support_resistance', 'uses', 0.8),
                ('risk_management', 'support_resistance', 'uses', 0.6),
                ('market_sentiment', 'fundamental_analysis', 'influences', 0.7)
            ]

            for source, target, rel_type, strength in relationships:
                edge = KnowledgeEdge(
                    edge_id=f"{source}_{target}_{rel_type}",
                    source_node=source,
                    target_node=target,
                    relationship_type=rel_type,
                    strength=strength,
                    attributes={},
                    timestamp=datetime.now()
                )
                self.knowledge_graph.add_edge(edge)

            self.logger.info("Digital brain initialized with base knowledge")

        except Exception as e:
            self.logger.error(f"Error initializing base knowledge: {e}")

    def process_market_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a market event and update knowledge"""
        try:
            # Safe regime extraction with comprehensive None checking
            regime_value = 'sideways'  # Default safe value
            
            if 'regime' in event_data and event_data['regime'] is not None:
                regime_data = event_data['regime']
                if isinstance(regime_data, str):
                    regime_value = regime_data.lower() if regime_data else 'sideways'
                else:
                    # Handle regime object with attributes
                    try:
                        if hasattr(regime_data, 'value'):
                            regime_value = regime_data.value.lower() if regime_data.value else 'sideways'
                        elif hasattr(regime_data, 'name'):
                            regime_value = regime_data.name.lower() if regime_data.name else 'sideways'
                        else:
                            regime_str = str(regime_data)
                            if regime_str and regime_str.lower() != 'none':
                                regime_value = regime_str.lower().replace('marketregime.', '')
                            else:
                                regime_value = 'sideways'
                    except Exception as e:
                        self.logger.debug(f"Regime processing error: {e}")
                        regime_value = 'sideways'
            
            # Create safe event data copy
            safe_event_data = event_data.copy()
            safe_event_data['regime'] = regime_value

            # Learn patterns from the event
            if 'outcome' in safe_event_data:
                self.pattern_engine.learn_pattern(
                    symbol=safe_event_data.get('symbol', 'MARKET'),
                    market_data=safe_event_data,
                    outcome=safe_event_data['outcome']
                )

            # Recognize current patterns
            current_patterns = self.pattern_engine.recognize_patterns(
                symbol=safe_event_data.get('symbol', 'MARKET'),
                current_data=safe_event_data
            )

            # Query related knowledge with safe parameters
            query = {}
            if 'symbol' in safe_event_data and safe_event_data['symbol']:
                query['symbol'] = safe_event_data['symbol']
            if 'pattern_type' in safe_event_data and safe_event_data['pattern_type']:
                query['pattern_type'] = safe_event_data['pattern_type']
            if 'event_type' in safe_event_data and safe_event_data['event_type']:
                query['event_type'] = safe_event_data['event_type']
            
            related_knowledge = self.knowledge_graph.query_knowledge(query)

            # Consolidate memory if needed
            if (datetime.now() - self.last_consolidation).seconds > self.memory_consolidation_interval:
                self._consolidate_memory()

            result = {
                'recognized_patterns': [(p.pattern_id, conf) for p, conf in current_patterns],
                'related_knowledge_count': len(related_knowledge),
                'new_knowledge_created': True,
                'confidence': max([conf for _, conf in current_patterns], default=0.0)
            }

            self.logger.debug(f"Processed market event for {safe_event_data.get('symbol', 'MARKET')}: "
                            f"{len(current_patterns)} patterns recognized")

            return result

        except Exception as e:
            self.logger.error(f"Error processing market event: {e}")
            return {'error': str(e)}

    def process_document(self, content: str, doc_type: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process a document and extract knowledge"""
        try:
            entity = self.document_processor.process_document(content, doc_type, metadata)

            if entity:
                result = {
                    'document_id': entity.entity_id,
                    'facts_extracted': len(entity.extracted_facts),
                    'sentiment_score': entity.sentiment_score,
                    'relevance_score': entity.relevance_score,
                    'symbols_mentioned': entity.symbols,
                    'knowledge_updated': True
                }

                self.logger.info(f"Processed document {entity.entity_id}: "
                               f"{len(entity.extracted_facts)} facts, "
                               f"sentiment: {entity.sentiment_score:.2f}")

                return result
            else:
                return {'error': 'Failed to process document'}

        except Exception as e:
            self.logger.error(f"Error processing document: {e}")
            return {'error': str(e)}

    def query_brain(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Query the digital brain for insights"""
        try:
            # Convert natural language query to structured query
            structured_query = self._parse_natural_query(query, context)

            # Query knowledge graph
            knowledge_results = self.knowledge_graph.query_knowledge(structured_query)

            # Get relevant patterns
            patterns = []
            if 'symbol' in structured_query:
                symbol_patterns = self.pattern_engine.recognize_patterns(
                    structured_query['symbol'], context or {}
                )
                patterns = [(p.pattern_type, conf) for p, conf in symbol_patterns[:5]]

            # Add original query to context for insights generation
            insights_context = context.copy() if context else {}
            insights_context['original_query'] = query

        # Generate insights
            insights = self._generate_insights(knowledge_results, patterns, insights_context)

            result = {
                'query': query,
                'structured_query': structured_query,
                'knowledge_matches': len(knowledge_results),
                'patterns_found': len(patterns),
                'insights': insights,
                'confidence': self._calculate_response_confidence(knowledge_results, patterns)
            }

            return result

        except Exception as e:
            self.logger.error(f"Error querying brain: {e}")
            return {'error': str(e)}

    def _parse_natural_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Parse natural language query into structured format"""
        structured = {}
        query_lower = query.lower()

        # Extract symbols
        import re
        symbols = re.findall(r'\b[A-Z]{1,5}\b', query)
        if symbols:
            structured['symbol'] = symbols[0]

        # Extract query type
        if any(word in query_lower for word in ['pattern', 'trend', 'signal']):
            structured['query_type'] = 'pattern'
        elif any(word in query_lower for word in ['news', 'document', 'report']):
            structured['query_type'] = 'document'
        elif any(word in query_lower for word in ['risk', 'volatility', 'drawdown']):
            structured['query_type'] = 'risk'

        # Extract time context
        if any(word in query_lower for word in ['recent', 'today', 'yesterday']):
            structured['timeframe'] = 'recent'
        elif any(word in query_lower for word in ['week', 'weekly']):
            structured['timeframe'] = 'week'
        elif any(word in query_lower for word in ['month', 'monthly']):
            structured['timeframe'] = 'month'

        # Add context if provided
        if context:
            structured.update({k: v for k, v in context.items() if k not in structured})

        return structured

    def _generate_insights(self, knowledge_results: List[Dict[str, Any]], 
                          patterns: List[Tuple[str, float]], context: Dict[str, Any] = None) -> List[str]:
        """Generate actionable insights from knowledge and patterns"""
        insights = []

        # Analyze knowledge results for chart patterns and concepts
        pattern_nodes = []
        concept_nodes = []

        for result in knowledge_results:
            if result['type'] == 'node':
                node = result['node']
                if node.node_type == 'chart_pattern':
                    pattern_nodes.append((node, result['score']))
                elif node.node_type == 'trading_concept':
                    concept_nodes.append((node, result['score']))

        # Pattern-based insights with specific knowledge
        for pattern_type, confidence in patterns:
            if confidence > 0.7:
                insights.append(f"🔴 Strong {pattern_type} signal detected (confidence: {confidence:.2f})")
                insights.append(f"   → Recommended action: Monitor for entry opportunity with volume confirmation")
            elif confidence > 0.5:
                insights.append(f"🟡 Moderate {pattern_type} pattern forming (confidence: {confidence:.2f})")
                insights.append(f"   → Wait for clearer confirmation before acting")

        # Knowledge graph pattern insights
        if pattern_nodes:
            top_pattern = pattern_nodes[0][0]  # Highest scoring pattern
            pattern_name = top_pattern.attributes.get('pattern_name', 'Unknown Pattern')

            # Add specific trading advice based on pattern type
            pattern_advice = {
                'head and shoulders': "⚠️ Bearish reversal - wait for neckline break with volume",
                'inverse head and shoulders': "📈 Bullish reversal - buy on neckline breakout",
                'double top': "🔴 Bearish signal - short on break below support valley",
                'double bottom': "🟢 Bullish signal - buy on break above resistance peak", 
                'ascending triangle': "📊 Bullish bias - buy breakout above horizontal resistance",
                'descending triangle': "📉 Bearish bias - short breakdown below horizontal support",
                'flag': "🚩 Continuation pattern - trade in direction of prior trend",
                'pennant': "📐 Brief consolidation - expect resumption of main trend",
                'breakout': "💥 Monitor volume - true breakouts need 1.5x average volume"
            }

            for pattern_key, advice in pattern_advice.items():
                if pattern_key in pattern_name.lower():
                    insights.append(advice)
                    break

        # Concept-based insights
        if concept_nodes and not pattern_nodes:
            top_concept = concept_nodes[0][0]
            concept_name = top_concept.attributes.get('concept_name', 'Unknown Concept')

            concept_guidance = {
                'support and resistance': "🎯 Key levels identified - trade bounces off support, breaks through resistance",
                'trend lines': "📈 Trend analysis - trade with the trend, watch for breakouts",
                'volume analysis': "📊 Volume confirms moves - high volume validates breakouts",
                'momentum': "⚡ Momentum shift detected - watch for trend continuation or reversal",
                'breakout trading': "🚀 Breakout setup - wait for volume confirmation and follow-through"
            }

            for concept_key, guidance in concept_guidance.items():
                if concept_key in concept_name.lower():
                    insights.append(guidance)
                    break

        # Risk management insights
        if context and 'volatility' in context:
            vol = context['volatility']
            if vol > 0.3:
                insights.append("⚠️ High volatility - reduce position size by 50%, use tighter stops")
            elif vol < 0.1:
                insights.append("📈 Low volatility - potential breakout environment, watch for expansion")

        # Market context insights
        if knowledge_results:
            high_confidence_count = len([k for k in knowledge_results if k['score'] > 0.8])
            if high_confidence_count >= 3:
                insights.append(f"🎯 Strong pattern match - {high_confidence_count} high-confidence signals aligned")
            elif high_confidence_count >= 1:
                insights.append(f"📊 Moderate pattern match - {high_confidence_count} reliable signal(s) detected")

        # Default insight if none generated
        if not insights:
            insights.append("📍 Pattern recognition active - monitoring market for trading opportunities")
            insights.append("💡 Tip: Ask specific questions like 'head and shoulders pattern' for detailed analysis")

        # For queries with no specific matches, provide general guidance based on query terms
        if not pattern_nodes and not concept_nodes:
            original_query = context.get('original_query', '') if context else ''
            query_lower = original_query.lower() if isinstance(original_query, str) else ""
            if 'support' in query_lower or 'resistance' in query_lower:
                insights.extend([
                    "📊 SUPPORT & RESISTANCE - Key Trading Concept:",
                    "• Support: Price level where buying interest prevents further decline",
                    "• Resistance: Price level where selling pressure prevents further advance", 
                    "• Role Reversal: Broken support becomes resistance (and vice versa)",
                    "• Strength: More tests = stronger level, volume adds significance",
                    "• Strategy: Buy near support, sell near resistance, trade breakouts"
                ])

        return insights

    def _calculate_response_confidence(self, knowledge_results: List[Dict[str, Any]], 
                                     patterns: List[Tuple[str, float]]) -> float:
        """Calculate confidence in the brain's response"""
        if not knowledge_results and not patterns:
            return 0.1

        knowledge_confidence = np.mean([k['score'] for k in knowledge_results]) if knowledge_results else 0
        pattern_confidence = np.mean([conf for _, conf in patterns]) if patterns else 0

        return (knowledge_confidence + pattern_confidence) / 2

    def _consolidate_memory(self):
        """Enhanced memory consolidation with load balancing"""
        try:
            consolidation_start = time.time()

            # Adaptive thresholds based on memory usage
            total_patterns = len(self.pattern_engine.patterns)
            total_docs = len(self.document_processor.processed_documents)
            total_nodes = len(self.knowledge_graph.nodes)

            # Dynamic thresholds based on load
            if total_patterns > 1000:
                min_success_rate = 0.4
                min_samples = 5
            elif total_patterns > 500:
                min_success_rate = 0.35
                min_samples = 3
            else:
                min_success_rate = 0.3
                min_samples = 3

            # Remove low-confidence patterns with load balancing
            patterns_to_remove = []
            pattern_items = list(self.pattern_engine.patterns.items())

            # Process in batches to avoid memory spikes
            batch_size = min(100, len(pattern_items) // 10 + 1)

            for i in range(0, len(pattern_items), batch_size):
                batch = pattern_items[i:i + batch_size]

                for pattern_id, pattern in batch:
                    # Enhanced removal criteria
                    age_days = (datetime.now() - pattern.last_seen).days

                    should_remove = (
                        (pattern.success_rate < min_success_rate and pattern.sample_size < min_samples) or
                        (age_days > 60 and pattern.success_rate < 0.5) or
                        (age_days > 90 and pattern.sample_size < 5)
                    )

                    if should_remove:
                        patterns_to_remove.append(pattern_id)

                # Yield control periodically
                if i % (batch_size * 5) == 0:
                    time.sleep(0.001)  # Micro-pause for system responsiveness

            # Remove patterns in batches
            for i in range(0, len(patterns_to_remove), batch_size):
                batch = patterns_to_remove[i:i + batch_size]
                for pattern_id in batch:
                    if pattern_id in self.pattern_engine.patterns:
                        del self.pattern_engine.patterns[pattern_id]

            # Enhanced document cleanup with relevance scoring
            docs_to_remove = []
            doc_items = list(self.document_processor.processed_documents.items())

            for i in range(0, len(doc_items), batch_size):
                batch = doc_items[i:i + batch_size]

                for doc_id, doc in batch:
                    age_days = (datetime.now() - doc.timestamp).days

                    # Dynamic relevance threshold
                    relevance_threshold = 0.3 if total_docs < 500 else 0.4

                    should_remove = (
                        (age_days > 30 and doc.relevance_score < relevance_threshold) or
                        (age_days > 60 and doc.relevance_score < 0.5) or
                        (age_days > 90)
                    )

                    if should_remove:
                        docs_to_remove.append(doc_id)

                if i % (batch_size * 5) == 0:
                    time.sleep(0.001)

            # Remove documents in batches
            for i in range(0, len(docs_to_remove), batch_size):
                batch = docs_to_remove[i:i + batch_size]
                for doc_id in batch:
                    if doc_id in self.document_processor.processed_documents:
                        del self.document_processor.processed_documents[doc_id]

            # Knowledge graph cleanup and re-indexing
            if total_nodes > 1000:
                self._cleanup_knowledge_graph()
                self._rebuild_indexes()

            consolidation_time = time.time() - consolidation_start
            self.last_consolidation = datetime.now()

            self.logger.info(f"Enhanced memory consolidation completed in {consolidation_time:.2f}s: "
                           f"removed {len(patterns_to_remove)} patterns and {len(docs_to_remove)} documents")

        except Exception as e:
            self.logger.error(f"Error during enhanced memory consolidation: {e}")

    def _cleanup_knowledge_graph(self):
        """Clean up orphaned nodes and low-value edges"""
        nodes_to_remove = []

        for node_id, node in self.knowledge_graph.nodes.items():
            # Remove nodes with very low confidence and no connections
            if (node.confidence < 0.2 and 
                len(self.knowledge_graph.adjacency[node_id]) == 0 and
                len(self.knowledge_graph.reverse_adjacency[node_id]) == 0):
                nodes_to_remove.append(node_id)

        for node_id in nodes_to_remove:
            if node_id in self.knowledge_graph.nodes:
                del self.knowledge_graph.nodes[node_id]
                self.knowledge_graph.node_index[self.knowledge_graph.nodes[node_id].node_type].discard(node_id)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get knowledge graph performance statistics"""
        return {
            'total_nodes': len(self.knowledge_graph.nodes),
            'total_edges': len(self.knowledge_graph.edges),
            'index_sizes': {
                'pattern_index': len(self.knowledge_graph.pattern_index),
                'symbol_index': len(self.knowledge_graph.symbol_index),
                'text_search_index': len(self.knowledge_graph.text_search_index),
                'confidence_buckets': len(self.knowledge_graph.confidence_index)
            },
            'cache_stats': {
                'cache_size': len(self.knowledge_graph.fuzzy_search_cache),
                'cache_enabled': self.knowledge_graph.enable_caching,
                'max_cache_size': self.knowledge_graph.max_cache_size
            },
            'query_performance': self.knowledge_graph.query_performance_stats,
            'node_types': dict(Counter(node.node_type for node in self.knowledge_graph.nodes.values()))
        }

    def optimize_indexes(self):
        """Optimize all indexes for better performance"""
        self.logger.info("Starting index optimization...")
        
        # Clear existing optimized indexes
        self.knowledge_graph.text_search_index.clear()
        self.knowledge_graph.pattern_category_index.clear()
        self.knowledge_graph.concept_hierarchy_index.clear()
        self.knowledge_graph.temporal_index.clear()
        
        # Rebuild all indexes with optimization
        for node_id, node in self.knowledge_graph.nodes.items():
            # Rebuild text search index
            searchable_text = self.knowledge_graph._extract_searchable_text(node)
            for keyword in searchable_text:
                if len(keyword) > 2:
                    self.knowledge_graph.text_search_index[keyword.lower()].add(node_id)
            
            # Rebuild pattern category index
            if node.node_type == 'chart_pattern':
                category = node.attributes.get('category', 'general')
                self.knowledge_graph.pattern_category_index[category].add(node_id)
            
            # Rebuild concept hierarchy
            if node.node_type == 'trading_concept':
                concept_name = node.attributes.get('concept_name', '')
                if concept_name:
                    self.knowledge_graph.concept_hierarchy_index[concept_name.lower()].add(node_id)
            
            # Rebuild temporal index
            date_bucket = node.timestamp.strftime("%Y-%m")
            if node_id not in self.knowledge_graph.temporal_index[date_bucket]:
                self.knowledge_graph.temporal_index[date_bucket].append(node_id)
        
        # Sort temporal indexes
        for date_bucket in self.knowledge_graph.temporal_index:
            self.knowledge_graph.temporal_index[date_bucket].sort(
                key=lambda nid: self.knowledge_graph.nodes[nid].timestamp
            )
        
        # Clear cache to force fresh results
        self.knowledge_graph.fuzzy_search_cache.clear()
        
        self.logger.info(f"Index optimization completed. Text search index: {len(self.knowledge_graph.text_search_index)} keywords")

    def _rebuild_indexes(self):
        """Rebuild knowledge graph indexes for optimal performance"""
        self.knowledge_graph.pattern_index.clear()
        self.knowledge_graph.symbol_index.clear()
        self.knowledge_graph.confidence_index.clear()
        self.knowledge_graph.attribute_index.clear()

        for node_id, node in self.knowledge_graph.nodes.items():
            # Rebuild pattern index
            if node.node_type == 'chart_pattern':
                pattern_name = node.attributes.get('pattern_name', '')
                if pattern_name:
                    self.knowledge_graph.pattern_index[pattern_name.lower()].add(node_id)

            # Rebuild symbol index
            symbols = node.attributes.get('symbols', [])
            for symbol in symbols:
                self.knowledge_graph.symbol_index[symbol.upper()].add(node_id)

            # Rebuild confidence index with sorting
            confidence_bucket = round(node.confidence, 1)
            if confidence_bucket not in self.knowledge_graph.confidence_index:
                self.knowledge_graph.confidence_index[confidence_bucket] = []
            self.knowledge_graph.confidence_index[confidence_bucket].append(node_id)
        
        # Sort confidence indexes
        for bucket in self.knowledge_graph.confidence_index:
            self.knowledge_graph.confidence_index[bucket].sort(
                key=lambda nid: self.knowledge_graph.nodes[nid].confidence, reverse=True
            )
        
        # Trigger full optimization
        self.optimize_indexes()

    def get_brain_status(self) -> Dict[str, Any]:
        """Get current status of the digital brain"""
        return {
            'knowledge_nodes': len(self.knowledge_graph.nodes),
            'knowledge_edges': len(self.knowledge_graph.edges),
            'learned_patterns': len(self.pattern_engine.patterns),
            'processed_documents': len(self.document_processor.processed_documents),
            'last_consolidation': self.last_consolidation.isoformat(),
            'brain_initialized': True,
            'average_pattern_confidence': np.mean([p.success_rate for p in self.pattern_engine.patterns.values()]) if self.pattern_engine.patterns else 0,
            'memory_health': 'optimal'
        }

    def save_brain_state(self, filepath: str) -> bool:
        """Save the entire brain state to file"""
        try:
            # Save knowledge graph
            kg_success = self.knowledge_graph.save_to_file(f"{filepath}_knowledge.json")

            # Save patterns
            patterns_data = {}
            for pattern_id, pattern in self.pattern_engine.patterns.items():
                patterns_data[pattern_id] = {
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

            with open(f"{filepath}_patterns.json", 'w') as f:
                json.dump(patterns_data, f, indent=2)

            # Save documents (metadata only)
            docs_data = {}
            for doc_id, doc in self.document_processor.processed_documents.items():
                docs_data[doc_id] = {
                    'entity_id': doc.entity_id,
                    'entity_type': doc.entity_type,
                    'metadata': doc.metadata,
                    'sentiment_score': doc.sentiment_score,
                    'relevance_score': doc.relevance_score,
                    'timestamp': doc.timestamp.isoformat(),
                    'symbols': doc.symbols,
                    'fact_count': len(doc.extracted_facts)
                }

            with open(f"{filepath}_documents.json", 'w') as f:
                json.dump(docs_data, f, indent=2)

            if kg_success:
                self.logger.info(f"Brain state saved to {filepath}")
                return True
            else:
                return False

        except Exception as e:
            self.logger.error(f"Error saving brain state: {e}")
            return False

    def _integrate_processed_documents(self):
        """Integrate processed documents into knowledge graph"""
        try:
            import os
            # Load document memory bank
            memory_bank_file = "document_memory_bank.json"
            if os.path.exists(memory_bank_file):
                with open(memory_bank_file, 'r') as f:
                    memory_data = json.load(f)

                stored_docs = memory_data.get('stored_documents', {})
                for doc_id, doc_info in stored_docs.items():
                    # Create knowledge nodes for each document
                    node_id = f"doc_{doc_id}"
                    node = KnowledgeNode(
                        node_id=node_id,
                        node_type='document',
                        attributes={
                            'title': doc_info.get('title', 'Unknown'),
                            'document_type': doc_info.get('document_type', 'trading_literature'),
                            'content_summary': doc_info.get('content', '')[:200] + '...',
                            'page_count': doc_info.get('page_count', 0),
                            'processed_date': doc_info.get('processed_date', ''),
                            'importance': 0.7
                        },
                        timestamp=datetime.now(),
                        confidence=0.8
                    )
                    self.knowledge_graph.add_node(node)

                    # Create relationships to chart patterns
                    if 'chart' in doc_info.get('title', '').lower() or 'pattern' in doc_info.get('title', '').lower():
                        edge = KnowledgeEdge(
                            edge_id=f"{node_id}_chart_patterns",
                            source_node=node_id,
                            target_node='chart_patterns',
                            relationship_type='contains_knowledge_about',
                            strength=0.9,
                            attributes={'relationship_type': 'document_to_concept'},
                            timestamp=datetime.now()
                        )
                        self.knowledge_graph.add_edge(edge)

                self.logger.info(f"Integrated {len(stored_docs)} documents into knowledge graph")

        except Exception as e:
            self.logger.error(f"Error integrating processed documents: {e}")