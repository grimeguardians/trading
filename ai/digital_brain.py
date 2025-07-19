"""
Digital Brain - Advanced AI knowledge engine for trading
"""

import asyncio
import logging
import json
import pickle
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path

import anthropic
from anthropic import Anthropic
from openai import OpenAI
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import Settings
from ai.conversational_ai import ConversationalAI
from utils.logger import setup_logger

@dataclass
class KnowledgeNode:
    """Represents a knowledge node in the digital brain"""
    node_id: str
    content: str
    category: str
    subcategory: str
    confidence: float
    source: str
    timestamp: datetime
    embeddings: Optional[List[float]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class TradingPattern:
    """Represents a trading pattern in the knowledge base"""
    pattern_id: str
    name: str
    description: str
    conditions: List[str]
    success_rate: float
    risk_level: str
    market_conditions: List[str]
    examples: List[Dict]
    timestamp: datetime

@dataclass
class MarketInsight:
    """Represents a market insight"""
    insight_id: str
    title: str
    content: str
    confidence: float
    timeframe: str
    affected_assets: List[str]
    impact_score: float
    timestamp: datetime

class DigitalBrain:
    """Advanced AI knowledge engine for trading decisions"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = setup_logger("DigitalBrain", settings.LOG_LEVEL)
        
        # Initialize AI clients
        self.anthropic_client = None
        self.openai_client = None
        self.conversational_ai = ConversationalAI(settings)
        
        # Knowledge storage
        self.knowledge_base: Dict[str, KnowledgeNode] = {}
        self.trading_patterns: Dict[str, TradingPattern] = {}
        self.market_insights: Dict[str, MarketInsight] = {}
        
        # Search and retrieval
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.knowledge_vectors = None
        self.is_vectorizer_fitted = False
        
        # Memory and context
        self.short_term_memory = []
        self.long_term_memory = []
        self.context_window = 10
        
        # Performance tracking
        self.query_history = []
        self.accuracy_metrics = {}
        
        # Storage paths
        self.knowledge_path = Path("data/knowledge_base.pkl")
        self.patterns_path = Path("data/trading_patterns.pkl")
        self.insights_path = Path("data/market_insights.pkl")
        
        # Ensure data directory exists
        Path("data").mkdir(exist_ok=True)
        
    async def initialize(self):
        """Initialize the Digital Brain"""
        self.logger.info("🧠 Initializing Digital Brain")
        
        try:
            # Initialize AI clients
            await self._initialize_ai_clients()
            
            # Load existing knowledge
            await self._load_knowledge_base()
            
            # Initialize core trading knowledge
            await self._initialize_trading_knowledge()
            
            # Build search indexes
            await self._build_search_indexes()
            
            self.logger.info("✅ Digital Brain initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Digital Brain: {e}")
            raise
    
    async def _initialize_ai_clients(self):
        """Initialize AI service clients"""
        try:
            # Initialize Anthropic client
            if self.settings.ANTHROPIC_API_KEY:
                self.anthropic_client = Anthropic(api_key=self.settings.ANTHROPIC_API_KEY)
                self.logger.info("✅ Anthropic client initialized for Digital Brain")
            
            # Initialize OpenAI client
            if self.settings.OPENAI_API_KEY:
                self.openai_client = OpenAI(api_key=self.settings.OPENAI_API_KEY)
                self.logger.info("✅ OpenAI client initialized for Digital Brain")
                
        except Exception as e:
            self.logger.error(f"❌ Error initializing AI clients: {e}")
    
    async def _load_knowledge_base(self):
        """Load existing knowledge base from storage"""
        try:
            # Load knowledge nodes
            if self.knowledge_path.exists():
                with open(self.knowledge_path, 'rb') as f:
                    self.knowledge_base = pickle.load(f)
                self.logger.info(f"📚 Loaded {len(self.knowledge_base)} knowledge nodes")
            
            # Load trading patterns
            if self.patterns_path.exists():
                with open(self.patterns_path, 'rb') as f:
                    self.trading_patterns = pickle.load(f)
                self.logger.info(f"📈 Loaded {len(self.trading_patterns)} trading patterns")
            
            # Load market insights
            if self.insights_path.exists():
                with open(self.insights_path, 'rb') as f:
                    self.market_insights = pickle.load(f)
                self.logger.info(f"💡 Loaded {len(self.market_insights)} market insights")
                
        except Exception as e:
            self.logger.error(f"Error loading knowledge base: {e}")
            # Initialize empty knowledge base
            self.knowledge_base = {}
            self.trading_patterns = {}
            self.market_insights = {}
    
    async def _initialize_trading_knowledge(self):
        """Initialize core trading knowledge"""
        try:
            # Core trading concepts
            trading_concepts = [
                {
                    "content": "Support and resistance levels are key price points where buying or selling pressure typically emerges. Support is a price level where buying interest is strong enough to prevent further decline, while resistance is where selling pressure prevents further advance.",
                    "category": "technical_analysis",
                    "subcategory": "support_resistance",
                    "confidence": 0.95
                },
                {
                    "content": "Risk management is fundamental to successful trading. The 2% rule suggests never risking more than 2% of your account on a single trade. Position sizing should be based on your risk tolerance and the distance to your stop loss.",
                    "category": "risk_management",
                    "subcategory": "position_sizing",
                    "confidence": 0.98
                },
                {
                    "content": "Moving averages are trend-following indicators that smooth out price action. The 50-day and 200-day moving averages are commonly watched by institutional traders. A golden cross occurs when the 50-day MA crosses above the 200-day MA.",
                    "category": "technical_analysis",
                    "subcategory": "moving_averages",
                    "confidence": 0.92
                },
                {
                    "content": "Market sentiment indicators like the VIX (volatility index) can help gauge market fear and greed. High VIX readings often correspond with market bottoms, while low VIX readings may indicate complacency.",
                    "category": "market_sentiment",
                    "subcategory": "volatility_indicators",
                    "confidence": 0.88
                },
                {
                    "content": "Fibonacci retracement levels (23.6%, 38.2%, 50%, 61.8%) are commonly used to identify potential reversal points. These levels often act as support or resistance during pullbacks in trending markets.",
                    "category": "technical_analysis",
                    "subcategory": "fibonacci",
                    "confidence": 0.85
                }
            ]
            
            # Add concepts to knowledge base
            for concept in trading_concepts:
                await self.add_knowledge(
                    content=concept["content"],
                    category=concept["category"],
                    subcategory=concept["subcategory"],
                    confidence=concept["confidence"],
                    source="core_trading_knowledge"
                )
            
            # Initialize trading patterns
            await self._initialize_trading_patterns()
            
            self.logger.info("✅ Core trading knowledge initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing trading knowledge: {e}")
    
    async def _initialize_trading_patterns(self):
        """Initialize common trading patterns"""
        try:
            patterns = [
                {
                    "name": "Bull Flag",
                    "description": "A brief consolidation after a strong upward move, typically breaks out to the upside",
                    "conditions": ["Strong upward move", "Brief consolidation", "Decreasing volume during consolidation", "Breakout on increased volume"],
                    "success_rate": 0.68,
                    "risk_level": "Medium",
                    "market_conditions": ["Uptrend", "Bull market"]
                },
                {
                    "name": "Head and Shoulders",
                    "description": "A reversal pattern with three peaks, the middle being the highest",
                    "conditions": ["Three peaks", "Middle peak highest", "Neckline support", "Volume confirmation"],
                    "success_rate": 0.72,
                    "risk_level": "Medium",
                    "market_conditions": ["Uptrend reversal", "Distribution phase"]
                },
                {
                    "name": "Double Bottom",
                    "description": "A bullish reversal pattern with two equal lows",
                    "conditions": ["Two equal lows", "Resistance between lows", "Volume confirmation on breakout"],
                    "success_rate": 0.65,
                    "risk_level": "Medium",
                    "market_conditions": ["Downtrend reversal", "Accumulation phase"]
                }
            ]
            
            for pattern_data in patterns:
                pattern = TradingPattern(
                    pattern_id=hashlib.md5(pattern_data["name"].encode()).hexdigest(),
                    name=pattern_data["name"],
                    description=pattern_data["description"],
                    conditions=pattern_data["conditions"],
                    success_rate=pattern_data["success_rate"],
                    risk_level=pattern_data["risk_level"],
                    market_conditions=pattern_data["market_conditions"],
                    examples=[],
                    timestamp=datetime.now()
                )
                
                self.trading_patterns[pattern.pattern_id] = pattern
            
            self.logger.info(f"✅ Initialized {len(patterns)} trading patterns")
            
        except Exception as e:
            self.logger.error(f"Error initializing trading patterns: {e}")
    
    async def _build_search_indexes(self):
        """Build search indexes for knowledge retrieval"""
        try:
            if not self.knowledge_base:
                return
            
            # Prepare documents for vectorization
            documents = []
            for node in self.knowledge_base.values():
                doc = f"{node.category} {node.subcategory} {node.content}"
                documents.append(doc)
            
            # Fit vectorizer and create vectors
            if documents:
                self.knowledge_vectors = self.vectorizer.fit_transform(documents)
                self.is_vectorizer_fitted = True
                self.logger.info("✅ Search indexes built successfully")
            
        except Exception as e:
            self.logger.error(f"Error building search indexes: {e}")
    
    async def query(self, query: str, context: Dict = None) -> str:
        """Process a query and return intelligent response"""
        try:
            # Add query to history
            self.query_history.append({
                "query": query,
                "timestamp": datetime.now(),
                "context": context
            })
            
            # Retrieve relevant knowledge
            relevant_knowledge = await self._retrieve_relevant_knowledge(query)
            
            # Build enhanced context
            enhanced_context = await self._build_enhanced_context(query, context, relevant_knowledge)
            
            # Generate response using AI
            response = await self._generate_ai_response(query, enhanced_context)
            
            # Update memory
            await self._update_memory(query, response, context)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return "I apologize, but I encountered an error processing your query. Please try again."
    
    async def _retrieve_relevant_knowledge(self, query: str, top_k: int = 5) -> List[KnowledgeNode]:
        """Retrieve relevant knowledge nodes for a query"""
        try:
            if not self.is_vectorizer_fitted or not self.knowledge_base:
                return []
            
            # Vectorize query
            query_vector = self.vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.knowledge_vectors)[0]
            
            # Get top k results
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # Return relevant nodes
            knowledge_nodes = list(self.knowledge_base.values())
            relevant_nodes = [knowledge_nodes[i] for i in top_indices if similarities[i] > 0.1]
            
            return relevant_nodes
            
        except Exception as e:
            self.logger.error(f"Error retrieving relevant knowledge: {e}")
            return []
    
    async def _build_enhanced_context(self, query: str, context: Dict, relevant_knowledge: List[KnowledgeNode]) -> Dict:
        """Build enhanced context for AI response generation"""
        try:
            enhanced_context = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "relevant_knowledge": [],
                "trading_patterns": [],
                "market_insights": [],
                "user_context": context or {},
                "memory_context": self._get_memory_context()
            }
            
            # Add relevant knowledge
            for node in relevant_knowledge:
                enhanced_context["relevant_knowledge"].append({
                    "content": node.content,
                    "category": node.category,
                    "subcategory": node.subcategory,
                    "confidence": node.confidence
                })
            
            # Add relevant trading patterns
            pattern_keywords = query.lower().split()
            for pattern in self.trading_patterns.values():
                if any(keyword in pattern.name.lower() or keyword in pattern.description.lower() 
                       for keyword in pattern_keywords):
                    enhanced_context["trading_patterns"].append({
                        "name": pattern.name,
                        "description": pattern.description,
                        "success_rate": pattern.success_rate,
                        "risk_level": pattern.risk_level
                    })
            
            # Add recent market insights
            recent_insights = [
                insight for insight in self.market_insights.values()
                if (datetime.now() - insight.timestamp).days <= 7
            ]
            
            for insight in recent_insights[:3]:  # Top 3 recent insights
                enhanced_context["market_insights"].append({
                    "title": insight.title,
                    "content": insight.content,
                    "confidence": insight.confidence,
                    "impact_score": insight.impact_score
                })
            
            return enhanced_context
            
        except Exception as e:
            self.logger.error(f"Error building enhanced context: {e}")
            return {"query": query, "error": "Context building failed"}
    
    async def _generate_ai_response(self, query: str, context: Dict) -> str:
        """Generate AI response using enhanced context"""
        try:
            # Create comprehensive prompt
            prompt = f"""
            You are an advanced AI trading assistant with deep market knowledge and experience.
            
            User Query: {query}
            
            Relevant Knowledge:
            {json.dumps(context.get('relevant_knowledge', []), indent=2)}
            
            Trading Patterns:
            {json.dumps(context.get('trading_patterns', []), indent=2)}
            
            Market Insights:
            {json.dumps(context.get('market_insights', []), indent=2)}
            
            User Context:
            {json.dumps(context.get('user_context', {}), indent=2)}
            
            Memory Context:
            {context.get('memory_context', 'No previous context')}
            
            Instructions:
            1. Provide a comprehensive, accurate response based on the knowledge available
            2. Include specific examples and actionable insights where relevant
            3. Consider risk management and market conditions
            4. Be precise about confidence levels and limitations
            5. If discussing specific trades or strategies, emphasize risk management
            
            Response should be professional, informative, and practical for trading decisions.
            """
            
            # Use Anthropic Claude as primary AI
            if self.anthropic_client:
                response = self.anthropic_client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            
            # Fallback to OpenAI
            elif self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000
                )
                return response.choices[0].message.content
            
            else:
                # Fallback to simple response
                return await self._generate_simple_response(query, context)
                
        except Exception as e:
            self.logger.error(f"Error generating AI response: {e}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again."
    
    async def _generate_simple_response(self, query: str, context: Dict) -> str:
        """Generate simple response when AI services are unavailable"""
        try:
            # Check if we have relevant knowledge
            relevant_knowledge = context.get('relevant_knowledge', [])
            
            if relevant_knowledge:
                response = f"Based on my knowledge about {query}:\n\n"
                
                for knowledge in relevant_knowledge[:3]:  # Top 3 results
                    response += f"• {knowledge['content']}\n\n"
                
                response += "For more detailed analysis, please ensure AI services are properly configured."
                return response
            
            else:
                return "I don't have specific knowledge about that topic in my current database. Please try a different query or add relevant knowledge to the system."
                
        except Exception as e:
            self.logger.error(f"Error generating simple response: {e}")
            return "I'm unable to provide a response at this time."
    
    def _get_memory_context(self) -> str:
        """Get relevant memory context"""
        try:
            if not self.short_term_memory:
                return "No recent context available"
            
            context = "Recent context:\n"
            for memory in self.short_term_memory[-3:]:  # Last 3 memories
                context += f"- {memory['summary']}\n"
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error getting memory context: {e}")
            return "Context retrieval failed"
    
    async def _update_memory(self, query: str, response: str, context: Dict):
        """Update memory with new interaction"""
        try:
            memory_entry = {
                "query": query,
                "response": response[:200] + "..." if len(response) > 200 else response,
                "summary": f"Discussed {query[:50]}...",
                "timestamp": datetime.now(),
                "context": context
            }
            
            # Add to short-term memory
            self.short_term_memory.append(memory_entry)
            
            # Keep only recent memories
            if len(self.short_term_memory) > self.context_window:
                self.short_term_memory = self.short_term_memory[-self.context_window:]
            
            # Optionally move to long-term memory
            if len(self.short_term_memory) >= self.context_window:
                self.long_term_memory.append(self.short_term_memory[0])
                
        except Exception as e:
            self.logger.error(f"Error updating memory: {e}")
    
    async def add_knowledge(self, content: str, category: str, subcategory: str, 
                          confidence: float, source: str = "user") -> str:
        """Add new knowledge to the brain"""
        try:
            node_id = hashlib.md5(f"{content}{category}{subcategory}".encode()).hexdigest()
            
            knowledge_node = KnowledgeNode(
                node_id=node_id,
                content=content,
                category=category,
                subcategory=subcategory,
                confidence=confidence,
                source=source,
                timestamp=datetime.now(),
                metadata={"added_by": source}
            )
            
            self.knowledge_base[node_id] = knowledge_node
            
            # Rebuild search indexes
            await self._build_search_indexes()
            
            # Save to storage
            await self._save_knowledge_base()
            
            self.logger.info(f"✅ Added knowledge node: {category}/{subcategory}")
            return node_id
            
        except Exception as e:
            self.logger.error(f"Error adding knowledge: {e}")
            raise
    
    async def add_trading_pattern(self, pattern: TradingPattern) -> str:
        """Add a new trading pattern"""
        try:
            self.trading_patterns[pattern.pattern_id] = pattern
            
            # Save to storage
            await self._save_knowledge_base()
            
            self.logger.info(f"✅ Added trading pattern: {pattern.name}")
            return pattern.pattern_id
            
        except Exception as e:
            self.logger.error(f"Error adding trading pattern: {e}")
            raise
    
    async def add_market_insight(self, insight: MarketInsight) -> str:
        """Add a new market insight"""
        try:
            self.market_insights[insight.insight_id] = insight
            
            # Save to storage
            await self._save_knowledge_base()
            
            self.logger.info(f"✅ Added market insight: {insight.title}")
            return insight.insight_id
            
        except Exception as e:
            self.logger.error(f"Error adding market insight: {e}")
            raise
    
    async def _save_knowledge_base(self):
        """Save knowledge base to storage"""
        try:
            # Save knowledge nodes
            with open(self.knowledge_path, 'wb') as f:
                pickle.dump(self.knowledge_base, f)
            
            # Save trading patterns
            with open(self.patterns_path, 'wb') as f:
                pickle.dump(self.trading_patterns, f)
            
            # Save market insights
            with open(self.insights_path, 'wb') as f:
                pickle.dump(self.market_insights, f)
            
            self.logger.debug("💾 Knowledge base saved to storage")
            
        except Exception as e:
            self.logger.error(f"Error saving knowledge base: {e}")
    
    async def analyze_market_sentiment(self, market_data: Dict) -> Dict:
        """Analyze market sentiment using digital brain"""
        try:
            sentiment_query = f"""
            Analyze the current market sentiment based on this data:
            {json.dumps(market_data, indent=2)}
            
            Consider:
            1. Price movements and trends
            2. Volume patterns
            3. Volatility indicators
            4. Market breadth
            5. Economic indicators
            
            Provide sentiment score (-1 to 1) and reasoning.
            """
            
            response = await self.query(sentiment_query, {"market_data": market_data})
            
            # Extract sentiment score (simplified)
            sentiment_score = 0.0
            if "bullish" in response.lower() or "positive" in response.lower():
                sentiment_score = 0.6
            elif "bearish" in response.lower() or "negative" in response.lower():
                sentiment_score = -0.6
            
            return {
                "sentiment_score": sentiment_score,
                "sentiment_text": response,
                "confidence": 0.8,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market sentiment: {e}")
            return {"sentiment_score": 0.0, "sentiment_text": "Analysis failed", "confidence": 0.0}
    
    async def get_trading_recommendation(self, symbol: str, analysis: Dict) -> Dict:
        """Get trading recommendation for a symbol"""
        try:
            recommendation_query = f"""
            Provide a trading recommendation for {symbol} based on this analysis:
            {json.dumps(analysis, indent=2)}
            
            Consider:
            1. Technical indicators
            2. Market conditions
            3. Risk factors
            4. Entry/exit points
            5. Risk management
            
            Provide: action (buy/sell/hold), confidence, reasoning, and risk level.
            """
            
            response = await self.query(recommendation_query, {"symbol": symbol, "analysis": analysis})
            
            # Extract recommendation (simplified)
            action = "hold"
            confidence = 0.5
            
            if "buy" in response.lower() and "strong" in response.lower():
                action = "buy"
                confidence = 0.8
            elif "sell" in response.lower() and "strong" in response.lower():
                action = "sell"
                confidence = 0.8
            elif "buy" in response.lower():
                action = "buy"
                confidence = 0.6
            elif "sell" in response.lower():
                action = "sell"
                confidence = 0.6
            
            return {
                "symbol": symbol,
                "action": action,
                "confidence": confidence,
                "reasoning": response,
                "risk_level": "medium",
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting trading recommendation: {e}")
            return {"symbol": symbol, "action": "hold", "confidence": 0.0, "reasoning": "Analysis failed"}
    
    async def get_status(self) -> Dict:
        """Get Digital Brain status"""
        return {
            "knowledge_nodes": len(self.knowledge_base),
            "trading_patterns": len(self.trading_patterns),
            "market_insights": len(self.market_insights),
            "vectorizer_fitted": self.is_vectorizer_fitted,
            "memory_entries": len(self.short_term_memory),
            "query_history": len(self.query_history),
            "ai_services": {
                "anthropic": self.anthropic_client is not None,
                "openai": self.openai_client is not None
            }
        }
    
    def get_knowledge_summary(self) -> Dict:
        """Get summary of knowledge base"""
        try:
            categories = {}
            for node in self.knowledge_base.values():
                if node.category not in categories:
                    categories[node.category] = {}
                if node.subcategory not in categories[node.category]:
                    categories[node.category][node.subcategory] = 0
                categories[node.category][node.subcategory] += 1
            
            return {
                "total_nodes": len(self.knowledge_base),
                "categories": categories,
                "average_confidence": sum(node.confidence for node in self.knowledge_base.values()) / len(self.knowledge_base) if self.knowledge_base else 0,
                "last_updated": max(node.timestamp for node in self.knowledge_base.values()) if self.knowledge_base else None
            }
            
        except Exception as e:
            self.logger.error(f"Error getting knowledge summary: {e}")
            return {"total_nodes": 0, "categories": {}, "error": str(e)}
