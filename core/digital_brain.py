"""
Digital Brain integration for advanced market intelligence
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass
import json

# Import the existing knowledge engine
from knowledge_engine import KnowledgeGraph, KnowledgeNode, KnowledgeEdge, MarketPattern

# AI imports
import anthropic
import openai
from anthropic import Anthropic

logger = logging.getLogger(__name__)

@dataclass
class MarketInsight:
    """Market insight generated by Digital Brain"""
    symbol: str
    insight_type: str
    confidence: float
    message: str
    supporting_evidence: List[str]
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class PatternMatch:
    """Pattern matching result"""
    pattern_id: str
    symbol: str
    confidence: float
    pattern_type: str
    conditions_met: List[str]
    expected_outcome: str
    historical_success_rate: float
    timestamp: datetime

class DigitalBrain:
    """Advanced AI-powered market intelligence system"""
    
    def __init__(self):
        self.knowledge_graph = KnowledgeGraph()
        self.market_patterns = {}
        self.learned_insights = {}
        self.conversation_history = []
        
        # AI clients
        self.anthropic_client = None
        self.openai_client = None
        
        # Learning parameters
        self.learning_rate = 0.01
        self.pattern_threshold = 0.75
        self.max_patterns = 10000
        
        # Performance tracking
        self.prediction_accuracy = {}
        self.pattern_success_rates = {}
        
        self.running = False
        self.learning_tasks = set()
        
        logger.info("DigitalBrain initialized")
    
    async def initialize(self):
        """Initialize Digital Brain components"""
        try:
            # Initialize AI clients
            await self.init_ai_clients()
            
            # Load existing knowledge
            await self.load_knowledge_base()
            
            # Start learning tasks
            await self.start_learning_tasks()
            
            self.running = True
            logger.info("DigitalBrain initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize DigitalBrain: {e}")
            raise
    
    async def init_ai_clients(self):
        """Initialize AI clients"""
        try:
            # Initialize Anthropic client
            import os
            anthropic_key = os.getenv('ANTHROPIC_API_KEY')
            if anthropic_key:
                self.anthropic_client = Anthropic(api_key=anthropic_key)
                logger.info("Anthropic client initialized")
            
            # Initialize OpenAI client
            openai_key = os.getenv('OPENAI_API_KEY')
            if openai_key:
                self.openai_client = openai.OpenAI(api_key=openai_key)
                logger.info("OpenAI client initialized")
                
        except Exception as e:
            logger.error(f"Error initializing AI clients: {e}")
    
    async def load_knowledge_base(self):
        """Load existing knowledge base"""
        try:
            # Load knowledge graph from file if exists
            try:
                with open('knowledge_graph_state.json', 'r') as f:
                    graph_data = json.load(f)
                    await self.restore_knowledge_graph(graph_data)
                    logger.info("Knowledge graph loaded from file")
            except FileNotFoundError:
                logger.info("No existing knowledge graph found, starting fresh")
            
            # Load market patterns
            await self.load_market_patterns()
            
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
    
    async def start_learning_tasks(self):
        """Start continuous learning tasks"""
        # Pattern learning task
        task = asyncio.create_task(self.continuous_pattern_learning())
        self.learning_tasks.add(task)
        task.add_done_callback(self.learning_tasks.discard)
        
        # Knowledge graph optimization task
        task = asyncio.create_task(self.optimize_knowledge_graph())
        self.learning_tasks.add(task)
        task.add_done_callback(self.learning_tasks.discard)
    
    async def analyze_patterns(self, symbol: str, market_data: pd.DataFrame, technical_indicators: Dict[str, float]) -> float:
        """Analyze market patterns using Digital Brain"""
        try:
            # Extract features from market data
            features = await self.extract_features(market_data, technical_indicators)
            
            # Find matching patterns
            pattern_matches = await self.find_pattern_matches(symbol, features)
            
            # Calculate overall pattern strength
            pattern_strength = 0.0
            if pattern_matches:
                pattern_strength = np.mean([match.confidence for match in pattern_matches])
            
            # Store pattern analysis
            await self.store_pattern_analysis(symbol, pattern_matches, features)
            
            return pattern_strength
            
        except Exception as e:
            logger.error(f"Error analyzing patterns for {symbol}: {e}")
            return 0.0
    
    async def extract_features(self, market_data: pd.DataFrame, technical_indicators: Dict[str, float]) -> Dict[str, float]:
        """Extract features for pattern analysis"""
        try:
            features = {}
            
            # Price features
            features["price_change_1d"] = (market_data["close"].iloc[-1] - market_data["close"].iloc[-2]) / market_data["close"].iloc[-2]
            features["price_change_5d"] = (market_data["close"].iloc[-1] - market_data["close"].iloc[-6]) / market_data["close"].iloc[-6]
            features["price_change_20d"] = (market_data["close"].iloc[-1] - market_data["close"].iloc[-21]) / market_data["close"].iloc[-21]
            
            # Volume features
            features["volume_ratio"] = market_data["volume"].iloc[-1] / market_data["volume"].rolling(20).mean().iloc[-1]
            features["volume_trend"] = market_data["volume"].rolling(5).mean().iloc[-1] / market_data["volume"].rolling(20).mean().iloc[-1]
            
            # Volatility features
            returns = market_data["close"].pct_change()
            features["volatility_5d"] = returns.rolling(5).std().iloc[-1]
            features["volatility_20d"] = returns.rolling(20).std().iloc[-1]
            
            # Technical indicator features
            features.update(technical_indicators)
            
            # Candlestick patterns
            features.update(await self.detect_candlestick_patterns(market_data))
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return {}
    
    async def detect_candlestick_patterns(self, data: pd.DataFrame) -> Dict[str, float]:
        """Detect candlestick patterns"""
        try:
            patterns = {}
            
            # Get last few candles
            if len(data) < 3:
                return patterns
            
            last_candles = data.tail(3)
            
            # Doji pattern
            for i, candle in last_candles.iterrows():
                body_size = abs(candle["close"] - candle["open"])
                candle_range = candle["high"] - candle["low"]
                if candle_range > 0 and body_size / candle_range < 0.1:
                    patterns["doji"] = 1.0
                    break
            
            # Hammer pattern
            last_candle = last_candles.iloc[-1]
            body_size = abs(last_candle["close"] - last_candle["open"])
            lower_shadow = min(last_candle["open"], last_candle["close"]) - last_candle["low"]
            upper_shadow = last_candle["high"] - max(last_candle["open"], last_candle["close"])
            
            if body_size > 0 and lower_shadow > 2 * body_size and upper_shadow < body_size:
                patterns["hammer"] = 1.0
            
            # Engulfing pattern
            if len(last_candles) >= 2:
                prev_candle = last_candles.iloc[-2]
                curr_candle = last_candles.iloc[-1]
                
                if (curr_candle["open"] < prev_candle["close"] and 
                    curr_candle["close"] > prev_candle["open"] and
                    curr_candle["close"] > curr_candle["open"]):
                    patterns["bullish_engulfing"] = 1.0
                elif (curr_candle["open"] > prev_candle["close"] and 
                      curr_candle["close"] < prev_candle["open"] and
                      curr_candle["close"] < curr_candle["open"]):
                    patterns["bearish_engulfing"] = 1.0
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting candlestick patterns: {e}")
            return {}
    
    async def find_pattern_matches(self, symbol: str, features: Dict[str, float]) -> List[PatternMatch]:
        """Find matching patterns in knowledge base"""
        try:
            matches = []
            
            # Query knowledge graph for similar patterns
            similar_patterns = await self.query_similar_patterns(features)
            
            for pattern_id, similarity in similar_patterns:
                if similarity > self.pattern_threshold:
                    pattern = self.market_patterns.get(pattern_id)
                    if pattern:
                        match = PatternMatch(
                            pattern_id=pattern_id,
                            symbol=symbol,
                            confidence=similarity,
                            pattern_type=pattern.pattern_type,
                            conditions_met=await self.get_conditions_met(pattern, features),
                            expected_outcome=await self.get_expected_outcome(pattern),
                            historical_success_rate=pattern.success_rate,
                            timestamp=datetime.now()
                        )
                        matches.append(match)
            
            return sorted(matches, key=lambda x: x.confidence, reverse=True)
            
        except Exception as e:
            logger.error(f"Error finding pattern matches: {e}")
            return []
    
    async def query_similar_patterns(self, features: Dict[str, float]) -> List[Tuple[str, float]]:
        """Query knowledge graph for similar patterns"""
        try:
            # Simplified similarity calculation
            # In a real implementation, this would use more sophisticated matching
            
            similar_patterns = []
            
            for pattern_id, pattern in self.market_patterns.items():
                similarity = await self.calculate_pattern_similarity(features, pattern)
                similar_patterns.append((pattern_id, similarity))
            
            return sorted(similar_patterns, key=lambda x: x[1], reverse=True)[:10]
            
        except Exception as e:
            logger.error(f"Error querying similar patterns: {e}")
            return []
    
    async def calculate_pattern_similarity(self, features: Dict[str, float], pattern: MarketPattern) -> float:
        """Calculate similarity between features and pattern"""
        try:
            # Simplified cosine similarity
            if not pattern.conditions:
                return 0.0
            
            common_keys = set(features.keys()) & set(pattern.conditions.keys())
            if not common_keys:
                return 0.0
            
            dot_product = sum(features[key] * pattern.conditions[key] for key in common_keys)
            norm_features = np.sqrt(sum(features[key]**2 for key in common_keys))
            norm_pattern = np.sqrt(sum(pattern.conditions[key]**2 for key in common_keys))
            
            if norm_features == 0 or norm_pattern == 0:
                return 0.0
            
            return dot_product / (norm_features * norm_pattern)
            
        except Exception as e:
            logger.error(f"Error calculating pattern similarity: {e}")
            return 0.0
    
    async def get_conditions_met(self, pattern: MarketPattern, features: Dict[str, float]) -> List[str]:
        """Get conditions met for pattern"""
        try:
            conditions_met = []
            
            for condition, threshold in pattern.conditions.items():
                if condition in features:
                    if features[condition] >= threshold:
                        conditions_met.append(f"{condition} >= {threshold}")
                    else:
                        conditions_met.append(f"{condition} < {threshold}")
            
            return conditions_met
            
        except Exception as e:
            logger.error(f"Error getting conditions met: {e}")
            return []
    
    async def get_expected_outcome(self, pattern: MarketPattern) -> str:
        """Get expected outcome for pattern"""
        try:
            if pattern.outcomes:
                return pattern.outcomes.get("direction", "neutral")
            return "neutral"
            
        except Exception as e:
            logger.error(f"Error getting expected outcome: {e}")
            return "neutral"
    
    async def enhance_signals(self, signals: List[Any], analysis: Any) -> List[Any]:
        """Enhance trading signals with Digital Brain insights"""
        try:
            enhanced_signals = []
            
            for signal in signals:
                # Get insights for signal
                insights = await self.get_symbol_insights(signal.symbol)
                
                # Adjust signal confidence based on insights
                confidence_adjustment = 0.0
                for insight in insights:
                    if insight.insight_type == "bullish" and signal.action == "buy":
                        confidence_adjustment += insight.confidence * 0.1
                    elif insight.insight_type == "bearish" and signal.action == "sell":
                        confidence_adjustment += insight.confidence * 0.1
                
                # Apply adjustment
                signal.confidence = min(1.0, signal.confidence + confidence_adjustment)
                
                # Add reasoning
                if insights:
                    signal.reasoning += f" Digital Brain insights: {', '.join([i.message for i in insights[:3]])}"
                
                enhanced_signals.append(signal)
            
            return enhanced_signals
            
        except Exception as e:
            logger.error(f"Error enhancing signals: {e}")
            return signals
    
    async def get_symbol_insights(self, symbol: str) -> List[MarketInsight]:
        """Get insights for a specific symbol"""
        try:
            insights = []
            
            # Check stored insights
            if symbol in self.learned_insights:
                insights.extend(self.learned_insights[symbol])
            
            # Generate new insights using AI
            if self.anthropic_client:
                ai_insights = await self.generate_ai_insights(symbol)
                insights.extend(ai_insights)
            
            # Filter recent insights
            cutoff_time = datetime.now() - timedelta(hours=24)
            recent_insights = [i for i in insights if i.timestamp > cutoff_time]
            
            return sorted(recent_insights, key=lambda x: x.confidence, reverse=True)[:5]
            
        except Exception as e:
            logger.error(f"Error getting symbol insights: {e}")
            return []
    
    async def generate_ai_insights(self, symbol: str) -> List[MarketInsight]:
        """Generate AI-powered insights"""
        try:
            insights = []
            
            # Create prompt for AI
            prompt = f"""
            Analyze the current market conditions for {symbol} and provide insights.
            Consider technical analysis, market sentiment, and recent price action.
            Provide 2-3 key insights with confidence levels.
            """
            
            # Get AI response
            response = await self.anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Parse response (simplified)
            ai_text = response.content[0].text
            
            # Create insight from AI response
            insight = MarketInsight(
                symbol=symbol,
                insight_type="analysis",
                confidence=0.7,
                message=ai_text[:200],
                supporting_evidence=["AI analysis"],
                timestamp=datetime.now(),
                metadata={"source": "anthropic"}
            )
            
            insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating AI insights: {e}")
            return []
    
    async def store_pattern_analysis(self, symbol: str, pattern_matches: List[PatternMatch], features: Dict[str, float]):
        """Store pattern analysis for learning"""
        try:
            # Create knowledge node for this analysis
            node = KnowledgeNode(
                node_id=f"analysis_{symbol}_{datetime.now().timestamp()}",
                node_type="pattern_analysis",
                attributes={
                    "symbol": symbol,
                    "features": features,
                    "pattern_matches": [match.__dict__ for match in pattern_matches],
                    "timestamp": datetime.now().isoformat()
                },
                timestamp=datetime.now()
            )
            
            # Add to knowledge graph
            self.knowledge_graph.add_node(node)
            
        except Exception as e:
            logger.error(f"Error storing pattern analysis: {e}")
    
    async def continuous_pattern_learning(self):
        """Continuous pattern learning task"""
        while self.running:
            try:
                # Learn from recent market data
                await self.learn_from_market_data()
                
                # Update pattern success rates
                await self.update_pattern_success_rates()
                
                # Prune old patterns
                await self.prune_old_patterns()
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in continuous pattern learning: {e}")
                await asyncio.sleep(60)
    
    async def learn_from_market_data(self):
        """Learn patterns from recent market data"""
        try:
            # This is a simplified learning process
            # In practice, this would analyze recent trades and outcomes
            
            # Get recent successful trades
            successful_patterns = await self.get_successful_patterns()
            
            # Update pattern success rates
            for pattern_id, success_rate in successful_patterns.items():
                if pattern_id in self.market_patterns:
                    pattern = self.market_patterns[pattern_id]
                    pattern.success_rate = (pattern.success_rate + success_rate) / 2
                    pattern.sample_size += 1
            
        except Exception as e:
            logger.error(f"Error learning from market data: {e}")
    
    async def get_successful_patterns(self) -> Dict[str, float]:
        """Get successful patterns from recent trades"""
        try:
            # This would query the database for recent successful trades
            # and identify which patterns were used
            
            # Placeholder implementation
            return {}
            
        except Exception as e:
            logger.error(f"Error getting successful patterns: {e}")
            return {}
    
    async def update_pattern_success_rates(self):
        """Update pattern success rates based on recent performance"""
        try:
            # This would analyze recent trade outcomes
            # and update pattern success rates accordingly
            
            pass
            
        except Exception as e:
            logger.error(f"Error updating pattern success rates: {e}")
    
    async def prune_old_patterns(self):
        """Remove old or poorly performing patterns"""
        try:
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(days=90)
            
            # Remove old patterns
            patterns_to_remove = []
            for pattern_id, pattern in self.market_patterns.items():
                if pattern.last_seen < cutoff_time or pattern.success_rate < 0.3:
                    patterns_to_remove.append(pattern_id)
            
            for pattern_id in patterns_to_remove:
                del self.market_patterns[pattern_id]
                logger.info(f"Removed old pattern: {pattern_id}")
            
        except Exception as e:
            logger.error(f"Error pruning old patterns: {e}")
    
    async def optimize_knowledge_graph(self):
        """Optimize knowledge graph performance"""
        while self.running:
            try:
                # Optimize graph structure
                await self.knowledge_graph.optimize_structure()
                
                # Save state
                await self.save_knowledge_state()
                
                await asyncio.sleep(1800)  # Run every 30 minutes
                
            except Exception as e:
                logger.error(f"Error optimizing knowledge graph: {e}")
                await asyncio.sleep(300)
    
    async def save_knowledge_state(self):
        """Save knowledge graph state"""
        try:
            # This would save the current state of the knowledge graph
            # to persistent storage
            
            pass
            
        except Exception as e:
            logger.error(f"Error saving knowledge state: {e}")
    
    async def restore_knowledge_graph(self, graph_data: Dict[str, Any]):
        """Restore knowledge graph from saved state"""
        try:
            # This would restore the knowledge graph from saved data
            
            pass
            
        except Exception as e:
            logger.error(f"Error restoring knowledge graph: {e}")
    
    async def load_market_patterns(self):
        """Load market patterns from database"""
        try:
            # This would load patterns from the database
            
            pass
            
        except Exception as e:
            logger.error(f"Error loading market patterns: {e}")
    
    async def process_natural_language_query(self, query: str) -> str:
        """Process natural language query"""
        try:
            if not self.anthropic_client:
                return "AI client not available"
            
            # Create context from knowledge graph
            context = await self.create_query_context(query)
            
            # Generate response
            response = await self.anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": f"Context: {context}\n\nQuery: {query}"}
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Error processing natural language query: {e}")
            return "Error processing query"
    
    async def create_query_context(self, query: str) -> str:
        """Create context for query processing"""
        try:
            # This would create relevant context from the knowledge graph
            # based on the query
            
            return "Market analysis context"
            
        except Exception as e:
            logger.error(f"Error creating query context: {e}")
            return ""
    
    def is_healthy(self) -> bool:
        """Check if Digital Brain is healthy"""
        return self.running and self.knowledge_graph is not None
    
    async def continuous_learning(self):
        """Continuous learning task"""
        while self.running:
            try:
                # This is the main learning loop
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in continuous learning: {e}")
                await asyncio.sleep(30)
    
    async def shutdown(self):
        """Shutdown Digital Brain"""
        logger.info("Shutting down DigitalBrain...")
        
        self.running = False
        
        # Cancel learning tasks
        for task in self.learning_tasks:
            task.cancel()
        
        # Save current state
        await self.save_knowledge_state()
        
        logger.info("DigitalBrain shutdown complete")
