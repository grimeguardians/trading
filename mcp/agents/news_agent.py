"""
News Agent for MCP System
Handles news analysis, event processing, and market sentiment from news sources
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import websockets
import uuid
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import feedparser
import requests
from bs4 import BeautifulSoup
import re

# AI imports
import os
from openai import OpenAI
import anthropic
from anthropic import Anthropic

# The newest OpenAI model is "gpt-4o", not "gpt-4"
# The newest Anthropic model is "claude-sonnet-4-20250514", not "claude-3-5-sonnet-20241022"

@dataclass
class NewsArticle:
    """News article data structure"""
    article_id: str
    title: str
    content: str
    source: str
    url: str
    published_at: datetime
    symbols: List[str]
    sentiment_score: float  # -1 to 1
    relevance_score: float  # 0 to 1
    impact_score: float  # 0 to 1
    categories: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MarketEvent:
    """Market event data structure"""
    event_id: str
    event_type: str
    title: str
    description: str
    timestamp: datetime
    impact_level: str  # "low", "medium", "high", "critical"
    affected_symbols: List[str]
    affected_sectors: List[str]
    expected_impact: Dict[str, float]
    source_articles: List[str]

class NewsAgent:
    """News Agent for comprehensive news analysis and market event processing"""
    
    def __init__(self, mcp_server_url: str = "ws://localhost:9000"):
        self.agent_id = "news_agent_001"
        self.agent_type = "news_agent"
        self.name = "News Agent"
        self.description = "Advanced news analysis and market event processing"
        self.capabilities = [
            "news_analysis",
            "event_processing",
            "sentiment_analysis",
            "impact_assessment",
            "real_time_monitoring",
            "earnings_analysis",
            "regulatory_monitoring"
        ]
        
        self.mcp_server_url = mcp_server_url
        self.websocket = None
        self.running = False
        
        # AI clients
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        # News sources and feeds
        self.news_sources = {
            "reuters": "https://feeds.reuters.com/reuters/businessNews",
            "bloomberg": "https://feeds.bloomberg.com/markets/news.rss",
            "cnbc": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
            "marketwatch": "http://feeds.marketwatch.com/marketwatch/marketpulse/",
            "yahoo_finance": "https://feeds.finance.yahoo.com/rss/2.0/headline",
            "seeking_alpha": "https://seekingalpha.com/api/sa/combined/A.xml",
            "finviz": "https://finviz.com/news.ashx"
        }
        
        # News data storage
        self.articles: Dict[str, NewsArticle] = {}
        self.events: Dict[str, MarketEvent] = {}
        self.sentiment_cache: Dict[str, Dict[str, float]] = {}
        
        # Processing parameters
        self.max_articles_per_source = 100
        self.sentiment_threshold = 0.3
        self.impact_threshold = 0.5
        self.cache_ttl = 3600  # 1 hour
        
        # Symbol tracking
        self.tracked_symbols = set()
        self.symbol_patterns = {}
        
        # Event detection
        self.event_keywords = {
            "earnings": ["earnings", "quarterly", "q1", "q2", "q3", "q4", "revenue", "profit"],
            "merger": ["merger", "acquisition", "takeover", "buyout", "deal"],
            "regulatory": ["fda", "sec", "regulation", "compliance", "investigation"],
            "leadership": ["ceo", "cfo", "chairman", "executive", "leadership"],
            "product": ["product", "launch", "recall", "innovation", "patent"],
            "financial": ["dividend", "split", "buyback", "debt", "financing"]
        }
        
        # Setup logging
        self.logger = logging.getLogger(f"NewsAgent_{self.agent_id}")
        
        # Performance metrics
        self.metrics = {
            "articles_processed": 0,
            "events_detected": 0,
            "sentiment_accuracy": 0.0,
            "processing_time": 0.0
        }
    
    async def start(self):
        """Start the news agent"""
        try:
            self.logger.info("Starting News Agent...")
            
            # Connect to MCP server
            self.websocket = await websockets.connect(self.mcp_server_url)
            
            # Initialize with server
            await self._send_initialize_message()
            
            self.running = True
            
            # Start background tasks
            asyncio.create_task(self._listen_for_messages())
            asyncio.create_task(self._monitor_news_feeds())
            asyncio.create_task(self._process_news_queue())
            asyncio.create_task(self._detect_market_events())
            asyncio.create_task(self._cleanup_old_data())
            asyncio.create_task(self._send_heartbeat())
            
            self.logger.info("News Agent started successfully")
            
        except Exception as e:
            self.logger.error(f"Error starting News Agent: {e}")
            raise
    
    async def stop(self):
        """Stop the news agent"""
        self.logger.info("Stopping News Agent...")
        
        self.running = False
        
        if self.websocket:
            await self.websocket.close()
        
        self.logger.info("News Agent stopped")
    
    async def _send_initialize_message(self):
        """Send initialization message to MCP server"""
        message = {
            "message_id": str(uuid.uuid4()),
            "message_type": "initialize",
            "sender": self.agent_id,
            "payload": {
                "agent_type": self.agent_type,
                "name": self.name,
                "description": self.description,
                "capabilities": self.capabilities
            }
        }
        
        await self.websocket.send(json.dumps(message))
    
    async def _listen_for_messages(self):
        """Listen for messages from MCP server"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                await self._handle_message(data)
        except websockets.exceptions.ConnectionClosed:
            self.logger.info("Connection to MCP server closed")
            self.running = False
        except Exception as e:
            self.logger.error(f"Error listening for messages: {e}")
    
    async def _handle_message(self, data: Dict[str, Any]):
        """Handle incoming messages"""
        message_type = data.get("message_type")
        
        if message_type == "ready":
            self.logger.info("Received ready message from server")
        elif message_type == "request":
            await self._handle_request(data)
        elif message_type == "notification":
            await self._handle_notification(data)
        elif message_type == "error":
            self.logger.error(f"Error from server: {data.get('payload', {}).get('error')}")
        else:
            self.logger.warning(f"Unknown message type: {message_type}")
    
    async def _handle_request(self, data: Dict[str, Any]):
        """Handle request messages"""
        request_type = data.get("payload", {}).get("request_type")
        
        if request_type == "news_analysis":
            await self._handle_news_analysis_request(data)
        elif request_type == "sentiment_analysis":
            await self._handle_sentiment_analysis_request(data)
        elif request_type == "event_detection":
            await self._handle_event_detection_request(data)
        elif request_type == "symbol_news":
            await self._handle_symbol_news_request(data)
        else:
            await self._send_error_response(data, f"Unknown request type: {request_type}")
    
    async def _handle_notification(self, data: Dict[str, Any]):
        """Handle notification messages"""
        notification_type = data.get("payload", {}).get("notification_type")
        
        if notification_type == "track_symbol":
            await self._handle_track_symbol(data)
        elif notification_type == "market_update":
            await self._handle_market_update(data)
    
    async def _monitor_news_feeds(self):
        """Monitor news feeds for new articles"""
        while self.running:
            try:
                for source, feed_url in self.news_sources.items():
                    try:
                        articles = await self._fetch_news_from_feed(source, feed_url)
                        
                        for article in articles:
                            if article.article_id not in self.articles:
                                self.articles[article.article_id] = article
                                
                                # Process article for sentiment and impact
                                await self._process_article(article)
                                
                                # Check if article mentions tracked symbols
                                if article.symbols:
                                    await self._send_symbol_news_notification(article)
                    
                    except Exception as e:
                        self.logger.error(f"Error fetching news from {source}: {e}")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error monitoring news feeds: {e}")
                await asyncio.sleep(300)
    
    async def _fetch_news_from_feed(self, source: str, feed_url: str) -> List[NewsArticle]:
        """Fetch news articles from RSS feed"""
        try:
            # Fetch RSS feed
            response = requests.get(feed_url, timeout=30)
            response.raise_for_status()
            
            # Parse RSS feed
            feed = feedparser.parse(response.content)
            articles = []
            
            for entry in feed.entries[:self.max_articles_per_source]:
                # Extract article content
                content = entry.get('summary', '') or entry.get('description', '')
                
                # Clean HTML content
                if content:
                    content = BeautifulSoup(content, 'html.parser').get_text()
                
                article = NewsArticle(
                    article_id=str(uuid.uuid4()),
                    title=entry.get('title', ''),
                    content=content,
                    source=source,
                    url=entry.get('link', ''),
                    published_at=datetime.now(),  # Parse entry.published
                    symbols=[],
                    sentiment_score=0.0,
                    relevance_score=0.0,
                    impact_score=0.0,
                    categories=[]
                )
                
                articles.append(article)
            
            return articles
            
        except Exception as e:
            self.logger.error(f"Error fetching news from {source}: {e}")
            return []
    
    async def _process_article(self, article: NewsArticle):
        """Process article for sentiment, symbols, and impact"""
        try:
            # Extract symbols from article
            article.symbols = self._extract_symbols(article.title + " " + article.content)
            
            # Analyze sentiment using AI
            sentiment_result = await self._analyze_article_sentiment(article)
            article.sentiment_score = sentiment_result["sentiment_score"]
            article.relevance_score = sentiment_result["relevance_score"]
            
            # Categorize article
            article.categories = self._categorize_article(article)
            
            # Calculate impact score
            article.impact_score = self._calculate_impact_score(article)
            
            # Update metrics
            self.metrics["articles_processed"] += 1
            
        except Exception as e:
            self.logger.error(f"Error processing article {article.article_id}: {e}")
    
    def _extract_symbols(self, text: str) -> List[str]:
        """Extract stock symbols from text"""
        symbols = []
        
        # Common patterns for stock symbols
        patterns = [
            r'\b[A-Z]{1,5}\b',  # Basic symbol pattern
            r'\$[A-Z]{1,5}',    # Symbol with $ prefix
            r'\([A-Z]{1,5}\)',  # Symbol in parentheses
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text.upper())
            symbols.extend(matches)
        
        # Filter out common words and validate symbols
        valid_symbols = []
        exclude_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'HAD', 'HAS', 'HIS', 'ITS', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WHO', 'BOY', 'DID', 'HAS', 'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE'}
        
        for symbol in symbols:
            symbol = symbol.replace('$', '').replace('(', '').replace(')', '')
            if len(symbol) >= 1 and len(symbol) <= 5 and symbol not in exclude_words:
                valid_symbols.append(symbol)
        
        return list(set(valid_symbols))
    
    async def _analyze_article_sentiment(self, article: NewsArticle) -> Dict[str, float]:
        """Analyze article sentiment using AI"""
        try:
            # Prepare text for analysis
            text = f"Title: {article.title}\n\nContent: {article.content[:2000]}"
            
            # Use OpenAI for sentiment analysis
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",  # The newest OpenAI model
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial news sentiment analyzer. Analyze the sentiment of the news article and provide scores between -1 and 1 for sentiment (-1 = very negative, 1 = very positive) and between 0 and 1 for market relevance (0 = not relevant, 1 = highly relevant). Respond in JSON format with 'sentiment_score' and 'relevance_score' fields."
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return {
                "sentiment_score": max(-1, min(1, result.get("sentiment_score", 0))),
                "relevance_score": max(0, min(1, result.get("relevance_score", 0)))
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment for article {article.article_id}: {e}")
            return {"sentiment_score": 0.0, "relevance_score": 0.0}
    
    def _categorize_article(self, article: NewsArticle) -> List[str]:
        """Categorize article based on content"""
        categories = []
        text = (article.title + " " + article.content).lower()
        
        for category, keywords in self.event_keywords.items():
            if any(keyword in text for keyword in keywords):
                categories.append(category)
        
        return categories
    
    def _calculate_impact_score(self, article: NewsArticle) -> float:
        """Calculate potential market impact score"""
        impact_score = 0.0
        
        # Base score on sentiment strength
        impact_score += abs(article.sentiment_score) * 0.3
        
        # Relevance contributes to impact
        impact_score += article.relevance_score * 0.3
        
        # Number of symbols mentioned
        impact_score += min(len(article.symbols) * 0.1, 0.2)
        
        # Source credibility
        source_weights = {
            "reuters": 0.2,
            "bloomberg": 0.2,
            "cnbc": 0.15,
            "marketwatch": 0.1,
            "yahoo_finance": 0.05
        }
        impact_score += source_weights.get(article.source, 0.05)
        
        return min(impact_score, 1.0)
    
    async def _detect_market_events(self):
        """Detect significant market events from news articles"""
        while self.running:
            try:
                # Analyze recent articles for events
                recent_articles = [
                    article for article in self.articles.values()
                    if (datetime.utcnow() - article.published_at).total_seconds() < 3600
                ]
                
                # Group articles by symbols and categories
                symbol_groups = {}
                for article in recent_articles:
                    for symbol in article.symbols:
                        if symbol not in symbol_groups:
                            symbol_groups[symbol] = []
                        symbol_groups[symbol].append(article)
                
                # Detect events for each symbol
                for symbol, articles in symbol_groups.items():
                    await self._detect_symbol_events(symbol, articles)
                
                await asyncio.sleep(900)  # Check every 15 minutes
                
            except Exception as e:
                self.logger.error(f"Error detecting market events: {e}")
                await asyncio.sleep(900)
    
    async def _detect_symbol_events(self, symbol: str, articles: List[NewsArticle]):
        """Detect events for a specific symbol"""
        try:
            # Calculate aggregate metrics
            total_sentiment = sum(article.sentiment_score for article in articles)
            avg_sentiment = total_sentiment / len(articles) if articles else 0
            total_impact = sum(article.impact_score for article in articles)
            
            # Check for significant events
            if abs(avg_sentiment) > self.sentiment_threshold or total_impact > self.impact_threshold:
                # Determine event type
                event_type = self._determine_event_type(articles)
                
                # Create market event
                event = MarketEvent(
                    event_id=str(uuid.uuid4()),
                    event_type=event_type,
                    title=f"{symbol} {event_type.title()} Event",
                    description=f"Significant {event_type} detected for {symbol}",
                    timestamp=datetime.utcnow(),
                    impact_level=self._calculate_impact_level(total_impact),
                    affected_symbols=[symbol],
                    affected_sectors=[],
                    expected_impact={"price": avg_sentiment, "volume": total_impact},
                    source_articles=[article.article_id for article in articles]
                )
                
                # Store event
                self.events[event.event_id] = event
                
                # Send notification
                await self._send_event_notification(event)
                
                # Update metrics
                self.metrics["events_detected"] += 1
                
        except Exception as e:
            self.logger.error(f"Error detecting events for {symbol}: {e}")
    
    def _determine_event_type(self, articles: List[NewsArticle]) -> str:
        """Determine the type of event based on articles"""
        category_counts = {}
        
        for article in articles:
            for category in article.categories:
                category_counts[category] = category_counts.get(category, 0) + 1
        
        if category_counts:
            return max(category_counts, key=category_counts.get)
        
        return "general"
    
    def _calculate_impact_level(self, impact_score: float) -> str:
        """Calculate impact level based on score"""
        if impact_score >= 0.8:
            return "critical"
        elif impact_score >= 0.6:
            return "high"
        elif impact_score >= 0.3:
            return "medium"
        else:
            return "low"
    
    async def _send_event_notification(self, event: MarketEvent):
        """Send event notification to MCP server"""
        notification = {
            "message_id": str(uuid.uuid4()),
            "message_type": "notification",
            "sender": self.agent_id,
            "payload": {
                "notification_type": "market_event",
                "event_id": event.event_id,
                "event_type": event.event_type,
                "title": event.title,
                "impact_level": event.impact_level,
                "affected_symbols": event.affected_symbols,
                "expected_impact": event.expected_impact,
                "timestamp": event.timestamp.isoformat()
            }
        }
        
        await self.websocket.send(json.dumps(notification))
    
    async def _send_symbol_news_notification(self, article: NewsArticle):
        """Send symbol news notification"""
        notification = {
            "message_id": str(uuid.uuid4()),
            "message_type": "notification",
            "sender": self.agent_id,
            "payload": {
                "notification_type": "symbol_news",
                "article_id": article.article_id,
                "title": article.title,
                "source": article.source,
                "symbols": article.symbols,
                "sentiment_score": article.sentiment_score,
                "impact_score": article.impact_score,
                "timestamp": article.published_at.isoformat()
            }
        }
        
        await self.websocket.send(json.dumps(notification))
    
    async def _handle_news_analysis_request(self, data: Dict[str, Any]):
        """Handle news analysis request"""
        payload = data.get("payload", {})
        symbol = payload.get("symbol")
        timeframe = payload.get("timeframe", "1h")
        
        try:
            # Get relevant articles
            articles = await self._get_symbol_articles(symbol, timeframe)
            
            # Analyze articles
            analysis = await self._analyze_symbol_news(symbol, articles)
            
            # Send response
            response = {
                "message_id": str(uuid.uuid4()),
                "message_type": "response",
                "sender": self.agent_id,
                "recipient": data["sender"],
                "payload": {
                    "request_type": "news_analysis",
                    "symbol": symbol,
                    "analysis": analysis,
                    "article_count": len(articles)
                },
                "correlation_id": data["message_id"]
            }
            
            await self.websocket.send(json.dumps(response))
            
        except Exception as e:
            await self._send_error_response(data, str(e))
    
    async def _get_symbol_articles(self, symbol: str, timeframe: str) -> List[NewsArticle]:
        """Get articles mentioning a specific symbol"""
        # Convert timeframe to hours
        timeframe_hours = {"1h": 1, "4h": 4, "1d": 24, "1w": 168}
        hours = timeframe_hours.get(timeframe, 1)
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        relevant_articles = []
        for article in self.articles.values():
            if (symbol in article.symbols and 
                article.published_at >= cutoff_time):
                relevant_articles.append(article)
        
        return sorted(relevant_articles, key=lambda x: x.published_at, reverse=True)
    
    async def _analyze_symbol_news(self, symbol: str, articles: List[NewsArticle]) -> Dict[str, Any]:
        """Analyze news for a specific symbol"""
        if not articles:
            return {
                "overall_sentiment": 0.0,
                "sentiment_trend": "neutral",
                "key_events": [],
                "confidence": 0.0
            }
        
        # Calculate overall sentiment
        overall_sentiment = sum(article.sentiment_score for article in articles) / len(articles)
        
        # Determine sentiment trend
        if overall_sentiment > 0.2:
            sentiment_trend = "positive"
        elif overall_sentiment < -0.2:
            sentiment_trend = "negative"
        else:
            sentiment_trend = "neutral"
        
        # Identify key events
        key_events = []
        for article in articles[:5]:  # Top 5 articles
            if article.impact_score > 0.3:
                key_events.append({
                    "title": article.title,
                    "source": article.source,
                    "sentiment": article.sentiment_score,
                    "impact": article.impact_score,
                    "published_at": article.published_at.isoformat()
                })
        
        # Calculate confidence
        confidence = min(len(articles) / 10, 1.0)  # More articles = higher confidence
        
        return {
            "overall_sentiment": overall_sentiment,
            "sentiment_trend": sentiment_trend,
            "key_events": key_events,
            "confidence": confidence,
            "article_count": len(articles)
        }
    
    async def _cleanup_old_data(self):
        """Clean up old articles and events"""
        while self.running:
            try:
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                
                # Remove old articles
                old_articles = [
                    article_id for article_id, article in self.articles.items()
                    if article.published_at < cutoff_time
                ]
                
                for article_id in old_articles:
                    del self.articles[article_id]
                
                # Remove old events
                old_events = [
                    event_id for event_id, event in self.events.items()
                    if event.timestamp < cutoff_time
                ]
                
                for event_id in old_events:
                    del self.events[event_id]
                
                await asyncio.sleep(3600)  # Clean every hour
                
            except Exception as e:
                self.logger.error(f"Error cleaning old data: {e}")
                await asyncio.sleep(3600)
    
    async def _process_news_queue(self):
        """Process news analysis queue"""
        while self.running:
            try:
                # Process high-impact articles
                high_impact_articles = [
                    article for article in self.articles.values()
                    if article.impact_score > 0.7 and not article.metadata.get("processed", False)
                ]
                
                for article in high_impact_articles:
                    # Perform deeper analysis
                    await self._deep_analyze_article(article)
                    article.metadata["processed"] = True
                
                await asyncio.sleep(60)  # Process every minute
                
            except Exception as e:
                self.logger.error(f"Error processing news queue: {e}")
                await asyncio.sleep(60)
    
    async def _deep_analyze_article(self, article: NewsArticle):
        """Perform deep analysis on high-impact articles"""
        try:
            # Use Anthropic for detailed analysis
            response = self.anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",  # The newest Anthropic model
                max_tokens=2000,
                temperature=0.3,
                messages=[
                    {
                        "role": "user",
                        "content": f"""
                        Analyze this financial news article in detail:
                        
                        Title: {article.title}
                        Content: {article.content[:3000]}
                        
                        Provide:
                        1. Key insights and implications
                        2. Potential market impact
                        3. Affected companies/sectors
                        4. Timeline of impact
                        5. Confidence level (0-1)
                        
                        Focus on actionable trading insights.
                        """
                    }
                ]
            )
            
            analysis = response.content[0].text
            article.metadata["deep_analysis"] = analysis
            
            # Send detailed analysis notification
            notification = {
                "message_id": str(uuid.uuid4()),
                "message_type": "notification",
                "sender": self.agent_id,
                "payload": {
                    "notification_type": "deep_analysis",
                    "article_id": article.article_id,
                    "title": article.title,
                    "analysis": analysis,
                    "symbols": article.symbols,
                    "impact_score": article.impact_score
                }
            }
            
            await self.websocket.send(json.dumps(notification))
            
        except Exception as e:
            self.logger.error(f"Error performing deep analysis: {e}")
    
    async def _send_error_response(self, original_message: Dict[str, Any], error: str):
        """Send error response"""
        response = {
            "message_id": str(uuid.uuid4()),
            "message_type": "error",
            "sender": self.agent_id,
            "recipient": original_message["sender"],
            "payload": {"error": error},
            "correlation_id": original_message["message_id"]
        }
        
        await self.websocket.send(json.dumps(response))
    
    async def _send_heartbeat(self):
        """Send heartbeat to MCP server"""
        while self.running:
            try:
                heartbeat = {
                    "message_id": str(uuid.uuid4()),
                    "message_type": "heartbeat",
                    "sender": self.agent_id,
                    "payload": {
                        "status": "online",
                        "metrics": self.metrics,
                        "articles_count": len(self.articles),
                        "events_count": len(self.events)
                    }
                }
                
                await self.websocket.send(json.dumps(heartbeat))
                await asyncio.sleep(30)  # Send every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error sending heartbeat: {e}")
                await asyncio.sleep(30)
    
    async def _handle_track_symbol(self, data: Dict[str, Any]):
        """Handle track symbol notification"""
        payload = data.get("payload", {})
        symbol = payload.get("symbol")
        
        if symbol:
            self.tracked_symbols.add(symbol)
            self.logger.info(f"Now tracking news for symbol: {symbol}")
    
    async def _handle_market_update(self, data: Dict[str, Any]):
        """Handle market update notification"""
        # Process market updates that might affect news analysis
        pass

if __name__ == "__main__":
    # Run the agent
    agent = NewsAgent()
    
    async def main():
        await agent.start()
        
        # Keep running
        while agent.running:
            await asyncio.sleep(1)
    
    asyncio.run(main())
