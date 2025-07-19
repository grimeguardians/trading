"""
News Agent - Monitors news and market sentiment
Provides real-time news analysis and sentiment scoring
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import aiohttp
import json
from textblob import TextBlob
import feedparser

from agents.base_agent import BaseAgent
from mcp_server import MessageType


class NewsAgent(BaseAgent):
    """
    News Agent for monitoring financial news and sentiment
    Provides real-time news analysis and market impact assessment
    """
    
    def __init__(self, mcp_server, knowledge_engine, config):
        super().__init__(
            agent_id="news_agent",
            agent_type="news_analyst",
            mcp_server=mcp_server,
            knowledge_engine=knowledge_engine,
            config=config
        )
        
        # News sources configuration
        self.news_sources = {
            "financial_news": [
                "https://feeds.reuters.com/reuters/businessNews",
                "https://feeds.bloomberg.com/markets/news.rss",
                "https://rss.cnn.com/rss/money_latest.rss",
                "https://feeds.marketwatch.com/marketwatch/topstories/",
                "https://feeds.finance.yahoo.com/rss/2.0/headline"
            ],
            "crypto_news": [
                "https://cointelegraph.com/rss",
                "https://feeds.coindesk.com/coindesk/rss/news",
                "https://feeds.cryptocoinsnews.com/cryptocoinsnews/all"
            ],
            "economic_indicators": [
                "https://www.federalreserve.gov/feeds/press_all.xml",
                "https://www.bls.gov/news.release/rss/all.xml"
            ]
        }
        
        # Sentiment analysis
        self.sentiment_keywords = {
            "positive": ["bullish", "rally", "surge", "growth", "profit", "gains", "strong", "positive", "upgrade"],
            "negative": ["bearish", "crash", "decline", "loss", "weak", "negative", "downgrade", "recession", "risk"],
            "neutral": ["stable", "unchanged", "steady", "consistent", "maintain", "hold"]
        }
        
        # Market impact tracking
        self.market_impact_scores = {}
        self.news_history = []
        self.sentiment_history = []
        
        # News processing settings
        self.fetch_interval = 300  # 5 minutes
        self.max_news_age = 3600  # 1 hour
        self.sentiment_threshold = 0.1  # Minimum sentiment score to report
        
        # Performance tracking
        self.news_processed = 0
        self.alerts_sent = 0
        self.sentiment_analyses = 0
        
        # Monitored symbols
        self.monitored_symbols = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "BTC", "ETH"]
        
        self.logger.info("📰 News Agent initialized with sentiment analysis")
    
    def _setup_capabilities(self):
        """Setup news agent capabilities"""
        self.capabilities = [
            "news_monitoring",
            "sentiment_analysis",
            "market_impact_assessment",
            "real_time_alerts",
            "economic_indicators",
            "social_media_monitoring",
            "news_categorization",
            "trend_analysis"
        ]
    
    def _setup_message_handlers(self):
        """Setup message handlers"""
        self.register_message_handler("news_request", self._handle_news_request)
        self.register_message_handler("sentiment_analysis_request", self._handle_sentiment_analysis_request)
        self.register_message_handler("market_impact_request", self._handle_market_impact_request)
        self.register_message_handler("news_alert_config", self._handle_news_alert_config)
    
    async def _agent_logic(self):
        """Main news agent logic"""
        self.logger.info("📰 News Agent started - monitoring financial news")
        
        # Start background tasks
        asyncio.create_task(self._news_fetcher())
        asyncio.create_task(self._sentiment_analyzer())
        asyncio.create_task(self._market_impact_analyzer())
        asyncio.create_task(self._news_alerter())
        
        while self.running:
            try:
                # Process news queue
                await self._process_news_queue()
                
                # Update sentiment trends
                await self._update_sentiment_trends()
                
                # Check for market-moving news
                await self._check_market_moving_news()
                
                # Clean old news data
                await self._cleanup_old_news()
                
                # Wait for next cycle
                await asyncio.sleep(30)  # 30 second cycle
                
            except Exception as e:
                self.logger.error(f"❌ News agent error: {e}")
                await asyncio.sleep(60)
    
    async def _news_fetcher(self):
        """Fetch news from various sources"""
        while self.running:
            try:
                # Fetch from RSS feeds
                await self._fetch_rss_news()
                
                # Fetch from APIs (if available)
                await self._fetch_api_news()
                
                # Wait for next fetch cycle
                await asyncio.sleep(self.fetch_interval)
                
            except Exception as e:
                self.logger.error(f"❌ News fetching error: {e}")
                await asyncio.sleep(self.fetch_interval)
    
    async def _fetch_rss_news(self):
        """Fetch news from RSS feeds"""
        try:
            for category, feeds in self.news_sources.items():
                for feed_url in feeds:
                    try:
                        # Parse RSS feed
                        feed = feedparser.parse(feed_url)
                        
                        for entry in feed.entries:
                            news_item = {
                                "id": entry.get("id", entry.get("link", "")),
                                "title": entry.get("title", ""),
                                "summary": entry.get("summary", ""),
                                "content": entry.get("content", [{}])[0].get("value", ""),
                                "link": entry.get("link", ""),
                                "published": entry.get("published", ""),
                                "source": feed.feed.get("title", "Unknown"),
                                "category": category,
                                "timestamp": datetime.utcnow().isoformat()
                            }
                            
                            # Check if news is recent
                            if self._is_recent_news(news_item):
                                await self._process_news_item(news_item)
                                
                    except Exception as e:
                        self.logger.error(f"❌ RSS feed error for {feed_url}: {e}")
                        
        except Exception as e:
            self.logger.error(f"❌ RSS news fetching error: {e}")
    
    async def _fetch_api_news(self):
        """Fetch news from APIs"""
        try:
            # This would integrate with news APIs like NewsAPI, Alpha Vantage, etc.
            # For now, using a placeholder implementation
            pass
            
        except Exception as e:
            self.logger.error(f"❌ API news fetching error: {e}")
    
    def _is_recent_news(self, news_item: Dict) -> bool:
        """Check if news item is recent enough to process"""
        try:
            # Check if we already processed this news
            if news_item["id"] in [n["id"] for n in self.news_history]:
                return False
            
            # Check news age
            published_str = news_item.get("published", "")
            if published_str:
                # Parse published date and check age
                # This is a simplified check - would need proper date parsing
                return True
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ News age check error: {e}")
            return False
    
    async def _process_news_item(self, news_item: Dict):
        """Process individual news item"""
        try:
            # Analyze sentiment
            sentiment_score = await self._analyze_sentiment(news_item)
            
            # Extract mentioned symbols
            mentioned_symbols = self._extract_symbols(news_item)
            
            # Calculate market impact
            market_impact = await self._calculate_market_impact(news_item, sentiment_score)
            
            # Enhanced news item
            enhanced_news = {
                **news_item,
                "sentiment_score": sentiment_score,
                "mentioned_symbols": mentioned_symbols,
                "market_impact": market_impact,
                "processed_at": datetime.utcnow().isoformat()
            }
            
            # Store in history
            self.news_history.append(enhanced_news)
            
            # Store in knowledge engine
            await self.update_knowledge("add_node", {
                "node_type": "news_item",
                "news_data": enhanced_news
            })
            
            # Send news update
            await self.broadcast_message({
                "type": "news_update",
                "news_item": enhanced_news
            })
            
            # Check if this is significant news
            if abs(sentiment_score) > self.sentiment_threshold or market_impact > 0.5:
                await self._send_news_alert(enhanced_news)
            
            self.news_processed += 1
            
        except Exception as e:
            self.logger.error(f"❌ News processing error: {e}")
    
    async def _analyze_sentiment(self, news_item: Dict) -> float:
        """Analyze sentiment of news item"""
        try:
            # Combine title and summary for analysis
            text = f"{news_item.get('title', '')} {news_item.get('summary', '')}"
            
            if not text.strip():
                return 0.0
            
            # Use TextBlob for basic sentiment analysis
            blob = TextBlob(text)
            sentiment_score = blob.sentiment.polarity
            
            # Enhance with keyword-based analysis
            keyword_score = self._calculate_keyword_sentiment(text)
            
            # Combine scores
            final_score = (sentiment_score + keyword_score) / 2
            
            self.sentiment_analyses += 1
            
            return max(-1.0, min(1.0, final_score))
            
        except Exception as e:
            self.logger.error(f"❌ Sentiment analysis error: {e}")
            return 0.0
    
    def _calculate_keyword_sentiment(self, text: str) -> float:
        """Calculate sentiment based on financial keywords"""
        try:
            text_lower = text.lower()
            positive_count = sum(1 for word in self.sentiment_keywords["positive"] if word in text_lower)
            negative_count = sum(1 for word in self.sentiment_keywords["negative"] if word in text_lower)
            
            total_sentiment_words = positive_count + negative_count
            
            if total_sentiment_words == 0:
                return 0.0
            
            return (positive_count - negative_count) / total_sentiment_words
            
        except Exception as e:
            self.logger.error(f"❌ Keyword sentiment calculation error: {e}")
            return 0.0
    
    def _extract_symbols(self, news_item: Dict) -> List[str]:
        """Extract mentioned stock symbols from news"""
        try:
            text = f"{news_item.get('title', '')} {news_item.get('summary', '')}"
            mentioned_symbols = []
            
            # Check for monitored symbols
            for symbol in self.monitored_symbols:
                if symbol in text.upper():
                    mentioned_symbols.append(symbol)
            
            # Check for company names (simplified)
            company_symbols = {
                "Apple": "AAPL",
                "Tesla": "TSLA",
                "Microsoft": "MSFT",
                "Google": "GOOGL",
                "Amazon": "AMZN",
                "NVIDIA": "NVDA",
                "Bitcoin": "BTC",
                "Ethereum": "ETH"
            }
            
            for company, symbol in company_symbols.items():
                if company.lower() in text.lower():
                    if symbol not in mentioned_symbols:
                        mentioned_symbols.append(symbol)
            
            return mentioned_symbols
            
        except Exception as e:
            self.logger.error(f"❌ Symbol extraction error: {e}")
            return []
    
    async def _calculate_market_impact(self, news_item: Dict, sentiment_score: float) -> float:
        """Calculate potential market impact of news"""
        try:
            impact_score = 0.0
            
            # Base impact from sentiment
            impact_score += abs(sentiment_score) * 0.3
            
            # Source credibility weight
            source = news_item.get("source", "").lower()
            if "reuters" in source or "bloomberg" in source:
                impact_score += 0.3
            elif "yahoo" in source or "marketwatch" in source:
                impact_score += 0.2
            else:
                impact_score += 0.1
            
            # Category weight
            category = news_item.get("category", "")
            if category == "economic_indicators":
                impact_score += 0.4
            elif category == "financial_news":
                impact_score += 0.3
            elif category == "crypto_news":
                impact_score += 0.2
            
            # Title keywords that suggest high impact
            title = news_item.get("title", "").lower()
            high_impact_keywords = ["fed", "earnings", "merger", "acquisition", "bankruptcy", "ipo", "split"]
            
            for keyword in high_impact_keywords:
                if keyword in title:
                    impact_score += 0.2
                    break
            
            return min(1.0, impact_score)
            
        except Exception as e:
            self.logger.error(f"❌ Market impact calculation error: {e}")
            return 0.0
    
    async def _sentiment_analyzer(self):
        """Analyze overall market sentiment"""
        while self.running:
            try:
                # Calculate overall sentiment
                overall_sentiment = await self._calculate_overall_sentiment()
                
                # Calculate symbol-specific sentiment
                symbol_sentiments = await self._calculate_symbol_sentiments()
                
                # Store sentiment data
                sentiment_data = {
                    "overall_sentiment": overall_sentiment,
                    "symbol_sentiments": symbol_sentiments,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                self.sentiment_history.append(sentiment_data)
                
                # Keep history size manageable
                if len(self.sentiment_history) > 100:
                    self.sentiment_history = self.sentiment_history[-50:]
                
                # Update knowledge engine
                await self.update_knowledge("add_node", {
                    "node_type": "sentiment_analysis",
                    "sentiment_data": sentiment_data
                })
                
                # Broadcast sentiment update
                await self.broadcast_message({
                    "type": "sentiment_update",
                    "sentiment_data": sentiment_data
                })
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                self.logger.error(f"❌ Sentiment analysis error: {e}")
                await asyncio.sleep(300)
    
    async def _calculate_overall_sentiment(self) -> float:
        """Calculate overall market sentiment"""
        try:
            # Get recent news (last hour)
            recent_news = [
                news for news in self.news_history
                if (datetime.utcnow() - datetime.fromisoformat(news["timestamp"])).total_seconds() < 3600
            ]
            
            if not recent_news:
                return 0.0
            
            # Calculate weighted average sentiment
            total_sentiment = 0.0
            total_weight = 0.0
            
            for news in recent_news:
                sentiment = news.get("sentiment_score", 0.0)
                weight = news.get("market_impact", 0.1)
                
                total_sentiment += sentiment * weight
                total_weight += weight
            
            return total_sentiment / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"❌ Overall sentiment calculation error: {e}")
            return 0.0
    
    async def _calculate_symbol_sentiments(self) -> Dict[str, float]:
        """Calculate sentiment for individual symbols"""
        try:
            symbol_sentiments = {}
            
            for symbol in self.monitored_symbols:
                # Get news mentioning this symbol
                symbol_news = [
                    news for news in self.news_history
                    if symbol in news.get("mentioned_symbols", [])
                    and (datetime.utcnow() - datetime.fromisoformat(news["timestamp"])).total_seconds() < 3600
                ]
                
                if symbol_news:
                    # Calculate average sentiment
                    avg_sentiment = sum(news.get("sentiment_score", 0.0) for news in symbol_news) / len(symbol_news)
                    symbol_sentiments[symbol] = avg_sentiment
                else:
                    symbol_sentiments[symbol] = 0.0
            
            return symbol_sentiments
            
        except Exception as e:
            self.logger.error(f"❌ Symbol sentiment calculation error: {e}")
            return {}
    
    async def _market_impact_analyzer(self):
        """Analyze market impact of news"""
        while self.running:
            try:
                # Analyze recent high-impact news
                high_impact_news = [
                    news for news in self.news_history
                    if news.get("market_impact", 0.0) > 0.7
                    and (datetime.utcnow() - datetime.fromisoformat(news["timestamp"])).total_seconds() < 1800  # 30 minutes
                ]
                
                if high_impact_news:
                    # Send market impact alert
                    await self.broadcast_message({
                        "type": "market_impact_alert",
                        "high_impact_news": high_impact_news
                    })
                
                await asyncio.sleep(600)  # Check every 10 minutes
                
            except Exception as e:
                self.logger.error(f"❌ Market impact analysis error: {e}")
                await asyncio.sleep(600)
    
    async def _news_alerter(self):
        """Send news alerts for significant events"""
        while self.running:
            try:
                # Check for breaking news
                breaking_news = await self._identify_breaking_news()
                
                if breaking_news:
                    for news in breaking_news:
                        await self._send_news_alert(news)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"❌ News alerter error: {e}")
                await asyncio.sleep(60)
    
    async def _identify_breaking_news(self) -> List[Dict]:
        """Identify breaking news that requires immediate attention"""
        try:
            breaking_news = []
            
            # Get very recent news (last 10 minutes)
            recent_threshold = datetime.utcnow() - timedelta(minutes=10)
            
            for news in self.news_history:
                news_time = datetime.fromisoformat(news["timestamp"])
                
                if news_time > recent_threshold:
                    # Check if it's breaking news
                    if (abs(news.get("sentiment_score", 0.0)) > 0.6 or
                        news.get("market_impact", 0.0) > 0.8 or
                        any(keyword in news.get("title", "").lower() 
                            for keyword in ["breaking", "urgent", "alert", "crash", "surge"])):
                        breaking_news.append(news)
            
            return breaking_news
            
        except Exception as e:
            self.logger.error(f"❌ Breaking news identification error: {e}")
            return []
    
    async def _send_news_alert(self, news_item: Dict):
        """Send news alert to other agents"""
        try:
            alert = {
                "type": "news_alert",
                "severity": "high" if news_item.get("market_impact", 0.0) > 0.7 else "medium",
                "news_item": news_item,
                "recommended_action": self._get_recommended_action(news_item)
            }
            
            # Send to all agents
            await self.broadcast_message(alert)
            
            # Send system event
            await self.send_system_event(
                "news_alert",
                alert["severity"],
                f"News alert: {news_item.get('title', 'Unknown')}",
                {"news_item": news_item}
            )
            
            self.alerts_sent += 1
            
        except Exception as e:
            self.logger.error(f"❌ News alert sending error: {e}")
    
    def _get_recommended_action(self, news_item: Dict) -> str:
        """Get recommended action based on news"""
        try:
            sentiment = news_item.get("sentiment_score", 0.0)
            impact = news_item.get("market_impact", 0.0)
            
            if sentiment > 0.5 and impact > 0.6:
                return "consider_buy"
            elif sentiment < -0.5 and impact > 0.6:
                return "consider_sell"
            elif impact > 0.8:
                return "review_positions"
            else:
                return "monitor"
                
        except Exception as e:
            self.logger.error(f"❌ Recommended action error: {e}")
            return "monitor"
    
    async def _process_news_queue(self):
        """Process queued news items"""
        # This would handle any queued news processing
        # For now, just a placeholder
        pass
    
    async def _update_sentiment_trends(self):
        """Update sentiment trend analysis"""
        try:
            if len(self.sentiment_history) < 2:
                return
            
            # Calculate sentiment trend
            recent_sentiment = self.sentiment_history[-1]["overall_sentiment"]
            previous_sentiment = self.sentiment_history[-2]["overall_sentiment"]
            
            trend = "bullish" if recent_sentiment > previous_sentiment else "bearish"
            trend_strength = abs(recent_sentiment - previous_sentiment)
            
            # Update knowledge engine
            await self.update_knowledge("add_node", {
                "node_type": "sentiment_trend",
                "trend": trend,
                "trend_strength": trend_strength,
                "timestamp": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"❌ Sentiment trend update error: {e}")
    
    async def _check_market_moving_news(self):
        """Check for market-moving news"""
        try:
            # Get recent high-impact news
            high_impact_threshold = 0.8
            recent_news = [
                news for news in self.news_history
                if news.get("market_impact", 0.0) > high_impact_threshold
                and (datetime.utcnow() - datetime.fromisoformat(news["timestamp"])).total_seconds() < 600  # 10 minutes
            ]
            
            if recent_news:
                # Send market moving news alert
                await self.broadcast_message({
                    "type": "market_moving_news",
                    "news_items": recent_news
                })
                
        except Exception as e:
            self.logger.error(f"❌ Market moving news check error: {e}")
    
    async def _cleanup_old_news(self):
        """Clean up old news data"""
        try:
            # Remove news older than 24 hours
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            
            self.news_history = [
                news for news in self.news_history
                if datetime.fromisoformat(news["timestamp"]) > cutoff_time
            ]
            
            # Remove old sentiment data
            self.sentiment_history = [
                sentiment for sentiment in self.sentiment_history
                if datetime.fromisoformat(sentiment["timestamp"]) > cutoff_time
            ]
            
        except Exception as e:
            self.logger.error(f"❌ News cleanup error: {e}")
    
    # Message handlers
    
    async def _handle_news_request(self, data: Dict):
        """Handle news request from other agents"""
        try:
            request_type = data.get("request_type", "recent")
            symbol = data.get("symbol", None)
            
            if request_type == "recent":
                # Get recent news
                news_items = self.news_history[-10:]  # Last 10 news items
            elif request_type == "symbol" and symbol:
                # Get news for specific symbol
                news_items = [
                    news for news in self.news_history
                    if symbol in news.get("mentioned_symbols", [])
                ][-5:]  # Last 5 news items for symbol
            else:
                news_items = []
            
            # Send response
            await self.send_direct_message(data["source"], {
                "type": "news_response",
                "news_items": news_items,
                "request_type": request_type
            })
            
        except Exception as e:
            self.logger.error(f"❌ News request handling error: {e}")
    
    async def _handle_sentiment_analysis_request(self, data: Dict):
        """Handle sentiment analysis request"""
        try:
            text = data.get("text", "")
            
            if text:
                # Analyze sentiment
                sentiment_score = await self._analyze_sentiment({"title": text, "summary": ""})
                
                # Send response
                await self.send_direct_message(data["source"], {
                    "type": "sentiment_analysis_response",
                    "text": text,
                    "sentiment_score": sentiment_score
                })
                
        except Exception as e:
            self.logger.error(f"❌ Sentiment analysis request handling error: {e}")
    
    async def _handle_market_impact_request(self, data: Dict):
        """Handle market impact request"""
        try:
            # Get current market impact summary
            impact_summary = {
                "high_impact_news_count": len([
                    news for news in self.news_history
                    if news.get("market_impact", 0.0) > 0.7
                    and (datetime.utcnow() - datetime.fromisoformat(news["timestamp"])).total_seconds() < 3600
                ]),
                "overall_sentiment": await self._calculate_overall_sentiment(),
                "symbol_sentiments": await self._calculate_symbol_sentiments(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Send response
            await self.send_direct_message(data["source"], {
                "type": "market_impact_response",
                "impact_summary": impact_summary
            })
            
        except Exception as e:
            self.logger.error(f"❌ Market impact request handling error: {e}")
    
    async def _handle_news_alert_config(self, data: Dict):
        """Handle news alert configuration"""
        try:
            # Update alert configuration
            config = data.get("config", {})
            
            if "sentiment_threshold" in config:
                self.sentiment_threshold = config["sentiment_threshold"]
            
            if "monitored_symbols" in config:
                self.monitored_symbols = config["monitored_symbols"]
            
            self.logger.info("📰 News alert configuration updated")
            
        except Exception as e:
            self.logger.error(f"❌ News alert config handling error: {e}")
    
    def get_news_metrics(self) -> Dict:
        """Get news agent metrics"""
        return {
            "news_processed": self.news_processed,
            "alerts_sent": self.alerts_sent,
            "sentiment_analyses": self.sentiment_analyses,
            "news_history_size": len(self.news_history),
            "sentiment_history_size": len(self.sentiment_history),
            "monitored_symbols": len(self.monitored_symbols),
            "active_sources": len([source for sources in self.news_sources.values() for source in sources]),
            "uptime": (datetime.utcnow() - self.metrics["uptime"]).total_seconds()
        }
