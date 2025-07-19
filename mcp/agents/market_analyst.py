"""
Market Analyst Agent for MCP System
Handles market analysis, sentiment analysis, and trend detection
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import websockets
import numpy as np
import pandas as pd
from dataclasses import dataclass
import uuid

# AI imports
import os
from openai import OpenAI
import anthropic
from anthropic import Anthropic

# The newest OpenAI model is "gpt-4o", not "gpt-4"
# The newest Anthropic model is "claude-sonnet-4-20250514", not "claude-3-5-sonnet-20241022"

@dataclass
class MarketAnalysis:
    """Market analysis result"""
    symbol: str
    exchange: str
    analysis_type: str
    sentiment_score: float  # -1 to 1
    confidence: float  # 0 to 1
    trend_direction: str  # 'bullish', 'bearish', 'neutral'
    key_levels: Dict[str, float]
    technical_indicators: Dict[str, Any]
    fundamental_data: Dict[str, Any]
    news_sentiment: Dict[str, Any]
    recommendation: str
    timestamp: datetime

class MarketAnalystAgent:
    """Market Analyst Agent for comprehensive market analysis"""
    
    def __init__(self, mcp_server_url: str = "ws://localhost:9000"):
        self.agent_id = "market_analyst_001"
        self.agent_type = "market_analyst"
        self.name = "Market Analyst Agent"
        self.description = "Advanced market analysis with AI-powered insights"
        self.capabilities = [
            "technical_analysis",
            "sentiment_analysis",
            "trend_detection",
            "support_resistance",
            "pattern_recognition",
            "news_analysis",
            "fundamental_analysis"
        ]
        
        self.mcp_server_url = mcp_server_url
        self.websocket = None
        self.running = False
        
        # AI clients
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        # Analysis cache
        self.analysis_cache: Dict[str, MarketAnalysis] = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Market data
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.news_data: Dict[str, List[Dict]] = {}
        
        # Technical indicators
        self.indicators = {
            "rsi_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "bb_period": 20,
            "bb_std": 2.0
        }
        
        # Sentiment analysis
        self.sentiment_sources = ["news", "social_media", "analyst_reports"]
        self.sentiment_weights = {"news": 0.4, "social_media": 0.3, "analyst_reports": 0.3}
        
        # Setup logging
        self.logger = logging.getLogger(f"MarketAnalyst_{self.agent_id}")
        
        # Performance metrics
        self.metrics = {
            "analyses_performed": 0,
            "accuracy_rate": 0.0,
            "response_time": 0.0,
            "cache_hit_rate": 0.0
        }
    
    async def start(self):
        """Start the market analyst agent"""
        try:
            self.logger.info("Starting Market Analyst Agent...")
            
            # Connect to MCP server
            self.websocket = await websockets.connect(self.mcp_server_url)
            
            # Initialize with server
            await self._send_initialize_message()
            
            self.running = True
            
            # Start background tasks
            asyncio.create_task(self._listen_for_messages())
            asyncio.create_task(self._periodic_analysis())
            asyncio.create_task(self._cleanup_cache())
            asyncio.create_task(self._send_heartbeat())
            
            self.logger.info("Market Analyst Agent started successfully")
            
        except Exception as e:
            self.logger.error(f"Error starting Market Analyst Agent: {e}")
            raise
    
    async def stop(self):
        """Stop the market analyst agent"""
        self.logger.info("Stopping Market Analyst Agent...")
        
        self.running = False
        
        if self.websocket:
            await self.websocket.close()
        
        self.logger.info("Market Analyst Agent stopped")
    
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
        
        if request_type == "market_analysis":
            await self._handle_market_analysis_request(data)
        elif request_type == "sentiment_analysis":
            await self._handle_sentiment_analysis_request(data)
        elif request_type == "technical_analysis":
            await self._handle_technical_analysis_request(data)
        elif request_type == "trend_detection":
            await self._handle_trend_detection_request(data)
        else:
            await self._send_error_response(data, f"Unknown request type: {request_type}")
    
    async def _handle_notification(self, data: Dict[str, Any]):
        """Handle notification messages"""
        notification_type = data.get("payload", {}).get("notification_type")
        
        if notification_type == "market_update":
            await self._handle_market_update(data)
        elif notification_type == "news_update":
            await self._handle_news_update(data)
    
    async def _handle_market_analysis_request(self, data: Dict[str, Any]):
        """Handle market analysis request"""
        payload = data.get("payload", {})
        symbol = payload.get("symbol")
        exchange = payload.get("exchange")
        analysis_type = payload.get("analysis_type", "comprehensive")
        
        if not symbol or not exchange:
            await self._send_error_response(data, "Symbol and exchange are required")
            return
        
        try:
            # Check cache first
            cache_key = f"{symbol}_{exchange}_{analysis_type}"
            if cache_key in self.analysis_cache:
                cached_analysis = self.analysis_cache[cache_key]
                if (datetime.utcnow() - cached_analysis.timestamp).total_seconds() < self.cache_ttl:
                    await self._send_analysis_response(data, cached_analysis)
                    self.metrics["cache_hit_rate"] += 1
                    return
            
            # Perform analysis
            start_time = datetime.utcnow()
            analysis = await self._perform_market_analysis(symbol, exchange, analysis_type)
            
            # Cache result
            self.analysis_cache[cache_key] = analysis
            
            # Update metrics
            self.metrics["analyses_performed"] += 1
            self.metrics["response_time"] = (datetime.utcnow() - start_time).total_seconds()
            
            # Send response
            await self._send_analysis_response(data, analysis)
            
        except Exception as e:
            self.logger.error(f"Error performing market analysis: {e}")
            await self._send_error_response(data, str(e))
    
    async def _perform_market_analysis(self, symbol: str, exchange: str, analysis_type: str) -> MarketAnalysis:
        """Perform comprehensive market analysis"""
        # Get market data
        market_data = await self._get_market_data(symbol, exchange)
        
        # Technical analysis
        technical_indicators = await self._calculate_technical_indicators(market_data)
        
        # Sentiment analysis
        sentiment_data = await self._analyze_sentiment(symbol)
        
        # Fundamental analysis
        fundamental_data = await self._analyze_fundamentals(symbol, exchange)
        
        # AI-powered analysis
        ai_analysis = await self._ai_market_analysis(symbol, market_data, technical_indicators, sentiment_data)
        
        # Determine key levels
        key_levels = await self._identify_key_levels(market_data, technical_indicators)
        
        # Generate recommendation
        recommendation = await self._generate_recommendation(
            symbol, technical_indicators, sentiment_data, fundamental_data, ai_analysis
        )
        
        return MarketAnalysis(
            symbol=symbol,
            exchange=exchange,
            analysis_type=analysis_type,
            sentiment_score=sentiment_data.get("overall_score", 0.0),
            confidence=ai_analysis.get("confidence", 0.5),
            trend_direction=ai_analysis.get("trend_direction", "neutral"),
            key_levels=key_levels,
            technical_indicators=technical_indicators,
            fundamental_data=fundamental_data,
            news_sentiment=sentiment_data,
            recommendation=recommendation,
            timestamp=datetime.utcnow()
        )
    
    async def _get_market_data(self, symbol: str, exchange: str) -> pd.DataFrame:
        """Get market data for analysis"""
        # Request market data from MCP server
        request_message = {
            "message_id": str(uuid.uuid4()),
            "message_type": "request",
            "sender": self.agent_id,
            "payload": {
                "request_type": "market_data",
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": "1d",
                "limit": 200
            }
        }
        
        await self.websocket.send(json.dumps(request_message))
        
        # For now, generate sample data (in production, wait for response)
        dates = pd.date_range(end=datetime.utcnow(), periods=200, freq='D')
        np.random.seed(42)  # For reproducible results
        
        price = 100
        data = []
        for date in dates:
            change = np.random.normal(0, 0.02)
            price *= (1 + change)
            volume = np.random.randint(1000000, 10000000)
            
            data.append({
                'date': date,
                'open': price * (1 + np.random.normal(0, 0.005)),
                'high': price * (1 + abs(np.random.normal(0, 0.01))),
                'low': price * (1 - abs(np.random.normal(0, 0.01))),
                'close': price,
                'volume': volume
            })
        
        return pd.DataFrame(data)
    
    async def _calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators"""
        indicators = {}
        
        # RSI
        indicators["rsi"] = self._calculate_rsi(data["close"], self.indicators["rsi_period"])
        
        # MACD
        macd_data = self._calculate_macd(
            data["close"], 
            self.indicators["macd_fast"], 
            self.indicators["macd_slow"], 
            self.indicators["macd_signal"]
        )
        indicators["macd"] = macd_data
        
        # Bollinger Bands
        bb_data = self._calculate_bollinger_bands(
            data["close"], 
            self.indicators["bb_period"], 
            self.indicators["bb_std"]
        )
        indicators["bollinger_bands"] = bb_data
        
        # Moving Averages
        indicators["sma_20"] = data["close"].rolling(window=20).mean().iloc[-1]
        indicators["sma_50"] = data["close"].rolling(window=50).mean().iloc[-1]
        indicators["sma_200"] = data["close"].rolling(window=200).mean().iloc[-1]
        
        # Volume indicators
        indicators["volume_sma"] = data["volume"].rolling(window=20).mean().iloc[-1]
        indicators["volume_ratio"] = data["volume"].iloc[-1] / indicators["volume_sma"]
        
        # Current price
        indicators["current_price"] = data["close"].iloc[-1]
        
        return indicators
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> float:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    
    def _calculate_macd(self, prices: pd.Series, fast: int, slow: int, signal: int) -> Dict[str, float]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        
        return {
            "macd": macd.iloc[-1],
            "signal": signal_line.iloc[-1],
            "histogram": histogram.iloc[-1]
        }
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int, std: float) -> Dict[str, float]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        rolling_std = prices.rolling(window=period).std()
        
        upper_band = sma + (rolling_std * std)
        lower_band = sma - (rolling_std * std)
        
        return {
            "upper": upper_band.iloc[-1],
            "middle": sma.iloc[-1],
            "lower": lower_band.iloc[-1],
            "current_position": (prices.iloc[-1] - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1])
        }
    
    async def _analyze_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Analyze sentiment from multiple sources"""
        sentiment_data = {
            "overall_score": 0.0,
            "sources": {},
            "news_count": 0,
            "social_mentions": 0
        }
        
        # News sentiment (simulated)
        news_sentiment = np.random.uniform(-0.5, 0.5)
        sentiment_data["sources"]["news"] = news_sentiment
        sentiment_data["news_count"] = np.random.randint(10, 50)
        
        # Social media sentiment (simulated)
        social_sentiment = np.random.uniform(-0.3, 0.3)
        sentiment_data["sources"]["social_media"] = social_sentiment
        sentiment_data["social_mentions"] = np.random.randint(100, 1000)
        
        # Analyst reports sentiment (simulated)
        analyst_sentiment = np.random.uniform(-0.2, 0.6)
        sentiment_data["sources"]["analyst_reports"] = analyst_sentiment
        
        # Calculate weighted overall score
        overall_score = 0.0
        for source, weight in self.sentiment_weights.items():
            if source in sentiment_data["sources"]:
                overall_score += sentiment_data["sources"][source] * weight
        
        sentiment_data["overall_score"] = overall_score
        
        return sentiment_data
    
    async def _analyze_fundamentals(self, symbol: str, exchange: str) -> Dict[str, Any]:
        """Analyze fundamental data"""
        # Simulated fundamental data
        fundamental_data = {
            "pe_ratio": np.random.uniform(15, 25),
            "market_cap": np.random.uniform(1e9, 1e12),
            "revenue_growth": np.random.uniform(-0.1, 0.3),
            "debt_to_equity": np.random.uniform(0.2, 1.5),
            "dividend_yield": np.random.uniform(0, 0.05),
            "earnings_growth": np.random.uniform(-0.2, 0.4)
        }
        
        return fundamental_data
    
    async def _ai_market_analysis(self, symbol: str, market_data: pd.DataFrame, 
                                 technical_indicators: Dict[str, Any], 
                                 sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """AI-powered market analysis"""
        try:
            # Prepare context for AI analysis
            context = f"""
            Analyze the market data for {symbol}:
            
            Technical Indicators:
            - RSI: {technical_indicators.get('rsi', 0):.2f}
            - MACD: {technical_indicators.get('macd', {}).get('macd', 0):.4f}
            - Current Price: ${technical_indicators.get('current_price', 0):.2f}
            - Bollinger Band Position: {technical_indicators.get('bollinger_bands', {}).get('current_position', 0.5):.2f}
            
            Sentiment Analysis:
            - Overall Score: {sentiment_data.get('overall_score', 0):.2f}
            - News Sentiment: {sentiment_data.get('sources', {}).get('news', 0):.2f}
            - Social Media: {sentiment_data.get('sources', {}).get('social_media', 0):.2f}
            
            Provide trend direction (bullish/bearish/neutral) and confidence level (0-1).
            """
            
            # Use Anthropic for analysis
            response = self.anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",  # The newest Anthropic model
                max_tokens=1000,
                temperature=0.3,
                messages=[
                    {
                        "role": "user",
                        "content": context
                    }
                ]
            )
            
            # Parse AI response (simplified)
            analysis_text = response.content[0].text
            
            # Extract trend direction and confidence
            trend_direction = "neutral"
            confidence = 0.5
            
            if "bullish" in analysis_text.lower():
                trend_direction = "bullish"
                confidence = 0.7
            elif "bearish" in analysis_text.lower():
                trend_direction = "bearish"
                confidence = 0.7
            
            return {
                "trend_direction": trend_direction,
                "confidence": confidence,
                "analysis": analysis_text
            }
            
        except Exception as e:
            self.logger.error(f"Error in AI analysis: {e}")
            return {
                "trend_direction": "neutral",
                "confidence": 0.5,
                "analysis": "Error in AI analysis"
            }
    
    async def _identify_key_levels(self, market_data: pd.DataFrame, 
                                  technical_indicators: Dict[str, Any]) -> Dict[str, float]:
        """Identify key support and resistance levels"""
        current_price = technical_indicators.get("current_price", 0)
        
        # Calculate support and resistance levels
        high_points = market_data["high"].rolling(window=20).max()
        low_points = market_data["low"].rolling(window=20).min()
        
        resistance = high_points.iloc[-1]
        support = low_points.iloc[-1]
        
        # Fibonacci levels
        price_range = resistance - support
        fib_levels = {
            "23.6%": resistance - (price_range * 0.236),
            "38.2%": resistance - (price_range * 0.382),
            "50.0%": resistance - (price_range * 0.5),
            "61.8%": resistance - (price_range * 0.618),
            "78.6%": resistance - (price_range * 0.786)
        }
        
        return {
            "support": support,
            "resistance": resistance,
            "current_price": current_price,
            "fibonacci_levels": fib_levels
        }
    
    async def _generate_recommendation(self, symbol: str, technical_indicators: Dict[str, Any],
                                     sentiment_data: Dict[str, Any], fundamental_data: Dict[str, Any],
                                     ai_analysis: Dict[str, Any]) -> str:
        """Generate trading recommendation"""
        rsi = technical_indicators.get("rsi", 50)
        macd = technical_indicators.get("macd", {})
        sentiment = sentiment_data.get("overall_score", 0)
        trend = ai_analysis.get("trend_direction", "neutral")
        
        # Simple recommendation logic
        if trend == "bullish" and rsi < 70 and sentiment > 0.1:
            return "BUY"
        elif trend == "bearish" and rsi > 30 and sentiment < -0.1:
            return "SELL"
        else:
            return "HOLD"
    
    async def _send_analysis_response(self, original_message: Dict[str, Any], analysis: MarketAnalysis):
        """Send analysis response"""
        response = {
            "message_id": str(uuid.uuid4()),
            "message_type": "response",
            "sender": self.agent_id,
            "recipient": original_message["sender"],
            "payload": {
                "request_type": "market_analysis",
                "analysis": {
                    "symbol": analysis.symbol,
                    "exchange": analysis.exchange,
                    "sentiment_score": analysis.sentiment_score,
                    "confidence": analysis.confidence,
                    "trend_direction": analysis.trend_direction,
                    "key_levels": analysis.key_levels,
                    "technical_indicators": analysis.technical_indicators,
                    "fundamental_data": analysis.fundamental_data,
                    "recommendation": analysis.recommendation,
                    "timestamp": analysis.timestamp.isoformat()
                }
            },
            "correlation_id": original_message["message_id"]
        }
        
        await self.websocket.send(json.dumps(response))
    
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
    
    async def _handle_market_update(self, data: Dict[str, Any]):
        """Handle market update notification"""
        # Process market updates and trigger analysis if needed
        payload = data.get("payload", {})
        symbol = payload.get("symbol")
        
        if symbol:
            # Invalidate cache for this symbol
            cache_keys_to_remove = [key for key in self.analysis_cache.keys() if key.startswith(symbol)]
            for key in cache_keys_to_remove:
                del self.analysis_cache[key]
    
    async def _periodic_analysis(self):
        """Perform periodic analysis on watchlist"""
        while self.running:
            try:
                # Perform periodic analysis on key symbols
                watchlist = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
                
                for symbol in watchlist:
                    try:
                        analysis = await self._perform_market_analysis(symbol, "alpaca", "periodic")
                        
                        # Send notification if significant change
                        if abs(analysis.sentiment_score) > 0.5 or analysis.confidence > 0.8:
                            notification = {
                                "message_id": str(uuid.uuid4()),
                                "message_type": "notification",
                                "sender": self.agent_id,
                                "payload": {
                                    "notification_type": "market_alert",
                                    "symbol": symbol,
                                    "analysis": analysis.recommendation,
                                    "confidence": analysis.confidence,
                                    "trend": analysis.trend_direction
                                }
                            }
                            
                            await self.websocket.send(json.dumps(notification))
                            
                    except Exception as e:
                        self.logger.error(f"Error in periodic analysis for {symbol}: {e}")
                
                await asyncio.sleep(1800)  # 30 minutes
                
            except Exception as e:
                self.logger.error(f"Error in periodic analysis: {e}")
                await asyncio.sleep(1800)
    
    async def _cleanup_cache(self):
        """Clean up expired cache entries"""
        while self.running:
            try:
                current_time = datetime.utcnow()
                expired_keys = []
                
                for key, analysis in self.analysis_cache.items():
                    if (current_time - analysis.timestamp).total_seconds() > self.cache_ttl:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self.analysis_cache[key]
                
                await asyncio.sleep(600)  # Clean every 10 minutes
                
            except Exception as e:
                self.logger.error(f"Error cleaning cache: {e}")
                await asyncio.sleep(600)
    
    async def _send_heartbeat(self):
        """Send heartbeat to MCP server"""
        while self.running:
            try:
                heartbeat = {
                    "message_id": str(uuid.uuid4()),
                    "message_type": "heartbeat",
                    "sender": self.agent_id,
                    "payload": {"status": "online", "metrics": self.metrics}
                }
                
                await self.websocket.send(json.dumps(heartbeat))
                await asyncio.sleep(30)  # Send every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error sending heartbeat: {e}")
                await asyncio.sleep(30)

if __name__ == "__main__":
    # Run the agent
    agent = MarketAnalystAgent()
    
    async def main():
        await agent.start()
        
        # Keep running
        while agent.running:
            await asyncio.sleep(1)
    
    asyncio.run(main())
