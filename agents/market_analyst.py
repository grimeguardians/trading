"""
Market Analyst Agent - Provides market analysis and insights
Specializes in technical analysis, pattern recognition, and market sentiment
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import yfinance as yf
import talib

from agents.base_agent import BaseAgent
from analysis.technical_indicators import TechnicalIndicators
from analysis.fibonacci import FibonacciAnalysis
from analysis.pattern_recognition import PatternRecognition
from mcp_server import MessageType


class MarketAnalyst(BaseAgent):
    """
    Market Analyst Agent for comprehensive market analysis
    Provides technical analysis, pattern recognition, and market insights
    """
    
    def __init__(self, mcp_server, knowledge_engine, config):
        super().__init__(
            agent_id="market_analyst",
            agent_type="analyst",
            mcp_server=mcp_server,
            knowledge_engine=knowledge_engine,
            config=config
        )
        
        # Initialize analysis modules
        self.technical_indicators = TechnicalIndicators()
        self.fibonacci_analysis = FibonacciAnalysis()
        self.pattern_recognition = PatternRecognition()
        
        # Analysis settings
        self.analysis_symbols = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "SPY", "QQQ"]
        self.analysis_timeframes = ["1d", "1h", "15m"]
        self.analysis_interval = 300  # 5 minutes
        
        # Market data cache
        self.market_data_cache = {}
        self.last_analysis_time = {}
        
        # Performance tracking
        self.analysis_count = 0
        self.patterns_found = 0
        self.signals_generated = 0
    
    def _setup_capabilities(self):
        """Setup market analyst capabilities"""
        self.capabilities = [
            "technical_analysis",
            "pattern_recognition",
            "fibonacci_analysis",
            "market_sentiment",
            "support_resistance",
            "trend_analysis",
            "volatility_analysis",
            "correlation_analysis"
        ]
    
    def _setup_message_handlers(self):
        """Setup message handlers"""
        self.register_message_handler("analysis_request", self._handle_analysis_request)
        self.register_message_handler("pattern_scan_request", self._handle_pattern_scan_request)
        self.register_message_handler("market_data_update", self._handle_market_data_update)
        self.register_message_handler("fibonacci_analysis_request", self._handle_fibonacci_analysis_request)
    
    async def _agent_logic(self):
        """Main market analyst logic"""
        self.logger.info("🔍 Market Analyst started - beginning market analysis")
        
        while self.running:
            try:
                # Perform scheduled market analysis
                await self._perform_market_analysis()
                
                # Look for trading patterns
                await self._scan_for_patterns()
                
                # Update market sentiment
                await self._update_market_sentiment()
                
                # Generate market insights
                await self._generate_market_insights()
                
                # Wait for next analysis cycle
                await asyncio.sleep(self.analysis_interval)
                
            except Exception as e:
                self.logger.error(f"❌ Market analysis error: {e}")
                await asyncio.sleep(30)
    
    async def _perform_market_analysis(self):
        """Perform comprehensive market analysis"""
        try:
            for symbol in self.analysis_symbols:
                # Get market data
                market_data = await self._get_market_data(symbol)
                
                if market_data is None or market_data.empty:
                    continue
                
                # Perform technical analysis
                technical_analysis = await self._analyze_technical_indicators(symbol, market_data)
                
                # Perform trend analysis
                trend_analysis = await self._analyze_trends(symbol, market_data)
                
                # Perform volatility analysis
                volatility_analysis = await self._analyze_volatility(symbol, market_data)
                
                # Combine analysis results
                analysis_result = {
                    "symbol": symbol,
                    "timestamp": datetime.utcnow().isoformat(),
                    "technical_analysis": technical_analysis,
                    "trend_analysis": trend_analysis,
                    "volatility_analysis": volatility_analysis,
                    "current_price": float(market_data['Close'].iloc[-1]),
                    "price_change_pct": self._calculate_price_change(market_data)
                }
                
                # Store in knowledge engine
                await self.update_knowledge("add_node", {
                    "node_type": "market_analysis",
                    "symbol": symbol,
                    "analysis": analysis_result
                })
                
                # Broadcast analysis to other agents
                await self.broadcast_message({
                    "type": "market_analysis_update",
                    "data": analysis_result
                })
                
                self.analysis_count += 1
                
        except Exception as e:
            self.logger.error(f"❌ Market analysis error: {e}")
    
    async def _get_market_data(self, symbol: str, period: str = "1d") -> Optional[pd.DataFrame]:
        """Get market data for analysis"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{period}"
            if cache_key in self.market_data_cache:
                cached_data, cache_time = self.market_data_cache[cache_key]
                if (datetime.utcnow() - cache_time).total_seconds() < 60:  # 1 minute cache
                    return cached_data
            
            # Fetch fresh data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="5d", interval="1h")
            
            if not data.empty:
                # Cache the data
                self.market_data_cache[cache_key] = (data, datetime.utcnow())
                return data
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching market data for {symbol}: {e}")
            return None
    
    async def _analyze_technical_indicators(self, symbol: str, data: pd.DataFrame) -> Dict:
        """Analyze technical indicators"""
        try:
            close_prices = data['Close'].values
            high_prices = data['High'].values
            low_prices = data['Low'].values
            volume = data['Volume'].values
            
            # Calculate technical indicators
            indicators = {
                "rsi": self.technical_indicators.calculate_rsi(close_prices),
                "macd": self.technical_indicators.calculate_macd(close_prices),
                "bollinger_bands": self.technical_indicators.calculate_bollinger_bands(close_prices),
                "stochastic": self.technical_indicators.calculate_stochastic(high_prices, low_prices, close_prices),
                "williams_r": self.technical_indicators.calculate_williams_r(high_prices, low_prices, close_prices),
                "sma_20": self.technical_indicators.calculate_sma(close_prices, 20),
                "sma_50": self.technical_indicators.calculate_sma(close_prices, 50),
                "ema_12": self.technical_indicators.calculate_ema(close_prices, 12),
                "ema_26": self.technical_indicators.calculate_ema(close_prices, 26),
                "atr": self.technical_indicators.calculate_atr(high_prices, low_prices, close_prices),
                "obv": self.technical_indicators.calculate_obv(close_prices, volume)
            }
            
            # Generate trading signals based on indicators
            signals = self._generate_technical_signals(indicators)
            
            return {
                "indicators": indicators,
                "signals": signals,
                "signal_strength": self._calculate_signal_strength(signals)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Technical analysis error for {symbol}: {e}")
            return {}
    
    def _generate_technical_signals(self, indicators: Dict) -> List[Dict]:
        """Generate trading signals from technical indicators"""
        signals = []
        
        try:
            # RSI signals
            rsi_current = indicators["rsi"][-1] if len(indicators["rsi"]) > 0 else 50
            if rsi_current < 30:
                signals.append({
                    "indicator": "RSI",
                    "signal": "BUY",
                    "strength": min((30 - rsi_current) / 10, 1.0),
                    "reason": f"RSI oversold at {rsi_current:.2f}"
                })
            elif rsi_current > 70:
                signals.append({
                    "indicator": "RSI",
                    "signal": "SELL",
                    "strength": min((rsi_current - 70) / 10, 1.0),
                    "reason": f"RSI overbought at {rsi_current:.2f}"
                })
            
            # MACD signals
            macd_data = indicators["macd"]
            if len(macd_data) >= 2:
                macd_current = macd_data[-1]
                macd_previous = macd_data[-2]
                
                if macd_current > 0 and macd_previous <= 0:
                    signals.append({
                        "indicator": "MACD",
                        "signal": "BUY",
                        "strength": 0.7,
                        "reason": "MACD bullish crossover"
                    })
                elif macd_current < 0 and macd_previous >= 0:
                    signals.append({
                        "indicator": "MACD",
                        "signal": "SELL",
                        "strength": 0.7,
                        "reason": "MACD bearish crossover"
                    })
            
            # Bollinger Bands signals
            bb_data = indicators["bollinger_bands"]
            if len(bb_data) > 0:
                current_price = bb_data[-1]["price"]
                lower_band = bb_data[-1]["lower"]
                upper_band = bb_data[-1]["upper"]
                
                if current_price <= lower_band:
                    signals.append({
                        "indicator": "Bollinger Bands",
                        "signal": "BUY",
                        "strength": 0.6,
                        "reason": "Price at lower Bollinger Band"
                    })
                elif current_price >= upper_band:
                    signals.append({
                        "indicator": "Bollinger Bands",
                        "signal": "SELL",
                        "strength": 0.6,
                        "reason": "Price at upper Bollinger Band"
                    })
            
            # Stochastic signals
            stoch_data = indicators["stochastic"]
            if len(stoch_data) > 0:
                stoch_k = stoch_data[-1]["k"]
                stoch_d = stoch_data[-1]["d"]
                
                if stoch_k < 20 and stoch_d < 20:
                    signals.append({
                        "indicator": "Stochastic",
                        "signal": "BUY",
                        "strength": 0.5,
                        "reason": "Stochastic oversold"
                    })
                elif stoch_k > 80 and stoch_d > 80:
                    signals.append({
                        "indicator": "Stochastic",
                        "signal": "SELL",
                        "strength": 0.5,
                        "reason": "Stochastic overbought"
                    })
            
        except Exception as e:
            self.logger.error(f"❌ Error generating technical signals: {e}")
        
        return signals
    
    def _calculate_signal_strength(self, signals: List[Dict]) -> float:
        """Calculate overall signal strength"""
        if not signals:
            return 0.0
        
        buy_strength = sum(s["strength"] for s in signals if s["signal"] == "BUY")
        sell_strength = sum(s["strength"] for s in signals if s["signal"] == "SELL")
        
        # Return net signal strength (-1 to 1)
        return (buy_strength - sell_strength) / len(signals)
    
    async def _analyze_trends(self, symbol: str, data: pd.DataFrame) -> Dict:
        """Analyze price trends"""
        try:
            close_prices = data['Close'].values
            
            # Calculate trend indicators
            sma_20 = self.technical_indicators.calculate_sma(close_prices, 20)
            sma_50 = self.technical_indicators.calculate_sma(close_prices, 50)
            
            # Determine trend direction
            current_price = close_prices[-1]
            short_trend = "bullish" if current_price > sma_20[-1] else "bearish"
            long_trend = "bullish" if sma_20[-1] > sma_50[-1] else "bearish"
            
            # Calculate trend strength
            trend_strength = abs(current_price - sma_20[-1]) / sma_20[-1]
            
            # Support and resistance levels
            support_resistance = self._find_support_resistance(data)
            
            return {
                "short_term_trend": short_trend,
                "long_term_trend": long_trend,
                "trend_strength": trend_strength,
                "support_levels": support_resistance["support"],
                "resistance_levels": support_resistance["resistance"],
                "current_price": current_price,
                "sma_20": sma_20[-1],
                "sma_50": sma_50[-1]
            }
            
        except Exception as e:
            self.logger.error(f"❌ Trend analysis error for {symbol}: {e}")
            return {}
    
    def _find_support_resistance(self, data: pd.DataFrame) -> Dict:
        """Find support and resistance levels"""
        try:
            high_prices = data['High'].values
            low_prices = data['Low'].values
            
            # Find local maxima and minima
            from scipy.signal import argrelextrema
            
            # Find resistance levels (local maxima)
            resistance_indices = argrelextrema(high_prices, np.greater, order=5)[0]
            resistance_levels = [high_prices[i] for i in resistance_indices[-5:]]  # Last 5 levels
            
            # Find support levels (local minima)
            support_indices = argrelextrema(low_prices, np.less, order=5)[0]
            support_levels = [low_prices[i] for i in support_indices[-5:]]  # Last 5 levels
            
            return {
                "support": sorted(support_levels),
                "resistance": sorted(resistance_levels, reverse=True)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Support/resistance analysis error: {e}")
            return {"support": [], "resistance": []}
    
    async def _analyze_volatility(self, symbol: str, data: pd.DataFrame) -> Dict:
        """Analyze price volatility"""
        try:
            close_prices = data['Close'].values
            
            # Calculate volatility metrics
            returns = np.diff(np.log(close_prices))
            volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
            
            # Calculate ATR for volatility measure
            atr = self.technical_indicators.calculate_atr(
                data['High'].values,
                data['Low'].values,
                data['Close'].values
            )
            
            current_atr = atr[-1] if len(atr) > 0 else 0
            
            # VIX-like calculation (simplified)
            rolling_vol = pd.Series(returns).rolling(window=20).std() * np.sqrt(252)
            
            return {
                "annualized_volatility": volatility,
                "current_atr": current_atr,
                "volatility_percentile": self._calculate_volatility_percentile(rolling_vol),
                "volatility_trend": "increasing" if rolling_vol.iloc[-1] > rolling_vol.iloc[-5] else "decreasing",
                "risk_level": self._assess_risk_level(volatility)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Volatility analysis error for {symbol}: {e}")
            return {}
    
    def _calculate_volatility_percentile(self, rolling_vol: pd.Series) -> float:
        """Calculate volatility percentile"""
        try:
            current_vol = rolling_vol.iloc[-1]
            return (rolling_vol < current_vol).mean() * 100
        except:
            return 50.0
    
    def _assess_risk_level(self, volatility: float) -> str:
        """Assess risk level based on volatility"""
        if volatility < 0.15:
            return "low"
        elif volatility < 0.30:
            return "medium"
        elif volatility < 0.50:
            return "high"
        else:
            return "very_high"
    
    def _calculate_price_change(self, data: pd.DataFrame) -> float:
        """Calculate price change percentage"""
        try:
            close_prices = data['Close'].values
            if len(close_prices) < 2:
                return 0.0
            
            return ((close_prices[-1] - close_prices[-2]) / close_prices[-2]) * 100
            
        except Exception as e:
            self.logger.error(f"❌ Price change calculation error: {e}")
            return 0.0
    
    async def _scan_for_patterns(self):
        """Scan for trading patterns"""
        try:
            for symbol in self.analysis_symbols:
                market_data = await self._get_market_data(symbol)
                
                if market_data is None or market_data.empty:
                    continue
                
                # Fibonacci analysis
                fibonacci_levels = self.fibonacci_analysis.calculate_fibonacci_levels(market_data)
                
                # Pattern recognition
                patterns = self.pattern_recognition.identify_patterns(market_data)
                
                if patterns:
                    self.patterns_found += len(patterns)
                    
                    # Send pattern alerts
                    await self.broadcast_message({
                        "type": "patterns_detected",
                        "symbol": symbol,
                        "patterns": patterns,
                        "fibonacci_levels": fibonacci_levels
                    })
                
        except Exception as e:
            self.logger.error(f"❌ Pattern scanning error: {e}")
    
    async def _update_market_sentiment(self):
        """Update overall market sentiment"""
        try:
            # Analyze major indices
            indices = ["SPY", "QQQ", "DIA", "IWM"]
            sentiment_scores = []
            
            for index in indices:
                market_data = await self._get_market_data(index)
                if market_data is not None and not market_data.empty:
                    # Calculate sentiment based on technical indicators
                    sentiment = self._calculate_sentiment_score(market_data)
                    sentiment_scores.append(sentiment)
            
            if sentiment_scores:
                overall_sentiment = np.mean(sentiment_scores)
                sentiment_label = self._classify_sentiment(overall_sentiment)
                
                # Broadcast sentiment update
                await self.broadcast_message({
                    "type": "market_sentiment_update",
                    "overall_sentiment": overall_sentiment,
                    "sentiment_label": sentiment_label,
                    "index_sentiments": dict(zip(indices, sentiment_scores))
                })
                
                # Update knowledge engine
                await self.update_knowledge("add_node", {
                    "node_type": "market_sentiment",
                    "sentiment_score": overall_sentiment,
                    "sentiment_label": sentiment_label,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
        except Exception as e:
            self.logger.error(f"❌ Market sentiment update error: {e}")
    
    def _calculate_sentiment_score(self, data: pd.DataFrame) -> float:
        """Calculate sentiment score from market data"""
        try:
            close_prices = data['Close'].values
            
            # Calculate various sentiment indicators
            rsi = self.technical_indicators.calculate_rsi(close_prices)
            macd = self.technical_indicators.calculate_macd(close_prices)
            
            # Price momentum
            price_momentum = (close_prices[-1] - close_prices[-10]) / close_prices[-10]
            
            # Combine indicators for sentiment score
            sentiment_score = 0.0
            
            if len(rsi) > 0:
                # RSI contribution (normalized to -1 to 1)
                rsi_score = (rsi[-1] - 50) / 50
                sentiment_score += rsi_score * 0.3
            
            if len(macd) > 0:
                # MACD contribution
                macd_score = np.tanh(macd[-1])  # Normalize with tanh
                sentiment_score += macd_score * 0.4
            
            # Price momentum contribution
            momentum_score = np.tanh(price_momentum * 10)  # Normalize with tanh
            sentiment_score += momentum_score * 0.3
            
            return np.clip(sentiment_score, -1.0, 1.0)
            
        except Exception as e:
            self.logger.error(f"❌ Sentiment score calculation error: {e}")
            return 0.0
    
    def _classify_sentiment(self, sentiment_score: float) -> str:
        """Classify sentiment score into label"""
        if sentiment_score > 0.5:
            return "very_bullish"
        elif sentiment_score > 0.2:
            return "bullish"
        elif sentiment_score > -0.2:
            return "neutral"
        elif sentiment_score > -0.5:
            return "bearish"
        else:
            return "very_bearish"
    
    async def _generate_market_insights(self):
        """Generate market insights and recommendations"""
        try:
            insights = []
            
            # Analyze cross-asset correlations
            correlations = await self._analyze_correlations()
            
            # Generate insights based on analysis
            if correlations:
                insights.append({
                    "type": "correlation_insight",
                    "message": "Cross-asset correlation analysis completed",
                    "data": correlations
                })
            
            # Market regime analysis
            regime = await self._analyze_market_regime()
            if regime:
                insights.append({
                    "type": "regime_insight",
                    "message": f"Market regime classified as: {regime['regime']}",
                    "data": regime
                })
            
            # Broadcast insights
            if insights:
                await self.broadcast_message({
                    "type": "market_insights",
                    "insights": insights,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
        except Exception as e:
            self.logger.error(f"❌ Market insights generation error: {e}")
    
    async def _analyze_correlations(self) -> Dict:
        """Analyze correlations between assets"""
        try:
            # Get data for correlation analysis
            correlation_data = {}
            for symbol in self.analysis_symbols[:5]:  # Limit to first 5 for performance
                data = await self._get_market_data(symbol)
                if data is not None and not data.empty:
                    correlation_data[symbol] = data['Close'].pct_change().dropna()
            
            if len(correlation_data) > 1:
                # Calculate correlation matrix
                df = pd.DataFrame(correlation_data)
                correlation_matrix = df.corr()
                
                return {
                    "correlation_matrix": correlation_matrix.to_dict(),
                    "highest_correlation": self._find_highest_correlation(correlation_matrix),
                    "lowest_correlation": self._find_lowest_correlation(correlation_matrix)
                }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"❌ Correlation analysis error: {e}")
            return {}
    
    def _find_highest_correlation(self, corr_matrix: pd.DataFrame) -> Dict:
        """Find highest correlation pair"""
        try:
            # Remove diagonal and get upper triangle
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            corr_values = corr_matrix.where(mask)
            
            # Find maximum correlation
            max_corr = corr_values.max().max()
            max_idx = corr_values.stack().idxmax()
            
            return {
                "pair": list(max_idx),
                "correlation": max_corr
            }
        except:
            return {}
    
    def _find_lowest_correlation(self, corr_matrix: pd.DataFrame) -> Dict:
        """Find lowest correlation pair"""
        try:
            # Remove diagonal and get upper triangle
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            corr_values = corr_matrix.where(mask)
            
            # Find minimum correlation
            min_corr = corr_values.min().min()
            min_idx = corr_values.stack().idxmin()
            
            return {
                "pair": list(min_idx),
                "correlation": min_corr
            }
        except:
            return {}
    
    async def _analyze_market_regime(self) -> Dict:
        """Analyze current market regime"""
        try:
            # Get SPY data for regime analysis
            spy_data = await self._get_market_data("SPY")
            if spy_data is None or spy_data.empty:
                return {}
            
            close_prices = spy_data['Close'].values
            
            # Calculate regime indicators
            volatility = np.std(np.diff(np.log(close_prices))) * np.sqrt(252)
            sma_20 = self.technical_indicators.calculate_sma(close_prices, 20)
            sma_50 = self.technical_indicators.calculate_sma(close_prices, 50)
            
            # Determine regime
            if volatility > 0.25:
                regime = "high_volatility"
            elif close_prices[-1] > sma_20[-1] and sma_20[-1] > sma_50[-1]:
                regime = "bull_market"
            elif close_prices[-1] < sma_20[-1] and sma_20[-1] < sma_50[-1]:
                regime = "bear_market"
            else:
                regime = "sideways"
            
            return {
                "regime": regime,
                "volatility": volatility,
                "trend_strength": abs(close_prices[-1] - sma_20[-1]) / sma_20[-1],
                "confidence": 0.8  # Simplified confidence score
            }
            
        except Exception as e:
            self.logger.error(f"❌ Market regime analysis error: {e}")
            return {}
    
    async def _handle_analysis_request(self, data: Dict):
        """Handle analysis request from other agents"""
        try:
            symbol = data.get("symbol")
            analysis_type = data.get("analysis_type", "full")
            
            if not symbol:
                return
            
            market_data = await self._get_market_data(symbol)
            if market_data is None or market_data.empty:
                return
            
            # Perform requested analysis
            if analysis_type == "technical":
                result = await self._analyze_technical_indicators(symbol, market_data)
            elif analysis_type == "trend":
                result = await self._analyze_trends(symbol, market_data)
            elif analysis_type == "volatility":
                result = await self._analyze_volatility(symbol, market_data)
            else:
                # Full analysis
                result = {
                    "technical": await self._analyze_technical_indicators(symbol, market_data),
                    "trend": await self._analyze_trends(symbol, market_data),
                    "volatility": await self._analyze_volatility(symbol, market_data)
                }
            
            # Send result back to requester
            await self.send_direct_message(data["source"], {
                "type": "analysis_result",
                "symbol": symbol,
                "analysis_type": analysis_type,
                "result": result
            })
            
        except Exception as e:
            self.logger.error(f"❌ Analysis request handling error: {e}")
    
    async def _handle_pattern_scan_request(self, data: Dict):
        """Handle pattern scan request"""
        try:
            symbol = data.get("symbol")
            if not symbol:
                return
            
            market_data = await self._get_market_data(symbol)
            if market_data is None or market_data.empty:
                return
            
            # Perform pattern analysis
            patterns = self.pattern_recognition.identify_patterns(market_data)
            fibonacci_levels = self.fibonacci_analysis.calculate_fibonacci_levels(market_data)
            
            # Send results
            await self.send_direct_message(data["source"], {
                "type": "pattern_scan_result",
                "symbol": symbol,
                "patterns": patterns,
                "fibonacci_levels": fibonacci_levels
            })
            
        except Exception as e:
            self.logger.error(f"❌ Pattern scan request handling error: {e}")
    
    async def _handle_market_data_update(self, data: Dict):
        """Handle market data updates"""
        try:
            # Update cache with new market data
            symbol = data.get("symbol")
            if symbol:
                # Invalidate cache for this symbol
                cache_keys_to_remove = [k for k in self.market_data_cache.keys() if k.startswith(symbol)]
                for key in cache_keys_to_remove:
                    del self.market_data_cache[key]
                
                self.logger.debug(f"📊 Market data cache updated for {symbol}")
                
        except Exception as e:
            self.logger.error(f"❌ Market data update handling error: {e}")
    
    async def _handle_fibonacci_analysis_request(self, data: Dict):
        """Handle Fibonacci analysis request"""
        try:
            symbol = data.get("symbol")
            if not symbol:
                return
            
            market_data = await self._get_market_data(symbol)
            if market_data is None or market_data.empty:
                return
            
            # Calculate Fibonacci levels
            fibonacci_levels = self.fibonacci_analysis.calculate_fibonacci_levels(market_data)
            
            # Send results
            await self.send_direct_message(data["source"], {
                "type": "fibonacci_analysis_result",
                "symbol": symbol,
                "fibonacci_levels": fibonacci_levels
            })
            
        except Exception as e:
            self.logger.error(f"❌ Fibonacci analysis request handling error: {e}")
