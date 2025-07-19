#!/usr/bin/env python3
"""
Enterprise Market Data System
Professional-grade real-time market data aggregation using institutional APIs
"""

import asyncio
import logging
import os
import json
import redis
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd

# Enterprise API Imports
try:
    from alpha_vantage.timeseries import TimeSeries
    from alpha_vantage.fundamentaldata import FundamentalData
    ALPHA_VANTAGE_AVAILABLE = True
except ImportError:
    ALPHA_VANTAGE_AVAILABLE = False

try:
    from polygon import RESTClient as PolygonClient
    POLYGON_AVAILABLE = True
except ImportError:
    POLYGON_AVAILABLE = False

try:
    from iexfinance.stocks import Stock
    from iexfinance.refdata import get_symbols
    IEX_AVAILABLE = True
except ImportError:
    IEX_AVAILABLE = False

try:
    import finnhub
    FINNHUB_AVAILABLE = True
except ImportError:
    FINNHUB_AVAILABLE = False

# Technical Analysis
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False

logger = logging.getLogger(__name__)

class EnterpriseMarketData:
    """Professional market data aggregator using institutional APIs"""
    
    def __init__(self):
        # API Keys
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_KEY")
        self.polygon_key = os.getenv("POLYGON_API_KEY") 
        self.iex_token = os.getenv("IEX_TOKEN")
        self.finnhub_key = os.getenv("FINNHUB_API_KEY")
        
        # Initialize clients
        self.alpha_vantage_ts = None
        self.alpha_vantage_fd = None
        self.polygon_client = None
        self.finnhub_client = None
        
        # Redis for caching and real-time data
        try:
            self.redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
            self.redis_client.ping()
            logger.info("✅ Redis connected for real-time data caching")
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
            self.redis_client = None
        
        # Initialize API clients
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize all available API clients"""
        
        # Alpha Vantage (Professional fundamental data)
        if ALPHA_VANTAGE_AVAILABLE and self.alpha_vantage_key:
            self.alpha_vantage_ts = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')
            self.alpha_vantage_fd = FundamentalData(key=self.alpha_vantage_key, output_format='pandas')
            logger.info("✅ Alpha Vantage client initialized (Professional)")
        
        # Polygon.io (Institutional-grade real-time data)
        if POLYGON_AVAILABLE and self.polygon_key:
            self.polygon_client = PolygonClient(self.polygon_key)
            logger.info("✅ Polygon.io client initialized (Institutional)")
        
        # Finnhub (Real-time news and sentiment)
        if FINNHUB_AVAILABLE and self.finnhub_key:
            self.finnhub_client = finnhub.Client(api_key=self.finnhub_key)
            logger.info("✅ Finnhub client initialized (News & Sentiment)")
    
    async def get_real_time_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time quote from multiple professional sources"""
        symbol = symbol.upper()
        
        # Check Redis cache first (for sub-second data)
        if self.redis_client:
            cached_data = self.redis_client.get(f"quote:{symbol}")
            if cached_data:
                data = json.loads(cached_data)
                # Use cached data if less than 5 seconds old
                if datetime.fromisoformat(data['timestamp']) > datetime.now() - timedelta(seconds=5):
                    data['source'] = f"{data['source']} (Cached)"
                    return data
        
        # Try multiple sources in order of preference
        quote_data = None
        
        # 1. Polygon.io (Most reliable for real-time)
        if self.polygon_client:
            try:
                quote_data = await self._get_polygon_quote(symbol)
                if quote_data:
                    quote_data['source'] = 'Polygon.io Professional'
            except Exception as e:
                logger.warning(f"Polygon API error for {symbol}: {e}")
        
        # 2. Alpha Vantage (Fallback with fundamentals)
        if not quote_data and self.alpha_vantage_ts:
            try:
                quote_data = await self._get_alpha_vantage_quote(symbol)
                if quote_data:
                    quote_data['source'] = 'Alpha Vantage Professional'
            except Exception as e:
                logger.warning(f"Alpha Vantage API error for {symbol}: {e}")
        
        # 3. IEX Cloud (High-quality alternative)
        if not quote_data and self.iex_token:
            try:
                quote_data = await self._get_iex_quote(symbol)
                if quote_data:
                    quote_data['source'] = 'IEX Cloud Professional'
            except Exception as e:
                logger.warning(f"IEX API error for {symbol}: {e}")
        
        # Cache the result
        if quote_data and self.redis_client:
            self.redis_client.setex(f"quote:{symbol}", 5, json.dumps(quote_data, default=str))
        
        return quote_data or {"error": f"No data available for {symbol}", "symbol": symbol}
    
    async def _get_polygon_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get quote from Polygon.io (institutional-grade)"""
        try:
            # Get latest trade
            trades = list(self.polygon_client.list_trades(symbol, limit=1))
            
            # Get latest quote
            quotes = list(self.polygon_client.list_quotes(symbol, limit=1))
            
            # Get daily bar for additional data
            aggs = list(self.polygon_client.list_aggs(
                symbol, 
                1, 
                "day", 
                (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d"),
                datetime.now().strftime("%Y-%m-%d")
            ))
            
            if trades and aggs:
                latest_trade = trades[0]
                latest_quote = quotes[0] if quotes else None
                today_bar = aggs[-1] if aggs else None
                yesterday_bar = aggs[-2] if len(aggs) > 1 else None
                
                current_price = latest_trade.price
                previous_close = yesterday_bar.close if yesterday_bar else current_price
                
                return {
                    "symbol": symbol,
                    "price": current_price,
                    "change": current_price - previous_close,
                    "change_percent": ((current_price - previous_close) / previous_close) * 100 if previous_close > 0 else 0,
                    "volume": today_bar.volume if today_bar else latest_trade.size,
                    "high": today_bar.high if today_bar else current_price,
                    "low": today_bar.low if today_bar else current_price,
                    "open": today_bar.open if today_bar else current_price,
                    "previous_close": previous_close,
                    "bid": latest_quote.bid_price if latest_quote else current_price,
                    "ask": latest_quote.ask_price if latest_quote else current_price,
                    "bid_size": latest_quote.bid_size if latest_quote else 0,
                    "ask_size": latest_quote.ask_size if latest_quote else 0,
                    "asset_type": "stock",
                    "currency": "USD",
                    "exchange": "Professional Grade",
                    "timestamp": datetime.now().isoformat(),
                    "last_trade_time": latest_trade.participant_timestamp,
                    "conditions": getattr(latest_trade, 'conditions', [])
                }
        except Exception as e:
            logger.error(f"Polygon error for {symbol}: {e}")
        return None
    
    async def _get_alpha_vantage_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get quote from Alpha Vantage with fundamentals"""
        try:
            # Get intraday data (most recent)
            data, meta_data = self.alpha_vantage_ts.get_intraday(symbol, interval='1min', outputsize='compact')
            
            if not data.empty:
                latest_data = data.iloc[0]  # Most recent minute
                previous_data = data.iloc[1] if len(data) > 1 else latest_data
                
                current_price = float(latest_data['4. close'])
                previous_price = float(previous_data['4. close'])
                
                return {
                    "symbol": symbol,
                    "price": current_price,
                    "change": current_price - previous_price,
                    "change_percent": ((current_price - previous_price) / previous_price) * 100 if previous_price > 0 else 0,
                    "volume": int(latest_data['5. volume']),
                    "high": float(latest_data['2. high']),
                    "low": float(latest_data['3. low']),
                    "open": float(latest_data['1. open']),
                    "previous_close": previous_price,
                    "asset_type": "stock",
                    "currency": "USD",
                    "exchange": "Professional Grade",
                    "timestamp": datetime.now().isoformat(),
                    "interval": "1min",
                    "data_points": len(data)
                }
        except Exception as e:
            logger.error(f"Alpha Vantage error for {symbol}: {e}")
        return None
    
    async def _get_iex_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get quote from IEX Cloud"""
        try:
            stock = Stock(symbol, token=self.iex_token)
            quote = stock.get_quote()
            
            if quote:
                return {
                    "symbol": symbol,
                    "price": quote['latestPrice'],
                    "change": quote['change'],
                    "change_percent": quote['changePercent'] * 100,
                    "volume": quote['latestVolume'],
                    "high": quote['high'],
                    "low": quote['low'],
                    "open": quote['open'],
                    "previous_close": quote['previousClose'],
                    "market_cap": quote.get('marketCap', 0),
                    "pe_ratio": quote.get('peRatio', 0),
                    "asset_type": "stock",
                    "currency": "USD",
                    "exchange": quote.get('primaryExchange', 'IEX'),
                    "timestamp": datetime.now().isoformat(),
                    "latest_source": quote.get('latestSource'),
                    "latest_time": quote.get('latestTime'),
                    "is_us_market_open": quote.get('isUSMarketOpen', False)
                }
        except Exception as e:
            logger.error(f"IEX error for {symbol}: {e}")
        return None
    
    async def get_technical_indicators(self, symbol: str, period: str = "1d") -> Dict[str, Any]:
        """Calculate real-time technical indicators"""
        if not PANDAS_TA_AVAILABLE:
            return {"error": "Technical analysis library not available"}
        
        try:
            # Get historical data for calculations
            if self.alpha_vantage_ts:
                data, _ = self.alpha_vantage_ts.get_daily(symbol, outputsize='compact')
                
                if not data.empty:
                    # Prepare data for pandas_ta
                    df = data.copy()
                    df.columns = ['open', 'high', 'low', 'close', 'volume']
                    df = df.astype(float)
                    
                    # Calculate indicators
                    df.ta.rsi(append=True)  # RSI
                    df.ta.macd(append=True)  # MACD
                    df.ta.bbands(append=True)  # Bollinger Bands
                    df.ta.sma(length=20, append=True)  # 20-day SMA
                    df.ta.ema(length=12, append=True)  # 12-day EMA
                    df.ta.stoch(append=True)  # Stochastic
                    
                    latest = df.iloc[0]  # Most recent data
                    
                    return {
                        "symbol": symbol,
                        "rsi": latest.get('RSI_14', 0),
                        "macd": latest.get('MACD_12_26_9', 0),
                        "macd_signal": latest.get('MACDs_12_26_9', 0),
                        "bb_upper": latest.get('BBU_5_2.0', 0),
                        "bb_middle": latest.get('BBM_5_2.0', 0),
                        "bb_lower": latest.get('BBL_5_2.0', 0),
                        "sma_20": latest.get('SMA_20', 0),
                        "ema_12": latest.get('EMA_12', 0),
                        "stoch_k": latest.get('STOCHk_14_3_3', 0),
                        "stoch_d": latest.get('STOCHd_14_3_3', 0),
                        "source": "Alpha Vantage + pandas_ta",
                        "timestamp": datetime.now().isoformat()
                    }
        except Exception as e:
            logger.error(f"Technical analysis error for {symbol}: {e}")
        
        return {"error": f"Technical analysis failed for {symbol}"}
    
    async def get_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get real-time news and sentiment from Finnhub"""
        if not self.finnhub_client:
            return {"error": "Finnhub API not configured"}
        
        try:
            # Get company news
            news = self.finnhub_client.company_news(symbol, 
                                                    _from=(datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
                                                    to=datetime.now().strftime("%Y-%m-%d"))
            
            # Get sentiment scores
            sentiment = self.finnhub_client.news_sentiment(symbol)
            
            return {
                "symbol": symbol,
                "sentiment_score": sentiment.get('sentiment', 0),
                "buzz_articlesInLastWeek": sentiment.get('buzz', {}).get('articlesInLastWeek', 0),
                "buzz_buzz": sentiment.get('buzz', {}).get('buzz', 0),
                "news_count": len(news),
                "recent_headlines": [
                    {
                        "headline": article.get('headline'),
                        "summary": article.get('summary'),
                        "source": article.get('source'),
                        "datetime": article.get('datetime'),
                        "url": article.get('url')
                    } for article in news[:5]
                ],
                "source": "Finnhub Professional",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Finnhub sentiment error for {symbol}: {e}")
            return {"error": f"Sentiment analysis failed for {symbol}"}
    
    async def get_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Get fundamental data from Alpha Vantage"""
        if not self.alpha_vantage_fd:
            return {"error": "Alpha Vantage fundamentals not available"}
        
        try:
            # Get company overview
            overview = self.alpha_vantage_fd.get_company_overview(symbol)
            
            if overview and not overview.empty:
                data = overview.iloc[0]
                
                return {
                    "symbol": symbol,
                    "company_name": data.get('Name', ''),
                    "sector": data.get('Sector', ''),
                    "industry": data.get('Industry', ''),
                    "market_cap": float(data.get('MarketCapitalization', 0)) if data.get('MarketCapitalization') != 'None' else 0,
                    "pe_ratio": float(data.get('PERatio', 0)) if data.get('PERatio') != 'None' else 0,
                    "peg_ratio": float(data.get('PEGRatio', 0)) if data.get('PEGRatio') != 'None' else 0,
                    "book_value": float(data.get('BookValue', 0)) if data.get('BookValue') != 'None' else 0,
                    "dividend_yield": float(data.get('DividendYield', 0)) if data.get('DividendYield') != 'None' else 0,
                    "eps": float(data.get('EPS', 0)) if data.get('EPS') != 'None' else 0,
                    "revenue_ttm": float(data.get('RevenueTTM', 0)) if data.get('RevenueTTM') != 'None' else 0,
                    "profit_margin": float(data.get('ProfitMargin', 0)) if data.get('ProfitMargin') != 'None' else 0,
                    "52_week_high": float(data.get('52WeekHigh', 0)) if data.get('52WeekHigh') != 'None' else 0,
                    "52_week_low": float(data.get('52WeekLow', 0)) if data.get('52WeekLow') != 'None' else 0,
                    "analyst_target_price": float(data.get('AnalystTargetPrice', 0)) if data.get('AnalystTargetPrice') != 'None' else 0,
                    "source": "Alpha Vantage Fundamentals",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Fundamentals error for {symbol}: {e}")
        
        return {"error": f"Fundamentals data not available for {symbol}"}

# Global instance
enterprise_data = EnterpriseMarketData()