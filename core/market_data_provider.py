"""
Real-time Market Data Provider
Integrates multiple data sources: Yahoo Finance, Alpha Vantage, Finnhub
"""

import os
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from decimal import Decimal
import requests
import requests_cache
from dataclasses import dataclass

# Set up caching to avoid rate limits
requests_cache.install_cache('market_data_cache', expire_after=300)  # 5 minutes

logger = logging.getLogger(__name__)

@dataclass
class MarketQuote:
    """Market quote data structure"""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    high: float
    low: float
    open: float
    previous_close: float
    timestamp: datetime
    source: str

class MarketDataProvider:
    """Multi-source market data provider"""
    
    def __init__(self):
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.finnhub_key = os.getenv('FINNHUB_API_KEY')
        self.polygon_key = os.getenv('POLYGON_API_KEY')
        
        # Initialize data sources
        self.sources = {
            'yahoo': self._get_yahoo_quote,
            'alpha_vantage': self._get_alpha_vantage_quote,
            'finnhub': self._get_finnhub_quote,
            'polygon': self._get_polygon_quote
        }
        
        logger.info("Market data provider initialized")
    
    def get_quote(self, symbol: str, source: str = 'auto') -> Optional[MarketQuote]:
        """Get market quote for a symbol"""
        try:
            if source == 'auto':
                # Try sources in order of preference
                for source_name in ['yahoo', 'alpha_vantage', 'finnhub', 'polygon']:
                    try:
                        quote = self.sources[source_name](symbol)
                        if quote:
                            return quote
                    except Exception as e:
                        logger.warning(f"Failed to get quote from {source_name}: {e}")
                        continue
                
                # If all sources fail, return None
                return None
            else:
                return self.sources.get(source, self._get_yahoo_quote)(symbol)
                
        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
            return None
    
    def _get_yahoo_quote(self, symbol: str) -> Optional[MarketQuote]:
        """Get quote from Yahoo Finance using direct API calls"""
        try:
            # Use direct Yahoo Finance API endpoint
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'chart' in data and data['chart']['result']:
                    result = data['chart']['result'][0]
                    meta = result.get('meta', {})
                    
                    current_price = meta.get('regularMarketPrice', 0)
                    previous_close = meta.get('previousClose', 0)
                    change = current_price - previous_close
                    change_percent = (change / previous_close * 100) if previous_close else 0
                    
                    return MarketQuote(
                        symbol=symbol,
                        price=float(current_price),
                        change=float(change),
                        change_percent=float(change_percent),
                        volume=int(meta.get('regularMarketVolume', 0)),
                        high=float(meta.get('regularMarketDayHigh', 0)),
                        low=float(meta.get('regularMarketDayLow', 0)),
                        open=float(meta.get('regularMarketOpen', 0)),
                        previous_close=float(previous_close),
                        timestamp=datetime.now(),
                        source='yahoo'
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Yahoo Finance error for {symbol}: {e}")
            return None
    
    def _get_alpha_vantage_quote(self, symbol: str) -> Optional[MarketQuote]:
        """Get quote from Alpha Vantage (requires API key)"""
        if not self.alpha_vantage_key:
            return None
            
        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.alpha_vantage_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'Global Quote' in data:
                quote_data = data['Global Quote']
                current_price = float(quote_data.get('05. price', 0))
                change = float(quote_data.get('09. change', 0))
                change_percent = float(quote_data.get('10. change percent', '0').replace('%', ''))
                
                return MarketQuote(
                    symbol=symbol,
                    price=current_price,
                    change=change,
                    change_percent=change_percent,
                    volume=int(quote_data.get('06. volume', 0)),
                    high=float(quote_data.get('03. high', 0)),
                    low=float(quote_data.get('04. low', 0)),
                    open=float(quote_data.get('02. open', 0)),
                    previous_close=float(quote_data.get('08. previous close', 0)),
                    timestamp=datetime.now(),
                    source='alpha_vantage'
                )
            
        except Exception as e:
            logger.error(f"Alpha Vantage error for {symbol}: {e}")
            return None
    
    def _get_finnhub_quote(self, symbol: str) -> Optional[MarketQuote]:
        """Get quote from Finnhub (requires API key)"""
        if not self.finnhub_key:
            return None
            
        try:
            import finnhub
            
            finnhub_client = finnhub.Client(api_key=self.finnhub_key)
            quote = finnhub_client.quote(symbol)
            
            current_price = quote.get('c', 0)  # Current price
            previous_close = quote.get('pc', 0)  # Previous close
            change = current_price - previous_close
            change_percent = (change / previous_close * 100) if previous_close else 0
            
            return MarketQuote(
                symbol=symbol,
                price=float(current_price),
                change=float(change),
                change_percent=float(change_percent),
                volume=0,  # Finnhub doesn't provide volume in quote
                high=float(quote.get('h', 0)),
                low=float(quote.get('l', 0)),
                open=float(quote.get('o', 0)),
                previous_close=float(previous_close),
                timestamp=datetime.now(),
                source='finnhub'
            )
            
        except Exception as e:
            logger.error(f"Finnhub error for {symbol}: {e}")
            return None
    
    def _get_polygon_quote(self, symbol: str) -> Optional[MarketQuote]:
        """Get quote from Polygon.io (requires API key)"""
        if not self.polygon_key:
            return None
            
        try:
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev"
            params = {
                'adjusted': 'true',
                'apikey': self.polygon_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'results' in data and data['results']:
                result = data['results'][0]
                current_price = float(result.get('c', 0))  # Close price
                open_price = float(result.get('o', 0))
                change = current_price - open_price
                change_percent = (change / open_price * 100) if open_price else 0
                
                return MarketQuote(
                    symbol=symbol,
                    price=current_price,
                    change=change,
                    change_percent=change_percent,
                    volume=int(result.get('v', 0)),
                    high=float(result.get('h', 0)),
                    low=float(result.get('l', 0)),
                    open=open_price,
                    previous_close=open_price,
                    timestamp=datetime.now(),
                    source='polygon'
                )
                
        except Exception as e:
            logger.error(f"Polygon error for {symbol}: {e}")
            return None
    
    def get_multiple_quotes(self, symbols: List[str]) -> Dict[str, MarketQuote]:
        """Get quotes for multiple symbols"""
        quotes = {}
        
        for symbol in symbols:
            quote = self.get_quote(symbol)
            if quote:
                quotes[symbol] = quote
        
        return quotes
    
    def get_market_status(self) -> Dict[str, Any]:
        """Get market status information"""
        try:
            # Use direct Yahoo Finance API to get market status
            url = "https://query1.finance.yahoo.com/v8/finance/chart/SPY"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'chart' in data and data['chart']['result']:
                    result = data['chart']['result'][0]
                    meta = result.get('meta', {})
                    
                    return {
                        'is_open': meta.get('marketState') == 'REGULAR',
                        'market_state': meta.get('marketState', 'unknown'),
                        'timezone': meta.get('timezone', 'America/New_York'),
                        'timestamp': datetime.now()
                    }
            
            return {
                'is_open': False,
                'market_state': 'unknown',
                'timezone': 'America/New_York',
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting market status: {e}")
            return {
                'is_open': False,
                'market_state': 'unknown',
                'timezone': 'America/New_York',
                'timestamp': datetime.now()
            }

# Global instance
market_data_provider = MarketDataProvider()