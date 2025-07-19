"""
Enhanced Market Data Provider with Comprehensive Asset Coverage
Supports: Stocks, ETFs, Crypto, Futures, Options, Forex, Commodities
"""

import os
import logging
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from decimal import Decimal
import requests
import requests_cache
from dataclasses import dataclass
from enum import Enum

# Set up caching
requests_cache.install_cache('enhanced_market_data_cache', expire_after=60)  # 1 minute for real-time data

logger = logging.getLogger(__name__)

class AssetType(Enum):
    STOCK = "stock"
    ETF = "etf"
    CRYPTO = "crypto"
    FUTURES = "futures"
    OPTIONS = "options"
    FOREX = "forex"
    COMMODITY = "commodity"
    INDEX = "index"

@dataclass
class EnhancedMarketQuote:
    """Enhanced market quote with asset type detection"""
    symbol: str
    asset_type: AssetType
    price: float
    change: float
    change_percent: float
    volume: int
    high: float
    low: float
    open: float
    previous_close: float
    market_cap: Optional[float] = None
    timestamp: datetime = None
    source: str = "unknown"
    currency: str = "USD"
    exchange: str = "unknown"
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class EnhancedMarketDataProvider:
    """Comprehensive market data provider for all asset types"""
    
    def __init__(self):
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.finnhub_key = os.getenv('FINNHUB_API_KEY')
        self.polygon_key = os.getenv('POLYGON_API_KEY')
        
        # Asset type detection patterns - expanded crypto list
        self.crypto_patterns = ['BTC', 'ETH', 'ADA', 'SOL', 'DOGE', 'XRP', 'LTC', 'BCH', 'DOT', 'UNI', 'LINK', 'MATIC', 'AVAX', 'SHIB', 'ATOM', 'HBAR', 'SUI', 'TRX', 'ALGO', 'FTM', 'NEAR', 'FLOW', 'MANA', 'SAND', 'AXS', 'HYPE', 'PEPE', 'FLOKI', 'BONK', 'WIF', 'POPCAT', 'PNUT', 'GALA', 'ENJ', 'CHZ', 'APE', 'LRC', 'IMX', 'ROSE', 'RNDR', 'FET', 'OCEAN', 'AGIX', 'TAO', 'THETA', 'TFUEL', 'VET', 'HOT', 'ONE', 'HARMONY', 'ZILLIQA', 'ZIL', 'IOTA', 'MIOTA', 'NANO', 'XNO', 'DASH', 'ZEC', 'XMR', 'ETC', 'BSV', 'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'USDD']
        self.forex_patterns = ['EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD', 'CNY', 'INR', 'MXN']
        self.commodity_patterns = ['GLD', 'SLV', 'OIL', 'GAS', 'GOLD', 'SILVER', 'CRUDE', 'BRENT']
        self.index_patterns = ['SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'DIA', 'TLT', 'GLD']
        
        logger.info("Enhanced market data provider initialized")
    
    def detect_asset_type(self, symbol: str) -> AssetType:
        """Detect asset type from symbol"""
        symbol_upper = symbol.upper()
        
        # Check for crypto patterns
        if any(crypto in symbol_upper for crypto in self.crypto_patterns):
            return AssetType.CRYPTO
        
        # Check for forex patterns (typically like EURUSD)
        if len(symbol_upper) == 6 and any(fx in symbol_upper for fx in self.forex_patterns):
            return AssetType.FOREX
        
        # Check for commodity patterns
        if any(commodity in symbol_upper for commodity in self.commodity_patterns):
            return AssetType.COMMODITY
        
        # Check for index patterns
        if symbol_upper in self.index_patterns:
            return AssetType.INDEX
        
        # Check for futures patterns (typically have month/year codes)
        if len(symbol_upper) > 4 and any(char.isdigit() for char in symbol_upper):
            return AssetType.FUTURES
        
        # Default to stock for most symbols
        return AssetType.STOCK
    
    def get_comprehensive_quote(self, symbol: str) -> Optional[EnhancedMarketQuote]:
        """Get comprehensive quote with asset type detection"""
        try:
            asset_type = self.detect_asset_type(symbol)
            
            # Try different data sources based on asset type
            if asset_type == AssetType.CRYPTO:
                return self._get_crypto_quote(symbol)
            elif asset_type == AssetType.FOREX:
                return self._get_forex_quote(symbol)
            elif asset_type == AssetType.COMMODITY:
                return self._get_commodity_quote(symbol)
            else:
                return self._get_stock_quote(symbol, asset_type)
                
        except Exception as e:
            logger.error(f"Error getting comprehensive quote for {symbol}: {e}")
            return None
    
    def _get_crypto_quote(self, symbol: str) -> Optional[EnhancedMarketQuote]:
        """Get cryptocurrency quote from multiple sources"""
        try:
            # Try CoinGecko first (free)
            quote = self._get_coingecko_quote(symbol)
            if quote:
                return quote
            
            # Fallback to Yahoo Finance crypto format
            crypto_symbol = f"{symbol}-USD"
            return self._get_yahoo_quote(crypto_symbol, AssetType.CRYPTO)
            
        except Exception as e:
            logger.error(f"Error getting crypto quote for {symbol}: {e}")
            return None
    
    def _get_coingecko_quote(self, symbol: str) -> Optional[EnhancedMarketQuote]:
        """Get crypto quote from CoinGecko (free API)"""
        try:
            # Comprehensive symbol mapping to CoinGecko IDs
            symbol_map = {
                'BTC': 'bitcoin',
                'ETH': 'ethereum',
                'ADA': 'cardano',
                'SOL': 'solana',
                'DOGE': 'dogecoin',
                'XRP': 'ripple',
                'LTC': 'litecoin',
                'BCH': 'bitcoin-cash',
                'DOT': 'polkadot',
                'UNI': 'uniswap',
                'LINK': 'chainlink',
                'MATIC': 'matic-network',
                'AVAX': 'avalanche-2',
                'SHIB': 'shiba-inu',
                'ATOM': 'cosmos',
                'HBAR': 'hedera-hashgraph',
                'SUI': 'sui',
                'TRX': 'tron',
                'ALGO': 'algorand',
                'FTM': 'fantom',
                'NEAR': 'near',
                'FLOW': 'flow',
                'MANA': 'decentraland',
                'SAND': 'the-sandbox',
                'AXS': 'axie-infinity',
                'HYPE': 'hyperliquid',
                'PEPE': 'pepe',
                'FLOKI': 'floki',
                'BONK': 'bonk',
                'WIF': 'dogwifcoin',
                'POPCAT': 'popcat',
                'PNUT': 'peanut-the-squirrel',
                'GALA': 'gala',
                'ENJ': 'enjincoin',
                'CHZ': 'chiliz',
                'APE': 'apecoin',
                'LRC': 'loopring',
                'IMX': 'immutable-x',
                'ROSE': 'oasis-network',
                'RNDR': 'render-token',
                'FET': 'fetch-ai',
                'OCEAN': 'ocean-protocol',
                'AGIX': 'singularitynet',
                'TAO': 'bittensor',
                'THETA': 'theta-token',
                'TFUEL': 'theta-fuel',
                'VET': 'vechain',
                'HOT': 'holo',
                'ONE': 'harmony',
                'ZIL': 'zilliqa',
                'IOTA': 'iota',
                'MIOTA': 'iota',
                'NANO': 'nano',
                'XNO': 'nano',
                'DASH': 'dash',
                'ZEC': 'zcash',
                'XMR': 'monero',
                'ETC': 'ethereum-classic',
                'BSV': 'bitcoin-sv',
                'USDT': 'tether',
                'USDC': 'usd-coin',
                'BUSD': 'binance-usd',
                'DAI': 'dai',
                'TUSD': 'trueusd',
                'USDD': 'usdd'
            }
            
            coin_id = symbol_map.get(symbol.upper())
            if not coin_id:
                # If not in our mapping, try to search CoinGecko for the symbol
                coin_id = self._search_coingecko_id(symbol)
                if not coin_id:
                    return None
            
            url = f"https://api.coingecko.com/api/v3/simple/price"
            params = {
                'ids': coin_id,
                'vs_currencies': 'usd',
                'include_24hr_change': 'true',
                'include_24hr_vol': 'true',
                'include_market_cap': 'true'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if coin_id in data:
                    coin_data = data[coin_id]
                    current_price = coin_data.get('usd', 0)
                    change_percent = coin_data.get('usd_24h_change', 0)
                    volume = coin_data.get('usd_24h_vol', 0)
                    market_cap = coin_data.get('usd_market_cap', 0)
                    
                    # Calculate change amount
                    change = (current_price * change_percent) / 100
                    
                    return EnhancedMarketQuote(
                        symbol=symbol.upper(),
                        asset_type=AssetType.CRYPTO,
                        price=float(current_price),
                        change=float(change),
                        change_percent=float(change_percent),
                        volume=int(volume),
                        high=0,  # CoinGecko simple API doesn't provide OHLC
                        low=0,
                        open=0,
                        previous_close=current_price - change,
                        market_cap=float(market_cap) if market_cap else None,
                        timestamp=datetime.now(),
                        source='coingecko',
                        currency='USD',
                        exchange='global'
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"CoinGecko error for {symbol}: {e}")
            return None
    
    def _search_coingecko_id(self, symbol: str) -> Optional[str]:
        """Search CoinGecko for coin ID by symbol"""
        try:
            # Use CoinGecko search API to find coin ID
            url = "https://api.coingecko.com/api/v3/search"
            params = {'query': symbol.upper()}
            
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                coins = data.get('coins', [])
                
                # Look for exact symbol match
                for coin in coins:
                    if coin.get('symbol', '').upper() == symbol.upper():
                        return coin.get('id')
                
                # If no exact match, return first result
                if coins:
                    return coins[0].get('id')
            
            return None
            
        except Exception as e:
            logger.error(f"CoinGecko search error for {symbol}: {e}")
            return None
    
    def _get_forex_quote(self, symbol: str) -> Optional[EnhancedMarketQuote]:
        """Get forex quote"""
        try:
            # Use Yahoo Finance for forex
            if len(symbol) == 6:
                # Format: EURUSD -> EUR=X
                base_currency = symbol[:3]
                quote_currency = symbol[3:]
                yahoo_symbol = f"{base_currency}{quote_currency}=X"
            else:
                yahoo_symbol = f"{symbol}=X"
            
            return self._get_yahoo_quote(yahoo_symbol, AssetType.FOREX)
            
        except Exception as e:
            logger.error(f"Error getting forex quote for {symbol}: {e}")
            return None
    
    def _get_commodity_quote(self, symbol: str) -> Optional[EnhancedMarketQuote]:
        """Get commodity quote"""
        try:
            return self._get_yahoo_quote(symbol, AssetType.COMMODITY)
            
        except Exception as e:
            logger.error(f"Error getting commodity quote for {symbol}: {e}")
            return None
    
    def _get_stock_quote(self, symbol: str, asset_type: AssetType) -> Optional[EnhancedMarketQuote]:
        """Get stock/ETF quote"""
        try:
            return self._get_yahoo_quote(symbol, asset_type)
            
        except Exception as e:
            logger.error(f"Error getting stock quote for {symbol}: {e}")
            return None
    
    def _get_yahoo_quote(self, symbol: str, asset_type: AssetType) -> Optional[EnhancedMarketQuote]:
        """Get quote from Yahoo Finance with asset type"""
        try:
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
                    
                    return EnhancedMarketQuote(
                        symbol=symbol,
                        asset_type=asset_type,
                        price=float(current_price),
                        change=float(change),
                        change_percent=float(change_percent),
                        volume=int(meta.get('regularMarketVolume', 0)),
                        high=float(meta.get('regularMarketDayHigh', 0)),
                        low=float(meta.get('regularMarketDayLow', 0)),
                        open=float(meta.get('regularMarketOpen', 0)),
                        previous_close=float(previous_close),
                        market_cap=meta.get('marketCap'),
                        timestamp=datetime.now(),
                        source='yahoo',
                        currency=meta.get('currency', 'USD'),
                        exchange=meta.get('exchangeName', 'unknown')
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Yahoo Finance error for {symbol}: {e}")
            return None
    
    def get_popular_assets(self) -> Dict[str, List[str]]:
        """Get popular assets by category"""
        return {
            'stocks': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA', 'NFLX'],
            'crypto': ['BTC', 'ETH', 'SOL', 'ADA', 'DOGE', 'XRP', 'HBAR', 'SUI', 'TRX', 'ALGO', 'FTM', 'NEAR', 'MANA', 'SAND', 'PEPE', 'FLOKI'],
            'etfs': ['SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'TLT'],
            'forex': ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD'],
            'commodities': ['GLD', 'SLV', 'USO', 'UNG', 'DBA', 'DBB']
        }

# Global instance
enhanced_market_data_provider = EnhancedMarketDataProvider()