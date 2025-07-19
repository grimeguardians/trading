"""
Exchange configuration and metadata
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

class ExchangeType(Enum):
    """Exchange types"""
    STOCK = "stock"
    CRYPTO = "crypto"
    FOREX = "forex"
    FUTURES = "futures"
    OPTIONS = "options"

class AssetType(Enum):
    """Asset types supported"""
    STOCK = "stock"
    CRYPTO = "crypto"
    ETF = "etf"
    OPTION = "option"
    FUTURE = "future"
    FOREX = "forex"

@dataclass
class ExchangeMetadata:
    """Exchange metadata and capabilities"""
    name: str
    display_name: str
    exchange_type: ExchangeType
    supported_assets: List[AssetType]
    supported_order_types: List[str]
    fee_structure: Dict[str, float]
    rate_limits: Dict[str, int]
    trading_hours: Dict[str, str]
    paper_trading: bool
    minimum_order_size: float
    countries_supported: List[str]
    api_documentation: str

# Exchange configurations
EXCHANGE_CONFIGS = {
    'alpaca': ExchangeMetadata(
        name='alpaca',
        display_name='Alpaca Markets',
        exchange_type=ExchangeType.STOCK,
        supported_assets=[AssetType.STOCK, AssetType.ETF, AssetType.CRYPTO, AssetType.OPTION],
        supported_order_types=['market', 'limit', 'stop', 'stop_limit', 'trailing_stop'],
        fee_structure={
            'stock_commission': 0.0,
            'option_commission': 0.65,
            'crypto_spread': 0.0025
        },
        rate_limits={
            'orders_per_minute': 200,
            'data_requests_per_minute': 200
        },
        trading_hours={
            'regular': '09:30-16:00 EST',
            'extended': '04:00-20:00 EST'
        },
        paper_trading=True,
        minimum_order_size=1.0,
        countries_supported=['US'],
        api_documentation='https://alpaca.markets/docs/'
    ),
    
    'binance': ExchangeMetadata(
        name='binance',
        display_name='Binance',
        exchange_type=ExchangeType.CRYPTO,
        supported_assets=[AssetType.CRYPTO],
        supported_order_types=['market', 'limit', 'stop_loss', 'stop_loss_limit', 'take_profit', 'take_profit_limit'],
        fee_structure={
            'maker_fee': 0.001,
            'taker_fee': 0.001,
            'bnb_discount': 0.25
        },
        rate_limits={
            'orders_per_second': 10,
            'requests_per_minute': 1200
        },
        trading_hours={
            'regular': '24/7',
            'extended': '24/7'
        },
        paper_trading=True,
        minimum_order_size=0.001,
        countries_supported=['Global (with restrictions)'],
        api_documentation='https://binance-docs.github.io/apidocs/'
    ),
    
    'td_ameritrade': ExchangeMetadata(
        name='td_ameritrade',
        display_name='TD Ameritrade',
        exchange_type=ExchangeType.STOCK,
        supported_assets=[AssetType.STOCK, AssetType.ETF, AssetType.OPTION, AssetType.FUTURE, AssetType.FOREX],
        supported_order_types=['market', 'limit', 'stop', 'stop_limit', 'trailing_stop', 'bracket'],
        fee_structure={
            'stock_commission': 0.0,
            'option_commission': 0.65,
            'future_commission': 2.25
        },
        rate_limits={
            'orders_per_minute': 120,
            'data_requests_per_minute': 120
        },
        trading_hours={
            'regular': '09:30-16:00 EST',
            'extended': '07:00-20:00 EST'
        },
        paper_trading=True,
        minimum_order_size=1.0,
        countries_supported=['US'],
        api_documentation='https://developer.tdameritrade.com/apis'
    ),
    
    'kucoin': ExchangeMetadata(
        name='kucoin',
        display_name='KuCoin',
        exchange_type=ExchangeType.CRYPTO,
        supported_assets=[AssetType.CRYPTO],
        supported_order_types=['market', 'limit', 'stop_loss', 'stop_limit'],
        fee_structure={
            'maker_fee': 0.001,
            'taker_fee': 0.001,
            'kcs_discount': 0.2
        },
        rate_limits={
            'orders_per_second': 10,
            'requests_per_minute': 100
        },
        trading_hours={
            'regular': '24/7',
            'extended': '24/7'
        },
        paper_trading=True,
        minimum_order_size=0.001,
        countries_supported=['Global (with restrictions)'],
        api_documentation='https://docs.kucoin.com/'
    )
}

# Trading pairs and symbols by exchange
EXCHANGE_SYMBOLS = {
    'alpaca': {
        'stocks': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD', 'NFLX', 'BABA'],
        'etfs': ['SPY', 'QQQ', 'IWM', 'VTI', 'VEA', 'VWO', 'BND', 'TLT', 'GLD', 'SLV'],
        'crypto': ['BTCUSD', 'ETHUSD', 'LTCUSD', 'BCHUSD', 'ADAUSD', 'DOTUSD']
    },
    'binance': {
        'crypto': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOTUSDT', 'XRPUSDT', 'LTCUSDT', 'BCHUSDT', 'LINKUSDT', 'SOLUSDT']
    },
    'td_ameritrade': {
        'stocks': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD', 'NFLX', 'BABA'],
        'etfs': ['SPY', 'QQQ', 'IWM', 'VTI', 'VEA', 'VWO', 'BND', 'TLT', 'GLD', 'SLV'],
        'futures': ['ES', 'NQ', 'YM', 'RTY', 'CL', 'GC', 'SI', 'ZN', 'ZB', 'ZF']
    },
    'kucoin': {
        'crypto': ['BTC-USDT', 'ETH-USDT', 'KCS-USDT', 'ADA-USDT', 'DOT-USDT', 'XRP-USDT', 'LTC-USDT', 'BCH-USDT', 'LINK-USDT', 'SOL-USDT']
    }
}

# Market data endpoints
MARKET_DATA_ENDPOINTS = {
    'alpaca': {
        'stocks': 'https://data.alpaca.markets/v2/stocks',
        'crypto': 'https://data.alpaca.markets/v1beta3/crypto',
        'options': 'https://data.alpaca.markets/v1beta1/options'
    },
    'binance': {
        'crypto': 'https://api.binance.com/api/v3'
    },
    'td_ameritrade': {
        'stocks': 'https://api.tdameritrade.com/v1/marketdata',
        'options': 'https://api.tdameritrade.com/v1/marketdata/chains',
        'futures': 'https://api.tdameritrade.com/v1/marketdata'
    },
    'kucoin': {
        'crypto': 'https://api.kucoin.com/api/v1'
    }
}

def get_exchange_config(exchange_name: str) -> ExchangeMetadata:
    """Get exchange configuration"""
    return EXCHANGE_CONFIGS.get(exchange_name.lower())

def get_supported_exchanges() -> List[str]:
    """Get list of supported exchanges"""
    return list(EXCHANGE_CONFIGS.keys())

def get_exchange_symbols(exchange_name: str) -> Dict[str, List[str]]:
    """Get supported symbols for an exchange"""
    return EXCHANGE_SYMBOLS.get(exchange_name.lower(), {})

def get_exchanges_by_asset(asset_type: AssetType) -> List[str]:
    """Get exchanges that support a specific asset type"""
    exchanges = []
    for exchange_name, config in EXCHANGE_CONFIGS.items():
        if asset_type in config.supported_assets:
            exchanges.append(exchange_name)
    return exchanges

def validate_exchange_symbol(exchange_name: str, symbol: str) -> bool:
    """Validate if a symbol is supported by an exchange"""
    symbols = get_exchange_symbols(exchange_name)
    for asset_symbols in symbols.values():
        if symbol.upper() in [s.upper() for s in asset_symbols]:
            return True
    return False
