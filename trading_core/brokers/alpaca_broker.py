"""
Streamlined Alpaca Paper Trading Integration
Connects your AI trading system to real Alpaca paper trading account
"""

import os
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ..data_models import TradeOrder, Position, MarketData, OrderType, OrderStatus

# Alpaca SDK
try:
    from alpaca.trading.client import TradingClient
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    ALPACA_AVAILABLE = True
    print("✅ Alpaca SDK (alpaca-py) loaded successfully!")
except ImportError:
    print("⚠️ Alpaca SDK not available - install with: pip install alpaca-py")
    ALPACA_AVAILABLE = False

@dataclass
class AlpacaConfig:
    """Alpaca configuration"""
    api_key: str
    secret_key: str
    paper: bool = True
    base_url: str = "https://paper-api.alpaca.markets"

class AlpacaBroker:
    """Streamlined Alpaca broker integration for paper trading"""
    
    def __init__(self, config: Optional[AlpacaConfig] = None):
        self.logger = logging.getLogger("AlpacaBroker")
        
        # Load configuration
        if config:
            self.config = config
        else:
            self.config = self._load_config_from_env()
        
        # Initialize clients
        self.trading_client = None
        self.data_client = None
        self.is_connected = False
        
        if ALPACA_AVAILABLE:
            self._initialize_clients()
        
    def _load_config_from_env(self) -> AlpacaConfig:
        """Load Alpaca configuration from environment variables"""
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY') 
        
        if not api_key or not secret_key:
            self.logger.warning("Alpaca API credentials not found in environment")
            # Return dummy config for demo mode
            return AlpacaConfig(
                api_key="DEMO_KEY",
                secret_key="DEMO_SECRET",
                paper=True
            )
        
        return AlpacaConfig(
            api_key=api_key,
            secret_key=secret_key,
            paper=True,  # Always use paper trading for safety
            base_url="https://paper-api.alpaca.markets"
        )
    
    def _initialize_clients(self):
        """Initialize Alpaca trading and data clients"""
        try:
            # Trading client for orders and positions
            self.trading_client = TradingClient(
                api_key=self.config.api_key,
                secret_key=self.config.secret_key,
                paper=self.config.paper
            )
            
            # Data client for market data
            self.data_client = StockHistoricalDataClient(
                api_key=self.config.api_key,
                secret_key=self.config.secret_key
            )
            
            # Test connection
            account = self.trading_client.get_account()
            self.is_connected = True
            
            self.logger.info(f"✅ Connected to Alpaca Paper Trading")
            self.logger.info(f"💰 Account Value: ${float(account.portfolio_value):,.2f}")
            self.logger.info(f"💵 Buying Power: ${float(account.buying_power):,.2f}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Alpaca: {e}")
            self.is_connected = False
    
    def place_order(self, signal_order: TradeOrder) -> bool:
        """Place order through Alpaca"""
        if not self.is_connected or not ALPACA_AVAILABLE:
            self.logger.warning("Alpaca not connected - using simulation mode")
            return self._simulate_order_execution(signal_order)
        
        try:
            # Convert to Alpaca order
            if signal_order.action == 'BUY':
                side = OrderSide.BUY
            elif signal_order.action == 'SELL':
                side = OrderSide.SELL
            else:
                self.logger.warning(f"Unknown order action: {signal_order.action}")
                return False
            
            # Create market order (for simplicity)
            order_request = MarketOrderRequest(
                symbol=signal_order.symbol,
                qty=signal_order.quantity,
                side=side,
                time_in_force=TimeInForce.DAY
            )
            
            # Submit order
            alpaca_order = self.trading_client.submit_order(order_request)
            
            # Update our order with Alpaca's order ID
            signal_order.order_id = alpaca_order.id
            signal_order.status = OrderStatus.FILLED  # Market orders typically fill immediately
            
            self.logger.info(f"✅ Order placed: {signal_order.action} {signal_order.quantity} {signal_order.symbol}")
            self.logger.info(f"   Alpaca Order ID: {alpaca_order.id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Order placement failed: {e}")
            signal_order.status = OrderStatus.REJECTED
            return False
    
    def _simulate_order_execution(self, order: TradeOrder) -> bool:
        """Simulate order execution when Alpaca not available"""
        self.logger.info(f"🎭 SIMULATED: {order.action} {order.quantity} {order.symbol}")
        order.status = OrderStatus.FILLED
        order.avg_fill_price = order.price
        return True
    
    def get_positions(self) -> Dict[str, Position]:
        """Get current positions from Alpaca"""
        if not self.is_connected or not ALPACA_AVAILABLE:
            return {}
        
        try:
            alpaca_positions = self.trading_client.get_all_positions()
            positions = {}
            
            for pos in alpaca_positions:
                position = Position(
                    symbol=pos.symbol,
                    quantity=int(pos.qty),
                    avg_price=float(pos.avg_entry_price),
                    current_price=float(pos.market_value) / int(pos.qty) if int(pos.qty) != 0 else 0,
                    unrealized_pnl=float(pos.unrealized_pl),
                    realized_pnl=0.0,  # Alpaca doesn't provide this directly
                    entry_timestamp=datetime.now()
                )
                positions[pos.symbol] = position
            
            return positions
            
        except Exception as e:
            self.logger.error(f"Error fetching positions: {e}")
            return {}
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information from Alpaca"""
        if not self.is_connected or not ALPACA_AVAILABLE:
            return {
                'cash': 100000.0,
                'portfolio_value': 100000.0,
                'buying_power': 100000.0,
                'status': 'SIMULATION'
            }
        
        try:
            account = self.trading_client.get_account()
            
            return {
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'buying_power': float(account.buying_power),
                'day_trade_count': int(account.daytrade_count),
                'status': account.status,
                'pattern_day_trader': account.pattern_day_trader,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching account info: {e}")
            return {}
    
    def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get real-time market data from Alpaca"""
        if not self.is_connected or not ALPACA_AVAILABLE:
            return self._simulate_market_data(symbol)
        
        try:
            # Get latest quote
            request = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
            quotes = self.data_client.get_stock_latest_quote(request)
            
            if symbol in quotes:
                quote = quotes[symbol]
                
                return MarketData(
                    symbol=symbol,
                    price=(quote.bid_price + quote.ask_price) / 2,  # Mid price
                    volume=0,  # Quote doesn't include volume
                    timestamp=quote.timestamp,
                    bid=quote.bid_price,
                    ask=quote.ask_price,
                    high_24h=0.0,  # Would need separate bars request
                    low_24h=0.0
                )
            
        except Exception as e:
            self.logger.error(f"Error fetching market data for {symbol}: {e}")
            
        return self._simulate_market_data(symbol)
    
    def _simulate_market_data(self, symbol: str) -> MarketData:
        """Simulate market data when Alpaca not available"""
        import random
        base_price = 100.0
        price = base_price + random.uniform(-5, 5)
        
        return MarketData(
            symbol=symbol,
            price=price,
            volume=random.randint(1000, 10000),
            timestamp=datetime.now(),
            bid=price - 0.01,
            ask=price + 0.01,
            high_24h=price * 1.02,
            low_24h=price * 0.98
        )
    
    def get_order_history(self) -> List[Dict[str, Any]]:
        """Get order history from Alpaca"""
        if not self.is_connected or not ALPACA_AVAILABLE:
            return []
        
        try:
            orders = self.trading_client.get_orders()
            order_history = []
            
            for order in orders[:10]:  # Last 10 orders
                order_history.append({
                    'id': order.id,
                    'symbol': order.symbol,
                    'side': order.side.value,
                    'quantity': int(order.qty),
                    'status': order.status.value,
                    'filled_qty': int(order.filled_qty) if order.filled_qty else 0,
                    'created_at': order.created_at.isoformat() if order.created_at else None
                })
            
            return order_history
            
        except Exception as e:
            self.logger.error(f"Error fetching order history: {e}")
            return []
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if not self.is_connected or not ALPACA_AVAILABLE:
            self.logger.info(f"🎭 SIMULATED: Cancel order {order_id}")
            return True
        
        try:
            self.trading_client.cancel_order_by_id(order_id)
            self.logger.info(f"✅ Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get broker connection status"""
        return {
            'broker': 'Alpaca',
            'connected': self.is_connected,
            'paper_trading': self.config.paper,
            'sdk_available': ALPACA_AVAILABLE,
            'base_url': self.config.base_url,
            'last_check': datetime.now().isoformat()
        }

def create_alpaca_broker() -> AlpacaBroker:
    """Factory function to create Alpaca broker"""
    return AlpacaBroker()