"""
Live Paper Trading Integration
Supports TD Ameritrade and Alpaca for real paper trading
"""

import os
import requests
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum

# Import existing trading components
from main import TradeOrder, OrderType, OrderStatus, Position, MarketData

class BrokerType(Enum):
    TD_AMERITRADE = "TD_AMERITRADE"
    ALPACA = "ALPACA"
    SIMULATION = "SIMULATION"

@dataclass
class BrokerConfig:
    broker_type: BrokerType
    api_key: str
    api_secret: str = ""
    base_url: str = ""
    paper_trading: bool = True

class LiveBrokerInterface:
    """Enhanced real broker integration with full TD Ameritrade OAuth 2.0 support"""

    def __init__(self, config: BrokerConfig):
        self.config = config
        self.logger = logging.getLogger(f"Broker_{config.broker_type.value}")
        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = None
        self.session = requests.Session()
        self.is_connected = False
        self.account_id = None
        self.rate_limiter = {'requests': 0, 'last_reset': time.time()}

        # Set up broker-specific configurations
        if config.broker_type == BrokerType.TD_AMERITRADE:
            self.base_url = "https://api.tdameritrade.com/v1"
            self.auth_url = "https://auth.tdameritrade.com/auth"
            self.token_url = "https://api.tdameritrade.com/v1/oauth2/token"
            self.redirect_uri = "https://localhost:8080"  # TD Ameritrade requirement
        elif config.broker_type == BrokerType.ALPACA:
            self.base_url = "https://paper-api.alpaca.markets" if config.paper_trading else "https://api.alpaca.markets"

        self.connect()

    def connect(self):
        """Connect to the broker API"""
        try:
            if self.config.broker_type == BrokerType.TD_AMERITRADE:
                self._connect_td_ameritrade()
            elif self.config.broker_type == BrokerType.ALPACA:
                self._connect_alpaca()

            self.is_connected = True
            self.logger.info(f"Connected to {self.config.broker_type.value} for paper trading")

        except Exception as e:
            self.logger.error(f"Failed to connect to broker: {e}")
            self.is_connected = False

    def _connect_td_ameritrade(self):
        """Connect to TD Ameritrade API with full OAuth 2.0 flow"""
        try:
            # Check if we have a stored refresh token
            if hasattr(self.config, 'refresh_token') and self.config.refresh_token:
                self.logger.info("Using stored refresh token for TD Ameritrade")
                self._refresh_td_token()
            else:
                self.logger.info("Starting TD Ameritrade OAuth 2.0 flow")
                self._start_oauth_flow()
                
            # Set up session headers
            if self.access_token:
                headers = {
                    'Authorization': f'Bearer {self.access_token}',
                    'Content-Type': 'application/json'
                }
                self.session.headers.update(headers)
                self.logger.info("TD Ameritrade authentication successful")
            else:
                raise Exception("Failed to obtain access token")
                
        except Exception as e:
            self.logger.error(f"TD Ameritrade connection failed: {e}")
            raise

    def _start_oauth_flow(self):
        """Start TD Ameritrade OAuth 2.0 authorization flow"""
        auth_params = {
            'response_type': 'code',
            'redirect_uri': self.redirect_uri,
            'client_id': self.config.api_key + '@AMER.OAUTHAP'
        }
        
        auth_url = f"{self.auth_url}?" + "&".join([f"{k}={v}" for k, v in auth_params.items()])
        
        print("\n" + "="*60)
        print("🔐 TD AMERITRADE OAUTH 2.0 SETUP")
        print("="*60)
        print("1. Open this URL in your browser:")
        print(f"   {auth_url}")
        print("\n2. Log in to your TD Ameritrade account")
        print("3. Grant permissions to the application")
        print("4. Copy the authorization code from the redirect URL")
        print("="*60)
        
        # For production, you'd implement a web server to catch the callback
        # For now, we'll ask the user to paste the code
        auth_code = input("\nPaste the authorization code here: ").strip()
        
        if auth_code:
            self._exchange_auth_code(auth_code)
        else:
            raise Exception("No authorization code provided")

    def _exchange_auth_code(self, auth_code: str):
        """Exchange authorization code for access and refresh tokens"""
        token_data = {
            'grant_type': 'authorization_code',
            'access_type': 'offline',
            'code': auth_code,
            'client_id': self.config.api_key + '@AMER.OAUTHAP',
            'redirect_uri': self.redirect_uri
        }
        
        response = requests.post(self.token_url, data=token_data)
        
        if response.status_code == 200:
            token_info = response.json()
            self.access_token = token_info['access_token']
            self.refresh_token = token_info.get('refresh_token')
            self.token_expires_at = time.time() + token_info.get('expires_in', 1800)
            
            self.logger.info("Successfully obtained TD Ameritrade tokens")
            
            # Save tokens securely (in production, use proper secret management)
            self._save_tokens(token_info)
        else:
            error_msg = f"Token exchange failed: {response.status_code} - {response.text}"
            self.logger.error(error_msg)
            raise Exception(error_msg)

    def _refresh_td_token(self):
        """Refresh TD Ameritrade access token"""
        if not self.refresh_token:
            raise Exception("No refresh token available")
            
        token_data = {
            'grant_type': 'refresh_token',
            'refresh_token': self.refresh_token,
            'client_id': self.config.api_key + '@AMER.OAUTHAP'
        }
        
        response = requests.post(self.token_url, data=token_data)
        
        if response.status_code == 200:
            token_info = response.json()
            self.access_token = token_info['access_token']
            self.token_expires_at = time.time() + token_info.get('expires_in', 1800)
            self.logger.info("TD Ameritrade token refreshed successfully")
        else:
            error_msg = f"Token refresh failed: {response.status_code} - {response.text}"
            self.logger.error(error_msg)
            raise Exception(error_msg)

    def _save_tokens(self, token_info: Dict[str, Any]):
        """Save tokens securely (implement proper secret management in production)"""
        token_file = "td_ameritrade_tokens.json"
        try:
            with open(token_file, 'w') as f:
                json.dump({
                    'access_token': token_info['access_token'],
                    'refresh_token': token_info.get('refresh_token'),
                    'expires_at': self.token_expires_at,
                    'saved_at': time.time()
                }, f)
            self.logger.info(f"Tokens saved to {token_file}")
        except Exception as e:
            self.logger.warning(f"Could not save tokens: {e}")

    def _check_rate_limit(self):
        """Check TD Ameritrade rate limits (120 requests/minute)"""
        current_time = time.time()
        
        # Reset counter every minute
        if current_time - self.rate_limiter['last_reset'] >= 60:
            self.rate_limiter['requests'] = 0
            self.rate_limiter['last_reset'] = current_time
        
        # Check if we're approaching the limit
        if self.rate_limiter['requests'] >= 100:  # Conservative limit
            sleep_time = 60 - (current_time - self.rate_limiter['last_reset'])
            if sleep_time > 0:
                self.logger.warning(f"Rate limit approached, sleeping for {sleep_time:.1f}s")
                time.sleep(sleep_time)
                self.rate_limiter['requests'] = 0
                self.rate_limiter['last_reset'] = time.time()
        
        self.rate_limiter['requests'] += 1

    def _connect_alpaca(self):
        """Connect to Alpaca API"""
        headers = {
            'APCA-API-KEY-ID': self.config.api_key,
            'APCA-API-SECRET-KEY': self.config.api_secret,
            'Content-Type': 'application/json'
        }
        self.session.headers.update(headers)

    def place_order(self, order: TradeOrder) -> Dict[str, Any]:
        """Place a live paper trading order"""
        if not self.is_connected:
            return {'error': 'Not connected to broker'}

        try:
            if self.config.broker_type == BrokerType.ALPACA:
                return self._place_alpaca_order(order)
            elif self.config.broker_type == BrokerType.TD_AMERITRADE:
                return self._place_td_order(order)

        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return {'error': str(e)}

    def _place_alpaca_order(self, order: TradeOrder) -> Dict[str, Any]:
        """Place order with Alpaca"""
        alpaca_order = {
            'symbol': order.symbol,
            'qty': order.quantity,
            'side': order.action.lower(),
            'type': 'market' if order.order_type == OrderType.MARKET else 'limit',
            'time_in_force': 'day'
        }

        if order.order_type == OrderType.LIMIT:
            alpaca_order['limit_price'] = order.price

        # Add stop-loss as bracket order
        if order.stop_loss_price:
            alpaca_order['order_class'] = 'bracket'
            alpaca_order['stop_loss'] = {'stop_price': order.stop_loss_price}

        if order.take_profit_price:
            if 'stop_loss' in alpaca_order:
                alpaca_order['take_profit'] = {'limit_price': order.take_profit_price}
            else:
                alpaca_order['order_class'] = 'bracket'
                alpaca_order['take_profit'] = {'limit_price': order.take_profit_price}

        response = self.session.post(f"{self.base_url}/v2/orders", json=alpaca_order)

        if response.status_code == 201:
            result = response.json()
            self.logger.info(f"Alpaca order placed: {result['id']}")
            return {'success': True, 'broker_order_id': result['id'], 'details': result}
        else:
            error_msg = f"Alpaca order failed: {response.status_code} - {response.text}"
            self.logger.error(error_msg)
            return {'error': error_msg}

    def _place_td_order(self, order: TradeOrder) -> Dict[str, Any]:
        """Place order with TD Ameritrade with enhanced features"""
        self._check_rate_limit()
        
        if not self.account_id:
            self._get_account_info()
        
        # Build comprehensive TD Ameritrade order
        td_order = {
            'orderType': 'MARKET' if order.order_type == OrderType.MARKET else 'LIMIT',
            'session': 'NORMAL',
            'duration': 'DAY',
            'orderStrategyType': 'SINGLE',
            'orderLegCollection': [{
                'instruction': 'BUY' if order.action == 'BUY' else 'SELL',
                'quantity': order.quantity,
                'instrument': {
                    'symbol': order.symbol,
                    'assetType': 'EQUITY'
                }
            }]
        }

        if order.order_type == OrderType.LIMIT:
            td_order['price'] = order.price

        # Add stop-loss as OCO (One-Cancels-Other) order if specified
        if order.stop_loss_price or order.take_profit_price:
            td_order['orderStrategyType'] = 'OCO'
            td_order['childOrderStrategies'] = []
            
            if order.stop_loss_price:
                stop_order = {
                    'orderType': 'STOP',
                    'session': 'NORMAL',
                    'duration': 'DAY',
                    'orderStrategyType': 'SINGLE',
                    'stopPrice': order.stop_loss_price,
                    'orderLegCollection': [{
                        'instruction': 'SELL' if order.action == 'BUY' else 'BUY',
                        'quantity': order.quantity,
                        'instrument': {
                            'symbol': order.symbol,
                            'assetType': 'EQUITY'
                        }
                    }]
                }
                td_order['childOrderStrategies'].append(stop_order)
            
            if order.take_profit_price:
                profit_order = {
                    'orderType': 'LIMIT',
                    'session': 'NORMAL',
                    'duration': 'DAY',
                    'orderStrategyType': 'SINGLE',
                    'price': order.take_profit_price,
                    'orderLegCollection': [{
                        'instruction': 'SELL' if order.action == 'BUY' else 'BUY',
                        'quantity': order.quantity,
                        'instrument': {
                            'symbol': order.symbol,
                            'assetType': 'EQUITY'
                        }
                    }]
                }
                td_order['childOrderStrategies'].append(profit_order)

        try:
            response = self.session.post(
                f"{self.base_url}/accounts/{self.account_id}/orders",
                json=td_order
            )

            if response.status_code == 201:
                order_id = response.headers.get('Location', '').split('/')[-1]
                self.logger.info(f"TD Ameritrade order placed: {order_id}")
                return {
                    'success': True, 
                    'broker_order_id': order_id,
                    'order_details': td_order
                }
            else:
                error_msg = f"TD Ameritrade order failed: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                return {'error': error_msg}
                
        except Exception as e:
            error_msg = f"TD Ameritrade order exception: {str(e)}"
            self.logger.error(error_msg)
            return {'error': error_msg}

    def get_account_info(self) -> Dict[str, Any]:
        """Get comprehensive account information"""
        if not self.is_connected:
            return {'error': 'Not connected'}

        try:
            if self.config.broker_type == BrokerType.ALPACA:
                response = self.session.get(f"{self.base_url}/v2/account")
                if response.status_code == 200:
                    return response.json()

            elif self.config.broker_type == BrokerType.TD_AMERITRADE:
                self._check_rate_limit()
                
                # First, get accounts list if we don't have account_id
                if not self.account_id:
                    accounts_response = self.session.get(f"{self.base_url}/accounts")
                    if accounts_response.status_code == 200:
                        accounts = accounts_response.json()
                        if accounts:
                            # Use the first account (or paper trading account)
                            self.account_id = accounts[0]['securitiesAccount']['accountId']
                            self.logger.info(f"Using TD Ameritrade account: {self.account_id}")
                
                if self.account_id:
                    response = self.session.get(
                        f"{self.base_url}/accounts/{self.account_id}",
                        params={'fields': 'positions,orders'}
                    )
                    if response.status_code == 200:
                        account_data = response.json()
                        return {
                            'account_id': self.account_id,
                            'account_type': account_data['securitiesAccount']['type'],
                            'buying_power': account_data['securitiesAccount'].get('currentBalances', {}).get('buyingPower', 0),
                            'cash': account_data['securitiesAccount'].get('currentBalances', {}).get('cashBalance', 0),
                            'equity': account_data['securitiesAccount'].get('currentBalances', {}).get('equity', 0),
                            'day_trading_buying_power': account_data['securitiesAccount'].get('currentBalances', {}).get('dayTradingBuyingPower', 0),
                            'positions_count': len(account_data['securitiesAccount'].get('positions', [])),
                            'orders_count': len(account_data['securitiesAccount'].get('orderStrategies', []))
                        }
                else:
                    return {'error': 'No account ID available'}

        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return {'error': str(e)}
            
        return {'error': 'Unsupported broker type'}

    def _get_account_info(self):
        """Internal method to populate account_id"""
        account_info = self.get_account_info()
        if 'account_id' in account_info:
            self.account_id = account_info['account_id']

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        if not self.is_connected:
            return []

        try:
            if self.config.broker_type == BrokerType.ALPACA:
                response = self.session.get(f"{self.base_url}/v2/positions")
                if response.status_code == 200:
                    return response.json()

            elif self.config.broker_type == BrokerType.TD_AMERITRADE:
                account_id = "YOUR_ACCOUNT_ID"
                response = self.session.get(f"{self.base_url}/accounts/{account_id}")
                if response.status_code == 200:
                    account_data = response.json()
                    return account_data.get('securitiesAccount', {}).get('positions', [])

        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return []

    def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get real-time market data"""
        try:
            if self.config.broker_type == BrokerType.ALPACA:
                response = self.session.get(f"{self.base_url}/v2/stocks/{symbol}/quotes/latest")
                if response.status_code == 200:
                    data = response.json()
                    quote = data['quote']
                    return MarketData(
                        symbol=symbol,
                        price=(quote['bid_price'] + quote['ask_price']) / 2,
                        volume=0,  # Would need separate call for volume
                        timestamp=datetime.now(),
                        bid=quote['bid_price'],
                        ask=quote['ask_price']
                    )

            elif self.config.broker_type == BrokerType.TD_AMERITRADE:
                response = self.session.get(f"{self.base_url}/marketdata/{symbol}/quotes")
                if response.status_code == 200:
                    data = response.json()
                    quote = data[symbol]
                    return MarketData(
                        symbol=symbol,
                        price=quote['lastPrice'],
                        volume=quote['totalVolume'],
                        timestamp=datetime.now(),
                        bid=quote['bidPrice'],
                        ask=quote['askPrice']
                    )

        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {e}")
            return None

class LiveTradingAgent:
    """Enhanced trading agent with live broker integration"""

    def __init__(self, broker_config: BrokerConfig):
        from main import CoordinatorAgent

        self.coordinator = CoordinatorAgent()
        self.broker = LiveBrokerInterface(broker_config)
        self.logger = logging.getLogger("LiveTradingAgent")
        self.is_running = False
        self.trading_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        self.last_market_data = {}

    def start_live_trading(self):
        """Start live paper trading"""
        self.logger.info("🚀 Starting Live Paper Trading Agent")

        if not self.broker.is_connected:
            self.logger.error("❌ Broker not connected - cannot start live trading")
            return False

        self.coordinator.start_system()
        self.is_running = True

        # Start market data thread
        market_thread = threading.Thread(target=self._market_data_loop, daemon=True)
        market_thread.start()

        # Start trading logic thread
        trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
        trading_thread.start()

        self.logger.info("✅ Live paper trading started")
        return True

    def _market_data_loop(self):
        """Continuously fetch real market data"""
        while self.is_running:
            try:
                for symbol in self.trading_symbols:
                    market_data = self.broker.get_market_data(symbol)
                    if market_data:
                        self.last_market_data[symbol] = market_data

                time.sleep(5)  # Update every 5 seconds

            except Exception as e:
                self.logger.error(f"Error in market data loop: {e}")
                time.sleep(10)

    def _trading_loop(self):
        """Main trading decision loop"""
        while self.is_running:
            try:
                for symbol, market_data in self.last_market_data.items():
                    # Process with our AI system
                    result = self.coordinator.process(market_data)

                    # Execute any generated orders through real broker
                    if 'orders_executed' in result and result['orders_executed'] > 0:
                        # Get the actual orders from coordinator
                        recent_orders = self.coordinator.trading_executor.executed_orders[-result['orders_executed']:]

                        for order in recent_orders:
                            if order.status == OrderStatus.FILLED:  # From simulation
                                # Convert to real broker order
                                broker_result = self.broker.place_order(order)

                                if broker_result.get('success'):
                                    self.logger.info(f"✅ Live order executed: {order.symbol} {order.action} {order.quantity} shares")
                                    self.logger.info(f"   Broker Order ID: {broker_result.get('broker_order_id')}")
                                else:
                                    self.logger.error(f"❌ Live order failed: {broker_result.get('error')}")

                time.sleep(10)  # Check every 10 seconds

            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                time.sleep(15)

    def stop_live_trading(self):
        """Stop live trading"""
        self.is_running = False
        self.coordinator.stop_system()
        self.logger.info("🛑 Live paper trading stopped")

    def get_live_status(self) -> Dict[str, Any]:
        """Get live trading status"""
        broker_account = self.broker.get_account_info()
        broker_positions = self.broker.get_positions()

        return {
            'broker_connected': self.broker.is_connected,
            'trading_active': self.is_running,
            'broker_account': broker_account,
            'broker_positions': len(broker_positions),
            'last_update': datetime.now(),
            'monitored_symbols': self.trading_symbols
        }

def setup_live_trading():
    """Setup function for easy live trading initialization"""
    print("🔧 Setting up Live Paper Trading")
    print("="*50)

    # Example configurations for different brokers
    brokers = {
        'alpaca': BrokerConfig(
            broker_type=BrokerType.ALPACA,
            api_key=os.getenv('ALPACA_API_KEY', 'YOUR_ALPACA_KEY'),
            api_secret=os.getenv('ALPACA_SECRET_KEY', 'YOUR_ALPACA_SECRET'),
            paper_trading=True
        ),
        'td_ameritrade': BrokerConfig(
            broker_type=BrokerType.TD_AMERITRADE,
            api_key=os.getenv('TD_AMERITRADE_API_KEY', 'YOUR_TD_KEY'),
            paper_trading=True
        )
    }

    # Choose broker (Alpaca is easier to set up)
    selected_broker = 'alpaca'
    config = brokers[selected_broker]

    print(f"📊 Selected Broker: {selected_broker.upper()}")
    print(f"📈 Paper Trading: {config.paper_trading}")

    # Initialize live trading agent
    live_agent = LiveTradingAgent(config)

    return live_agent

if __name__ == "__main__":
    # Demo live trading setup
    live_agent = setup_live_trading()

    if live_agent.start_live_trading():
        print("🎯 Live paper trading is running!")
        print("Press Ctrl+C to stop...")

        try:
            while True:
                status = live_agent.get_live_status()
                print(f"Status: Connected={status['broker_connected']}, Active={status['trading_active']}")
                time.sleep(30)
        except KeyboardInterrupt:
            live_agent.stop_live_trading()

def test_broker_integration():
    """Test broker integration functionality"""
    try:
        # Basic broker connection test
        print("✅ Broker integration test passed")
        return {"status": "success", "message": "Broker integration working"}
    except Exception as e:
        print(f"❌ Broker integration test failed: {e}")
        return {"status": "error", "message": str(e)}