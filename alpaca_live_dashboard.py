#!/usr/bin/env python3
"""
Alpaca Paper Trading with Live Web Dashboard
Real positions, charts, and portfolio tracking
"""

import os
import json
import time
import threading
import random
from datetime import datetime
from flask import Flask, render_template, jsonify
import requests
# Using local SimplePaperTrading class instead of alternative_brokers

# Import new Alpaca SDK
try:
    from alpaca.trading.client import TradingClient
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.live import StockDataStream
    from alpaca.data.requests import StockLatestQuoteRequest
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    ALPACA_SDK_AVAILABLE = True
    print("✅ New Alpaca SDK (alpaca-py) loaded successfully!")
except ImportError:
    print("⚠️ New Alpaca SDK not available - using legacy mode")
    ALPACA_SDK_AVAILABLE = False

# Simple paper trading system (since alternative_brokers is missing)
class QuickPaperTradingSystem:
    def __init__(self, broker_type=None):
        self.broker = SimplePaperTrading()

class AlternativeBrokerType:
    YAHOO_FINANCE = "yahoo_finance"

class SimplePaperTrading:
    def __init__(self):
        self.positions = {}
        self.executed_orders = []
        self.cash_balance = 100000.0
        self.initial_balance = 100000.0
        self.trade_count = 0

    def get_real_time_quote(self, symbol):
        import random
        # Use persistent price tracking for more realistic simulation
        if not hasattr(self, '_price_cache'):
            self._price_cache = {}
        
        if symbol not in self._price_cache:
            base_prices = {'AAPL': 175.0, 'GOOGL': 135.0, 'MSFT': 340.0, 'TSLA': 240.0, 'NVDA': 450.0}
            self._price_cache[symbol] = base_prices.get(symbol, 100.0)
        
        # Add some price movement
        price_change = random.uniform(-0.02, 0.02)
        self._price_cache[symbol] *= (1 + price_change)
        
        # Keep prices reasonable
        self._price_cache[symbol] = max(50, min(500, self._price_cache[symbol]))
        
        change_percent = price_change * 100

        return {
            'price': round(self._price_cache[symbol], 2),
            'change_percent': round(change_percent, 2),
            'symbol': symbol
        }

    def place_paper_order(self, order):
        try:
            self.executed_orders.append(order)
            self.trade_count += 1
            return True
        except:
            return False

    def get_portfolio_summary(self):
        total_value = self.cash_balance
        position_list = []
        
        for symbol, quantity in self.positions.items():
            if quantity > 0:
                quote = self.get_real_time_quote(symbol)
                position_value = quantity * quote['price']
                total_value += position_value
                
                position_list.append({
                    'symbol': symbol,
                    'quantity': quantity,
                    'current_price': quote['price'],
                    'value': position_value,
                    'change_percent': quote.get('change_percent', 0)
                })
        
        total_return_pct = ((total_value - self.initial_balance) / self.initial_balance) * 100
        
        return {
            'cash_balance': self.cash_balance,
            'total_value': total_value,
            'total_return_percent': total_return_pct,
            'positions': position_list,
            'trade_count': self.trade_count
        }

class AlpacaLiveDashboard:
    def __init__(self):
        self.app = Flask(__name__)
        self.trading_system = None
        self.setup_routes()

        # Check for Alpaca keys
        self.alpaca_key = os.getenv('ALPACA_API_KEY', '')
        self.alpaca_secret = os.getenv('ALPACA_SECRET_KEY', '')
        self.has_alpaca_keys = bool(self.alpaca_key and self.alpaca_secret)

        if self.has_alpaca_keys:
            print("✅ Alpaca API keys found - connecting to live paper trading")
            self.setup_alpaca_connection()
        else:
            print("📝 No Alpaca keys - using simulation mode")
            print("🔗 Get free keys at: https://alpaca.markets/")
            self.setup_simulation_mode()

    def setup_alpaca_connection(self):
        """Setup Alpaca paper trading connection using new SDK"""
        try:
            if ALPACA_SDK_AVAILABLE:
                # Initialize new Alpaca SDK clients
                self.trading_client = TradingClient(
                    api_key=self.alpaca_key,
                    secret_key=self.alpaca_secret,
                    paper=True  # Enable paper trading
                )
                
                self.data_client = StockHistoricalDataClient(
                    api_key=self.alpaca_key,
                    secret_key=self.alpaca_secret
                )

                # Test connection with new SDK
                account = self.trading_client.get_account()
                print(f"🎯 Connected to Alpaca Paper Trading (New SDK)")
                print(f"💰 Buying Power: ${float(account.buying_power):,.2f}")
                print(f"📊 Portfolio Value: ${float(account.portfolio_value):,.2f}")
                self.mode = "LIVE"
            else:
                # Fallback to legacy method
                self.base_url = "https://paper-api.alpaca.markets/v2"
                self.headers = {
                    'APCA-API-KEY-ID': self.alpaca_key,
                    'APCA-API-SECRET-KEY': self.alpaca_secret,
                    'Content-Type': 'application/json'
                }

                # Test connection
                response = requests.get(f"{self.base_url}/account", headers=self.headers)
                if response.status_code == 200:
                    account = response.json()
                    print(f"🎯 Connected to Alpaca Paper Trading (Legacy)")
                    print(f"💰 Buying Power: ${float(account['buying_power']):,.2f}")
                    self.mode = "LIVE"
                else:
                    raise Exception(f"Legacy connection failed: {response.status_code}")

        except Exception as e:
            print(f"❌ Alpaca connection failed: {e} - using simulation")
            print(f"   Error details: {type(e).__name__}")
            if hasattr(e, 'response'):
                print(f"   HTTP Status: {e.response.status_code if hasattr(e.response, 'status_code') else 'Unknown'}")
            self.setup_simulation_mode()

    def setup_simulation_mode(self):
        """Setup simulation trading"""
        self.trading_system = QuickPaperTradingSystem(AlternativeBrokerType.YAHOO_FINANCE)
        self.mode = "SIMULATION"

        # Start background trading
        self.trading_thread = threading.Thread(target=self._simulation_trading_loop, daemon=True)
        self.trading_thread.start()

    def _simulation_trading_loop(self):
        """Background simulation trading"""
        # Enhanced momentum trading with multiple strategies
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        trade_count = 0

        while True:
            try:
                trade_count += 1

                for symbol in symbols:
                    quote = self.trading_system.broker.get_real_time_quote(symbol)
                    change_pct = quote.get('change_percent', 0)
                    current_price = quote['price']
                    position = self.trading_system.broker.positions.get(symbol, 0)

                    # Strategy 1: Strong momentum (buy on 3%+ gains)
                    if change_pct > 3.0 and position < 20:
                        from main import TradeOrder, OrderType
                        order = TradeOrder(
                            symbol=symbol,
                            action='BUY',
                            quantity=10,
                            order_type=OrderType.MARKET,
                            price=current_price
                        )
                        self.trading_system.broker.place_paper_order(order)
                        print(f"🚀 Strong momentum BUY: {symbol} at ${current_price:.2f} (+{change_pct:.1f}%)")

                    # Strategy 2: Moderate momentum (smaller positions)
                    elif change_pct > 1.5 and position < 10:
                        from main import TradeOrder, OrderType
                        order = TradeOrder(
                            symbol=symbol,
                            action='BUY',
                            quantity=5,
                            order_type=OrderType.MARKET,
                            price=current_price
                        )
                        self.trading_system.broker.place_paper_order(order)
                        print(f"📈 Momentum BUY: {symbol} at ${current_price:.2f} (+{change_pct:.1f}%)")

                    # Strategy 3: Take profits on gains
                    elif change_pct < -1.5 and position > 0:
                        sell_qty = min(position, 10)
                        from main import TradeOrder, OrderType
                        order = TradeOrder(
                            symbol=symbol,
                            action='SELL',
                            quantity=sell_qty,
                            order_type=OrderType.MARKET,
                            price=current_price
                        )
                        self.trading_system.broker.place_paper_order(order)
                        print(f"🔻 Stop loss SELL: {symbol} at ${current_price:.2f} ({change_pct:.1f}%)")

                    # Strategy 4: Profit taking on strong positions
                    elif position > 15 and change_pct > 0:
                        sell_qty = min(5, position)
                        from main import TradeOrder, OrderType
                        order = TradeOrder(
                            symbol=symbol,
                            action='SELL',
                            quantity=sell_qty,
                            order_type=OrderType.MARKET,
                            price=current_price
                        )
                        self.trading_system.broker.place_paper_order(order)
                        print(f"💰 Profit taking: {symbol} at ${current_price:.2f} ({sell_qty} shares)")

                # Print status every 5 trades
                if trade_count % 5 == 0:
                    portfolio = self.trading_system.broker.get_portfolio_summary()
                    print(f"📊 Portfolio Update #{trade_count}: ${portfolio['total_value']:,.2f} "
                          f"({portfolio['total_return_percent']:+.2f}%) | "
                          f"Positions: {len([p for p in portfolio.get('positions', []) if p['quantity'] > 0])}")

                time.sleep(15)  # Trade every 15 seconds for more action

            except Exception as e:
                print(f"Trading loop error: {e}")
                time.sleep(60)

    def _get_price_change_percent(self, symbol, current_price):
        """Calculate real price change percentage using yesterday's close"""
        try:
            if ALPACA_SDK_AVAILABLE and hasattr(self, 'data_client'):
                from alpaca.data.requests import StockBarsRequest
                from alpaca.data.timeframe import TimeFrame
                from datetime import timedelta
                
                # Get yesterday's data
                end_date = datetime.now().date()
                start_date = end_date - timedelta(days=5)  # Get a few days to ensure we have data
                
                request_params = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Day,
                    start=start_date,
                    end=end_date
                )
                
                bars = self.data_client.get_stock_bars(request_params)
                
                if symbol in bars and len(bars[symbol]) > 0:
                    # Get the most recent bar (yesterday's close)
                    latest_bar = bars[symbol][-1]
                    previous_close = float(latest_bar.close)
                    
                    change_percent = ((current_price - previous_close) / previous_close) * 100
                    return round(change_percent, 2)
                    
        except Exception as e:
            print(f"Could not get historical data for {symbol}: {e}")
            
        # Fallback to random change
        return round(random.uniform(-3.0, 3.0), 2)


                    

    def setup_routes(self):
        """Setup Flask routes"""

        @self.app.route('/')
        def dashboard():
            return render_template('dashboard.html')

        @self.app.route('/api/account')
        def get_account():
            if self.mode == "LIVE":
                return self._get_alpaca_account()
            else:
                return self._get_simulation_account()

        @self.app.route('/api/positions')
        def get_positions():
            if self.mode == "LIVE":
                return self._get_alpaca_positions()
            else:
                return self._get_simulation_positions()

        @self.app.route('/api/orders')
        def get_orders():
            if self.mode == "LIVE":
                return self._get_alpaca_orders()
            else:
                return self._get_simulation_orders()

        @self.app.route('/api/status')
        def get_api_status():
            """Get API connection status"""
            status = {
                'mode': self.mode,
                'alpaca_connected': self.has_alpaca_keys and hasattr(self, 'data_client'),
                'data_client_available': ALPACA_SDK_AVAILABLE and hasattr(self, 'data_client'),
                'trading_client_available': ALPACA_SDK_AVAILABLE and hasattr(self, 'trading_client'),
                'timestamp': datetime.now().isoformat()
            }
            
            # Test a quick quote if connected
            if status['data_client_available']:
                try:
                    # Try a quick test quote
                    from alpaca.data.requests import StockLatestQuoteRequest
                    request_params = StockLatestQuoteRequest(symbol_or_symbols='AAPL')
                    test_quotes = self.data_client.get_stock_latest_quote(request_params)
                    status['last_quote_test'] = 'SUCCESS'
                    status['test_quote_time'] = datetime.now().isoformat()
                except Exception as e:
                    status['last_quote_test'] = f'FAILED: {str(e)}'
                    
            return jsonify(status)

        @self.app.route('/api/quotes/<symbol>')
        def get_quote(symbol):
            if self.mode == "LIVE":
                return self._get_alpaca_quote(symbol)
            else:
                return self._get_simulation_quote(symbol)

    def _get_alpaca_account(self):
        """Get Alpaca account info using new SDK"""
        try:
            if ALPACA_SDK_AVAILABLE and hasattr(self, 'trading_client'):
                # Use new SDK
                account = self.trading_client.get_account()
                return jsonify({
                    'buying_power': float(account.buying_power),
                    'cash': float(account.cash),
                    'portfolio_value': float(account.portfolio_value),
                    'equity': float(account.equity),
                    'day_trade_count': int(account.daytrade_count),
                    'mode': 'LIVE_PAPER_NEW_SDK'
                })
            else:
                # Legacy fallback
                response = requests.get(f"{self.base_url}/account", headers=self.headers)
                if response.status_code == 200:
                    account = response.json()
                    return jsonify({
                        'buying_power': float(account['buying_power']),
                        'cash': float(account['cash']),
                        'portfolio_value': float(account['portfolio_value']),
                        'equity': float(account['equity']),
                        'day_trade_count': int(account['daytrade_count']),
                        'mode': 'LIVE_PAPER_LEGACY'
                    })
        except Exception as e:
            return jsonify({'error': str(e)})

    def _get_alpaca_positions(self):
        """Get Alpaca positions using new SDK"""
        try:
            if ALPACA_SDK_AVAILABLE and hasattr(self, 'trading_client'):
                # Use new SDK
                positions = self.trading_client.get_all_positions()
                formatted = []
                for pos in positions:
                    formatted.append({
                        'symbol': pos.symbol,
                        'quantity': int(pos.qty),
                        'market_value': float(pos.market_value),
                        'unrealized_pl': float(pos.unrealized_pl),
                        'unrealized_plpc': float(pos.unrealized_plpc) * 100,
                        'avg_entry_price': float(pos.avg_entry_price),
                        'side': pos.side.value
                    })
                return jsonify(formatted)
            else:
                # Legacy fallback
                response = requests.get(f"{self.base_url}/positions", headers=self.headers)
                if response.status_code == 200:
                    positions = response.json()
                    formatted = []
                    for pos in positions:
                        formatted.append({
                            'symbol': pos['symbol'],
                            'quantity': int(pos['qty']),
                            'market_value': float(pos['market_value']),
                            'unrealized_pl': float(pos['unrealized_pl']),
                            'unrealized_plpc': float(pos['unrealized_plpc']) * 100,
                            'avg_entry_price': float(pos['avg_entry_price'])
                        })
                    return jsonify(formatted)
        except Exception as e:
            return jsonify({'error': str(e)})

    def _get_alpaca_orders(self):
        """Get Alpaca orders"""
        try:
            if ALPACA_SDK_AVAILABLE and hasattr(self, 'trading_client'):
                # Use new SDK
                orders = self.trading_client.get_orders()
                formatted = []
                for order in orders:
                    formatted.append({
                        'id': order.id,
                        'symbol': order.symbol,
                        'side': order.side.value,
                        'qty': int(order.qty),
                        'filled_qty': int(order.filled_qty or 0),
                        'status': order.status.value,
                        'submitted_at': order.submitted_at.isoformat()
                    })
                return jsonify(formatted)
            else:
                # Legacy API
                response = requests.get(f"{self.base_url}/orders", headers=self.headers)
                if response.status_code == 200:
                    orders = response.json()
                    formatted = []
                    for order in orders:
                        formatted.append({
                            'id': order['id'],
                            'symbol': order['symbol'],
                            'side': order['side'],
                            'qty': int(order['qty']),
                            'filled_qty': int(order['filled_qty']),
                            'status': order['status'],
                            'submitted_at': order['submitted_at']
                        })
                    return jsonify(formatted)
                else:
                    return jsonify([])
        except Exception as e:
            print(f"Error getting Alpaca orders: {str(e)}")
            # Always return empty array on error to prevent JavaScript errors
            return jsonify([])

    def _get_alpaca_quote(self, symbol):
        """Get Alpaca real-time quote using new SDK"""
        try:
            if ALPACA_SDK_AVAILABLE and hasattr(self, 'data_client'):
                # Try to get real quote using new Alpaca SDK
                print(f"Fetching live quote for {symbol} from Alpaca...")
                
                from alpaca.data.requests import StockLatestQuoteRequest
                
                # Create quote request
                request_params = StockLatestQuoteRequest(symbol_or_symbols=symbol)
                
                # Get latest quote
                quotes = self.data_client.get_stock_latest_quote(request_params)
                
                if symbol in quotes:
                    quote = quotes[symbol]
                    
                    # Calculate mid price
                    bid_price = float(quote.bid_price) if quote.bid_price else 0
                    ask_price = float(quote.ask_price) if quote.ask_price else 0
                    mid_price = (bid_price + ask_price) / 2 if bid_price and ask_price else bid_price or ask_price
                    
                    # Calculate real price change percentage
                    change_percent = self._get_price_change_percent(symbol, mid_price)
                    
                    quote_data = {
                        'symbol': symbol,
                        'price': round(mid_price, 2),
                        'bid': round(bid_price, 2),
                        'ask': round(ask_price, 2),
                        'change_percent': round(change_percent, 2),
                        'timestamp': quote.timestamp.isoformat() if quote.timestamp else datetime.now().isoformat(),
                        'source': 'alpaca_live'
                    }
                    
                    print(f"Live Alpaca quote for {symbol}: ${quote_data['price']:.2f} (bid: ${bid_price:.2f}, ask: ${ask_price:.2f})")
                    return jsonify(quote_data)
                else:
                    print(f"No quote data returned for {symbol} from Alpaca")
                    raise Exception("No quote data in response")
                    
            else:
                raise Exception("Alpaca data client not available")
                
        except Exception as e:
            print(f"Live quote failed for {symbol}: {str(e)} - falling back to simulation")
            
            # Fallback to simulation
            import random
            
            # Generate realistic quote data
            base_prices = {
                'AAPL': 175.0, 'GOOGL': 135.0, 'MSFT': 340.0, 
                'TSLA': 240.0, 'NVDA': 450.0
            }
            
            base_price = base_prices.get(symbol, 100.0)
            price_variation = random.uniform(-0.02, 0.02)  # ±2% variation
            current_price = base_price * (1 + price_variation)
            
            change_percent = random.uniform(-3.0, 3.0)
            
            quote_data = {
                'symbol': symbol,
                'price': round(current_price, 2),
                'bid': round(current_price - 0.01, 2),
                'ask': round(current_price + 0.01, 2),
                'change_percent': round(change_percent, 2),
                'timestamp': datetime.now().isoformat(),
                'source': 'simulation_fallback',
                'error_note': f'Fallback due to: {str(e)}'
            }
            
            print(f"Simulation fallback quote for {symbol}: ${quote_data['price']:.2f} ({quote_data['change_percent']:+.2f}%)")
            return jsonify(quote_data)

    def _get_simulation_account(self):
        """Get simulation account info"""
        if not self.trading_system:
            return jsonify({'error': 'Trading system not initialized'})

        portfolio = self.trading_system.broker.get_portfolio_summary()
        return jsonify({
            'buying_power': portfolio['cash_balance'],
            'cash': portfolio['cash_balance'],
            'portfolio_value': portfolio['total_value'],
            'equity': portfolio['total_value'],
            'return_percent': portfolio['total_return_percent'],
            'mode': 'SIMULATION'
        })

    def _get_simulation_positions(self):
        """Get simulation positions"""
        if not self.trading_system:
            return jsonify([])

        portfolio = self.trading_system.broker.get_portfolio_summary()
        return jsonify(portfolio.get('positions', []))

    def _get_simulation_orders(self):
        """Get simulation orders"""
        try:
            if not self.trading_system:
                return jsonify([])

            orders = []
            # Check if executed_orders exists and has data
            if hasattr(self.trading_system.broker, 'executed_orders') and self.trading_system.broker.executed_orders:
                for order in self.trading_system.broker.executed_orders[-10:]:  # Last 10 orders
                    try:
                        orders.append({
                            'id': order.get('order_id', f'sim_{len(orders)}'),
                            'symbol': order.get('symbol', 'UNKNOWN'),
                            'side': order.get('action', 'BUY'),
                            'qty': order.get('quantity', 0),
                            'filled_qty': order.get('quantity', 0),
                            'status': order.get('status', 'FILLED'),
                            'price': order.get('price', 0),
                            'submitted_at': order.get('timestamp', datetime.now()).isoformat() if hasattr(order.get('timestamp', datetime.now()), 'isoformat') else str(order.get('timestamp', datetime.now()))
                        })
                    except Exception as order_error:
                        print(f"Error processing order {order}: {order_error}")
                        continue
            
            return jsonify(orders)
            
        except Exception as e:
            print(f"Error getting simulation orders: {str(e)}")
            # Always return empty array on error to prevent JavaScript errors
            return jsonify([])

    def _get_simulation_quote(self, symbol):
        """Get simulation quote"""
        try:
            if not self.trading_system:
                return jsonify({'error': 'Trading system not initialized'})

            quote = self.trading_system.broker.get_real_time_quote(symbol)
            
            # Ensure quote has all required fields
            if isinstance(quote, dict) and 'price' in quote:
                # Add missing fields if they don't exist
                if 'change_percent' not in quote:
                    quote['change_percent'] = random.uniform(-3, 3)
                if 'symbol' not in quote:
                    quote['symbol'] = symbol
                if 'timestamp' not in quote:
                    quote['timestamp'] = datetime.now().isoformat()
                    
                return jsonify(quote)
            else:
                # Generate fallback quote
                price = random.uniform(95, 105)
                return jsonify({
                    'symbol': symbol,
                    'price': price,
                    'bid': price - 0.05,
                    'ask': price + 0.05,
                    'change_percent': random.uniform(-3, 3),
                    'timestamp': datetime.now().isoformat()
                })
                
        except Exception as e:
            print(f"Simulation quote error for {symbol}: {str(e)}")
            # Return fallback quote on any error
            price = random.uniform(95, 105)
            return jsonify({
                'symbol': symbol,
                'price': price,
                'bid': price - 0.05,
                'ask': price + 0.05,
                'change_percent': random.uniform(-3, 3),
                'timestamp': datetime.now().isoformat(),
                'error_note': f'Fallback quote due to: {str(e)}'
            })

    def run(self, host='0.0.0.0', port=5000):
        """Run the dashboard"""
        print(f"\n🚀 Starting Live Paper Trading Dashboard")
        print(f"📊 Mode: {self.mode}")
        print(f"🌐 Dashboard: http://localhost:{port}")
        print(f"💼 Access your positions, charts, and trades!")

        self.app.run(host=host, port=port, debug=False)

# Create HTML template
dashboard_html = '''
<!DOCTYPE html>
<html>
<head>
    <title>Live Paper Trading Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', sans-serif; background: #0a0a0a; color: #fff; }
        .header { background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 20px; text-align: center; }
        .header h1 { color: #00ff88; margin-bottom: 10px; }
        .status { display: inline-block; padding: 5px 15px; background: #00ff88; color: #000; border-radius: 20px; font-weight: bold; }

        .dashboard { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; padding: 20px; max-width: 1200px; margin: 0 auto; }

        .card { background: #1a1a1a; border-radius: 10px; padding: 20px; border: 1px solid #333; }
        .card h3 { color: #00ff88; margin-bottom: 15px; display: flex; align-items: center; }
        .card h3::before { content: '📊'; margin-right: 10px; }

        .account-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        .metric { text-align: center; padding: 15px; background: #2a2a2a; border-radius: 8px; }
        .metric-value { font-size: 24px; font-weight: bold; color: #00ff88; }
        .metric-label { font-size: 12px; color: #888; margin-top: 5px; }

        .position { display: flex; justify-content: space-between; align-items: center; padding: 10px; margin: 5px 0; background: #2a2a2a; border-radius: 5px; }
        .position-symbol { font-weight: bold; color: #00ff88; }
        .position-pnl.positive { color: #00ff88; }
        .position-pnl.negative { color: #ff4444; }

        .order { display: flex; justify-content: space-between; align-items: center; padding: 8px; margin: 3px 0; background: #2a2a2a; border-radius: 5px; font-size: 14px; }
        .order-buy { border-left: 3px solid #00ff88; }
        .order-sell { border-left: 3px solid #ff4444; }

        .quotes { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
        .quote { text-align: center; padding: 10px; background: #2a2a2a; border-radius: 5px; }
        .quote-price { font-size: 18px; font-weight: bold; color: #fff; }
        .quote-change.positive { color: #00ff88; }
        .quote-change.negative { color: #ff4444; }

        @media (max-width: 768px) {
            .dashboard { grid-template-columns: 1fr; }
            .account-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Live Paper Trading Dashboard</h1>
        <div class="status" id="status">Loading...</div>
    </div>

    <div class="dashboard">
        <div class="card">
            <h3>Account Summary</h3>
            <div class="account-grid" id="account-grid">
                <div class="metric">
                    <div class="metric-value" id="portfolio-value">$0</div>
                    <div class="metric-label">Portfolio Value</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="buying-power">$0</div>
                    <div class="metric-label">Buying Power</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="cash">$0</div>
                    <div class="metric-label">Cash</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="return">0%</div>
                    <div class="metric-label">Total Return</div>
                </div>
            </div>
        </div>

        <div class="card">
            <h3>Live Positions</h3>
            <div id="positions">No positions</div>
        </div>

        <div class="card">
            <h3>Recent Orders</h3>
            <div id="orders">No orders</div>
        </div>

        <div class="card">
            <h3>Live Quotes</h3>
            <div class="quotes" id="quotes"></div>
        </div>
    </div>

    <script>
        const symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA'];

        async function updateAccount() {
            try {
                const response = await fetch('/api/account');
                const account = await response.json();

                document.getElementById('portfolio-value').textContent = `$${account.portfolio_value?.toLocaleString() || account.total_value?.toLocaleString() || '0'}`;
                document.getElementById('buying-power').textContent = `$${account.buying_power?.toLocaleString() || '0'}`;
                document.getElementById('cash').textContent = `$${account.cash?.toLocaleString() || '0'}`;

                const returnValue = account.return_percent || 0;
                const returnElement = document.getElementById('return');
                returnElement.textContent = `${returnValue.toFixed(2)}%`;
                returnElement.style.color = returnValue >= 0 ? '#00ff88' : '#ff4444';

                document.getElementById('status').textContent = account.mode || 'ACTIVE';
            } catch (error) {
                console.error('Error updating account:', error);
            }
        }

        async function updatePositions() {
            try {
                const response = await fetch('/api/positions');
                const positions = await response.json();

                const positionsDiv = document.getElementById('positions');
                if (positions.length === 0) {
                    positionsDiv.innerHTML = '<div style="text-align: center; color: #888;">No open positions</div>';
                    return;
                }

                positionsDiv.innerHTML = positions.map(pos => `
                    <div class="position">
                        <div>
                            <span class="position-symbol">${pos.symbol}</span>
                            <span style="color: #888;"> ${pos.quantity || pos.qty} shares</span>
                        </div>
                        <div>
                            <span>$${(pos.market_value || pos.current_price * pos.quantity)?.toLocaleString()}</span>
                            <span class="position-pnl ${(pos.unrealized_pl || pos.change_percent) >= 0 ? 'positive' : 'negative'}">
                                ${pos.unrealized_plpc ? pos.unrealized_plpc.toFixed(2) + '%' : pos.change_percent?.toFixed(2) + '%' || '0%'}
                            </span>
                        </div>
                    </div>
                `).join('');
            } catch (error) {
                console.error('Error updating positions:', error);
            }
        }

        async function updateOrders() {
            try {
                const response = await fetch('/api/orders');
                const orders = await response.json();

                const ordersDiv = document.getElementById('orders');
                if (orders.length === 0) {
                    ordersDiv.innerHTML = '<div style="text-align: center; color: #888;">No recent orders</div>';
                    return;
                }

                ordersDiv.innerHTML = orders.slice(-5).map(order => `
                    <div class="order ${order.side?.toLowerCase() === 'buy' || order.action === 'BUY' ? 'order-buy' : 'order-sell'}">
                        <div>
                            <strong>${order.symbol}</strong> ${order.side || order.action} ${order.qty || order.quantity}
                        </div>
                        <div style="font-size: 12px; color: #888;">
                            ${order.status} ${order.price ? '@$' + order.price : ''}
                        </div>
                    </div>
                `).join('');
            } catch (error) {
                console.error('Error updating orders:', error);
            }
        }

        async function updateQuotes() {
            try {
                const quotesDiv = document.getElementById('quotes');
                const quotePromises = symbols.map(async symbol => {
                    try {
                        const response = await fetch(`/api/quotes/${symbol}`);
                        if (!response.ok) {
                            throw new Error(`HTTP ${response.status}`);
                        }
                        const quote = await response.json();
                        
                        // Check if quote has error
                        if (quote.error) {
                            console.warn(`Quote error for ${symbol}:`, quote.error);
                            return {
                                symbol: symbol,
                                price: 0,
                                change_percent: 0,
                                error: true
                            };
                        }
                        
                        return quote;
                    } catch (error) {
                        console.error(`Failed to fetch quote for ${symbol}:`, error);
                        return {
                            symbol: symbol,
                            price: 0,
                            change_percent: 0,
                            error: true
                        };
                    }
                });

                const quotes = await Promise.all(quotePromises);

                quotesDiv.innerHTML = quotes.map(quote => `
                    <div class="quote">
                        <div style="font-weight: bold; color: #00ff88;">${quote.symbol || 'Unknown'}</div>
                        <div class="quote-price">$${(quote.price || 0).toFixed(2)}</div>
                        <div class="quote-change ${(quote.change_percent || 0) >= 0 ? 'positive' : 'negative'}">
                            ${quote.error ? 'Error' : (quote.change_percent ? (quote.change_percent > 0 ? '+' : '') + quote.change_percent.toFixed(2) + '%' : '0%')}
                        </div>
                    </div>
                `).join('');
            } catch (error) {
                console.error('Error updating quotes:', error);
                // Display error message in quotes section
                const quotesDiv = document.getElementById('quotes');
                quotesDiv.innerHTML = '<div style="color: #ff4444; text-align: center;">Error loading quotes</div>';
            }
        }

        function updateAll() {
            updateAccount();
            updatePositions();
            updateOrders();
            updateQuotes();
        }

        // Initial load
        updateAll();

        // Update every 5 seconds
        setInterval(updateAll, 5000);
    </script>
</body>
</html>
'''

# Save the HTML template
if __name__ == "__main__":
    # Create templates directory
    import os
    os.makedirs('templates', exist_ok=True)

    with open('templates/dashboard.html', 'w') as f:
        f.write(dashboard_html)

    # Start the dashboard
    dashboard = AlpacaLiveDashboard()

    print("\n" + "="*60)
    print("🏦 ALPACA PAPER TRADING SETUP")
    print("="*60)

    if not dashboard.has_alpaca_keys:
        print("📝 To connect to Alpaca paper trading:")
        print("   1. Sign up at: https://alpaca.markets/")
        print("   2. Get your PAPER TRADING API keys (not live keys!)")
        print("   3. Set environment variables:")
        print("      ALPACA_API_KEY=your_paper_key")
        print("      ALPACA_SECRET_KEY=your_paper_secret")
        print("   4. System automatically uses paper-api.alpaca.markets domain")
        print("\n🎮 Running in simulation mode for now...")

    dashboard.run()