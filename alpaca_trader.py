#!/usr/bin/env python3
"""
Minimal Alpaca Paper Trading - Get Trading ASAP
Visual dashboard with real positions and trades
"""

import os
import time
import threading
from datetime import datetime
from flask import Flask, render_template, jsonify
import json

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Alpaca SDK
try:
    from alpaca.trading.client import TradingClient
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    ALPACA_AVAILABLE = True
except ImportError:
    print("⚠️ Installing Alpaca SDK...")
    import subprocess
    subprocess.run(["pip3", "install", "alpaca-py"], check=True)
    from alpaca.trading.client import TradingClient
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    ALPACA_AVAILABLE = True

class AlpacaTrader:
    def __init__(self):
        # Get credentials
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        if not self.api_key or not self.secret_key:
            print("❌ Alpaca credentials not found!")
            print("📋 Add to .env file:")
            print("ALPACA_API_KEY=your_key_here")
            print("ALPACA_SECRET_KEY=your_secret_here")
            print("🔗 Get keys at: https://app.alpaca.markets/signup")
            exit(1)
        
        # Initialize clients
        self.trading_client = TradingClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=True
        )
        
        self.data_client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.secret_key
        )
        
        # Test connection
        try:
            self.account = self.trading_client.get_account()
            print("✅ Connected to Alpaca Paper Trading!")
            print(f"💰 Portfolio Value: ${float(self.account.portfolio_value):,.2f}")
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            exit(1)
        
        # Trading state
        self.symbols = ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'NVDA']
        self.running = False
        
    def get_portfolio_status(self):
        """Get current portfolio status"""
        try:
            account = self.trading_client.get_account()
            positions = self.trading_client.get_all_positions()
            
            portfolio_data = {
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'buying_power': float(account.buying_power),
                'day_trade_count': int(account.daytrade_count),
                'total_return': ((float(account.portfolio_value) - 100000) / 100000) * 100,
                'positions': []
            }
            
            for pos in positions:
                portfolio_data['positions'].append({
                    'symbol': pos.symbol,
                    'quantity': int(pos.qty),
                    'avg_price': float(pos.avg_entry_price),
                    'current_value': float(pos.market_value),
                    'unrealized_pnl': float(pos.unrealized_pl),
                    'unrealized_pnl_pct': float(pos.unrealized_plpc) * 100
                })
            
            return portfolio_data
        except Exception as e:
            print(f"Error getting portfolio: {e}")
            return None
    
    def get_quote(self, symbol):
        """Get current quote for symbol"""
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
            quotes = self.data_client.get_stock_latest_quote(request)
            
            if symbol in quotes:
                quote = quotes[symbol]
                return {
                    'symbol': symbol,
                    'bid': float(quote.bid_price),
                    'ask': float(quote.ask_price),
                    'price': (float(quote.bid_price) + float(quote.ask_price)) / 2,
                    'timestamp': quote.timestamp.isoformat()
                }
        except Exception as e:
            print(f"Error getting quote for {symbol}: {e}")
        return None
    
    def place_order(self, symbol, side, quantity):
        """Place market order"""
        try:
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=OrderSide.BUY if side == 'BUY' else OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            
            order = self.trading_client.submit_order(order_request)
            print(f"✅ Order placed: {side} {quantity} {symbol} (ID: {order.id})")
            return True
            
        except Exception as e:
            print(f"❌ Order failed: {e}")
            return False
    
    def simple_trading_logic(self):
        """Simple trading logic for demo"""
        import random
        
        while self.running:
            try:
                for symbol in self.symbols:
                    quote = self.get_quote(symbol)
                    
                    if quote and random.random() < 0.1:  # 10% chance to trade
                        portfolio = self.get_portfolio_status()
                        
                        if portfolio:
                            # Simple logic: buy if we have cash, sell if we have position
                            has_position = any(pos['symbol'] == symbol for pos in portfolio['positions'])
                            
                            if not has_position and portfolio['cash'] > quote['price'] * 10:
                                # Buy 10 shares if we have cash and no position
                                if random.random() > 0.5:  # 50% chance
                                    self.place_order(symbol, 'BUY', 10)
                            
                            elif has_position and random.random() > 0.7:  # 30% chance to sell
                                # Sell position
                                pos = next(pos for pos in portfolio['positions'] if pos['symbol'] == symbol)
                                if pos['quantity'] > 0:
                                    self.place_order(symbol, 'SELL', pos['quantity'])
                
                time.sleep(30)  # Wait 30 seconds between trades
                
            except Exception as e:
                print(f"Trading error: {e}")
                time.sleep(10)

# Flask Web Dashboard
app = Flask(__name__)
trader = None

@app.route('/')
def dashboard():
    """Main dashboard"""
    return render_template('dashboard.html')

@app.route('/api/portfolio')
def api_portfolio():
    """API endpoint for portfolio data"""
    if trader:
        portfolio = trader.get_portfolio_status()
        if portfolio:
            return jsonify(portfolio)
    return jsonify({'error': 'No data available'})

@app.route('/api/quotes')
def api_quotes():
    """API endpoint for current quotes"""
    if trader:
        quotes = {}
        for symbol in trader.symbols:
            quote = trader.get_quote(symbol)
            if quote:
                quotes[symbol] = quote
        return jsonify(quotes)
    return jsonify({})

@app.route('/api/buy/<symbol>/<int:quantity>')
def api_buy(symbol, quantity):
    """API endpoint to place buy order"""
    if trader:
        success = trader.place_order(symbol, 'BUY', quantity)
        return jsonify({'success': success})
    return jsonify({'success': False})

@app.route('/api/sell/<symbol>/<int:quantity>')
def api_sell(symbol, quantity):
    """API endpoint to place sell order"""
    if trader:
        success = trader.place_order(symbol, 'SELL', quantity)
        return jsonify({'success': success})
    return jsonify({'success': False})

def create_dashboard_template():
    """Create the HTML dashboard template"""
    html_content = '''<!DOCTYPE html>
<html>
<head>
    <title>Alpaca Paper Trading Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .card { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .portfolio { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        .metric { text-align: center; }
        .metric h3 { margin: 0; color: #34495e; }
        .metric .value { font-size: 24px; font-weight: bold; margin: 10px 0; }
        .positive { color: #27ae60; }
        .negative { color: #e74c3c; }
        .positions table { width: 100%; border-collapse: collapse; }
        .positions th, .positions td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        .positions th { background: #f8f9fa; }
        .quotes { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; }
        .quote { background: #ecf0f1; padding: 15px; border-radius: 8px; text-align: center; }
        .quote h4 { margin: 0 0 10px 0; }
        .btn { padding: 8px 16px; margin: 5px; border: none; border-radius: 4px; cursor: pointer; }
        .btn-buy { background: #27ae60; color: white; }
        .btn-sell { background: #e74c3c; color: white; }
        .btn:hover { opacity: 0.8; }
        #status { padding: 10px; background: #d4edda; border-radius: 4px; margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🦙 Alpaca Paper Trading Dashboard</h1>
            <p>Live portfolio tracking and trading</p>
        </div>
        
        <div id="status" class="card">
            <strong>Status:</strong> <span id="connection-status">Connecting...</span>
        </div>
        
        <div class="card">
            <h2>📊 Portfolio Overview</h2>
            <div class="portfolio">
                <div class="metric">
                    <h3>Portfolio Value</h3>
                    <div class="value" id="portfolio-value">$0.00</div>
                </div>
                <div class="metric">
                    <h3>Cash Balance</h3>
                    <div class="value" id="cash-balance">$0.00</div>
                </div>
                <div class="metric">
                    <h3>Buying Power</h3>
                    <div class="value" id="buying-power">$0.00</div>
                </div>
                <div class="metric">
                    <h3>Total Return</h3>
                    <div class="value" id="total-return">0.00%</div>
                </div>
            </div>
        </div>
        
        <div class="card positions">
            <h2>📈 Current Positions</h2>
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Quantity</th>
                        <th>Avg Price</th>
                        <th>Current Value</th>
                        <th>P&L</th>
                        <th>P&L %</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody id="positions-table">
                    <tr><td colspan="7">No positions</td></tr>
                </tbody>
            </table>
        </div>
        
        <div class="card">
            <h2>💹 Live Quotes & Trading</h2>
            <div class="quotes" id="quotes-grid">
                <!-- Quotes will be loaded here -->
            </div>
        </div>
    </div>

    <script>
        function updateDashboard() {
            // Update portfolio
            fetch('/api/portfolio')
                .then(response => response.json())
                .then(data => {
                    if (data.error) return;
                    
                    document.getElementById('portfolio-value').textContent = '$' + data.portfolio_value.toLocaleString('en-US', {minimumFractionDigits: 2});
                    document.getElementById('cash-balance').textContent = '$' + data.cash.toLocaleString('en-US', {minimumFractionDigits: 2});
                    document.getElementById('buying-power').textContent = '$' + data.buying_power.toLocaleString('en-US', {minimumFractionDigits: 2});
                    
                    const returnEl = document.getElementById('total-return');
                    returnEl.textContent = data.total_return.toFixed(2) + '%';
                    returnEl.className = 'value ' + (data.total_return >= 0 ? 'positive' : 'negative');
                    
                    // Update positions
                    const positionsTable = document.getElementById('positions-table');
                    if (data.positions.length === 0) {
                        positionsTable.innerHTML = '<tr><td colspan="7">No positions</td></tr>';
                    } else {
                        positionsTable.innerHTML = data.positions.map(pos => 
                            `<tr>
                                <td><strong>${pos.symbol}</strong></td>
                                <td>${pos.quantity}</td>
                                <td>$${pos.avg_price.toFixed(2)}</td>
                                <td>$${pos.current_value.toLocaleString('en-US', {minimumFractionDigits: 2})}</td>
                                <td class="${pos.unrealized_pnl >= 0 ? 'positive' : 'negative'}">$${pos.unrealized_pnl.toFixed(2)}</td>
                                <td class="${pos.unrealized_pnl >= 0 ? 'positive' : 'negative'}">${pos.unrealized_pnl_pct.toFixed(2)}%</td>
                                <td><button class="btn btn-sell" onclick="sellPosition('${pos.symbol}', ${pos.quantity})">Sell All</button></td>
                            </tr>`
                        ).join('');
                    }
                    
                    document.getElementById('connection-status').textContent = 'Connected ✅';
                })
                .catch(error => {
                    document.getElementById('connection-status').textContent = 'Connection Error ❌';
                });
            
            // Update quotes
            fetch('/api/quotes')
                .then(response => response.json())
                .then(data => {
                    const quotesGrid = document.getElementById('quotes-grid');
                    quotesGrid.innerHTML = Object.entries(data).map(([symbol, quote]) =>
                        `<div class="quote">
                            <h4>${symbol}</h4>
                            <div>$${quote.price.toFixed(2)}</div>
                            <div style="font-size: 12px; color: #666;">
                                Bid: $${quote.bid.toFixed(2)}<br>
                                Ask: $${quote.ask.toFixed(2)}
                            </div>
                            <div style="margin-top: 10px;">
                                <button class="btn btn-buy" onclick="buyStock('${symbol}', 10)">Buy 10</button>
                                <button class="btn btn-buy" onclick="buyStock('${symbol}', 1)">Buy 1</button>
                            </div>
                        </div>`
                    ).join('');
                });
        }
        
        function buyStock(symbol, quantity) {
            fetch(`/api/buy/${symbol}/${quantity}`)
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert(`✅ Bought ${quantity} shares of ${symbol}`);
                        setTimeout(updateDashboard, 1000);
                    } else {
                        alert(`❌ Failed to buy ${symbol}`);
                    }
                });
        }
        
        function sellPosition(symbol, quantity) {
            fetch(`/api/sell/${symbol}/${quantity}`)
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert(`✅ Sold ${quantity} shares of ${symbol}`);
                        setTimeout(updateDashboard, 1000);
                    } else {
                        alert(`❌ Failed to sell ${symbol}`);
                    }
                });
        }
        
        // Update dashboard every 10 seconds
        updateDashboard();
        setInterval(updateDashboard, 10000);
    </script>
</body>
</html>'''
    
    # Create templates directory if it doesn't exist
    import os
    os.makedirs('templates', exist_ok=True)
    
    with open('templates/dashboard.html', 'w') as f:
        f.write(html_content)

def main():
    global trader
    
    print("🚀 Starting Alpaca Paper Trading Dashboard")
    print("=" * 50)
    
    # Create HTML template
    create_dashboard_template()
    
    # Initialize trader
    trader = AlpacaTrader()
    
    # Start simple trading logic in background
    trader.running = True
    trading_thread = threading.Thread(target=trader.simple_trading_logic)
    trading_thread.daemon = True
    trading_thread.start()
    
    print("\n🌐 Starting web dashboard...")
    print("📊 Access your dashboard at:")
    print("   • http://127.0.0.1:5000")
    print("   • http://localhost:5000") 
    print("⚡ Simple trading algorithm running in background")
    print("\n💡 If URLs don't work, try opening a new terminal and run:")
    print("   curl http://127.0.0.1:5000")
    print("\nPress Ctrl+C to stop\n")
    
    try:
        app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        trader.running = False
        print("\n🛑 Stopping trader...")

if __name__ == "__main__":
    main()