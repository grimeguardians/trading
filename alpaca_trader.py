#!/usr/bin/env python3
"""
Enhanced Multi-Asset Alpaca Trading System
Visual dashboard with stocks, ETFs, crypto, and futures
"""

import os
import time
import threading
from datetime import datetime
from flask import Flask, render_template, jsonify
import json

# Load configuration and strategy engine
from config import config
from intelligent_strategy_engine import strategy_engine, TradingSignal

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
        # Get credentials from config
        self.api_key = config.alpaca_api_key
        self.secret_key = config.alpaca_secret_key
        
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
        
        # Load symbols from config
        self.symbols = config.symbols
        self.enabled_assets = config.enabled_assets
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
    
    def intelligent_trading_logic(self):
        """Literature-driven intelligent trading with Digital Brain"""
        print("🧠 Intelligent Trading Engine Active")
        print("📚 Using strategies from uploaded trading literature")
        print("⚡ Dynamic stop-loss management enabled")
        
        while self.running:
            try:
                # Get portfolio status
                portfolio = self.get_portfolio_status()
                if not portfolio:
                    time.sleep(30)
                    continue
                
                print(f"📊 Portfolio: ${portfolio['portfolio_value']:,.2f} | Cash: ${portfolio['cash']:,.2f}")
                
                # Check all enabled symbols for trading opportunities
                for asset_class in self.enabled_assets:
                    if asset_class in self.symbols:
                        for symbol in self.symbols[asset_class][:3]:  # Limit to 3 per class
                            
                            # Get market data
                            quote = self.get_quote(symbol)
                            if not quote:
                                continue
                                
                            # Prepare market data for analysis
                            market_data = {
                                'price': quote['price'],
                                'bid': quote['bid'],
                                'ask': quote['ask'],
                                'volume': 1000,  # Placeholder
                                'ma_20': quote['price'] * 0.98,  # Simple approximation
                                'ma_50': quote['price'] * 0.96,
                                'recent_high': quote['price'] * 1.05,
                                'recent_low': quote['price'] * 0.95,
                                'support': quote['price'] * 0.97,
                                'resistance': quote['price'] * 1.03,
                                'atr': quote['price'] * 0.02  # 2% ATR estimate
                            }
                            
                            # Get intelligent signal from strategy engine
                            signal = strategy_engine.analyze_market_conditions(symbol, market_data)
                            
                            # Execute trades based on signal (lowered threshold for more activity)
                            if signal.confidence > 0.25:  # Lower threshold for more active trading
                                self._execute_intelligent_trade(signal, portfolio, market_data)
                            elif signal.confidence > 0.15:  # Even lower threshold for monitoring
                                print(f"🔍 Monitoring {symbol}: {signal.action} signal ({signal.confidence:.1%} confidence)")
                            
                            # Manage existing positions
                            self._manage_existing_positions(symbol, market_data, portfolio)
                
                time.sleep(config.portfolio_check_interval)  # Configurable interval
                
            except Exception as e:
                print(f"🚨 Trading engine error: {e}")
                time.sleep(30)
    
    def _execute_intelligent_trade(self, signal: TradingSignal, portfolio: Dict, market_data: Dict):
        """Execute trade based on intelligent signal"""
        symbol = signal.symbol
        
        # Check if we already have a position
        has_position = any(pos['symbol'] == symbol for pos in portfolio['positions'])
        
        if signal.action == 'BUY' and not has_position and portfolio['cash'] > 100:
            
            entry_price = market_data['price']
            
            # Calculate intelligent stop loss using literature
            stop_loss = strategy_engine.calculate_dynamic_stop_loss(
                symbol, entry_price, 'BUY', market_data
            )
            
            # Calculate position size using risk management
            position_size = strategy_engine.get_position_size(
                symbol, entry_price, stop_loss, portfolio['portfolio_value']
            )
            
            if position_size > 0 and position_size * entry_price <= portfolio['cash']:
                print(f"🚀 INTELLIGENT BUY: {symbol}")
                print(f"📊 Confidence: {signal.confidence:.1%}")
                print(f"🧠 Strategy: {signal.strategy}")
                print(f"💭 Reasoning: {signal.reasoning[0]}")
                print(f"🛡️ Stop Loss: ${stop_loss:.2f}")
                
                success = self.place_order(symbol, 'BUY', position_size)
                
                if success:
                    # Track position for stop loss management
                    strategy_engine.position_tracking[symbol] = {
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'direction': 'BUY',
                        'entry_time': datetime.now(),
                        'strategy': signal.strategy,
                        'literature_source': signal.literature_source
                    }
        
        elif signal.action == 'SELL' and has_position:
            # Get position details
            position = next(pos for pos in portfolio['positions'] if pos['symbol'] == symbol)
            if position['quantity'] > 0:
                print(f"📉 INTELLIGENT SELL: {symbol}")
                print(f"📊 Confidence: {signal.confidence:.1%}")
                print(f"🧠 Strategy: {signal.strategy}")
                print(f"💭 Reasoning: {signal.reasoning[0]}")
                
                self.place_order(symbol, 'SELL', position['quantity'])
                
                # Remove from tracking
                if symbol in strategy_engine.position_tracking:
                    del strategy_engine.position_tracking[symbol]
    
    def _manage_existing_positions(self, symbol: str, market_data: Dict, portfolio: Dict):
        """Manage existing positions with intelligent stop loss"""
        
        if symbol not in strategy_engine.position_tracking:
            return
            
        position_info = strategy_engine.position_tracking[symbol]
        current_price = market_data['price']
        
        # Check if we should move the stop loss
        should_move, new_stop = strategy_engine.should_move_stop_loss(
            symbol, 
            position_info['entry_price'],
            current_price,
            position_info['stop_loss'],
            position_info['direction']
        )
        
        if should_move:
            print(f"📈 MOVING STOP: {symbol} from ${position_info['stop_loss']:.2f} to ${new_stop:.2f}")
            position_info['stop_loss'] = new_stop
        
        # Check if stop loss is hit
        if position_info['direction'] == 'BUY' and current_price <= position_info['stop_loss']:
            print(f"🛑 STOP LOSS HIT: {symbol} at ${current_price:.2f}")
            
            # Find position and sell
            position = next((pos for pos in portfolio['positions'] if pos['symbol'] == symbol), None)
            if position and position['quantity'] > 0:
                self.place_order(symbol, 'SELL', position['quantity'])
                del strategy_engine.position_tracking[symbol]

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
    """API endpoint for current quotes - all asset classes"""
    if trader:
        quotes = {}
        for asset_class, symbols in trader.symbols.items():
            quotes[asset_class] = {}
            for symbol in symbols:
                quote = trader.get_quote(symbol)
                if quote:
                    quotes[asset_class][symbol] = quote
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

@app.route('/api/strategy-status')
def api_strategy_status():
    """API endpoint for intelligent strategy status"""
    if trader and strategy_engine:
        status = {
            'engine_active': trader.running,
            'literature_sources': len(strategy_engine.active_strategies),
            'active_positions': len(strategy_engine.position_tracking),
            'brain_available': strategy_engine.brain is not None,
            'strategies': list(strategy_engine.active_strategies.keys()),
            'recent_signals': []  # Could add recent signal history
        }
        return jsonify(status)
    return jsonify({'error': 'Strategy engine not available'})

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
        .asset-tabs { margin-bottom: 20px; }
        .tab-btn { padding: 10px 20px; margin-right: 10px; border: none; background: #ecf0f1; cursor: pointer; border-radius: 4px; }
        .tab-btn.active { background: #3498db; color: white; }
        .asset-section { display: none; }
        .asset-section.active { display: block; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Intelligent Trading Dashboard</h1>
            <p>Literature-driven strategies with dynamic stop management</p>
            <div style="margin-top: 10px; font-size: 14px;">
                📚 Active Strategies: <span id="strategy-count">Loading...</span> | 
                🎯 Digital Brain: <span id="brain-status">Loading...</span> |
                📊 Tracked Positions: <span id="tracked-positions">0</span>
            </div>
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
            <h2>💹 Live Quotes & Multi-Asset Trading</h2>
            
            <div class="asset-tabs">
                <button class="tab-btn active" onclick="showAssetClass('stocks')">📈 Stocks</button>
                <button class="tab-btn" onclick="showAssetClass('etfs')">📊 ETFs</button>
                <button class="tab-btn" onclick="showAssetClass('crypto')">₿ Crypto</button>
                <button class="tab-btn" onclick="showAssetClass('futures')">📋 Futures</button>
            </div>
            
            <div id="stocks-quotes" class="asset-section active">
                <h3>📈 Stocks</h3>
                <div class="quotes" id="stocks-grid"></div>
            </div>
            
            <div id="etfs-quotes" class="asset-section">
                <h3>📊 ETFs</h3>
                <div class="quotes" id="etfs-grid"></div>
            </div>
            
            <div id="crypto-quotes" class="asset-section">
                <h3>₿ Cryptocurrency</h3>
                <div class="quotes" id="crypto-grid"></div>
            </div>
            
            <div id="futures-quotes" class="asset-section">
                <h3>📋 Futures</h3>
                <div class="quotes" id="futures-grid"></div>
            </div>
        </div>
    </div>

    <script>
        function updateDashboard() {
            // Update strategy status
            fetch('/api/strategy-status')
                .then(response => response.json())
                .then(data => {
                    if (!data.error) {
                        document.getElementById('strategy-count').textContent = data.literature_sources;
                        document.getElementById('brain-status').textContent = data.brain_available ? 'Active' : 'Offline';
                        document.getElementById('tracked-positions').textContent = data.active_positions;
                    }
                })
                .catch(error => console.log('Strategy status update failed'));
            
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
            
            // Update quotes for all asset classes
            fetch('/api/quotes')
                .then(response => response.json())
                .then(data => {
                    updateAssetQuotes('stocks', data.stocks || {});
                    updateAssetQuotes('etfs', data.etfs || {});
                    updateAssetQuotes('crypto', data.crypto || {});
                    updateAssetQuotes('futures', data.futures || {});
                });
        }
        
        function updateAssetQuotes(assetClass, quotes) {
            const grid = document.getElementById(`${assetClass}-grid`);
            if (grid) {
                grid.innerHTML = Object.entries(quotes).map(([symbol, quote]) =>
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
            }
        }
        
        function showAssetClass(assetClass) {
            // Hide all sections
            document.querySelectorAll('.asset-section').forEach(section => {
                section.classList.remove('active');
            });
            document.querySelectorAll('.tab-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            
            // Show selected section
            document.getElementById(`${assetClass}-quotes`).classList.add('active');
            event.target.classList.add('active');
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
    
    # Start intelligent trading logic in background
    trader.running = True
    trading_thread = threading.Thread(target=trader.intelligent_trading_logic)
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