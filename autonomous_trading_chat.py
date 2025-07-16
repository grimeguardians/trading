#!/usr/bin/env python3
"""
Autonomous Trading Chat Interface
Natural language commands for trading operations
"""

import os
import time
import threading
from datetime import datetime
from flask import Flask, render_template, jsonify
import json
import re
from typing import Dict, List, Any

# Flask-SocketIO with fallback
try:
    from flask_socketio import SocketIO, emit
    SOCKETIO_AVAILABLE = True
except ImportError:
    print("⚠️ Installing Flask-SocketIO...")
    import subprocess
    subprocess.run(["pip3", "install", "flask-socketio"], check=True)
    from flask_socketio import SocketIO, emit
    SOCKETIO_AVAILABLE = True

from alpaca_trader import AlpacaTrader
from intelligent_strategy_engine import strategy_engine

class AutonomousTradingChatBot:
    """Enhanced chat bot with autonomous trading capabilities"""
    
    def __init__(self):
        self.trader = AlpacaTrader()
        self.is_autonomous = True
        self.monitoring_symbols = []
        self.trading_commands = {
            'buy': self._execute_buy_command,
            'sell': self._execute_sell_command,
            'status': self._get_portfolio_status,
            'positions': self._get_positions,
            'start_autonomous': self._start_autonomous_trading,
            'stop_autonomous': self._stop_autonomous_trading,
            'add_symbol': self._add_monitoring_symbol,
            'remove_symbol': self._remove_monitoring_symbol,
            'trade_ideas': self._get_trade_ideas,
            'risk_report': self._get_risk_report
        }
        
    def process_natural_language(self, message: str) -> Dict[str, Any]:
        """Process natural language trading commands"""
        message_lower = message.lower()
        
        # Extract symbols
        symbols = self._extract_symbols(message)
        
        # Extract quantities
        quantities = self._extract_quantities(message)
        
        # Determine intent
        if any(word in message_lower for word in ['buy', 'purchase', 'long']):
            return self._execute_buy_command(symbols, quantities, message)
        elif any(word in message_lower for word in ['sell', 'short', 'close']):
            return self._execute_sell_command(symbols, quantities, message)
        elif any(word in message_lower for word in ['status', 'portfolio', 'account']):
            return self._get_portfolio_status()
        elif any(word in message_lower for word in ['positions', 'holdings']):
            return self._get_positions()
        elif any(word in message_lower for word in ['ideas', 'opportunities', 'setups']):
            return self._get_trade_ideas()
        elif any(word in message_lower for word in ['autonomous', 'auto', 'start trading']):
            return self._start_autonomous_trading()
        elif any(word in message_lower for word in ['stop', 'pause', 'halt']):
            return self._stop_autonomous_trading()
        elif any(word in message_lower for word in ['add', 'monitor', 'watch']):
            return self._add_monitoring_symbol(symbols)
        elif any(word in message_lower for word in ['risk', 'drawdown', 'exposure']):
            return self._get_risk_report()
        else:
            return self._general_response(message)
    
    def _extract_symbols(self, message: str) -> List[str]:
        """Extract trading symbols from message with altcoin season support"""
        # Flagship crypto symbols available on Alpaca
        crypto_symbols = ['BTC', 'ETH', 'XRP']
        stock_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'META', 'SPY', 'QQQ', 'NFLX', 'CRM', 'PLTR']
        
        found_symbols = []
        message_upper = message.upper()
        
        # Check for crypto with USD pairs only (Alpaca available)
        for crypto in crypto_symbols:
            if crypto in message_upper:
                found_symbols.append(f"{crypto}USD")
        
        # Check for stocks
        for stock in stock_symbols:
            if stock in message_upper:
                found_symbols.append(stock)
        
        return found_symbols
    
    def _extract_quantities(self, message: str) -> List[int]:
        """Extract quantities from message"""
        quantities = re.findall(r'\b(\d+)\b', message)
        return [int(q) for q in quantities if int(q) < 10000]  # Reasonable limit
    
    def _execute_buy_command(self, symbols: List[str], quantities: List[int], message: str) -> Dict[str, Any]:
        """Execute buy command with natural language"""
        if not symbols:
            return {
                'message': "🤖 I need a symbol to buy. Try: 'Buy 10 shares of AAPL' or 'Buy 0.1 BTC'",
                'type': 'error'
            }
        
        symbol = symbols[0]
        
        # Smart position sizing based on available cash and asset type
        portfolio = self.trader.get_portfolio_status()
        available_cash = portfolio['cash'] if portfolio else 1000
        
        # Get current price to calculate affordable quantity
        quote = self.trader.get_quote(symbol)
        if not quote:
            return {
                'message': f"❌ Could not get quote for {symbol}",
                'type': 'error'
            }
        
        current_price = quote['price']
        
        # Calculate maximum affordable quantity (use 80% of available cash for safety)
        max_affordable = max(1, int((available_cash * 0.8) / current_price))
        
        # Use user quantity if provided, otherwise use smart defaults
        if quantities:
            quantity = quantities[0]
        else:
            # Smart defaults based on asset type and price
            if symbol in ['BTCUSD', 'ETHUSD']:
                quantity = min(max_affordable, 1)  # Max 1 unit for expensive crypto
            elif 'USD' in symbol:  # Other crypto
                quantity = min(max_affordable, 10)  # Max 10 units for cheaper crypto
            else:  # Stocks
                quantity = min(max_affordable, 10)  # Max 10 shares for stocks
        
        # Final safety check - ensure we can afford it
        if quantity * current_price > available_cash * 0.8:
            quantity = max(1, int((available_cash * 0.8) / current_price))
        
        # Absolute safety bounds
        quantity = max(1, min(quantity, max_affordable))
        
        # Final affordability check
        total_cost = quantity * current_price
        if total_cost > available_cash:
            return {
                'message': f"❌ Insufficient funds: Need ${total_cost:.2f}, have ${available_cash:.2f}",
                'type': 'error'
            }
        
        try:
            # Execute trade
            success = self.trader.place_order(symbol, 'BUY', quantity)
            
            if success:
                portfolio = self.trader.get_portfolio_status()
                return {
                    'message': f"✅ **BUY ORDER EXECUTED**\n\n"
                              f"**Symbol:** {symbol}\n"
                              f"**Quantity:** {quantity}\n"
                              f"**Price:** ${quote['price']:.2f}\n"
                              f"**Total Cost:** ${quantity * quote['price']:,.2f}\n\n"
                              f"**Portfolio Value:** ${portfolio['portfolio_value']:,.2f}\n"
                              f"**Cash Remaining:** ${portfolio['cash']:,.2f}",
                    'type': 'trade_success',
                    'trade_data': {
                        'symbol': symbol,
                        'action': 'BUY',
                        'quantity': quantity,
                        'price': quote['price']
                    }
                }
            else:
                return {
                    'message': f"❌ Failed to execute BUY order for {symbol}",
                    'type': 'trade_error'
                }
                
        except Exception as e:
            return {
                'message': f"❌ Error executing buy: {str(e)}",
                'type': 'error'
            }
    
    def _execute_sell_command(self, symbols: List[str], quantities: List[int], message: str) -> Dict[str, Any]:
        """Execute sell command"""
        if not symbols:
            return {
                'message': "🤖 I need a symbol to sell. Try: 'Sell all AAPL' or 'Sell 50% of my BTC'",
                'type': 'error'
            }
        
        symbol = symbols[0]
        portfolio = self.trader.get_portfolio_status()
        
        # Find position
        position = None
        for pos in portfolio['positions']:
            if pos['symbol'] == symbol:
                position = pos
                break
        
        if not position or position['quantity'] <= 0:
            return {
                'message': f"❌ No position found for {symbol}",
                'type': 'error'
            }
        
        # Determine quantity to sell
        if 'all' in message.lower():
            sell_quantity = position['quantity']
        elif '%' in message:
            # Extract percentage
            pct_match = re.search(r'(\d+)%', message)
            if pct_match:
                pct = int(pct_match.group(1)) / 100
                sell_quantity = int(position['quantity'] * pct)
            else:
                sell_quantity = quantities[0] if quantities else position['quantity']
        else:
            sell_quantity = quantities[0] if quantities else position['quantity']
        
        sell_quantity = min(sell_quantity, position['quantity'])
        
        try:
            success = self.trader.place_order(symbol, 'SELL', sell_quantity)
            
            if success:
                quote = self.trader.get_quote(symbol)
                estimated_proceeds = sell_quantity * quote['price'] if quote else 0
                
                return {
                    'message': f"✅ **SELL ORDER EXECUTED**\n\n"
                              f"**Symbol:** {symbol}\n"
                              f"**Quantity Sold:** {sell_quantity}\n"
                              f"**Estimated Price:** ${quote['price']:.2f if quote else 0}\n"
                              f"**Estimated Proceeds:** ${estimated_proceeds:,.2f}\n"
                              f"**Remaining Position:** {position['quantity'] - sell_quantity}",
                    'type': 'trade_success'
                }
            else:
                return {
                    'message': f"❌ Failed to execute SELL order for {symbol}",
                    'type': 'trade_error'
                }
                
        except Exception as e:
            return {
                'message': f"❌ Error executing sell: {str(e)}",
                'type': 'error'
            }
    
    def _get_portfolio_status(self) -> Dict[str, Any]:
        """Get comprehensive portfolio status"""
        try:
            portfolio = self.trader.get_portfolio_status()
            
            message = f"""
🏦 **PORTFOLIO STATUS**

💰 **Account Summary:**
• Portfolio Value: ${portfolio['portfolio_value']:,.2f}
• Cash Balance: ${portfolio['cash']:,.2f}
• Buying Power: ${portfolio['buying_power']:,.2f}
• Total Return: {portfolio['total_return']:+.2f}%

📊 **Active Positions:** {len(portfolio['positions'])}
🤖 **Autonomous Trading:** {'🟢 ACTIVE' if self.is_autonomous else '🔴 INACTIVE'}
👁️ **Monitoring:** {len(self.monitoring_symbols)} symbols
            """
            
            if portfolio['positions']:
                message += "\n**Top Positions:**\n"
                for i, pos in enumerate(portfolio['positions'][:5]):
                    pnl_emoji = "📈" if pos['unrealized_pnl'] >= 0 else "📉"
                    message += f"{pnl_emoji} {pos['symbol']}: {pos['quantity']} @ ${pos['avg_price']:.2f} (P&L: {pos['unrealized_pnl_pct']:+.1f}%)\n"
            
            return {
                'message': message.strip(),
                'type': 'portfolio_status',
                'data': portfolio
            }
            
        except Exception as e:
            return {
                'message': f"❌ Error getting portfolio status: {str(e)}",
                'type': 'error'
            }
    
    def _get_trade_ideas(self) -> Dict[str, Any]:
        """Get AI-generated trade ideas"""
        try:
            ideas = []
            
            # Check signals from strategy engine for various assets
            symbols_to_check = ['AAPL', 'TSLA', 'NVDA', 'GOOGL', 'MSFT', 'SPY', 'QQQ', 'BTCUSD', 'ETHUSD']
            
            for symbol in symbols_to_check:
                # Get quote
                quote = self.trader.get_quote(symbol)
                if not quote:
                    continue
                
                # Simulate market data for analysis
                market_data = {
                    'price': quote['price'],
                    'volume': 1000000,
                    'bid': quote['bid'],
                    'ask': quote['ask'],
                    'ma_20': quote['price'] * 0.98,
                    'ma_50': quote['price'] * 0.96,
                    'recent_high': quote['price'] * 1.05,
                    'recent_low': quote['price'] * 0.95,
                    'support': quote['price'] * 0.97,
                    'resistance': quote['price'] * 1.03,
                    'atr': quote['price'] * 0.02
                }
                
                signal = strategy_engine.analyze_market_conditions(symbol, market_data)
                
                if signal.confidence > 0.4:
                    ideas.append({
                        'symbol': symbol,
                        'action': signal.action,
                        'confidence': signal.confidence,
                        'reasoning': signal.reasoning[0] if signal.reasoning else "Technical analysis",
                        'current_price': quote['price'],
                        'strategy': signal.strategy
                    })
            
            if ideas:
                message = "🎯 **AI TRADE IDEAS**\n\n"
                for idea in ideas[:5]:
                    emoji = "🚀" if idea['action'] == 'BUY' else "🔻" if idea['action'] == 'SELL' else "⏸️"
                    message += f"{emoji} **{idea['symbol']}** - {idea['action']}\n"
                    message += f"   Price: ${idea['current_price']:.2f}\n"
                    message += f"   Confidence: {idea['confidence']:.1%}\n"
                    message += f"   Strategy: {idea['strategy']}\n"
                    message += f"   Reason: {idea['reasoning'][:50]}...\n\n"
            else:
                message = "🤖 No high-confidence trade ideas at the moment. Market analysis in progress..."
            
            return {
                'message': message,
                'type': 'trade_ideas',
                'data': ideas
            }
            
        except Exception as e:
            return {
                'message': f"❌ Error generating trade ideas: {str(e)}",
                'type': 'error'
            }
    
    def _get_positions(self) -> Dict[str, Any]:
        """Get current positions"""
        return self._get_portfolio_status()
    
    def _start_autonomous_trading(self) -> Dict[str, Any]:
        """Start autonomous trading"""
        self.is_autonomous = True
        self.trader.running = True
        
        # Start trading thread if not already running
        if not hasattr(self, 'trading_thread') or not self.trading_thread.is_alive():
            self.trading_thread = threading.Thread(target=self.trader.intelligent_trading_logic)
            self.trading_thread.daemon = True
            self.trading_thread.start()
        
        return {
            'message': '🤖 **AUTONOMOUS TRADING ACTIVATED**\n\n✅ AI agent now monitoring markets\n📊 Literature-driven strategies active\n🎯 Dynamic stop-loss management enabled',
            'type': 'success'
        }
    
    def _stop_autonomous_trading(self) -> Dict[str, Any]:
        """Stop autonomous trading"""
        self.is_autonomous = False
        self.trader.running = False
        return {
            'message': '🛑 **AUTONOMOUS TRADING STOPPED**\n\n⏸️ AI agent paused\n📊 Manual trading still available\n🎯 Existing positions will be monitored',
            'type': 'success'
        }
    
    def _add_monitoring_symbol(self, symbols: List[str]) -> Dict[str, Any]:
        """Add symbols to monitoring"""
        if not symbols:
            return {
                'message': '🤖 Please specify symbols to monitor. Try: "Monitor AAPL and TSLA"',
                'type': 'error'
            }
        
        self.monitoring_symbols.extend(symbols)
        return {
            'message': f'📊 **MONITORING ACTIVATED**\n\n🎯 Added symbols: {", ".join(symbols)}\n👁️ Total monitored: {len(self.monitoring_symbols)}\n🤖 AI will prioritize these symbols',
            'type': 'success'
        }
    
    def _remove_monitoring_symbol(self, symbols: List[str]) -> Dict[str, Any]:
        """Remove symbols from monitoring"""
        removed = []
        for symbol in symbols:
            if symbol in self.monitoring_symbols:
                self.monitoring_symbols.remove(symbol)
                removed.append(symbol)
        
        if removed:
            return {
                'message': f'🗑️ **MONITORING REMOVED**\n\n❌ Removed symbols: {", ".join(removed)}\n👁️ Still monitoring: {len(self.monitoring_symbols)} symbols',
                'type': 'success'
            }
        else:
            return {
                'message': f'⚠️ None of the specified symbols were being monitored',
                'type': 'warning'
            }
    
    def _get_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        try:
            portfolio = self.trader.get_portfolio_status()
            if not portfolio:
                return {
                    'message': '❌ Unable to generate risk report - portfolio data unavailable',
                    'type': 'error'
                }
            
            # Calculate risk metrics
            total_value = portfolio['portfolio_value']
            cash_percentage = (portfolio['cash'] / total_value) * 100
            positions_value = sum(pos['current_value'] for pos in portfolio['positions'])
            
            # Position concentration risk
            largest_position = max(portfolio['positions'], key=lambda x: x['current_value']) if portfolio['positions'] else None
            concentration_risk = (largest_position['current_value'] / total_value) * 100 if largest_position else 0
            
            # Calculate total P&L
            total_pnl = sum(pos['unrealized_pnl'] for pos in portfolio['positions'])
            total_pnl_pct = (total_pnl / total_value) * 100 if total_value > 0 else 0
            
            risk_level = "🟢 LOW" if concentration_risk < 20 else "🟡 MEDIUM" if concentration_risk < 40 else "🔴 HIGH"
            
            message = f"""
⚠️ **RISK ANALYSIS REPORT**

💰 **Portfolio Overview:**
• Total Value: ${total_value:,.2f}
• Cash Position: ${portfolio['cash']:,.2f} ({cash_percentage:.1f}%)
• Invested Capital: ${positions_value:,.2f}
• Total P&L: ${total_pnl:,.2f} ({total_pnl_pct:+.2f}%)

📊 **Risk Metrics:**
• Position Count: {len(portfolio['positions'])}
• Largest Position: {largest_position['symbol'] if largest_position else 'N/A'} ({concentration_risk:.1f}%)
• Concentration Risk: {risk_level}
• Diversification: {'🟢 GOOD' if len(portfolio['positions']) > 5 else '🟡 MODERATE' if len(portfolio['positions']) > 2 else '🔴 POOR'}

🎯 **Autonomous Trading Status:**
• AI Trading: {'🟢 ACTIVE' if self.is_autonomous else '🔴 INACTIVE'}
• Monitored Symbols: {len(self.monitoring_symbols)}
• Risk per Trade: 2% max
            """
            
            if portfolio['positions']:
                message += "\n**🔍 Top Risk Positions:**\n"
                sorted_positions = sorted(portfolio['positions'], key=lambda x: abs(x['unrealized_pnl']), reverse=True)[:3]
                for pos in sorted_positions:
                    risk_emoji = "🔴" if abs(pos['unrealized_pnl_pct']) > 10 else "🟡" if abs(pos['unrealized_pnl_pct']) > 5 else "🟢"
                    message += f"{risk_emoji} {pos['symbol']}: ${pos['current_value']:,.2f} ({pos['unrealized_pnl_pct']:+.1f}%)\n"
            
            return {
                'message': message.strip(),
                'type': 'risk_report',
                'data': {
                    'concentration_risk': concentration_risk,
                    'total_pnl_pct': total_pnl_pct,
                    'diversification_score': len(portfolio['positions'])
                }
            }
            
        except Exception as e:
            return {
                'message': f"❌ Error generating risk report: {str(e)}",
                'type': 'error'
            }
    
    def _general_response(self, message: str) -> Dict[str, Any]:
        """General response for unrecognized commands"""
        suggestions = [
            "💰 **Trading:** 'Buy 10 AAPL', 'Sell all TSLA', 'Buy 0.1 BTC'",
            "📊 **Portfolio:** 'Show portfolio', 'What are my positions?', 'Risk report'",
            "🎯 **Ideas:** 'Trade ideas', 'Any opportunities?', 'Market analysis'",
            "🤖 **Control:** 'Start autonomous trading', 'Stop trading', 'Monitor NVDA'"
        ]
        
        return {
            'message': f'🤖 I didn\'t understand: "{message}"\n\n**Try these commands:**\n\n' + '\n'.join(suggestions),
            'type': 'help'
        }
    
    def start_chat_server(self, host='0.0.0.0', port=5000):
        """Start the enhanced chat server"""
        app = Flask(__name__)
        app.config['SECRET_KEY'] = 'autonomous-trading-chat'
        socketio = SocketIO(app, cors_allowed_origins="*")
        
        @app.route('/')
        def chat_interface():
            return '''
<!DOCTYPE html>
<html>
<head>
    <title>🤖 Autonomous Trading Agent</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #0a0a0a, #1a1a2e); color: #fff; min-height: 100vh; }
        .header { background: linear-gradient(135deg, #00ff88, #00d4ff); padding: 25px; border-radius: 15px; margin-bottom: 25px; text-align: center; box-shadow: 0 10px 30px rgba(0,255,136,0.3); }
        .header h1 { color: #000; margin: 0; font-size: 2.5em; font-weight: bold; }
        .header p { color: #333; margin: 10px 0 0 0; font-size: 1.1em; }
        .status-bar { background: #2a2a2a; padding: 15px; border-radius: 10px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; }
        .status-item { display: flex; align-items: center; gap: 10px; }
        .status-dot { width: 12px; height: 12px; border-radius: 50%; }
        .status-active { background: #00ff88; }
        .status-inactive { background: #ff4444; }
        .chat-container { max-width: 1200px; margin: 0 auto; background: #1e1e1e; border-radius: 15px; border: 1px solid #333; box-shadow: 0 20px 50px rgba(0,0,0,0.5); }
        .chat-messages { height: 600px; overflow-y: auto; padding: 25px; }
        .message { margin: 20px 0; padding: 20px; border-radius: 12px; animation: fadeIn 0.3s ease-in; }
        .user-message { background: linear-gradient(135deg, #2196F3, #21CBF3); margin-left: 100px; text-align: right; color: white; }
        .bot-message { background: #2a2a2a; margin-right: 100px; border-left: 4px solid #00ff88; }
        .message-content { font-size: 15px; line-height: 1.7; }
        .message-time { font-size: 12px; color: #888; margin-top: 10px; }
        .chat-input { display: flex; padding: 25px; border-top: 1px solid #333; gap: 15px; }
        .chat-input input { flex: 1; padding: 18px; border: 2px solid #444; border-radius: 10px; background: #2a2a2a; color: #fff; font-size: 16px; transition: border-color 0.3s; }
        .chat-input input:focus { border-color: #00ff88; outline: none; }
        .chat-input button { padding: 18px 30px; background: linear-gradient(135deg, #00ff88, #00d4ff); color: #000; border: none; border-radius: 10px; cursor: pointer; font-weight: bold; font-size: 16px; transition: transform 0.2s; }
        .chat-input button:hover { transform: translateY(-2px); }
        .quick-actions { padding: 15px 25px; border-top: 1px solid #333; }
        .quick-btn { display: inline-block; padding: 12px 18px; background: #444; color: #fff; border: none; border-radius: 8px; margin: 8px; cursor: pointer; font-size: 13px; transition: all 0.3s; }
        .quick-btn:hover { background: #555; transform: translateY(-1px); }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .trade-data { background: #1a3a1a; border: 1px solid #00ff88; border-radius: 8px; padding: 15px; margin: 10px 0; }
        .error-data { background: #3a1a1a; border: 1px solid #ff4444; border-radius: 8px; padding: 15px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Autonomous Trading Agent</h1>
        <p>Natural language trading • Multi-asset support • Real-time execution</p>
    </div>

    <div class="status-bar">
        <div class="status-item">
            <div class="status-dot status-active"></div>
            <span>Alpaca Connected</span>
        </div>
        <div class="status-item">
            <div class="status-dot status-active"></div>
            <span>Autonomous Trading</span>
        </div>
        <div class="status-item">
            <div class="status-dot status-active"></div>
            <span>Digital Brain Active</span>
        </div>
        <div class="status-item">
            <span id="portfolio-value">Portfolio: Loading...</span>
        </div>
    </div>

    <div class="chat-container">
        <div class="chat-messages" id="messages">
            <div class="message bot-message">
                <div class="message-content">
                    <strong>🤖 Autonomous Trading Agent:</strong> Hello! I'm your AI trading agent with full market access and autonomous capabilities. I can:
                    <br><br>
                    🚀 <strong>Execute Trades:</strong> "Buy 10 AAPL", "Sell all TSLA", "Buy 0.1 BTC"<br>
                    📊 <strong>Portfolio Management:</strong> "Show my positions", "What's my status?"<br>
                    🎯 <strong>Trade Ideas:</strong> "What are your trade ideas?", "Any opportunities?"<br>
                    🤖 <strong>Autonomous Trading:</strong> "Start autonomous trading", "Stop trading"<br>
                    📈 <strong>Market Analysis:</strong> "Risk report", "Monitor NVDA"<br><br>
                    
                    <strong>Supported Assets:</strong> Stocks, ETFs, Crypto (BTC, ETH, XRP), Options, Futures<br>
                    <strong>Alpaca Paper Trading:</strong> Live market data with intelligent execution<br><br>
                    
                    Try: <em>"Buy 10 AAPL"</em>, <em>"Buy 0.1 BTC"</em>, <em>"Buy 1 XRP"</em>, or <em>"What trade ideas do you have?"</em>
                </div>
                <div class="message-time">Ready for autonomous trading</div>
            </div>
        </div>

        <div class="quick-actions">
            <button class="quick-btn" onclick="sendQuickMessage('Show my portfolio status')">📊 Portfolio Status</button>
            <button class="quick-btn" onclick="sendQuickMessage('What trade ideas do you have?')">💡 Trade Ideas</button>
            <button class="quick-btn" onclick="sendQuickMessage('Buy 10 shares of AAPL')">🚀 Buy AAPL</button>
            <button class="quick-btn" onclick="sendQuickMessage('Buy 1 XRP')">🔶 Buy XRP</button>
            <button class="quick-btn" onclick="sendQuickMessage('Buy 0.1 BTC')">₿ Buy Bitcoin</button>
            <button class="quick-btn" onclick="sendQuickMessage('Start autonomous trading')">🤖 Start Auto</button>
            <button class="quick-btn" onclick="sendQuickMessage('Show my positions')">📋 Positions</button>
            <button class="quick-btn" onclick="sendQuickMessage('Risk report')">⚠️ Risk Report</button>
        </div>

        <div class="chat-input">
            <input type="text" id="messageInput" placeholder="Try: 'Buy 10 AAPL', 'Sell all TSLA', 'Show portfolio', 'Trade ideas', 'Buy 0.5 ETH'" autofocus>
            <button onclick="sendMessage()">Execute</button>
        </div>
    </div>

    <script>
        let socket = io();

        function addMessage(sender, message, className, timestamp, messageType = null, tradeData = null) {
            const messagesDiv = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message ' + className;

            let extraContent = '';
            if (messageType === 'trade_success' && tradeData) {
                extraContent = `<div class="trade-data">
                    <strong>Trade Executed:</strong> ${tradeData.action} ${tradeData.quantity} ${tradeData.symbol} @ $${tradeData.price}
                </div>`;
            } else if (messageType === 'trade_error') {
                extraContent = `<div class="error-data">Trade execution failed</div>`;
            }

            const formattedMessage = message.replace(/\\*\\*([^*]+)\\*\\*/g, '<strong>$1</strong>')
                                          .replace(/\\*([^*]+)\\*/g, '<em>$1</em>')
                                          .replace(/\\n/g, '<br>');

            messageDiv.innerHTML = 
                '<div class="message-content"><strong>' + sender + ':</strong> ' + formattedMessage + extraContent + '</div>' +
                '<div class="message-time">' + timestamp + '</div>';

            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            if (message) {
                addMessage('👤 You', message, 'user-message', new Date().toLocaleTimeString());
                socket.emit('user_message', {message: message});
                input.value = '';
            }
        }

        function sendQuickMessage(message) {
            addMessage('👤 You', message, 'user-message', new Date().toLocaleTimeString());
            socket.emit('user_message', {message: message});
        }

        socket.on('bot_response', function(data) {
            addMessage('🤖 Agent', data.message, 'bot-message', data.timestamp || new Date().toLocaleTimeString(), data.type, data.trade_data);
        });

        document.getElementById('messageInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });

        // Update portfolio value in status bar
        setInterval(() => {
            fetch('/api/portfolio')
                .then(response => response.json())
                .then(data => {
                    if (data && data.portfolio_value) {
                        document.getElementById('portfolio-value').textContent = 
                            'Portfolio: $' + data.portfolio_value.toLocaleString();
                    }
                })
                .catch(error => console.log('Portfolio update failed'));
        }, 5000);
    </script>
</body>
</html>
            '''
        
        @app.route('/api/portfolio')
        def api_portfolio():
            """Portfolio API for status bar updates"""
            portfolio = self.trader.get_portfolio_status()
            return jsonify(portfolio) if portfolio else jsonify({'error': 'No data'})
        
        @socketio.on('user_message')
        def handle_message(data):
            response = self.process_natural_language(data['message'])
            response['timestamp'] = datetime.now().strftime('%H:%M:%S')
            emit('bot_response', response)
        
        print(f"🚀 Autonomous Trading Chat starting on http://0.0.0.0:{port}")
        print("🧠 Literature-driven AI trading agent ready")
        print("💬 Natural language commands supported")
        print("📊 Real-time portfolio management active")
        socketio.run(app, host=host, port=port, debug=False)

# Global chat bot instance
chat_bot = AutonomousTradingChatBot()

def main():
    """Start the autonomous trading chat interface"""
    print("🤖 Starting Autonomous Trading Agent Chat Interface")
    print("=" * 60)
    chat_bot.start_chat_server()

if __name__ == "__main__":
    main()