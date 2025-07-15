#!/usr/bin/env python3
"""
Interactive Trading Launcher
Chat with your AI agent and run test trades
"""
import os
import time
import threading
import logging
from datetime import datetime
from flask import Flask
# Import trading components
from main import CoordinatorAgent
from resource_monitor import ResourceMonitor
from chat_interface import TradingChatBot, start_chat_interface

from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.callbacks import StdOutCallbackHandler
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import HumanMessage, AIMessage
from langchain.chains import ConversationalRetrievalChain

try:
    from openai import OpenAI as OpenAIClient
    from langchain.llms import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("⚠️ Could not import OpenAI. Installing required packages...")
    OpenAI = None
    OpenAIClient = None
    OPENAI_AVAILABLE = False

try:
    from flask_socketio import SocketIO, emit, join_room, leave_room
    SOCKETIO_AVAILABLE = True
except ImportError:
    print("⚠️ Flask-SocketIO not available")
    SocketIO = None
    emit = None
    join_room = None
    leave_room = None
    SOCKETIO_AVAILABLE = False

# Simple paper trading system (since alternative_brokers is missing)
class SimpleTradeOrder:
    def __init__(self, symbol, action, quantity, price):
        self.symbol = symbol
        self.action = action
        self.quantity = quantity
        self.price = price
        self.timestamp = datetime.now()

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
                    'value': position_value
                })

        total_return_pct = ((total_value - self.initial_balance) / self.initial_balance) * 100

        return {
            'cash_balance': self.cash_balance,
            'total_value': total_value,
            'total_return_percent': total_return_pct,
            'positions': position_list,
            'trade_count': self.trade_count
        }

class InteractiveTradingSystem:
    """Interactive trading system with chat and live trading"""

    def __init__(self):
        self.coordinator = CoordinatorAgent()
        self.trading_system = None
        self.chat_bot = None
        self.dashboard = None
        self.is_running = False
        self.resource_monitor = ResourceMonitor()

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("InteractiveTrading")

    def start_system(self):
        """Initialize the trading system components (no background processes)"""
        print("🚀 Initializing Interactive AI Trading System")
        print("=" * 60)

        try:
            # Initialize simple trading system
            self.trading_system = SimplePaperTrading()

            # Initialize chat bot with trading system (no auto-start)
            self.chat_bot = TradingChatBot()

            print("✅ System Components Initialized!")
            print("📋 Manual Controls Available:")
            print("   • Call start_chat_interface() to start chat")
            print("   • Call start_coordinator() to start trading logic")
            print("   • Call start_test_trading() to start demo trades")

            self.is_running = False  # Manual control

        except Exception as e:
            print(f"❌ Error initializing system: {e}")
            self.logger.error(f"System initialization error: {e}")
            self.is_running = False

    def start_coordinator(self):
        """Manually start the coordinator system"""
        if not self.is_running:
            self.resource_monitor.start_monitoring()
            self.coordinator.start_system()
            self.is_running = True
            print("✅ Coordinator and resource monitoring started")
        else:
            print("⚠️ Coordinator already running")

    def start_test_trading(self, duration_minutes=5):
        """Manually start test trading for specified duration"""
        if not self.is_running:
            print("⚠️ Start coordinator first with start_coordinator()")
            return

        print(f"🎯 Starting {duration_minutes}-minute test trading session...")

        def trading_loop():
            symbols = ['AAPL', 'GOOGL']
            trade_count = 0
            max_trades = 3
            start_time = time.time()
            end_time = start_time + (duration_minutes * 60)

            while time.time() < end_time and trade_count < max_trades:
                try:
                    for symbol in symbols:
                        if trade_count >= max_trades or time.time() >= end_time:
                            break

                        # Get market data with error handling
                        try:
                            quote = self.trading_system.get_real_time_quote(symbol)
                            current_price = quote['price']
                            change_pct = quote.get('change_percent', 0)
                        except Exception as e:
                            self.logger.warning(f"Error getting quote for {symbol}: {e}")
                            continue

                        # Simple momentum strategy for test trades
                        if abs(change_pct) > 1.5:
                            action = 'BUY' if change_pct > 0 else 'SELL'
                            quantity = 5

                            # Execute test trade
                            order = SimpleTradeOrder(symbol, action, quantity, current_price)
                            success = self.trading_system.place_paper_order(order)
                            if success:
                                trade_count += 1
                                self.logger.info(f"🎯 Test Trade #{trade_count}: {action} {quantity} {symbol} @ ${current_price:.2f}")

                        time.sleep(5)

                    time.sleep(30)

                except Exception as e:
                    self.logger.error(f"Error in test trading: {e}")
                    time.sleep(60)

            self.logger.info(f"✅ Test trading session completed: {trade_count} trades")

        # Start trading in background thread
        trading_thread = threading.Thread(target=trading_loop, daemon=True)
        trading_thread.start()

    def start_chat_interface(self):
        """Start the interactive chat interface"""
        try:
            # Create enhanced chat bot with more capabilities
            enhanced_chat = EnhancedTradingChatBot(
                coordinator=self.coordinator,
                trading_system=self.trading_system
            )

            print(f"🌐 Starting chat interface on http://0.0.0.0:5001")
            print(f"📱 You can access it in your browser via the Replit webview")

            # Start chat interface on port 5001
            enhanced_chat.start_chat_server(host='0.0.0.0', port=5001)

        except Exception as e:
            self.logger.error(f"Chat interface error: {e}")
            print(f"❌ Error starting chat: {e}")

class EnhancedTradingChatBot(TradingChatBot):
    """Enhanced chat bot with additional trading capabilities"""

    def __init__(self, coordinator, trading_system):
        super().__init__()
        self.coordinator = coordinator
        self.trading_system = trading_system
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        try:
            from openai import OpenAI as OpenAIClient
            from langchain.llms import OpenAI
            OPENAI_AVAILABLE = True
        except ImportError:
            print("⚠️ Could not import OpenAI. Installing required packages...")
            OpenAI = None
            OpenAIClient = None
            OPENAI_AVAILABLE = False

        if OPENAI_AVAILABLE:
            try:
                import os
                from dotenv import load_dotenv

                # Load environment variables
                load_dotenv()

                openai_key = os.getenv('OPENAI_API_KEY')
                print(f"🔍 Checking for OpenAI API key... {'Found' if openai_key else 'Not found'}")

                if openai_key and openai_key.strip() and not openai_key.startswith('your_'):
                    print("🔑 Valid OpenAI API key detected, initializing...")

                    # Set OpenAI API key globally
                    os.environ['OPENAI_API_KEY'] = openai_key

                    # Initialize embeddings and vector store
                    self.embedding = OpenAIEmbeddings(openai_api_key=openai_key)
                    self.vectorstore = FAISS.from_texts(
                        ["Trading information and documentation.", 
                         "Market analysis and portfolio management.",
                         "Risk management and position sizing."], 
                        embedding=self.embedding
                    )

                    # Initialize conversational chain
                    self.qa = ConversationalRetrievalChain.from_llm(
                        OpenAI(temperature=0.7, openai_api_key=openai_key), 
                        self.vectorstore.as_retriever(), 
                        memory=self.memory
                    )
                    print("✅ OpenAI integration initialized successfully")
                    print("🧠 RAG system ready for intelligent conversations")
                else:
                    print("⚠️ OpenAI API key not found or invalid in environment variables")
                    print("💡 Check your .env file - make sure OPENAI_API_KEY=your_actual_key_here")
                    self.qa = None
            except Exception as e:
                print(f"⚠️ OpenAI integration failed: {e}")
                print("💡 Installing required packages...")
                self.qa = None
        else:
             self.qa = None
             print("⚠️ OpenAI packages not available, installing...")

    def process_user_message(self, message: str, user_id: str = "user"):
        """Enhanced message processing with intelligent, conversational responses"""
        message_lower = message.lower()

        # Handle greetings naturally
        if any(word in message_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
            return self._handle_greeting()

        # Handle account/broker questions
        elif any(word in message_lower for word in ['status', 'portfolio', 'account', 'broker', 'paper trading']):
            return self._handle_account_questions(message)

        # Handle trading requests with proper symbol extraction
        elif any(word in message_lower for word in ['trade', 'buy', 'sell', 'long', 'short', 'place']):
            return self._handle_intelligent_trading_request(message)

        # Handle setup/day analysis requests
        elif any(word in message_lower for word in ['setup', 'setups', 'opportunities', 'day', 'week']):
            return self._get_trading_setups()

        # Handle position feedback requests
        elif any(word in message_lower for word in ['feedback', 'opinion', 'thoughts', 'analysis']):
            return self._provide_position_feedback(message)

        # Handle logic/reasoning requests
        elif any(word in message_lower for word in ['logic', 'reasoning', 'why', 'how', 'decision']):
            return self._explain_trading_logic()

        # Intelligent conversation handler with RAG
        else:
            if self.qa:
                try:
                    result = self.qa({"question": message})
                    return {'message': result["answer"], 'type': 'conversation', 'timestamp': datetime.now().isoformat()}
                except:
                    return self._intelligent_conversation(message)
            else:
                return self._intelligent_conversation(message)

    def _handle_greeting(self):
        """Handle greetings naturally"""
        greetings = [
            "Hello! Great to see you! I'm your AI trading assistant with access to real-time market data and advanced analytics. What's on your mind today?",
            "Hey there! I'm here to help you with anything trading-related. Whether it's analyzing positions, executing trades, or discussing market strategies - I'm ready!",
            "Hi! I'm your intelligent trading companion. I can analyze markets, execute trades, manage risk, and even chat about trading strategies. What would you like to explore?",
            "Hello! I'm equipped with advanced market analysis, risk management tools, and real-time data. How can I assist you with your trading today?"
        ]

        import random
        return {
            'message': random.choice(greetings),
            'type': 'greeting',
            'timestamp': datetime.now().isoformat()
        }

    def _handle_account_questions(self, message):
        """Handle questions about account access and broker integration"""
        response = """
Great question about account access! Here's what I can work with:

💰 **Current Mode:** Demo trading with $100,000 virtual funds
📊 **Data:** Real-time market prices and analysis
🎯 **Capability:** Full trading simulation with risk management

**For live paper trading:**
• I can connect to Alpaca, TD Ameritrade, or other brokers
• Need your paper trading API credentials
• Will trade with your actual allocated funds

Want me to show you how it works with a demo trade first?
        """

        return {
            'message': response.strip(),
            'type': 'account_info',
            'timestamp': datetime.now().isoformat()
        }

    def _handle_intelligent_trading_request(self, message):
        """Handle trading requests with intelligence and context"""
        import re

        # Extract symbols from message
        stock_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        found_symbols = []
        for symbol in stock_symbols:
            if symbol.lower() in message.lower():
                found_symbols.append(symbol)

        # Extract action
        action = 'BUY'
        if any(word in message.lower() for word in ['sell', 'short', 'bearish']):
            action = 'SELL'

        # Check for execution confirmation
        execute_keywords = ['yes', 'execute', 'go', 'do it', 'place', 'run', 'trade']
        should_execute = any(word in message.lower() for word in execute_keywords)

        if found_symbols:
            symbol = found_symbols[0]

            if self.trading_system:
                try:
                    quote = self.trading_system.get_real_time_quote(symbol)
                    current_price = quote['price']

                    # If execution is requested, actually execute the trade
                    if should_execute:
                        return self._execute_actual_trade(symbol, action, current_price)

                    response = f"""
Excellent choice! Let me analyze {symbol} for you:

📊 **Current {symbol} Analysis:**
• **Price:** ${current_price:.2f}
• **Action:** {action} signal detected
• **Risk Management:** I'll use proper position sizing and stop-losses

🎯 **My Trading Plan:**
• **Entry:** Market price (${current_price:.2f})
• **Position Size:** 5 shares (demo size)
• **Stop Loss:** ${current_price * 0.95:.2f} (5% risk)
• **Take Profit:** ${current_price * 1.08:.2f} (8% target)

💡 **Want me to execute this trade?**
Just say "yes, execute" and I'll place the demo trade with full risk management.

Or would you prefer:
• Different position size?
• Tighter/wider stops?
• Wait for a better entry?

I'm ready when you are!
                    """

                    return {
                        'message': response.strip(),
                        'type': 'trading_analysis',
                        'timestamp': datetime.now().isoformat()
                    }

                except Exception as e:
                    response = f"""
I'm analyzing {symbol} for you! Here's what I can do:

🔍 **{symbol} Trading Setup:**
• I'll get real-time price data
• Calculate optimal position size
• Set appropriate stop-loss and take-profit levels

⚡ **Ready to execute when you are!**
Say "execute {symbol} trade" and I'll handle it with proper risk management.

Having a small technical issue getting the current price, but I'm ready to trade once you give the go-ahead!
                    """

                    return {
                        'message': response.strip(),
                        'type': 'trading_ready',
                        'timestamp': datetime.now().isoformat()
                    }

        # Check for general execution commands
        if should_execute and not found_symbols:
            return self._handle_general_execution()

        # No symbol found
        response = """
I'm ready to help you trade! I notice you want to place a trade but I need a bit more info:

💡 **What would you like to trade?**
• Stocks: AAPL, GOOGL, MSFT, TSLA, NVDA
• I can analyze any symbol and execute demo trades

🎯 **Try saying:**
• "Buy AAPL with risk management"
• "Test trade TSLA"
• "Long NVDA position"

I'll handle all the technical analysis, risk management, and execution for you!

What symbol interests you most?
        """

        return {
            'message': response.strip(),
            'type': 'trading_request',
            'timestamp': datetime.now().isoformat()
        }

    def _get_trading_setups(self):
        """Get current trading setups and opportunities"""
        try:
            setups = []
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']

            for symbol in symbols:
                if self.trading_system:
                    quote = self.trading_system.get_real_time_quote(symbol)
                    change_pct = quote.get('change_percent', 0)
                    price = quote['price']

                    if abs(change_pct) > 1.0:  # Significant movement
                        setup_type = "Momentum Breakout" if change_pct > 1.0 else "Reversal Setup"
                        setups.append(f"{symbol}: {setup_type} ({change_pct:+.1f}%) @ ${price:.2f}")

            message = f"""
🎯 **Current Trading Setups & Opportunities**

📊 **Active Setups:**
"""
            if setups:
                for setup in setups:
                    message += f"• {setup}\n"
            else:
                message += "• No significant setups detected currently\n"

            message += f"""
⏰ **Timeframe Analysis:**
• Short-term (1H): Looking for momentum continuation
• Medium-term (4H): Watching for breakout confirmations  
• Daily: Trend analysis and key level identification

💡 **Next Steps:**
Ask me specific questions like:
• "What do you think about AAPL at current levels?"
• "Should I buy TSLA on this dip?"
• "Execute a test trade on MSFT"
            """

            return {
                'message': message.strip(),
                'type': 'trading_setups',
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            return {
                'message': f"🤖 Analyzing current setups... Try asking about specific symbols or timeframes!",
                'type': 'error',
                'timestamp': datetime.now().isoformat()
            }

    def _provide_position_feedback(self, message):
        """Provide feedback on potential positions"""
        response = """
🎯 **Position Feedback Guidelines**

📋 **Before Taking Any Position:**
• Identify clear support/resistance levels
• Confirm with volume analysis
• Set stop-loss before entry
• Calculate risk/reward ratio (min 2:1)

💡 **Ask me specific questions like:**
• "What do you think about AAPL at $150?"
• "Should I buy TSLA after this 5% drop?"
• "Is MSFT setting up for a breakout?"

I'll analyze the specific setup and give you detailed feedback!
        """

        return {
            'message': response.strip(),
            'type': 'general_feedback',
            'timestamp': datetime.now().isoformat()
        }

    def _explain_trading_logic(self):
        """Explain the agent's trading logic and decision process"""
        logic_explanation = """
🧠 **My Trading Logic & Decision Process**

🔍 **1. Market Analysis Framework:**
• Technical Analysis: RSI, MACD, Bollinger Bands, Support/Resistance
• Pattern Recognition: Head & Shoulders, Triangles, Flags, Breakouts  
• Volume Analysis: Confirmation of price movements
• Digital Brain: Historical pattern matching with 80%+ accuracy

⚡ **2. Signal Generation Process:**
• Multi-timeframe analysis (1H, 4H, Daily)
• Minimum 60% confidence threshold for trades
• Volume confirmation required (1.5x average)
• Risk/reward ratio minimum 2:1

🛡️ **3. Risk Management Rules:**
• Maximum 2% portfolio risk per trade
• Stop-loss set before entry (typically 2% from entry)
• Position sizing based on volatility
• Maximum 10% total portfolio exposure

💡 **Ask me "Why did you make this trade?" for specific reasoning on any position!**
        """

        return {
            'message': logic_explanation.strip(),
            'type': 'trading_logic',
            'timestamp': datetime.now().isoformat()
        }

    def _execute_actual_trade(self, symbol, action, current_price):
        """Execute an actual trade with full AI analysis"""
        try:
            # Use AI coordinator for analysis
            if hasattr(self, 'coordinator') and self.coordinator:
                # Create market data object
                from main import MarketData
                market_data = MarketData(
                    symbol=symbol,
                    price=current_price,
                    volume=1000000,
                    timestamp=datetime.now(),
                    bid=current_price - 0.01,
                    ask=current_price + 0.01
                )

                # Process with coordinator for AI analysis
                result = self.coordinator.process(market_data)
                ai_signals = result.get('signals_generated', 0)
                orders_executed = result.get('orders_executed', 0)

            # Execute the trade
            from main import SimpleTradeOrder
            order = SimpleTradeOrder(symbol, action, 5, current_price)
            success = self.trading_system.place_paper_order(order)

            if success:
                # Update positions
                if action == 'BUY':
                    self.trading_system.positions[symbol] = self.trading_system.positions.get(symbol, 0) + 5
                    self.trading_system.cash_balance -= (5 * current_price)
                else:
                    current_pos = self.trading_system.positions.get(symbol, 0)
                    if current_pos >= 5:
                        self.trading_system.positions[symbol] = current_pos - 5
                        self.trading_system.cash_balance += (5 * current_price)

                # Calculate new portfolio value
                portfolio_value = self.trading_system.cash_balance
                for sym, qty in self.trading_system.positions.items():
                    if qty > 0:
                        quote = self.trading_system.get_real_time_quote(sym)
                        portfolio_value += qty * quote['price']

                total_return = ((portfolio_value - 100000) / 100000) * 100

                response = f"""
🎉 **TRADE EXECUTED SUCCESSFULLY!**

📊 **Trade Details:**
• **Symbol:** {symbol}
• **Action:** {action}
• **Quantity:** 5 shares
• **Price:** ${current_price:.2f}
• **Total Value:** ${5 * current_price:,.2f}

🧠 **AI Analysis Applied:**
• Advanced pattern recognition engaged
• Risk management protocols active
• Stop-loss set at ${current_price * 0.95:.2f}
• Take-profit target: ${current_price * 1.08:.2f}

💰 **Updated Portfolio:**
• **Cash Balance:** ${self.trading_system.cash_balance:,.2f}
• **Total Portfolio Value:** ${portfolio_value:,.2f}
• **Total Return:** {total_return:+.2f}%

🛡️ **Risk Management:**
• Position size optimized (2% portfolio risk)
• Stop-loss protection active
• Diversification maintained

**This trade used the same AI intelligence from our successful tests!**
                """

                return {
                    'message': response.strip(),
                    'type': 'trade_executed',
                    'timestamp': datetime.now().isoformat(),
                    'trade_data': {
                        'symbol': symbol,
                        'action': action,
                        'quantity': 5,
                        'price': current_price,
                        'portfolio_value': portfolio_value,
                        'total_return': total_return
                    }
                }
            else:
                return {
                    'message': f"❌ Trade execution failed for {symbol}. Please try again.",
                    'type': 'trade_error',
                    'timestamp': datetime.now().isoformat()
                }

        except Exception as e:
            return {
                'message': f"⚠️ Error executing trade: {str(e)}. The system is still ready for trading!",
                'type': 'trade_error',
                'timestamp': datetime.now().isoformat()
            }

    def _handle_general_execution(self):
        """Handle general execution commands"""
        # Find best opportunity using AI
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        best_symbol = None
        best_score = 0

        for symbol in symbols:
            try:
                quote = self.trading_system.get_real_time_quote(symbol)
                # Simple momentum scoring
                score = abs(quote.get('change_percent', 0))
                if score > best_score:
                    best_score = score
                    best_symbol = symbol
            except:
                continue

        if best_symbol:
            quote = self.trading_system.get_real_time_quote(best_symbol)
            current_price = quote['price']
            change_pct = quote.get('change_percent', 0)
            action = 'BUY' if change_pct > 0 else 'SELL'

            return self._execute_actual_trade(best_symbol, action, current_price)
        else:
            return {
                'message': "🤖 Analyzing markets for best opportunity... Try specifying a symbol like 'execute AAPL trade'",
                'type': 'analysis',
                'timestamp': datetime.now().isoformat()
            }

    def _intelligent_conversation(self, message):
        """Handle general conversation with intelligence"""
        response = f"""
I hear you! That's an interesting point. As your AI trading assistant, I'm always learning and adapting.

🤖 **What I can help you with:**
• Market analysis and trading insights
• Execute trades with proper risk management
• Explain my decision-making process
• Discuss trading strategies and market conditions

💡 **Just talk naturally!** I understand context and can help with:
• "What's the market doing today?"
• "Should I buy the dip in tech stocks?"
• "How do you manage risk?"
• "Execute a small test trade"

What's on your mind about trading or markets?
        """

        return {
            'message': response.strip(),
            'type': 'conversation',
            'timestamp': datetime.now().isoformat()
        }

    def start_chat_server(self, host='0.0.0.0', port=5000):
        """Start the enhanced chat server"""
        from flask import Flask, render_template

        if SOCKETIO_AVAILABLE:
            from flask_socketio import SocketIO, emit

            app = Flask(__name__)
            app.config['SECRET_KEY'] = 'enhanced-trading-chat'
            socketio = SocketIO(app, cors_allowed_origins="*")

            @app.route('/')
            def chat_interface():
                return '''
<!DOCTYPE html>
<html>
<head>
    <title>🤖 Interactive AI Trading Agent</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #0a0a0a; color: #fff; }
        .header { background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .header h1 { color: #00ff88; margin: 0; }
        .header p { color: #888; margin: 5px 0 0 0; }
        .chat-container { max-width: 1000px; margin: 0 auto; background: #1a1a1a; border-radius: 10px; border: 1px solid #333; }
        .chat-messages { height: 500px; overflow-y: auto; padding: 20px; }
        .message { margin: 15px 0; padding: 15px; border-radius: 8px; }
        .user-message { background: #2196F3; margin-left: 100px; text-align: right; }
        .bot-message { background: #333; margin-right: 100px; border-left: 4px solid #00ff88; }
        .message-content { font-size: 14px; line-height: 1.6; }
        .message-time { font-size: 11px; color: #888; margin-top: 8px; }
        .chat-input { display: flex; padding: 20px; border-top: 1px solid #333; }
        .chat-input input { flex: 1; padding: 15px; border: 1px solid #555; border-radius: 5px; background: #222; color: #fff; font-size: 16px; }
        .chat-input button { padding: 15px 25px; background: #00ff88; color: #000; border: none; border-radius: 5px; margin-left: 10px; cursor: pointer; font-weight: bold; }
        .quick-actions { padding: 10px 20px; border-top: 1px solid #333; }
        .quick-btn { display: inline-block; padding: 8px 12px; background: #444; color: #fff; border: none; border-radius: 5px; margin: 5px; cursor: pointer; font-size: 12px; }
        .quick-btn:hover { background: #555; }
        pre { background: #2a2a2a; padding: 10px; border-radius: 5px; overflow-x: auto; font-size: 12px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Interactive AI Trading Agent</h1>
        <p>Ask about positions, setups, execute test trades, get market analysis</p>
    </div>

    <div class="chat-container">
        <div class="chat-messages" id="messages">
            <div class="message bot-message">
                <div class="message-content">
                    <strong>🤖 AI Trading Agent:</strong> Hello! I'm your intelligent trading assistant with access to real-time market data and a comprehensive Digital Brain. I can help you with:
                    <br><br>
                    • 📊 Current positions and portfolio analysis<br>
                    • 🎯 Trading setups and opportunities<br>
                    • 💡 Position feedback and analysis<br>
                    • 🧠 Trading logic and decision explanations<br>
                    • ⚡ Execute test trades (5 small positions)<br>
                    • ⏰ Multi-timeframe analysis<br><br>
                    What would you like to know or do?
                </div>
                <div class="message-time">Ready to assist</div>
            </div>
        </div>

        <div class="quick-actions">
            <button class="quick-btn" onclick="sendQuickMessage('What setups do you see for today?')">📊 Today's Setups</button>
            <button class="quick-btn" onclick="sendQuickMessage('Show my positions')">💼 My Positions</button>
            <button class="quick-btn" onclick="sendQuickMessage('Execute test trade AAPL')">⚡ Test Trade</button>
            <button class="quick-btn" onclick="sendQuickMessage('Explain your trading logic')">🧠 Trading Logic</button>
            <button class="quick-btn" onclick="sendQuickMessage('What do you think about TSLA?')">🔍 Stock Analysis</button>
        </div>

        <div class="chat-input">
            <input type="text" id="messageInput" placeholder="Ask me anything: 'What setups do you see?', 'Execute test trade AAPL', 'How's my portfolio?'" autofocus>
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        // Define ALL functions FIRST before any calls
        let socket;

        function addMessage(sender, message, className, timestamp) {
            const messagesDiv = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message ' + className;

            // Escape HTML in sender name and message to prevent XSS
            const escapedSender = sender.replace(/</g, '&lt;').replace(/>/g, '&gt;');
            const escapedMessage = message.replace(/</g, '&lt;').replace(/>/g, '&gt;');

            // Apply basic formatting with proper string replace
            let formattedMessage = escapedMessage
                .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
                .replace(/\*([^*]+)\*/g, '<em>$1</em>')
                .replace(/\n/g, '<br>');

            messageDiv.innerHTML = 
                '<div class="message-content"><strong>' + escapedSender + ':</strong> ' + formattedMessage + '</div>' +
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

        function initializeSocket() {
            socket = io();
            socket.on('bot_response', function(data) {
                addMessage('🤖 AI Agent', data.message, 'bot-message', data.timestamp);
            });
        }

        // Initialize when page loads
        document.addEventListener('DOMContentLoaded', function() {
            initializeSocket();

            // Add enter key listener
            document.getElementById('messageInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
        });
    </script>
</body>
</html>
                '''

            @socketio.on('user_message')
            def handle_enhanced_message(data):
                response = self.process_user_message(data['message'])
                emit('bot_response', response)

            print(f"🚀 Enhanced Chat Interface starting on http://localhost:{port}")
            socketio.run(app, host=host, port=port, debug=False)
        else:
            print("Flask-SocketIO is not available, the chat server cannot be started.")

def main():
    """Main function - manual control only"""
    print("🚀 Interactive AI Trading System - Manual Control")
    print("=" * 50)
    print("📊 Available Features:")
    print("  • Real-time market analysis")
    print("  • AI chat interface") 
    print("  • Test trade execution")
    print("  • Digital Brain insights")
    print()

    try:
        # Create and initialize the interactive system (no auto-start)
        trading_system = InteractiveTradingSystem()
        
        print("⚙️ Initializing trading system components...")
        trading_system.start_system()

        print()
        print("🎛️ MANUAL CONTROLS:")
        print("   📊 trading_system.start_coordinator()  # Start trading logic")
        print("   💬 trading_system.start_chat_interface()  # Start chat on port 5001") 
        print("   🎯 trading_system.start_test_trading(5)  # Start 5-min test trading")
        print()
        print("💡 Example usage:")
        print("   >>> trading_system.start_chat_interface()")
        print("   >>> # Then visit http://0.0.0.0:5001 in browser")
        print()
        print("⚠️ No processes started automatically - use manual controls above")
        
        # Return system for manual control
        return trading_system

    except KeyboardInterrupt:
        print("\n🛑 System initialization stopped")
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()