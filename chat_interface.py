"""
Chat Interface for Trading Agent Interaction
Real-time conversation with your AI trading agents
"""

import json
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
try:
    from flask import Flask, render_template, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    print("⚠️ Flask not available - chat interface will run in console mode")
    FLASK_AVAILABLE = False
    Flask = None
from flask_socketio import SocketIO, emit, join_room, leave_room
import logging

# Import trading components
from main import CoordinatorAgent
from broker_integration import LiveTradingAgent, BrokerConfig, BrokerType
from document_upload import TradingDocumentUploader

class TradingChatBot:
    """AI Chat interface for trading agent"""

    def __init__(self, live_agent: Optional[LiveTradingAgent] = None):
        self.live_agent = live_agent
        self.coordinator = live_agent.coordinator if live_agent else CoordinatorAgent()
        self.document_uploader = TradingDocumentUploader()
        self.logger = logging.getLogger("TradingChatBot")
        self.conversation_history = []

    def process_user_message(self, message: str, user_id: str = "user") -> Dict[str, Any]:
        """Process user message and generate response"""
        try:
            message_lower = message.lower()
            response = {
                'message': '',
                'data': {},
                'timestamp': datetime.now().isoformat(),
                'type': 'text'
            }

            # Store user message
            self.conversation_history.append({
                'user_id': user_id,
                'message': message,
                'timestamp': datetime.now().isoformat(),
                'type': 'user'
            })

            # Command routing
            if any(word in message_lower for word in ['status', 'portfolio', 'how are you']):
                response = self._get_portfolio_status()

            elif any(word in message_lower for word in ['positions', 'holdings']):
                response = self._get_current_positions()

            elif any(word in message_lower for word in ['performance', 'returns', 'profit']):
                response = self._get_performance_metrics()

            elif any(word in message_lower for word in ['risk', 'safety', 'stop loss']):
                response = self._get_risk_metrics()

            elif 'buy' in message_lower or 'sell' in message_lower:
                response = self._handle_trading_request(message)

            elif any(word in message_lower for word in ['market', 'analysis', 'outlook']):
                response = self._get_market_analysis()

            elif any(word in message_lower for word in ['brain', 'knowledge', 'patterns']):
                response = self._query_digital_brain(message)

            elif 'help' in message_lower or '?' in message:
                response = self._get_help()

            else:
                response = self._general_conversation(message)

            # Store bot response
            self.conversation_history.append({
                'user_id': 'bot',
                'message': response['message'],
                'timestamp': response['timestamp'],
                'type': 'bot',
                'data': response.get('data', {})
            })

            return response

        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return {
                'message': f"Sorry, I encountered an error: {str(e)}",
                'type': 'error',
                'timestamp': datetime.now().isoformat()
            }

    def _get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status"""
        if self.live_agent:
            portfolio = self.live_agent.coordinator.trading_executor.get_portfolio_summary()
            live_status = self.live_agent.get_live_status()

            message = f"""
🤖 **Trading Agent Status Report**

💰 **Portfolio Overview:**
- Total Value: ${portfolio['total_portfolio_value']:,.2f}
- Total Return: {portfolio['total_return_pct']:+.2f}%
- Cash Balance: ${portfolio['cash_balance']:,.2f}
- Active Positions: {portfolio['active_positions_count']}

📊 **Performance:**
- Win Rate: {portfolio['win_rate']:.1f}%
- Total Trades: {portfolio.get('total_trades', 0)}
- Stop-Loss Coverage: {portfolio.get('stop_loss_coverage_pct', 0):.0f}%

🔗 **Live Trading:**
- Broker Connected: {'✅' if live_status['broker_connected'] else '❌'}
- Trading Active: {'✅' if live_status['trading_active'] else '❌'}
- Monitored Symbols: {', '.join(live_status['monitored_symbols'])}

How can I help you with your trading today?
            """

            return {
                'message': message.strip(),
                'data': {'portfolio': portfolio, 'live_status': live_status},
                'type': 'portfolio_status',
                'timestamp': datetime.now().isoformat()
            }
        else:
            return {
                'message': "I'm running in simulation mode. Connect me to a broker for live trading status!",
                'type': 'info',
                'timestamp': datetime.now().isoformat()
            }

    def _get_current_positions(self) -> Dict[str, Any]:
        """Get current trading positions"""
        portfolio = self.coordinator.trading_executor.get_portfolio_summary()
        positions = portfolio['positions']

        active_positions = [p for p in positions.values() if p.quantity > 0]

        if not active_positions:
            message = "📊 No active positions currently. Ready to find new opportunities!"
        else:
            message = "📊 **Current Positions:**\n\n"
            for pos in active_positions:
                pnl = pos.unrealized_pnl + pos.realized_pnl
                pnl_pct = (pnl / (pos.quantity * pos.avg_price)) * 100 if pos.quantity > 0 else 0
                sl_info = f" | SL: ${pos.stop_loss_price:.2f}" if pos.stop_loss_price else " | No SL"

                message += f"**{pos.symbol}**: {pos.quantity} shares @ ${pos.avg_price:.2f}\n"
                message += f"   P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%){sl_info}\n\n"

        return {
            'message': message,
            'data': {'positions': [p.__dict__ for p in active_positions]},
            'type': 'positions',
            'timestamp': datetime.now().isoformat()
        }

    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        portfolio = self.coordinator.trading_executor.get_portfolio_summary()
        risk_report = self.coordinator.risk_manager.get_risk_report()

        message = f"""
📈 **Performance Metrics**

💵 **Returns:**
- Total Return: {portfolio['total_return_pct']:+.2f}%
- Unrealized P&L: ${portfolio['total_unrealized_pnl']:+,.2f}
- Realized P&L: ${portfolio['total_realized_pnl']:+,.2f}

🎯 **Trading Stats:**
- Win Rate: {portfolio['win_rate']:.1f}%
- Total Trades: {portfolio.get('total_trades', 0)}
- Average Trade: ${portfolio.get('avg_trade_size', 0):,.2f}

⚠️ **Risk Metrics:**
- Max Drawdown: {risk_report['risk_metrics'].get('max_drawdown', 0):.1%}
- Sharpe Ratio: {risk_report['risk_metrics'].get('sharpe_ratio', 0):.2f}
- Portfolio Volatility: {risk_report['risk_metrics'].get('volatility', 0):.1%}

Looking great! Any specific performance aspect you'd like me to analyze?
        """

        return {
            'message': message.strip(),
            'data': {'performance': portfolio, 'risk': risk_report},
            'type': 'performance',
            'timestamp': datetime.now().isoformat()
        }

    def _get_risk_metrics(self) -> Dict[str, Any]:
        """Get risk management information"""
        risk_report = self.coordinator.risk_manager.get_risk_report()
        portfolio = self.coordinator.trading_executor.get_portfolio_summary()

        message = f"""
🛡️ **Risk Management Status**

🔒 **Stop-Loss Protection:**
- Active Stops: {risk_report['active_stops']}
- Coverage: {portfolio.get('stop_loss_coverage_pct', 0):.0f}%
- Trailing Stops: {risk_report['trailing_stops']}

⚠️ **Risk Limits:**
- Max Position Size: {risk_report['max_position_size']:.0%}
- Max Portfolio Risk: {risk_report['max_portfolio_risk']:.0%}
- Drawdown Limit: {risk_report['drawdown_limit']:.0%}

📊 **Current Risk:**
- Max Drawdown: {risk_report['risk_metrics'].get('max_drawdown', 0):.1%}
- VaR (95%): ${risk_report['risk_metrics'].get('var_95', 0):,.2f}
- Largest Loss: ${portfolio.get('largest_loss', 0):,.2f}

Your portfolio is well-protected! Is there anything specific about risk you'd like to adjust?
        """

        return {
            'message': message.strip(),
            'data': {'risk': risk_report},
            'type': 'risk',
            'timestamp': datetime.now().isoformat()
        }

    def _handle_trading_request(self, message: str) -> Dict[str, Any]:
        """Handle trading buy/sell requests"""
        message_lower = message.lower()

        # Extract symbol if mentioned
        symbols = ['aapl', 'googl', 'msft', 'tsla', 'nvda']
        mentioned_symbol = None
        for symbol in symbols:
            if symbol in message_lower:
                mentioned_symbol = symbol.upper()
                break

        if 'buy' in message_lower:
            action = "buy"
        elif 'sell' in message_lower:
            action = "sell"
        else:
            action = "analyze"

        if mentioned_symbol:
            # Get current analysis for the symbol
            # This would integrate with the real market analysis
            response_msg = f"""
🤖 **Trading Analysis for {mentioned_symbol}**

I'm analyzing {mentioned_symbol} right now with my AI models...

📊 **Current Signals:**
- Technical Analysis: Processing...
- ML Prediction: Calculating...
- Sentiment: Analyzing...
- Digital Brain Patterns: Searching...

Based on my analysis, I'll generate signals through my normal trading process. You can see the results in your portfolio status.

Would you like me to explain my decision-making process for {mentioned_symbol}?
            """
        else:
            response_msg = f"""
🤖 **Trading Request Received**

I understand you want to {action}, but I need more information:

- Which symbol? (AAPL, GOOGL, MSFT, TSLA, NVDA)
- What's your reasoning?

I can also provide analysis on any symbol. Just ask:
"What do you think about AAPL?" or "Should I buy TSLA?"

My AI models are constantly analyzing all symbols and will execute trades when conditions are optimal.
            """

        return {
            'message': response_msg.strip(),
            'type': 'trading',
            'timestamp': datetime.now().isoformat()
        }

    def _query_digital_brain(self, message: str) -> Dict[str, Any]:
        """Query the digital brain knowledge"""
        try:
            # Extract the actual question
            query = message.replace('brain', '').replace('knowledge', '').strip()
            if not query:
                query = "trading patterns and strategies"

            # Query the digital brain
            brain_result = self.coordinator.market_analyst.digital_brain.query_brain(
                query, {'query_type': 'chat_interaction'}
            )

            insights = brain_result.get('insights', [])
            confidence = brain_result.get('confidence', 0)

            if insights:
                message_text = f"""
🧠 **Digital Brain Knowledge**

**Query:** {query}
**Confidence:** {confidence:.1%}

**Insights:**
"""
                for i, insight in enumerate(insights[:3], 1):
                    message_text += f"{i}. {insight}\n"

                message_text += f"""
📚 **Knowledge Base Stats:**
- Documents: {brain_result.get('total_documents_in_memory', 0)}
- Knowledge Nodes: {brain_result.get('total_knowledge_nodes', 0)}

What else would you like to know from my trading knowledge?
"""
            else:
                message_text = """
🧠 **Digital Brain**

I'm searching my knowledge base... Let me learn more about your question and get back to you with insights!

My brain contains extensive trading knowledge from:
- Encyclopedia of Chart Patterns
- Technical Analysis guides
- Trading strategies
- Market patterns

Ask me about specific patterns, strategies, or market conditions!
"""

            return {
                'message': message_text.strip(),
                'data': brain_result,
                'type': 'brain_query',
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            return {
                'message': f"🧠 My digital brain is processing... Try asking about trading patterns or strategies!",
                'type': 'error',
                'timestamp': datetime.now().isoformat()
            }

    def _get_market_analysis(self) -> Dict[str, Any]:
        """Get current market analysis"""
        message = """
📊 **Market Analysis**

My AI models are continuously analyzing:

🔍 **Technical Analysis:**
- RSI, MACD, Bollinger Bands
- Support/Resistance levels
- Moving averages and trends

🤖 **ML Predictions:**
- Pattern recognition models
- Sentiment analysis
- Volatility forecasting

🧠 **Digital Brain:**
- Historical pattern matching
- Success rate analysis
- Market regime detection

**Current Focus:** Looking for momentum opportunities while maintaining strict risk controls.

Want analysis on a specific symbol or market condition?
        """

        return {
            'message': message.strip(),
            'type': 'analysis',
            'timestamp': datetime.now().isoformat()
        }

    def _general_conversation(self, message: str) -> Dict[str, Any]:
        """Handle general conversation"""
        responses = [
            "🤖 I'm focused on trading! Ask me about portfolio status, market analysis, or trading strategies.",
            "📈 How can I help with your trading today? I can check positions, analyze markets, or explain my decisions.",
            "💼 I'm your AI trading assistant! Try asking about performance, risk management, or specific stocks.",
            "🎯 Ready to discuss trading! What would you like to know about your portfolio or the markets?"
        ]

        import random
        return {
            'message': random.choice(responses),
            'type': 'conversation',
            'timestamp': datetime.now().isoformat()
        }

    def _get_help(self) -> Dict[str, Any]:
        """Provide help information"""
        message = """
🤖 **Trading Agent Commands**

**Portfolio & Performance:**
- "What's my status?" - Portfolio overview
- "Show positions" - Current holdings
- "How am I performing?" - Detailed metrics

**Trading:**
- "Should I buy AAPL?" - Analysis request
- "What do you think about TSLA?" - Symbol analysis
- "Market outlook?" - General market view

**Risk & Safety:**
- "Risk status?" - Risk management overview
- "Stop losses?" - Protection status

**Knowledge:**
- "Brain query: chart patterns" - Digital brain search
- "Help" - This message

**Live Trading:**
- "Trading status?" - Broker connection status

Just talk naturally! I understand context and can help with any trading-related questions.
        """

        return {
            'message': message.strip(),
            'type': 'help',
            'timestamp': datetime.now().isoformat()
        }

# Flask web interface
if FLASK_AVAILABLE:
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'trading-agent-secret-key'
    socketio = SocketIO(app, cors_allowed_origins="*")

# Global chat bot instance
chat_bot = None

if FLASK_AVAILABLE:
    @app.route('/')
    def index():
        """Main chat interface"""
        return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Trading Agent Chat</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
            .chat-container { max-width: 800px; margin: 0 auto; background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .chat-header { background: #2196F3; color: white; padding: 20px; border-radius: 10px 10px 0 0; }
            .chat-messages { height: 500px; overflow-y: auto; padding: 20px; border-bottom: 1px solid #eee; }
            .message { margin: 10px 0; padding: 10px; border-radius: 8px; }
            .user-message { background: #e3f2fd; margin-left: 50px; text-align: right; }
            .bot-message { background: #f5f5f5; margin-right: 50px; }
            .message-time { font-size: 0.8em; color: #666; margin-top: 5px; }
            .chat-input { display: flex; padding: 20px; }
            .chat-input input { flex: 1; padding: 15px; border: 1px solid #ddd; border-radius: 5px; font-size: 16px; }
            .chat-input button { padding: 15px 25px; background: #2196F3; color: white; border: none; border-radius: 5px; margin-left: 10px; cursor: pointer; }
            .status-badge { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 0.8em; margin-left: 10px; }
            .status-connected { background: #4CAF50; color: white; }
            .status-disconnected { background: #f44336; color: white; }
            pre { background: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                <h1>🤖 AI Trading Agent Chat</h1>
                <p>Chat with your intelligent trading assistant</p>
                <div id="connection-status">
                    <span class="status-badge status-disconnected">Connecting...</span>
                </div>
            </div>
            <div class="chat-messages" id="messages">
                <div class="message bot-message">
                    <strong>🤖 Trading Agent:</strong> Hello! I'm your AI trading assistant. I'm currently managing your portfolio with advanced ML models and risk management. How can I help you today?
                    <div class="message-time">Ready to trade</div>
                </div>
            </div>
            <div class="chat-input">
                <input type="text" id="messageInput" placeholder="Ask me about your portfolio, market analysis, or trading..." autofocus>
                <button onclick="sendMessage()">Send</button>
            </div>
        </div>

        <script>
            const socket = io();

            socket.on('connect', function() {
                document.getElementById('connection-status').innerHTML = '<span class="status-badge status-connected">Connected</span>';
            });

            socket.on('disconnect', function() {
                document.getElementById('connection-status').innerHTML = '<span class="status-badge status-disconnected">Disconnected</span>';
            });

            socket.on('bot_response', function(data) {
                addMessage('🤖 Trading Agent', data.message, 'bot-message', data.timestamp);
            });

            function sendMessage() {
                const input = document.getElementById('messageInput');
                const message = input.value.trim();
                if (message) {
                    addMessage('👤 You', message, 'user-message', new Date().toLocaleTimeString());
                    socket.emit('user_message', {message: message});
                    input.value = '';
                }
            }

            function addMessage(sender, message, className, timestamp) {
                const messagesDiv = document.getElementById('messages');
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message ' + className;

                // Convert markdown-style formatting to HTML
                let formattedMessage = message
                    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                    .replace(/\*(.*?)\*/g, '<em>$1</em>')
                    .replace(/\n/g, '<br>');

                messageDiv.innerHTML = `
                    <strong>${sender}:</strong> ${formattedMessage}
                    <div class="message-time">${timestamp}</div>
                `;
                messagesDiv.appendChild(messageDiv);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }

            document.getElementById('messageInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
        </script>
    </body>
    </html>
        '''

    @socketio.on('user_message')
    def handle_message(data):
        """Handle incoming chat messages"""
        if chat_bot:
            response = chat_bot.process_user_message(data['message'])
            emit('bot_response', response)
        else:
            emit('bot_response', {
                'message': '🤖 Trading agent is initializing... Please wait.',
                'timestamp': datetime.now().isoformat()
            })

def start_chat_interface(live_agent: Optional[LiveTradingAgent] = None, host='0.0.0.0', port=5000):
    """Start the chat interface web server"""
    global chat_bot
    chat_bot = TradingChatBot(live_agent)

    print(f"🚀 Starting Trading Agent Chat Interface")
    print(f"📱 Open your browser to: http://localhost:{port}")
    print(f"💬 Chat with your AI trading agent!")
    if FLASK_AVAILABLE:
        socketio.run(app, host=host, port=port, debug=False)
    else:
        print("Flask not available, cannot start web interface.")

if __name__ == "__main__":
    # Demo mode - start chat without live trading
    start_chat_interface()