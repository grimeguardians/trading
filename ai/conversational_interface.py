"""
Conversational AI Interface for natural language trading interactions
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from .digital_brain import DigitalBrain
from ..config import Config
from ..utils.logger import get_logger

class ConversationType(Enum):
    QUERY = "query"
    COMMAND = "command"
    ANALYSIS = "analysis"
    EDUCATION = "education"
    ALERT = "alert"

@dataclass
class ConversationContext:
    """Context for conversation tracking"""
    user_id: str
    session_id: str
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    current_topic: Optional[str] = None
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    last_interaction: datetime = field(default_factory=datetime.now)

class ConversationalInterface:
    """Natural language interface for trading system"""
    
    def __init__(self, digital_brain: DigitalBrain, config: Config):
        self.digital_brain = digital_brain
        self.config = config
        self.logger = get_logger("ConversationalInterface")
        
        # User contexts
        self.user_contexts: Dict[str, ConversationContext] = {}
        
        # Command patterns
        self.command_patterns = {
            'market_analysis': [
                'analyze', 'analysis', 'what do you think about',
                'market sentiment', 'outlook', 'forecast'
            ],
            'position_query': [
                'position', 'holdings', 'portfolio', 'current trades',
                'what am i holding', 'my positions'
            ],
            'trade_execution': [
                'buy', 'sell', 'trade', 'execute', 'place order',
                'close position', 'enter position'
            ],
            'risk_management': [
                'risk', 'stop loss', 'position size', 'drawdown',
                'what is my risk', 'risk level'
            ],
            'strategy_query': [
                'strategy', 'strategies', 'which strategy',
                'trading approach', 'method'
            ],
            'news_impact': [
                'news', 'impact', 'earnings', 'announcement',
                'what happened', 'why is it moving'
            ],
            'education': [
                'explain', 'what is', 'how does', 'teach me',
                'learn', 'definition', 'meaning'
            ],
            'performance': [
                'performance', 'profit', 'loss', 'returns',
                'how am i doing', 'pnl', 'results'
            ]
        }
        
        # Response templates
        self.response_templates = {
            'greeting': [
                "Hello! I'm your AI trading assistant. How can I help you today?",
                "Hi! Ready to discuss your trading strategies?",
                "Welcome! What would you like to know about the markets?"
            ],
            'acknowledgment': [
                "I understand you're asking about {topic}.",
                "Let me help you with {topic}.",
                "Good question about {topic}."
            ],
            'clarification': [
                "Could you be more specific about {topic}?",
                "I need a bit more information about {topic}.",
                "Can you clarify what you mean by {topic}?"
            ],
            'error': [
                "I'm sorry, I didn't understand that. Could you rephrase?",
                "I'm having trouble with that request. Can you try again?",
                "That doesn't seem right. Let me know if you need help."
            ]
        }
        
        # Trading command examples
        self.trading_examples = [
            "Current market sentiment?",
            "AAPL stock analysis?",
            "Promising setups now?",
            "Adjust stop-losses?",
            "Bitcoin volatility outlook?",
            "Compare strategy Alpha vs Beta?",
            "Today's economic news summary?",
            "Current correlated assets?",
            "Portfolio risk assessment?",
            "Best entry points for SPY?",
            "Fibonacci levels for TSLA?",
            "Volume analysis for QQQ?",
            "Momentum indicators showing?",
            "Options flow today?",
            "Crypto market regime?",
            "Earnings calendar impact?",
            "Sector rotation signals?",
            "Fear and greed index?",
            "Support and resistance levels?",
            "Technical pattern alerts?",
            "Risk-adjusted returns?",
            "Correlation matrix update?",
            "Volatility surface analysis?",
            "Market breadth indicators?",
            "Liquidity conditions?"
        ]
    
    async def process_message(self, message: str, user_id: str) -> Dict[str, Any]:
        """Process incoming message from user"""
        try:
            # Get or create user context
            context = self.get_user_context(user_id)
            
            # Update last interaction
            context.last_interaction = datetime.now()
            
            # Add message to conversation history
            context.conversation_history.append({
                'timestamp': datetime.now(),
                'type': 'user',
                'content': message
            })
            
            # Classify message type
            message_type = self.classify_message(message)
            
            # Extract intent and entities
            intent = await self.extract_intent(message, context)
            entities = await self.extract_entities(message)
            
            # Generate response
            response = await self.generate_response(message, intent, entities, context)
            
            # Add response to conversation history
            context.conversation_history.append({
                'timestamp': datetime.now(),
                'type': 'assistant',
                'content': response['content']
            })
            
            # Update context
            if intent.get('topic'):
                context.current_topic = intent['topic']
            
            return {
                'response': response['content'],
                'intent': intent,
                'entities': entities,
                'context': {
                    'topic': context.current_topic,
                    'conversation_length': len(context.conversation_history)
                },
                'suggestions': self.get_suggestions(intent, context)
            }
            
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return {
                'response': "I apologize, but I encountered an error processing your message. Please try again.",
                'error': str(e)
            }
    
    def get_user_context(self, user_id: str) -> ConversationContext:
        """Get or create user context"""
        if user_id not in self.user_contexts:
            self.user_contexts[user_id] = ConversationContext(
                user_id=user_id,
                session_id=f"session_{datetime.now().timestamp()}"
            )
        
        return self.user_contexts[user_id]
    
    def classify_message(self, message: str) -> ConversationType:
        """Classify message type"""
        try:
            message_lower = message.lower()
            
            # Check for questions
            if message_lower.startswith(('what', 'how', 'why', 'when', 'where', 'which')):
                return ConversationType.QUERY
            
            # Check for commands
            if any(word in message_lower for word in ['buy', 'sell', 'execute', 'place', 'close']):
                return ConversationType.COMMAND
            
            # Check for analysis requests
            if any(word in message_lower for word in ['analyze', 'analysis', 'outlook', 'forecast']):
                return ConversationType.ANALYSIS
            
            # Check for educational content
            if any(word in message_lower for word in ['explain', 'teach', 'learn', 'definition']):
                return ConversationType.EDUCATION
            
            # Default to query
            return ConversationType.QUERY
            
        except Exception as e:
            self.logger.error(f"Error classifying message: {e}")
            return ConversationType.QUERY
    
    async def extract_intent(self, message: str, context: ConversationContext) -> Dict[str, Any]:
        """Extract intent from message"""
        try:
            message_lower = message.lower()
            intent = {'confidence': 0.0, 'topic': None, 'action': None}
            
            # Match against command patterns
            for topic, patterns in self.command_patterns.items():
                for pattern in patterns:
                    if pattern in message_lower:
                        intent['topic'] = topic
                        intent['confidence'] = 0.8
                        break
                
                if intent['topic']:
                    break
            
            # Extract action
            if 'buy' in message_lower or 'purchase' in message_lower:
                intent['action'] = 'buy'
            elif 'sell' in message_lower or 'close' in message_lower:
                intent['action'] = 'sell'
            elif 'analyze' in message_lower or 'analysis' in message_lower:
                intent['action'] = 'analyze'
            elif 'explain' in message_lower or 'what is' in message_lower:
                intent['action'] = 'explain'
            
            # Use conversation context to improve intent
            if context.current_topic and intent['confidence'] < 0.5:
                intent['topic'] = context.current_topic
                intent['confidence'] = 0.6
            
            return intent
            
        except Exception as e:
            self.logger.error(f"Error extracting intent: {e}")
            return {'confidence': 0.0, 'topic': None, 'action': None}
    
    async def extract_entities(self, message: str) -> Dict[str, Any]:
        """Extract entities from message"""
        try:
            entities = {
                'symbols': [],
                'numbers': [],
                'dates': [],
                'timeframes': [],
                'indicators': []
            }
            
            # Extract stock symbols
            import re
            
            # Common stock symbols
            symbol_pattern = r'\b[A-Z]{1,5}\b'
            potential_symbols = re.findall(symbol_pattern, message.upper())
            
            # Filter for known symbols
            known_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'SPY', 'QQQ', 'NVDA', 'AMZN']
            for symbol in potential_symbols:
                if symbol in known_symbols:
                    entities['symbols'].append(symbol)
            
            # Extract numbers
            number_pattern = r'\d+\.?\d*'
            numbers = re.findall(number_pattern, message)
            entities['numbers'] = [float(n) for n in numbers]
            
            # Extract timeframes
            timeframe_patterns = ['1h', '4h', '1d', '1w', '1m', 'hourly', 'daily', 'weekly', 'monthly']
            for pattern in timeframe_patterns:
                if pattern in message.lower():
                    entities['timeframes'].append(pattern)
            
            # Extract indicators
            indicator_patterns = ['rsi', 'macd', 'bollinger', 'moving average', 'ma', 'sma', 'ema']
            for pattern in indicator_patterns:
                if pattern in message.lower():
                    entities['indicators'].append(pattern)
            
            return entities
            
        except Exception as e:
            self.logger.error(f"Error extracting entities: {e}")
            return {'symbols': [], 'numbers': [], 'dates': [], 'timeframes': [], 'indicators': []}
    
    async def generate_response(self, message: str, intent: Dict[str, Any], 
                               entities: Dict[str, Any], context: ConversationContext) -> Dict[str, Any]:
        """Generate response based on intent and entities"""
        try:
            topic = intent.get('topic')
            action = intent.get('action')
            
            # Handle different topics
            if topic == 'market_analysis':
                response = await self.handle_market_analysis(message, entities, context)
            elif topic == 'position_query':
                response = await self.handle_position_query(message, entities, context)
            elif topic == 'trade_execution':
                response = await self.handle_trade_execution(message, entities, context)
            elif topic == 'risk_management':
                response = await self.handle_risk_management(message, entities, context)
            elif topic == 'strategy_query':
                response = await self.handle_strategy_query(message, entities, context)
            elif topic == 'news_impact':
                response = await self.handle_news_impact(message, entities, context)
            elif topic == 'education':
                response = await self.handle_education(message, entities, context)
            elif topic == 'performance':
                response = await self.handle_performance(message, entities, context)
            else:
                response = await self.handle_general_query(message, entities, context)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return {'content': "I'm sorry, I encountered an error generating a response."}
    
    async def handle_market_analysis(self, message: str, entities: Dict[str, Any], 
                                   context: ConversationContext) -> Dict[str, Any]:
        """Handle market analysis requests"""
        try:
            symbols = entities.get('symbols', ['SPY'])  # Default to SPY
            
            response_parts = []
            
            for symbol in symbols:
                # Get market analysis from digital brain
                market_data = {
                    'symbol': symbol,
                    'timestamp': datetime.now(),
                    'request_type': 'analysis'
                }
                
                # Query digital brain
                insight = await self.digital_brain.generate_insight(market_data, "market_analysis")
                
                if insight:
                    response_parts.append(f"**{symbol} Analysis:**\n{insight.content}")
                    
                    if insight.actionable:
                        response_parts.append(f"📊 **Actionable Insight:** This analysis suggests specific trading opportunities.")
                    
                    response_parts.append(f"🎯 **Confidence:** {insight.confidence:.0%}")
                    response_parts.append(f"📈 **Market Impact:** {insight.market_impact.title()}")
                else:
                    response_parts.append(f"**{symbol}:** I don't have current analysis available.")
            
            if not response_parts:
                response_parts.append("I don't have current market analysis available. Please try again later.")
            
            return {'content': '\n\n'.join(response_parts)}
            
        except Exception as e:
            self.logger.error(f"Error handling market analysis: {e}")
            return {'content': "I'm having trouble accessing market analysis right now."}
    
    async def handle_position_query(self, message: str, entities: Dict[str, Any], 
                                   context: ConversationContext) -> Dict[str, Any]:
        """Handle position and portfolio queries"""
        try:
            # This would typically query the portfolio manager
            response = """
            📊 **Current Portfolio Overview:**
            
            **Active Positions:**
            • AAPL: 100 shares at $150.00 (+2.5%)
            • GOOGL: 50 shares at $2,500.00 (-1.2%)
            • SPY: 200 shares at $400.00 (+0.8%)
            
            **Portfolio Value:** $125,000
            **Daily P&L:** +$1,250 (+1.0%)
            **Total Return:** +8.5%
            
            **Risk Metrics:**
            • Current allocation: 70% stocks, 20% ETFs, 10% cash
            • Portfolio beta: 1.15
            • Max drawdown: -5.2%
            
            Would you like me to analyze any specific position or provide more detailed metrics?
            """
            
            return {'content': response}
            
        except Exception as e:
            self.logger.error(f"Error handling position query: {e}")
            return {'content': "I'm having trouble accessing your portfolio information right now."}
    
    async def handle_trade_execution(self, message: str, entities: Dict[str, Any], 
                                   context: ConversationContext) -> Dict[str, Any]:
        """Handle trade execution commands"""
        try:
            # Extract trading parameters
            symbols = entities.get('symbols', [])
            numbers = entities.get('numbers', [])
            
            if not symbols:
                return {'content': "Which symbol would you like to trade? Please specify a stock symbol."}
            
            symbol = symbols[0]
            quantity = numbers[0] if numbers else None
            
            # Determine action
            if 'buy' in message.lower():
                action = 'buy'
            elif 'sell' in message.lower():
                action = 'sell'
            else:
                return {'content': f"Would you like to buy or sell {symbol}? Please specify the action."}
            
            # Safety check - don't execute actual trades in demo
            response = f"""
            ⚠️ **Trade Confirmation Required**
            
            **Proposed Trade:**
            • Action: {action.upper()}
            • Symbol: {symbol}
            • Quantity: {quantity if quantity else 'Market order'}
            • Estimated cost: ${(quantity or 100) * 150:.2f}
            
            **Pre-trade Analysis:**
            • Current price: $150.00
            • 52-week range: $120 - $180
            • RSI: 65 (neutral)
            • Trend: Bullish
            
            **Risk Assessment:**
            • Position size: 2.5% of portfolio
            • Stop loss recommended: $142.50 (-5%)
            • Take profit target: $165.00 (+10%)
            
            To execute this trade, please confirm with your broker directly. 
            This system is for analysis and education only.
            """
            
            return {'content': response}
            
        except Exception as e:
            self.logger.error(f"Error handling trade execution: {e}")
            return {'content': "I'm having trouble processing your trade request."}
    
    async def handle_risk_management(self, message: str, entities: Dict[str, Any], 
                                   context: ConversationContext) -> Dict[str, Any]:
        """Handle risk management queries"""
        try:
            response = """
            🛡️ **Risk Management Dashboard**
            
            **Current Risk Metrics:**
            • Portfolio Value at Risk (VaR): $2,500 (2.0%)
            • Maximum Drawdown: -5.2%
            • Risk-Adjusted Return (Sharpe): 1.65
            • Beta vs S&P 500: 1.15
            
            **Position Risk:**
            • Largest position: 15% (AAPL)
            • Concentration risk: Medium
            • Sector exposure: Tech 40%, Finance 25%, Healthcare 20%
            
            **Risk Recommendations:**
            ✅ Stop losses are set for 85% of positions
            ⚠️ Consider reducing tech exposure (currently 40%)
            ✅ Portfolio diversification is adequate
            ⚠️ Cash position low (10%) - consider increasing to 15%
            
            **Upcoming Risk Events:**
            • FOMC meeting next week
            • Earnings season in 2 weeks
            • Options expiration Friday
            
            Would you like me to analyze risk for a specific position or strategy?
            """
            
            return {'content': response}
            
        except Exception as e:
            self.logger.error(f"Error handling risk management: {e}")
            return {'content': "I'm having trouble accessing risk management data."}
    
    async def handle_strategy_query(self, message: str, entities: Dict[str, Any], 
                                  context: ConversationContext) -> Dict[str, Any]:
        """Handle strategy-related queries"""
        try:
            response = """
            📈 **Trading Strategy Overview**
            
            **Active Strategies:**
            
            **1. Swing Trading Strategy**
            • Status: Active
            • Performance: +12.5% (last 30 days)
            • Win rate: 68%
            • Risk level: Medium
            
            **2. Mean Reversion Strategy**
            • Status: Active
            • Performance: +5.2% (last 30 days)
            • Win rate: 72%
            • Risk level: Low
            
            **3. Momentum Strategy**
            • Status: Paused
            • Performance: -2.1% (last 30 days)
            • Win rate: 45%
            • Risk level: High
            
            **Strategy Recommendations:**
            ✅ Swing trading performing well in current market
            ⚠️ Consider reactivating momentum strategy if market volatility increases
            ✅ Mean reversion providing steady returns
            
            **Market Regime:** Currently favorable for swing trading
            **Suggested allocation:** 60% swing, 40% mean reversion
            
            Would you like details on any specific strategy?
            """
            
            return {'content': response}
            
        except Exception as e:
            self.logger.error(f"Error handling strategy query: {e}")
            return {'content': "I'm having trouble accessing strategy information."}
    
    async def handle_news_impact(self, message: str, entities: Dict[str, Any], 
                                context: ConversationContext) -> Dict[str, Any]:
        """Handle news and market impact queries"""
        try:
            symbols = entities.get('symbols', [])
            
            if symbols:
                symbol = symbols[0]
                response = f"""
                📰 **News Impact Analysis for {symbol}**
                
                **Recent News:**
                • Earnings beat expectations by 5% (2 hours ago)
                • Analyst upgrade from Morgan Stanley (1 day ago)
                • New product launch announcement (3 days ago)
                
                **Market Impact:**
                • Price reaction: +3.2% on earnings
                • Volume: 2.5x average
                • Analyst sentiment: Bullish (80% buy ratings)
                
                **Technical Response:**
                • Broke resistance at $148
                • RSI moved to 72 (slightly overbought)
                • MACD showing bullish crossover
                
                **Outlook:**
                📈 **Positive momentum likely to continue**
                🎯 **Target:** $165 (next resistance level)
                ⚠️ **Risk:** Profit-taking at current levels
                """
            else:
                response = """
                📰 **Market News Summary**
                
                **Today's Key Events:**
                • Fed Chair speech at 2 PM EST
                • Tech earnings continue (GOOGL, MSFT after close)
                • Oil prices up 2% on supply concerns
                
                **Market Sentiment:**
                • VIX: 18.5 (moderate fear)
                • Put/Call ratio: 0.85 (slightly bullish)
                • Market breadth: 65% of stocks advancing
                
                **Sector Impact:**
                📈 Technology: +1.2%
                📈 Energy: +2.1%
                📉 Utilities: -0.8%
                📉 Real Estate: -1.1%
                
                **Tomorrow's Catalyst:**
                • CPI inflation data (8:30 AM)
                • Retail sales report (8:30 AM)
                """
            
            return {'content': response}
            
        except Exception as e:
            self.logger.error(f"Error handling news impact: {e}")
            return {'content': "I'm having trouble accessing current news information."}
    
    async def handle_education(self, message: str, entities: Dict[str, Any], 
                             context: ConversationContext) -> Dict[str, Any]:
        """Handle educational queries"""
        try:
            # Use digital brain for educational content
            response = await self.digital_brain.chat_query(message, {
                'type': 'education',
                'user_level': 'intermediate'
            })
            
            if not response or len(response) < 50:
                # Fallback educational content
                response = """
                📚 **Trading Education**
                
                I'd be happy to explain trading concepts! Here are some topics I can help with:
                
                **Technical Analysis:**
                • RSI (Relative Strength Index)
                • MACD (Moving Average Convergence Divergence)
                • Bollinger Bands
                • Support and Resistance
                • Chart patterns
                
                **Fundamental Analysis:**
                • P/E ratios
                • Earnings analysis
                • Economic indicators
                • Company valuation
                
                **Risk Management:**
                • Position sizing
                • Stop losses
                • Portfolio diversification
                • Risk-reward ratios
                
                **Options Trading:**
                • Calls and Puts
                • Greeks (Delta, Gamma, Theta, Vega)
                • Strategies (Covered Call, Protective Put)
                
                What specific topic would you like me to explain?
                """
            
            return {'content': response}
            
        except Exception as e:
            self.logger.error(f"Error handling education: {e}")
            return {'content': "I'd be happy to help with trading education. What would you like to learn about?"}
    
    async def handle_performance(self, message: str, entities: Dict[str, Any], 
                               context: ConversationContext) -> Dict[str, Any]:
        """Handle performance queries"""
        try:
            response = """
            📊 **Performance Dashboard**
            
            **Overall Performance:**
            • Total Return: +15.2% (YTD)
            • Benchmark (S&P 500): +12.8%
            • Alpha: +2.4%
            • Sharpe Ratio: 1.65
            
            **Monthly Performance:**
            • January: +3.1%
            • February: +2.8%
            • March: +1.9%
            • April: +4.2%
            • May: +3.2%
            
            **Strategy Performance:**
            • Swing Trading: +18.5% (best performer)
            • Mean Reversion: +8.2%
            • Momentum: +2.1%
            
            **Risk Metrics:**
            • Maximum Drawdown: -5.2%
            • Volatility: 12.8%
            • Beta: 1.15
            • Win Rate: 62%
            
            **Recent Trades:**
            ✅ AAPL: +5.2% (closed yesterday)
            ✅ GOOGL: +3.1% (closed last week)
            ❌ TSLA: -2.1% (stopped out)
            
            **Key Insights:**
            🎯 Outperforming market by 2.4%
            📈 Consistent positive returns
            🛡️ Well-managed risk profile
            """
            
            return {'content': response}
            
        except Exception as e:
            self.logger.error(f"Error handling performance: {e}")
            return {'content': "I'm having trouble accessing performance data right now."}
    
    async def handle_general_query(self, message: str, entities: Dict[str, Any], 
                                 context: ConversationContext) -> Dict[str, Any]:
        """Handle general queries"""
        try:
            # Use digital brain for general queries
            response = await self.digital_brain.chat_query(message, {
                'conversation_context': context.current_topic
            })
            
            if not response or len(response) < 20:
                # Fallback response
                response = """
                I'm here to help with your trading questions! I can assist with:
                
                📈 **Market Analysis** - Get insights on stocks, trends, and opportunities
                📊 **Portfolio Management** - Review your positions and performance
                🎯 **Strategy Development** - Discuss trading approaches and techniques
                🛡️ **Risk Management** - Analyze and optimize your risk profile
                📰 **News Impact** - Understand how events affect your trades
                📚 **Education** - Learn about trading concepts and techniques
                
                What would you like to explore?
                """
            
            return {'content': response}
            
        except Exception as e:
            self.logger.error(f"Error handling general query: {e}")
            return {'content': "I'm here to help with your trading questions. What would you like to know?"}
    
    def get_suggestions(self, intent: Dict[str, Any], context: ConversationContext) -> List[str]:
        """Get conversation suggestions based on context"""
        try:
            topic = intent.get('topic')
            suggestions = []
            
            if topic == 'market_analysis':
                suggestions = [
                    "Show me sector performance",
                    "What's the market sentiment today?",
                    "Analyze volatility trends",
                    "Check correlation matrix"
                ]
            elif topic == 'position_query':
                suggestions = [
                    "Show risk metrics",
                    "What's my best performer?",
                    "Rebalance recommendations",
                    "Exit strategy analysis"
                ]
            elif topic == 'education':
                suggestions = [
                    "Explain Fibonacci levels",
                    "What are options Greeks?",
                    "How to read candlestick patterns",
                    "Risk management basics"
                ]
            else:
                # Default suggestions
                suggestions = [
                    "Analyze my portfolio",
                    "Show market outlook",
                    "Check risk levels",
                    "Find trading opportunities"
                ]
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error getting suggestions: {e}")
            return ["How can I help you today?"]
    
    def get_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversation history for user"""
        try:
            if user_id in self.user_contexts:
                history = self.user_contexts[user_id].conversation_history
                return history[-limit:] if len(history) > limit else history
            return []
        except Exception as e:
            self.logger.error(f"Error getting conversation history: {e}")
            return []
    
    def clear_conversation_history(self, user_id: str):
        """Clear conversation history for user"""
        try:
            if user_id in self.user_contexts:
                self.user_contexts[user_id].conversation_history.clear()
                self.user_contexts[user_id].current_topic = None
        except Exception as e:
            self.logger.error(f"Error clearing conversation history: {e}")
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get user interaction statistics"""
        try:
            if user_id not in self.user_contexts:
                return {}
            
            context = self.user_contexts[user_id]
            
            return {
                'total_messages': len(context.conversation_history),
                'session_duration': (datetime.now() - context.last_interaction).total_seconds(),
                'current_topic': context.current_topic,
                'last_interaction': context.last_interaction
            }
            
        except Exception as e:
            self.logger.error(f"Error getting user stats: {e}")
            return {}
