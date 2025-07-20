"""
Conversational AI for Trading Assistant
Provides natural language conversations about trading, markets, and portfolio management
"""

import os
import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import openai
from openai import OpenAI

logger = logging.getLogger(__name__)

class TradingConversationalAI:
    """AI-powered conversational interface for trading assistant"""
    
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY') or os.getenv('OPENAI_KEY')
        if not self.api_key:
            logger.warning("OpenAI API key not found. Chat will use fallback responses.")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)
            logger.info("OpenAI client initialized successfully")
        
        # Initialize conversation history first
        self.conversation_history = []
        self.max_history = 20  # Keep last 20 messages for context
        
        # Import trading functions for command execution
        self._setup_trading_functions()
        
        self.system_prompt = """You are an AI trading assistant with LIVE trading capabilities. You can execute real commands and trades.

CRITICAL EXECUTION RULE: 
- ONLY claim trades were executed successfully if you see Command Execution Results showing success
- If there are no Command Execution Results or they show failures, DO NOT claim success
- Always report the actual results from the Command Execution Results section

**COMMAND EXECUTION CAPABILITIES:**
- "Close all positions" → Execute mass position closure
- "Buy 10 shares of AAPL" → Place buy order  
- "Check my portfolio" → Get account status and positions
- "Cancel all orders" → Cancel pending orders
- "What's the price of TSLA?" → Get real-time quote

**Response Guidelines:**
1. If you see Command Execution Results showing success → Report the actual success details
2. If you see Command Execution Results showing failure → Explain the actual error and suggest solutions
3. If you see NO Command Execution Results → Explain that the command wasn't detected/executed
4. Never fabricate execution results or claim trades completed without verification

**Example Responses:**
✅ Good: "I attempted to close your positions. The results show: 1 position failed to close due to wash trade restrictions. AAPL position (3 shares) remains open."
❌ Bad: "All positions have been successfully closed" (when no execution results confirm this)

You are connected to Alpaca paper trading. When users request trading actions, execute them and report the ACTUAL results from the Command Execution Results."""
        
        # Initialize conversation history first
        self.conversation_history = []
        self.max_history = 20  # Keep last 20 messages for context
    
    def _setup_trading_functions(self):
        """Setup trading function references"""
        try:
            # These will be imported when needed to avoid circular imports
            self.trading_functions = {
                'get_account': self._get_account_info,
                'get_positions': self._get_positions,
                'close_position': self._close_position,
                'close_all_positions': self._close_all_positions,
                'place_order': self._place_order,
                'cancel_orders': self._cancel_orders,
                'get_market_data': self._get_market_data
            }
            logger.info("Trading functions setup complete")
        except Exception as e:
            logger.error(f"Error setting up trading functions: {e}")
    
    def _get_account_info(self):
        """Get Alpaca account information"""
        try:
            from core.alpaca_paper_trading import alpaca_trader
            return alpaca_trader.get_account_info()
        except Exception as e:
            return {"error": f"Failed to get account info: {str(e)}"}
    
    def _get_positions(self):
        """Get current positions"""
        try:
            from core.alpaca_paper_trading import alpaca_trader
            return alpaca_trader.get_positions()
        except Exception as e:
            return {"error": f"Failed to get positions: {str(e)}"}
    
    def _close_position(self, symbol: str):
        """Close a specific position"""
        try:
            from core.alpaca_paper_trading import alpaca_trader
            
            # First get position details
            positions_data = alpaca_trader.get_positions()
            if "error" in positions_data:
                return positions_data
            
            positions = positions_data.get("positions", [])
            target_position = None
            
            for pos in positions:
                if pos["symbol"].upper() == symbol.upper():
                    target_position = pos
                    break
            
            if not target_position:
                return {"error": f"No position found for {symbol}"}
            
            # Close position by selling/buying opposite side
            qty = abs(float(target_position["qty"]))
            side = "sell" if float(target_position["qty"]) > 0 else "buy"
            
            result = alpaca_trader.place_order(
                symbol=symbol,
                qty=qty,
                side=side,
                order_type="market"
            )
            
            if result.get("status") == "success":
                result["action"] = f"Closed position: {side} {qty} shares of {symbol}"
            
            return result
            
        except Exception as e:
            return {"error": f"Failed to close position: {str(e)}"}
    
    def _close_all_positions(self):
        """Close all open positions"""
        try:
            from core.alpaca_paper_trading import alpaca_trader
            
            # Get all positions
            positions_data = alpaca_trader.get_positions()
            if "error" in positions_data:
                return positions_data
            
            positions = positions_data.get("positions", [])
            
            if not positions:
                return {"status": "success", "message": "No positions to close"}
            
            results = []
            for pos in positions:
                symbol = pos["symbol"]
                qty = abs(float(pos["qty"]))
                side = "sell" if float(pos["qty"]) > 0 else "buy"
                
                result = alpaca_trader.place_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    order_type="market"
                )
                
                results.append({
                    "symbol": symbol,
                    "qty": qty,
                    "side": side,
                    "result": result.get("status", "failed"),
                    "message": result.get("message", result.get("error", "Unknown"))
                })
            
            successful_closes = [r for r in results if r["result"] == "success"]
            failed_closes = [r for r in results if r["result"] != "success"]
            
            return {
                "status": "success",
                "total_positions": len(positions),
                "successful_closes": len(successful_closes),
                "failed_closes": len(failed_closes),
                "details": results,
                "summary": f"Closed {len(successful_closes)} of {len(positions)} positions"
            }
            
        except Exception as e:
            return {"error": f"Failed to close all positions: {str(e)}"}
    
    def _place_order(self, symbol: str, qty: float, side: str, order_type: str = "market", limit_price: float = None):
        """Place a trading order"""
        try:
            from core.alpaca_paper_trading import alpaca_trader
            return alpaca_trader.place_order(symbol, qty, side, order_type, limit_price=limit_price)
        except Exception as e:
            return {"error": f"Failed to place order: {str(e)}"}
    
    def _cancel_orders(self):
        """Cancel all open orders"""
        try:
            from core.alpaca_paper_trading import alpaca_trader
            
            # Get open orders
            orders_data = alpaca_trader.get_orders("open")
            if "error" in orders_data:
                return orders_data
            
            orders = orders_data.get("orders", [])
            
            if not orders:
                return {"status": "success", "message": "No open orders to cancel"}
            
            results = []
            for order in orders:
                result = alpaca_trader.cancel_order(order["id"])
                results.append({
                    "order_id": order["id"],
                    "symbol": order["symbol"],
                    "result": result.get("status", "failed"),
                    "message": result.get("message", result.get("error", "Unknown"))
                })
            
            successful_cancels = [r for r in results if r["result"] == "success"]
            
            return {
                "status": "success",
                "total_orders": len(orders),
                "cancelled": len(successful_cancels),
                "details": results,
                "summary": f"Cancelled {len(successful_cancels)} of {len(orders)} orders"
            }
            
        except Exception as e:
            return {"error": f"Failed to cancel orders: {str(e)}"}
    
    def _get_market_data(self, symbol: str):
        """Get market data for a symbol"""
        try:
            from core.enhanced_market_data_provider import enhanced_market_data_provider
            quote = enhanced_market_data_provider.get_comprehensive_quote(symbol.upper())
            
            if quote:
                return {
                    "status": "success",
                    "symbol": quote.symbol,
                    "price": quote.price,
                    "change": quote.change,
                    "change_percent": quote.change_percent,
                    "volume": quote.volume,
                    "high": quote.high,
                    "low": quote.low,
                    "market_cap": quote.market_cap,
                    "source": quote.source
                }
            else:
                return {"error": f"No market data found for {symbol}"}
                
        except Exception as e:
            return {"error": f"Failed to get market data: {str(e)}"}
    
    def _detect_and_execute_command(self, user_message: str) -> Optional[Dict[str, Any]]:
        """Detect trading commands in user message and execute them"""
        message_lower = user_message.lower()
        
        try:
            # Close all positions - expanded detection
            close_all_phrases = [
                'close all positions', 'close all my positions', 'sell all positions', 'exit all positions',
                'close all trades', 'close all my trades', 'sell all trades', 'exit all trades',
                'close everything', 'sell everything', 'liquidate all', 'liquidate everything'
            ]
            if any(phrase in message_lower for phrase in close_all_phrases):
                logger.info(f"🎯 COMMAND DETECTED: Close all positions from message: '{user_message}'")
                result = self._close_all_positions()
                logger.info(f"🎯 COMMAND RESULT: {result}")
                return result
            
            # Close specific position - more flexible pattern matching
            elif 'close' in message_lower and ('position' in message_lower or any(symbol in message_lower for symbol in ['aapl', 'googl', 'msft', 'tsla', 'nvda', 'amzn', 'meta'])):
                # Smart symbol extraction - prioritize known stock symbols
                words = user_message.upper().split()
                known_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'META', 'SPY', 'QQQ', 'IWM']
                excluded_words = ['CLOSE', 'POSITION', 'MY', 'THE', 'IN', 'ALPACA', 'CAN', 'YOU', 'PLEASE']
                
                # First try to find known symbols
                for word in words:
                    if word in known_symbols:
                        logger.info(f"🎯 COMMAND DETECTED: Close position {word} (known symbol) from message: '{user_message}'")
                        result = self._close_position(word)
                        logger.info(f"🎯 COMMAND RESULT: {result}")
                        return result
                
                # Then try to find any valid ticker-like words
                for word in words:
                    if len(word) >= 2 and len(word) <= 5 and word.isalpha() and word not in excluded_words:
                        logger.info(f"🎯 COMMAND DETECTED: Close position {word} (detected ticker) from message: '{user_message}'")
                        result = self._close_position(word)
                        logger.info(f"🎯 COMMAND RESULT: {result}")
                        return result
                        
                return {"error": "Could not identify which position to close. Please specify symbol (e.g., 'close AAPL position')"}
            
            # Cancel orders
            elif any(phrase in message_lower for phrase in ['cancel all orders', 'cancel orders', 'cancel my orders']):
                logger.info("Executing: Cancel all orders")
                return self._cancel_orders()
            
            # Check account/portfolio
            elif any(phrase in message_lower for phrase in ['check my portfolio', 'show my portfolio', 'account balance', 'my account', 'portfolio status']):
                logger.info("Executing: Get account info")
                return self._get_account_info()
            
            # Check positions
            elif any(phrase in message_lower for phrase in ['show my positions', 'current positions', 'what positions', 'my positions']):
                logger.info("Executing: Get positions")
                return self._get_positions()
            
            # Buy orders
            elif 'buy' in message_lower:
                # Extract quantity and symbol using simple approach
                import re
                words = user_message.upper().split()
                qty = 1
                symbol = None
                
                # Find quantity (number before 'shares' or near 'buy')
                for i, word in enumerate(words):
                    if word == 'BUY' and i + 1 < len(words):
                        # Check if next word is a number
                        try:
                            qty = float(words[i + 1])
                            break
                        except ValueError:
                            pass
                
                # Find symbol (2-5 letter word that looks like a ticker, exclude common words)
                excluded_words = ['BUY', 'SHARES', 'SHARE', 'OF', 'THE', 'STOCK', 'STOCKS']
                for word in words:
                    if len(word) >= 2 and len(word) <= 5 and word.isalpha() and word not in excluded_words:
                        symbol = word
                        break
                
                if symbol:
                    logger.info(f"🎯 COMMAND DETECTED: Buy {qty} shares of {symbol}")
                    result = self._place_order(symbol, qty, "buy")
                    logger.info(f"🎯 COMMAND RESULT: {result}")
                    return result
                
            # Sell orders
            elif 'sell' in message_lower:
                import re
                sell_pattern = r'sell\s+(?:(\d+(?:\.\d+)?)\s+)?([A-Z]{1,5})'
                match = re.search(sell_pattern, user_message.upper())
                if match:
                    qty = float(match.group(1)) if match.group(1) else 1
                    symbol = match.group(2)
                    logger.info(f"Executing: Sell {qty} shares of {symbol}")
                    return self._place_order(symbol, qty, "sell")
            
            # Get market data/price
            elif any(phrase in message_lower for phrase in ['price of', 'what is', 'how much is', 'quote for']):
                import re
                # Look for stock symbols in the message
                symbol_pattern = r'\b([A-Z]{1,5})\b'
                matches = re.findall(symbol_pattern, user_message.upper())
                symbols = [s for s in matches if s not in ['PRICE', 'WHAT', 'IS', 'HOW', 'MUCH', 'QUOTE', 'FOR', 'THE', 'OF']]
                if symbols:
                    symbol = symbols[0]
                    logger.info(f"Executing: Get market data for {symbol}")
                    return self._get_market_data(symbol)
            
            return None  # No command detected
            
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            return {"error": f"Failed to execute command: {str(e)}"}
    
    def get_response(self, user_message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get AI response to user message with optional context"""
        try:
            if not self.client:
                return self._get_fallback_response(user_message)
            
            # First, check for and execute trading commands
            logger.info(f"🔍 Checking for commands in: '{user_message}'")
            command_result = self._detect_and_execute_command(user_message)
            logger.info(f"🔍 Command detection result: {command_result}")
            
            # Prepare context information
            context_info = ""
            if context:
                if 'portfolio' in context:
                    portfolio = context['portfolio']
                    context_info += f"\nUser's Portfolio Context:\n"
                    context_info += f"- Total Value: ${portfolio.get('total_value', 0):,.2f}\n"
                    context_info += f"- Day P&L: ${portfolio.get('day_pnl', 0):,.2f} ({portfolio.get('day_change', 0):.2f}%)\n"
                    context_info += f"- Positions: {portfolio.get('positions_count', 0)}\n"
                
                if 'market_data' in context:
                    market_data = context['market_data']
                    context_info += f"\nMarket Data Context:\n"
                    for symbol, data in market_data.items():
                        context_info += f"- {symbol}: ${data.get('price', 0):.2f} ({data.get('change_percent', 0):+.2f}%)\n"
            
            # Add command execution results to context
            if command_result:
                context_info += f"\nCommand Execution Results:\n{json.dumps(command_result, indent=2)}\n"
                
                if context and 'market_status' in context:
                    status = context['market_status']
                    context_info += f"\nMarket Status: {status.get('market_state', 'Unknown')}\n"
            
            # Prepare messages for OpenAI
            messages = [
                {"role": "system", "content": self.system_prompt + context_info}
            ]
            
            # Add conversation history
            for msg in self.conversation_history[-self.max_history:]:
                messages.append(msg)
            
            # Add current user message
            messages.append({"role": "user", "content": user_message})
            
            # Get response from OpenAI
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Use latest model
                messages=messages,
                max_tokens=1000,
                temperature=0.7,
                presence_penalty=0.1,
                frequency_penalty=0.1
            )
            
            ai_response = response.choices[0].message.content
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": ai_response})
            
            # Keep history manageable
            if len(self.conversation_history) > self.max_history * 2:
                self.conversation_history = self.conversation_history[-self.max_history:]
            
            result = {
                "response": ai_response,
                "timestamp": datetime.now().isoformat(),
                "model": "gpt-4o",
                "context_used": bool(context),
                "conversation_length": len(self.conversation_history)
            }
            
            # Add command execution info if command was executed
            if command_result is not None:
                result["command_executed"] = True
                result["command_result"] = command_result
            else:
                result["command_executed"] = False
                
            return result
            
        except Exception as e:
            logger.error(f"Error getting AI response: {e}")
            return self._get_fallback_response(user_message)
    
    def _get_fallback_response(self, user_message: str) -> Dict[str, Any]:
        """Intelligent fallback responses when OpenAI is not available"""
        message_lower = user_message.lower()
        
        # Basic conversational responses
        if any(word in message_lower for word in ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']):
            response = f"Hello! I'm your AI trading assistant. I'm here to help you with market analysis, trading strategies, and portfolio management. What would you like to discuss today?"
        
        elif any(word in message_lower for word in ['how are you', 'how are you doing', 'how is it going']):
            response = "I'm doing well, thank you for asking! I'm ready to help you with trading and investment questions. The markets are always moving, so there's always something interesting to analyze. What's on your mind today?"
        
        elif 'today' in message_lower and any(word in message_lower for word in ['day', 'date', 'what day']):
            response = f"Today is {datetime.now().strftime('%A, %B %d, %Y')}. The markets are {'open' if datetime.now().weekday() < 5 and 9 <= datetime.now().hour < 16 else 'closed'}. What would you like to know about today's market action?"
        
        elif 'time' in message_lower:
            response = f"It's currently {datetime.now().strftime('%I:%M %p')} on {datetime.now().strftime('%A, %B %d, %Y')}. How can I help you with your trading today?"
        
        # Asset-specific responses
        elif 'nvda' in message_lower or 'nvidia' in message_lower:
            response = """NVIDIA (NVDA) is a fascinating stock to analyze! Here's my perspective:

**Strengths:**
• AI/ML chip dominance with their GPU technology
• Strong data center revenue growth
• Gaming market leadership
• Growing automotive and edge computing segments

**Considerations:**
• High valuation means higher volatility
• Competition from AMD, Intel, and custom chips
• Cyclical nature of semiconductor industry
• Regulatory risks in key markets

**Trading Perspective:**
• Popular for both swing and momentum trading
• Watch for earnings reactions and AI news catalysts
• Consider position sizing given volatility
• Technical levels often respected due to high volume

What specific aspect of NVDA interests you most? Technical analysis, fundamental outlook, or trading strategies?"""
        
        elif any(word in message_lower for word in ['bitcoin', 'btc', 'crypto', 'ethereum', 'eth']):
            response = """Cryptocurrency markets are incredibly dynamic! Here's my take:

**Market Dynamics:**
• 24/7 trading creates unique opportunities and risks
• High volatility means significant profit potential but also risk
• Correlation with traditional markets has increased
• Regulatory developments drive major price movements

**Trading Considerations:**
• DCA strategies work well for long-term accumulation
• Technical analysis often more reliable than fundamentals
• Risk management is crucial - crypto can move 10%+ daily
• Consider market cycles and Bitcoin halving patterns

**Popular Strategies:**
• Swing trading on weekly/monthly timeframes
• Momentum trading during bull runs
• Range trading during consolidation phases

Which crypto assets are you most interested in? I can help analyze specific coins or discuss broader market trends."""
        
        elif any(word in message_lower for word in ['portfolio', 'position', 'holdings']):
            response = """Portfolio management is the foundation of successful investing! Here's my approach:

**Key Principles:**
• Diversification across asset classes (stocks, bonds, crypto, REITs)
• Position sizing based on conviction and risk tolerance
• Regular rebalancing to maintain target allocations
• Risk management through stop losses and position limits

**Analysis Framework:**
• Review correlation between holdings
• Assess sector and geographic concentration
• Monitor performance vs benchmarks
• Track risk-adjusted returns (Sharpe ratio)

**Optimization Strategies:**
• Tax-loss harvesting for tax efficiency
• Dollar-cost averaging for new positions
• Momentum vs mean-reversion rebalancing

I'd love to help analyze your specific portfolio! What's your current allocation or what areas would you like to focus on?"""
        
        elif any(word in message_lower for word in ['strategy', 'trading', 'buy', 'sell']):
            response = """Trading strategies should align with your personality and market conditions. Here are my thoughts:

**Strategy Types:**
• **Swing Trading**: 3-30 day holds, good for busy professionals
• **Day Trading**: Same-day entries/exits, requires full attention
• **Momentum**: Follow trends and breakouts
• **Mean Reversion**: Buy oversold, sell overbought

**Risk Management:**
• Never risk more than 1-2% per trade
• Use stop losses religiously
• Position sizing based on volatility
• Keep detailed trading journal

**Market Analysis:**
• Combine technical and fundamental analysis
• Watch for volume confirmation
• Consider market sentiment and news flow
• Adapt strategies to market conditions

What type of trading timeframe appeals to you most? I can dive deeper into specific strategies that might work for your situation."""
        
        else:
            response = f"""I'm your comprehensive AI trading assistant! I can discuss:

**Market Analysis**: Stocks, crypto, ETFs, forex, commodities
**Trading Strategies**: Swing, day, momentum, options strategies  
**Portfolio Management**: Diversification, risk management, rebalancing
**Technical Analysis**: Chart patterns, indicators, support/resistance
**Risk Management**: Position sizing, stop losses, portfolio protection

**Current Market Context**: It's {datetime.now().strftime('%A, %B %d')} and the markets are {'likely active' if datetime.now().weekday() < 5 else 'closed for the weekend'}.

I'm designed to provide thoughtful, personalized guidance based on your specific questions and situation. What would you like to explore today?

Feel free to ask about specific assets, strategies, or market conditions - I'm here to help you make more informed trading decisions!"""
        
        return {
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "model": "intelligent_fallback",
            "context_used": False,
            "conversation_length": 0
        }
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation"""
        return {
            "total_messages": len(self.conversation_history),
            "conversation_started": len(self.conversation_history) > 0,
            "last_message_time": datetime.now().isoformat() if self.conversation_history else None,
            "api_available": self.client is not None
        }

# Global instance
conversational_ai = TradingConversationalAI()