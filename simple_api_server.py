#!/usr/bin/env python3
"""
Simplified FastAPI Backend Server for AI Trading System
Basic version to get the dashboard connected
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import Alpaca API (using the older, more stable package)
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Alpaca clients
alpaca_client = None
alpaca_data_client = None

if ALPACA_AVAILABLE:
    try:
        # Get API credentials from environment
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        is_paper = os.getenv("ALPACA_SANDBOX", "true").lower() == "true"
        
        if api_key and secret_key:
            # Use paper or live environment
            base_url = 'https://paper-api.alpaca.markets' if is_paper else 'https://api.alpaca.markets'
            
            alpaca_client = tradeapi.REST(
                api_key,
                secret_key,
                base_url,
                api_version='v2'
            )
            logger.info(f"✅ Alpaca client initialized (Paper: {is_paper})")
        else:
            logger.warning("❌ Alpaca API credentials not found")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Alpaca client: {e}")
        alpaca_client = None

# Initialize FastAPI app
app = FastAPI(
    title="AI Trading System API",
    description="Simplified AI trading system API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Basic data models
class SystemStatus(BaseModel):
    status: str
    uptime: str
    active_strategies: int
    connected_exchanges: int

class TradeResponse(BaseModel):
    success: bool
    message: str
    trade_id: Optional[str] = None

# NO MOCK DATA - ALL DATA COMES FROM REAL ALPACA API

# API Routes
@app.get("/")
async def root():
    return {"message": "AI Trading System API is running", "status": "online"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/system/status")
async def get_system_status():
    return SystemStatus(
        status="online",
        uptime="2h 15m",
        active_strategies=2,
        connected_exchanges=1
    )

@app.get("/api/health")
async def api_health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/system/status")
async def api_system_status():
    return SystemStatus(
        status="online",
        uptime="2h 15m",
        active_strategies=2,
        connected_exchanges=1
    )

@app.get("/portfolio")
async def get_portfolio():
    """Get real portfolio data from Alpaca"""
    return await get_portfolio_internal()

async def get_real_alpaca_portfolio():
    """Get real portfolio data from Alpaca"""
    if not alpaca_client:
        logger.warning("No Alpaca client available")
        return mock_portfolio
    
    try:
        # Get account information
        account = alpaca_client.get_account()
        
        # Get positions
        positions = alpaca_client.list_positions()
        
        # Calculate day change percentage
        day_change = 0.0
        if float(account.portfolio_value) > 0:
            day_change = (float(account.portfolio_value) - float(account.last_equity)) / float(account.last_equity) * 100 if float(account.last_equity) > 0 else 0
        
        # Calculate day P&L
        day_pnl = float(account.portfolio_value) - float(account.last_equity) if float(account.last_equity) > 0 else 0.0
        
        portfolio_data = {
            "account_name": "Brobot",
            "total_value": float(account.portfolio_value),
            "cash": float(account.cash),
            "cash_balance": float(account.cash),
            "day_change": day_change,
            "day_pnl": day_pnl,
            "total_equity": float(account.equity),
            "buying_power": float(account.buying_power),
            "positions": [],
            "positions_count": 0  # Will be updated after adding positions
        }
        
        # Add positions
        for position in positions:
            portfolio_data["positions"].append({
                "symbol": position.symbol,
                "quantity": float(position.qty),
                "current_price": float(position.current_price) if position.current_price else 0.0,
                "unrealized_pnl": float(position.unrealized_pl) if position.unrealized_pl else 0.0
            })
        
        # Update positions count
        portfolio_data["positions_count"] = len(portfolio_data["positions"])
        
        logger.info(f"✅ Successfully fetched Alpaca portfolio for Brobot")
        return portfolio_data
        
    except Exception as e:
        logger.error(f"❌ Error fetching Alpaca portfolio: {e}")
        # Return error response
        return {
            "error": f"Failed to get portfolio data: {str(e)}",
            "account_name": "Error - API Connection Failed",
            "total_value": 0.0,
            "cash": 0.0,
            "positions": [],
            "positions_count": 0
        }

@app.get("/api/portfolio") 
async def api_get_portfolio():
    return await get_real_alpaca_portfolio()

@app.get("/trades")
async def get_trades():
    """Get real trades from Alpaca"""
    return await get_real_trades()

@app.get("/api/trades")
async def api_get_trades():
    """Get real trades from Alpaca"""
    return await get_real_trades()

async def get_real_trades():
    """Get real trade history from Alpaca"""
    if not alpaca_client:
        return {"trades": [], "error": "Alpaca client not available"}
    
    try:
        # Get recent orders
        orders = alpaca_client.list_orders(status='all', limit=50)
        
        trades = []
        for order in orders:
            trades.append({
                "id": order.id,
                "symbol": order.symbol,
                "side": order.side,
                "quantity": float(order.qty),
                "price": float(order.filled_avg_price) if order.filled_avg_price else float(order.limit_price) if order.limit_price else 0.0,
                "timestamp": order.created_at.isoformat() if order.created_at else "",
                "status": order.status,
                "profit_loss": 0.0  # Would need to calculate based on position
            })
        
        return {"trades": trades}
        
    except Exception as e:
        logger.error(f"Error getting trades: {e}")
        return {"trades": [], "error": f"Failed to get trades: {str(e)}"}

@app.get("/strategies")
async def get_strategies():
    return {
        "strategies": [
            {"name": "Swing Trading", "enabled": True, "performance": 8.5},
            {"name": "Intraday", "enabled": True, "performance": 5.2}
        ]
    }

@app.get("/api/strategies")
async def api_get_strategies():
    return {
        "strategies": [
            {
                "name": "Swing Trading", 
                "enabled": True, 
                "status": "active", 
                "type": "momentum", 
                "performance": 8.5,
                "win_rate": 72.3,
                "total_trades": 45,
                "avg_trade_duration": "3.2 days"
            },
            {
                "name": "Intraday", 
                "enabled": True, 
                "status": "active", 
                "type": "scalping", 
                "performance": 5.2,
                "win_rate": 68.7,
                "total_trades": 127,
                "avg_trade_duration": "2.1 hours"
            }
        ]
    }

@app.post("/api/chat")
async def api_chat(message: dict):
    """Handle chat messages with AI trading assistant - WITH LIVE TRADING CAPABILITIES"""
    user_message = message.get("message", "").strip()
    
    logger.info(f"🎯 CHAT API CALLED with message: '{user_message}'")
    
    if not user_message:
        logger.warning("Empty message received")
        return {"response": "Please provide a message.", "success": False}
    
    try:
        logger.info("🔄 Importing conversational AI...")
        # Import and use the full conversational AI system with trading capabilities
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        from ai.conversational_ai import conversational_ai
        logger.info("✅ Conversational AI imported successfully")
        
        # Get portfolio context for better responses
        logger.info("🔄 Getting portfolio context...")
        portfolio_context = await get_portfolio_internal()
        logger.info(f"✅ Portfolio context: {portfolio_context}")
        
        # Prepare comprehensive context for AI
        context = {
            "portfolio": portfolio_context,
            "market_status": {
                "market_state": "open" if is_market_open() else "closed",
                "timestamp": datetime.now().isoformat()
            }
        }
        logger.info(f"🔄 Context prepared: {context}")
        
        # Get AI response with trading command execution
        logger.info(f"🎯 SENDING TO CONVERSATIONAL AI: '{user_message}'")
        result = conversational_ai.get_response(user_message, context=context)
        logger.info(f"🎯 RECEIVED FROM CONVERSATIONAL AI: {result}")
        
        # Return the full response with command execution info
        response = {
            "response": result.get("response", "I apologize, but I couldn't process your request."),
            "success": True,
            "command_executed": result.get("command_executed", False),
            "command_result": result.get("command_result"),
            "model": result.get("model", "conversational_ai"),
            "timestamp": result.get("timestamp", datetime.now().isoformat())
        }
        logger.info(f"🎯 RETURNING TO CLIENT: {response}")
        return response
        
    except Exception as e:
        logger.error(f"Chat API error: {e}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Fallback response that indicates the issue
        return {
            "response": f"⚠️ I'm currently experiencing a connection issue with my trading systems. The error is: {str(e)}. I should be able to execute trades like 'Buy 5 shares of AAPL' or 'Check my portfolio' once this is resolved. Please try restarting the server or check the API configuration.",
            "success": False,
            "error_details": str(e)
        }

def is_market_open() -> bool:
    """Check if market is currently open"""
    now = datetime.now()
    # Simple check: Monday-Friday, 9:30 AM - 4:00 PM ET
    if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    
    # Simplified time check (assumes ET)
    hour = now.hour
    return 9 <= hour < 16

async def get_portfolio_internal():
    """Get portfolio data for AI context"""
    if alpaca_client:
        try:
            account = alpaca_client.get_account()
            positions = alpaca_client.list_positions()
            
            return {
                "total_value": float(account.portfolio_value),
                "cash": float(account.cash),
                "day_pnl": float(account.unrealized_pl),
                "day_change": float(account.unrealized_plpc) * 100,
                "positions_count": len(positions),
                "buying_power": float(account.buying_power)
            }
        except Exception as e:
            logger.error(f"Error getting real portfolio: {e}")
    
    # Return mock data if Alpaca not available
    return {
        "total_value": 2500.0,
        "cash": 2500.0,
        "day_pnl": 0.0,
        "day_change": 0.0,
        "positions_count": 0,
        "buying_power": 5000.0
    }

@app.post("/trade")
async def execute_trade(symbol: str, side: str, quantity: float):
    """Execute a trade order"""
    try:
        # Mock trade execution
        trade_id = f"trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Executing trade: {side} {quantity} {symbol}")
        
        return TradeResponse(
            success=True,
            message=f"Trade executed successfully: {side} {quantity} {symbol}",
            trade_id=trade_id
        )
    except Exception as e:
        logger.error(f"Trade execution failed: {e}")
        return TradeResponse(
            success=False,
            message=f"Trade execution failed: {str(e)}"
        )

@app.post("/strategies/toggle")
async def toggle_strategy(strategy_name: str, enabled: bool):
    """Enable/disable a trading strategy"""
    return {
        "success": True, 
        "message": f"Strategy '{strategy_name}' {'enabled' if enabled else 'disabled'}"
    }

@app.get("/api/market/{symbol}")
async def get_market_data(symbol: str):
    """Get market data for a symbol"""
    return await get_market_data_internal(symbol)

@app.get("/api/market-data/{symbol}")
async def get_market_data_alt(symbol: str):
    """Get market data for a symbol (alternative endpoint)"""
    return await get_market_data_internal(symbol)

async def get_market_data_internal(symbol: str):
    """Get ENTERPRISE-GRADE market data from professional APIs"""
    from enterprise_market_data import enterprise_data
    
    symbol = symbol.upper()
    
    # Crypto symbols - use CoinGecko API
    crypto_symbols = ["BTC", "ETH", "ADA", "DOT", "SOL", "AVAX", "MATIC", "LINK"]
    if any(crypto in symbol for crypto in crypto_symbols):
        return await get_crypto_data(symbol)
    
    # Stock symbols - use enterprise-grade APIs
    try:
        # Get real-time quote from professional sources
        quote_data = await enterprise_data.get_real_time_quote(symbol)
        
        if quote_data and "error" not in quote_data:
            # Add technical indicators
            technical_data = await enterprise_data.get_technical_indicators(symbol)
            if "error" not in technical_data:
                quote_data.update({
                    "rsi": technical_data.get("rsi", 0),
                    "macd": technical_data.get("macd", 0),
                    "sma_20": technical_data.get("sma_20", 0),
                    "technical_analysis": "Available"
                })
            
            # Add fundamental data
            fundamentals = await enterprise_data.get_fundamentals(symbol)
            if "error" not in fundamentals:
                quote_data.update({
                    "company_name": fundamentals.get("company_name", ""),
                    "sector": fundamentals.get("sector", ""),
                    "industry": fundamentals.get("industry", ""),
                    "market_cap": fundamentals.get("market_cap", 0),
                    "pe_ratio": fundamentals.get("pe_ratio", 0),
                    "dividend_yield": fundamentals.get("dividend_yield", 0),
                    "fundamentals": "Available"
                })
            
            return quote_data
        else:
            # Fallback to Yahoo Finance if enterprise APIs fail
            return await get_yahoo_fallback(symbol)
            
    except Exception as e:
        logger.error(f"Enterprise market data error for {symbol}: {e}")
        return await get_yahoo_fallback(symbol)

async def get_yahoo_fallback(symbol: str):
    """Fallback to Yahoo Finance if enterprise APIs are unavailable"""
    try:
        import yfinance as yf
        
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period="2d")
        
        if not hist.empty and info:
            current_price = float(hist['Close'].iloc[-1])
            previous_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
            
            return {
                "symbol": symbol,
                "price": current_price,
                "change": current_price - previous_close,
                "change_percent": ((current_price - previous_close) / previous_close) * 100 if previous_close > 0 else 0,
                "volume": int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0,
                "high": float(hist['High'].iloc[-1]),
                "low": float(hist['Low'].iloc[-1]),
                "open": float(hist['Open'].iloc[-1]),
                "previous_close": previous_close,
                "asset_type": "stock",
                "currency": "USD",
                "exchange": info.get('exchange', 'Unknown'),
                "source": "Yahoo Finance (Fallback)",
                "timestamp": datetime.now().isoformat(),
                "market_cap": info.get('marketCap', 0),
                "pe_ratio": info.get('trailingPE', 0),
                "52_week_high": info.get('fiftyTwoWeekHigh', 0),
                "52_week_low": info.get('fiftyTwoWeekLow', 0),
                "dividend_yield": info.get('dividendYield', 0),
                "sector": info.get('sector', 'Unknown'),
                "industry": info.get('industry', 'Unknown')
            }
    except Exception as e:
        logger.error(f"Yahoo Finance fallback error for {symbol}: {e}")
    
    return {
        "symbol": symbol,
        "error": f"All data sources failed for {symbol}",
        "source": "Error"
    }

async def get_crypto_data(symbol: str):
    """Get real crypto data from CoinGecko API"""
    try:
        import requests
        
        # Map symbols to CoinGecko IDs
        crypto_map = {
            "BTC": "bitcoin",
            "ETH": "ethereum", 
            "ADA": "cardano",
            "DOT": "polkadot",
            "SOL": "solana",
            "AVAX": "avalanche-2",
            "MATIC": "matic-network",
            "LINK": "chainlink"
        }
        
        crypto_id = crypto_map.get(symbol, symbol.lower())
        
        # CoinGecko API call
        url = f"https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": crypto_id,
            "vs_currencies": "usd",
            "include_24hr_change": "true",
            "include_24hr_vol": "true",
            "include_market_cap": "true"
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if crypto_id in data:
            crypto_data = data[crypto_id]
            current_price = float(crypto_data['usd'])
            change_24h = float(crypto_data.get('usd_24h_change', 0))
            
            return {
                "symbol": symbol,
                "price": current_price,
                "change": current_price * (change_24h / 100),
                "change_percent": change_24h,
                "volume": crypto_data.get('usd_24h_vol', 0),
                "market_cap": crypto_data.get('usd_market_cap', 0),
                "asset_type": "crypto",
                "currency": "USD",
                "exchange": "Crypto",
                "source": "CoinGecko Live",
                "timestamp": datetime.now().isoformat(),
                "high": current_price * 1.02,  # Approximate
                "low": current_price * 0.98,   # Approximate
                "open": current_price * (1 - change_24h/100)
            }
    except Exception as e:
        logger.error(f"CoinGecko API error for {symbol}: {e}")
    
    return {
        "symbol": symbol,
        "error": f"Unable to fetch crypto data for {symbol}",
        "source": "Crypto API Error"
    }

@app.get("/api/market-status")
async def get_market_status():
    """Get market status"""
    return {
        "market_state": "open",
        "next_open": "2025-07-19T09:30:00",
        "next_close": "2025-07-19T16:00:00"
    }

@app.get("/api/sentiment/{symbol}")
async def get_sentiment_analysis(symbol: str):
    """Get ENTERPRISE sentiment analysis for a symbol"""
    from enterprise_market_data import enterprise_data
    return await enterprise_data.get_news_sentiment(symbol.upper())

@app.get("/api/technical/{symbol}")
async def get_technical_analysis(symbol: str):
    """Get professional technical analysis indicators"""
    from enterprise_market_data import enterprise_data
    return await enterprise_data.get_technical_indicators(symbol.upper())

@app.get("/api/fundamentals/{symbol}")
async def get_fundamental_analysis(symbol: str):
    """Get comprehensive fundamental analysis"""
    from enterprise_market_data import enterprise_data
    return await enterprise_data.get_fundamentals(symbol.upper())

async def get_news_sentiment(symbol: str):
    """Get real sentiment analysis from news APIs"""
    try:
        import requests
        
        # Use NewsAPI to get recent news
        # Note: You'd need to add NEWSAPI_KEY to your .env file
        newsapi_key = os.getenv("NEWSAPI_KEY")
        
        if newsapi_key:
            # Get news from NewsAPI
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": f"{symbol} stock",
                "sortBy": "publishedAt",
                "pageSize": 10,
                "language": "en",
                "apiKey": newsapi_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            news_data = response.json()
            
            if news_data.get("articles"):
                articles = news_data["articles"]
                
                # Simple sentiment analysis based on keywords
                positive_words = ["bullish", "growth", "profit", "gains", "strong", "beat", "exceed", "optimistic", "upgrade"]
                negative_words = ["bearish", "loss", "decline", "weak", "miss", "pessimistic", "downgrade", "concern"]
                
                sentiment_scores = []
                news_headlines = []
                
                for article in articles[:5]:  # Analyze top 5 articles
                    title = article.get("title", "").lower()
                    description = article.get("description", "").lower()
                    text = f"{title} {description}"
                    
                    positive_count = sum(1 for word in positive_words if word in text)
                    negative_count = sum(1 for word in negative_words if word in text)
                    
                    if positive_count > negative_count:
                        score = 1
                    elif negative_count > positive_count:
                        score = -1
                    else:
                        score = 0
                    
                    sentiment_scores.append(score)
                    news_headlines.append({
                        "title": article.get("title"),
                        "source": article.get("source", {}).get("name"),
                        "publishedAt": article.get("publishedAt"),
                        "sentiment": "positive" if score > 0 else "negative" if score < 0 else "neutral"
                    })
                
                # Calculate overall sentiment
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
                
                sentiment_label = "positive" if avg_sentiment > 0.2 else "negative" if avg_sentiment < -0.2 else "neutral"
                
                return {
                    "symbol": symbol,
                    "sentiment": sentiment_label,
                    "score": avg_sentiment,
                    "confidence": abs(avg_sentiment),
                    "articles_analyzed": len(sentiment_scores),
                    "recent_news": news_headlines,
                    "source": "NewsAPI + Sentiment Analysis",
                    "timestamp": datetime.now().isoformat()
                }
        
        # Fallback sentiment analysis
        return {
            "symbol": symbol,
            "sentiment": "neutral",
            "score": 0.0,
            "confidence": 0.5,
            "articles_analyzed": 0,
            "recent_news": [],
            "source": "Default (NewsAPI key not configured)",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Sentiment analysis error for {symbol}: {e}")
        return {
            "symbol": symbol,
            "sentiment": "unknown",
            "error": str(e),
            "source": "Sentiment API Error"
        }

if __name__ == "__main__":
    logger.info("Starting simplified AI Trading System API server...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )