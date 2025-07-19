"""
Simplified FastAPI Backend Server for AI Trading System
Fixed version without problematic dependencies
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from decimal import Decimal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="AI Trading System API",
    description="Advanced AI-powered trading system with multi-exchange support",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API requests/responses
class TradeRequest(BaseModel):
    symbol: str
    side: str
    quantity: float
    order_type: str = "market"
    price: Optional[float] = None
    exchange: str = "alpaca"

class ChatMessage(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None

class SystemStatus(BaseModel):
    status: str
    timestamp: str
    components: Dict[str, bool]

# Root endpoint to serve info
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Trading System API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "health": "/api/health",
            "portfolio": "/api/portfolio",
            "positions": "/api/positions",
            "chat": "/api/chat",
            "strategies": "/api/strategies",
            "market_data": "/api/market-data/{symbol}",
            "docs": "/docs"
        }
    }

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "components": {
            "api_server": True,
            "database": True,
            "trading_engine": True
        }
    }

# Portfolio endpoint
@app.get("/api/portfolio")
async def get_portfolio():
    """Get portfolio status"""
    return {
        "total_value": 100000.0,
        "day_change": 0.25,
        "day_pnl": 125.50,
        "positions_count": 8,
        "cash_balance": 25000.0,
        "buying_power": 50000.0
    }

# Positions endpoint
@app.get("/api/positions")
async def get_positions():
    """Get current positions"""
    return [
        {
            "symbol": "AAPL",
            "quantity": 100,
            "avg_price": 150.25,
            "current_price": 152.75,
            "pnl": 250.0,
            "pnl_percent": 1.67
        },
        {
            "symbol": "GOOGL",
            "quantity": 25,
            "avg_price": 2750.00,
            "current_price": 2785.50,
            "pnl": 887.50,
            "pnl_percent": 1.29
        }
    ]

# Enhanced AI chat endpoint
@app.post("/api/chat")
async def chat_with_ai(message: ChatMessage):
    """Chat with AI assistant using OpenAI GPT-4"""
    try:
        from ai.conversational_ai import conversational_ai
        
        # Prepare context from user's request
        context = message.context or {}
        
        # Add current portfolio context if available
        if not context.get('portfolio'):
            context['portfolio'] = {
                'total_value': 100000.0,
                'day_change': 0.25,
                'day_pnl': 125.50,
                'positions_count': 8,
                'cash_balance': 25000.0,
                'buying_power': 50000.0
            }
        
        # Get AI response
        ai_response = conversational_ai.get_response(message.message, context)
        
        # Return all AI response fields including command execution results
        return ai_response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

# Alpaca paper trading endpoints
@app.get("/api/alpaca/account")
async def get_alpaca_account():
    """Get Alpaca account information"""
    try:
        from core.alpaca_paper_trading import alpaca_trader
        return alpaca_trader.get_account_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting account info: {str(e)}")

@app.get("/api/alpaca/positions")
async def get_alpaca_positions():
    """Get Alpaca positions"""
    try:
        from core.alpaca_paper_trading import alpaca_trader
        return alpaca_trader.get_positions()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting positions: {str(e)}")

@app.get("/api/alpaca/orders")
async def get_alpaca_orders(status: str = "open"):
    """Get Alpaca orders"""
    try:
        from core.alpaca_paper_trading import alpaca_trader
        return alpaca_trader.get_orders(status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting orders: {str(e)}")

@app.post("/api/alpaca/trade")
async def place_alpaca_trade(trade_request: TradeRequest):
    """Place a real trade on Alpaca paper trading account"""
    try:
        from core.alpaca_paper_trading import alpaca_trader
        
        result = alpaca_trader.place_order(
            symbol=trade_request.symbol,
            qty=trade_request.quantity,
            side=trade_request.side,
            order_type=trade_request.order_type,
            limit_price=trade_request.price
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error placing trade: {str(e)}")

@app.delete("/api/alpaca/orders/{order_id}")
async def cancel_alpaca_order(order_id: str):
    """Cancel an Alpaca order"""
    try:
        from core.alpaca_paper_trading import alpaca_trader
        return alpaca_trader.cancel_order(order_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cancelling order: {str(e)}")

# Trade execution endpoint (legacy)
@app.post("/api/trade")
async def execute_trade(trade_request: TradeRequest):
    """Execute a trade (legacy endpoint)"""
    try:
        # Simulate trade execution
        return {
            "success": True,
            "order_id": f"ORD_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "message": f"Trade executed: {trade_request.side.upper()} {trade_request.quantity} shares of {trade_request.symbol}",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing trade: {str(e)}")

# Strategies endpoint
@app.get("/api/strategies")
async def get_strategies():
    """Get available strategies"""
    return {
        "strategies": [
            {
                "name": "Swing Trading",
                "type": "swing",
                "status": "active",
                "performance": 12.5,
                "win_rate": 65.0
            },
            {
                "name": "Scalping",
                "type": "scalping", 
                "status": "inactive",
                "performance": 8.3,
                "win_rate": 58.0
            },
            {
                "name": "Options Strategy",
                "type": "options",
                "status": "active",
                "performance": 18.7,
                "win_rate": 72.0
            }
        ]
    }

# Enhanced market data endpoint
@app.get("/api/market-data/{symbol}")
async def get_market_data(symbol: str):
    """Get comprehensive market data for any asset type"""
    try:
        from core.enhanced_market_data_provider import enhanced_market_data_provider
        
        # Get comprehensive market data
        quote = enhanced_market_data_provider.get_comprehensive_quote(symbol.upper())
        
        if quote:
            return {
                "symbol": quote.symbol,
                "asset_type": quote.asset_type.value,
                "price": quote.price,
                "change": quote.change,
                "change_percent": quote.change_percent,
                "volume": quote.volume,
                "high": quote.high,
                "low": quote.low,
                "open": quote.open,
                "previous_close": quote.previous_close,
                "market_cap": quote.market_cap,
                "timestamp": quote.timestamp.isoformat(),
                "source": quote.source,
                "currency": quote.currency,
                "exchange": quote.exchange
            }
        else:
            # Return error if no data found
            return {
                "error": f"No market data found for symbol {symbol}",
                "symbol": symbol,
                "timestamp": datetime.utcnow().isoformat()
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching market data: {str(e)}")

# Market status endpoint
@app.get("/api/market-status")
async def get_market_status():
    """Get market status information"""
    try:
        from core.market_data_provider import market_data_provider
        
        status = market_data_provider.get_market_status()
        return status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching market status: {str(e)}")

# Multiple quotes endpoint
@app.post("/api/market-data/batch")
async def get_multiple_market_data(symbols: List[str]):
    """Get market data for multiple symbols"""
    try:
        from core.market_data_provider import market_data_provider
        
        quotes = market_data_provider.get_multiple_quotes([s.upper() for s in symbols])
        
        result = {}
        for symbol, quote in quotes.items():
            result[symbol] = {
                "symbol": quote.symbol,
                "price": quote.price,
                "change": quote.change,
                "change_percent": quote.change_percent,
                "volume": quote.volume,
                "high": quote.high,
                "low": quote.low,
                "open": quote.open,
                "previous_close": quote.previous_close,
                "timestamp": quote.timestamp.isoformat(),
                "source": quote.source
            }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching market data: {str(e)}")

# API Key management endpoints
@app.get("/api/data-sources")
async def get_data_sources():
    """Get available data sources and their status"""
    try:
        from config.api_keys import api_key_manager
        
        status = api_key_manager.get_status()
        available_providers = api_key_manager.get_available_providers()
        
        return {
            "available_sources": [
                {
                    "name": "Yahoo Finance",
                    "key": "yahoo",
                    "free": True,
                    "status": "active",
                    "description": "Free real-time stock quotes (limited features)"
                },
                {
                    "name": "Alpha Vantage",
                    "key": "alpha_vantage",
                    "free": False,
                    "status": "configured" if status["alpha_vantage"] else "needs_key",
                    "description": "Premium data with 500 requests/day free tier"
                },
                {
                    "name": "Finnhub",
                    "key": "finnhub",
                    "free": False,
                    "status": "configured" if status["finnhub"] else "needs_key",
                    "description": "Real-time data with 60 requests/minute free tier"
                },
                {
                    "name": "Polygon.io",
                    "key": "polygon",
                    "free": False,
                    "status": "configured" if status["polygon"] else "needs_key",
                    "description": "Professional market data (subscription required)"
                }
            ],
            "configured_providers": available_providers,
            "total_sources": len(status),
            "active_sources": len(available_providers) + 1  # +1 for free Yahoo
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting data sources: {str(e)}")

# Popular assets endpoint
@app.get("/api/popular-assets")
async def get_popular_assets():
    """Get popular assets by category"""
    try:
        from core.enhanced_market_data_provider import enhanced_market_data_provider
        
        popular_assets = enhanced_market_data_provider.get_popular_assets()
        
        return {
            "categories": popular_assets,
            "total_assets": sum(len(assets) for assets in popular_assets.values()),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting popular assets: {str(e)}")

# Clear chat history endpoint
@app.post("/api/chat/clear")
async def clear_chat_history():
    """Clear chat conversation history"""
    try:
        from ai.conversational_ai import conversational_ai
        
        conversational_ai.clear_conversation()
        
        return {
            "message": "Chat history cleared successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing chat history: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)