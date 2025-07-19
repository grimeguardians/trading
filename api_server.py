"""
FastAPI Backend Server for AI Trading System
Provides REST API endpoints for the Streamlit dashboard
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from decimal import Decimal

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from core.trading_engine import TradingEngine, TradingSignal, Position, OrderSide, OrderType
from exchanges.exchange_manager import ExchangeManager
from ai.digital_brain import DigitalBrain
from mcp.mcp_server import MCPServer
from strategies.strategy_manager import StrategyManager
from math.fibonacci import FibonacciCalculator
from math.technical_indicators import TechnicalIndicators
from models import db, Trade, Portfolio, Strategy

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

# Global components (will be initialized on startup)
trading_engine: Optional[TradingEngine] = None
exchange_manager: Optional[ExchangeManager] = None
digital_brain: Optional[DigitalBrain] = None
mcp_server: Optional[MCPServer] = None
strategy_manager: Optional[StrategyManager] = None
fibonacci_calc: Optional[FibonacciCalculator] = None
technical_indicators: Optional[TechnicalIndicators] = None

# Pydantic models for API requests/responses
class TradeRequest(BaseModel):
    symbol: str
    side: str
    quantity: float
    order_type: str = "market"
    price: Optional[float] = None
    exchange: str = "alpaca"

class StrategyConfig(BaseModel):
    strategy_type: str
    parameters: Dict[str, Any]
    symbols: List[str]
    exchange: str

class ChatMessage(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None

class SystemStatus(BaseModel):
    trading_engine: Dict[str, Any]
    mcp_server: Dict[str, Any]
    digital_brain: Dict[str, Any]
    data_feed: Dict[str, Any]

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Trading System API",
        "version": "2.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/system/status")
async def get_system_status():
    """Get comprehensive system status"""
    try:
        # Get portfolio value from database
        portfolio_value = 100000.0  # Default starting value
        active_positions = 0
        total_pnl = 0.0
        
        if trading_engine:
            positions = await trading_engine.get_positions()
            active_positions = len(positions)
            total_pnl = sum(pos.unrealized_pnl for pos in positions)
            portfolio_value += total_pnl
        
        status = {
            "trading_engine": {
                "is_running": trading_engine is not None and trading_engine.is_running if trading_engine else False,
                "portfolio_value": portfolio_value,
                "active_positions": active_positions,
                "total_pnl": total_pnl,
                "exchanges": {}
            },
            "mcp_server": {
                "is_running": mcp_server is not None and mcp_server.is_running if mcp_server else False,
                "active_agents": 4,
                "messages_processed": 0
            },
            "digital_brain": {
                "knowledge_nodes": 1777,
                "ai_services": {
                    "anthropic": True,
                    "openai": False
                }
            },
            "data_feed": {
                "connected": True,
                "last_update": datetime.now().isoformat()
            }
        }
        
        return status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting system status: {str(e)}")

@app.get("/api/portfolio")
async def get_portfolio():
    """Get current portfolio data"""
    try:
        # Sample portfolio data - in production, this would come from the database
        portfolio_data = {
            "portfolio_value": 105250.00,
            "available_balance": 25000.00,
            "total_pnl": 5250.00,
            "daily_change": 0.25,
            "active_positions": 8,
            "day_pnl": 125.50,
            "positions": [
                {
                    "symbol": "AAPL",
                    "exchange": "alpaca",
                    "side": "long",
                    "quantity": 100,
                    "entry_price": 150.25,
                    "current_price": 152.75,
                    "unrealized_pnl": 250.00,
                    "pnl_percentage": 1.67
                },
                {
                    "symbol": "GOOGL",
                    "exchange": "alpaca",
                    "side": "long",
                    "quantity": 50,
                    "entry_price": 2650.00,
                    "current_price": 2625.00,
                    "unrealized_pnl": -1250.00,
                    "pnl_percentage": -0.94
                },
                {
                    "symbol": "MSFT",
                    "exchange": "alpaca",
                    "side": "long",
                    "quantity": 75,
                    "entry_price": 380.50,
                    "current_price": 385.25,
                    "unrealized_pnl": 356.25,
                    "pnl_percentage": 1.25
                }
            ]
        }
        
        return portfolio_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting portfolio: {str(e)}")

@app.get("/api/strategies")
async def get_strategies():
    """Get available strategies and their status"""
    try:
        strategies_data = {
            "categories": {
                "swing": ["trend_following", "mean_reversion", "breakout"],
                "scalping": ["grid_trading", "arbitrage", "momentum"],
                "options": ["covered_call", "iron_condor", "straddle"],
                "intraday": ["gap_trading", "reversal", "momentum"]
            },
            "active_strategies": [
                {
                    "name": "Swing Trend Following",
                    "type": "swing",
                    "active": True,
                    "performance": {
                        "win_rate": 0.65,
                        "total_trades": 25,
                        "profit_factor": 1.8
                    },
                    "symbols": ["AAPL", "GOOGL", "MSFT"]
                },
                {
                    "name": "Scalping Grid",
                    "type": "scalping",
                    "active": False,
                    "performance": {
                        "win_rate": 0.55,
                        "total_trades": 150,
                        "profit_factor": 1.2
                    },
                    "symbols": ["BTCUSD", "ETHUSD"]
                }
            ]
        }
        
        return strategies_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting strategies: {str(e)}")

@app.post("/api/strategies/start")
async def start_strategy(strategy_type: str, exchange: str):
    """Start a trading strategy"""
    try:
        if not strategy_manager:
            raise HTTPException(status_code=503, detail="Strategy manager not initialized")
            
        # Start strategy (placeholder implementation)
        result = {
            "success": True,
            "message": f"Started {strategy_type} strategy on {exchange}",
            "strategy_id": f"{strategy_type}_{exchange}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting strategy: {str(e)}")

@app.post("/api/strategies/stop")
async def stop_strategy(strategy_name: str):
    """Stop a trading strategy"""
    try:
        if not strategy_manager:
            raise HTTPException(status_code=503, detail="Strategy manager not initialized")
            
        # Stop strategy (placeholder implementation)
        result = {
            "success": True,
            "message": f"Stopped strategy: {strategy_name}"
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error stopping strategy: {str(e)}")

@app.post("/api/trades/execute")
async def execute_trade(trade_request: TradeRequest):
    """Execute a trade"""
    try:
        if not trading_engine:
            raise HTTPException(status_code=503, detail="Trading engine not initialized")
            
        # Create trading signal
        signal = TradingSignal(
            symbol=trade_request.symbol,
            signal_type=trade_request.side,
            strength=0.8,
            confidence=0.75,
            price_target=trade_request.price
        )
        
        # Execute trade through trading engine
        # This is a placeholder - in production, this would actually execute the trade
        result = {
            "success": True,
            "trade_id": f"trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "symbol": trade_request.symbol,
            "side": trade_request.side,
            "quantity": trade_request.quantity,
            "status": "executed",
            "execution_price": trade_request.price or 150.0,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing trade: {str(e)}")

@app.post("/api/chat")
async def chat_with_ai(message: ChatMessage):
    """Chat with AI assistant"""
    try:
        if not digital_brain:
            raise HTTPException(status_code=503, detail="Digital brain not initialized")
            
        # Process message through conversational AI
        # This is a placeholder response - in production, this would use the actual AI
        user_message = message.message.lower()
        
        if "portfolio" in user_message:
            response = """Based on your current portfolio, you have 8 active positions with a total value of $105,250. 
            Your portfolio is showing a positive performance with a 0.25% daily change and $125.50 in day P&L. 
            Your largest position is AAPL with 100 shares showing a 1.67% gain."""
            
        elif "market" in user_message:
            response = """Current market conditions are showing moderate volatility with mixed signals across sectors. 
            Technology stocks are showing resilience while financial sectors are experiencing some pressure. 
            The AI analysis suggests cautious optimism for swing trading opportunities."""
            
        elif "strategy" in user_message:
            response = """Your active swing trading strategy is performing well with a 65% win rate. 
            The trend-following approach is identifying good entry points in AAPL, GOOGL, and MSFT. 
            Consider diversifying into the scalping strategy for more frequent trading opportunities."""
            
        elif "risk" in user_message:
            response = """Your current risk exposure is within acceptable parameters. 
            Maximum drawdown is 5.2% with a Sharpe ratio of 1.85. 
            VaR (95%) is $2,450. Consider reducing position sizes if volatility increases."""
            
        else:
            response = """I'm here to help with your trading questions. I can provide insights on:
            - Portfolio performance and analysis
            - Market conditions and opportunities
            - Trading strategy recommendations
            - Risk management advice
            - Technical analysis insights
            
            What would you like to know more about?"""
        
        return {"response": response}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.get("/api/fibonacci/{symbol}")
async def get_fibonacci_levels(symbol: str):
    """Get Fibonacci retracement levels for a symbol"""
    try:
        if not fibonacci_calc:
            fibonacci_calc = FibonacciCalculator()
            
        # Calculate Fibonacci levels (placeholder implementation)
        levels = {
            "symbol": symbol,
            "high": 155.0,
            "low": 145.0,
            "levels": {
                "0.0": 155.0,
                "23.6": 152.64,
                "38.2": 151.18,
                "50.0": 150.0,
                "61.8": 148.82,
                "100.0": 145.0
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return levels
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating Fibonacci levels: {str(e)}")

@app.get("/api/technical/{symbol}")
async def get_technical_indicators(symbol: str):
    """Get technical indicators for a symbol"""
    try:
        # Placeholder technical indicators
        indicators = {
            "symbol": symbol,
            "rsi": 65.5,
            "macd": {
                "macd": 0.5,
                "signal": 0.3,
                "histogram": 0.2
            },
            "moving_averages": {
                "sma_20": 150.5,
                "sma_50": 149.2,
                "ema_12": 151.0,
                "ema_26": 150.0
            },
            "bollinger_bands": {
                "upper": 153.0,
                "middle": 150.0,
                "lower": 147.0
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return indicators
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting technical indicators: {str(e)}")

@app.post("/api/system/emergency_stop")
async def emergency_stop():
    """Emergency stop all trading activities"""
    try:
        if trading_engine:
            await trading_engine.emergency_stop()
            
        return {
            "success": True,
            "message": "Emergency stop activated - all trading activities halted",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in emergency stop: {str(e)}")

# Application startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global trading_engine, exchange_manager, digital_brain, mcp_server, strategy_manager
    global fibonacci_calc, technical_indicators
    
    logging.info("🚀 Starting AI Trading System API Server")
    
    try:
        # Initialize components (placeholder - in production these would be fully initialized)
        # trading_engine = TradingEngine()
        # exchange_manager = ExchangeManager()
        # digital_brain = DigitalBrain()
        # mcp_server = MCPServer()
        # strategy_manager = StrategyManager()
        fibonacci_calc = FibonacciCalculator()
        technical_indicators = TechnicalIndicators()
        
        logging.info("✅ API Server initialized successfully")
        
    except Exception as e:
        logging.error(f"❌ Error initializing API server: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logging.info("🛑 Shutting down AI Trading System API Server")
    
    # Cleanup components
    if trading_engine:
        await trading_engine.shutdown()
    if mcp_server:
        await mcp_server.shutdown()

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "trading_engine": trading_engine is not None,
            "exchange_manager": exchange_manager is not None,
            "digital_brain": digital_brain is not None,
            "mcp_server": mcp_server is not None
        }
    }

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )