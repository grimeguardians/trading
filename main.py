"""
Advanced AI Trading System with Multi-Exchange Support and MCP Integration
Built on Freqtrade-inspired architecture with Digital Brain integration
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import streamlit as st
from contextlib import asynccontextmanager

# Local imports
from config.settings import Settings, get_settings
from mcp.server import MCPServer
from exchanges.exchange_manager import ExchangeManager
from strategies.strategy_manager import StrategyManager
from trading.portfolio_manager import PortfolioManager
from trading.risk_manager import RiskManager
from ai.digital_brain import DigitalBrain
from ai.conversational_interface import ConversationalInterface
from dashboard.app import create_dashboard
from data.database_models import init_database
from utils.logging_config import setup_logging

# Global instances
settings = get_settings()
mcp_server = None
exchange_manager = None
strategy_manager = None
portfolio_manager = None
risk_manager = None
digital_brain = None
conversational_ai = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global mcp_server, exchange_manager, strategy_manager, portfolio_manager, risk_manager, digital_brain, conversational_ai
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger("TradingSystem")
    logger.info("Starting Advanced AI Trading System...")
    
    # Initialize database
    await init_database()
    
    # Initialize core components
    digital_brain = DigitalBrain()
    await digital_brain.initialize()
    
    conversational_ai = ConversationalInterface(digital_brain)
    
    exchange_manager = ExchangeManager()
    await exchange_manager.initialize()
    
    strategy_manager = StrategyManager()
    portfolio_manager = PortfolioManager(exchange_manager)
    risk_manager = RiskManager()
    
    # Initialize MCP server for multi-agent coordination
    mcp_server = MCPServer(
        digital_brain=digital_brain,
        exchange_manager=exchange_manager,
        strategy_manager=strategy_manager,
        portfolio_manager=portfolio_manager,
        risk_manager=risk_manager
    )
    
    # Start MCP server
    await mcp_server.start()
    
    logger.info("All systems initialized successfully")
    
    yield
    
    # Cleanup
    logger.info("Shutting down trading system...")
    if mcp_server:
        await mcp_server.stop()
    if exchange_manager:
        await exchange_manager.shutdown()
    
    logger.info("Trading system shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Advanced AI Trading System",
    description="Multi-Exchange AI Trading Platform with Digital Brain Integration",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="dashboard/static"), name="static")

# Templates
templates = Jinja2Templates(directory="dashboard/templates")

# WebSocket connections manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

connection_manager = ConnectionManager()

# API Routes
@app.get("/", response_class=HTMLResponse)
async def dashboard(request):
    """Main dashboard"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "components": {
            "mcp_server": mcp_server.is_running() if mcp_server else False,
            "exchange_manager": exchange_manager.is_connected() if exchange_manager else False,
            "digital_brain": digital_brain.is_ready() if digital_brain else False
        }
    }

@app.get("/api/exchanges")
async def get_exchanges():
    """Get available exchanges"""
    if not exchange_manager:
        raise HTTPException(status_code=503, detail="Exchange manager not initialized")
    
    return {
        "exchanges": exchange_manager.get_available_exchanges(),
        "connected": exchange_manager.get_connected_exchanges()
    }

@app.get("/api/strategies")
async def get_strategies():
    """Get available strategies"""
    if not strategy_manager:
        raise HTTPException(status_code=503, detail="Strategy manager not initialized")
    
    return {
        "strategies": strategy_manager.get_available_strategies(),
        "categories": {
            "swing": strategy_manager.get_strategies_by_category("swing"),
            "scalping": strategy_manager.get_strategies_by_category("scalping"),
            "options": strategy_manager.get_strategies_by_category("options"),
            "intraday": strategy_manager.get_strategies_by_category("intraday")
        }
    }

@app.get("/api/portfolio")
async def get_portfolio():
    """Get portfolio status"""
    if not portfolio_manager:
        raise HTTPException(status_code=503, detail="Portfolio manager not initialized")
    
    return await portfolio_manager.get_portfolio_summary()

@app.get("/api/positions")
async def get_positions():
    """Get current positions"""
    if not portfolio_manager:
        raise HTTPException(status_code=503, detail="Portfolio manager not initialized")
    
    return await portfolio_manager.get_positions()

@app.post("/api/chat")
async def chat_with_ai(request: Dict[str, Any]):
    """Chat with the AI trading assistant"""
    if not conversational_ai:
        raise HTTPException(status_code=503, detail="Conversational AI not initialized")
    
    message = request.get("message", "")
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")
    
    try:
        response = await conversational_ai.process_message(message)
        return {"response": response, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.post("/api/strategies/activate")
async def activate_strategy(request: Dict[str, Any]):
    """Activate a trading strategy"""
    if not strategy_manager:
        raise HTTPException(status_code=503, detail="Strategy manager not initialized")
    
    strategy_name = request.get("strategy_name")
    exchange = request.get("exchange")
    parameters = request.get("parameters", {})
    
    if not strategy_name or not exchange:
        raise HTTPException(status_code=400, detail="Strategy name and exchange are required")
    
    try:
        result = await strategy_manager.activate_strategy(strategy_name, exchange, parameters)
        return {"success": True, "strategy_id": result, "message": "Strategy activated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error activating strategy: {str(e)}")

@app.post("/api/strategies/deactivate")
async def deactivate_strategy(request: Dict[str, Any]):
    """Deactivate a trading strategy"""
    if not strategy_manager:
        raise HTTPException(status_code=503, detail="Strategy manager not initialized")
    
    strategy_id = request.get("strategy_id")
    if not strategy_id:
        raise HTTPException(status_code=400, detail="Strategy ID is required")
    
    try:
        await strategy_manager.deactivate_strategy(strategy_id)
        return {"success": True, "message": "Strategy deactivated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deactivating strategy: {str(e)}")

@app.get("/api/market-data/{symbol}")
async def get_market_data(symbol: str):
    """Get market data for a symbol"""
    if not exchange_manager:
        raise HTTPException(status_code=503, detail="Exchange manager not initialized")
    
    try:
        data = await exchange_manager.get_market_data(symbol)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching market data: {str(e)}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await connection_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle real-time requests
            await connection_manager.send_personal_message(f"Echo: {data}", websocket)
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)

# Run Streamlit dashboard in parallel
def run_streamlit_dashboard():
    """Run the Streamlit dashboard"""
    os.system("streamlit run dashboard/app.py --server.port 5000 --server.address 0.0.0.0")

if __name__ == "__main__":
    # Start the FastAPI server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
