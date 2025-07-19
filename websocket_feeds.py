#!/usr/bin/env python3
"""
Enterprise WebSocket Real-Time Data Feeds
Professional-grade real-time market data streaming
"""

import asyncio
import json
import logging
import websockets
from datetime import datetime
from typing import Set, Dict, Any
from enterprise_market_data import enterprise_data

logger = logging.getLogger(__name__)

class RealTimeDataFeed:
    """Enterprise WebSocket server for real-time market data"""
    
    def __init__(self, host="localhost", port=8765):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.subscriptions: Dict[str, Set[websockets.WebSocketServerProtocol]] = {}
        self.running = False
    
    async def register_client(self, websocket):
        """Register a new client connection"""
        self.clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")
    
    async def unregister_client(self, websocket):
        """Unregister a client connection"""
        self.clients.discard(websocket)
        # Remove from all subscriptions
        for symbol, subscribers in self.subscriptions.items():
            subscribers.discard(websocket)
        logger.info(f"Client disconnected. Total clients: {len(self.clients)}")
    
    async def subscribe_symbol(self, websocket, symbol: str):
        """Subscribe client to symbol updates"""
        symbol = symbol.upper()
        if symbol not in self.subscriptions:
            self.subscriptions[symbol] = set()
        
        self.subscriptions[symbol].add(websocket)
        
        # Send immediate data for newly subscribed symbol
        try:
            data = await enterprise_data.get_real_time_quote(symbol)
            if data and "error" not in data:
                message = {
                    "type": "quote_update",
                    "symbol": symbol,
                    "data": data,
                    "timestamp": datetime.now().isoformat()
                }
                await websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending initial data for {symbol}: {e}")
        
        logger.info(f"Client subscribed to {symbol}. Subscribers: {len(self.subscriptions[symbol])}")
    
    async def unsubscribe_symbol(self, websocket, symbol: str):
        """Unsubscribe client from symbol updates"""
        symbol = symbol.upper()
        if symbol in self.subscriptions:
            self.subscriptions[symbol].discard(websocket)
            if not self.subscriptions[symbol]:
                del self.subscriptions[symbol]
        logger.info(f"Client unsubscribed from {symbol}")
    
    async def broadcast_to_subscribers(self, symbol: str, data: Dict[str, Any]):
        """Broadcast data to all subscribers of a symbol"""
        if symbol in self.subscriptions and self.subscriptions[symbol]:
            message = {
                "type": "quote_update",
                "symbol": symbol,
                "data": data,
                "timestamp": datetime.now().isoformat()
            }
            
            # Send to all subscribers
            disconnected = set()
            for websocket in self.subscriptions[symbol]:
                try:
                    await websocket.send(json.dumps(message))
                except websockets.exceptions.ConnectionClosed:
                    disconnected.add(websocket)
                except Exception as e:
                    logger.error(f"Error broadcasting to client: {e}")
                    disconnected.add(websocket)
            
            # Clean up disconnected clients
            for websocket in disconnected:
                await self.unregister_client(websocket)
    
    async def data_update_loop(self):
        """Continuously update market data for subscribed symbols"""
        while self.running:
            try:
                # Update data for all subscribed symbols
                for symbol in list(self.subscriptions.keys()):
                    if self.subscriptions[symbol]:  # Has active subscribers
                        try:
                            # Get fresh market data
                            data = await enterprise_data.get_real_time_quote(symbol)
                            if data and "error" not in data:
                                await self.broadcast_to_subscribers(symbol, data)
                        except Exception as e:
                            logger.error(f"Error updating data for {symbol}: {e}")
                
                # Wait before next update (adjust frequency as needed)
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in data update loop: {e}")
                await asyncio.sleep(10)  # Wait longer on error
    
    async def handle_client_message(self, websocket, message: str):
        """Handle incoming messages from clients"""
        try:
            data = json.loads(message)
            action = data.get("action")
            symbol = data.get("symbol", "").upper()
            
            if action == "subscribe" and symbol:
                await self.subscribe_symbol(websocket, symbol)
                response = {"type": "subscription_confirmed", "symbol": symbol}
                await websocket.send(json.dumps(response))
            
            elif action == "unsubscribe" and symbol:
                await self.unsubscribe_symbol(websocket, symbol)
                response = {"type": "unsubscription_confirmed", "symbol": symbol}
                await websocket.send(json.dumps(response))
            
            elif action == "get_quote" and symbol:
                # Get immediate quote
                quote_data = await enterprise_data.get_real_time_quote(symbol)
                response = {
                    "type": "quote_response",
                    "symbol": symbol,
                    "data": quote_data,
                    "timestamp": datetime.now().isoformat()
                }
                await websocket.send(json.dumps(response))
            
            elif action == "get_technical" and symbol:
                # Get technical analysis
                technical_data = await enterprise_data.get_technical_indicators(symbol)
                response = {
                    "type": "technical_response",
                    "symbol": symbol,
                    "data": technical_data,
                    "timestamp": datetime.now().isoformat()
                }
                await websocket.send(json.dumps(response))
            
            else:
                error_response = {"type": "error", "message": "Invalid action or missing symbol"}
                await websocket.send(json.dumps(error_response))
                
        except json.JSONDecodeError:
            error_response = {"type": "error", "message": "Invalid JSON format"}
            await websocket.send(json.dumps(error_response))
        except Exception as e:
            logger.error(f"Error handling client message: {e}")
            error_response = {"type": "error", "message": str(e)}
            await websocket.send(json.dumps(error_response))
    
    async def client_handler(self, websocket, path):
        """Handle individual client connections"""
        await self.register_client(websocket)
        
        # Send welcome message
        welcome = {
            "type": "welcome",
            "message": "Connected to Enterprise Market Data Feed",
            "timestamp": datetime.now().isoformat(),
            "instructions": {
                "subscribe": {"action": "subscribe", "symbol": "AAPL"},
                "unsubscribe": {"action": "unsubscribe", "symbol": "AAPL"},
                "get_quote": {"action": "get_quote", "symbol": "AAPL"},
                "get_technical": {"action": "get_technical", "symbol": "AAPL"}
            }
        }
        await websocket.send(json.dumps(welcome))
        
        try:
            async for message in websocket:
                await self.handle_client_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            logger.error(f"Error in client handler: {e}")
        finally:
            await self.unregister_client(websocket)
    
    async def start_server(self):
        """Start the WebSocket server"""
        self.running = True
        
        # Start the data update loop
        update_task = asyncio.create_task(self.data_update_loop())
        
        # Start the WebSocket server
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        server = await websockets.serve(
            self.client_handler,
            self.host,
            self.port,
            ping_interval=30,
            ping_timeout=10
        )
        
        logger.info("✅ Enterprise WebSocket Real-Time Data Feed started")
        
        try:
            await server.wait_closed()
        except KeyboardInterrupt:
            logger.info("Shutting down WebSocket server...")
        finally:
            self.running = False
            update_task.cancel()
            try:
                await update_task
            except asyncio.CancelledError:
                pass

# Global instance
real_time_feed = RealTimeDataFeed()

if __name__ == "__main__":
    # Run the WebSocket server
    asyncio.run(real_time_feed.start_server())