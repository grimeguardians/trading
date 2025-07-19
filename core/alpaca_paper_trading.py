"""
Alpaca Paper Trading Integration
Simple interface for placing trades on Alpaca paper trading account
"""

import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import requests
import json

logger = logging.getLogger(__name__)

class AlpacaPaperTrading:
    """Simple Alpaca paper trading interface"""
    
    def __init__(self):
        # Use Alpaca paper trading endpoints
        self.base_url = "https://paper-api.alpaca.markets"
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.api_secret = os.getenv('ALPACA_SECRET_KEY')
        
        if not self.api_key or not self.api_secret:
            logger.warning("Alpaca API credentials not found. Paper trading disabled.")
            self.enabled = False
        else:
            self.enabled = True
            logger.info("✅ Alpaca paper trading initialized")
        
        self.headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret,
            'Content-Type': 'application/json'
        }
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        if not self.enabled:
            return {"error": "Alpaca credentials not configured"}
        
        try:
            response = requests.get(
                f"{self.base_url}/v2/account",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                account = response.json()
                return {
                    "status": "success",
                    "account_number": account.get('account_number'),
                    "buying_power": float(account.get('buying_power', 0)),
                    "cash": float(account.get('cash', 0)),
                    "portfolio_value": float(account.get('portfolio_value', 0)),
                    "day_trade_count": account.get('day_trade_count', 0),
                    "trading_blocked": account.get('trading_blocked', False),
                    "pattern_day_trader": account.get('pattern_day_trader', False)
                }
            else:
                return {"error": f"API error: {response.status_code} - {response.text}"}
                
        except Exception as e:
            logger.error(f"Account info error: {e}")
            return {"error": f"Failed to get account info: {str(e)}"}
    
    def get_positions(self) -> Dict[str, Any]:
        """Get current positions"""
        if not self.enabled:
            return {"error": "Alpaca credentials not configured"}
        
        try:
            response = requests.get(
                f"{self.base_url}/v2/positions",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                positions = response.json()
                return {
                    "status": "success",
                    "positions": [
                        {
                            "symbol": pos.get('symbol'),
                            "qty": float(pos.get('qty', 0)),
                            "market_value": float(pos.get('market_value', 0)),
                            "cost_basis": float(pos.get('cost_basis', 0)),
                            "unrealized_pl": float(pos.get('unrealized_pl', 0)),
                            "unrealized_plpc": float(pos.get('unrealized_plpc', 0)),
                            "current_price": float(pos.get('current_price', 0)),
                            "side": pos.get('side')
                        } for pos in positions
                    ]
                }
            else:
                return {"error": f"API error: {response.status_code} - {response.text}"}
                
        except Exception as e:
            logger.error(f"Positions error: {e}")
            return {"error": f"Failed to get positions: {str(e)}"}
    
    def place_order(self, symbol: str, qty: float, side: str, order_type: str = "market", 
                   time_in_force: str = "day", limit_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Place a trading order
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            qty: Number of shares
            side: 'buy' or 'sell'
            order_type: 'market', 'limit', 'stop', 'stop_limit'
            time_in_force: 'day', 'gtc', 'ioc', 'fok'
            limit_price: Required for limit orders
        """
        if not self.enabled:
            return {"error": "Alpaca credentials not configured"}
        
        # Validate inputs
        if side not in ['buy', 'sell']:
            return {"error": "Side must be 'buy' or 'sell'"}
        
        if order_type not in ['market', 'limit', 'stop', 'stop_limit']:
            return {"error": "Invalid order type"}
        
        if order_type in ['limit', 'stop_limit'] and not limit_price:
            return {"error": "Limit price required for limit orders"}
        
        try:
            # For most stocks, use whole shares. For fractional, use notional amount
            if qty < 1.0:
                # Use notional (dollar amount) for fractional orders
                order_data = {
                    "symbol": symbol.upper(),
                    "notional": str(abs(qty * 200)),  # Approximate dollar amount (qty * ~$200 AAPL price)
                    "side": side,
                    "type": order_type,
                    "time_in_force": time_in_force
                }
            else:
                # Use quantity for whole shares
                order_data = {
                    "symbol": symbol.upper(),
                    "qty": str(int(abs(qty))),
                    "side": side,
                    "type": order_type,
                    "time_in_force": time_in_force
                }
            
            if limit_price:
                order_data["limit_price"] = str(limit_price)
            
            response = requests.post(
                f"{self.base_url}/v2/orders",
                headers=self.headers,
                json=order_data,
                timeout=10
            )
            
            if response.status_code in [201, 200]:  # Created or OK
                order = response.json()
                order_status = order.get('status', 'unknown')
                
                # Alpaca returns various statuses: new, pending_new, accepted, filled, etc.
                if order_status in ['new', 'pending_new', 'accepted', 'partially_filled', 'filled']:
                    return {
                        "status": "success",
                        "order_id": order.get('id'),
                        "symbol": order.get('symbol'),
                        "qty": int(order.get('qty', 0)),
                        "side": order.get('side'),
                        "order_type": order.get('type', order.get('order_type')),
                        "order_status": order_status,
                        "submitted_at": order.get('submitted_at'),
                        "filled_qty": int(order.get('filled_qty', 0)),
                        "message": f"✅ Order {order_status}: {side.upper()} {qty} shares of {symbol.upper()}",
                        "alpaca_response": order
                    }
                else:
                    return {
                        "error": f"Order status '{order_status}' - may need review",
                        "alpaca_response": order
                    }
            else:
                error_msg = response.text
                try:
                    error_data = response.json()
                    error_msg = error_data.get('message', error_msg)
                except:
                    pass
                return {"error": f"Order failed: {error_msg}"}
                
        except Exception as e:
            logger.error(f"Order placement error: {e}")
            return {"error": f"Failed to place order: {str(e)}"}
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order"""
        if not self.enabled:
            return {"error": "Alpaca credentials not configured"}
        
        try:
            response = requests.delete(
                f"{self.base_url}/v2/orders/{order_id}",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 204:  # No Content (success)
                return {
                    "status": "success",
                    "message": f"Order {order_id} cancelled successfully"
                }
            else:
                return {"error": f"Cancel failed: {response.status_code} - {response.text}"}
                
        except Exception as e:
            logger.error(f"Order cancellation error: {e}")
            return {"error": f"Failed to cancel order: {str(e)}"}
    
    def get_orders(self, status: str = "open") -> Dict[str, Any]:
        """Get orders by status"""
        if not self.enabled:
            return {"error": "Alpaca credentials not configured"}
        
        try:
            params = {"status": status, "limit": 50}
            response = requests.get(
                f"{self.base_url}/v2/orders",
                headers=self.headers,
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                orders = response.json()
                return {
                    "status": "success",
                    "orders": [
                        {
                            "id": order.get('id'),
                            "symbol": order.get('symbol'),
                            "qty": int(order.get('qty', 0)),
                            "side": order.get('side'),
                            "order_type": order.get('order_type'),
                            "status": order.get('status'),
                            "submitted_at": order.get('submitted_at'),
                            "filled_qty": int(order.get('filled_qty', 0)),
                            "limit_price": order.get('limit_price')
                        } for order in orders
                    ]
                }
            else:
                return {"error": f"API error: {response.status_code} - {response.text}"}
                
        except Exception as e:
            logger.error(f"Orders retrieval error: {e}")
            return {"error": f"Failed to get orders: {str(e)}"}

# Global instance
alpaca_trader = AlpacaPaperTrading()