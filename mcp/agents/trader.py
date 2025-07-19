"""
Trader Agent for MCP-based trading system
Handles trade execution, position management, and order routing
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
import logging
from decimal import Decimal

from mcp.agents.base_agent import BaseAgent
from exchanges.exchange_manager import ExchangeManager
from strategies.base_strategy import BaseStrategy
from math_models.risk_metrics import RiskMetrics

class TraderAgent(BaseAgent):
    """Trading execution agent with advanced order management"""
    
    def __init__(self, agent_id: str, mcp_server):
        super().__init__(agent_id, mcp_server, "trader")
        
        # Trading components
        self.exchange_manager = None
        self.risk_metrics = RiskMetrics()
        
        # Active positions and orders
        self.active_positions = {}
        self.pending_orders = {}
        self.order_history = []
        
        # Trading parameters
        self.max_position_size = 10000.0
        self.max_daily_loss = 2000.0
        self.max_positions = 10
        self.risk_per_trade = 0.02  # 2% risk per trade
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Strategy execution
        self.active_strategies = {}
        self.strategy_performance = {}
        
        # Risk management
        self.position_limits = {}
        self.stop_losses = {}
        self.take_profits = {}
        
    async def initialize(self):
        """Initialize trader agent components"""
        try:
            self.logger.info("💼 Initializing Trader Agent")
            
            # Initialize exchange manager
            self.exchange_manager = ExchangeManager()
            await self.exchange_manager.initialize()
            
            # Load saved positions and orders
            await self._load_saved_state()
            
            # Initialize risk management
            await self._initialize_risk_management()
            
            self.logger.info("✅ Trader Agent initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing Trader Agent: {e}")
            raise
            
    async def cleanup(self):
        """Cleanup trader agent resources"""
        try:
            # Close all positions
            await self._close_all_positions()
            
            # Cancel pending orders
            await self._cancel_all_orders()
            
            # Save state
            await self._save_state()
            
            if self.exchange_manager:
                await self.exchange_manager.cleanup()
                
        except Exception as e:
            self.logger.error(f"❌ Error cleaning up Trader Agent: {e}")
            
    def get_message_handlers(self) -> Dict[str, Any]:
        """Get trader specific message handlers"""
        return {
            'execute_trade': self._handle_execute_trade,
            'close_position': self._handle_close_position,
            'update_stop_loss': self._handle_update_stop_loss,
            'get_positions': self._handle_get_positions,
            'get_orders': self._handle_get_orders,
            'risk_check': self._handle_risk_check,
            'strategy_signal': self._handle_strategy_signal
        }
        
    async def get_background_tasks(self) -> List[asyncio.Task]:
        """Get background tasks for trader agent"""
        return [
            asyncio.create_task(self._position_monitor()),
            asyncio.create_task(self._order_monitor()),
            asyncio.create_task(self._risk_monitor()),
            asyncio.create_task(self._performance_tracker())
        ]
        
    async def _position_monitor(self):
        """Monitor active positions"""
        while self.running:
            try:
                for position_id, position in self.active_positions.items():
                    # Check stop losses
                    await self._check_stop_loss(position_id, position)
                    
                    # Check take profits
                    await self._check_take_profit(position_id, position)
                    
                    # Update position PnL
                    await self._update_position_pnl(position_id, position)
                    
                    # Check position risk
                    await self._check_position_risk(position_id, position)
                    
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"❌ Error in position monitor: {e}")
                await asyncio.sleep(10)
                
    async def _order_monitor(self):
        """Monitor pending orders"""
        while self.running:
            try:
                for order_id, order in list(self.pending_orders.items()):
                    # Check order status
                    status = await self._check_order_status(order_id, order)
                    
                    if status == 'filled':
                        await self._handle_order_filled(order_id, order)
                    elif status == 'cancelled':
                        await self._handle_order_cancelled(order_id, order)
                    elif status == 'expired':
                        await self._handle_order_expired(order_id, order)
                        
                await asyncio.sleep(2)  # Check every 2 seconds
                
            except Exception as e:
                self.logger.error(f"❌ Error in order monitor: {e}")
                await asyncio.sleep(10)
                
    async def _risk_monitor(self):
        """Monitor risk metrics"""
        while self.running:
            try:
                # Check daily loss limit
                if self.daily_pnl < -self.max_daily_loss:
                    await self._trigger_emergency_stop("Daily loss limit exceeded")
                    
                # Check position limits
                if len(self.active_positions) >= self.max_positions:
                    await self._send_risk_alert("Maximum positions reached")
                    
                # Check individual position risk
                for position_id, position in self.active_positions.items():
                    risk_metrics = await self._calculate_position_risk(position)
                    if risk_metrics['risk_level'] > 0.8:  # High risk
                        await self._send_risk_alert(f"High risk position: {position_id}")
                        
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"❌ Error in risk monitor: {e}")
                await asyncio.sleep(60)
                
    async def _performance_tracker(self):
        """Track performance metrics"""
        while self.running:
            try:
                # Calculate performance metrics
                total_trades = self.winning_trades + self.losing_trades
                win_rate = self.winning_trades / total_trades if total_trades > 0 else 0
                
                performance_data = {
                    'daily_pnl': self.daily_pnl,
                    'total_trades': total_trades,
                    'win_rate': win_rate,
                    'active_positions': len(self.active_positions),
                    'pending_orders': len(self.pending_orders)
                }
                
                # Broadcast performance update
                await self.send_message(
                    recipient='broadcast',
                    payload={
                        'notification_type': 'performance_update',
                        'agent_id': self.agent_id,
                        'performance': performance_data
                    },
                    message_type='notification'
                )
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                self.logger.error(f"❌ Error in performance tracker: {e}")
                await asyncio.sleep(60)
                
    async def execute_trade(self, trade_params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trade with risk management"""
        try:
            # Validate trade parameters
            if not self._validate_trade_params(trade_params):
                return {'success': False, 'error': 'Invalid trade parameters'}
                
            # Pre-trade risk check
            risk_check = await self._pre_trade_risk_check(trade_params)
            if not risk_check['approved']:
                return {'success': False, 'error': risk_check['reason']}
                
            # Calculate position size
            position_size = await self._calculate_position_size(trade_params)
            trade_params['quantity'] = position_size
            
            # Execute trade on exchange
            exchange_name = trade_params.get('exchange', 'alpaca')
            result = await self.exchange_manager.execute_trade(
                exchange=exchange_name,
                symbol=trade_params['symbol'],
                side=trade_params['side'],
                quantity=position_size,
                order_type=trade_params.get('order_type', 'market'),
                price=trade_params.get('price'),
                stop_loss=trade_params.get('stop_loss'),
                take_profit=trade_params.get('take_profit')
            )
            
            if result['success']:
                # Create position record
                position_id = result['order_id']
                position_data = {
                    'id': position_id,
                    'symbol': trade_params['symbol'],
                    'side': trade_params['side'],
                    'quantity': position_size,
                    'entry_price': result['fill_price'],
                    'exchange': exchange_name,
                    'timestamp': datetime.now(),
                    'strategy': trade_params.get('strategy', 'manual'),
                    'stop_loss': trade_params.get('stop_loss'),
                    'take_profit': trade_params.get('take_profit'),
                    'current_pnl': 0.0,
                    'unrealized_pnl': 0.0
                }
                
                self.active_positions[position_id] = position_data
                self.total_trades += 1
                
                # Log successful trade
                self.logger.info(f"✅ Trade executed: {trade_params['symbol']} {trade_params['side']} {position_size}")
                
                return {
                    'success': True,
                    'position_id': position_id,
                    'fill_price': result['fill_price'],
                    'quantity': position_size
                }
            else:
                return {'success': False, 'error': result['error']}
                
        except Exception as e:
            self.logger.error(f"❌ Error executing trade: {e}")
            return {'success': False, 'error': str(e)}
            
    async def close_position(self, position_id: str, reason: str = 'manual') -> Dict[str, Any]:
        """Close a position"""
        try:
            if position_id not in self.active_positions:
                return {'success': False, 'error': 'Position not found'}
                
            position = self.active_positions[position_id]
            
            # Determine opposite side
            close_side = 'sell' if position['side'] == 'buy' else 'buy'
            
            # Execute closing trade
            result = await self.exchange_manager.execute_trade(
                exchange=position['exchange'],
                symbol=position['symbol'],
                side=close_side,
                quantity=position['quantity'],
                order_type='market'
            )
            
            if result['success']:
                # Calculate final PnL
                exit_price = result['fill_price']
                entry_price = position['entry_price']
                
                if position['side'] == 'buy':
                    pnl = (exit_price - entry_price) * position['quantity']
                else:
                    pnl = (entry_price - exit_price) * position['quantity']
                    
                # Update performance metrics
                self.daily_pnl += pnl
                if pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
                    
                # Move to history
                position['exit_price'] = exit_price
                position['exit_timestamp'] = datetime.now()
                position['final_pnl'] = pnl
                position['close_reason'] = reason
                
                self.order_history.append(position)
                del self.active_positions[position_id]
                
                self.logger.info(f"✅ Position closed: {position_id} PnL: ${pnl:.2f}")
                
                return {
                    'success': True,
                    'position_id': position_id,
                    'exit_price': exit_price,
                    'pnl': pnl
                }
            else:
                return {'success': False, 'error': result['error']}
                
        except Exception as e:
            self.logger.error(f"❌ Error closing position: {e}")
            return {'success': False, 'error': str(e)}
            
    async def _handle_execute_trade(self, message):
        """Handle trade execution request"""
        try:
            trade_params = message.payload.get('trade_params', {})
            
            # Add strategy context if available
            if 'strategy_context' in message.payload:
                trade_params['strategy'] = message.payload['strategy_context']
                
            result = await self.execute_trade(trade_params)
            
            await self.send_message(
                recipient=message.sender,
                payload={
                    'response_type': 'trade_execution',
                    'result': result
                },
                message_type='response'
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error handling trade execution: {e}")
            
    async def _handle_close_position(self, message):
        """Handle position closing request"""
        try:
            position_id = message.payload.get('position_id')
            reason = message.payload.get('reason', 'manual')
            
            if not position_id:
                await self.send_message(
                    recipient=message.sender,
                    payload={
                        'response_type': 'error',
                        'error': 'Position ID required'
                    },
                    message_type='response'
                )
                return
                
            result = await self.close_position(position_id, reason)
            
            await self.send_message(
                recipient=message.sender,
                payload={
                    'response_type': 'position_closed',
                    'result': result
                },
                message_type='response'
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error handling position close: {e}")
            
    async def _handle_update_stop_loss(self, message):
        """Handle stop loss update request"""
        try:
            position_id = message.payload.get('position_id')
            new_stop_loss = message.payload.get('stop_loss')
            
            if position_id in self.active_positions:
                self.active_positions[position_id]['stop_loss'] = new_stop_loss
                
                # Update stop loss on exchange
                result = await self.exchange_manager.update_stop_loss(
                    exchange=self.active_positions[position_id]['exchange'],
                    position_id=position_id,
                    stop_loss=new_stop_loss
                )
                
                await self.send_message(
                    recipient=message.sender,
                    payload={
                        'response_type': 'stop_loss_updated',
                        'result': result
                    },
                    message_type='response'
                )
            else:
                await self.send_message(
                    recipient=message.sender,
                    payload={
                        'response_type': 'error',
                        'error': 'Position not found'
                    },
                    message_type='response'
                )
                
        except Exception as e:
            self.logger.error(f"❌ Error updating stop loss: {e}")
            
    async def _handle_get_positions(self, message):
        """Handle get positions request"""
        try:
            positions_data = []
            
            for position_id, position in self.active_positions.items():
                # Update current PnL
                current_pnl = await self._calculate_current_pnl(position)
                position_data = {
                    **position,
                    'current_pnl': current_pnl,
                    'unrealized_pnl': current_pnl
                }
                positions_data.append(position_data)
                
            await self.send_message(
                recipient=message.sender,
                payload={
                    'response_type': 'positions',
                    'positions': positions_data
                },
                message_type='response'
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error getting positions: {e}")
            
    async def _handle_get_orders(self, message):
        """Handle get orders request"""
        try:
            await self.send_message(
                recipient=message.sender,
                payload={
                    'response_type': 'orders',
                    'pending_orders': list(self.pending_orders.values()),
                    'order_history': self.order_history[-50:]  # Last 50 orders
                },
                message_type='response'
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error getting orders: {e}")
            
    async def _handle_risk_check(self, message):
        """Handle risk check request"""
        try:
            risk_params = message.payload.get('risk_params', {})
            
            # Perform comprehensive risk check
            risk_assessment = await self._comprehensive_risk_check(risk_params)
            
            await self.send_message(
                recipient=message.sender,
                payload={
                    'response_type': 'risk_assessment',
                    'assessment': risk_assessment
                },
                message_type='response'
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error in risk check: {e}")
            
    async def _handle_strategy_signal(self, message):
        """Handle strategy signal"""
        try:
            signal_data = message.payload
            
            # Process strategy signal
            if signal_data.get('action') == 'buy' or signal_data.get('action') == 'sell':
                # Convert signal to trade parameters
                trade_params = {
                    'symbol': signal_data['symbol'],
                    'side': signal_data['action'],
                    'exchange': signal_data.get('exchange', 'alpaca'),
                    'strategy': signal_data.get('strategy', 'unknown'),
                    'confidence': signal_data.get('confidence', 0.5),
                    'stop_loss': signal_data.get('stop_loss'),
                    'take_profit': signal_data.get('take_profit')
                }
                
                # Execute trade if confidence is high enough
                if signal_data.get('confidence', 0) > 0.7:
                    result = await self.execute_trade(trade_params)
                    
                    # Send result back to strategy
                    await self.send_message(
                        recipient=message.sender,
                        payload={
                            'response_type': 'signal_processed',
                            'trade_result': result
                        },
                        message_type='response'
                    )
                    
        except Exception as e:
            self.logger.error(f"❌ Error handling strategy signal: {e}")
            
    def _validate_trade_params(self, trade_params: Dict[str, Any]) -> bool:
        """Validate trade parameters"""
        required_fields = ['symbol', 'side', 'exchange']
        
        for field in required_fields:
            if field not in trade_params:
                return False
                
        # Validate side
        if trade_params['side'] not in ['buy', 'sell']:
            return False
            
        # Validate exchange
        if trade_params['exchange'] not in ['alpaca', 'binance', 'td_ameritrade', 'kucoin']:
            return False
            
        return True
        
    async def _pre_trade_risk_check(self, trade_params: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-trade risk assessment"""
        try:
            # Check daily loss limit
            if self.daily_pnl < -self.max_daily_loss:
                return {
                    'approved': False,
                    'reason': 'Daily loss limit exceeded'
                }
                
            # Check position limits
            if len(self.active_positions) >= self.max_positions:
                return {
                    'approved': False,
                    'reason': 'Maximum positions reached'
                }
                
            # Check symbol exposure
            symbol = trade_params['symbol']
            symbol_exposure = sum(
                pos['quantity'] for pos in self.active_positions.values()
                if pos['symbol'] == symbol and pos['side'] == trade_params['side']
            )
            
            if symbol_exposure > self.max_position_size:
                return {
                    'approved': False,
                    'reason': f'Maximum exposure for {symbol} exceeded'
                }
                
            return {
                'approved': True,
                'reason': 'Risk check passed'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error in pre-trade risk check: {e}")
            return {
                'approved': False,
                'reason': 'Risk check failed'
            }
            
    async def _calculate_position_size(self, trade_params: Dict[str, Any]) -> float:
        """Calculate appropriate position size"""
        try:
            # Get current price
            symbol = trade_params['symbol']
            exchange = trade_params['exchange']
            
            current_price = await self.exchange_manager.get_current_price(exchange, symbol)
            
            # Calculate position size based on risk
            if trade_params.get('stop_loss'):
                price_diff = abs(current_price - trade_params['stop_loss'])
                risk_amount = self.max_position_size * self.risk_per_trade
                position_size = risk_amount / price_diff
            else:
                # Default position size
                position_size = 100  # Default to 100 shares/units
                
            # Ensure position size is within limits
            max_size = self.max_position_size / current_price
            position_size = min(position_size, max_size)
            
            return round(position_size, 2)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating position size: {e}")
            return 100  # Default size
            
    async def _check_stop_loss(self, position_id: str, position: Dict[str, Any]):
        """Check stop loss conditions"""
        try:
            if not position.get('stop_loss'):
                return
                
            current_price = await self.exchange_manager.get_current_price(
                position['exchange'], position['symbol']
            )
            
            should_close = False
            
            if position['side'] == 'buy' and current_price <= position['stop_loss']:
                should_close = True
            elif position['side'] == 'sell' and current_price >= position['stop_loss']:
                should_close = True
                
            if should_close:
                await self.close_position(position_id, 'stop_loss')
                
        except Exception as e:
            self.logger.error(f"❌ Error checking stop loss: {e}")
            
    async def _check_take_profit(self, position_id: str, position: Dict[str, Any]):
        """Check take profit conditions"""
        try:
            if not position.get('take_profit'):
                return
                
            current_price = await self.exchange_manager.get_current_price(
                position['exchange'], position['symbol']
            )
            
            should_close = False
            
            if position['side'] == 'buy' and current_price >= position['take_profit']:
                should_close = True
            elif position['side'] == 'sell' and current_price <= position['take_profit']:
                should_close = True
                
            if should_close:
                await self.close_position(position_id, 'take_profit')
                
        except Exception as e:
            self.logger.error(f"❌ Error checking take profit: {e}")
            
    async def _calculate_current_pnl(self, position: Dict[str, Any]) -> float:
        """Calculate current PnL for a position"""
        try:
            current_price = await self.exchange_manager.get_current_price(
                position['exchange'], position['symbol']
            )
            
            entry_price = position['entry_price']
            quantity = position['quantity']
            
            if position['side'] == 'buy':
                pnl = (current_price - entry_price) * quantity
            else:
                pnl = (entry_price - current_price) * quantity
                
            return round(pnl, 2)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating PnL: {e}")
            return 0.0
            
    async def _trigger_emergency_stop(self, reason: str):
        """Trigger emergency stop - close all positions"""
        try:
            self.logger.warning(f"⚠️ Emergency stop triggered: {reason}")
            
            # Close all positions
            for position_id in list(self.active_positions.keys()):
                await self.close_position(position_id, f'emergency_stop: {reason}')
                
            # Cancel all orders
            await self._cancel_all_orders()
            
            # Send emergency alert
            await self.send_message(
                recipient='broadcast',
                payload={
                    'notification_type': 'emergency_stop',
                    'reason': reason,
                    'agent_id': self.agent_id
                },
                message_type='notification',
                priority='critical'
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error in emergency stop: {e}")
            
    async def _send_risk_alert(self, message: str):
        """Send risk alert"""
        await self.send_message(
            recipient='broadcast',
            payload={
                'notification_type': 'risk_alert',
                'message': message,
                'agent_id': self.agent_id
            },
            message_type='notification',
            priority='high'
        )
        
    async def _load_saved_state(self):
        """Load saved positions and orders"""
        # In production, this would load from database
        pass
        
    async def _save_state(self):
        """Save current state"""
        # In production, this would save to database
        pass
        
    async def _initialize_risk_management(self):
        """Initialize risk management parameters"""
        # Set up position limits, risk parameters, etc.
        pass
        
    async def _close_all_positions(self):
        """Close all active positions"""
        for position_id in list(self.active_positions.keys()):
            await self.close_position(position_id, 'shutdown')
            
    async def _cancel_all_orders(self):
        """Cancel all pending orders"""
        for order_id in list(self.pending_orders.keys()):
            try:
                await self.exchange_manager.cancel_order(
                    self.pending_orders[order_id]['exchange'],
                    order_id
                )
                del self.pending_orders[order_id]
            except Exception as e:
                self.logger.error(f"❌ Error cancelling order {order_id}: {e}")
                
    async def _check_order_status(self, order_id: str, order: Dict[str, Any]) -> str:
        """Check order status on exchange"""
        try:
            status = await self.exchange_manager.get_order_status(
                order['exchange'], order_id
            )
            return status
        except Exception as e:
            self.logger.error(f"❌ Error checking order status: {e}")
            return 'unknown'
            
    async def _handle_order_filled(self, order_id: str, order: Dict[str, Any]):
        """Handle filled order"""
        # Move order to history and update positions
        self.order_history.append(order)
        if order_id in self.pending_orders:
            del self.pending_orders[order_id]
            
    async def _handle_order_cancelled(self, order_id: str, order: Dict[str, Any]):
        """Handle cancelled order"""
        self.order_history.append(order)
        if order_id in self.pending_orders:
            del self.pending_orders[order_id]
            
    async def _handle_order_expired(self, order_id: str, order: Dict[str, Any]):
        """Handle expired order"""
        self.order_history.append(order)
        if order_id in self.pending_orders:
            del self.pending_orders[order_id]
            
    async def _update_position_pnl(self, position_id: str, position: Dict[str, Any]):
        """Update position PnL"""
        current_pnl = await self._calculate_current_pnl(position)
        position['current_pnl'] = current_pnl
        position['unrealized_pnl'] = current_pnl
        
    async def _check_position_risk(self, position_id: str, position: Dict[str, Any]):
        """Check individual position risk"""
        # Implement position-specific risk checks
        pass
        
    async def _calculate_position_risk(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate position risk metrics"""
        # Implement position risk calculation
        return {'risk_level': 0.3}
        
    async def _comprehensive_risk_check(self, risk_params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive risk assessment"""
        # Implement comprehensive risk assessment
        return {
            'overall_risk': 0.4,
            'position_risk': 0.3,
            'portfolio_risk': 0.5,
            'recommendations': []
        }
