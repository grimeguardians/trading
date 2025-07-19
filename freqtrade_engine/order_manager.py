"""
Order Management System for Multi-Exchange Trading
Handles order placement, tracking, and execution across exchanges
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
import pandas as pd

from config import Config
from freqtrade_engine.exchange_manager import ExchangeManager
from utils.logger import get_logger

logger = get_logger(__name__)

class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "buy"
    SELL = "sell"

@dataclass
class Order:
    """Order data structure"""
    order_id: str
    exchange: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float]
    stop_price: Optional[float]
    status: OrderStatus
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    average_fill_price: Optional[float] = None
    fees: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    strategy: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_order_id: Optional[str] = None
    child_orders: List[str] = field(default_factory=list)

@dataclass
class OrderExecution:
    """Order execution record"""
    execution_id: str
    order_id: str
    exchange: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    fees: float
    timestamp: datetime
    trade_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class OrderManager:
    """
    Advanced order management system with multi-exchange support
    """
    
    def __init__(self, config: Config, exchange_manager: ExchangeManager):
        self.config = config
        self.exchange_manager = exchange_manager
        
        # Order tracking
        self.active_orders: Dict[str, Order] = {}
        self.completed_orders: Dict[str, Order] = {}
        self.order_executions: Dict[str, List[OrderExecution]] = {}
        
        # Order queues
        self.pending_orders: asyncio.Queue = asyncio.Queue()
        self.cancellation_queue: asyncio.Queue = asyncio.Queue()
        
        # Order processing
        self.order_processor_task: Optional[asyncio.Task] = None
        self.status_monitor_task: Optional[asyncio.Task] = None
        self.processing_active = False
        
        # Order limits and controls
        self.max_orders_per_symbol = 10
        self.max_orders_per_exchange = 100
        self.order_timeout = timedelta(minutes=30)
        
        # Performance tracking
        self.order_stats = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'canceled_orders': 0,
            'average_fill_time': 0.0,
            'total_fees': 0.0
        }
        
        logger.info("OrderManager initialized")
    
    async def initialize(self):
        """Initialize order manager"""
        try:
            logger.info("Initializing OrderManager...")
            
            # Start order processing
            self.processing_active = True
            self.order_processor_task = asyncio.create_task(self._process_orders())
            self.status_monitor_task = asyncio.create_task(self._monitor_order_status())
            
            # Load existing orders from database
            await self._load_existing_orders()
            
            logger.info("✅ OrderManager initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize OrderManager: {e}")
            raise
    
    async def place_order(self, exchange: str, symbol: str, action: str, 
                         quantity: float, price: Optional[float] = None,
                         order_type: str = "market", strategy: Optional[str] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Place a new order"""
        try:
            # Validate inputs
            if not await self._validate_order_request(exchange, symbol, action, quantity, price):
                return {
                    'success': False,
                    'error': 'Order validation failed'
                }
            
            # Check order limits
            if not await self._check_order_limits(exchange, symbol):
                return {
                    'success': False,
                    'error': 'Order limits exceeded'
                }
            
            # Create order
            order_id = str(uuid.uuid4())
            order = Order(
                order_id=order_id,
                exchange=exchange,
                symbol=symbol,
                side=OrderSide.BUY if action.lower() == 'buy' else OrderSide.SELL,
                order_type=OrderType(order_type.lower()),
                quantity=quantity,
                price=price,
                status=OrderStatus.PENDING,
                remaining_quantity=quantity,
                strategy=strategy,
                metadata=metadata or {}
            )
            
            # Add to pending orders
            await self.pending_orders.put(order)
            
            # Store in active orders
            self.active_orders[order_id] = order
            
            logger.info(f"Order placed: {order_id} - {action} {quantity} {symbol} on {exchange}")
            
            return {
                'success': True,
                'order_id': order_id,
                'message': 'Order submitted successfully'
            }
            
        except Exception as e:
            logger.error(f"❌ Error placing order: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order"""
        try:
            if order_id not in self.active_orders:
                return {
                    'success': False,
                    'error': 'Order not found'
                }
            
            order = self.active_orders[order_id]
            
            # Add to cancellation queue
            await self.cancellation_queue.put(order_id)
            
            logger.info(f"Order cancellation requested: {order_id}")
            
            return {
                'success': True,
                'message': 'Order cancellation requested'
            }
            
        except Exception as e:
            logger.error(f"❌ Error canceling order: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def modify_order(self, order_id: str, new_quantity: Optional[float] = None,
                          new_price: Optional[float] = None) -> Dict[str, Any]:
        """Modify an existing order"""
        try:
            if order_id not in self.active_orders:
                return {
                    'success': False,
                    'error': 'Order not found'
                }
            
            order = self.active_orders[order_id]
            
            # For simplicity, we'll cancel and replace
            # In a real implementation, you'd use exchange-specific modify APIs
            
            # Cancel existing order
            await self.cancel_order(order_id)
            
            # Place new order with modified parameters
            new_order_result = await self.place_order(
                exchange=order.exchange,
                symbol=order.symbol,
                action=order.side.value,
                quantity=new_quantity or order.quantity,
                price=new_price or order.price,
                order_type=order.order_type.value,
                strategy=order.strategy,
                metadata=order.metadata
            )
            
            return new_order_result
            
        except Exception as e:
            logger.error(f"❌ Error modifying order: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _process_orders(self):
        """Process pending orders"""
        while self.processing_active:
            try:
                # Process new orders
                try:
                    order = await asyncio.wait_for(self.pending_orders.get(), timeout=1.0)
                    await self._submit_order_to_exchange(order)
                except asyncio.TimeoutError:
                    pass
                
                # Process cancellations
                try:
                    order_id = await asyncio.wait_for(self.cancellation_queue.get(), timeout=0.1)
                    await self._cancel_order_on_exchange(order_id)
                except asyncio.TimeoutError:
                    pass
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ Error in order processing: {e}")
                await asyncio.sleep(1)
    
    async def _submit_order_to_exchange(self, order: Order):
        """Submit order to exchange"""
        try:
            order.status = OrderStatus.SUBMITTED
            order.updated_at = datetime.utcnow()
            
            # Submit to exchange
            result = await self.exchange_manager.place_order(
                exchange_name=order.exchange,
                symbol=order.symbol,
                order_type=order.order_type.value,
                side=order.side.value,
                quantity=order.quantity,
                price=order.price
            )
            
            if result['success']:
                # Update order with exchange order ID
                order.metadata['exchange_order_id'] = result['order_id']
                
                # Update order status based on exchange response
                exchange_status = result.get('status', 'unknown')
                if exchange_status == 'filled':
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = order.quantity
                    order.remaining_quantity = 0.0
                    order.average_fill_price = result.get('price')
                    
                    # Move to completed orders
                    self._complete_order(order)
                elif exchange_status == 'partially_filled':
                    order.status = OrderStatus.PARTIALLY_FILLED
                    order.filled_quantity = result.get('filled_quantity', 0)
                    order.remaining_quantity = order.quantity - order.filled_quantity
                
                self.order_stats['successful_orders'] += 1
                logger.info(f"✅ Order submitted successfully: {order.order_id}")
                
            else:
                order.status = OrderStatus.REJECTED
                order.metadata['rejection_reason'] = result.get('error', 'Unknown error')
                
                self.order_stats['failed_orders'] += 1
                logger.error(f"❌ Order rejected: {order.order_id} - {result.get('error')}")
                
                # Move to completed orders
                self._complete_order(order)
            
            self.order_stats['total_orders'] += 1
            
        except Exception as e:
            logger.error(f"❌ Error submitting order to exchange: {e}")
            order.status = OrderStatus.REJECTED
            order.metadata['error'] = str(e)
            self._complete_order(order)
    
    async def _cancel_order_on_exchange(self, order_id: str):
        """Cancel order on exchange"""
        try:
            if order_id not in self.active_orders:
                return
            
            order = self.active_orders[order_id]
            exchange_order_id = order.metadata.get('exchange_order_id')
            
            if exchange_order_id:
                # Cancel on exchange
                result = await self.exchange_manager.cancel_order(
                    exchange_name=order.exchange,
                    order_id=exchange_order_id,
                    symbol=order.symbol
                )
                
                if result['success']:
                    order.status = OrderStatus.CANCELED
                    order.updated_at = datetime.utcnow()
                    
                    self.order_stats['canceled_orders'] += 1
                    logger.info(f"✅ Order canceled successfully: {order_id}")
                    
                    # Move to completed orders
                    self._complete_order(order)
                else:
                    logger.error(f"❌ Failed to cancel order: {order_id} - {result.get('error')}")
            else:
                # Order not yet submitted to exchange
                order.status = OrderStatus.CANCELED
                order.updated_at = datetime.utcnow()
                self._complete_order(order)
            
        except Exception as e:
            logger.error(f"❌ Error canceling order on exchange: {e}")
    
    async def _monitor_order_status(self):
        """Monitor order status updates"""
        while self.processing_active:
            try:
                for order_id, order in list(self.active_orders.items()):
                    # Skip orders that haven't been submitted yet
                    if order.status == OrderStatus.PENDING:
                        continue
                    
                    # Check for timeout
                    if datetime.utcnow() - order.created_at > self.order_timeout:
                        logger.warning(f"Order {order_id} timed out")
                        await self.cancel_order(order_id)
                        continue
                    
                    # Get status from exchange
                    exchange_order_id = order.metadata.get('exchange_order_id')
                    if exchange_order_id:
                        status_result = await self.exchange_manager.get_order_status(
                            exchange_name=order.exchange,
                            order_id=exchange_order_id,
                            symbol=order.symbol
                        )
                        
                        if status_result['success']:
                            await self._update_order_status(order, status_result)
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"❌ Error monitoring order status: {e}")
                await asyncio.sleep(10)
    
    async def _update_order_status(self, order: Order, status_result: Dict[str, Any]):
        """Update order status from exchange response"""
        try:
            exchange_status = status_result.get('status', '').lower()
            
            if exchange_status == 'filled':
                order.status = OrderStatus.FILLED
                order.filled_quantity = status_result.get('filled_quantity', order.quantity)
                order.remaining_quantity = 0.0
                order.average_fill_price = status_result.get('price')
                order.updated_at = datetime.utcnow()
                
                # Create execution record
                execution = OrderExecution(
                    execution_id=str(uuid.uuid4()),
                    order_id=order.order_id,
                    exchange=order.exchange,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.filled_quantity,
                    price=order.average_fill_price,
                    fees=0.0,  # Would be calculated from exchange data
                    timestamp=datetime.utcnow(),
                    trade_id=status_result.get('trade_id', ''),
                    metadata=status_result
                )
                
                if order.order_id not in self.order_executions:
                    self.order_executions[order.order_id] = []
                self.order_executions[order.order_id].append(execution)
                
                # Move to completed orders
                self._complete_order(order)
                
            elif exchange_status == 'partially_filled':
                order.status = OrderStatus.PARTIALLY_FILLED
                order.filled_quantity = status_result.get('filled_quantity', 0)
                order.remaining_quantity = order.quantity - order.filled_quantity
                order.updated_at = datetime.utcnow()
                
            elif exchange_status == 'canceled':
                order.status = OrderStatus.CANCELED
                order.updated_at = datetime.utcnow()
                self._complete_order(order)
                
        except Exception as e:
            logger.error(f"❌ Error updating order status: {e}")
    
    def _complete_order(self, order: Order):
        """Move order to completed orders"""
        if order.order_id in self.active_orders:
            del self.active_orders[order.order_id]
            self.completed_orders[order.order_id] = order
            
            # Update statistics
            if order.status == OrderStatus.FILLED:
                fill_time = (order.updated_at - order.created_at).total_seconds()
                self._update_fill_time_stats(fill_time)
    
    def _update_fill_time_stats(self, fill_time: float):
        """Update fill time statistics"""
        current_avg = self.order_stats['average_fill_time']
        successful_orders = self.order_stats['successful_orders']
        
        if successful_orders > 0:
            self.order_stats['average_fill_time'] = (
                (current_avg * (successful_orders - 1) + fill_time) / successful_orders
            )
        else:
            self.order_stats['average_fill_time'] = fill_time
    
    async def _validate_order_request(self, exchange: str, symbol: str, action: str,
                                    quantity: float, price: Optional[float]) -> bool:
        """Validate order request"""
        try:
            # Check exchange availability
            if not await self.exchange_manager.is_exchange_available(exchange):
                logger.error(f"Exchange {exchange} not available")
                return False
            
            # Check symbol availability
            if not await self.exchange_manager.is_symbol_available(exchange, symbol):
                logger.error(f"Symbol {symbol} not available on {exchange}")
                return False
            
            # Validate action
            if action.lower() not in ['buy', 'sell']:
                logger.error(f"Invalid action: {action}")
                return False
            
            # Validate quantity
            if quantity <= 0:
                logger.error(f"Invalid quantity: {quantity}")
                return False
            
            # Validate price if provided
            if price is not None and price <= 0:
                logger.error(f"Invalid price: {price}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error validating order request: {e}")
            return False
    
    async def _check_order_limits(self, exchange: str, symbol: str) -> bool:
        """Check order limits"""
        try:
            # Count active orders for symbol
            symbol_orders = sum(1 for order in self.active_orders.values() 
                              if order.exchange == exchange and order.symbol == symbol)
            
            if symbol_orders >= self.max_orders_per_symbol:
                logger.error(f"Maximum orders per symbol exceeded: {symbol}")
                return False
            
            # Count active orders for exchange
            exchange_orders = sum(1 for order in self.active_orders.values() 
                                if order.exchange == exchange)
            
            if exchange_orders >= self.max_orders_per_exchange:
                logger.error(f"Maximum orders per exchange exceeded: {exchange}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error checking order limits: {e}")
            return False
    
    async def _load_existing_orders(self):
        """Load existing orders from database"""
        # This would load orders from database
        # For now, we'll just initialize empty
        pass
    
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status"""
        try:
            # Check active orders
            if order_id in self.active_orders:
                order = self.active_orders[order_id]
                return self._order_to_dict(order)
            
            # Check completed orders
            if order_id in self.completed_orders:
                order = self.completed_orders[order_id]
                return self._order_to_dict(order)
            
            return {
                'success': False,
                'error': 'Order not found'
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting order status: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _order_to_dict(self, order: Order) -> Dict[str, Any]:
        """Convert order to dictionary"""
        return {
            'success': True,
            'order_id': order.order_id,
            'exchange': order.exchange,
            'symbol': order.symbol,
            'side': order.side.value,
            'order_type': order.order_type.value,
            'quantity': order.quantity,
            'price': order.price,
            'filled_quantity': order.filled_quantity,
            'remaining_quantity': order.remaining_quantity,
            'average_fill_price': order.average_fill_price,
            'status': order.status.value,
            'created_at': order.created_at.isoformat(),
            'updated_at': order.updated_at.isoformat(),
            'strategy': order.strategy,
            'metadata': order.metadata
        }
    
    async def get_active_orders(self) -> List[Dict[str, Any]]:
        """Get all active orders"""
        return [self._order_to_dict(order) for order in self.active_orders.values()]
    
    async def get_order_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get order history"""
        # Sort by updated_at descending
        sorted_orders = sorted(
            self.completed_orders.values(),
            key=lambda x: x.updated_at,
            reverse=True
        )
        
        return [self._order_to_dict(order) for order in sorted_orders[:limit]]
    
    async def get_order_statistics(self) -> Dict[str, Any]:
        """Get order statistics"""
        return {
            'active_orders': len(self.active_orders),
            'completed_orders': len(self.completed_orders),
            'statistics': self.order_stats,
            'last_updated': datetime.utcnow().isoformat()
        }
    
    async def stop(self):
        """Stop order manager"""
        try:
            logger.info("Stopping OrderManager...")
            
            # Stop processing
            self.processing_active = False
            
            # Cancel background tasks
            if self.order_processor_task:
                self.order_processor_task.cancel()
            
            if self.status_monitor_task:
                self.status_monitor_task.cancel()
            
            # Cancel all active orders
            for order_id in list(self.active_orders.keys()):
                await self.cancel_order(order_id)
            
            logger.info("✅ OrderManager stopped successfully")
            
        except Exception as e:
            logger.error(f"❌ Error stopping OrderManager: {e}")
