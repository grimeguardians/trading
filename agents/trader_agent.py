"""
Trader Agent - Executes trading decisions and manages positions
Handles order placement, position management, and trade execution
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import uuid

from agents.base_agent import BaseAgent
from exchanges.alpaca_exchange import AlpacaExchange
from exchanges.binance_exchange import BinanceExchange
from exchanges.td_ameritrade_exchange import TDAmeritradeExchange
from exchanges.kucoin_exchange import KuCoinExchange
from exchanges.base_exchange import OrderType, OrderSide, OrderStatus
from strategies.swing_strategy import SwingStrategy
from strategies.scalping_strategy import ScalpingStrategy
from strategies.options_strategy import OptionsStrategy
from strategies.intraday_strategy import IntradayStrategy
from analysis.risk_management import RiskManager
from mcp_server import MessageType


class TraderAgent(BaseAgent):
    """
    Trader Agent for executing trades and managing positions
    Coordinates with multiple exchanges and implements various strategies
    """
    
    def __init__(self, mcp_server, knowledge_engine, config):
        super().__init__(
            agent_id="trader_agent",
            agent_type="trader",
            mcp_server=mcp_server,
            knowledge_engine=knowledge_engine,
            config=config
        )
        
        # Initialize exchanges
        self.exchanges = {}
        self.active_exchange = None
        self._initialize_exchanges()
        
        # Initialize strategies
        self.strategies = {}
        self.active_strategies = {}
        self._initialize_strategies()
        
        # Initialize risk manager
        self.risk_manager = RiskManager(config)
        
        # Trading state
        self.portfolio = {}
        self.positions = {}
        self.pending_orders = {}
        self.order_history = []
        
        # Trading limits
        self.max_positions = config.STRATEGIES.get("swing", {}).get("max_positions", 5)
        self.max_risk_per_trade = config.RISK_PER_TRADE
        self.max_portfolio_risk = config.MAX_DRAWDOWN
        
        # Performance tracking
        self.trades_executed = 0
        self.successful_trades = 0
        self.total_pnl = 0.0
        self.win_rate = 0.0
        
        # Signal processing
        self.signal_queue = asyncio.Queue()
        self.signal_history = []
        
        self.logger.info("💼 Trader Agent initialized with multi-exchange support")
    
    def _setup_capabilities(self):
        """Setup trader capabilities"""
        self.capabilities = [
            "order_execution",
            "position_management", 
            "risk_management",
            "multi_exchange_trading",
            "strategy_execution",
            "portfolio_management",
            "stop_loss_management",
            "profit_taking"
        ]
    
    def _setup_message_handlers(self):
        """Setup message handlers"""
        self.register_message_handler("trading_signal", self._handle_trading_signal)
        self.register_message_handler("order_request", self._handle_order_request)
        self.register_message_handler("position_update", self._handle_position_update)
        self.register_message_handler("market_data_update", self._handle_market_data_update)
        self.register_message_handler("risk_alert", self._handle_risk_alert)
        self.register_message_handler("strategy_update", self._handle_strategy_update)
    
    def _initialize_exchanges(self):
        """Initialize supported exchanges"""
        try:
            # Initialize Alpaca (Primary)
            if self.config.EXCHANGES["alpaca"].enabled:
                self.exchanges["alpaca"] = AlpacaExchange(self.config, "alpaca")
                self.active_exchange = "alpaca"
                self.logger.info("✅ Alpaca exchange initialized (Primary)")
            
            # Initialize Binance
            if self.config.EXCHANGES["binance"].enabled:
                self.exchanges["binance"] = BinanceExchange(self.config, "binance")
                self.logger.info("✅ Binance exchange initialized")
            
            # Initialize TD Ameritrade
            if self.config.EXCHANGES["td_ameritrade"].enabled:
                self.exchanges["td_ameritrade"] = TDAmeritradeExchange(self.config, "td_ameritrade")
                self.logger.info("✅ TD Ameritrade exchange initialized")
            
            # Initialize KuCoin
            if self.config.EXCHANGES["kucoin"].enabled:
                self.exchanges["kucoin"] = KuCoinExchange(self.config, "kucoin")
                self.logger.info("✅ KuCoin exchange initialized")
            
            if not self.exchanges:
                raise Exception("No exchanges configured")
                
        except Exception as e:
            self.logger.error(f"❌ Exchange initialization failed: {e}")
            raise
    
    def _initialize_strategies(self):
        """Initialize trading strategies"""
        try:
            # Initialize strategies
            self.strategies["swing"] = SwingStrategy(self.config)
            self.strategies["scalping"] = ScalpingStrategy(self.config)
            self.strategies["options"] = OptionsStrategy(self.config)
            self.strategies["intraday"] = IntradayStrategy(self.config)
            
            # Activate enabled strategies
            for strategy_name, strategy_config in self.config.STRATEGIES.items():
                if strategy_config.get("enabled", False):
                    self.active_strategies[strategy_name] = self.strategies[strategy_name]
                    self.logger.info(f"✅ {strategy_name} strategy activated")
            
            if not self.active_strategies:
                self.logger.warning("⚠️ No strategies enabled")
                
        except Exception as e:
            self.logger.error(f"❌ Strategy initialization failed: {e}")
            raise
    
    async def _agent_logic(self):
        """Main trader agent logic"""
        self.logger.info("🚀 Trader Agent started - ready for trading")
        
        # Connect to exchanges
        await self._connect_exchanges()
        
        # Start background tasks
        asyncio.create_task(self._signal_processor())
        asyncio.create_task(self._position_monitor())
        asyncio.create_task(self._risk_monitor())
        asyncio.create_task(self._portfolio_updater())
        
        while self.running:
            try:
                # Process active strategies
                await self._process_strategies()
                
                # Update portfolio metrics
                await self._update_portfolio_metrics()
                
                # Check for strategy signals
                await self._check_strategy_signals()
                
                # Manage existing positions
                await self._manage_positions()
                
                # Wait for next cycle
                await asyncio.sleep(1)  # 1 second cycle
                
            except Exception as e:
                self.logger.error(f"❌ Trader agent error: {e}")
                await asyncio.sleep(5)
    
    async def _connect_exchanges(self):
        """Connect to all configured exchanges"""
        for exchange_name, exchange in self.exchanges.items():
            try:
                await exchange.connect()
                await exchange.authenticate()
                self.logger.info(f"✅ Connected to {exchange_name}")
            except Exception as e:
                self.logger.error(f"❌ Failed to connect to {exchange_name}: {e}")
    
    async def _signal_processor(self):
        """Process incoming trading signals"""
        while self.running:
            try:
                # Wait for signal with timeout
                signal = await asyncio.wait_for(self.signal_queue.get(), timeout=1.0)
                
                # Process the signal
                await self._process_trading_signal(signal)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"❌ Signal processing error: {e}")
                await asyncio.sleep(1)
    
    async def _process_trading_signal(self, signal: Dict):
        """Process a trading signal"""
        try:
            symbol = signal.get("symbol")
            signal_type = signal.get("signal_type")
            confidence = signal.get("confidence", 0.0)
            source = signal.get("source", "unknown")
            
            # Validate signal
            if not symbol or not signal_type:
                self.logger.warning("⚠️ Invalid signal received")
                return
            
            # Check confidence threshold
            if confidence < 0.6:
                self.logger.debug(f"📊 Signal confidence too low: {confidence}")
                return
            
            # Risk check
            if not await self._check_risk_limits(symbol, signal):
                self.logger.warning(f"⚠️ Risk limits exceeded for {symbol}")
                return
            
            # Determine exchange for this signal
            exchange_name = signal.get("exchange", self.active_exchange)
            if exchange_name not in self.exchanges:
                self.logger.error(f"❌ Exchange not available: {exchange_name}")
                return
            
            exchange = self.exchanges[exchange_name]
            
            # Execute trade based on signal
            if signal_type.upper() == "BUY":
                await self._execute_buy_signal(exchange, symbol, signal)
            elif signal_type.upper() == "SELL":
                await self._execute_sell_signal(exchange, symbol, signal)
            
            # Store signal in history
            self.signal_history.append({
                **signal,
                "processed_at": datetime.utcnow().isoformat(),
                "executed": True
            })
            
            # Keep history size manageable
            if len(self.signal_history) > 1000:
                self.signal_history = self.signal_history[-500:]
            
        except Exception as e:
            self.logger.error(f"❌ Error processing signal: {e}")
    
    async def _execute_buy_signal(self, exchange, symbol: str, signal: Dict):
        """Execute buy signal"""
        try:
            # Calculate position size
            position_size = await self._calculate_position_size(symbol, signal)
            
            if position_size <= 0:
                return
            
            # Get current price
            ticker = await exchange.get_ticker(symbol)
            current_price = ticker.get("price", 0)
            
            if current_price <= 0:
                self.logger.error(f"❌ Invalid price for {symbol}: {current_price}")
                return
            
            # Calculate stop loss and take profit
            stop_loss_price = current_price * (1 - self.config.STOP_LOSS_PERCENTAGE)
            take_profit_price = current_price * (1 + self.config.TAKE_PROFIT_PERCENTAGE)
            
            # Place market buy order
            order = await exchange.place_order(
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=position_size
            )
            
            if order:
                self.logger.info(f"✅ Buy order placed: {symbol} x{position_size}")
                
                # Track the order
                self.pending_orders[order["order_id"]] = {
                    "order": order,
                    "signal": signal,
                    "stop_loss_price": stop_loss_price,
                    "take_profit_price": take_profit_price
                }
                
                # Send order update
                await self.broadcast_message({
                    "type": "order_placed",
                    "order": order,
                    "signal": signal
                })
                
                self.trades_executed += 1
            
        except Exception as e:
            self.logger.error(f"❌ Buy execution error for {symbol}: {e}")
    
    async def _execute_sell_signal(self, exchange, symbol: str, signal: Dict):
        """Execute sell signal"""
        try:
            # Check if we have a position to sell
            position = await self._get_position(symbol)
            
            if not position or position.get("quantity", 0) <= 0:
                self.logger.warning(f"⚠️ No position to sell for {symbol}")
                return
            
            # Place market sell order
            order = await exchange.place_order(
                symbol=symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=position["quantity"]
            )
            
            if order:
                self.logger.info(f"✅ Sell order placed: {symbol} x{position['quantity']}")
                
                # Track the order
                self.pending_orders[order["order_id"]] = {
                    "order": order,
                    "signal": signal,
                    "position": position
                }
                
                # Send order update
                await self.broadcast_message({
                    "type": "order_placed",
                    "order": order,
                    "signal": signal
                })
                
                self.trades_executed += 1
            
        except Exception as e:
            self.logger.error(f"❌ Sell execution error for {symbol}: {e}")
    
    async def _calculate_position_size(self, symbol: str, signal: Dict) -> float:
        """Calculate position size based on risk management"""
        try:
            # Get account balance
            balance = await self._get_account_balance()
            available_cash = balance.get("cash", 0)
            
            # Calculate maximum position value
            max_position_value = available_cash * self.config.MAX_POSITION_SIZE
            
            # Calculate position size based on risk
            risk_amount = available_cash * self.max_risk_per_trade
            
            # Get current price
            exchange = self.exchanges[self.active_exchange]
            ticker = await exchange.get_ticker(symbol)
            current_price = ticker.get("price", 0)
            
            if current_price <= 0:
                return 0.0
            
            # Calculate position size based on stop loss
            stop_loss_distance = current_price * self.config.STOP_LOSS_PERCENTAGE
            position_size_by_risk = risk_amount / stop_loss_distance
            
            # Use smaller of the two calculations
            max_shares = max_position_value / current_price
            position_size = min(position_size_by_risk, max_shares)
            
            # Apply confidence multiplier
            confidence = signal.get("confidence", 1.0)
            position_size *= confidence
            
            return max(0.0, position_size)
            
        except Exception as e:
            self.logger.error(f"❌ Position size calculation error: {e}")
            return 0.0
    
    async def _get_account_balance(self) -> Dict:
        """Get account balance from active exchange"""
        try:
            exchange = self.exchanges[self.active_exchange]
            return await exchange.get_balance()
        except Exception as e:
            self.logger.error(f"❌ Account balance error: {e}")
            return {"cash": 0.0, "total": 0.0}
    
    async def _get_position(self, symbol: str) -> Optional[Dict]:
        """Get position for symbol"""
        try:
            exchange = self.exchanges[self.active_exchange]
            positions = await exchange.get_positions()
            
            for position in positions:
                if position.get("symbol") == symbol:
                    return position
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Position lookup error: {e}")
            return None
    
    async def _check_risk_limits(self, symbol: str, signal: Dict) -> bool:
        """Check if trade complies with risk limits"""
        try:
            # Check maximum positions
            current_positions = len(self.positions)
            if current_positions >= self.max_positions:
                return False
            
            # Check portfolio risk
            portfolio_risk = await self._calculate_portfolio_risk()
            if portfolio_risk >= self.max_portfolio_risk:
                return False
            
            # Check symbol-specific risk
            if symbol in self.positions:
                # Already have position in this symbol
                return False
            
            # Check correlation risk
            correlation_risk = await self._check_correlation_risk(symbol)
            if correlation_risk > 0.8:  # High correlation threshold
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Risk check error: {e}")
            return False
    
    async def _calculate_portfolio_risk(self) -> float:
        """Calculate current portfolio risk"""
        try:
            # Simplified portfolio risk calculation
            # In production, this would be more sophisticated
            balance = await self._get_account_balance()
            total_value = balance.get("total", 0)
            
            if total_value <= 0:
                return 0.0
            
            # Calculate value at risk
            position_values = []
            for symbol, position in self.positions.items():
                position_value = position.get("market_value", 0)
                position_values.append(position_value)
            
            if not position_values:
                return 0.0
            
            # Simple risk calculation based on position concentration
            max_position_value = max(position_values)
            portfolio_risk = max_position_value / total_value
            
            return portfolio_risk
            
        except Exception as e:
            self.logger.error(f"❌ Portfolio risk calculation error: {e}")
            return 0.0
    
    async def _check_correlation_risk(self, symbol: str) -> float:
        """Check correlation risk with existing positions"""
        try:
            # Request correlation analysis from market analyst
            await self.send_direct_message("market_analyst", {
                "type": "correlation_analysis_request",
                "symbol": symbol,
                "existing_positions": list(self.positions.keys())
            })
            
            # For now, return low correlation risk
            # In production, wait for response from market analyst
            return 0.0
            
        except Exception as e:
            self.logger.error(f"❌ Correlation risk check error: {e}")
            return 0.0
    
    async def _position_monitor(self):
        """Monitor existing positions"""
        while self.running:
            try:
                # Update positions from exchanges
                await self._update_positions()
                
                # Check stop losses and take profits
                await self._check_stop_losses()
                await self._check_take_profits()
                
                # Update position metrics
                await self._update_position_metrics()
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"❌ Position monitor error: {e}")
                await asyncio.sleep(10)
    
    async def _update_positions(self):
        """Update positions from exchanges"""
        try:
            for exchange_name, exchange in self.exchanges.items():
                positions = await exchange.get_positions()
                
                for position in positions:
                    symbol = position.get("symbol")
                    if symbol:
                        self.positions[symbol] = {
                            **position,
                            "exchange": exchange_name,
                            "last_updated": datetime.utcnow().isoformat()
                        }
                        
        except Exception as e:
            self.logger.error(f"❌ Position update error: {e}")
    
    async def _check_stop_losses(self):
        """Check and execute stop losses"""
        try:
            for symbol, position in self.positions.items():
                if position.get("side") == "long":
                    current_price = position.get("current_price", 0)
                    stop_loss_price = position.get("stop_loss_price", 0)
                    
                    if current_price > 0 and stop_loss_price > 0:
                        if current_price <= stop_loss_price:
                            await self._execute_stop_loss(symbol, position)
                            
        except Exception as e:
            self.logger.error(f"❌ Stop loss check error: {e}")
    
    async def _execute_stop_loss(self, symbol: str, position: Dict):
        """Execute stop loss order"""
        try:
            exchange_name = position.get("exchange", self.active_exchange)
            exchange = self.exchanges[exchange_name]
            
            # Place stop loss order
            order = await exchange.place_order(
                symbol=symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=position.get("quantity", 0)
            )
            
            if order:
                self.logger.warning(f"⚠️ Stop loss executed: {symbol}")
                
                # Send alert
                await self.broadcast_message({
                    "type": "stop_loss_executed",
                    "symbol": symbol,
                    "order": order,
                    "position": position
                })
                
        except Exception as e:
            self.logger.error(f"❌ Stop loss execution error: {e}")
    
    async def _check_take_profits(self):
        """Check and execute take profits"""
        try:
            for symbol, position in self.positions.items():
                if position.get("side") == "long":
                    current_price = position.get("current_price", 0)
                    take_profit_price = position.get("take_profit_price", 0)
                    
                    if current_price > 0 and take_profit_price > 0:
                        if current_price >= take_profit_price:
                            await self._execute_take_profit(symbol, position)
                            
        except Exception as e:
            self.logger.error(f"❌ Take profit check error: {e}")
    
    async def _execute_take_profit(self, symbol: str, position: Dict):
        """Execute take profit order"""
        try:
            exchange_name = position.get("exchange", self.active_exchange)
            exchange = self.exchanges[exchange_name]
            
            # Place take profit order
            order = await exchange.place_order(
                symbol=symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=position.get("quantity", 0)
            )
            
            if order:
                self.logger.info(f"✅ Take profit executed: {symbol}")
                
                # Send alert
                await self.broadcast_message({
                    "type": "take_profit_executed",
                    "symbol": symbol,
                    "order": order,
                    "position": position
                })
                
                self.successful_trades += 1
                
        except Exception as e:
            self.logger.error(f"❌ Take profit execution error: {e}")
    
    async def _risk_monitor(self):
        """Monitor portfolio risk"""
        while self.running:
            try:
                # Calculate current risk metrics
                portfolio_risk = await self._calculate_portfolio_risk()
                
                # Check risk limits
                if portfolio_risk > self.max_portfolio_risk:
                    await self._handle_risk_breach(portfolio_risk)
                
                # Update risk metrics
                await self.update_knowledge("add_node", {
                    "node_type": "risk_metrics",
                    "portfolio_risk": portfolio_risk,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"❌ Risk monitor error: {e}")
                await asyncio.sleep(60)
    
    async def _handle_risk_breach(self, portfolio_risk: float):
        """Handle portfolio risk breach"""
        try:
            self.logger.warning(f"⚠️ Portfolio risk breach: {portfolio_risk:.2%}")
            
            # Send risk alert
            await self.send_system_event(
                "risk_breach",
                "warning",
                f"Portfolio risk exceeded limit: {portfolio_risk:.2%}",
                {"portfolio_risk": portfolio_risk, "limit": self.max_portfolio_risk}
            )
            
            # Consider reducing positions
            await self._reduce_positions()
            
        except Exception as e:
            self.logger.error(f"❌ Risk breach handling error: {e}")
    
    async def _reduce_positions(self):
        """Reduce positions to manage risk"""
        try:
            # Sort positions by risk/loss
            position_risks = []
            for symbol, position in self.positions.items():
                unrealized_pnl = position.get("unrealized_pnl", 0)
                position_risks.append((symbol, unrealized_pnl))
            
            # Sort by loss (most negative first)
            position_risks.sort(key=lambda x: x[1])
            
            # Close losing positions first
            positions_to_close = position_risks[:2]  # Close worst 2 positions
            
            for symbol, _ in positions_to_close:
                await self._close_position(symbol)
                
        except Exception as e:
            self.logger.error(f"❌ Position reduction error: {e}")
    
    async def _close_position(self, symbol: str):
        """Close a position"""
        try:
            position = self.positions.get(symbol)
            if not position:
                return
            
            exchange_name = position.get("exchange", self.active_exchange)
            exchange = self.exchanges[exchange_name]
            
            # Place market sell order to close position
            order = await exchange.place_order(
                symbol=symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=position.get("quantity", 0)
            )
            
            if order:
                self.logger.info(f"✅ Position closed: {symbol}")
                
                # Remove from positions
                if symbol in self.positions:
                    del self.positions[symbol]
                
                # Send update
                await self.broadcast_message({
                    "type": "position_closed",
                    "symbol": symbol,
                    "order": order,
                    "reason": "risk_management"
                })
                
        except Exception as e:
            self.logger.error(f"❌ Position closing error: {e}")
    
    async def _portfolio_updater(self):
        """Update portfolio metrics"""
        while self.running:
            try:
                await self._update_portfolio_metrics()
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                self.logger.error(f"❌ Portfolio updater error: {e}")
                await asyncio.sleep(120)
    
    async def _update_portfolio_metrics(self):
        """Update portfolio performance metrics"""
        try:
            # Calculate portfolio value
            total_value = 0.0
            unrealized_pnl = 0.0
            
            for symbol, position in self.positions.items():
                position_value = position.get("market_value", 0)
                position_pnl = position.get("unrealized_pnl", 0)
                
                total_value += position_value
                unrealized_pnl += position_pnl
            
            # Calculate win rate
            if self.trades_executed > 0:
                self.win_rate = (self.successful_trades / self.trades_executed) * 100
            
            # Update metrics
            portfolio_metrics = {
                "total_value": total_value,
                "unrealized_pnl": unrealized_pnl,
                "trades_executed": self.trades_executed,
                "successful_trades": self.successful_trades,
                "win_rate": self.win_rate,
                "active_positions": len(self.positions),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Store in knowledge engine
            await self.update_knowledge("add_node", {
                "node_type": "portfolio_metrics",
                "metrics": portfolio_metrics
            })
            
            # Broadcast update
            await self.broadcast_message({
                "type": "portfolio_update",
                "metrics": portfolio_metrics
            })
            
        except Exception as e:
            self.logger.error(f"❌ Portfolio metrics update error: {e}")
    
    async def _process_strategies(self):
        """Process active strategies"""
        try:
            for strategy_name, strategy in self.active_strategies.items():
                # Get strategy signals
                signals = await strategy.get_signals()
                
                # Process each signal
                for signal in signals:
                    await self.signal_queue.put(signal)
                    
        except Exception as e:
            self.logger.error(f"❌ Strategy processing error: {e}")
    
    async def _check_strategy_signals(self):
        """Check for new strategy signals"""
        try:
            # This would integrate with the strategy engine
            # For now, just a placeholder
            pass
            
        except Exception as e:
            self.logger.error(f"❌ Strategy signal check error: {e}")
    
    async def _manage_positions(self):
        """Manage existing positions"""
        try:
            # Update position prices
            for symbol, position in self.positions.items():
                exchange_name = position.get("exchange", self.active_exchange)
                exchange = self.exchanges[exchange_name]
                
                # Get current price
                ticker = await exchange.get_ticker(symbol)
                current_price = ticker.get("price", 0)
                
                if current_price > 0:
                    # Update position with current price
                    entry_price = position.get("entry_price", 0)
                    quantity = position.get("quantity", 0)
                    
                    if entry_price > 0:
                        unrealized_pnl = (current_price - entry_price) * quantity
                        market_value = current_price * quantity
                        
                        self.positions[symbol].update({
                            "current_price": current_price,
                            "unrealized_pnl": unrealized_pnl,
                            "market_value": market_value,
                            "last_updated": datetime.utcnow().isoformat()
                        })
                        
        except Exception as e:
            self.logger.error(f"❌ Position management error: {e}")
    
    async def _update_position_metrics(self):
        """Update position-specific metrics"""
        try:
            for symbol, position in self.positions.items():
                # Calculate position metrics
                entry_price = position.get("entry_price", 0)
                current_price = position.get("current_price", 0)
                
                if entry_price > 0 and current_price > 0:
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    
                    # Update position metrics
                    self.positions[symbol].update({
                        "pnl_percentage": pnl_pct,
                        "metrics_updated": datetime.utcnow().isoformat()
                    })
                    
        except Exception as e:
            self.logger.error(f"❌ Position metrics update error: {e}")
    
    # Message handlers
    
    async def _handle_trading_signal(self, data: Dict):
        """Handle trading signal message"""
        try:
            signal = data.get("data", {})
            await self.signal_queue.put(signal)
            
        except Exception as e:
            self.logger.error(f"❌ Trading signal handling error: {e}")
    
    async def _handle_order_request(self, data: Dict):
        """Handle order request message"""
        try:
            order_request = data.get("data", {})
            # Process order request
            await self._process_order_request(order_request)
            
        except Exception as e:
            self.logger.error(f"❌ Order request handling error: {e}")
    
    async def _handle_position_update(self, data: Dict):
        """Handle position update message"""
        try:
            # Update position information
            await self._update_positions()
            
        except Exception as e:
            self.logger.error(f"❌ Position update handling error: {e}")
    
    async def _handle_market_data_update(self, data: Dict):
        """Handle market data update"""
        try:
            # Update position prices with new market data
            await self._manage_positions()
            
        except Exception as e:
            self.logger.error(f"❌ Market data update handling error: {e}")
    
    async def _handle_risk_alert(self, data: Dict):
        """Handle risk alert message"""
        try:
            alert = data.get("data", {})
            alert_type = alert.get("type", "")
            
            if alert_type == "stop_loss":
                symbol = alert.get("symbol", "")
                if symbol in self.positions:
                    await self._execute_stop_loss(symbol, self.positions[symbol])
                    
        except Exception as e:
            self.logger.error(f"❌ Risk alert handling error: {e}")
    
    async def _handle_strategy_update(self, data: Dict):
        """Handle strategy update message"""
        try:
            # Update strategy configuration
            strategy_name = data.get("strategy", "")
            if strategy_name in self.strategies:
                await self.strategies[strategy_name].update_config(data.get("config", {}))
                
        except Exception as e:
            self.logger.error(f"❌ Strategy update handling error: {e}")
    
    async def _process_order_request(self, order_request: Dict):
        """Process order request"""
        try:
            symbol = order_request.get("symbol", "")
            side = order_request.get("side", "")
            quantity = order_request.get("quantity", 0)
            order_type = order_request.get("type", "market")
            
            if not symbol or not side or quantity <= 0:
                return
            
            # Convert to enums
            order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
            order_type_enum = OrderType.MARKET if order_type.upper() == "MARKET" else OrderType.LIMIT
            
            # Execute order
            exchange = self.exchanges[self.active_exchange]
            order = await exchange.place_order(
                symbol=symbol,
                side=order_side,
                order_type=order_type_enum,
                quantity=quantity,
                price=order_request.get("price")
            )
            
            if order:
                self.logger.info(f"✅ Order executed: {symbol} {side} {quantity}")
                
        except Exception as e:
            self.logger.error(f"❌ Order request processing error: {e}")
    
    def get_trading_metrics(self) -> Dict:
        """Get trading performance metrics"""
        return {
            "trades_executed": self.trades_executed,
            "successful_trades": self.successful_trades,
            "win_rate": self.win_rate,
            "total_pnl": self.total_pnl,
            "active_positions": len(self.positions),
            "active_strategies": len(self.active_strategies),
            "connected_exchanges": len([e for e in self.exchanges.values() if e.connected]),
            "uptime": (datetime.utcnow() - self.metrics["uptime"]).total_seconds()
        }
