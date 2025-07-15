#!/usr/bin/env python3
"""
Enhanced Paper Trading System with Advanced Digital Brain Integration
Real-time trading with sophisticated AI features
"""

import logging
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import random
import threading
from dataclasses import dataclass
from collections import defaultdict, deque

from main import TradingSimulation, MarketData, CoordinatorAgent
from advanced_brain_features import AdvancedBrainFeatures, AdvancedSignal
from knowledge_engine import DigitalBrain
from real_time_brain_integration import RealTimeBrainIntegration

@dataclass
class EnhancedTrade:
    """Enhanced trade with AI insights"""
    trade_id: str
    symbol: str
    action: str
    quantity: int
    entry_price: float
    exit_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    entry_time: datetime
    exit_time: Optional[datetime]
    ai_confidence: float
    strategy_type: str
    reasoning_chain: List[str]
    risk_reward_ratio: float
    actual_return: Optional[float]
    status: str  # 'open', 'closed', 'stopped_out'

class EnhancedPaperTradingSystem:
    """Enhanced paper trading with advanced AI features"""

    def __init__(self):
        self.logger = logging.getLogger("EnhancedPaperTrading")

        # Core components
        self.digital_brain = DigitalBrain()
        self.advanced_features = AdvancedBrainFeatures(self.digital_brain)
        self.coordinator = CoordinatorAgent()

        # Trading state
        self.portfolio_value = 100000.0
        self.cash_balance = 100000.0
        self.positions = {}
        self.open_trades = {}
        self.closed_trades = []
        self.trade_counter = 0

        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.ai_performance_metrics = {
            'total_signals': 0,
            'successful_signals': 0,
            'ai_accuracy': 0.0,
            'strategy_breakdown': defaultdict(int),
            'confidence_correlation': []
        }

        # Real-time features
        self.symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        self.market_data_cache = {}
        self.is_running = False

    async def start_enhanced_trading(self, duration_minutes: int = 5):
        """Start enhanced paper trading session"""
        print("🚀 Enhanced Paper Trading System with Advanced AI")
        print("=" * 70)

        # Initialize systems
        await self.advanced_features.start_advanced_processing()
        self.coordinator.start_system()
        self.is_running = True

        try:
            # Start real-time trading loop
            await self._real_time_trading_loop(duration_minutes)

        finally:
            await self._shutdown_systems()

    async def _real_time_trading_loop(self, duration_minutes: int):
        """Main real-time trading loop"""
        start_time = time.time()
        iteration = 0

        print(f"🎯 Starting {duration_minutes}-minute enhanced trading session")
        print(f"💰 Initial Portfolio Value: ${self.portfolio_value:,.2f}")

        while self.is_running and (time.time() - start_time) < (duration_minutes * 60):
            iteration += 1

            # Process each symbol
            for symbol in self.symbols:
                await self._process_symbol_enhanced(symbol, iteration)

            # Update portfolio and display status
            if iteration % 5 == 0:  # Every 10 seconds
                await self._update_portfolio_status()
                self._display_enhanced_status(iteration)

            # Portfolio optimization every 30 iterations
            if iteration % 30 == 0:
                await self._optimize_portfolio()

            await asyncio.sleep(2)  # 2-second intervals

        # Final summary
        await self._display_final_summary()

    async def _process_symbol_enhanced(self, symbol: str, iteration: int):
        """Process symbol with advanced AI features"""
        try:
            # Generate realistic market data
            market_data = self._generate_enhanced_market_data(symbol)
            self.market_data_cache[symbol] = market_data

            # Get advanced AI signals
            advanced_signals = self.advanced_features.generate_advanced_signals(
                symbol, self._convert_to_dict(market_data)
            )

            # Process signals for trading
            for signal in advanced_signals:
                await self._process_advanced_signal(signal, market_data)

            # Update existing positions
            await self._update_existing_positions(symbol, market_data)

            # Track AI performance
            self._track_ai_performance(symbol, advanced_signals)

        except Exception as e:
            self.logger.error(f"Error processing {symbol}: {e}")

    async def _process_advanced_signal(self, signal: AdvancedSignal, market_data: MarketData):
        """Process advanced signal for potential trade execution"""
        try:
            # Risk assessment
            if not self._validate_signal_risk(signal):
                return

            # Position sizing with AI confidence
            position_size = self._calculate_ai_position_size(signal)

            if position_size > 0:
                # Execute trade
                trade = await self._execute_enhanced_trade(signal, position_size, market_data)

                if trade:
                    self.open_trades[trade.trade_id] = trade
                    self.trade_counter += 1

                    self.logger.info(f"🎯 Executed {trade.action} {trade.symbol}: "
                                   f"{trade.quantity} shares @ ${trade.entry_price:.2f}")
                    self.logger.info(f"   AI Strategy: {trade.strategy_type} "
                                   f"(Confidence: {trade.ai_confidence:.1%})")

        except Exception as e:
            self.logger.error(f"Error processing signal for {signal.symbol}: {e}")

    async def _execute_enhanced_trade(self, signal: AdvancedSignal, quantity: int, 
                                    market_data: MarketData) -> Optional[EnhancedTrade]:
        """Execute enhanced trade with AI insights"""
        try:
            current_price = market_data.price
            trade_value = quantity * current_price

            # Check available capital
            if signal.action == 'BUY' and trade_value > self.cash_balance:
                return None

            # Check existing position for SELL
            if signal.action == 'SELL' and self.positions.get(signal.symbol, 0) < quantity:
                return None

            # Create enhanced trade
            trade = EnhancedTrade(
                trade_id=f"T{self.trade_counter:04d}_{signal.symbol}_{signal.action}",
                symbol=signal.symbol,
                action=signal.action,
                quantity=quantity,
                entry_price=current_price,
                exit_price=None,
                stop_loss=self._calculate_dynamic_stop_loss(signal, current_price),
                take_profit=self._calculate_dynamic_take_profit(signal, current_price),
                entry_time=datetime.now(),
                exit_time=None,
                ai_confidence=signal.confidence,
                strategy_type=signal.strategy_type,
                reasoning_chain=signal.reasoning_chain,
                risk_reward_ratio=signal.risk_reward_ratio,
                actual_return=None,
                status='open'
            )

            # Update portfolio
            if signal.action == 'BUY':
                self.cash_balance -= trade_value
                self.positions[signal.symbol] = self.positions.get(signal.symbol, 0) + quantity
            else:  # SELL
                self.cash_balance += trade_value
                self.positions[signal.symbol] = self.positions.get(signal.symbol, 0) - quantity

            return trade

        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return None

    async def _update_existing_positions(self, symbol: str, market_data: MarketData):
        """Update existing positions with current market data"""
        try:
            current_price = market_data.price

            # Check stop-loss and take-profit levels
            trades_to_close = []

            for trade_id, trade in self.open_trades.items():
                if trade.symbol == symbol and trade.status == 'open':

                    # Check stop-loss
                    if trade.stop_loss:
                        if ((trade.action == 'BUY' and current_price <= trade.stop_loss) or
                            (trade.action == 'SELL' and current_price >= trade.stop_loss)):

                            trades_to_close.append((trade_id, current_price, 'stopped_out'))

                    # Check take-profit
                    elif trade.take_profit:
                        if ((trade.action == 'BUY' and current_price >= trade.take_profit) or
                            (trade.action == 'SELL' and current_price <= trade.take_profit)):

                            trades_to_close.append((trade_id, current_price, 'take_profit'))

            # Close triggered trades
            for trade_id, exit_price, reason in trades_to_close:
                await self._close_trade(trade_id, exit_price, reason)

        except Exception as e:
            self.logger.error(f"Error updating positions for {symbol}: {e}")

    async def _close_trade(self, trade_id: str, exit_price: float, reason: str):
        """Close trade and update portfolio"""
        try:
            trade = self.open_trades.get(trade_id)
            if not trade:
                return

            # Calculate return
            if trade.action == 'BUY':
                trade_return = (exit_price - trade.entry_price) / trade.entry_price
                self.cash_balance += trade.quantity * exit_price
                self.positions[trade.symbol] -= trade.quantity
            else:  # SELL
                trade_return = (trade.entry_price - exit_price) / trade.entry_price
                self.cash_balance -= trade.quantity * exit_price
                self.positions[trade.symbol] += trade.quantity

            # Update trade
            trade.exit_price = exit_price
            trade.exit_time = datetime.now()
            trade.actual_return = trade_return
            trade.status = reason

            # Move to closed trades
            self.closed_trades.append(trade)
            del self.open_trades[trade_id]

            # Log trade closure
            return_pct = trade_return * 100
            duration = (trade.exit_time - trade.entry_time).total_seconds() / 60

            self.logger.info(f"📈 Closed {trade.symbol} {trade.action}: "
                           f"{return_pct:+.2f}% in {duration:.1f}min ({reason})")

            # Update AI performance tracking
            self._update_ai_performance(trade)

        except Exception as e:
            self.logger.error(f"Error closing trade {trade_id}: {e}")

    def _calculate_ai_position_size(self, signal: AdvancedSignal) -> int:
        """Calculate position size based on AI confidence and risk assessment"""
        try:
            # Base position size (2% of portfolio)
            base_position_value = self.portfolio_value * 0.02

            # Confidence multiplier
            confidence_multiplier = 0.5 + (signal.confidence * 1.5)  # 0.5x to 2.0x

            # Risk-reward adjustment
            rr_multiplier = min(signal.risk_reward_ratio / 2.0, 1.5)  # Cap at 1.5x

            # Strategy-specific adjustment
            strategy_multipliers = {
                'momentum': 1.2,
                'mean_reversion': 0.9,
                'breakout': 1.3,
                'adaptive': 1.0
            }
            strategy_multiplier = strategy_multipliers.get(signal.strategy_type, 1.0)

            # Priority adjustment
            priority_multiplier = 2.0 - (signal.execution_priority * 0.2)  # Higher priority = larger size

            # Final position value
            adjusted_value = (base_position_value * confidence_multiplier * 
                            rr_multiplier * strategy_multiplier * priority_multiplier)

            # Adaptive parameter adjustment
            adaptive_multiplier = signal.adaptive_parameters.get('position_size_multiplier', 1.0)
            final_value = adjusted_value * adaptive_multiplier

            # Convert to shares (assuming reasonable price range)
            estimated_price = 150.0  # Rough estimate for demo
            quantity = int(final_value / estimated_price)

            return max(1, min(quantity, 100))  # Min 1, max 100 shares

        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 1

    def _calculate_dynamic_stop_loss(self, signal: AdvancedSignal, current_price: float) -> Optional[float]:
        """Calculate dynamic stop-loss based on AI insights"""
        try:
            # Base stop distance (2% for demo)
            base_stop_distance = current_price * 0.02

            # Adjust based on AI confidence
            confidence_adjustment = 1.0 + (1.0 - signal.confidence)  # Lower confidence = wider stop

            # Adjust based on market conditions
            volatility = signal.market_conditions.get('volatility', 0.2)
            volatility_adjustment = 1.0 + volatility  # Higher volatility = wider stop

            # Adaptive parameter adjustment
            stop_buffer = signal.adaptive_parameters.get('stop_loss_buffer', 1.0)

            # Final stop distance
            final_distance = base_stop_distance * confidence_adjustment * volatility_adjustment * stop_buffer

            if signal.action == 'BUY':
                return current_price - final_distance
            else:
                return current_price + final_distance

        except Exception as e:
            self.logger.error(f"Error calculating stop-loss: {e}")
            return None

    def _calculate_dynamic_take_profit(self, signal: AdvancedSignal, current_price: float) -> Optional[float]:
        """Calculate dynamic take-profit based on AI insights"""
        try:
            # Use risk-reward ratio from AI
            stop_loss = self._calculate_dynamic_stop_loss(signal, current_price)
            if not stop_loss:
                return None

            stop_distance = abs(current_price - stop_loss)
            profit_distance = stop_distance * signal.risk_reward_ratio

            if signal.action == 'BUY':
                return current_price + profit_distance
            else:
                return current_price - profit_distance

        except Exception as e:
            self.logger.error(f"Error calculating take-profit: {e}")
            return None

    def _validate_signal_risk(self, signal: AdvancedSignal) -> bool:
        """Validate signal against risk parameters"""
        # Minimum confidence threshold
        if signal.confidence < 0.6:
            return False

        # Execution priority check
        if signal.execution_priority > 4:  # Only execute priority 1-4
            return False

        # Risk-reward ratio check
        if signal.risk_reward_ratio < 1.2:
            return False

        return True

    async def _update_portfolio_status(self):
        """Update portfolio valuation"""
        try:
            # Calculate current portfolio value
            total_value = self.cash_balance

            for symbol, quantity in self.positions.items():
                if quantity > 0 and symbol in self.market_data_cache:
                    current_price = self.market_data_cache[symbol].price
                    total_value += quantity * current_price

            self.portfolio_value = total_value

            # Add to performance history
            self.performance_history.append({
                'timestamp': datetime.now(),
                'portfolio_value': total_value,
                'total_return': (total_value - 100000) / 100000,
                'open_trades': len(self.open_trades),
                'closed_trades': len(self.closed_trades)
            })

        except Exception as e:
            self.logger.error(f"Error updating portfolio status: {e}")

    def _display_enhanced_status(self, iteration: int):
        """Display enhanced trading status"""
        if iteration % 20 == 0:  # Every 40 seconds
            total_return = (self.portfolio_value - 100000) / 100000

            print(f"\n🤖 Enhanced AI Trading Status - Iteration {iteration}")
            print("-" * 60)
            print(f"💰 Portfolio Value: ${self.portfolio_value:,.2f} ({total_return:+.2%})")
            print(f"💵 Cash Balance: ${self.cash_balance:,.2f}")
            print(f"📊 Open Trades: {len(self.open_trades)} | Closed: {len(self.closed_trades)}")

            # AI Performance
            ai_accuracy = self.ai_performance_metrics['ai_accuracy']
            total_signals = self.ai_performance_metrics['total_signals']
            print(f"🧠 AI Accuracy: {ai_accuracy:.1%} ({total_signals} signals)")

            # Active positions
            if self.positions:
                print(f"📈 Active Positions:")
                for symbol, qty in self.positions.items():
                    if qty > 0 and symbol in self.market_data_cache:
                        current_price = self.market_data_cache[symbol].price
                        value = qty * current_price
                        print(f"   {symbol}: {qty} shares @ ${current_price:.2f} = ${value:,.2f}")

            # Recent closed trades
            if self.closed_trades:
                recent_trades = self.closed_trades[-3:]
                print(f"📊 Recent Trades:")
                for trade in recent_trades:
                    return_pct = trade.actual_return * 100 if trade.actual_return else 0
                    duration = ((trade.exit_time or datetime.now()) - trade.entry_time).total_seconds() / 60
                    print(f"   {trade.symbol} {trade.action}: {return_pct:+.1f}% in {duration:.0f}min")

    async def _optimize_portfolio(self):
        """Perform portfolio optimization"""
        try:
            # Get advanced insights
            insights = self.advanced_features.get_advanced_insights()

            # Log optimization insights
            if insights:
                regime_forecast = insights.get('market_regime_forecast', {})
                if regime_forecast:
                    self.logger.info(f"🔮 Market Regime Forecast: {regime_forecast}")

                correlations = insights.get('cross_asset_correlations', {})
                if correlations:
                    self.logger.info(f"🌐 Cross-Asset Analysis: {correlations}")

        except Exception as e:
            self.logger.error(f"Error in portfolio optimization: {e}")

    def _track_ai_performance(self, symbol: str, signals: List[AdvancedSignal]):
        """Track AI signal performance"""
        try:
            self.ai_performance_metrics['total_signals'] += len(signals)

            for signal in signals:
                # Track strategy breakdown
                self.ai_performance_metrics['strategy_breakdown'][signal.strategy_type] += 1

                # Track confidence correlation
                self.ai_performance_metrics['confidence_correlation'].append(signal.confidence)

        except Exception as e:
            self.logger.error(f"Error tracking AI performance: {e}")

    def _update_ai_performance(self, trade: EnhancedTrade):
        """Update AI performance based on trade outcome"""
        try:
            if trade.actual_return and trade.actual_return > 0:
                self.ai_performance_metrics['successful_signals'] += 1

            # Calculate accuracy
            total = self.ai_performance_metrics['total_signals']
            successful = self.ai_performance_metrics['successful_signals']

            if total > 0:
                self.ai_performance_metrics['ai_accuracy'] = successful / total

        except Exception as e:
            self.logger.error(f"Error updating AI performance: {e}")

    async def _display_final_summary(self):
        """Display comprehensive final summary"""
        print(f"\n" + "=" * 70)
        print(f"🎉 Enhanced Paper Trading Session Complete!")
        print(f"=" * 70)

        # Portfolio performance
        total_return = (self.portfolio_value - 100000) / 100000
        print(f"💰 Final Portfolio Value: ${self.portfolio_value:,.2f}")
        print(f"📈 Total Return: {total_return:+.2%}")
        print(f"💵 Cash Balance: ${self.cash_balance:,.2f}")

        # Trading statistics
        total_trades = len(self.closed_trades)
        winning_trades = sum(1 for trade in self.closed_trades if trade.actual_return and trade.actual_return > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        print(f"\n📊 Trading Statistics:")
        print(f"   Total Trades: {total_trades}")
        print(f"   Winning Trades: {winning_trades}")
        print(f"   Win Rate: {win_rate:.1%}")
        print(f"   Open Positions: {len(self.open_trades)}")

        # AI Performance
        ai_metrics = self.ai_performance_metrics
        print(f"\n🧠 AI Performance:")
        print(f"   Signals Generated: {ai_metrics['total_signals']}")
        print(f"   AI Accuracy: {ai_metrics['ai_accuracy']:.1%}")

        # Strategy breakdown
        strategy_breakdown = ai_metrics['strategy_breakdown']
        if strategy_breakdown:
            print(f"   Strategy Usage:")
            for strategy, count in strategy_breakdown.items():
                print(f"      {strategy.title()}: {count}")

        # Best trades
        if self.closed_trades:
            best_trades = sorted(self.closed_trades, 
                               key=lambda t: t.actual_return or 0, reverse=True)[:3]
            print(f"\n🏆 Best Trades:")
            for i, trade in enumerate(best_trades, 1):
                return_pct = (trade.actual_return or 0) * 100
                print(f"   {i}. {trade.symbol} {trade.action}: {return_pct:+.1f}% "
                     f"({trade.strategy_type}, AI: {trade.ai_confidence:.0%})")

        # Advanced insights summary
        insights = self.advanced_features.get_advanced_insights()
        print(f"\n🔬 Advanced Insights Summary:")
        print(f"   Strategy Adaptations: {len(insights.get('adaptation_summary', {}).get('adaptations', []))}")
        print(f"   Pattern Evolution Tracked: {len(insights.get('pattern_evolution', {}))}")
        print(f"   Cross-Asset Correlations: Active")
        print(f"   Market Regime Prediction: Active")

        print(f"\n🚀 System Ready for Full Integration!")

    async def _shutdown_systems(self):
        """Shutdown all systems gracefully"""
        try:
            self.is_running = False
            await self.advanced_features.stop_advanced_processing()
            self.coordinator.stop_system()
            print(f"✅ All systems shutdown gracefully")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

    def _generate_enhanced_market_data(self, symbol: str) -> MarketData:
        """Generate enhanced market data with realistic patterns"""
        if not hasattr(self, '_price_trends'):
            self._price_trends = {s: 100.0 for s in self.symbols}
            self._volatility_states = {s: 0.2 for s in self.symbols}

        # Evolving price with persistence
        trend = random.uniform(-2.0, 2.0)
        volatility_component = random.uniform(-5.0, 5.0) * self._volatility_states[symbol]

        self._price_trends[symbol] += trend + volatility_component
        self._price_trends[symbol] = max(50, min(200, self._price_trends[symbol]))

        # Evolving volatility
        self._volatility_states[symbol] += random.uniform(-0.05, 0.05)
        self._volatility_states[symbol] = max(0.1, min(0.5, self._volatility_states[symbol]))

        base_price = self._price_trends[symbol]
        spread = base_price * 0.001

        return MarketData(
            symbol=symbol,
            price=base_price,
            volume=random.randint(20000000, 100000000),
            timestamp=datetime.now(),
            bid=base_price - spread,
            ask=base_price + spread,
            high_24h=base_price * (1 + random.uniform(0, 0.03)),
            low_24h=base_price * (1 - random.uniform(0, 0.03))
        )

    def _convert_to_dict(self, market_data: MarketData) -> Dict[str, Any]:
        """Convert MarketData to dictionary"""
        return {
            'symbol': market_data.symbol,
            'price': market_data.price,
            'volume': market_data.volume,
            'bid': market_data.bid,
            'ask': market_data.ask,
            'high_24h': market_data.high_24h,
            'low_24h': market_data.low_24h,
            'volatility': random.uniform(0.15, 0.35),
            'rsi': random.uniform(30, 80),
            'macd': random.uniform(-2, 2),
            'trend_strength': random.uniform(0.3, 0.9),
            'volume_ratio': market_data.volume / 50000000,
            'sentiment': random.uniform(-0.3, 0.7)
        }

async def main():
    """Main enhanced paper trading demo - manual start only"""
    print("🚀 Enhanced Paper Trading System - Manual Control Mode")
    print("=" * 50)
    print("⚙️ System initialized but not started")
    print("🎛️ MANUAL CONTROLS:")
    print("   >>> system = EnhancedPaperTradingSystem()")
    print("   >>> await system.start_enhanced_trading(5)  # Start 5-min session")
    print()
    print("⚠️ NO AUTO-START - Use manual commands only")
    
    system = EnhancedPaperTradingSystem()
    return system

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())ystem._shutdown_systems()

if __name__ == "__main__":
    asyncio.run(main())