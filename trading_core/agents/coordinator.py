"""Streamlined coordinator agent"""

import time
import random
from datetime import datetime
from typing import Dict, Any, List

from .base_agent import BaseAgent
from .market_analyst import MarketAnalystAgent
from .risk_manager import RiskManagerAgent
from .trading_executor import TradingExecutorAgent
from ..data_models import MarketData

class CoordinatorAgent(BaseAgent):
    """Streamlined central coordinator"""

    def __init__(self):
        super().__init__("Coordinator")
        self.market_analyst = MarketAnalystAgent()
        self.risk_manager = RiskManagerAgent()
        self.trading_executor = TradingExecutorAgent()
        self.is_running = False
        self.iteration_count = 0
        self.performance_history = []

    def start_system(self):
        """Start the trading system"""
        self.logger.info("Starting Streamlined Multi-Agent Trading System")
        
        self.market_analyst.start()
        self.risk_manager.start()
        self.trading_executor.start()
        self.start()
        
        self.is_running = True

    def stop_system(self):
        """Stop the trading system"""
        self.logger.info("Stopping Streamlined Trading System")
        
        self.is_running = False
        self.market_analyst.stop()
        self.risk_manager.stop()
        self.trading_executor.stop()
        self.stop()

    def process(self, market_data: MarketData) -> Dict[str, Any]:
        """Main coordination process"""
        try:
            self.iteration_count += 1

            # Step 1: Update positions
            self.trading_executor.update_positions(market_data)

            # Step 2: Check stop-losses
            stop_orders = self.risk_manager.process_market_update(market_data)
            stop_executions = 0
            for stop_order in stop_orders:
                if self.trading_executor.process(stop_order):
                    stop_executions += 1

            # Step 3: Generate trading signals
            signals = self.market_analyst.process(market_data)

            # Step 4: Risk management and order execution
            orders_executed = 0
            risk_alerts = []

            for signal in signals:
                current_positions = self.trading_executor.positions
                order = self.risk_manager.process(signal, current_positions)

                if order:
                    success = self.trading_executor.process(order)
                    if success:
                        orders_executed += 1
                        
                        # Create stop-loss orders for new positions
                        if order.action == 'BUY' and order.stop_loss_price:
                            position = self.trading_executor.positions.get(order.symbol)
                            if position:
                                self.risk_manager.create_stop_loss_orders(position)
                else:
                    risk_alerts.append(f"Order blocked for {signal.symbol}")

            # Step 5: Portfolio optimization (every 20 iterations)
            portfolio_optimization = None
            if self.iteration_count % 20 == 0:
                current_positions = self.trading_executor.positions
                technical_data = {}
                for symbol in current_positions.keys():
                    ta = self.market_analyst.technical_indicators.calculate_indicators(symbol)
                    if ta:
                        technical_data[symbol] = ta
                
                portfolio_optimization = self.market_analyst.portfolio_optimizer.optimize_portfolio(
                    current_positions, technical_data)

            # Step 6: Generate summary
            portfolio_summary = self.trading_executor.get_portfolio_summary()
            risk_report = self.risk_manager.get_risk_report()
            
            result = {
                'signals_generated': len(signals),
                'orders_executed': orders_executed,
                'stop_loss_executions': stop_executions,
                'portfolio_summary': portfolio_summary,
                'risk_report': risk_report,
                'alerts': risk_alerts,
                'system_health': self._check_system_health(),
                'timestamp': datetime.now()
            }

            if portfolio_optimization:
                result['portfolio_optimization'] = portfolio_optimization

            return result

        except Exception as e:
            self.logger.error(f"Error in coordination process: {e}")
            return {'error': str(e)}

    def _check_system_health(self) -> Dict[str, str]:
        """Check system health"""
        health = {
            'market_analyst': 'HEALTHY' if self.market_analyst.is_active else 'INACTIVE',
            'risk_manager': 'HEALTHY' if self.risk_manager.is_active else 'INACTIVE',
            'trading_executor': 'HEALTHY' if self.trading_executor.is_active else 'INACTIVE',
            'overall_status': 'OPERATIONAL'
        }

        # Check Digital Brain status
        if self.market_analyst.digital_brain:
            try:
                brain_status = self.market_analyst.digital_brain.get_brain_status()
                health['digital_brain'] = f"ACTIVE ({brain_status['knowledge_nodes']} nodes)"
            except:
                health['digital_brain'] = 'ERROR'
        else:
            health['digital_brain'] = 'UNAVAILABLE'

        return health

class TradingSimulation:
    """Streamlined trading simulation"""

    def __init__(self):
        self.coordinator = CoordinatorAgent()
        self.symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        self.simulation_running = False
        self._price_trends = {symbol: 100.0 for symbol in self.symbols}

    def start_simulation(self, duration_minutes: int = 3):
        """Start the trading simulation"""
        print("🚀 Starting Streamlined Multi-Agent Trading System")
        print("Features: Digital Brain Integration, Advanced Analytics, Risk Management")
        print("=" * 70)

        self.coordinator.start_system()
        self.simulation_running = True

        start_time = time.time()
        iteration = 0

        try:
            while self.simulation_running and (time.time() - start_time) < (duration_minutes * 60):
                iteration += 1

                # Process each symbol
                for symbol in self.symbols:
                    market_data = self._generate_market_data(symbol)
                    result = self.coordinator.process(market_data)

                    if 'error' not in result:
                        # Show progress every 5 iterations
                        if iteration % 5 == 0:
                            print(f"\n[Iteration {iteration}] {symbol}: "
                                  f"Signals: {result['signals_generated']} | "
                                  f"Orders: {result['orders_executed']} | "
                                  f"Stops: {result.get('stop_loss_executions', 0)}")

                        # Full dashboard every 10 iterations
                        if iteration % 10 == 0 and symbol == self.symbols[0]:
                            self._display_dashboard(result, iteration)

                time.sleep(1)  # Reduced from 2 seconds for faster simulation

        except KeyboardInterrupt:
            print("\n⚠️ Simulation interrupted by user")
        finally:
            self.stop_simulation()

    def _display_dashboard(self, result: Dict[str, Any], iteration: int):
        """Display trading dashboard"""
        portfolio = result['portfolio_summary']
        health = result.get('system_health', {})

        print(f"\n{'='*70}")
        print(f"🤖 STREAMLINED TRADING DASHBOARD - Iteration {iteration}")
        print(f"{'='*70}")

        # Portfolio status
        print(f"💰 PORTFOLIO:")
        print(f"  Value: ${portfolio['total_portfolio_value']:,.2f} | "
              f"Return: {portfolio['total_return_pct']:+.2f}% | "
              f"Positions: {portfolio['active_positions_count']}")

        # Performance
        print(f"📈 PERFORMANCE:")
        print(f"  Win Rate: {portfolio['win_rate']:.1f}% | "
              f"Trades: {portfolio.get('total_trades', 0)} | "
              f"Stop Coverage: {portfolio.get('stop_loss_coverage_pct', 0):.0f}%")

        # System health
        print(f"🔧 SYSTEM: {health.get('overall_status', 'UNKNOWN')}")
        if 'digital_brain' in health:
            print(f"🧠 Digital Brain: {health['digital_brain']}")

        # Show alerts if any
        alerts = result.get('alerts', [])
        if alerts:
            print(f"🚨 ALERTS: {', '.join(alerts[:2])}")

        print(f"{'='*70}")

    def stop_simulation(self):
        """Stop the simulation"""
        print("\n🛑 Stopping simulation...")
        self.simulation_running = False
        self.coordinator.stop_system()

        # Final summary
        final_summary = self.coordinator.trading_executor.get_portfolio_summary()
        print("\n" + "=" * 70)
        print("📈 FINAL PORTFOLIO SUMMARY")
        print("=" * 70)
        print(f"Portfolio Value: ${final_summary['total_portfolio_value']:,.2f}")
        print(f"Total Return: {final_summary['total_return_pct']:+.2f}%")
        print(f"Total Trades: {final_summary.get('total_trades', 0)}")
        print(f"Win Rate: {final_summary['win_rate']:.1f}%")

        active_positions = [p for p in final_summary['positions'].values() if p.quantity > 0]
        if active_positions:
            print(f"\nActive Positions ({len(active_positions)}):")
            for position in active_positions[:5]:  # Show first 5
                pnl = position.unrealized_pnl + position.realized_pnl
                print(f"  {position.symbol}: {position.quantity} shares @ ${position.avg_price:.2f} (P&L: ${pnl:.2f})")

    def _generate_market_data(self, symbol: str) -> MarketData:
        """Generate realistic market data"""
        # More realistic price movements
        trend = random.uniform(-0.5, 0.5)
        volatility = random.uniform(-2.0, 2.0)
        self._price_trends[symbol] += trend + volatility

        # Keep price in reasonable range
        self._price_trends[symbol] = max(80, min(120, self._price_trends[symbol]))

        base_price = self._price_trends[symbol]
        spread = 0.02

        return MarketData(
            symbol=symbol,
            price=base_price,
            volume=random.randint(1000, 8000),
            timestamp=datetime.now(),
            bid=base_price - spread,
            ask=base_price + spread,
            high_24h=base_price * (1 + random.uniform(0, 0.02)),
            low_24h=base_price * (1 - random.uniform(0, 0.02))
        )