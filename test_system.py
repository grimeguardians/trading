
#!/usr/bin/env python3
"""
Comprehensive Test Script for Multi-Agent Trading System
"""

import time
import random
from datetime import datetime
from main import TradingSimulation, MarketData, TechnicalIndicators
from knowledge_engine import DigitalBrain
from document_upload import TradingDocumentUploader

def test_technical_indicators():
    """Test technical indicators calculation"""
    print("🔧 Testing Technical Indicators...")
    
    ti = TechnicalIndicators()
    symbol = "TEST"
    
    # Generate test market data
    for i in range(30):
        price = 100 + random.uniform(-5, 5)
        volume = random.randint(1000, 5000)
        market_data = MarketData(
            symbol=symbol,
            price=price,
            volume=volume,
            timestamp=datetime.now(),
            bid=price - 0.05,
            ask=price + 0.05
        )
        ti.update_data(market_data)
    
    # Calculate indicators
    analysis = ti.calculate_indicators(symbol)
    
    if analysis:
        print(f"✅ Technical indicators working for {symbol}")
        print(f"   RSI: {analysis.rsi:.2f}")
        print(f"   MACD: {analysis.macd:.3f}")
        print(f"   Volatility: {analysis.volatility:.3f}")
        return True
    else:
        print(f"❌ Technical indicators failed for {symbol}")
        return False

def test_digital_brain():
    """Test Digital Brain functionality"""
    print("\n🧠 Testing Digital Brain...")
    
    try:
        brain = DigitalBrain()
        
        # Test basic brain status
        status = brain.get_brain_status()
        print(f"✅ Digital Brain initialized")
        print(f"   Knowledge nodes: {status['knowledge_nodes']}")
        print(f"   Memory health: {status['memory_health']}")
        
        # Test market event processing
        test_event = {
            'symbol': 'TEST',
            'price': 100.0,
            'volume': 1000,
            'event_type': 'test_event',
            'rsi': 65.0,
            'macd': 0.5
        }
        
        result = brain.process_market_event(test_event)
        print(f"✅ Market event processed: {len(result.get('recognized_patterns', []))} patterns")
        
        # Test brain query
        query_result = brain.query_brain("What are bullish patterns?", {'symbol': 'TEST'})
        print(f"✅ Brain query completed: confidence {query_result.get('confidence', 0):.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Digital Brain test failed: {e}")
        return False

def test_document_uploader():
    """Test document upload functionality"""
    print("\n📚 Testing Document Uploader...")
    
    try:
        uploader = TradingDocumentUploader()
        
        # Test memory bank stats
        stats = uploader.get_memory_bank_stats()
        print(f"✅ Document uploader initialized")
        print(f"   Total documents: {stats['total_documents']}")
        print(f"   Brain status: {stats.get('brain_status', {}).get('memory_health', 'unknown')}")
        
        # Test query
        query_result = uploader.query_memory_bank("What are chart patterns?")
        print(f"✅ Memory bank query: {query_result.get('confidence', 0):.2f} confidence")
        
        return True
        
    except Exception as e:
        print(f"❌ Document uploader test failed: {e}")
        return False

def test_trading_simulation():
    """Test full trading simulation"""
    print("\n🚀 Testing Trading Simulation...")
    
    try:
        simulation = TradingSimulation()
        
        print("✅ Trading simulation initialized")
        print("   Coordinator created")
        print("   Agents initialized")
        
        # Test single iteration
        coordinator = simulation.coordinator
        coordinator.start_system()
        
        # Generate test market data
        test_data = MarketData(
            symbol="AAPL",
            price=150.0,
            volume=2000,
            timestamp=datetime.now(),
            bid=149.95,
            ask=150.05
        )
        
        # Process single market data point
        result = coordinator.process(test_data)
        
        print(f"✅ Single iteration processed")
        print(f"   Signals generated: {result.get('signals_generated', 0)}")
        print(f"   System health: {result.get('system_health', {}).get('overall_status', 'unknown')}")
        
        coordinator.stop_system()
        return True
        
    except Exception as e:
        print(f"❌ Trading simulation test failed: {e}")
        return False

def run_mini_simulation():
    """Run a mini 30-second simulation"""
    print("\n⚡ Running Mini Simulation (30 seconds)...")
    
    try:
        simulation = TradingSimulation()
        simulation.coordinator.start_system()
        
        symbols = ['AAPL', 'GOOGL']
        base_prices = {'AAPL': 150.0, 'GOOGL': 120.0}
        
        for i in range(15):  # 15 iterations over 30 seconds
            for symbol in symbols:
                # Generate realistic price movement
                price_change = random.uniform(-0.02, 0.02)
                base_prices[symbol] *= (1 + price_change)
                
                market_data = MarketData(
                    symbol=symbol,
                    price=base_prices[symbol],
                    volume=random.randint(1000, 5000),
                    timestamp=datetime.now(),
                    bid=base_prices[symbol] - 0.05,
                    ask=base_prices[symbol] + 0.05
                )
                
                result = simulation.coordinator.process(market_data)
                
                if i % 5 == 0:  # Print every 5th iteration
                    print(f"   Iteration {i+1}: {symbol} @ ${base_prices[symbol]:.2f} | "
                          f"Signals: {result.get('signals_generated', 0)} | "
                          f"Orders: {result.get('orders_executed', 0)}")
            
            time.sleep(2)  # 2 seconds between iterations
        
        # Get final portfolio summary
        portfolio = simulation.coordinator.trading_executor.get_portfolio_summary()
        print(f"\n✅ Mini simulation completed!")
        print(f"   Portfolio value: ${portfolio['total_portfolio_value']:,.2f}")
        print(f"   Total orders: {portfolio['executed_orders_count']}")
        print(f"   Active positions: {portfolio['active_positions_count']}")
        
        simulation.coordinator.stop_system()
        return True
        
    except Exception as e:
        print(f"❌ Mini simulation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 COMPREHENSIVE TRADING SYSTEM TESTS")
    print("=" * 50)
    
    test_results = []
    
    # Run individual component tests
    test_results.append(("Technical Indicators", test_technical_indicators()))
    test_results.append(("Digital Brain", test_digital_brain()))
    test_results.append(("Document Uploader", test_document_uploader()))
    test_results.append(("Trading Simulation", test_trading_simulation()))
    
    # Run mini simulation
    test_results.append(("Mini Simulation", run_mini_simulation()))
    
    # Print test summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20s} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(test_results)} tests passed")
    
    if passed == len(test_results):
        print("🎉 All tests passed! System is ready for full simulation.")
        print("\nTo run the full simulation, use:")
        print("   python main.py")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        print("\nYou can still try running the main simulation:")
        print("   python main.py")
    
    print("\nOther testing options:")
    print("   python test_brain_query.py  # Test Digital Brain queries")
    print("   python upload_pdf_book.py   # Upload trading documents")

if __name__ == "__main__":
    main()
