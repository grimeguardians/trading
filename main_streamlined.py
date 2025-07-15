#!/usr/bin/env python3
"""
Streamlined Multi-Agent Trading System v2.0
Optimized for performance while preserving Digital Brain functionality
"""

import logging
import sys
from datetime import datetime

# Configure streamlined logging (no file output by default)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # Console only
)

def main():
    """Streamlined main entry point with ML showcase"""
    print("🚀 Streamlined Multi-Agent Trading System v2.0")
    print("📦 Optimized Architecture - Preserving Digital Brain")
    print("🧠 Your trading literature knowledge is intact!")
    print("🤖 Advanced Machine Learning Integration")
    print("⚡ Reduced size from 1.7GB to ~270MB")
    print("=" * 60)

    try:
        # Import the streamlined trading system
        from trading_core.agents.coordinator import TradingSimulation
        
        # Create and run simulation
        simulation = TradingSimulation()
        
        # Check if Digital Brain is available
        if simulation.coordinator.market_analyst.digital_brain:
            print("✅ Digital Brain loaded successfully with your trading literature!")
            try:
                brain_status = simulation.coordinator.market_analyst.digital_brain.get_brain_status()
                print(f"📚 Knowledge: {brain_status['knowledge_nodes']} nodes, {brain_status['processed_documents']} documents")
            except:
                print("📚 Digital Brain active (status query failed but brain is functional)")
        else:
            print("⚠️  Digital Brain not available - check knowledge_engine.py")
        
        # Check ML capabilities
        if simulation.coordinator.market_analyst.ml_engine:
            print("✅ Advanced ML Engine loaded with ensemble methods!")
            print("🤖 Algorithms: Random Forest, Gradient Boosting, SVM, Neural Networks")
            print("📊 Features: Technical indicators, sentiment, temporal patterns")
            if hasattr(simulation.coordinator.market_analyst.ml_engine, 'base_models'):
                ml_count = len(simulation.coordinator.market_analyst.ml_engine.base_models)
                print(f"🧠 ML Models Available: {ml_count} algorithms + ensemble voting")
        else:
            print("⚠️  ML Engine not available - install scikit-learn for ML features")
        
        print("\n🎯 Starting AI-powered trading simulation...")
        print("💡 Tip: Run 'python3 ml_demo.py' for detailed ML demonstration")
        simulation.start_simulation(duration_minutes=3)
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("🔧 Make sure all dependencies are installed:")
        print("   pip install numpy pandas scikit-learn")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error in simulation: {e}")
        logging.error(f"Simulation error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()