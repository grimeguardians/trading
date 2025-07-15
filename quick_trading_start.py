
#!/usr/bin/env python3
"""
Quick Trading Start - Immediate Paper Trading
No waiting for TD Ameritrade approval needed
"""

from alternative_brokers import QuickPaperTradingSystem, AlternativeBrokerType

def quick_start_trading():
    """Quick paper trading system - manual start only"""
    print("🏃‍♂️ QUICK START PAPER TRADING - Manual Control Mode")
    print("=" * 50)
    print("✨ No approval waiting required!")
    print("📊 Real market data")
    print("🎯 Instant execution")
    print()
    print("⚙️ System initialized but not started")
    print("🎛️ MANUAL CONTROLS:")
    print("   >>> quick_start_trading()  # Initialize system")
    print("   >>> system.start_quick_trading(10)  # Start 10-min session")
    print()
    print("⚠️ NO AUTO-START - Use manual commands only")
    
    # Use Yahoo Finance for immediate start (no API key needed)
    from alternative_brokers import QuickPaperTradingSystem, AlternativeBrokerType
    system = QuickPaperTradingSystem(AlternativeBrokerType.YAHOO_FINANCE)
    return system

if __name__ == "__main__":
    quick_start_trading()in__":
    quick_start_trading()
