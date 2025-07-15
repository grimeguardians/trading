#!/usr/bin/env python3
"""Quick test of Alpaca connection and dashboard"""

import os
from dotenv import load_dotenv
load_dotenv()

try:
    from alpaca.trading.client import TradingClient
    
    # Test Alpaca connection
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    trading_client = TradingClient(
        api_key=api_key,
        secret_key=secret_key,
        paper=True
    )
    
    account = trading_client.get_account()
    print("✅ Alpaca Connection Success!")
    print(f"💰 Portfolio Value: ${float(account.portfolio_value):,.2f}")
    print(f"💵 Cash: ${float(account.cash):,.2f}")
    
    positions = trading_client.get_all_positions()
    print(f"📊 Positions: {len(positions)}")
    
    print("\n🚀 Your trading system is ready!")
    print("💡 Upload to Replit for web dashboard access")
    
except Exception as e:
    print(f"❌ Error: {e}")