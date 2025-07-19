#!/usr/bin/env python3
"""Test chat command execution directly"""

import sys
sys.path.append('.')

from ai.conversational_ai import TradingConversationalAI

def test_command_detection():
    """Test command detection and execution"""
    ai = TradingConversationalAI()
    
    test_messages = [
        "Close all of my trades",
        "close all positions", 
        "liquidate everything",
        "check my portfolio",
        "buy 5 shares of AAPL"
    ]
    
    for msg in test_messages:
        print(f"\n🧪 Testing: '{msg}'")
        try:
            # Test command detection directly
            result = ai._detect_and_execute_command(msg)
            print(f"   Command result: {result}")
            
            # Test full response
            response = ai.get_response(msg)
            print(f"   Command executed: {response.get('command_executed', False)}")
            print(f"   Response preview: {response.get('response', 'No response')[:100]}...")
            
        except Exception as e:
            print(f"   Error: {e}")

if __name__ == "__main__":
    test_command_detection()