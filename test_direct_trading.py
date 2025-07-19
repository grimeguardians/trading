#!/usr/bin/env python3
"""Test direct trading functionality and chat integration"""

import sys
import requests
import json
sys.path.append('.')

from ai.conversational_ai import TradingConversationalAI

def test_direct_command_execution():
    """Test command execution directly"""
    print("🧪 Testing Direct Command Execution")
    ai = TradingConversationalAI()
    
    # Test 1: Direct buy command
    print("\n1. Testing buy command detection:")
    result = ai._detect_and_execute_command("buy 1 share of AAPL")
    print(f"   Result: {result}")
    
    # Test 2: Full AI response with command
    print("\n2. Testing full AI response:")
    response = ai.get_response("buy 1 share of AAPL")
    print(f"   Command executed: {response.get('command_executed', False)}")
    print(f"   Command result: {response.get('command_result', 'None')}")
    
def test_api_endpoint():
    """Test chat API endpoint"""
    print("\n🧪 Testing API Endpoint")
    
    try:
        response = requests.post(
            "http://localhost:8000/api/chat",
            json={"message": "buy 1 share of TSLA"},
            headers={"Content-Type": "application/json"}
        )
        data = response.json()
        print(f"   API Response Keys: {list(data.keys())}")
        print(f"   Command executed: {data.get('command_executed', False)}")
        print(f"   Command result: {data.get('command_result', 'None')}")
        
    except Exception as e:
        print(f"   API Error: {e}")

def test_alpaca_positions():
    """Check current Alpaca positions"""
    print("\n🧪 Testing Alpaca Positions")
    
    try:
        response = requests.get("http://localhost:8000/api/alpaca/positions")
        data = response.json()
        positions = data.get('positions', [])
        print(f"   Current positions: {len(positions)}")
        for pos in positions:
            print(f"     {pos['symbol']}: {pos['qty']} shares")
            
    except Exception as e:
        print(f"   Positions Error: {e}")

if __name__ == "__main__":
    test_direct_command_execution()
    test_api_endpoint() 
    test_alpaca_positions()