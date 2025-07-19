"""
Alpaca Paper Trading Interface for Dashboard
"""

import streamlit as st
import requests
from datetime import datetime
import time

API_BASE_URL = "http://localhost:8000"

def api_call(endpoint, method="GET", data=None):
    """Simple API call with error handling"""
    try:
        if method == "GET":
            response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=10)
        elif method == "POST":
            response = requests.post(f"{API_BASE_URL}{endpoint}", json=data, timeout=10)
        elif method == "DELETE":
            response = requests.delete(f"{API_BASE_URL}{endpoint}", timeout=10)
        
        if response.status_code in [200, 201, 204]:
            if response.content:
                return response.json()
            else:
                return {"status": "success"}
        else:
            return {"error": f"API returned status {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API server"}
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}

def show_alpaca_trading_interface():
    """Show Alpaca paper trading interface"""
    st.header("🎯 Alpaca Paper Trading")
    
    # Account info section
    st.subheader("📊 Account Overview")
    
    if st.button("🔄 Refresh Account", use_container_width=True):
        with st.spinner("Getting account information..."):
            account_data = api_call("/api/alpaca/account")
            
            if "error" not in account_data:
                if account_data.get("status") == "success":
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Portfolio Value", f"${account_data.get('portfolio_value', 0):,.2f}")
                    
                    with col2:
                        st.metric("Buying Power", f"${account_data.get('buying_power', 0):,.2f}")
                    
                    with col3:
                        st.metric("Cash", f"${account_data.get('cash', 0):,.2f}")
                    
                    with col4:
                        st.metric("Day Trades", account_data.get('day_trade_count', 0))
                    
                    # Account status
                    if account_data.get('trading_blocked'):
                        st.error("⚠️ Trading is currently blocked on this account")
                    elif account_data.get('pattern_day_trader'):
                        st.warning("📊 Account is flagged as Pattern Day Trader")
                    else:
                        st.success("✅ Account is active and ready for trading")
                else:
                    st.error(f"❌ {account_data.get('error', 'Unknown error')}")
            else:
                st.error(f"❌ Account Error: {account_data['error']}")
    
    st.markdown("---")
    
    # Trading section
    st.subheader("📈 Place Trade")
    
    col1, col2 = st.columns(2)
    
    with col1:
        symbol = st.text_input("Symbol", value="AAPL", help="Enter stock symbol (e.g., AAPL, TSLA, NVDA)")
        quantity = st.number_input("Quantity", min_value=1, value=1, help="Number of shares")
        side = st.selectbox("Side", ["buy", "sell"])
    
    with col2:
        order_type = st.selectbox("Order Type", ["market", "limit"])
        limit_price = None
        if order_type == "limit":
            limit_price = st.number_input("Limit Price", min_value=0.01, value=100.00, step=0.01)
        
        time_in_force = st.selectbox("Time in Force", ["day", "gtc", "ioc", "fok"])
    
    # Trade execution
    if st.button(f"🎯 {side.upper()} {quantity} shares of {symbol}", use_container_width=True, type="primary"):
        if symbol:
            with st.spinner(f"Placing {side} order for {quantity} shares of {symbol}..."):
                trade_data = {
                    "symbol": symbol.upper(),
                    "quantity": quantity,
                    "side": side,
                    "order_type": order_type,
                    "price": limit_price
                }
                
                result = api_call("/api/alpaca/trade", "POST", trade_data)
                
                if "error" not in result:
                    if result.get("status") == "success":
                        st.success(f"✅ {result.get('message', 'Order placed successfully!')}")
                        
                        # Show order details in a more user-friendly way
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Order ID", result.get('order_id', 'N/A')[:8] + "...")
                        with col2:
                            st.metric("Status", result.get('order_status', 'Unknown').title())
                        with col3:
                            st.metric("Filled", f"{result.get('filled_qty', 0)}/{result.get('qty', 0)}")
                        
                        # Show full details in expander
                        with st.expander("📋 Full Order Details"):
                            st.json(result.get('alpaca_response', result))
                    else:
                        st.error(f"❌ Trade failed: {result.get('error', 'Unknown error')}")
                else:
                    st.error(f"❌ API Error: {result['error']}")
        else:
            st.warning("Please enter a symbol")
    
    st.markdown("---")
    
    # Positions section
    st.subheader("💼 Current Positions")
    
    if st.button("📊 Refresh Positions", use_container_width=True):
        with st.spinner("Getting positions..."):
            positions_data = api_call("/api/alpaca/positions")
            
            if "error" not in positions_data:
                if positions_data.get("status") == "success":
                    positions = positions_data.get("positions", [])
                    
                    if positions:
                        for pos in positions:
                            col1, col2, col3, col4, col5 = st.columns(5)
                            
                            with col1:
                                st.write(f"**{pos['symbol']}**")
                            
                            with col2:
                                qty_color = "🟢" if float(pos['qty']) > 0 else "🔴"
                                st.write(f"{qty_color} {pos['qty']} shares")
                            
                            with col3:
                                st.write(f"${pos['current_price']:.2f}")
                            
                            with col4:
                                st.write(f"${pos['market_value']:.2f}")
                            
                            with col5:
                                pnl_color = "🟢" if float(pos['unrealized_pl']) >= 0 else "🔴"
                                st.write(f"{pnl_color} ${pos['unrealized_pl']:.2f} ({pos['unrealized_plpc']:.2f}%)")
                            
                            st.write("---")
                    else:
                        st.info("📭 No positions found")
                else:
                    st.error(f"❌ {positions_data.get('error', 'Unknown error')}")
            else:
                st.error(f"❌ Positions Error: {positions_data['error']}")
    
    st.markdown("---")
    
    # Orders section
    st.subheader("📋 Orders")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Open Orders", use_container_width=True):
            with st.spinner("Getting open orders..."):
                orders_data = api_call("/api/alpaca/orders?status=open")
                show_orders(orders_data, "Open")
    
    with col2:
        if st.button("📋 Recent Orders", use_container_width=True):
            with st.spinner("Getting recent orders..."):
                orders_data = api_call("/api/alpaca/orders?status=filled")
                show_orders(orders_data, "Recent")

def show_orders(orders_data, title):
    """Display orders data"""
    if "error" not in orders_data:
        if orders_data.get("status") == "success":
            orders = orders_data.get("orders", [])
            
            if orders:
                st.write(f"**{title} Orders:**")
                for order in orders:
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.write(f"**{order['symbol']}**")
                    
                    with col2:
                        side_color = "🟢" if order['side'] == 'buy' else "🔴"
                        st.write(f"{side_color} {order['side'].upper()}")
                    
                    with col3:
                        st.write(f"{order['qty']} shares")
                    
                    with col4:
                        st.write(f"{order['status'].title()}")
                    
                    with col5:
                        if order['status'] == 'pending_new' or order['status'] == 'new':
                            if st.button("❌", key=f"cancel_{order['id']}", help="Cancel order"):
                                cancel_result = api_call(f"/api/alpaca/orders/{order['id']}", "DELETE")
                                if "error" not in cancel_result:
                                    st.success("Order cancelled!")
                                    st.rerun()
                                else:
                                    st.error(f"Cancel failed: {cancel_result['error']}")
                    
                    st.write("---")
            else:
                st.info(f"📭 No {title.lower()} orders found")
        else:
            st.error(f"❌ {orders_data.get('error', 'Unknown error')}")
    else:
        st.error(f"❌ Orders Error: {orders_data['error']}")

def show_setup_instructions():
    """Show Alpaca setup instructions"""
    st.header("⚙️ Alpaca Setup")
    
    st.markdown("""
    ### 🚀 Get Started with Alpaca Paper Trading
    
    To connect your Alpaca paper trading account:
    
    1. **Create Alpaca Account**:
       - Go to [alpaca.markets](https://alpaca.markets)
       - Sign up for a free account
       - Verify your email and complete registration
    
    2. **Get API Keys**:
       - Login to your Alpaca dashboard
       - Go to "Paper Trading" section
       - Navigate to "API Keys" or "Your API Keys"
       - Generate new API key pair
       - Copy both the "Key ID" and "Secret Key"
    
    3. **Add Keys to Replit**:
       - In this Replit project, go to Secrets (🔒 lock icon)
       - Add secret: `ALPACA_API_KEY` = your Key ID
       - Add secret: `ALPACA_SECRET_KEY` = your Secret Key
       - Save the secrets
    
    4. **Test Connection**:
       - Once keys are added, refresh this page
       - Click "Refresh Account" to test the connection
       - You should see your paper trading account balance
    
    ### 📋 Features Available:
    - ✅ Real-time account information
    - ✅ Portfolio positions and P&L
    - ✅ Place market and limit orders
    - ✅ View and cancel pending orders
    - ✅ Track order history
    - ✅ Paper trading with $100,000 virtual money
    
    ### 💡 Trading Tips:
    - Start with small quantities to test the system
    - Use limit orders for better price control
    - Monitor your day trade count (PDT rules apply)
    - Paper trading is risk-free - perfect for learning!
    """)
    
    if st.button("🔄 Test Connection Now", use_container_width=True):
        with st.spinner("Testing Alpaca connection..."):
            account_data = api_call("/api/alpaca/account")
            
            if "error" not in account_data:
                if account_data.get("status") == "success":
                    st.success("✅ Alpaca connection successful!")
                    st.json(account_data)
                else:
                    st.error(f"❌ Connection failed: {account_data.get('error')}")
            else:
                st.error(f"❌ API Error: {account_data['error']}")
                if "credentials not configured" in account_data['error'].lower():
                    st.info("💡 Please add your Alpaca API keys to Replit Secrets as described above.")