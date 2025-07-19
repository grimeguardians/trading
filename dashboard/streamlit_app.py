"""
Streamlit Dashboard for AI Trading System
"""

import streamlit as st
import requests
import json
from datetime import datetime
import time

# Page config
st.set_page_config(
    page_title="AI Trading System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .status-healthy {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# API Base URL
import os
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

@st.cache_data(ttl=5)
def get_api_data(endpoint):
    """Get data from API endpoint with caching"""
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API server. Please check if the API server is running."}
    except requests.exceptions.Timeout:
        return {"error": "API request timed out. Please try again."}
    except Exception as e:
        return {"error": f"Connection Error: {str(e)}"}

def send_chat_message(message):
    """Send message to AI chat"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/chat",
            json={"message": message},
            timeout=15
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Chat Error: {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to AI chat service. Please check if the API server is running."}
    except requests.exceptions.Timeout:
        return {"error": "Chat request timed out. Please try again."}
    except Exception as e:
        return {"error": f"Chat Connection Error: {str(e)}"}

# Main header
st.markdown('<h1 class="main-header">🤖 AI Trading System Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.selectbox(
        "Choose a page:",
        ["Dashboard", "Portfolio", "Chat", "Strategies", "Market Data"]
    )
    
    st.markdown("---")
    
    # System Status
    st.subheader("System Status")
    try:
        health_data = get_api_data("/api/health")
        if "error" not in health_data:
            st.markdown('<span class="status-healthy">● System Online</span>', unsafe_allow_html=True)
            st.text(f"Version: {health_data.get('version', 'Unknown')}")
        else:
            st.markdown('<span class="status-warning">● System Offline</span>', unsafe_allow_html=True)
            st.warning(health_data["error"])
    except Exception as e:
        st.markdown('<span class="status-warning">● System Offline</span>', unsafe_allow_html=True)
        st.error(f"Status check failed: {str(e)}")

# Main content based on selected page
if page == "Dashboard":
    st.header("📊 Trading Dashboard")
    
    # Portfolio Overview
    portfolio_data = get_api_data("/api/portfolio")
    if "error" not in portfolio_data:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Value", f"${portfolio_data['total_value']:,.2f}")
        
        with col2:
            st.metric("Day Change", f"{portfolio_data['day_change']:.2f}%")
        
        with col3:
            st.metric("Day P&L", f"${portfolio_data['day_pnl']:,.2f}")
        
        with col4:
            st.metric("Positions", portfolio_data['positions_count'])
    
    # Recent Activity
    st.subheader("Recent Activity")
    st.info("No recent trading activity to display")
    
    # Quick Actions
    st.subheader("Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📈 View Portfolio"):
            st.success("Portfolio view selected")
    
    with col2:
        if st.button("💬 Chat with AI"):
            st.success("Chat activated")
    
    with col3:
        if st.button("⚙️ Strategies"):
            st.success("Strategies panel opened")

elif page == "Portfolio":
    st.header("💼 Portfolio Management")
    
    # Portfolio summary
    portfolio_data = get_api_data("/api/portfolio")
    if "error" not in portfolio_data:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Portfolio Summary")
            st.metric("Total Value", f"${portfolio_data['total_value']:,.2f}")
            st.metric("Cash Balance", f"${portfolio_data['cash_balance']:,.2f}")
            st.metric("Buying Power", f"${portfolio_data['buying_power']:,.2f}")
        
        with col2:
            st.subheader("Performance")
            st.metric("Day Change", f"{portfolio_data['day_change']:.2f}%")
            st.metric("Day P&L", f"${portfolio_data['day_pnl']:,.2f}")
            st.metric("Positions", portfolio_data['positions_count'])
    
    # Positions
    st.subheader("Current Positions")
    positions_data = get_api_data("/api/positions")
    if "error" not in positions_data and positions_data:
        for position in positions_data:
            with st.expander(f"{position['symbol']} - {position['quantity']} shares"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.text(f"Avg Price: ${position['avg_price']:.2f}")
                    st.text(f"Current Price: ${position['current_price']:.2f}")
                
                with col2:
                    st.text(f"P&L: ${position['pnl']:.2f}")
                    st.text(f"P&L %: {position['pnl_percent']:.2f}%")
                
                with col3:
                    if st.button(f"Sell {position['symbol']}", key=f"sell_{position['symbol']}"):
                        st.success(f"Sell order for {position['symbol']} submitted")

elif page == "Chat":
    st.header("💬 AI Trading Assistant")
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about your portfolio, market conditions, or trading strategies..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_data = send_chat_message(prompt)
                
                if "error" not in response_data:
                    response = response_data["response"]
                    st.write(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    st.error(response_data["error"])

elif page == "Strategies":
    st.header("⚙️ Trading Strategies")
    
    strategies_data = get_api_data("/api/strategies")
    if "error" not in strategies_data:
        for strategy in strategies_data["strategies"]:
            with st.expander(f"{strategy['name']} - {strategy['status'].upper()}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.text(f"Type: {strategy['type']}")
                    st.text(f"Status: {strategy['status']}")
                
                with col2:
                    st.text(f"Performance: {strategy['performance']:.1f}%")
                    st.text(f"Win Rate: {strategy['win_rate']:.1f}%")
                
                if strategy['status'] == 'active':
                    if st.button(f"Deactivate {strategy['name']}", key=f"deactivate_{strategy['name']}"):
                        st.success(f"{strategy['name']} deactivated")
                else:
                    if st.button(f"Activate {strategy['name']}", key=f"activate_{strategy['name']}"):
                        st.success(f"{strategy['name']} activated")

elif page == "Market Data":
    st.header("📈 Market Data")
    
    # Symbol input
    symbol = st.text_input("Enter symbol (e.g., AAPL, GOOGL, TSLA):", value="AAPL")
    
    if symbol:
        market_data = get_api_data(f"/api/market-data/{symbol}")
        if "error" not in market_data:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Price", f"${market_data['price']:.2f}")
            
            with col2:
                st.metric("Change", f"${market_data['change']:.2f}", f"{market_data['change_percent']:.2f}%")
            
            with col3:
                st.metric("Volume", f"{market_data['volume']:,}")
        else:
            st.error(market_data["error"])

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        AI Trading System v2.0 | Built with Streamlit & FastAPI
    </div>
    """,
    unsafe_allow_html=True
)