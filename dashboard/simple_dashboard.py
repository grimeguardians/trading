"""
Simple Trading Dashboard - Minimal version without WebSocket issues
"""

import streamlit as st
import requests
from datetime import datetime
import time

# Minimal page config
st.set_page_config(
    page_title="AI Trading System",
    page_icon="🤖",
    layout="wide"
)

# API Base URL
API_BASE_URL = "http://localhost:8000"

# Simple API call function
def api_call(endpoint, method="GET", data=None):
    """Simple API call with error handling"""
    try:
        if method == "GET":
            response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=5)
        elif method == "POST":
            response = requests.post(f"{API_BASE_URL}{endpoint}", json=data, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API returned status {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API server"}
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}

# Main header
st.title("🤖 AI Trading System")
st.markdown("---")

# Status check
col1, col2 = st.columns([3, 1])

with col1:
    st.header("System Status")
    
with col2:
    if st.button("🔄 Refresh"):
        st.rerun()

# Check API health
health_data = api_call("/api/health")
if "error" not in health_data:
    st.success("✅ System is online and operational")
    st.json(health_data)
else:
    st.error(f"❌ System offline: {health_data['error']}")

st.markdown("---")

# Navigation tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Portfolio", "💬 Chat", "🎯 Trading", "⚙️ Strategies", "📈 Market"])

with tab1:
    st.header("Portfolio Overview")
    
    # Get portfolio data
    portfolio_data = api_call("/api/portfolio")
    if "error" not in portfolio_data:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Value", f"${portfolio_data.get('total_value', 0):,.2f}")
        
        with col2:
            st.metric("Day Change", f"{portfolio_data.get('day_change', 0):.2f}%")
        
        with col3:
            st.metric("Day P&L", f"${portfolio_data.get('day_pnl', 0):,.2f}")
        
        with col4:
            st.metric("Positions", portfolio_data.get('positions_count', 0))
        
        st.subheader("Current Positions")
        positions_data = api_call("/api/positions")
        if "error" not in positions_data and positions_data:
            for position in positions_data:
                st.write(f"**{position['symbol']}** - {position['quantity']} shares")
                st.write(f"P&L: ${position['pnl']:.2f} ({position['pnl_percent']:.2f}%)")
                st.write("---")
    else:
        st.error(f"Failed to load portfolio: {portfolio_data['error']}")

with tab2:
    st.header("💬 AI Trading Assistant")
    
    # Chat interface with conversation history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about trading, markets, or your portfolio..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chat_response = api_call("/api/chat", "POST", {"message": prompt})
                
                if "error" not in chat_response:
                    response = chat_response.get("response", "No response received")
                    model = chat_response.get("model", "unknown")
                    
                    st.write(response)
                    
                    # Show model info
                    if model == "gpt-4o":
                        st.caption("🤖 Powered by GPT-4o")
                    elif model == "intelligent_fallback":
                        st.caption("⚠️ Using intelligent fallback responses (OpenAI API not available)")
                    elif model == "fallback":
                        st.caption("⚠️ Using fallback responses (OpenAI API not available)")
                    else:
                        st.caption(f"🤖 Model: {model}")
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    st.error(f"Chat failed: {chat_response['error']}")
    
    # Chat controls
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            # Also clear server-side history
            clear_response = api_call("/api/chat/clear", "POST", {})
            if "error" not in clear_response:
                st.success("Chat history cleared!")
            else:
                st.warning("Local chat cleared, but server-side clear failed")
    
    with col2:
        if st.button("💡 Chat Tips"):
            st.info("""
            **You can ask me about:**
            - Portfolio analysis and recommendations
            - Market insights and trends
            - Trading strategies and risk management
            - Technical analysis and indicators
            - Specific stocks, crypto, or other assets
            - Investment education and concepts
            
            **Examples:**
            - "What's your opinion on NVDA?"
            - "How should I diversify my portfolio?"
            - "Explain swing trading strategies"
            - "What are the risks of crypto trading?"
            """)
    
    # Show conversation stats
    if st.session_state.messages:
        st.caption(f"💬 {len(st.session_state.messages)} messages in this conversation")

with tab3:
    # Alpaca Paper Trading Interface
    try:
        from dashboard.trading_interface import show_alpaca_trading_interface, show_setup_instructions
        
        # Check if Alpaca credentials are configured
        alpaca_status = api_call("/api/alpaca/account")
        
        if "error" in alpaca_status and "credentials not configured" in alpaca_status['error'].lower():
            show_setup_instructions()
        else:
            show_alpaca_trading_interface()
            
    except ImportError:
        st.error("Trading interface not available")
    except Exception as e:
        st.error(f"Trading interface error: {str(e)}")

with tab4:
    st.header("Trading Strategies")
    
    strategies_data = api_call("/api/strategies")
    if "error" not in strategies_data:
        for strategy in strategies_data.get("strategies", []):
            st.subheader(f"{strategy['name']}")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Type:** {strategy['type']}")
                st.write(f"**Status:** {strategy['status']}")
            
            with col2:
                st.write(f"**Performance:** {strategy['performance']:.1f}%")
                st.write(f"**Win Rate:** {strategy['win_rate']:.1f}%")
            
            st.write("---")
    else:
        st.error(f"Failed to load strategies: {strategies_data['error']}")

with tab5:
    st.header("Market Data")
    
    # Data sources info
    st.subheader("📊 Data Sources")
    sources_data = api_call("/api/data-sources")
    if "error" not in sources_data:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Available Sources", sources_data.get("total_sources", 0))
        
        with col2:
            st.metric("Active Sources", sources_data.get("active_sources", 0))
        
        # Show source status
        st.subheader("Data Source Status")
        for source in sources_data.get("available_sources", []):
            status_color = "🟢" if source["status"] == "active" else "🔴" if source["status"] == "needs_key" else "🟡"
            st.write(f"{status_color} **{source['name']}** - {source['description']}")
    
    st.markdown("---")
    
    # Market data lookup
    st.subheader("📈 Universal Asset Lookup")
    symbol = st.text_input("Enter symbol (e.g., AAPL, BTC, EUR/USD, SPY, GLD):", value="AAPL")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Get Quote", use_container_width=True):
            if symbol:
                with st.spinner("Fetching real-time market data..."):
                    market_data = api_call(f"/api/market-data/{symbol}")
                    if "error" not in market_data:
                        # Main metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Price", f"${market_data.get('price', 0):.2f}")
                        
                        with col2:
                            change = market_data.get('change', 0)
                            change_pct = market_data.get('change_percent', 0)
                            delta_color = "normal" if change >= 0 else "inverse"
                            st.metric("Change", f"${change:.2f}", f"{change_pct:.2f}%", delta_color=delta_color)
                        
                        with col3:
                            st.metric("Volume", f"{market_data.get('volume', 0):,}")
                        
                        # Additional market data
                        st.subheader("📋 Additional Details")
                        col4, col5, col6 = st.columns(3)
                        
                        with col4:
                            st.write(f"**Asset Type:** {market_data.get('asset_type', 'unknown').title()}")
                            st.write(f"**Open:** ${market_data.get('open', 0):.2f}")
                            st.write(f"**High:** ${market_data.get('high', 0):.2f}")
                        
                        with col5:
                            st.write(f"**Low:** ${market_data.get('low', 0):.2f}")
                            st.write(f"**Previous Close:** ${market_data.get('previous_close', 0):.2f}")
                            st.write(f"**Currency:** {market_data.get('currency', 'USD')}")
                        
                        with col6:
                            st.write(f"**Exchange:** {market_data.get('exchange', 'Unknown').title()}")
                            st.write(f"**Data Source:** {market_data.get('source', 'Unknown').title()}")
                            st.write(f"**Last Updated:** {market_data.get('timestamp', 'Unknown')[:19]}")
                        
                        # Market cap for stocks/crypto
                        if market_data.get('market_cap'):
                            st.write(f"**Market Cap:** ${market_data.get('market_cap', 0):,.0f}")
                        
                        # Success message with asset type
                        asset_type = market_data.get('asset_type', 'asset')
                        st.success(f"✅ Real-time {asset_type} data for {symbol} retrieved successfully!")
                        
                    else:
                        st.error(f"❌ Failed to get market data: {market_data['error']}")
            else:
                st.warning("Please enter a stock symbol")
    
    with col2:
        if st.button("📈 Market Status", use_container_width=True):
            with st.spinner("Checking market status..."):
                status_data = api_call("/api/market-status")
                if "error" not in status_data:
                    market_state = status_data.get("market_state", "unknown")
                    is_open = status_data.get("is_open", False)
                    
                    if is_open:
                        st.success(f"🟢 Market is OPEN ({market_state})")
                    else:
                        st.info(f"🔴 Market is CLOSED ({market_state})")
                    
                    st.write(f"**Timezone:** {status_data.get('timezone', 'Unknown')}")
                    st.write(f"**Check Time:** {status_data.get('timestamp', 'Unknown')[:19]}")
                else:
                    st.error("Failed to get market status")
    
    # Popular assets by category
    st.subheader("🚀 Popular Assets")
    
    # Get popular assets from API
    popular_assets = api_call("/api/popular-assets")
    if "error" not in popular_assets:
        categories = popular_assets.get("categories", {})
        
        # Create tabs for different asset categories
        asset_tabs = st.tabs(["📈 Stocks", "🪙 Crypto", "📊 ETFs", "💱 Forex", "🏗️ Commodities"])
        
        # Stocks tab
        with asset_tabs[0]:
            stocks = categories.get("stocks", [])
            cols = st.columns(4)
            for i, stock in enumerate(stocks):
                with cols[i % 4]:
                    if st.button(stock, key=f"stock_{stock}"):
                        with st.spinner(f"Getting {stock} data..."):
                            market_data = api_call(f"/api/market-data/{stock}")
                            if "error" not in market_data:
                                price = market_data.get('price', 0)
                                change_pct = market_data.get('change_percent', 0)
                                asset_type = market_data.get('asset_type', 'stock')
                                change_color = "🟢" if change_pct >= 0 else "🔴"
                                st.write(f"{change_color} **{stock}**: ${price:.2f} ({change_pct:+.2f}%)")
                                st.write(f"*{asset_type.title()}*")
                            else:
                                st.error(f"Failed to get {stock} data")
        
        # Crypto tab
        with asset_tabs[1]:
            cryptos = categories.get("crypto", [])
            cols = st.columns(4)
            for i, crypto in enumerate(cryptos):
                with cols[i % 4]:
                    if st.button(crypto, key=f"crypto_{crypto}"):
                        with st.spinner(f"Getting {crypto} data..."):
                            market_data = api_call(f"/api/market-data/{crypto}")
                            if "error" not in market_data:
                                price = market_data.get('price', 0)
                                change_pct = market_data.get('change_percent', 0)
                                asset_type = market_data.get('asset_type', 'crypto')
                                change_color = "🟢" if change_pct >= 0 else "🔴"
                                st.write(f"{change_color} **{crypto}**: ${price:.2f} ({change_pct:+.2f}%)")
                                st.write(f"*{asset_type.title()}*")
                            else:
                                st.error(f"Failed to get {crypto} data")
        
        # ETFs tab
        with asset_tabs[2]:
            etfs = categories.get("etfs", [])
            cols = st.columns(4)
            for i, etf in enumerate(etfs):
                with cols[i % 4]:
                    if st.button(etf, key=f"etf_{etf}"):
                        with st.spinner(f"Getting {etf} data..."):
                            market_data = api_call(f"/api/market-data/{etf}")
                            if "error" not in market_data:
                                price = market_data.get('price', 0)
                                change_pct = market_data.get('change_percent', 0)
                                asset_type = market_data.get('asset_type', 'etf')
                                change_color = "🟢" if change_pct >= 0 else "🔴"
                                st.write(f"{change_color} **{etf}**: ${price:.2f} ({change_pct:+.2f}%)")
                                st.write(f"*{asset_type.title()}*")
                            else:
                                st.error(f"Failed to get {etf} data")
        
        # Forex tab
        with asset_tabs[3]:
            forex = categories.get("forex", [])
            cols = st.columns(4)
            for i, fx in enumerate(forex):
                with cols[i % 4]:
                    if st.button(fx, key=f"forex_{fx}"):
                        with st.spinner(f"Getting {fx} data..."):
                            market_data = api_call(f"/api/market-data/{fx}")
                            if "error" not in market_data:
                                price = market_data.get('price', 0)
                                change_pct = market_data.get('change_percent', 0)
                                asset_type = market_data.get('asset_type', 'forex')
                                change_color = "🟢" if change_pct >= 0 else "🔴"
                                st.write(f"{change_color} **{fx}**: {price:.4f} ({change_pct:+.2f}%)")
                                st.write(f"*{asset_type.title()}*")
                            else:
                                st.error(f"Failed to get {fx} data")
        
        # Commodities tab
        with asset_tabs[4]:
            commodities = categories.get("commodities", [])
            cols = st.columns(4)
            for i, commodity in enumerate(commodities):
                with cols[i % 4]:
                    if st.button(commodity, key=f"commodity_{commodity}"):
                        with st.spinner(f"Getting {commodity} data..."):
                            market_data = api_call(f"/api/market-data/{commodity}")
                            if "error" not in market_data:
                                price = market_data.get('price', 0)
                                change_pct = market_data.get('change_percent', 0)
                                asset_type = market_data.get('asset_type', 'commodity')
                                change_color = "🟢" if change_pct >= 0 else "🔴"
                                st.write(f"{change_color} **{commodity}**: ${price:.2f} ({change_pct:+.2f}%)")
                                st.write(f"*{asset_type.title()}*")
                            else:
                                st.error(f"Failed to get {commodity} data")
    else:
        # Fallback to static list
        st.write("Using default popular assets...")
        popular_stocks = ["AAPL", "GOOGL", "MSFT", "TSLA", "BTC", "ETH", "SPY", "QQQ"]
        cols = st.columns(4)
        for i, stock in enumerate(popular_stocks):
            with cols[i % 4]:
                if st.button(stock, key=f"fallback_{stock}"):
                    with st.spinner(f"Getting {stock} data..."):
                        market_data = api_call(f"/api/market-data/{stock}")
                        if "error" not in market_data:
                            price = market_data.get('price', 0)
                            change_pct = market_data.get('change_percent', 0)
                            change_color = "🟢" if change_pct >= 0 else "🔴"
                            st.write(f"{change_color} **{stock}**: ${price:.2f} ({change_pct:+.2f}%)")
                        else:
                            st.error(f"Failed to get {stock} data")

# Footer
st.markdown("---")
st.markdown("**AI Trading System v2.0** - Simple Dashboard Mode")
st.markdown("This is a simplified version to avoid WebSocket connection issues.")