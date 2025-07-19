"""
Portfolio View Component for the Trading Dashboard
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np

class PortfolioView:
    """Component for displaying portfolio information and metrics"""
    
    def __init__(self):
        self.api_base_url = "http://localhost:8000"
    
    def render(self, portfolio_data: Dict = None):
        """Render the portfolio view component"""
        if not portfolio_data:
            portfolio_data = self._get_sample_portfolio_data()
        
        # Portfolio overview
        self._render_portfolio_overview(portfolio_data)
        
        # Portfolio charts
        self._render_portfolio_charts(portfolio_data)
        
        # Positions table
        self._render_positions_table(portfolio_data)
        
        # Performance metrics
        self._render_performance_metrics(portfolio_data)
    
    def _render_portfolio_overview(self, portfolio_data: Dict):
        """Render portfolio overview metrics"""
        st.markdown("### Portfolio Overview")
        
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            portfolio_value = portfolio_data.get('portfolio_value', 0)
            daily_change = portfolio_data.get('daily_change', 0)
            st.metric(
                "Portfolio Value",
                f"${portfolio_value:,.2f}",
                delta=f"{daily_change:+.2f}%"
            )
        
        with col2:
            available_balance = portfolio_data.get('available_balance', 0)
            st.metric(
                "Available Balance",
                f"${available_balance:,.2f}"
            )
        
        with col3:
            total_pnl = portfolio_data.get('total_pnl', 0)
            pnl_percentage = (total_pnl / portfolio_value * 100) if portfolio_value > 0 else 0
            st.metric(
                "Total P&L",
                f"${total_pnl:,.2f}",
                delta=f"{pnl_percentage:+.2f}%"
            )
        
        with col4:
            active_positions = portfolio_data.get('active_positions', 0)
            st.metric(
                "Active Positions",
                active_positions
            )
        
        with col5:
            day_pnl = portfolio_data.get('day_pnl', 0)
            st.metric(
                "Day P&L",
                f"${day_pnl:,.2f}",
                delta=f"{day_pnl:+.2f}"
            )
    
    def _render_portfolio_charts(self, portfolio_data: Dict):
        """Render portfolio charts"""
        st.markdown("### Portfolio Charts")
        
        # Create tabs for different chart types
        tab1, tab2, tab3 = st.tabs(["Performance", "Allocation", "Risk"])
        
        with tab1:
            self._render_performance_chart(portfolio_data)
        
        with tab2:
            self._render_allocation_charts(portfolio_data)
        
        with tab3:
            self._render_risk_charts(portfolio_data)
    
    def _render_performance_chart(self, portfolio_data: Dict):
        """Render portfolio performance chart"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Portfolio Value Over Time")
            
            # Generate sample performance data
            dates = pd.date_range(start='2024-01-01', end=datetime.now(), freq='D')
            portfolio_values = self._generate_portfolio_performance(len(dates))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=portfolio_values,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#1f77b4', width=2),
                fill='tonexty'
            ))
            
            # Add benchmark
            benchmark_values = self._generate_benchmark_performance(len(dates))
            fig.add_trace(go.Scatter(
                x=dates,
                y=benchmark_values,
                mode='lines',
                name='S&P 500',
                line=dict(color='#ff7f0e', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title="Portfolio vs Benchmark Performance",
                xaxis_title="Date",
                yaxis_title="Value ($)",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Daily P&L Distribution")
            
            # Generate daily P&L data
            daily_pnl = np.random.normal(50, 200, 100)  # Sample data
            
            fig = go.Figure(data=[go.Histogram(
                x=daily_pnl,
                nbinsx=20,
                marker_color='#1f77b4',
                opacity=0.7
            )])
            
            fig.update_layout(
                title="Daily P&L Distribution",
                xaxis_title="Daily P&L ($)",
                yaxis_title="Frequency",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_allocation_charts(self, portfolio_data: Dict):
        """Render portfolio allocation charts"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Asset Allocation")
            
            # Sample allocation data
            allocation_data = {
                'Asset': ['Stocks', 'Options', 'Cash', 'Crypto'],
                'Value': [75000, 15000, 8000, 2000],
                'Percentage': [75, 15, 8, 2]
            }
            
            fig = px.pie(
                allocation_data,
                values='Value',
                names='Asset',
                title="Portfolio Allocation by Asset Class"
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Sector Allocation")
            
            # Sample sector data
            sector_data = {
                'Sector': ['Technology', 'Healthcare', 'Finance', 'Energy', 'Consumer'],
                'Value': [30000, 20000, 15000, 8000, 7000],
                'Percentage': [37.5, 25, 18.75, 10, 8.75]
            }
            
            fig = go.Figure(data=[go.Bar(
                x=sector_data['Sector'],
                y=sector_data['Value'],
                marker_color='#1f77b4'
            )])
            
            fig.update_layout(
                title="Portfolio Allocation by Sector",
                xaxis_title="Sector",
                yaxis_title="Value ($)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_risk_charts(self, portfolio_data: Dict):
        """Render risk analysis charts"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Risk Metrics Over Time")
            
            # Generate risk metrics data
            dates = pd.date_range(start='2024-01-01', end=datetime.now(), freq='D')
            var_95 = np.random.normal(-2000, 500, len(dates))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=var_95,
                mode='lines',
                name='VaR (95%)',
                line=dict(color='#ff7f0e', width=2)
            ))
            
            fig.update_layout(
                title="Value at Risk (95% Confidence)",
                xaxis_title="Date",
                yaxis_title="VaR ($)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Correlation Heatmap")
            
            # Sample correlation data
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
            correlation_matrix = np.random.rand(5, 5)
            correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
            np.fill_diagonal(correlation_matrix, 1)
            
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix,
                x=symbols,
                y=symbols,
                colorscale='RdBu',
                zmid=0
            ))
            
            fig.update_layout(
                title="Position Correlation Matrix",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_positions_table(self, portfolio_data: Dict):
        """Render positions table"""
        st.markdown("### Current Positions")
        
        positions = portfolio_data.get('positions', [])
        
        if positions:
            # Convert to DataFrame
            positions_df = pd.DataFrame(positions)
            
            # Format columns
            if not positions_df.empty:
                positions_df['entry_price'] = positions_df['entry_price'].apply(lambda x: f"${x:.2f}")
                positions_df['current_price'] = positions_df['current_price'].apply(lambda x: f"${x:.2f}")
                positions_df['unrealized_pnl'] = positions_df['unrealized_pnl'].apply(lambda x: f"${x:,.2f}")
                positions_df['pnl_percentage'] = positions_df['pnl_percentage'].apply(lambda x: f"{x:.2f}%")
                
                # Color coding for P&L
                def color_pnl(val):
                    if val.startswith('$-'):
                        return 'color: red'
                    elif val.startswith('$') and not val.startswith('$-') and not val.startswith('$0'):
                        return 'color: green'
                    return ''
                
                styled_df = positions_df.style.applymap(color_pnl, subset=['unrealized_pnl'])
                
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    column_config={
                        "symbol": st.column_config.TextColumn("Symbol"),
                        "exchange": st.column_config.TextColumn("Exchange"),
                        "side": st.column_config.TextColumn("Side"),
                        "quantity": st.column_config.NumberColumn("Quantity"),
                        "entry_price": st.column_config.TextColumn("Entry Price"),
                        "current_price": st.column_config.TextColumn("Current Price"),
                        "unrealized_pnl": st.column_config.TextColumn("Unrealized P&L"),
                        "pnl_percentage": st.column_config.TextColumn("P&L %")
                    }
                )
                
                # Position actions
                st.markdown("#### Position Actions")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    selected_symbol = st.selectbox("Select Position", positions_df['symbol'].tolist())
                
                with col2:
                    if st.button("Close Position"):
                        st.warning(f"Close position for {selected_symbol}")
                
                with col3:
                    if st.button("Modify Stop Loss"):
                        st.info(f"Modify stop loss for {selected_symbol}")
        else:
            st.info("No active positions")
    
    def _render_performance_metrics(self, portfolio_data: Dict):
        """Render detailed performance metrics"""
        st.markdown("### Performance Metrics")
        
        # Create tabs for different metric categories
        tab1, tab2, tab3 = st.tabs(["Returns", "Risk", "Ratios"])
        
        with tab1:
            self._render_return_metrics(portfolio_data)
        
        with tab2:
            self._render_risk_metrics(portfolio_data)
        
        with tab3:
            self._render_ratio_metrics(portfolio_data)
    
    def _render_return_metrics(self, portfolio_data: Dict):
        """Render return metrics"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Daily Return", "0.25%", delta="0.15%")
            st.metric("Weekly Return", "1.2%", delta="0.8%")
            st.metric("Monthly Return", "3.5%", delta="2.1%")
        
        with col2:
            st.metric("YTD Return", "15.2%", delta="8.5%")
            st.metric("1Y Return", "18.7%", delta="12.3%")
            st.metric("3Y Return", "45.6%", delta="32.1%")
        
        with col3:
            st.metric("Best Day", "2.8%")
            st.metric("Worst Day", "-1.9%")
            st.metric("Avg Daily Return", "0.12%")
    
    def _render_risk_metrics(self, portfolio_data: Dict):
        """Render risk metrics"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Max Drawdown", "5.2%", delta="-0.3%")
            st.metric("Current Drawdown", "1.1%", delta="-0.5%")
            st.metric("Volatility", "12.8%", delta="1.2%")
        
        with col2:
            st.metric("VaR (95%)", "$2,450", delta="-$120")
            st.metric("Expected Shortfall", "$3,200", delta="-$180")
            st.metric("Beta", "0.85", delta="-0.05")
        
        with col3:
            st.metric("Correlation to S&P", "0.72", delta="0.08")
            st.metric("Tracking Error", "8.5%", delta="0.7%")
            st.metric("R-Squared", "0.68", delta="0.12")
    
    def _render_ratio_metrics(self, portfolio_data: Dict):
        """Render ratio metrics"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Sharpe Ratio", "1.85", delta="0.12")
            st.metric("Sortino Ratio", "2.34", delta="0.18")
            st.metric("Calmar Ratio", "3.56", delta="0.25")
        
        with col2:
            st.metric("Information Ratio", "0.92", delta="0.15")
            st.metric("Treynor Ratio", "0.22", delta="0.03")
            st.metric("Alpha", "2.3%", delta="0.5%")
        
        with col3:
            st.metric("Win Rate", "65%", delta="3%")
            st.metric("Profit Factor", "1.8", delta="0.2")
            st.metric("Recovery Factor", "4.2", delta="0.8")
    
    def _generate_portfolio_performance(self, days: int) -> List[float]:
        """Generate sample portfolio performance data"""
        np.random.seed(42)  # For reproducible results
        returns = np.random.normal(0.001, 0.02, days)  # Daily returns
        prices = [100000]  # Starting value
        
        for i in range(1, days):
            prices.append(prices[i-1] * (1 + returns[i]))
        
        return prices
    
    def _generate_benchmark_performance(self, days: int) -> List[float]:
        """Generate sample benchmark performance data"""
        np.random.seed(123)  # Different seed for benchmark
        returns = np.random.normal(0.0008, 0.015, days)  # Slightly different characteristics
        prices = [100000]  # Starting value
        
        for i in range(1, days):
            prices.append(prices[i-1] * (1 + returns[i]))
        
        return prices
    
    def _get_sample_portfolio_data(self) -> Dict:
        """Get sample portfolio data for demonstration"""
        return {
            "portfolio_value": 105250.00,
            "available_balance": 25000.00,
            "total_pnl": 5250.00,
            "daily_change": 0.25,
            "active_positions": 8,
            "day_pnl": 125.50,
            "positions": [
                {
                    "symbol": "AAPL",
                    "exchange": "alpaca",
                    "side": "long",
                    "quantity": 100,
                    "entry_price": 150.25,
                    "current_price": 152.80,
                    "unrealized_pnl": 255.00,
                    "pnl_percentage": 1.70
                },
                {
                    "symbol": "GOOGL",
                    "exchange": "alpaca",
                    "side": "long",
                    "quantity": 50,
                    "entry_price": 2650.00,
                    "current_price": 2625.00,
                    "unrealized_pnl": -1250.00,
                    "pnl_percentage": -0.94
                },
                {
                    "symbol": "MSFT",
                    "exchange": "alpaca",
                    "side": "long",
                    "quantity": 75,
                    "entry_price": 380.50,
                    "current_price": 385.20,
                    "unrealized_pnl": 352.50,
                    "pnl_percentage": 1.24
                },
                {
                    "symbol": "TSLA",
                    "exchange": "alpaca",
                    "side": "short",
                    "quantity": 25,
                    "entry_price": 220.75,
                    "current_price": 215.30,
                    "unrealized_pnl": 136.25,
                    "pnl_percentage": 2.47
                }
            ]
        }
