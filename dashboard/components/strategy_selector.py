"""
Strategy Selector Component for the Trading Dashboard
"""

import streamlit as st
from typing import Dict, List, Optional
import requests

class StrategySelector:
    """Component for selecting and managing trading strategies"""
    
    def __init__(self):
        self.api_base_url = "http://localhost:8000"
        self.strategy_categories = {
            "swing": {
                "name": "Swing Trading",
                "description": "Medium-term trend following strategy (3-7 days)",
                "suitable_for": ["Trending markets", "Medium volatility"],
                "risk_level": "Medium",
                "icon": "📈"
            },
            "scalping": {
                "name": "Scalping",
                "description": "High-frequency short-term trading (1-5 minutes)",
                "suitable_for": ["High-volume instruments", "Tight spreads"],
                "risk_level": "High",
                "icon": "⚡"
            },
            "options": {
                "name": "Options Trading",
                "description": "Complex derivatives trading strategies",
                "suitable_for": ["Experienced traders", "Volatility trading"],
                "risk_level": "High",
                "icon": "🎯"
            },
            "intraday": {
                "name": "Intraday Trading",
                "description": "Same-day trading with market open/close focus",
                "suitable_for": ["Active traders", "Volatile markets"],
                "risk_level": "Medium-High",
                "icon": "⏰"
            }
        }
    
    def render(self):
        """Render the strategy selector component"""
        st.markdown("### Select Trading Strategy")
        
        # Create columns for strategy cards
        cols = st.columns(2)
        
        for i, (category, info) in enumerate(self.strategy_categories.items()):
            with cols[i % 2]:
                self._render_strategy_card(category, info)
        
        # Strategy configuration
        self._render_strategy_configuration()
    
    def _render_strategy_card(self, category: str, info: Dict):
        """Render a strategy card"""
        with st.container():
            st.markdown(f"""
            <div style="
                border: 1px solid #ddd;
                border-radius: 10px;
                padding: 1rem;
                margin: 0.5rem 0;
                background: #f8f9fa;
            ">
                <h4>{info['icon']} {info['name']}</h4>
                <p>{info['description']}</p>
                <p><strong>Risk Level:</strong> {info['risk_level']}</p>
                <p><strong>Suitable for:</strong> {', '.join(info['suitable_for'])}</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button(f"Select {info['name']}", key=f"select_{category}"):
                    st.session_state.selected_strategy = category
                    st.success(f"Selected {info['name']} strategy")
            
            with col2:
                if st.button(f"Configure", key=f"config_{category}"):
                    self._show_strategy_config(category)
    
    def _render_strategy_configuration(self):
        """Render strategy configuration section"""
        if st.session_state.get('selected_strategy'):
            strategy = st.session_state.selected_strategy
            strategy_info = self.strategy_categories[strategy]
            
            st.markdown("### Strategy Configuration")
            
            with st.expander(f"Configure {strategy_info['name']} Strategy"):
                # Common parameters
                col1, col2 = st.columns(2)
                
                with col1:
                    st.number_input(
                        "Position Size (%)",
                        min_value=1,
                        max_value=20,
                        value=5,
                        key=f"{strategy}_position_size"
                    )
                    
                    st.number_input(
                        "Stop Loss (%)",
                        min_value=0.5,
                        max_value=10.0,
                        value=2.0,
                        step=0.1,
                        key=f"{strategy}_stop_loss"
                    )
                
                with col2:
                    st.number_input(
                        "Take Profit (%)",
                        min_value=1.0,
                        max_value=20.0,
                        value=4.0,
                        step=0.1,
                        key=f"{strategy}_take_profit"
                    )
                    
                    st.number_input(
                        "Max Positions",
                        min_value=1,
                        max_value=10,
                        value=3,
                        key=f"{strategy}_max_positions"
                    )
                
                # Strategy-specific parameters
                self._render_strategy_specific_params(strategy)
                
                # Target symbols
                st.markdown("#### Target Symbols")
                symbols = st.text_input(
                    "Enter symbols (comma-separated)",
                    value="AAPL,GOOGL,MSFT,TSLA,AMZN",
                    key=f"{strategy}_symbols"
                )
                
                # Exchange selection
                st.markdown("#### Exchange Selection")
                exchange = st.selectbox(
                    "Target Exchange",
                    ["alpaca", "binance", "kucoin", "td_ameritrade"],
                    key=f"{strategy}_exchange"
                )
                
                # Save configuration
                if st.button(f"Save Configuration", key=f"save_{strategy}"):
                    self._save_strategy_config(strategy)
    
    def _render_strategy_specific_params(self, strategy: str):
        """Render strategy-specific parameters"""
        st.markdown("#### Strategy-Specific Parameters")
        
        if strategy == "swing":
            col1, col2 = st.columns(2)
            
            with col1:
                st.number_input(
                    "Min Trend Strength",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.6,
                    step=0.1,
                    key="swing_trend_strength"
                )
                
                st.number_input(
                    "RSI Oversold",
                    min_value=10,
                    max_value=40,
                    value=30,
                    key="swing_rsi_oversold"
                )
            
            with col2:
                st.number_input(
                    "Holding Period (days)",
                    min_value=1,
                    max_value=14,
                    value=5,
                    key="swing_holding_period"
                )
                
                st.number_input(
                    "RSI Overbought",
                    min_value=60,
                    max_value=90,
                    value=70,
                    key="swing_rsi_overbought"
                )
        
        elif strategy == "scalping":
            col1, col2 = st.columns(2)
            
            with col1:
                st.number_input(
                    "Min Spread",
                    min_value=0.001,
                    max_value=0.1,
                    value=0.01,
                    step=0.001,
                    key="scalping_min_spread"
                )
                
                st.number_input(
                    "Fast EMA Period",
                    min_value=3,
                    max_value=15,
                    value=5,
                    key="scalping_fast_ema"
                )
            
            with col2:
                st.number_input(
                    "Max Spread",
                    min_value=0.01,
                    max_value=0.2,
                    value=0.05,
                    step=0.01,
                    key="scalping_max_spread"
                )
                
                st.number_input(
                    "Slow EMA Period",
                    min_value=10,
                    max_value=30,
                    value=13,
                    key="scalping_slow_ema"
                )
        
        elif strategy == "options":
            col1, col2 = st.columns(2)
            
            with col1:
                st.number_input(
                    "Min Days to Expiration",
                    min_value=7,
                    max_value=90,
                    value=30,
                    key="options_min_dte"
                )
                
                st.number_input(
                    "Min Implied Volatility",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.15,
                    step=0.05,
                    key="options_min_iv"
                )
            
            with col2:
                st.number_input(
                    "Max Days to Expiration",
                    min_value=30,
                    max_value=365,
                    value=90,
                    key="options_max_dte"
                )
                
                st.number_input(
                    "Max Implied Volatility",
                    min_value=0.2,
                    max_value=2.0,
                    value=0.60,
                    step=0.05,
                    key="options_max_iv"
                )
            
            # Options strategy types
            st.multiselect(
                "Options Strategies",
                ["covered_call", "cash_secured_put", "iron_condor", "straddle", "strangle"],
                default=["covered_call", "cash_secured_put"],
                key="options_strategies"
            )
        
        elif strategy == "intraday":
            col1, col2 = st.columns(2)
            
            with col1:
                st.number_input(
                    "Min Gap Percentage",
                    min_value=0.005,
                    max_value=0.05,
                    value=0.01,
                    step=0.005,
                    key="intraday_min_gap"
                )
                
                st.number_input(
                    "Min Volume Ratio",
                    min_value=1.0,
                    max_value=5.0,
                    value=1.5,
                    step=0.1,
                    key="intraday_min_volume"
                )
            
            with col2:
                st.number_input(
                    "Max Gap Percentage",
                    min_value=0.02,
                    max_value=0.1,
                    value=0.05,
                    step=0.005,
                    key="intraday_max_gap"
                )
                
                st.selectbox(
                    "Intraday Strategies",
                    ["gap_trading", "breakout_trading", "reversal_trading", "momentum_trading"],
                    key="intraday_sub_strategy"
                )
    
    def _save_strategy_config(self, strategy: str):
        """Save strategy configuration"""
        # Collect all configuration parameters
        config = {
            "strategy": strategy,
            "position_size": st.session_state.get(f"{strategy}_position_size", 5),
            "stop_loss": st.session_state.get(f"{strategy}_stop_loss", 2.0),
            "take_profit": st.session_state.get(f"{strategy}_take_profit", 4.0),
            "max_positions": st.session_state.get(f"{strategy}_max_positions", 3),
            "symbols": st.session_state.get(f"{strategy}_symbols", "").split(","),
            "exchange": st.session_state.get(f"{strategy}_exchange", "alpaca")
        }
        
        # Add strategy-specific parameters
        if strategy == "swing":
            config.update({
                "trend_strength": st.session_state.get("swing_trend_strength", 0.6),
                "rsi_oversold": st.session_state.get("swing_rsi_oversold", 30),
                "rsi_overbought": st.session_state.get("swing_rsi_overbought", 70),
                "holding_period": st.session_state.get("swing_holding_period", 5)
            })
        
        elif strategy == "scalping":
            config.update({
                "min_spread": st.session_state.get("scalping_min_spread", 0.01),
                "max_spread": st.session_state.get("scalping_max_spread", 0.05),
                "fast_ema": st.session_state.get("scalping_fast_ema", 5),
                "slow_ema": st.session_state.get("scalping_slow_ema", 13)
            })
        
        elif strategy == "options":
            config.update({
                "min_dte": st.session_state.get("options_min_dte", 30),
                "max_dte": st.session_state.get("options_max_dte", 90),
                "min_iv": st.session_state.get("options_min_iv", 0.15),
                "max_iv": st.session_state.get("options_max_iv", 0.60),
                "strategies": st.session_state.get("options_strategies", ["covered_call", "cash_secured_put"])
            })
        
        elif strategy == "intraday":
            config.update({
                "min_gap": st.session_state.get("intraday_min_gap", 0.01),
                "max_gap": st.session_state.get("intraday_max_gap", 0.05),
                "min_volume": st.session_state.get("intraday_min_volume", 1.5),
                "sub_strategy": st.session_state.get("intraday_sub_strategy", "gap_trading")
            })
        
        # Save to session state
        st.session_state[f"{strategy}_config"] = config
        
        st.success(f"Configuration saved for {self.strategy_categories[strategy]['name']} strategy!")
    
    def _show_strategy_config(self, strategy: str):
        """Show strategy configuration in a popup"""
        st.info(f"Configuration for {self.strategy_categories[strategy]['name']} strategy will be shown here.")
    
    def get_strategy_config(self, strategy: str) -> Optional[Dict]:
        """Get saved configuration for a strategy"""
        return st.session_state.get(f"{strategy}_config")
    
    def get_selected_strategy(self) -> Optional[str]:
        """Get currently selected strategy"""
        return st.session_state.get('selected_strategy')
