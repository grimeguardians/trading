"""
Options Trading Strategy
Specialized strategy for options trading with Greeks analysis
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from math import sqrt, log, exp
from scipy.stats import norm

from strategies.base_strategy import BaseStrategy, TradingSignal
from config import Config


class OptionsStrategy(BaseStrategy):
    """
    Options trading strategy implementation
    Focuses on volatility, Greeks, and time decay considerations
    """
    
    def __init__(self, config: Config):
        super().__init__(config, "options")
        
        # Options-specific parameters
        self.iv_percentile_threshold = self.parameters.get("iv_percentile_threshold", 30)
        self.iv_rank_threshold = self.parameters.get("iv_rank_threshold", 50)
        self.delta_threshold = self.parameters.get("delta_threshold", 0.3)
        self.gamma_threshold = self.parameters.get("gamma_threshold", 0.1)
        self.theta_threshold = self.parameters.get("theta_threshold", -0.05)
        self.vega_threshold = self.parameters.get("vega_threshold", 0.1)
        
        # Options strategy types
        self.strategy_types = [
            "long_call", "long_put", "short_call", "short_put",
            "bull_call_spread", "bear_put_spread", "iron_condor",
            "straddle", "strangle", "butterfly"
        ]
        
        # Risk management
        self.max_dte = 45  # Maximum days to expiration
        self.min_dte = 7   # Minimum days to expiration
        self.max_position_size = 0.05  # 5% of portfolio per position
        self.profit_target = 0.50  # 50% profit target
        self.stop_loss = 0.20  # 20% stop loss
        
        # Market environment
        self.risk_free_rate = 0.02  # 2% risk-free rate
        self.current_iv_environment = "normal"  # normal, high, low
        
        self.logger.info("📊 Options Strategy initialized")
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """Get default options strategy parameters"""
        return {
            "iv_percentile_threshold": 30,
            "iv_rank_threshold": 50,
            "delta_threshold": 0.3,
            "gamma_threshold": 0.1,
            "theta_threshold": -0.05,
            "vega_threshold": 0.1,
            "min_open_interest": 100,
            "min_volume": 50,
            "max_bid_ask_spread": 0.05,
            "volatility_lookback": 20
        }
    
    async def analyze_market(self, symbol: str, market_data: pd.DataFrame) -> Optional[TradingSignal]:
        """Analyze market for options trading opportunities"""
        try:
            if len(market_data) < 50:
                return None
            
            # Calculate underlying analysis
            underlying_analysis = self._analyze_underlying(market_data)
            
            # Calculate implied volatility analysis
            iv_analysis = self._analyze_implied_volatility(market_data)
            
            # Get options chain data (simulated)
            options_chain = self._get_options_chain(symbol, market_data['close'].iloc[-1])
            
            # Analyze options opportunities
            opportunities = self._analyze_options_opportunities(
                underlying_analysis, iv_analysis, options_chain
            )
            
            # Select best opportunity
            best_opportunity = self._select_best_opportunity(opportunities)
            
            if best_opportunity:
                return self._create_options_signal(symbol, best_opportunity, market_data)
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Options analysis error for {symbol}: {e}")
            return None
    
    def _analyze_underlying(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze underlying stock for options trading"""
        try:
            current_price = market_data['close'].iloc[-1]
            
            # Calculate historical volatility
            returns = market_data['close'].pct_change().dropna()
            hist_vol = returns.std() * sqrt(252)  # Annualized
            
            # Calculate support and resistance
            support_resistance = self.calculate_support_resistance(market_data)
            
            # Calculate trend
            sma_20 = market_data['close'].rolling(window=20).mean().iloc[-1]
            sma_50 = market_data['close'].rolling(window=50).mean().iloc[-1]
            
            trend = "bullish" if current_price > sma_20 > sma_50 else "bearish" if current_price < sma_20 < sma_50 else "neutral"
            
            # Calculate price momentum
            momentum = (current_price - market_data['close'].iloc[-5]) / market_data['close'].iloc[-5]
            
            # Calculate volatility regime
            vol_percentile = self._calculate_volatility_percentile(returns)
            
            return {
                "current_price": current_price,
                "historical_volatility": hist_vol,
                "support_levels": support_resistance["support"],
                "resistance_levels": support_resistance["resistance"],
                "trend": trend,
                "momentum": momentum,
                "volatility_percentile": vol_percentile,
                "sma_20": sma_20,
                "sma_50": sma_50
            }
            
        except Exception as e:
            self.logger.error(f"❌ Underlying analysis error: {e}")
            return {}
    
    def _analyze_implied_volatility(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze implied volatility environment"""
        try:
            # Simulate IV data (in production, would get from options data)
            returns = market_data['close'].pct_change().dropna()
            hist_vol = returns.std() * sqrt(252)
            
            # Simulate current IV as function of historical volatility
            iv_multiplier = 1.0 + np.random.normal(0, 0.2)  # Random IV premium
            current_iv = hist_vol * iv_multiplier
            
            # Calculate IV percentile (simulated)
            iv_percentile = min(max(np.random.normal(50, 20), 0), 100)
            
            # Calculate IV rank (simulated)
            iv_rank = min(max(np.random.normal(45, 15), 0), 100)
            
            # Determine IV environment
            if iv_percentile > 70:
                iv_environment = "high"
            elif iv_percentile < 30:
                iv_environment = "low"
            else:
                iv_environment = "normal"
            
            return {
                "current_iv": current_iv,
                "historical_volatility": hist_vol,
                "iv_percentile": iv_percentile,
                "iv_rank": iv_rank,
                "iv_environment": iv_environment,
                "iv_premium": current_iv - hist_vol
            }
            
        except Exception as e:
            self.logger.error(f"❌ IV analysis error: {e}")
            return {}
    
    def _get_options_chain(self, symbol: str, current_price: float) -> Dict[str, Any]:
        """Get options chain data (simulated)"""
        try:
            # Simulate options chain
            expirations = []
            current_date = datetime.utcnow()
            
            # Generate expiration dates
            for weeks in [1, 2, 3, 4, 6, 8]:
                exp_date = current_date + timedelta(weeks=weeks)
                # Adjust to Friday
                exp_date = exp_date + timedelta(days=(4 - exp_date.weekday()))
                expirations.append(exp_date)
            
            options_chain = {}
            
            for exp_date in expirations:
                dte = (exp_date - current_date).days
                
                # Generate strike prices
                strikes = []
                for i in range(-10, 11):
                    strike = current_price + (i * 5)  # $5 intervals
                    strikes.append(strike)
                
                options_chain[exp_date.strftime("%Y-%m-%d")] = {
                    "expiration": exp_date,
                    "dte": dte,
                    "calls": self._generate_options_data(strikes, current_price, dte, "call"),
                    "puts": self._generate_options_data(strikes, current_price, dte, "put")
                }
            
            return options_chain
            
        except Exception as e:
            self.logger.error(f"❌ Options chain error: {e}")
            return {}
    
    def _generate_options_data(self, strikes: List[float], current_price: float, dte: int, option_type: str) -> List[Dict]:
        """Generate simulated options data"""
        try:
            options_data = []
            
            for strike in strikes:
                # Calculate basic Greeks using Black-Scholes
                greeks = self._calculate_greeks(current_price, strike, dte, option_type)
                
                # Simulate market data
                theoretical_price = greeks["theoretical_price"]
                bid = theoretical_price * 0.95
                ask = theoretical_price * 1.05
                
                volume = max(0, int(np.random.exponential(50)))
                open_interest = max(0, int(np.random.exponential(200)))
                
                options_data.append({
                    "strike": strike,
                    "bid": bid,
                    "ask": ask,
                    "last": (bid + ask) / 2,
                    "volume": volume,
                    "open_interest": open_interest,
                    "implied_volatility": greeks["iv"],
                    "delta": greeks["delta"],
                    "gamma": greeks["gamma"],
                    "theta": greeks["theta"],
                    "vega": greeks["vega"],
                    "theoretical_price": theoretical_price
                })
            
            return options_data
            
        except Exception as e:
            self.logger.error(f"❌ Options data generation error: {e}")
            return []
    
    def _calculate_greeks(self, S: float, K: float, T: int, option_type: str) -> Dict[str, float]:
        """Calculate option Greeks using Black-Scholes"""
        try:
            # Convert days to years
            T_years = T / 365.0
            
            # Use estimated volatility
            sigma = 0.25  # 25% volatility assumption
            r = self.risk_free_rate
            
            # Avoid division by zero
            if T_years <= 0:
                T_years = 1/365
            
            # Black-Scholes calculations
            d1 = (log(S/K) + (r + 0.5 * sigma**2) * T_years) / (sigma * sqrt(T_years))
            d2 = d1 - sigma * sqrt(T_years)
            
            if option_type == "call":
                theoretical_price = S * norm.cdf(d1) - K * exp(-r * T_years) * norm.cdf(d2)
                delta = norm.cdf(d1)
            else:  # put
                theoretical_price = K * exp(-r * T_years) * norm.cdf(-d2) - S * norm.cdf(-d1)
                delta = -norm.cdf(-d1)
            
            # Common Greeks
            gamma = norm.pdf(d1) / (S * sigma * sqrt(T_years))
            theta = (-S * norm.pdf(d1) * sigma / (2 * sqrt(T_years)) - 
                    r * K * exp(-r * T_years) * norm.cdf(d2 if option_type == "call" else -d2))
            
            if option_type == "put":
                theta += r * K * exp(-r * T_years)
            
            theta = theta / 365  # Per day
            
            vega = S * norm.pdf(d1) * sqrt(T_years) / 100  # Per 1% volatility change
            
            return {
                "theoretical_price": max(0, theoretical_price),
                "delta": delta,
                "gamma": gamma,
                "theta": theta,
                "vega": vega,
                "iv": sigma
            }
            
        except Exception as e:
            self.logger.error(f"❌ Greeks calculation error: {e}")
            return {
                "theoretical_price": 0,
                "delta": 0,
                "gamma": 0,
                "theta": 0,
                "vega": 0,
                "iv": 0.25
            }
    
    def _calculate_volatility_percentile(self, returns: pd.Series) -> float:
        """Calculate volatility percentile"""
        try:
            current_vol = returns.std() * sqrt(252)
            historical_vols = returns.rolling(window=20).std() * sqrt(252)
            
            percentile = (historical_vols < current_vol).mean() * 100
            return percentile
            
        except Exception as e:
            self.logger.error(f"❌ Volatility percentile calculation error: {e}")
            return 50.0
    
    def _analyze_options_opportunities(self, underlying_analysis: Dict, iv_analysis: Dict, 
                                     options_chain: Dict) -> List[Dict]:
        """Analyze options trading opportunities"""
        try:
            opportunities = []
            
            current_price = underlying_analysis["current_price"]
            iv_environment = iv_analysis["iv_environment"]
            trend = underlying_analysis["trend"]
            
            for exp_date, exp_data in options_chain.items():
                dte = exp_data["dte"]
                
                # Skip if outside DTE range
                if dte < self.min_dte or dte > self.max_dte:
                    continue
                
                # Analyze different strategies
                if iv_environment == "high":
                    # High IV: prefer selling strategies
                    opportunities.extend(self._analyze_selling_strategies(exp_data, current_price, trend))
                elif iv_environment == "low":
                    # Low IV: prefer buying strategies
                    opportunities.extend(self._analyze_buying_strategies(exp_data, current_price, trend))
                else:
                    # Normal IV: mixed strategies
                    opportunities.extend(self._analyze_mixed_strategies(exp_data, current_price, trend))
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"❌ Options opportunities analysis error: {e}")
            return []
    
    def _analyze_selling_strategies(self, exp_data: Dict, current_price: float, trend: str) -> List[Dict]:
        """Analyze premium selling strategies"""
        try:
            opportunities = []
            calls = exp_data["calls"]
            puts = exp_data["puts"]
            
            # Find ATM options
            atm_call = min(calls, key=lambda x: abs(x["strike"] - current_price))
            atm_put = min(puts, key=lambda x: abs(x["strike"] - current_price))
            
            # Short straddle (high IV)
            if atm_call["implied_volatility"] > 0.30:  # High IV threshold
                opportunities.append({
                    "strategy": "short_straddle",
                    "legs": [
                        {"action": "sell", "option": atm_call},
                        {"action": "sell", "option": atm_put}
                    ],
                    "max_profit": atm_call["ask"] + atm_put["ask"],
                    "max_loss": float('inf'),
                    "breakeven_upper": atm_call["strike"] + atm_call["ask"] + atm_put["ask"],
                    "breakeven_lower": atm_put["strike"] - atm_call["ask"] - atm_put["ask"],
                    "score": self._calculate_strategy_score("short_straddle", exp_data, current_price)
                })
            
            # Covered calls (if bullish/neutral)
            if trend in ["bullish", "neutral"]:
                otm_calls = [c for c in calls if c["strike"] > current_price and c["delta"] < 0.3]
                for call in otm_calls:
                    if call["volume"] > 50 and call["open_interest"] > 100:
                        opportunities.append({
                            "strategy": "covered_call",
                            "legs": [
                                {"action": "sell", "option": call}
                            ],
                            "max_profit": call["ask"] + (call["strike"] - current_price),
                            "max_loss": current_price - call["ask"],
                            "breakeven": current_price - call["ask"],
                            "score": self._calculate_strategy_score("covered_call", exp_data, current_price)
                        })
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"❌ Selling strategies analysis error: {e}")
            return []
    
    def _analyze_buying_strategies(self, exp_data: Dict, current_price: float, trend: str) -> List[Dict]:
        """Analyze premium buying strategies"""
        try:
            opportunities = []
            calls = exp_data["calls"]
            puts = exp_data["puts"]
            
            # Long calls (bullish trend)
            if trend == "bullish":
                otm_calls = [c for c in calls if c["strike"] > current_price and c["delta"] > 0.2]
                for call in otm_calls:
                    if call["volume"] > 50 and call["implied_volatility"] < 0.25:  # Low IV
                        opportunities.append({
                            "strategy": "long_call",
                            "legs": [
                                {"action": "buy", "option": call}
                            ],
                            "max_profit": float('inf'),
                            "max_loss": call["ask"],
                            "breakeven": call["strike"] + call["ask"],
                            "score": self._calculate_strategy_score("long_call", exp_data, current_price)
                        })
            
            # Long puts (bearish trend)
            if trend == "bearish":
                otm_puts = [p for p in puts if p["strike"] < current_price and abs(p["delta"]) > 0.2]
                for put in otm_puts:
                    if put["volume"] > 50 and put["implied_volatility"] < 0.25:  # Low IV
                        opportunities.append({
                            "strategy": "long_put",
                            "legs": [
                                {"action": "buy", "option": put}
                            ],
                            "max_profit": put["strike"] - put["ask"],
                            "max_loss": put["ask"],
                            "breakeven": put["strike"] - put["ask"],
                            "score": self._calculate_strategy_score("long_put", exp_data, current_price)
                        })
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"❌ Buying strategies analysis error: {e}")
            return []
    
    def _analyze_mixed_strategies(self, exp_data: Dict, current_price: float, trend: str) -> List[Dict]:
        """Analyze mixed strategies for normal IV environment"""
        try:
            opportunities = []
            calls = exp_data["calls"]
            puts = exp_data["puts"]
            
            # Bull call spread
            if trend == "bullish":
                # Find suitable strikes
                itm_calls = [c for c in calls if c["strike"] < current_price and c["delta"] > 0.6]
                otm_calls = [c for c in calls if c["strike"] > current_price and c["delta"] < 0.4]
                
                for long_call in itm_calls:
                    for short_call in otm_calls:
                        if short_call["strike"] > long_call["strike"]:
                            spread_cost = long_call["ask"] - short_call["bid"]
                            max_profit = (short_call["strike"] - long_call["strike"]) - spread_cost
                            
                            if max_profit > 0:
                                opportunities.append({
                                    "strategy": "bull_call_spread",
                                    "legs": [
                                        {"action": "buy", "option": long_call},
                                        {"action": "sell", "option": short_call}
                                    ],
                                    "max_profit": max_profit,
                                    "max_loss": spread_cost,
                                    "breakeven": long_call["strike"] + spread_cost,
                                    "score": self._calculate_strategy_score("bull_call_spread", exp_data, current_price)
                                })
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"❌ Mixed strategies analysis error: {e}")
            return []
    
    def _calculate_strategy_score(self, strategy: str, exp_data: Dict, current_price: float) -> float:
        """Calculate score for options strategy"""
        try:
            score = 0.0
            
            # Base score by strategy type
            strategy_scores = {
                "long_call": 0.6,
                "long_put": 0.6,
                "short_straddle": 0.7,
                "covered_call": 0.8,
                "bull_call_spread": 0.75,
                "bear_put_spread": 0.75
            }
            
            score = strategy_scores.get(strategy, 0.5)
            
            # Adjust for DTE
            dte = exp_data["dte"]
            if 15 <= dte <= 30:
                score += 0.1  # Optimal DTE range
            elif dte < 15:
                score -= 0.1  # Too close to expiration
            
            # Adjust for liquidity (simplified)
            score += 0.05  # Assume decent liquidity
            
            return min(score, 1.0)
            
        except Exception as e:
            self.logger.error(f"❌ Strategy score calculation error: {e}")
            return 0.5
    
    def _select_best_opportunity(self, opportunities: List[Dict]) -> Optional[Dict]:
        """Select the best options opportunity"""
        try:
            if not opportunities:
                return None
            
            # Sort by score
            opportunities.sort(key=lambda x: x["score"], reverse=True)
            
            # Return the best opportunity
            return opportunities[0]
            
        except Exception as e:
            self.logger.error(f"❌ Best opportunity selection error: {e}")
            return None
    
    def _create_options_signal(self, symbol: str, opportunity: Dict, market_data: pd.DataFrame) -> TradingSignal:
        """Create options trading signal"""
        try:
            current_price = market_data['close'].iloc[-1]
            strategy = opportunity["strategy"]
            
            # Calculate confidence based on strategy score and market conditions
            confidence = opportunity["score"]
            
            # Create reasoning
            reasoning = f"Options strategy: {strategy.replace('_', ' ').title()}"
            
            # Calculate risk metrics
            max_profit = opportunity.get("max_profit", 0)
            max_loss = opportunity.get("max_loss", 0)
            
            if max_loss > 0 and max_profit > 0:
                risk_reward = max_profit / max_loss
                reasoning += f" (Risk/Reward: {risk_reward:.2f})"
            
            # Create signal
            signal = TradingSignal(
                symbol=symbol,
                signal_type="BUY",  # Options strategies are typically "BUY" orders
                confidence=confidence,
                price_target=current_price,
                stop_loss=current_price * 0.9 if max_loss == float('inf') else None,
                take_profit=current_price * 1.1 if max_profit == float('inf') else None,
                timeframe="options",
                reasoning=reasoning,
                metadata={
                    "strategy": "options",
                    "options_strategy": strategy,
                    "max_profit": max_profit,
                    "max_loss": max_loss,
                    "breakeven": opportunity.get("breakeven", current_price),
                    "legs": opportunity["legs"]
                }
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"❌ Options signal creation error: {e}")
            return None
    
    async def should_exit_position(self, symbol: str, position: Dict, current_price: float) -> bool:
        """Determine if options position should be exited"""
        try:
            entry_time = position.get('entry_time', datetime.utcnow())
            entry_price = position.get('entry_price', current_price)
            strategy = position.get('strategy', 'long_call')
            
            # Check time decay (theta)
            days_held = (datetime.utcnow() - entry_time).days
            
            # Exit if approaching expiration
            if days_held > 30:  # Assuming monthly options
                return True
            
            # Check profit/loss
            pnl_pct = (current_price - entry_price) / entry_price
            
            # Exit if profit target reached
            if pnl_pct >= self.profit_target:
                return True
            
            # Exit if stop loss hit
            if pnl_pct <= -self.stop_loss:
                return True
            
            # Strategy-specific exit rules
            if strategy in ["short_straddle", "short_strangle"]:
                # Exit if volatility increases significantly
                if days_held > 7 and pnl_pct < -0.1:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Options exit analysis error: {e}")
            return False
    
    def get_optimization_parameters(self) -> Dict[str, Tuple[float, float, float]]:
        """Get parameters for optimization"""
        return {
            "iv_percentile_threshold": (20, 40, 5),
            "iv_rank_threshold": (40, 60, 5),
            "delta_threshold": (0.2, 0.4, 0.05),
            "profit_target": (0.3, 0.7, 0.1),
            "stop_loss": (0.15, 0.25, 0.05),
            "max_dte": (30, 60, 15),
            "min_dte": (5, 15, 5)
        }
    
    def get_options_metrics(self) -> Dict[str, Any]:
        """Get options-specific metrics"""
        return {
            "iv_environment": self.current_iv_environment,
            "max_dte": self.max_dte,
            "min_dte": self.min_dte,
            "profit_target": self.profit_target,
            "stop_loss": self.stop_loss,
            "supported_strategies": self.strategy_types,
            "risk_free_rate": self.risk_free_rate
        }
