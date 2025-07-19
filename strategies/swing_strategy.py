"""
Swing Trading Strategy
Medium-term trading strategy focusing on capturing price swings over days to weeks
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

from strategies.base_strategy import BaseStrategy, TradingSignal
from config import Config


class SwingStrategy(BaseStrategy):
    """
    Swing trading strategy implementation
    Focuses on capturing medium-term price movements with technical analysis
    """
    
    def __init__(self, config: Config):
        super().__init__(config, "swing")
        
        # Swing-specific parameters
        self.trend_period = self.parameters.get("trend_period", 20)
        self.momentum_period = self.parameters.get("momentum_period", 14)
        self.volume_threshold = self.parameters.get("volume_threshold", 1.5)
        self.rsi_oversold = self.parameters.get("rsi_oversold", 30)
        self.rsi_overbought = self.parameters.get("rsi_overbought", 70)
        self.macd_signal_threshold = self.parameters.get("macd_signal_threshold", 0.001)
        
        # Swing trading specific settings
        self.min_swing_duration = timedelta(hours=12)  # Minimum swing duration
        self.max_swing_duration = timedelta(days=14)   # Maximum swing duration
        self.profit_target_multiplier = 2.0  # Risk:Reward ratio
        
        self.logger.info("📊 Swing Strategy initialized")
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """Get default swing strategy parameters"""
        return {
            "trend_period": 20,
            "momentum_period": 14,
            "volume_threshold": 1.5,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "macd_signal_threshold": 0.001,
            "bb_std_multiplier": 2.0,
            "min_volume_ratio": 1.2,
            "trend_strength_threshold": 0.02
        }
    
    async def analyze_market(self, symbol: str, market_data: pd.DataFrame) -> Optional[TradingSignal]:
        """Analyze market for swing trading opportunities"""
        try:
            if len(market_data) < 50:  # Need sufficient data
                return None
            
            # Calculate technical indicators
            indicators = self.calculate_technical_indicators(market_data)
            
            # Get current values
            current_price = market_data['close'].iloc[-1]
            current_rsi = indicators['rsi'].iloc[-1]
            current_macd = indicators['macd'].iloc[-1]
            current_macd_signal = indicators['macd_signal'].iloc[-1]
            current_bb_upper = indicators['bb_upper'].iloc[-1]
            current_bb_lower = indicators['bb_lower'].iloc[-1]
            current_volume_ratio = indicators['volume_ratio'].iloc[-1]
            
            # Analyze trend
            trend_analysis = self._analyze_trend(market_data, indicators)
            
            # Analyze momentum
            momentum_analysis = self._analyze_momentum(indicators)
            
            # Analyze volume
            volume_analysis = self._analyze_volume(market_data, indicators)
            
            # Generate signal based on analysis
            signal = self._generate_swing_signal(
                symbol=symbol,
                current_price=current_price,
                trend_analysis=trend_analysis,
                momentum_analysis=momentum_analysis,
                volume_analysis=volume_analysis,
                rsi=current_rsi,
                macd=current_macd,
                macd_signal=current_macd_signal,
                bb_upper=current_bb_upper,
                bb_lower=current_bb_lower,
                volume_ratio=current_volume_ratio
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"❌ Swing analysis error for {symbol}: {e}")
            return None
    
    def _analyze_trend(self, market_data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trend direction and strength"""
        try:
            sma_20 = indicators['sma_20'].iloc[-1]
            sma_50 = indicators['sma_50'].iloc[-1]
            current_price = market_data['close'].iloc[-1]
            
            # Determine trend direction
            if current_price > sma_20 > sma_50:
                trend_direction = "bullish"
            elif current_price < sma_20 < sma_50:
                trend_direction = "bearish"
            else:
                trend_direction = "neutral"
            
            # Calculate trend strength
            price_change = (current_price - market_data['close'].iloc[-self.trend_period]) / market_data['close'].iloc[-self.trend_period]
            trend_strength = abs(price_change)
            
            # Check for trend reversal signals
            recent_highs = market_data['high'].rolling(window=5).max()
            recent_lows = market_data['low'].rolling(window=5).min()
            
            higher_highs = recent_highs.iloc[-1] > recent_highs.iloc[-6]
            higher_lows = recent_lows.iloc[-1] > recent_lows.iloc[-6]
            lower_highs = recent_highs.iloc[-1] < recent_highs.iloc[-6]
            lower_lows = recent_lows.iloc[-1] < recent_lows.iloc[-6]
            
            reversal_signal = None
            if trend_direction == "bearish" and higher_highs and higher_lows:
                reversal_signal = "bullish_reversal"
            elif trend_direction == "bullish" and lower_highs and lower_lows:
                reversal_signal = "bearish_reversal"
            
            return {
                "direction": trend_direction,
                "strength": trend_strength,
                "reversal_signal": reversal_signal,
                "sma_20": sma_20,
                "sma_50": sma_50,
                "price_above_sma20": current_price > sma_20,
                "price_above_sma50": current_price > sma_50
            }
            
        except Exception as e:
            self.logger.error(f"❌ Trend analysis error: {e}")
            return {"direction": "neutral", "strength": 0.0, "reversal_signal": None}
    
    def _analyze_momentum(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze momentum indicators"""
        try:
            current_rsi = indicators['rsi'].iloc[-1]
            current_macd = indicators['macd'].iloc[-1]
            current_macd_signal = indicators['macd_signal'].iloc[-1]
            current_stoch_k = indicators['stoch_k'].iloc[-1]
            current_stoch_d = indicators['stoch_d'].iloc[-1]
            
            # RSI analysis
            rsi_signal = None
            if current_rsi < self.rsi_oversold:
                rsi_signal = "oversold"
            elif current_rsi > self.rsi_overbought:
                rsi_signal = "overbought"
            
            # MACD analysis
            macd_signal = None
            macd_histogram = current_macd - current_macd_signal
            
            if current_macd > current_macd_signal and macd_histogram > self.macd_signal_threshold:
                macd_signal = "bullish"
            elif current_macd < current_macd_signal and macd_histogram < -self.macd_signal_threshold:
                macd_signal = "bearish"
            
            # Stochastic analysis
            stoch_signal = None
            if current_stoch_k < 20 and current_stoch_d < 20:
                stoch_signal = "oversold"
            elif current_stoch_k > 80 and current_stoch_d > 80:
                stoch_signal = "overbought"
            
            # Combined momentum score
            momentum_score = 0.0
            if rsi_signal == "oversold" or stoch_signal == "oversold":
                momentum_score += 0.3
            if rsi_signal == "overbought" or stoch_signal == "overbought":
                momentum_score -= 0.3
            if macd_signal == "bullish":
                momentum_score += 0.4
            if macd_signal == "bearish":
                momentum_score -= 0.4
            
            return {
                "rsi_signal": rsi_signal,
                "macd_signal": macd_signal,
                "stoch_signal": stoch_signal,
                "momentum_score": momentum_score,
                "rsi_value": current_rsi,
                "macd_histogram": macd_histogram
            }
            
        except Exception as e:
            self.logger.error(f"❌ Momentum analysis error: {e}")
            return {"momentum_score": 0.0}
    
    def _analyze_volume(self, market_data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze volume patterns"""
        try:
            current_volume = market_data['volume'].iloc[-1]
            volume_ratio = indicators['volume_ratio'].iloc[-1]
            
            # Volume trend analysis
            volume_trend = "neutral"
            if volume_ratio > self.volume_threshold:
                volume_trend = "increasing"
            elif volume_ratio < 0.7:
                volume_trend = "decreasing"
            
            # Price-volume relationship
            price_change = (market_data['close'].iloc[-1] - market_data['close'].iloc[-2]) / market_data['close'].iloc[-2]
            volume_change = (current_volume - market_data['volume'].iloc[-2]) / market_data['volume'].iloc[-2]
            
            volume_confirmation = None
            if price_change > 0 and volume_change > 0:
                volume_confirmation = "bullish_confirmation"
            elif price_change < 0 and volume_change > 0:
                volume_confirmation = "bearish_confirmation"
            elif price_change > 0 and volume_change < 0:
                volume_confirmation = "bullish_divergence"
            elif price_change < 0 and volume_change < 0:
                volume_confirmation = "bearish_divergence"
            
            return {
                "trend": volume_trend,
                "ratio": volume_ratio,
                "confirmation": volume_confirmation,
                "current_volume": current_volume,
                "volume_strength": min(volume_ratio / self.volume_threshold, 2.0)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Volume analysis error: {e}")
            return {"trend": "neutral", "ratio": 1.0, "confirmation": None}
    
    def _generate_swing_signal(self, symbol: str, current_price: float, trend_analysis: Dict,
                              momentum_analysis: Dict, volume_analysis: Dict, rsi: float,
                              macd: float, macd_signal: float, bb_upper: float, bb_lower: float,
                              volume_ratio: float) -> Optional[TradingSignal]:
        """Generate swing trading signal"""
        try:
            signal_type = "HOLD"
            confidence = 0.0
            reasoning = []
            
            # Bullish signal conditions
            bullish_score = 0.0
            bearish_score = 0.0
            
            # Trend analysis scoring
            if trend_analysis["direction"] == "bullish":
                bullish_score += 0.2
                reasoning.append("Bullish trend detected")
            elif trend_analysis["direction"] == "bearish":
                bearish_score += 0.2
                reasoning.append("Bearish trend detected")
            
            if trend_analysis["reversal_signal"] == "bullish_reversal":
                bullish_score += 0.3
                reasoning.append("Bullish reversal pattern")
            elif trend_analysis["reversal_signal"] == "bearish_reversal":
                bearish_score += 0.3
                reasoning.append("Bearish reversal pattern")
            
            # Momentum analysis scoring
            if momentum_analysis["rsi_signal"] == "oversold":
                bullish_score += 0.2
                reasoning.append("RSI oversold")
            elif momentum_analysis["rsi_signal"] == "overbought":
                bearish_score += 0.2
                reasoning.append("RSI overbought")
            
            if momentum_analysis["macd_signal"] == "bullish":
                bullish_score += 0.2
                reasoning.append("MACD bullish crossover")
            elif momentum_analysis["macd_signal"] == "bearish":
                bearish_score += 0.2
                reasoning.append("MACD bearish crossover")
            
            # Volume analysis scoring
            if volume_analysis["confirmation"] == "bullish_confirmation":
                bullish_score += 0.15
                reasoning.append("Volume confirms bullish move")
            elif volume_analysis["confirmation"] == "bearish_confirmation":
                bearish_score += 0.15
                reasoning.append("Volume confirms bearish move")
            
            # Bollinger Bands analysis
            if current_price <= bb_lower:
                bullish_score += 0.15
                reasoning.append("Price at lower Bollinger Band")
            elif current_price >= bb_upper:
                bearish_score += 0.15
                reasoning.append("Price at upper Bollinger Band")
            
            # Determine signal
            if bullish_score > bearish_score and bullish_score >= 0.6:
                signal_type = "BUY"
                confidence = min(bullish_score, 0.95)
            elif bearish_score > bullish_score and bearish_score >= 0.6:
                signal_type = "SELL"
                confidence = min(bearish_score, 0.95)
            
            # Skip signals below threshold
            if confidence < self.signal_threshold:
                return None
            
            # Calculate stop loss and take profit
            stop_loss, take_profit = self._calculate_swing_levels(
                current_price, signal_type, trend_analysis, bb_upper, bb_lower
            )
            
            # Create signal
            signal = TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                price_target=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                timeframe=self.timeframe,
                reasoning="; ".join(reasoning),
                metadata={
                    "strategy": "swing",
                    "trend_direction": trend_analysis["direction"],
                    "trend_strength": trend_analysis["strength"],
                    "momentum_score": momentum_analysis["momentum_score"],
                    "volume_ratio": volume_ratio,
                    "rsi": rsi,
                    "macd": macd,
                    "bb_position": self._get_bb_position(current_price, bb_upper, bb_lower)
                }
            )
            
            # Validate signal
            if self.validate_signal(signal):
                return signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Swing signal generation error: {e}")
            return None
    
    def _calculate_swing_levels(self, current_price: float, signal_type: str, 
                               trend_analysis: Dict, bb_upper: float, bb_lower: float) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels for swing trades"""
        try:
            if signal_type == "BUY":
                # Stop loss below recent support or lower BB
                stop_loss = min(current_price * 0.97, bb_lower * 0.995)  # 3% or just below lower BB
                
                # Take profit based on risk:reward ratio
                risk = current_price - stop_loss
                take_profit = current_price + (risk * self.profit_target_multiplier)
                
                # Consider resistance levels
                if bb_upper > current_price:
                    take_profit = min(take_profit, bb_upper * 0.995)
                
            else:  # SELL
                # Stop loss above recent resistance or upper BB
                stop_loss = max(current_price * 1.03, bb_upper * 1.005)  # 3% or just above upper BB
                
                # Take profit based on risk:reward ratio
                risk = stop_loss - current_price
                take_profit = current_price - (risk * self.profit_target_multiplier)
                
                # Consider support levels
                if bb_lower < current_price:
                    take_profit = max(take_profit, bb_lower * 1.005)
            
            return stop_loss, take_profit
            
        except Exception as e:
            self.logger.error(f"❌ Swing levels calculation error: {e}")
            return current_price * 0.97, current_price * 1.06  # Default 3% stop, 6% target
    
    def _get_bb_position(self, current_price: float, bb_upper: float, bb_lower: float) -> str:
        """Get position relative to Bollinger Bands"""
        try:
            bb_range = bb_upper - bb_lower
            if bb_range == 0:
                return "middle"
            
            position = (current_price - bb_lower) / bb_range
            
            if position <= 0.2:
                return "lower"
            elif position >= 0.8:
                return "upper"
            else:
                return "middle"
                
        except Exception as e:
            self.logger.error(f"❌ BB position calculation error: {e}")
            return "middle"
    
    async def should_exit_position(self, symbol: str, position: Dict, current_price: float) -> bool:
        """Determine if swing position should be exited"""
        try:
            entry_price = position.get('entry_price', 0)
            entry_time = position.get('entry_time', datetime.utcnow())
            side = position.get('side', 'long')
            
            # Check holding period
            holding_period = datetime.utcnow() - entry_time
            if holding_period > self.max_swing_duration:
                return True
            
            # Check if minimum holding period met
            if holding_period < self.min_swing_duration:
                return False
            
            # Get fresh market data for exit analysis
            market_data = await self._get_market_data(symbol)
            if market_data is None:
                return False
            
            indicators = self.calculate_technical_indicators(market_data)
            
            # Check for exit conditions
            current_rsi = indicators['rsi'].iloc[-1]
            current_macd = indicators['macd'].iloc[-1]
            current_macd_signal = indicators['macd_signal'].iloc[-1]
            
            # Exit conditions for long positions
            if side == 'long':
                # Exit if RSI overbought and MACD turns bearish
                if current_rsi > 70 and current_macd < current_macd_signal:
                    return True
                
                # Exit if price drops below SMA20 with volume confirmation
                sma_20 = indicators['sma_20'].iloc[-1]
                if current_price < sma_20 and indicators['volume_ratio'].iloc[-1] > 1.2:
                    return True
            
            # Exit conditions for short positions
            elif side == 'short':
                # Exit if RSI oversold and MACD turns bullish
                if current_rsi < 30 and current_macd > current_macd_signal:
                    return True
                
                # Exit if price rises above SMA20 with volume confirmation
                sma_20 = indicators['sma_20'].iloc[-1]
                if current_price > sma_20 and indicators['volume_ratio'].iloc[-1] > 1.2:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Swing exit analysis error: {e}")
            return False
    
    def get_optimization_parameters(self) -> Dict[str, Tuple[float, float, float]]:
        """Get parameters for optimization"""
        return {
            "trend_period": (10, 30, 5),
            "momentum_period": (10, 20, 2),
            "volume_threshold": (1.2, 2.0, 0.1),
            "rsi_oversold": (25, 35, 2),
            "rsi_overbought": (65, 75, 2),
            "macd_signal_threshold": (0.0005, 0.002, 0.0005),
            "bb_std_multiplier": (1.5, 2.5, 0.25)
        }
