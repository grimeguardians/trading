"""
Intraday Trading Strategy
Day trading strategy focusing on intraday price movements
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

from strategies.base_strategy import BaseStrategy, TradingSignal
from config import Config


class IntradayStrategy(BaseStrategy):
    """
    Intraday trading strategy implementation
    Focuses on day trading opportunities with end-of-day position closure
    """
    
    def __init__(self, config: Config):
        super().__init__(config, "intraday")
        
        # Intraday-specific parameters
        self.morning_range_minutes = self.parameters.get("morning_range_minutes", 60)
        self.breakout_threshold = self.parameters.get("breakout_threshold", 0.005)
        self.volume_surge_threshold = self.parameters.get("volume_surge_threshold", 2.0)
        self.atr_multiplier = self.parameters.get("atr_multiplier", 1.5)
        self.momentum_period = self.parameters.get("momentum_period", 10)
        
        # Trading hours (EST)
        self.market_open = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
        self.market_close = datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)
        self.power_hour_start = datetime.now().replace(hour=15, minute=0, second=0, microsecond=0)
        
        # Intraday-specific settings
        self.avoid_lunch_time = True
        self.lunch_start = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
        self.lunch_end = datetime.now().replace(hour=13, minute=0, second=0, microsecond=0)
        
        # Position management
        self.max_intraday_loss = 0.02  # 2% max loss per day
        self.profit_target_atr = 2.0  # 2x ATR profit target
        self.stop_loss_atr = 1.0  # 1x ATR stop loss
        
        # Pattern recognition
        self.gap_threshold = 0.01  # 1% gap threshold
        self.breakout_confirmation_minutes = 15
        
        self.logger.info("📈 Intraday Strategy initialized")
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """Get default intraday strategy parameters"""
        return {
            "morning_range_minutes": 60,
            "breakout_threshold": 0.005,
            "volume_surge_threshold": 2.0,
            "atr_multiplier": 1.5,
            "momentum_period": 10,
            "rsi_overbought": 75,
            "rsi_oversold": 25,
            "vwap_deviation_threshold": 0.5,
            "gap_fade_threshold": 0.02
        }
    
    async def analyze_market(self, symbol: str, market_data: pd.DataFrame) -> Optional[TradingSignal]:
        """Analyze market for intraday trading opportunities"""
        try:
            if len(market_data) < 30:
                return None
            
            # Check if we're in trading hours
            if not self._is_trading_hours():
                return None
            
            # Skip lunch hour if configured
            if self.avoid_lunch_time and self._is_lunch_time():
                return None
            
            # Calculate intraday indicators
            indicators = self._calculate_intraday_indicators(market_data)
            
            # Analyze morning range
            morning_analysis = self._analyze_morning_range(market_data)
            
            # Analyze gaps
            gap_analysis = self._analyze_gaps(market_data)
            
            # Analyze breakouts
            breakout_analysis = self._analyze_breakouts(market_data, indicators)
            
            # Analyze momentum
            momentum_analysis = self._analyze_intraday_momentum(market_data, indicators)
            
            # Analyze volume
            volume_analysis = self._analyze_intraday_volume(market_data, indicators)
            
            # Generate intraday signal
            signal = self._generate_intraday_signal(
                symbol=symbol,
                market_data=market_data,
                indicators=indicators,
                morning_analysis=morning_analysis,
                gap_analysis=gap_analysis,
                breakout_analysis=breakout_analysis,
                momentum_analysis=momentum_analysis,
                volume_analysis=volume_analysis
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"❌ Intraday analysis error for {symbol}: {e}")
            return None
    
    def _is_trading_hours(self) -> bool:
        """Check if current time is within trading hours"""
        try:
            now = datetime.now()
            current_time = now.time()
            
            # Market hours: 9:30 AM - 4:00 PM EST
            market_open_time = self.market_open.time()
            market_close_time = self.market_close.time()
            
            return market_open_time <= current_time <= market_close_time
            
        except Exception as e:
            self.logger.error(f"❌ Trading hours check error: {e}")
            return False
    
    def _is_lunch_time(self) -> bool:
        """Check if current time is lunch hour"""
        try:
            now = datetime.now()
            current_time = now.time()
            
            lunch_start_time = self.lunch_start.time()
            lunch_end_time = self.lunch_end.time()
            
            return lunch_start_time <= current_time <= lunch_end_time
            
        except Exception as e:
            self.logger.error(f"❌ Lunch time check error: {e}")
            return False
    
    def _calculate_intraday_indicators(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate indicators specific to intraday trading"""
        try:
            indicators = self.calculate_technical_indicators(market_data)
            
            # VWAP (Volume Weighted Average Price)
            indicators['vwap'] = self._calculate_vwap(market_data)
            
            # ATR for position sizing
            indicators['atr'] = self._calculate_atr(market_data)
            
            # Intraday momentum
            indicators['intraday_momentum'] = market_data['close'].pct_change(periods=self.momentum_period)
            
            # Price distance from VWAP
            indicators['vwap_distance'] = (market_data['close'] - indicators['vwap']) / indicators['vwap']
            
            # High/Low of day
            indicators['hod'] = market_data['high'].expanding().max()
            indicators['lod'] = market_data['low'].expanding().min()
            
            # Morning range
            morning_data = market_data.head(self.morning_range_minutes)
            if len(morning_data) > 0:
                indicators['morning_high'] = morning_data['high'].max()
                indicators['morning_low'] = morning_data['low'].min()
                indicators['morning_range'] = indicators['morning_high'] - indicators['morning_low']
            
            # Volume profile
            indicators['volume_profile'] = self._calculate_volume_profile(market_data)
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"❌ Intraday indicators calculation error: {e}")
            return {}
    
    def _calculate_vwap(self, market_data: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        try:
            typical_price = (market_data['high'] + market_data['low'] + market_data['close']) / 3
            vwap = (typical_price * market_data['volume']).cumsum() / market_data['volume'].cumsum()
            return vwap
            
        except Exception as e:
            self.logger.error(f"❌ VWAP calculation error: {e}")
            return pd.Series(index=market_data.index, data=market_data['close'])
    
    def _calculate_atr(self, market_data: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range"""
        try:
            high_low = market_data['high'] - market_data['low']
            high_close = np.abs(market_data['high'] - market_data['close'].shift())
            low_close = np.abs(market_data['low'] - market_data['close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean()
            
            return atr
            
        except Exception as e:
            self.logger.error(f"❌ ATR calculation error: {e}")
            return pd.Series(index=market_data.index, data=1.0)
    
    def _calculate_volume_profile(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume profile for support/resistance"""
        try:
            # Simplified volume profile
            price_ranges = pd.cut(market_data['close'], bins=20)
            volume_profile = market_data.groupby(price_ranges)['volume'].sum()
            
            # Find volume peaks
            max_volume_level = volume_profile.idxmax()
            
            return {
                "profile": volume_profile.to_dict(),
                "poc": max_volume_level.mid,  # Point of Control
                "max_volume": volume_profile.max()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Volume profile calculation error: {e}")
            return {"profile": {}, "poc": 0, "max_volume": 0}
    
    def _analyze_morning_range(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze morning range for breakout opportunities"""
        try:
            current_time = datetime.now().time()
            market_open_time = self.market_open.time()
            
            # Check if we're past morning range period
            morning_end = (self.market_open + timedelta(minutes=self.morning_range_minutes)).time()
            
            if current_time < morning_end:
                return {"range_established": False}
            
            # Get morning data
            morning_data = market_data.head(self.morning_range_minutes)
            
            if len(morning_data) == 0:
                return {"range_established": False}
            
            morning_high = morning_data['high'].max()
            morning_low = morning_data['low'].min()
            morning_range = morning_high - morning_low
            current_price = market_data['close'].iloc[-1]
            
            # Analyze breakout potential
            breakout_signal = None
            if current_price > morning_high:
                breakout_signal = "bullish_breakout"
            elif current_price < morning_low:
                breakout_signal = "bearish_breakout"
            
            return {
                "range_established": True,
                "morning_high": morning_high,
                "morning_low": morning_low,
                "morning_range": morning_range,
                "current_price": current_price,
                "breakout_signal": breakout_signal,
                "range_size_pct": morning_range / morning_low if morning_low > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"❌ Morning range analysis error: {e}")
            return {"range_established": False}
    
    def _analyze_gaps(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze gap patterns"""
        try:
            if len(market_data) < 2:
                return {"gap_type": "none", "gap_size": 0}
            
            # Compare current open with previous close
            current_open = market_data['open'].iloc[-1]
            previous_close = market_data['close'].iloc[-2]
            
            gap_size = (current_open - previous_close) / previous_close
            
            # Classify gap
            gap_type = "none"
            if gap_size > self.gap_threshold:
                gap_type = "gap_up"
            elif gap_size < -self.gap_threshold:
                gap_type = "gap_down"
            
            # Analyze gap fill
            current_price = market_data['close'].iloc[-1]
            gap_fill_pct = 0
            
            if gap_type == "gap_up":
                gap_fill_pct = (current_open - current_price) / (current_open - previous_close)
            elif gap_type == "gap_down":
                gap_fill_pct = (current_price - current_open) / (previous_close - current_open)
            
            return {
                "gap_type": gap_type,
                "gap_size": abs(gap_size),
                "gap_fill_pct": gap_fill_pct,
                "current_open": current_open,
                "previous_close": previous_close,
                "fade_signal": gap_fill_pct > 0.5  # Gap is being filled
            }
            
        except Exception as e:
            self.logger.error(f"❌ Gap analysis error: {e}")
            return {"gap_type": "none", "gap_size": 0}
    
    def _analyze_breakouts(self, market_data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze breakout patterns"""
        try:
            current_price = market_data['close'].iloc[-1]
            
            # Check for breakouts from morning range
            morning_analysis = {}
            if 'morning_high' in indicators and 'morning_low' in indicators:
                morning_high = indicators['morning_high']
                morning_low = indicators['morning_low']
                
                if current_price > morning_high:
                    morning_analysis = {
                        "type": "morning_range_breakout",
                        "direction": "bullish",
                        "level": morning_high,
                        "strength": (current_price - morning_high) / morning_high
                    }
                elif current_price < morning_low:
                    morning_analysis = {
                        "type": "morning_range_breakout",
                        "direction": "bearish",
                        "level": morning_low,
                        "strength": (morning_low - current_price) / morning_low
                    }
            
            # Check for VWAP breakouts
            vwap_analysis = {}
            if 'vwap' in indicators:
                vwap = indicators['vwap'].iloc[-1]
                vwap_distance = indicators['vwap_distance'].iloc[-1]
                
                if abs(vwap_distance) > 0.01:  # 1% deviation
                    vwap_analysis = {
                        "type": "vwap_breakout",
                        "direction": "bullish" if vwap_distance > 0 else "bearish",
                        "level": vwap,
                        "strength": abs(vwap_distance)
                    }
            
            # Check for support/resistance breakouts
            sr_levels = self.calculate_support_resistance(market_data)
            sr_analysis = {}
            
            for resistance in sr_levels['resistance']:
                if current_price > resistance * 1.002:  # 0.2% above resistance
                    sr_analysis = {
                        "type": "resistance_breakout",
                        "direction": "bullish",
                        "level": resistance,
                        "strength": (current_price - resistance) / resistance
                    }
                    break
            
            for support in sr_levels['support']:
                if current_price < support * 0.998:  # 0.2% below support
                    sr_analysis = {
                        "type": "support_breakout",
                        "direction": "bearish",
                        "level": support,
                        "strength": (support - current_price) / support
                    }
                    break
            
            return {
                "morning_range": morning_analysis,
                "vwap": vwap_analysis,
                "support_resistance": sr_analysis,
                "has_breakout": bool(morning_analysis or vwap_analysis or sr_analysis)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Breakout analysis error: {e}")
            return {"has_breakout": False}
    
    def _analyze_intraday_momentum(self, market_data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze intraday momentum"""
        try:
            current_momentum = indicators['intraday_momentum'].iloc[-1]
            rsi = indicators['rsi'].iloc[-1]
            
            # Momentum classification
            momentum_signal = None
            if current_momentum > 0.01:  # 1% momentum
                momentum_signal = "strong_bullish"
            elif current_momentum > 0.005:  # 0.5% momentum
                momentum_signal = "bullish"
            elif current_momentum < -0.01:  # -1% momentum
                momentum_signal = "strong_bearish"
            elif current_momentum < -0.005:  # -0.5% momentum
                momentum_signal = "bearish"
            else:
                momentum_signal = "neutral"
            
            # RSI momentum
            rsi_momentum = None
            if rsi > 70:
                rsi_momentum = "overbought"
            elif rsi < 30:
                rsi_momentum = "oversold"
            
            # Price momentum vs VWAP
            vwap_momentum = None
            if 'vwap_distance' in indicators:
                vwap_distance = indicators['vwap_distance'].iloc[-1]
                if vwap_distance > 0.005:  # 0.5% above VWAP
                    vwap_momentum = "bullish"
                elif vwap_distance < -0.005:  # 0.5% below VWAP
                    vwap_momentum = "bearish"
            
            return {
                "momentum_signal": momentum_signal,
                "rsi_momentum": rsi_momentum,
                "vwap_momentum": vwap_momentum,
                "momentum_value": current_momentum,
                "rsi_value": rsi
            }
            
        except Exception as e:
            self.logger.error(f"❌ Intraday momentum analysis error: {e}")
            return {"momentum_signal": "neutral"}
    
    def _analyze_intraday_volume(self, market_data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze intraday volume patterns"""
        try:
            volume_ratio = indicators['volume_ratio'].iloc[-1]
            current_volume = market_data['volume'].iloc[-1]
            
            # Volume surge detection
            volume_surge = volume_ratio > self.volume_surge_threshold
            
            # Volume trend
            volume_trend = "neutral"
            if volume_ratio > 1.5:
                volume_trend = "increasing"
            elif volume_ratio < 0.7:
                volume_trend = "decreasing"
            
            # Price-volume relationship
            price_change = market_data['close'].pct_change().iloc[-1]
            volume_confirmation = None
            
            if price_change > 0.002 and volume_ratio > 1.3:  # 0.2% price move with volume
                volume_confirmation = "bullish_confirmation"
            elif price_change < -0.002 and volume_ratio > 1.3:
                volume_confirmation = "bearish_confirmation"
            
            return {
                "volume_surge": volume_surge,
                "volume_trend": volume_trend,
                "volume_ratio": volume_ratio,
                "volume_confirmation": volume_confirmation,
                "current_volume": current_volume,
                "relative_volume": volume_ratio
            }
            
        except Exception as e:
            self.logger.error(f"❌ Intraday volume analysis error: {e}")
            return {"volume_surge": False, "volume_trend": "neutral"}
    
    def _generate_intraday_signal(self, symbol: str, market_data: pd.DataFrame, indicators: Dict[str, Any],
                                 morning_analysis: Dict, gap_analysis: Dict, breakout_analysis: Dict,
                                 momentum_analysis: Dict, volume_analysis: Dict) -> Optional[TradingSignal]:
        """Generate intraday trading signal"""
        try:
            signal_type = "HOLD"
            confidence = 0.0
            reasoning = []
            
            current_price = market_data['close'].iloc[-1]
            
            # Scoring system
            bullish_score = 0.0
            bearish_score = 0.0
            
            # Morning range breakout scoring
            if breakout_analysis.get("morning_range"):
                morning_breakout = breakout_analysis["morning_range"]
                if morning_breakout["direction"] == "bullish":
                    bullish_score += 0.3
                    reasoning.append("Morning range bullish breakout")
                elif morning_breakout["direction"] == "bearish":
                    bearish_score += 0.3
                    reasoning.append("Morning range bearish breakout")
            
            # Gap analysis scoring
            if gap_analysis["gap_type"] != "none":
                if gap_analysis["gap_type"] == "gap_up" and not gap_analysis.get("fade_signal", False):
                    bullish_score += 0.2
                    reasoning.append("Gap up continuation")
                elif gap_analysis["gap_type"] == "gap_down" and not gap_analysis.get("fade_signal", False):
                    bearish_score += 0.2
                    reasoning.append("Gap down continuation")
                elif gap_analysis.get("fade_signal", False):
                    if gap_analysis["gap_type"] == "gap_up":
                        bearish_score += 0.15
                        reasoning.append("Gap up fade")
                    else:
                        bullish_score += 0.15
                        reasoning.append("Gap down fade")
            
            # VWAP analysis scoring
            if breakout_analysis.get("vwap"):
                vwap_breakout = breakout_analysis["vwap"]
                if vwap_breakout["direction"] == "bullish":
                    bullish_score += 0.2
                    reasoning.append("VWAP bullish breakout")
                elif vwap_breakout["direction"] == "bearish":
                    bearish_score += 0.2
                    reasoning.append("VWAP bearish breakout")
            
            # Momentum scoring
            momentum_signal = momentum_analysis.get("momentum_signal", "neutral")
            if momentum_signal == "strong_bullish":
                bullish_score += 0.25
                reasoning.append("Strong bullish momentum")
            elif momentum_signal == "bullish":
                bullish_score += 0.15
                reasoning.append("Bullish momentum")
            elif momentum_signal == "strong_bearish":
                bearish_score += 0.25
                reasoning.append("Strong bearish momentum")
            elif momentum_signal == "bearish":
                bearish_score += 0.15
                reasoning.append("Bearish momentum")
            
            # Volume confirmation scoring
            if volume_analysis.get("volume_confirmation") == "bullish_confirmation":
                bullish_score += 0.15
                reasoning.append("Volume confirms bullish move")
            elif volume_analysis.get("volume_confirmation") == "bearish_confirmation":
                bearish_score += 0.15
                reasoning.append("Volume confirms bearish move")
            
            # Support/resistance breakout scoring
            if breakout_analysis.get("support_resistance"):
                sr_breakout = breakout_analysis["support_resistance"]
                if sr_breakout["direction"] == "bullish":
                    bullish_score += 0.2
                    reasoning.append("Resistance breakout")
                elif sr_breakout["direction"] == "bearish":
                    bearish_score += 0.2
                    reasoning.append("Support breakdown")
            
            # Time-based adjustments
            current_time = datetime.now().time()
            power_hour_time = self.power_hour_start.time()
            
            # Power hour boost
            if current_time >= power_hour_time:
                if bullish_score > bearish_score:
                    bullish_score += 0.1
                    reasoning.append("Power hour momentum")
                elif bearish_score > bullish_score:
                    bearish_score += 0.1
                    reasoning.append("Power hour momentum")
            
            # Determine signal
            if bullish_score > bearish_score and bullish_score >= 0.7:
                signal_type = "BUY"
                confidence = min(bullish_score, 0.95)
            elif bearish_score > bullish_score and bearish_score >= 0.7:
                signal_type = "SELL"
                confidence = min(bearish_score, 0.95)
            
            # Skip weak signals
            if confidence < 0.7:
                return None
            
            # Calculate stop loss and take profit
            stop_loss, take_profit = self._calculate_intraday_levels(
                current_price, signal_type, indicators
            )
            
            # Create signal
            signal = TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                price_target=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                timeframe="15m",
                reasoning="; ".join(reasoning),
                metadata={
                    "strategy": "intraday",
                    "momentum_signal": momentum_signal,
                    "volume_surge": volume_analysis.get("volume_surge", False),
                    "breakout_type": breakout_analysis.get("morning_range", {}).get("type", "none"),
                    "gap_type": gap_analysis.get("gap_type", "none"),
                    "vwap_distance": indicators.get('vwap_distance', pd.Series([0])).iloc[-1],
                    "expected_duration": "intraday"
                }
            )
            
            # Validate signal
            if self.validate_signal(signal):
                return signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Intraday signal generation error: {e}")
            return None
    
    def _calculate_intraday_levels(self, current_price: float, signal_type: str, 
                                  indicators: Dict[str, Any]) -> Tuple[float, float]:
        """Calculate stop loss and take profit for intraday trades"""
        try:
            # Use ATR for dynamic levels
            atr = indicators.get('atr', pd.Series([current_price * 0.01])).iloc[-1]
            
            if signal_type == "BUY":
                stop_loss = current_price - (atr * self.stop_loss_atr)
                take_profit = current_price + (atr * self.profit_target_atr)
            else:  # SELL
                stop_loss = current_price + (atr * self.stop_loss_atr)
                take_profit = current_price - (atr * self.profit_target_atr)
            
            return stop_loss, take_profit
            
        except Exception as e:
            self.logger.error(f"❌ Intraday levels calculation error: {e}")
            return current_price * 0.98, current_price * 1.04
    
    async def should_exit_position(self, symbol: str, position: Dict, current_price: float) -> bool:
        """Determine if intraday position should be exited"""
        try:
            entry_time = position.get('entry_time', datetime.utcnow())
            current_time = datetime.now()
            
            # Always exit before market close
            if current_time.time() >= datetime.now().replace(hour=15, minute=45).time():
                return True
            
            # Exit if held for more than 2 hours
            if (current_time - entry_time).total_seconds() > 7200:
                return True
            
            # Check for reversal signals
            market_data = await self._get_market_data(symbol)
            if market_data is not None and len(market_data) > 10:
                indicators = self._calculate_intraday_indicators(market_data)
                
                # Check VWAP reversal
                vwap_distance = indicators.get('vwap_distance', pd.Series([0])).iloc[-1]
                
                if position.get('side') == 'long' and vwap_distance < -0.01:
                    return True  # Price moved significantly below VWAP
                elif position.get('side') == 'short' and vwap_distance > 0.01:
                    return True  # Price moved significantly above VWAP
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Intraday exit analysis error: {e}")
            return False
    
    def get_optimization_parameters(self) -> Dict[str, Tuple[float, float, float]]:
        """Get parameters for optimization"""
        return {
            "morning_range_minutes": (30, 90, 15),
            "breakout_threshold": (0.003, 0.008, 0.001),
            "volume_surge_threshold": (1.5, 3.0, 0.25),
            "atr_multiplier": (1.0, 2.0, 0.25),
            "momentum_period": (5, 15, 2),
            "profit_target_atr": (1.5, 3.0, 0.5),
            "stop_loss_atr": (0.5, 1.5, 0.25)
        }
    
    def get_intraday_metrics(self) -> Dict[str, Any]:
        """Get intraday-specific metrics"""
        return {
            "trading_hours": f"{self.market_open.time()} - {self.market_close.time()}",
            "avoid_lunch_time": self.avoid_lunch_time,
            "lunch_hours": f"{self.lunch_start.time()} - {self.lunch_end.time()}" if self.avoid_lunch_time else None,
            "power_hour_start": self.power_hour_start.time(),
            "max_intraday_loss": self.max_intraday_loss,
            "gap_threshold": self.gap_threshold,
            "morning_range_minutes": self.morning_range_minutes
        }
