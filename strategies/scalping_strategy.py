"""
Scalping Strategy
High-frequency trading strategy for capturing small price movements
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

from strategies.base_strategy import BaseStrategy, TradingSignal
from config import Config


class ScalpingStrategy(BaseStrategy):
    """
    Scalping strategy implementation
    Focuses on capturing small, quick price movements with high frequency
    """
    
    def __init__(self, config: Config):
        super().__init__(config, "scalping")
        
        # Scalping-specific parameters
        self.fast_ema_period = self.parameters.get("fast_ema_period", 5)
        self.slow_ema_period = self.parameters.get("slow_ema_period", 10)
        self.rsi_period = self.parameters.get("rsi_period", 7)
        self.bb_period = self.parameters.get("bb_period", 10)
        self.volume_spike_threshold = self.parameters.get("volume_spike_threshold", 2.0)
        self.spread_threshold = self.parameters.get("spread_threshold", 0.001)  # 0.1%
        
        # Scalping-specific settings
        self.max_holding_time = timedelta(minutes=30)  # Maximum holding time
        self.min_holding_time = timedelta(seconds=30)  # Minimum holding time
        self.profit_target_pips = 0.002  # 0.2% profit target
        self.stop_loss_pips = 0.001  # 0.1% stop loss
        self.max_daily_trades = 50  # Maximum trades per day
        
        # Performance tracking
        self.daily_trades = 0
        self.last_trade_date = None
        
        self.logger.info("⚡ Scalping Strategy initialized")
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """Get default scalping strategy parameters"""
        return {
            "fast_ema_period": 5,
            "slow_ema_period": 10,
            "rsi_period": 7,
            "bb_period": 10,
            "volume_spike_threshold": 2.0,
            "spread_threshold": 0.001,
            "momentum_threshold": 0.0005,
            "volatility_threshold": 0.01,
            "liquidity_threshold": 1.5
        }
    
    async def analyze_market(self, symbol: str, market_data: pd.DataFrame) -> Optional[TradingSignal]:
        """Analyze market for scalping opportunities"""
        try:
            if len(market_data) < 20:  # Need recent data
                return None
            
            # Check daily trade limit
            if not self._check_daily_limit():
                return None
            
            # Calculate short-term indicators
            indicators = self._calculate_scalping_indicators(market_data)
            
            # Get current values
            current_price = market_data['close'].iloc[-1]
            current_spread = self._calculate_spread(market_data)
            
            # Check spread condition
            if current_spread > self.spread_threshold:
                return None  # Spread too wide for scalping
            
            # Analyze micro-trend
            micro_trend = self._analyze_micro_trend(market_data, indicators)
            
            # Analyze momentum
            momentum = self._analyze_scalping_momentum(indicators)
            
            # Analyze volume
            volume_analysis = self._analyze_scalping_volume(market_data, indicators)
            
            # Check volatility
            volatility = self._calculate_volatility(market_data)
            if volatility['atr'] > current_price * 0.02:  # Too volatile for scalping
                return None
            
            # Generate scalping signal
            signal = self._generate_scalping_signal(
                symbol=symbol,
                current_price=current_price,
                micro_trend=micro_trend,
                momentum=momentum,
                volume_analysis=volume_analysis,
                indicators=indicators
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"❌ Scalping analysis error for {symbol}: {e}")
            return None
    
    def _check_daily_limit(self) -> bool:
        """Check if daily trade limit is reached"""
        try:
            current_date = datetime.utcnow().date()
            
            # Reset counter if new day
            if self.last_trade_date != current_date:
                self.daily_trades = 0
                self.last_trade_date = current_date
            
            return self.daily_trades < self.max_daily_trades
            
        except Exception as e:
            self.logger.error(f"❌ Daily limit check error: {e}")
            return False
    
    def _calculate_scalping_indicators(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate indicators specific to scalping"""
        try:
            indicators = {}
            
            # Fast and slow EMAs
            indicators['fast_ema'] = market_data['close'].ewm(span=self.fast_ema_period).mean()
            indicators['slow_ema'] = market_data['close'].ewm(span=self.slow_ema_period).mean()
            
            # EMA crossover
            indicators['ema_diff'] = indicators['fast_ema'] - indicators['slow_ema']
            
            # RSI (shorter period)
            delta = market_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands (shorter period)
            indicators['bb_middle'] = market_data['close'].rolling(window=self.bb_period).mean()
            bb_std = market_data['close'].rolling(window=self.bb_period).std()
            indicators['bb_upper'] = indicators['bb_middle'] + (bb_std * 1.5)  # Tighter bands
            indicators['bb_lower'] = indicators['bb_middle'] - (bb_std * 1.5)
            
            # Volume indicators
            indicators['volume_sma'] = market_data['volume'].rolling(window=10).mean()
            indicators['volume_ratio'] = market_data['volume'] / indicators['volume_sma']
            
            # Price momentum
            indicators['momentum'] = market_data['close'].pct_change(periods=3)
            
            # Volatility
            indicators['volatility'] = market_data['close'].rolling(window=10).std()
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"❌ Scalping indicators calculation error: {e}")
            return {}
    
    def _calculate_spread(self, market_data: pd.DataFrame) -> float:
        """Calculate bid-ask spread estimate"""
        try:
            # Estimate spread from high-low range
            recent_data = market_data.tail(5)
            avg_range = (recent_data['high'] - recent_data['low']).mean()
            current_price = market_data['close'].iloc[-1]
            
            return avg_range / current_price if current_price > 0 else 0.01
            
        except Exception as e:
            self.logger.error(f"❌ Spread calculation error: {e}")
            return 0.01
    
    def _analyze_micro_trend(self, market_data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze very short-term trend"""
        try:
            # EMA crossover analysis
            fast_ema = indicators['fast_ema'].iloc[-1]
            slow_ema = indicators['slow_ema'].iloc[-1]
            ema_diff = indicators['ema_diff'].iloc[-1]
            prev_ema_diff = indicators['ema_diff'].iloc[-2]
            
            # Trend direction
            if fast_ema > slow_ema:
                trend_direction = "bullish"
            elif fast_ema < slow_ema:
                trend_direction = "bearish"
            else:
                trend_direction = "neutral"
            
            # Crossover detection
            crossover = None
            if ema_diff > 0 and prev_ema_diff <= 0:
                crossover = "bullish_crossover"
            elif ema_diff < 0 and prev_ema_diff >= 0:
                crossover = "bearish_crossover"
            
            # Trend strength
            trend_strength = abs(ema_diff) / slow_ema if slow_ema > 0 else 0
            
            # Recent price action
            recent_prices = market_data['close'].tail(5)
            price_momentum = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
            
            return {
                "direction": trend_direction,
                "strength": trend_strength,
                "crossover": crossover,
                "momentum": price_momentum,
                "fast_ema": fast_ema,
                "slow_ema": slow_ema
            }
            
        except Exception as e:
            self.logger.error(f"❌ Micro trend analysis error: {e}")
            return {"direction": "neutral", "strength": 0.0, "crossover": None}
    
    def _analyze_scalping_momentum(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze momentum for scalping"""
        try:
            current_rsi = indicators['rsi'].iloc[-1]
            current_momentum = indicators['momentum'].iloc[-1]
            
            # RSI signals (different thresholds for scalping)
            rsi_signal = None
            if current_rsi < 40:
                rsi_signal = "oversold"
            elif current_rsi > 60:
                rsi_signal = "overbought"
            
            # Momentum analysis
            momentum_signal = None
            if current_momentum > 0.001:  # 0.1% momentum
                momentum_signal = "bullish"
            elif current_momentum < -0.001:
                momentum_signal = "bearish"
            
            # Combined momentum score
            momentum_score = 0.0
            if rsi_signal == "oversold":
                momentum_score += 0.3
            elif rsi_signal == "overbought":
                momentum_score -= 0.3
            
            if momentum_signal == "bullish":
                momentum_score += 0.4
            elif momentum_signal == "bearish":
                momentum_score -= 0.4
            
            return {
                "rsi_signal": rsi_signal,
                "momentum_signal": momentum_signal,
                "momentum_score": momentum_score,
                "rsi_value": current_rsi,
                "momentum_value": current_momentum
            }
            
        except Exception as e:
            self.logger.error(f"❌ Scalping momentum analysis error: {e}")
            return {"momentum_score": 0.0}
    
    def _analyze_scalping_volume(self, market_data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze volume for scalping signals"""
        try:
            volume_ratio = indicators['volume_ratio'].iloc[-1]
            current_volume = market_data['volume'].iloc[-1]
            
            # Volume spike detection
            volume_spike = volume_ratio > self.volume_spike_threshold
            
            # Volume trend
            volume_trend = "neutral"
            if volume_ratio > 1.5:
                volume_trend = "increasing"
            elif volume_ratio < 0.7:
                volume_trend = "decreasing"
            
            # Price-volume confirmation
            price_change = market_data['close'].pct_change().iloc[-1]
            volume_confirmation = None
            
            if price_change > 0 and volume_ratio > 1.2:
                volume_confirmation = "bullish"
            elif price_change < 0 and volume_ratio > 1.2:
                volume_confirmation = "bearish"
            
            return {
                "spike": volume_spike,
                "trend": volume_trend,
                "ratio": volume_ratio,
                "confirmation": volume_confirmation,
                "current_volume": current_volume
            }
            
        except Exception as e:
            self.logger.error(f"❌ Scalping volume analysis error: {e}")
            return {"spike": False, "trend": "neutral", "ratio": 1.0}
    
    def _generate_scalping_signal(self, symbol: str, current_price: float, micro_trend: Dict,
                                 momentum: Dict, volume_analysis: Dict, indicators: Dict) -> Optional[TradingSignal]:
        """Generate scalping signal"""
        try:
            signal_type = "HOLD"
            confidence = 0.0
            reasoning = []
            
            # Scoring system
            bullish_score = 0.0
            bearish_score = 0.0
            
            # Micro-trend scoring
            if micro_trend["crossover"] == "bullish_crossover":
                bullish_score += 0.4
                reasoning.append("Bullish EMA crossover")
            elif micro_trend["crossover"] == "bearish_crossover":
                bearish_score += 0.4
                reasoning.append("Bearish EMA crossover")
            
            if micro_trend["direction"] == "bullish":
                bullish_score += 0.2
            elif micro_trend["direction"] == "bearish":
                bearish_score += 0.2
            
            # Momentum scoring
            if momentum["momentum_signal"] == "bullish":
                bullish_score += 0.3
                reasoning.append("Bullish momentum")
            elif momentum["momentum_signal"] == "bearish":
                bearish_score += 0.3
                reasoning.append("Bearish momentum")
            
            # Volume scoring
            if volume_analysis["confirmation"] == "bullish":
                bullish_score += 0.2
                reasoning.append("Volume confirms bullish move")
            elif volume_analysis["confirmation"] == "bearish":
                bearish_score += 0.2
                reasoning.append("Volume confirms bearish move")
            
            # Bollinger Bands positioning
            bb_upper = indicators['bb_upper'].iloc[-1]
            bb_lower = indicators['bb_lower'].iloc[-1]
            
            if current_price <= bb_lower:
                bullish_score += 0.15
                reasoning.append("Price at lower BB")
            elif current_price >= bb_upper:
                bearish_score += 0.15
                reasoning.append("Price at upper BB")
            
            # Volume spike bonus
            if volume_analysis["spike"]:
                if bullish_score > bearish_score:
                    bullish_score += 0.1
                    reasoning.append("Volume spike supports signal")
                elif bearish_score > bullish_score:
                    bearish_score += 0.1
                    reasoning.append("Volume spike supports signal")
            
            # Determine signal
            if bullish_score > bearish_score and bullish_score >= 0.7:
                signal_type = "BUY"
                confidence = min(bullish_score, 0.95)
            elif bearish_score > bullish_score and bearish_score >= 0.7:
                signal_type = "SELL"
                confidence = min(bearish_score, 0.95)
            
            # Skip weak signals
            if confidence < 0.7:  # Higher threshold for scalping
                return None
            
            # Calculate tight stop loss and take profit
            stop_loss, take_profit = self._calculate_scalping_levels(current_price, signal_type)
            
            # Create signal
            signal = TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                price_target=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                timeframe="1m",  # Very short timeframe
                reasoning="; ".join(reasoning),
                metadata={
                    "strategy": "scalping",
                    "micro_trend": micro_trend["direction"],
                    "momentum_score": momentum["momentum_score"],
                    "volume_ratio": volume_analysis["ratio"],
                    "expected_duration": "5-30 minutes",
                    "risk_level": "high_frequency"
                }
            )
            
            # Validate signal
            if self.validate_signal(signal):
                self.daily_trades += 1
                return signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Scalping signal generation error: {e}")
            return None
    
    def _calculate_scalping_levels(self, current_price: float, signal_type: str) -> Tuple[float, float]:
        """Calculate tight stop loss and take profit for scalping"""
        try:
            if signal_type == "BUY":
                stop_loss = current_price * (1 - self.stop_loss_pips)
                take_profit = current_price * (1 + self.profit_target_pips)
            else:  # SELL
                stop_loss = current_price * (1 + self.stop_loss_pips)
                take_profit = current_price * (1 - self.profit_target_pips)
            
            return stop_loss, take_profit
            
        except Exception as e:
            self.logger.error(f"❌ Scalping levels calculation error: {e}")
            return current_price * 0.999, current_price * 1.002
    
    async def should_exit_position(self, symbol: str, position: Dict, current_price: float) -> bool:
        """Determine if scalping position should be exited"""
        try:
            entry_time = position.get('entry_time', datetime.utcnow())
            entry_price = position.get('entry_price', current_price)
            side = position.get('side', 'long')
            
            # Check maximum holding time
            holding_time = datetime.utcnow() - entry_time
            if holding_time > self.max_holding_time:
                return True
            
            # Check minimum holding time
            if holding_time < self.min_holding_time:
                return False
            
            # Quick profit taking for scalping
            profit_pct = abs(current_price - entry_price) / entry_price
            
            # Exit if small profit target reached
            if profit_pct >= self.profit_target_pips:
                return True
            
            # Exit if trend reverses quickly
            market_data = await self._get_market_data(symbol)
            if market_data is not None and len(market_data) > 10:
                indicators = self._calculate_scalping_indicators(market_data)
                
                # Check for EMA crossover reversal
                ema_diff = indicators['ema_diff'].iloc[-1]
                prev_ema_diff = indicators['ema_diff'].iloc[-2]
                
                if side == 'long' and ema_diff < 0 and prev_ema_diff > 0:
                    return True  # Fast EMA crossed below slow EMA
                elif side == 'short' and ema_diff > 0 and prev_ema_diff < 0:
                    return True  # Fast EMA crossed above slow EMA
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Scalping exit analysis error: {e}")
            return False
    
    def get_optimization_parameters(self) -> Dict[str, Tuple[float, float, float]]:
        """Get parameters for optimization"""
        return {
            "fast_ema_period": (3, 8, 1),
            "slow_ema_period": (8, 15, 1),
            "rsi_period": (5, 10, 1),
            "bb_period": (8, 15, 1),
            "volume_spike_threshold": (1.5, 3.0, 0.25),
            "spread_threshold": (0.0005, 0.002, 0.0005),
            "momentum_threshold": (0.0002, 0.001, 0.0002)
        }
    
    def get_scalping_metrics(self) -> Dict[str, Any]:
        """Get scalping-specific metrics"""
        return {
            "daily_trades": self.daily_trades,
            "max_daily_trades": self.max_daily_trades,
            "trades_remaining": self.max_daily_trades - self.daily_trades,
            "last_trade_date": self.last_trade_date.isoformat() if self.last_trade_date else None,
            "max_holding_time_minutes": self.max_holding_time.total_seconds() / 60,
            "profit_target_pips": self.profit_target_pips,
            "stop_loss_pips": self.stop_loss_pips
        }
