"""
Swing Trading Strategy
Holds positions for days to weeks, focuses on trend following and momentum
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import numpy as np

from .base_strategy import BaseStrategy, TradingSignal, SignalType, StrategyConfig, StrategyType
from exchanges.base_exchange import BaseExchange

class SwingTradingStrategy(BaseStrategy):
    """Swing trading strategy implementation"""
    
    def __init__(self, config: StrategyConfig, exchange: BaseExchange):
        # Set default parameters for swing trading
        default_params = {
            "min_confidence": 0.75,
            "max_positions": 3,
            "position_size_pct": 0.05,  # 5% per position
            "stop_loss_pct": 0.08,      # 8% stop loss
            "take_profit_pct": 0.15,    # 15% take profit
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "macd_threshold": 0.001,
            "fibonacci_retracement_levels": [0.236, 0.382, 0.618],
            "volume_threshold": 1.2,    # Volume should be 20% above average
            "trend_strength_threshold": 0.6,
            "lookback_period": 50
        }
        
        # Merge with provided parameters
        config.parameters = {**default_params, **config.parameters}
        config.strategy_type = StrategyType.SWING_TRADING
        
        super().__init__(config, exchange)
        
        self.logger.info(f"✅ Swing Trading Strategy initialized: {config.name}")
    
    async def analyze_market(self, symbol: str, timeframe: str = "1D") -> TradingSignal:
        """Analyze market for swing trading opportunities"""
        try:
            # Get market data with indicators
            market_data = await self.get_market_data_with_indicators(
                symbol, timeframe, self.config.parameters["lookback_period"]
            )
            
            if not market_data or not market_data.get("indicators"):
                return None
            
            indicators = market_data["indicators"]
            current_price = market_data["current_price"]
            
            # Analyze trend strength
            trend_strength = self._analyze_trend_strength(indicators)
            
            # Check for swing trading signals
            signal_type = await self._identify_swing_signal(indicators, current_price)
            
            if signal_type == SignalType.HOLD:
                return None
            
            # Calculate confidence
            confidence = self._calculate_confidence(indicators, signal_type, trend_strength)
            
            if confidence < self.config.parameters["min_confidence"]:
                return None
            
            # Calculate stop loss and take profit
            stop_loss, take_profit = self._calculate_exit_levels(current_price, signal_type, indicators)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(indicators, signal_type, trend_strength, confidence)
            
            return TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                price=current_price,
                confidence=confidence,
                reasoning=reasoning,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing market for {symbol}: {e}")
            return None
    
    def _analyze_trend_strength(self, indicators: Dict[str, Any]) -> float:
        """Analyze trend strength using multiple indicators"""
        try:
            trend_signals = []
            
            # SMA trend analysis
            if indicators.get("price_sma_20") and indicators.get("price_sma_50"):
                sma_20 = indicators["price_sma_20"][-1]
                sma_50 = indicators["price_sma_50"][-1]
                
                if sma_20 > sma_50:
                    trend_signals.append(1)  # Uptrend
                elif sma_20 < sma_50:
                    trend_signals.append(-1)  # Downtrend
                else:
                    trend_signals.append(0)   # Sideways
            
            # MACD trend analysis
            if indicators.get("macd"):
                macd_line = indicators["macd"]["macd_line"][-1]
                signal_line = indicators["macd"]["signal_line"][-1]
                
                if macd_line > signal_line:
                    trend_signals.append(1)
                elif macd_line < signal_line:
                    trend_signals.append(-1)
                else:
                    trend_signals.append(0)
            
            # RSI trend analysis
            if indicators.get("rsi"):
                rsi = indicators["rsi"][-1]
                
                if rsi > 50:
                    trend_signals.append(1)
                elif rsi < 50:
                    trend_signals.append(-1)
                else:
                    trend_signals.append(0)
            
            # Calculate trend strength
            if trend_signals:
                trend_strength = abs(sum(trend_signals)) / len(trend_signals)
                return min(trend_strength, 1.0)
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error analyzing trend strength: {e}")
            return 0.0
    
    async def _identify_swing_signal(self, indicators: Dict[str, Any], current_price: float) -> SignalType:
        """Identify swing trading signals"""
        try:
            buy_signals = 0
            sell_signals = 0
            
            # RSI analysis
            if indicators.get("rsi"):
                rsi = indicators["rsi"][-1]
                
                if rsi < self.config.parameters["rsi_oversold"]:
                    buy_signals += 1
                elif rsi > self.config.parameters["rsi_overbought"]:
                    sell_signals += 1
            
            # MACD analysis
            if indicators.get("macd"):
                macd_line = indicators["macd"]["macd_line"][-1]
                signal_line = indicators["macd"]["signal_line"][-1]
                histogram = indicators["macd"]["histogram"][-1]
                
                # MACD crossover
                if (macd_line > signal_line and 
                    histogram > self.config.parameters["macd_threshold"]):
                    buy_signals += 1
                elif (macd_line < signal_line and 
                      histogram < -self.config.parameters["macd_threshold"]):
                    sell_signals += 1
            
            # Bollinger Bands analysis
            if indicators.get("bollinger_bands"):
                bb = indicators["bollinger_bands"]
                lower_band = bb["lower_band"][-1]
                upper_band = bb["upper_band"][-1]
                
                if current_price <= lower_band:
                    buy_signals += 1
                elif current_price >= upper_band:
                    sell_signals += 1
            
            # Moving Average analysis
            if (indicators.get("price_sma_20") and 
                indicators.get("price_sma_50")):
                sma_20 = indicators["price_sma_20"][-1]
                sma_50 = indicators["price_sma_50"][-1]
                
                # Golden cross
                if sma_20 > sma_50 and current_price > sma_20:
                    buy_signals += 1
                # Death cross
                elif sma_20 < sma_50 and current_price < sma_20:
                    sell_signals += 1
            
            # Fibonacci retracement analysis
            if indicators.get("fibonacci_levels"):
                fib_levels = indicators["fibonacci_levels"]
                
                for level in self.config.parameters["fibonacci_retracement_levels"]:
                    level_key = f"{level*100:.1f}"
                    if level_key in fib_levels:
                        fib_price = fib_levels[level_key]
                        
                        # Price near Fibonacci support (buy signal)
                        if abs(current_price - fib_price) / current_price < 0.02:
                            if current_price > fib_price:
                                buy_signals += 1
            
            # Volume analysis
            if indicators.get("volume_sma"):
                current_volume = indicators.get("current_volume", 0)
                avg_volume = indicators["volume_sma"][-1]
                
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
                
                if volume_ratio > self.config.parameters["volume_threshold"]:
                    # High volume supports the signal
                    if buy_signals > sell_signals:
                        buy_signals += 1
                    elif sell_signals > buy_signals:
                        sell_signals += 1
            
            # Determine signal
            if buy_signals > sell_signals and buy_signals >= 2:
                return SignalType.BUY
            elif sell_signals > buy_signals and sell_signals >= 2:
                return SignalType.SELL
            else:
                return SignalType.HOLD
                
        except Exception as e:
            self.logger.error(f"Error identifying swing signal: {e}")
            return SignalType.HOLD
    
    def _calculate_confidence(self, indicators: Dict[str, Any], signal_type: SignalType, trend_strength: float) -> float:
        """Calculate confidence score for the signal"""
        try:
            confidence_factors = []
            
            # Trend strength factor
            confidence_factors.append(trend_strength)
            
            # RSI confidence
            if indicators.get("rsi"):
                rsi = indicators["rsi"][-1]
                
                if signal_type == SignalType.BUY:
                    # More confident when RSI is oversold
                    rsi_confidence = max(0, (50 - rsi) / 20)  # 0-1 scale
                elif signal_type == SignalType.SELL:
                    # More confident when RSI is overbought
                    rsi_confidence = max(0, (rsi - 50) / 20)  # 0-1 scale
                else:
                    rsi_confidence = 0.5
                
                confidence_factors.append(min(rsi_confidence, 1.0))
            
            # MACD confidence
            if indicators.get("macd"):
                histogram = indicators["macd"]["histogram"][-1]
                
                if signal_type == SignalType.BUY and histogram > 0:
                    macd_confidence = min(abs(histogram) * 1000, 1.0)
                elif signal_type == SignalType.SELL and histogram < 0:
                    macd_confidence = min(abs(histogram) * 1000, 1.0)
                else:
                    macd_confidence = 0.3
                
                confidence_factors.append(macd_confidence)
            
            # Volume confidence
            if indicators.get("volume_sma"):
                current_volume = indicators.get("current_volume", 0)
                avg_volume = indicators["volume_sma"][-1]
                
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
                volume_confidence = min(volume_ratio / 2, 1.0)  # Normalize to 0-1
                
                confidence_factors.append(volume_confidence)
            
            # Calculate overall confidence
            if confidence_factors:
                return sum(confidence_factors) / len(confidence_factors)
            
            return 0.5  # Default confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _calculate_exit_levels(self, current_price: float, signal_type: SignalType, indicators: Dict[str, Any]) -> tuple:
        """Calculate stop loss and take profit levels"""
        try:
            if signal_type == SignalType.BUY:
                # Stop loss below recent support or percentage-based
                stop_loss = current_price * (1 - self.config.parameters["stop_loss_pct"])
                take_profit = current_price * (1 + self.config.parameters["take_profit_pct"])
                
                # Adjust based on Fibonacci levels
                if indicators.get("fibonacci_levels"):
                    fib_levels = indicators["fibonacci_levels"]
                    
                    # Find nearest Fibonacci support for stop loss
                    for level in ["38.2", "50.0", "61.8"]:
                        if level in fib_levels:
                            fib_price = fib_levels[level]
                            if fib_price < current_price:
                                stop_loss = max(stop_loss, fib_price * 0.99)  # Slightly below Fib level
                                break
                
            elif signal_type == SignalType.SELL:
                # Stop loss above recent resistance or percentage-based
                stop_loss = current_price * (1 + self.config.parameters["stop_loss_pct"])
                take_profit = current_price * (1 - self.config.parameters["take_profit_pct"])
                
                # Adjust based on Fibonacci levels
                if indicators.get("fibonacci_levels"):
                    fib_levels = indicators["fibonacci_levels"]
                    
                    # Find nearest Fibonacci resistance for stop loss
                    for level in ["38.2", "50.0", "61.8"]:
                        if level in fib_levels:
                            fib_price = fib_levels[level]
                            if fib_price > current_price:
                                stop_loss = min(stop_loss, fib_price * 1.01)  # Slightly above Fib level
                                break
            
            else:
                return None, None
            
            return stop_loss, take_profit
            
        except Exception as e:
            self.logger.error(f"Error calculating exit levels: {e}")
            return None, None
    
    def _generate_reasoning(self, indicators: Dict[str, Any], signal_type: SignalType, 
                          trend_strength: float, confidence: float) -> str:
        """Generate human-readable reasoning for the signal"""
        try:
            reasoning_parts = []
            
            # Trend analysis
            if trend_strength > 0.7:
                reasoning_parts.append(f"Strong trend detected (strength: {trend_strength:.2f})")
            elif trend_strength > 0.5:
                reasoning_parts.append(f"Moderate trend detected (strength: {trend_strength:.2f})")
            else:
                reasoning_parts.append(f"Weak trend detected (strength: {trend_strength:.2f})")
            
            # RSI analysis
            if indicators.get("rsi"):
                rsi = indicators["rsi"][-1]
                
                if rsi < 30:
                    reasoning_parts.append(f"RSI oversold ({rsi:.1f})")
                elif rsi > 70:
                    reasoning_parts.append(f"RSI overbought ({rsi:.1f})")
                else:
                    reasoning_parts.append(f"RSI neutral ({rsi:.1f})")
            
            # MACD analysis
            if indicators.get("macd"):
                macd_line = indicators["macd"]["macd_line"][-1]
                signal_line = indicators["macd"]["signal_line"][-1]
                
                if macd_line > signal_line:
                    reasoning_parts.append("MACD bullish crossover")
                elif macd_line < signal_line:
                    reasoning_parts.append("MACD bearish crossover")
            
            # Bollinger Bands analysis
            if indicators.get("bollinger_bands"):
                bb = indicators["bollinger_bands"]
                middle_band = bb["middle_band"][-1]
                
                if signal_type == SignalType.BUY:
                    reasoning_parts.append("Price near lower Bollinger Band (potential bounce)")
                elif signal_type == SignalType.SELL:
                    reasoning_parts.append("Price near upper Bollinger Band (potential reversal)")
            
            # Moving Average analysis
            if (indicators.get("price_sma_20") and indicators.get("price_sma_50")):
                sma_20 = indicators["price_sma_20"][-1]
                sma_50 = indicators["price_sma_50"][-1]
                
                if sma_20 > sma_50:
                    reasoning_parts.append("20-day SMA above 50-day SMA (bullish)")
                elif sma_20 < sma_50:
                    reasoning_parts.append("20-day SMA below 50-day SMA (bearish)")
            
            # Fibonacci analysis
            if indicators.get("fibonacci_levels"):
                reasoning_parts.append("Fibonacci levels support the signal")
            
            # Final reasoning
            signal_action = "BUY" if signal_type == SignalType.BUY else "SELL"
            reasoning = f"{signal_action} signal - {'; '.join(reasoning_parts)}. Confidence: {confidence:.2f}"
            
            return reasoning
            
        except Exception as e:
            self.logger.error(f"Error generating reasoning: {e}")
            return f"{signal_type.value.upper()} signal detected"
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        """Validate swing trading signal"""
        try:
            # Check confidence threshold
            if signal.confidence < self.config.parameters["min_confidence"]:
                return False
            
            # Check if we already have a position in this symbol
            positions = await self.exchange.get_positions()
            for position in positions:
                if position.symbol == signal.symbol and position.quantity != 0:
                    return False  # Already have a position
            
            # Check maximum positions
            active_positions = sum(1 for pos in positions if pos.quantity != 0)
            if active_positions >= self.config.parameters["max_positions"]:
                return False
            
            # Check market hours
            if not await self.exchange.check_trading_hours(signal.symbol):
                return False
            
            # Check if price hasn't moved too much since signal generation
            current_market_data = await self.exchange.get_market_data(signal.symbol)
            price_change = abs(current_market_data.price - signal.price) / signal.price
            
            if price_change > 0.02:  # 2% price change threshold
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating signal: {e}")
            return False
    
    def get_strategy_parameters(self) -> Dict[str, Any]:
        """Get swing trading strategy parameters"""
        return {
            "strategy_type": "swing_trading",
            "description": "Medium-term trend following strategy",
            "typical_holding_period": "3-14 days",
            "risk_per_trade": f"{self.config.parameters['position_size_pct']*100:.1f}%",
            "stop_loss": f"{self.config.parameters['stop_loss_pct']*100:.1f}%",
            "take_profit": f"{self.config.parameters['take_profit_pct']*100:.1f}%",
            "parameters": self.config.parameters
        }
