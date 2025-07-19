"""
Scalping Strategy
Very short-term trading strategy focused on small price movements
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import numpy as np

from .base_strategy import BaseStrategy, TradingSignal, SignalType, StrategyConfig, StrategyType
from exchanges.base_exchange import BaseExchange

class ScalpingStrategy(BaseStrategy):
    """Scalping strategy implementation"""
    
    def __init__(self, config: StrategyConfig, exchange: BaseExchange):
        # Set default parameters for scalping
        default_params = {
            "min_confidence": 0.8,
            "max_positions": 10,
            "position_size_pct": 0.01,  # 1% per position
            "stop_loss_pct": 0.005,     # 0.5% stop loss
            "take_profit_pct": 0.01,    # 1% take profit
            "spread_threshold": 0.0005, # 0.05% max spread
            "volume_threshold": 2.0,    # Volume should be 2x average
            "volatility_threshold": 0.002, # 0.2% minimum volatility
            "rsi_oversold": 25,
            "rsi_overbought": 75,
            "stoch_oversold": 20,
            "stoch_overbought": 80,
            "williams_r_oversold": -80,
            "williams_r_overbought": -20,
            "lookback_period": 20
        }
        
        # Merge with provided parameters
        config.parameters = {**default_params, **config.parameters}
        config.strategy_type = StrategyType.SCALPING
        
        super().__init__(config, exchange)
        
        self.logger.info(f"✅ Scalping Strategy initialized: {config.name}")
    
    async def analyze_market(self, symbol: str, timeframe: str = "1m") -> TradingSignal:
        """Analyze market for scalping opportunities"""
        try:
            # Get market data with indicators (shorter timeframe for scalping)
            market_data = await self.get_market_data_with_indicators(
                symbol, timeframe, self.config.parameters["lookback_period"]
            )
            
            if not market_data or not market_data.get("indicators"):
                return None
            
            indicators = market_data["indicators"]
            current_price = market_data["current_price"]
            
            # Check market conditions for scalping
            if not await self._check_scalping_conditions(indicators, current_price):
                return None
            
            # Check for scalping signals
            signal_type = await self._identify_scalping_signal(indicators, current_price)
            
            if signal_type == SignalType.HOLD:
                return None
            
            # Calculate confidence
            confidence = self._calculate_confidence(indicators, signal_type)
            
            if confidence < self.config.parameters["min_confidence"]:
                return None
            
            # Calculate stop loss and take profit (tight levels for scalping)
            stop_loss, take_profit = self._calculate_exit_levels(current_price, signal_type)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(indicators, signal_type, confidence)
            
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
            self.logger.error(f"Error analyzing market for scalping {symbol}: {e}")
            return None
    
    async def _check_scalping_conditions(self, indicators: Dict[str, Any], current_price: float) -> bool:
        """Check if market conditions are suitable for scalping"""
        try:
            # Check volatility
            if indicators.get("bollinger_bands"):
                bb = indicators["bollinger_bands"]
                upper_band = bb["upper_band"][-1]
                lower_band = bb["lower_band"][-1]
                
                # Calculate volatility as percentage of price
                volatility = (upper_band - lower_band) / current_price
                
                if volatility < self.config.parameters["volatility_threshold"]:
                    return False  # Not enough volatility for scalping
            
            # Check volume
            if indicators.get("volume_sma"):
                current_volume = indicators.get("current_volume", 0)
                avg_volume = indicators["volume_sma"][-1]
                
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
                
                if volume_ratio < self.config.parameters["volume_threshold"]:
                    return False  # Not enough volume for scalping
            
            # Check spread (if available)
            # This would require real-time order book data
            # For now, we'll assume spread is acceptable
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking scalping conditions: {e}")
            return False
    
    async def _identify_scalping_signal(self, indicators: Dict[str, Any], current_price: float) -> SignalType:
        """Identify scalping signals using fast indicators"""
        try:
            buy_signals = 0
            sell_signals = 0
            
            # RSI analysis (faster settings)
            if indicators.get("rsi"):
                rsi = indicators["rsi"][-1]
                
                if rsi < self.config.parameters["rsi_oversold"]:
                    buy_signals += 2  # Strong buy signal
                elif rsi > self.config.parameters["rsi_overbought"]:
                    sell_signals += 2  # Strong sell signal
            
            # Stochastic oscillator (excellent for scalping)
            if indicators.get("stochastic"):
                stoch_k = indicators["stochastic"]["k"][-1]
                stoch_d = indicators["stochastic"]["d"][-1]
                
                # Stochastic oversold
                if stoch_k < self.config.parameters["stoch_oversold"] and stoch_d < self.config.parameters["stoch_oversold"]:
                    buy_signals += 2
                # Stochastic overbought
                elif stoch_k > self.config.parameters["stoch_overbought"] and stoch_d > self.config.parameters["stoch_overbought"]:
                    sell_signals += 2
                
                # Stochastic crossover
                if len(indicators["stochastic"]["k"]) >= 2:
                    prev_k = indicators["stochastic"]["k"][-2]
                    prev_d = indicators["stochastic"]["d"][-2]
                    
                    # Bullish crossover
                    if stoch_k > stoch_d and prev_k <= prev_d:
                        buy_signals += 1
                    # Bearish crossover
                    elif stoch_k < stoch_d and prev_k >= prev_d:
                        sell_signals += 1
            
            # Williams %R (another good scalping indicator)
            if indicators.get("williams_r"):
                williams_r = indicators["williams_r"][-1]
                
                if williams_r < self.config.parameters["williams_r_oversold"]:
                    buy_signals += 1
                elif williams_r > self.config.parameters["williams_r_overbought"]:
                    sell_signals += 1
            
            # MACD for momentum (fast settings)
            if indicators.get("macd"):
                macd_line = indicators["macd"]["macd_line"][-1]
                signal_line = indicators["macd"]["signal_line"][-1]
                histogram = indicators["macd"]["histogram"][-1]
                
                # MACD histogram turning positive
                if histogram > 0 and len(indicators["macd"]["histogram"]) >= 2:
                    prev_histogram = indicators["macd"]["histogram"][-2]
                    if prev_histogram <= 0:
                        buy_signals += 1
                
                # MACD histogram turning negative
                elif histogram < 0 and len(indicators["macd"]["histogram"]) >= 2:
                    prev_histogram = indicators["macd"]["histogram"][-2]
                    if prev_histogram >= 0:
                        sell_signals += 1
            
            # Bollinger Bands for mean reversion
            if indicators.get("bollinger_bands"):
                bb = indicators["bollinger_bands"]
                lower_band = bb["lower_band"][-1]
                upper_band = bb["upper_band"][-1]
                middle_band = bb["middle_band"][-1]
                
                # Price touching lower band (buy signal)
                if current_price <= lower_band * 1.001:  # Small tolerance
                    buy_signals += 2
                # Price touching upper band (sell signal)
                elif current_price >= upper_band * 0.999:  # Small tolerance
                    sell_signals += 2
            
            # EMA crossover (fast scalping signal)
            if indicators.get("price_ema_12") and indicators.get("price_ema_26"):
                ema_12 = indicators["price_ema_12"][-1]
                ema_26 = indicators["price_ema_26"][-1]
                
                if len(indicators["price_ema_12"]) >= 2 and len(indicators["price_ema_26"]) >= 2:
                    prev_ema_12 = indicators["price_ema_12"][-2]
                    prev_ema_26 = indicators["price_ema_26"][-2]
                    
                    # Bullish EMA crossover
                    if ema_12 > ema_26 and prev_ema_12 <= prev_ema_26:
                        buy_signals += 1
                    # Bearish EMA crossover
                    elif ema_12 < ema_26 and prev_ema_12 >= prev_ema_26:
                        sell_signals += 1
            
            # Volume confirmation
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
            
            # Determine signal (need strong consensus for scalping)
            if buy_signals > sell_signals and buy_signals >= 3:
                return SignalType.BUY
            elif sell_signals > buy_signals and sell_signals >= 3:
                return SignalType.SELL
            else:
                return SignalType.HOLD
                
        except Exception as e:
            self.logger.error(f"Error identifying scalping signal: {e}")
            return SignalType.HOLD
    
    def _calculate_confidence(self, indicators: Dict[str, Any], signal_type: SignalType) -> float:
        """Calculate confidence score for scalping signal"""
        try:
            confidence_factors = []
            
            # RSI confidence
            if indicators.get("rsi"):
                rsi = indicators["rsi"][-1]
                
                if signal_type == SignalType.BUY:
                    # Higher confidence when RSI is very oversold
                    rsi_confidence = max(0, (30 - rsi) / 30)
                elif signal_type == SignalType.SELL:
                    # Higher confidence when RSI is very overbought
                    rsi_confidence = max(0, (rsi - 70) / 30)
                else:
                    rsi_confidence = 0.5
                
                confidence_factors.append(min(rsi_confidence, 1.0))
            
            # Stochastic confidence
            if indicators.get("stochastic"):
                stoch_k = indicators["stochastic"]["k"][-1]
                stoch_d = indicators["stochastic"]["d"][-1]
                
                if signal_type == SignalType.BUY:
                    stoch_confidence = max(0, (20 - min(stoch_k, stoch_d)) / 20)
                elif signal_type == SignalType.SELL:
                    stoch_confidence = max(0, (max(stoch_k, stoch_d) - 80) / 20)
                else:
                    stoch_confidence = 0.5
                
                confidence_factors.append(min(stoch_confidence, 1.0))
            
            # Williams %R confidence
            if indicators.get("williams_r"):
                williams_r = indicators["williams_r"][-1]
                
                if signal_type == SignalType.BUY:
                    williams_confidence = max(0, (-80 - williams_r) / 20)
                elif signal_type == SignalType.SELL:
                    williams_confidence = max(0, (williams_r + 20) / 20)
                else:
                    williams_confidence = 0.5
                
                confidence_factors.append(min(williams_confidence, 1.0))
            
            # Volume confidence
            if indicators.get("volume_sma"):
                current_volume = indicators.get("current_volume", 0)
                avg_volume = indicators["volume_sma"][-1]
                
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
                volume_confidence = min(volume_ratio / 3, 1.0)  # Normalize to 0-1
                
                confidence_factors.append(volume_confidence)
            
            # Volatility confidence (higher volatility = higher confidence for scalping)
            if indicators.get("bollinger_bands"):
                bb = indicators["bollinger_bands"]
                upper_band = bb["upper_band"][-1]
                lower_band = bb["lower_band"][-1]
                middle_band = bb["middle_band"][-1]
                
                volatility = (upper_band - lower_band) / middle_band
                volatility_confidence = min(volatility * 50, 1.0)  # Normalize
                
                confidence_factors.append(volatility_confidence)
            
            # Calculate overall confidence
            if confidence_factors:
                return sum(confidence_factors) / len(confidence_factors)
            
            return 0.5  # Default confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _calculate_exit_levels(self, current_price: float, signal_type: SignalType) -> tuple:
        """Calculate tight stop loss and take profit levels for scalping"""
        try:
            if signal_type == SignalType.BUY:
                stop_loss = current_price * (1 - self.config.parameters["stop_loss_pct"])
                take_profit = current_price * (1 + self.config.parameters["take_profit_pct"])
                
            elif signal_type == SignalType.SELL:
                stop_loss = current_price * (1 + self.config.parameters["stop_loss_pct"])
                take_profit = current_price * (1 - self.config.parameters["take_profit_pct"])
            
            else:
                return None, None
            
            return stop_loss, take_profit
            
        except Exception as e:
            self.logger.error(f"Error calculating exit levels: {e}")
            return None, None
    
    def _generate_reasoning(self, indicators: Dict[str, Any], signal_type: SignalType, confidence: float) -> str:
        """Generate reasoning for scalping signal"""
        try:
            reasoning_parts = []
            
            # RSI analysis
            if indicators.get("rsi"):
                rsi = indicators["rsi"][-1]
                
                if rsi < 25:
                    reasoning_parts.append(f"RSI extremely oversold ({rsi:.1f})")
                elif rsi > 75:
                    reasoning_parts.append(f"RSI extremely overbought ({rsi:.1f})")
                else:
                    reasoning_parts.append(f"RSI at {rsi:.1f}")
            
            # Stochastic analysis
            if indicators.get("stochastic"):
                stoch_k = indicators["stochastic"]["k"][-1]
                stoch_d = indicators["stochastic"]["d"][-1]
                
                if stoch_k < 20 and stoch_d < 20:
                    reasoning_parts.append(f"Stochastic oversold (K:{stoch_k:.1f}, D:{stoch_d:.1f})")
                elif stoch_k > 80 and stoch_d > 80:
                    reasoning_parts.append(f"Stochastic overbought (K:{stoch_k:.1f}, D:{stoch_d:.1f})")
                else:
                    reasoning_parts.append(f"Stochastic neutral (K:{stoch_k:.1f}, D:{stoch_d:.1f})")
            
            # Williams %R analysis
            if indicators.get("williams_r"):
                williams_r = indicators["williams_r"][-1]
                
                if williams_r < -80:
                    reasoning_parts.append(f"Williams %R oversold ({williams_r:.1f})")
                elif williams_r > -20:
                    reasoning_parts.append(f"Williams %R overbought ({williams_r:.1f})")
            
            # Volume analysis
            if indicators.get("volume_sma"):
                current_volume = indicators.get("current_volume", 0)
                avg_volume = indicators["volume_sma"][-1]
                
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
                
                if volume_ratio > 2.0:
                    reasoning_parts.append(f"High volume ({volume_ratio:.1f}x average)")
                elif volume_ratio > 1.5:
                    reasoning_parts.append(f"Above average volume ({volume_ratio:.1f}x)")
            
            # Bollinger Bands analysis
            if indicators.get("bollinger_bands"):
                reasoning_parts.append("Price at Bollinger Band extreme")
            
            # Final reasoning
            signal_action = "BUY" if signal_type == SignalType.BUY else "SELL"
            reasoning = f"SCALP {signal_action} - {'; '.join(reasoning_parts)}. Confidence: {confidence:.2f}"
            
            return reasoning
            
        except Exception as e:
            self.logger.error(f"Error generating reasoning: {e}")
            return f"SCALP {signal_type.value.upper()} signal detected"
    
    async def validate_signal(self, signal: TradingSignal) -> bool:
        """Validate scalping signal with stricter requirements"""
        try:
            # Check confidence threshold (higher for scalping)
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
            
            # Check if price hasn't moved too much (stricter for scalping)
            current_market_data = await self.exchange.get_market_data(signal.symbol)
            price_change = abs(current_market_data.price - signal.price) / signal.price
            
            if price_change > 0.005:  # 0.5% price change threshold
                return False
            
            # Check spread (if available)
            if hasattr(current_market_data, 'bid') and hasattr(current_market_data, 'ask'):
                if current_market_data.bid > 0 and current_market_data.ask > 0:
                    spread = (current_market_data.ask - current_market_data.bid) / current_market_data.price
                    if spread > self.config.parameters["spread_threshold"]:
                        return False  # Spread too wide for scalping
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating scalping signal: {e}")
            return False
    
    def get_strategy_parameters(self) -> Dict[str, Any]:
        """Get scalping strategy parameters"""
        return {
            "strategy_type": "scalping",
            "description": "Very short-term trading strategy",
            "typical_holding_period": "1-5 minutes",
            "risk_per_trade": f"{self.config.parameters['position_size_pct']*100:.1f}%",
            "stop_loss": f"{self.config.parameters['stop_loss_pct']*100:.2f}%",
            "take_profit": f"{self.config.parameters['take_profit_pct']*100:.2f}%",
            "parameters": self.config.parameters
        }
