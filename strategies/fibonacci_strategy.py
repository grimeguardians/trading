"""
Fibonacci-based trading strategy with advanced retracement and extension analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyResult
from exchanges.base_exchange import BaseExchange
from analytics.fibonacci_calculator import FibonacciCalculator

logger = logging.getLogger(__name__)

class FibonacciStrategy(BaseStrategy):
    """Advanced Fibonacci retracement and extension trading strategy"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.fibonacci_calculator = FibonacciCalculator()
        
        # Fibonacci-specific parameters
        self.swing_lookback = self.parameters.get("swing_lookback", 50)
        self.min_swing_size = self.parameters.get("min_swing_size", 0.02)  # 2% minimum swing
        self.confluence_threshold = self.parameters.get("confluence_threshold", 2)  # Min confluences
        self.volume_confirmation = self.parameters.get("volume_confirmation", True)
        self.trend_confirmation = self.parameters.get("trend_confirmation", True)
        
        # Fibonacci levels for entry/exit
        self.entry_levels = self.parameters.get("entry_levels", [0.382, 0.5, 0.618])
        self.profit_levels = self.parameters.get("profit_levels", [1.272, 1.618, 2.618])
        
        # Risk management
        self.max_risk_per_trade = self.parameters.get("max_risk_per_trade", 0.02)
        self.confluence_multiplier = self.parameters.get("confluence_multiplier", 1.5)
        
    async def analyze(self, symbol: str, exchange: BaseExchange, timeframe: str = "1h") -> StrategyResult:
        """Analyze market data using Fibonacci retracements and extensions"""
        start_time = datetime.now()
        
        try:
            # Get market data
            df = await self.get_market_data(symbol, timeframe, exchange, limit=200)
            
            if df.empty or len(df) < self.swing_lookback:
                return StrategyResult(
                    signals=[],
                    performance_metrics={},
                    risk_metrics={},
                    execution_time=0.0,
                    debug_info={"error": "Insufficient data"}
                )
                
            # Calculate technical indicators
            indicators = await self._calculate_indicators(df)
            
            # Identify swing points
            swing_points = self._identify_swing_points(df)
            
            # Calculate Fibonacci levels
            fibonacci_analysis = await self._calculate_fibonacci_levels(df, swing_points)
            
            # Identify trading opportunities
            signals = await self._generate_fibonacci_signals(
                df, indicators, fibonacci_analysis, symbol
            )
            
            # Calculate performance metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            performance_metrics = self.update_performance_metrics(signals, execution_time)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(signals, fibonacci_analysis)
            
            return StrategyResult(
                signals=signals,
                performance_metrics=performance_metrics,
                risk_metrics=risk_metrics,
                execution_time=execution_time,
                debug_info={
                    "swing_points": len(swing_points),
                    "fibonacci_levels": fibonacci_analysis,
                    "indicators": {k: v.iloc[-1] if hasattr(v, 'iloc') else v for k, v in indicators.items()}
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in Fibonacci analysis for {symbol}: {e}")
            return StrategyResult(
                signals=[],
                performance_metrics={},
                risk_metrics={},
                execution_time=0.0,
                debug_info={"error": str(e)}
            )
            
    async def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators for confirmation"""
        indicators = {}
        
        # RSI for momentum confirmation
        indicators['rsi'] = self.calculate_rsi(df, period=14)
        
        # MACD for trend confirmation
        macd_data = self.calculate_macd(df)
        indicators.update(macd_data)
        
        # Bollinger Bands for volatility context
        bb_data = self.calculate_bollinger_bands(df)
        indicators.update(bb_data)
        
        # Stochastic for overbought/oversold
        stoch_data = self.calculate_stochastic(df)
        indicators.update(stoch_data)
        
        # Volume indicators
        indicators['volume_sma'] = df['volume'].rolling(window=20).mean()
        indicators['volume_ratio'] = df['volume'] / indicators['volume_sma']
        
        # Support/Resistance levels
        sr_levels = self.calculate_support_resistance(df)
        indicators.update(sr_levels)
        
        return indicators
        
    def _identify_swing_points(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify significant swing highs and lows"""
        swing_points = []
        
        try:
            # Calculate swing highs and lows using pivot points
            highs = df['high'].rolling(window=5, center=True).max()
            lows = df['low'].rolling(window=5, center=True).min()
            
            # Find swing highs
            swing_highs = df[
                (df['high'] == highs) & 
                (df['high'].shift(1) < df['high']) & 
                (df['high'].shift(-1) < df['high'])
            ]
            
            # Find swing lows
            swing_lows = df[
                (df['low'] == lows) & 
                (df['low'].shift(1) > df['low']) & 
                (df['low'].shift(-1) > df['low'])
            ]
            
            # Filter significant swings
            for idx, row in swing_highs.iterrows():
                if len(swing_points) == 0:
                    swing_points.append({
                        'type': 'high',
                        'timestamp': idx,
                        'price': row['high'],
                        'volume': row['volume']
                    })
                else:
                    last_point = swing_points[-1]
                    price_change = abs(row['high'] - last_point['price']) / last_point['price']
                    
                    if price_change >= self.min_swing_size:
                        swing_points.append({
                            'type': 'high',
                            'timestamp': idx,
                            'price': row['high'],
                            'volume': row['volume']
                        })
                        
            for idx, row in swing_lows.iterrows():
                if len(swing_points) == 0:
                    swing_points.append({
                        'type': 'low',
                        'timestamp': idx,
                        'price': row['low'],
                        'volume': row['volume']
                    })
                else:
                    last_point = swing_points[-1]
                    price_change = abs(row['low'] - last_point['price']) / last_point['price']
                    
                    if price_change >= self.min_swing_size:
                        swing_points.append({
                            'type': 'low',
                            'timestamp': idx,
                            'price': row['low'],
                            'volume': row['volume']
                        })
                        
            # Sort by timestamp
            swing_points.sort(key=lambda x: x['timestamp'])
            
            return swing_points[-20:]  # Keep last 20 swing points
            
        except Exception as e:
            self.logger.error(f"Error identifying swing points: {e}")
            return []
            
    async def _calculate_fibonacci_levels(self, df: pd.DataFrame, swing_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate Fibonacci retracement and extension levels"""
        if len(swing_points) < 2:
            return {}
            
        try:
            # Get the most recent significant swing
            recent_swings = swing_points[-4:]  # Last 4 swing points
            
            if len(recent_swings) < 2:
                return {}
                
            # Determine trend direction
            latest_high = max([s for s in recent_swings if s['type'] == 'high'], 
                            key=lambda x: x['price'], default=None)
            latest_low = min([s for s in recent_swings if s['type'] == 'low'], 
                           key=lambda x: x['price'], default=None)
            
            if not latest_high or not latest_low:
                return {}
                
            # Calculate Fibonacci levels
            fibonacci_analysis = self.fibonacci_calculator.calculate_fibonacci_levels(
                high_price=latest_high['price'],
                low_price=latest_low['price'],
                current_price=df['close'].iloc[-1]
            )
            
            # Add confluence analysis
            confluence_levels = self._find_confluence_levels(df, fibonacci_analysis)
            
            fibonacci_analysis.update({
                'swing_high': latest_high,
                'swing_low': latest_low,
                'confluence_levels': confluence_levels,
                'trend_direction': self._determine_trend_direction(recent_swings)
            })
            
            return fibonacci_analysis
            
        except Exception as e:
            self.logger.error(f"Error calculating Fibonacci levels: {e}")
            return {}
            
    def _find_confluence_levels(self, df: pd.DataFrame, fibonacci_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find confluence levels where multiple indicators align"""
        confluence_levels = []
        
        try:
            current_price = df['close'].iloc[-1]
            
            # Get Fibonacci levels
            fib_levels = fibonacci_analysis.get('retracement_levels', {})
            ext_levels = fibonacci_analysis.get('extension_levels', {})
            
            # Check for confluences with support/resistance
            sr_levels = self.calculate_support_resistance(df)
            
            all_fib_levels = {**fib_levels, **ext_levels}
            
            for level_name, level_price in all_fib_levels.items():
                confluences = []
                
                # Check support/resistance confluence
                for sr_name, sr_price in sr_levels.items():
                    if abs(level_price - sr_price) / sr_price < 0.01:  # Within 1%
                        confluences.append(f"SR_{sr_name}")
                        
                # Check moving average confluence
                sma_20 = df['close'].rolling(window=20).mean().iloc[-1]
                sma_50 = df['close'].rolling(window=50).mean().iloc[-1]
                
                if abs(level_price - sma_20) / sma_20 < 0.01:
                    confluences.append("SMA_20")
                if abs(level_price - sma_50) / sma_50 < 0.01:
                    confluences.append("SMA_50")
                    
                if len(confluences) >= self.confluence_threshold:
                    confluence_levels.append({
                        'level': level_name,
                        'price': level_price,
                        'confluences': confluences,
                        'strength': len(confluences),
                        'distance_from_current': abs(level_price - current_price) / current_price
                    })
                    
            # Sort by strength and proximity
            confluence_levels.sort(key=lambda x: (x['strength'], -x['distance_from_current']), reverse=True)
            
            return confluence_levels[:5]  # Top 5 confluence levels
            
        except Exception as e:
            self.logger.error(f"Error finding confluence levels: {e}")
            return []
            
    def _determine_trend_direction(self, swing_points: List[Dict[str, Any]]) -> str:
        """Determine overall trend direction from swing points"""
        if len(swing_points) < 4:
            return "neutral"
            
        try:
            highs = [s['price'] for s in swing_points if s['type'] == 'high']
            lows = [s['price'] for s in swing_points if s['type'] == 'low']
            
            if len(highs) >= 2 and len(lows) >= 2:
                # Check for higher highs and higher lows (uptrend)
                if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
                    return "uptrend"
                # Check for lower highs and lower lows (downtrend)
                elif highs[-1] < highs[-2] and lows[-1] < lows[-2]:
                    return "downtrend"
                    
            return "neutral"
            
        except Exception as e:
            self.logger.error(f"Error determining trend direction: {e}")
            return "neutral"
            
    async def _generate_fibonacci_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], 
                                        fibonacci_analysis: Dict[str, Any], symbol: str) -> List[Signal]:
        """Generate trading signals based on Fibonacci analysis"""
        signals = []
        
        try:
            current_price = df['close'].iloc[-1]
            current_volume = df['volume'].iloc[-1]
            
            confluence_levels = fibonacci_analysis.get('confluence_levels', [])
            trend_direction = fibonacci_analysis.get('trend_direction', 'neutral')
            
            # Get current indicator values
            current_rsi = indicators['rsi'].iloc[-1] if 'rsi' in indicators else 50
            current_macd = indicators['macd'].iloc[-1] if 'macd' in indicators else 0
            current_macd_signal = indicators['macd_signal'].iloc[-1] if 'macd_signal' in indicators else 0
            volume_ratio = indicators['volume_ratio'].iloc[-1] if 'volume_ratio' in indicators else 1
            
            # Check for buy signals
            for confluence in confluence_levels:
                level_price = confluence['price']
                distance_pct = abs(current_price - level_price) / level_price
                
                # Buy signal conditions
                if (distance_pct < 0.005 and  # Within 0.5% of level
                    trend_direction in ['uptrend', 'neutral'] and
                    current_rsi < 70 and  # Not overbought
                    current_macd > current_macd_signal and  # MACD bullish
                    (not self.volume_confirmation or volume_ratio > 1.2)):  # Volume confirmation
                    
                    # Calculate confidence based on confluence strength
                    confidence = min(0.9, 0.5 + (confluence['strength'] * 0.1))
                    
                    # Adjust for trend alignment
                    if trend_direction == 'uptrend':
                        confidence *= 1.2
                        
                    # Create buy signal
                    signal = self.create_signal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        price=current_price,
                        confidence=min(confidence, 1.0),
                        fibonacci_levels=fibonacci_analysis.get('retracement_levels', {}),
                        metadata={
                            'confluence_level': confluence['level'],
                            'confluence_strength': confluence['strength'],
                            'trend_direction': trend_direction,
                            'rsi': current_rsi,
                            'macd_signal': current_macd > current_macd_signal,
                            'volume_ratio': volume_ratio,
                            'fibonacci_price': level_price,
                            'strategy': 'fibonacci_retracement'
                        }
                    )
                    
                    if signal:
                        signals.append(signal)
                        
                # Sell signal conditions
                elif (distance_pct < 0.005 and  # Within 0.5% of level
                      trend_direction in ['downtrend', 'neutral'] and
                      current_rsi > 30 and  # Not oversold
                      current_macd < current_macd_signal and  # MACD bearish
                      (not self.volume_confirmation or volume_ratio > 1.2)):  # Volume confirmation
                    
                    # Calculate confidence based on confluence strength
                    confidence = min(0.9, 0.5 + (confluence['strength'] * 0.1))
                    
                    # Adjust for trend alignment
                    if trend_direction == 'downtrend':
                        confidence *= 1.2
                        
                    # Create sell signal
                    signal = self.create_signal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        price=current_price,
                        confidence=min(confidence, 1.0),
                        fibonacci_levels=fibonacci_analysis.get('extension_levels', {}),
                        metadata={
                            'confluence_level': confluence['level'],
                            'confluence_strength': confluence['strength'],
                            'trend_direction': trend_direction,
                            'rsi': current_rsi,
                            'macd_signal': current_macd < current_macd_signal,
                            'volume_ratio': volume_ratio,
                            'fibonacci_price': level_price,
                            'strategy': 'fibonacci_extension'
                        }
                    )
                    
                    if signal:
                        signals.append(signal)
                        
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating Fibonacci signals: {e}")
            return []
            
    def _calculate_risk_metrics(self, signals: List[Signal], fibonacci_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk metrics for Fibonacci strategy"""
        try:
            if not signals:
                return {}
                
            # Calculate average risk/reward ratio
            avg_risk_reward = np.mean([s.risk_reward_ratio for s in signals if s.risk_reward_ratio > 0])
            
            # Calculate signal distribution
            buy_signals = len([s for s in signals if s.signal_type == SignalType.BUY])
            sell_signals = len([s for s in signals if s.signal_type == SignalType.SELL])
            
            # Calculate confluence strength distribution
            confluence_strengths = []
            for signal in signals:
                if 'confluence_strength' in signal.metadata:
                    confluence_strengths.append(signal.metadata['confluence_strength'])
                    
            avg_confluence_strength = np.mean(confluence_strengths) if confluence_strengths else 0
            
            return {
                'average_risk_reward': avg_risk_reward,
                'signal_distribution': {
                    'buy_signals': buy_signals,
                    'sell_signals': sell_signals,
                    'total_signals': len(signals)
                },
                'confluence_analysis': {
                    'average_strength': avg_confluence_strength,
                    'max_strength': max(confluence_strengths) if confluence_strengths else 0,
                    'levels_count': len(fibonacci_analysis.get('confluence_levels', []))
                },
                'trend_alignment': fibonacci_analysis.get('trend_direction', 'neutral'),
                'fibonacci_levels_active': len(fibonacci_analysis.get('retracement_levels', {}))
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return {}
            
    def get_required_indicators(self) -> List[str]:
        """Get list of required technical indicators"""
        return [
            'rsi',
            'macd',
            'macd_signal',
            'macd_histogram',
            'bollinger_bands',
            'stochastic',
            'volume_sma',
            'support_resistance',
            'fibonacci_levels'
        ]
        
    def validate_parameters(self) -> bool:
        """Validate strategy parameters"""
        try:
            # Check required parameters
            required_params = ['swing_lookback', 'min_swing_size', 'confluence_threshold']
            
            for param in required_params:
                if param not in self.parameters:
                    self.logger.error(f"Missing required parameter: {param}")
                    return False
                    
            # Validate parameter ranges
            if self.swing_lookback < 20 or self.swing_lookback > 100:
                self.logger.error("swing_lookback must be between 20 and 100")
                return False
                
            if self.min_swing_size < 0.01 or self.min_swing_size > 0.1:
                self.logger.error("min_swing_size must be between 0.01 and 0.1")
                return False
                
            if self.confluence_threshold < 1 or self.confluence_threshold > 5:
                self.logger.error("confluence_threshold must be between 1 and 5")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating parameters: {e}")
            return False
