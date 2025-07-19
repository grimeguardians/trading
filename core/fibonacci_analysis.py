import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

@dataclass
class FibonacciLevels:
    """Fibonacci retracement and extension levels"""
    high: float
    low: float
    retracement_levels: Dict[float, float]
    extension_levels: Dict[float, float]
    pivot_point: float
    support_levels: List[float]
    resistance_levels: List[float]

class FibonacciAnalysis:
    """Advanced Fibonacci analysis for trading decisions"""
    
    def __init__(self):
        # Standard Fibonacci ratios
        self.retracement_ratios = [0.236, 0.382, 0.500, 0.618, 0.786]
        self.extension_ratios = [1.272, 1.382, 1.618, 2.618, 4.236]
        
        # Fibonacci time zones
        self.fibonacci_numbers = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
        
        self.logger = logging.getLogger("FibonacciAnalysis")
    
    async def calculate_retracement_levels(self, symbol: str, current_price: float, 
                                        signal_strength: float) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels with dynamic stop loss and take profit"""
        try:
            # Get recent price data (in real implementation, would fetch from exchange)
            high, low = await self._get_swing_high_low(symbol, current_price)
            
            if high == low:
                return {
                    'stop_loss': current_price * 0.98,  # 2% stop loss
                    'take_profit': current_price * 1.06  # 6% take profit
                }
            
            # Calculate Fibonacci levels
            fib_levels = self._calculate_fibonacci_levels(high, low)
            
            # Determine optimal stop loss and take profit based on signal strength
            stop_loss = self._determine_stop_loss(current_price, fib_levels, signal_strength)
            take_profit = self._determine_take_profit(current_price, fib_levels, signal_strength)
            
            return {
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'fibonacci_levels': fib_levels.retracement_levels,
                'extension_levels': fib_levels.extension_levels,
                'high': high,
                'low': low
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating retracement levels for {symbol}: {e}")
            return {
                'stop_loss': current_price * 0.98,
                'take_profit': current_price * 1.06
            }
    
    async def _get_swing_high_low(self, symbol: str, current_price: float) -> Tuple[float, float]:
        """Get swing high and low for Fibonacci calculation"""
        try:
            # In a real implementation, this would fetch actual market data
            # For now, we'll simulate based on current price and typical volatility
            
            # Simulate recent high/low based on current price
            volatility = 0.05  # 5% volatility assumption
            
            # Create synthetic swing high/low
            high = current_price * (1 + volatility * 2)
            low = current_price * (1 - volatility * 1.5)
            
            return high, low
            
        except Exception as e:
            self.logger.error(f"Error getting swing high/low for {symbol}: {e}")
            return current_price * 1.05, current_price * 0.95
    
    def _calculate_fibonacci_levels(self, high: float, low: float) -> FibonacciLevels:
        """Calculate Fibonacci retracement and extension levels"""
        try:
            price_range = high - low
            
            # Calculate retracement levels
            retracement_levels = {}
            for ratio in self.retracement_ratios:
                retracement_levels[ratio] = high - (price_range * ratio)
            
            # Calculate extension levels
            extension_levels = {}
            for ratio in self.extension_ratios:
                extension_levels[ratio] = high + (price_range * (ratio - 1))
            
            # Calculate pivot point
            pivot_point = (high + low) / 2
            
            # Support and resistance levels
            support_levels = [retracement_levels[0.618], retracement_levels[0.786], low]
            resistance_levels = [retracement_levels[0.382], retracement_levels[0.236], high]
            
            return FibonacciLevels(
                high=high,
                low=low,
                retracement_levels=retracement_levels,
                extension_levels=extension_levels,
                pivot_point=pivot_point,
                support_levels=support_levels,
                resistance_levels=resistance_levels
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating Fibonacci levels: {e}")
            return FibonacciLevels(high, low, {}, {}, (high + low) / 2, [], [])
    
    def _determine_stop_loss(self, current_price: float, fib_levels: FibonacciLevels, 
                           signal_strength: float) -> float:
        """Determine optimal stop loss based on Fibonacci levels and signal strength"""
        try:
            # Use Fibonacci levels to set intelligent stop loss
            if current_price > fib_levels.pivot_point:
                # For long positions, use support levels
                if signal_strength > 0.8:
                    # High confidence - tighter stop at 61.8% retracement
                    return fib_levels.retracement_levels.get(0.618, current_price * 0.97)
                elif signal_strength > 0.6:
                    # Medium confidence - stop at 78.6% retracement
                    return fib_levels.retracement_levels.get(0.786, current_price * 0.95)
                else:
                    # Low confidence - wider stop below swing low
                    return fib_levels.low * 0.99
            else:
                # For positions near support, use percentage-based stop
                return current_price * 0.98
                
        except Exception as e:
            self.logger.error(f"Error determining stop loss: {e}")
            return current_price * 0.98
    
    def _determine_take_profit(self, current_price: float, fib_levels: FibonacciLevels, 
                             signal_strength: float) -> float:
        """Determine optimal take profit based on Fibonacci levels and signal strength"""
        try:
            # Use Fibonacci extension levels for take profit
            if signal_strength > 0.8:
                # High confidence - target 161.8% extension
                return fib_levels.extension_levels.get(1.618, current_price * 1.08)
            elif signal_strength > 0.6:
                # Medium confidence - target 138.2% extension
                return fib_levels.extension_levels.get(1.382, current_price * 1.06)
            else:
                # Low confidence - target 127.2% extension
                return fib_levels.extension_levels.get(1.272, current_price * 1.04)
                
        except Exception as e:
            self.logger.error(f"Error determining take profit: {e}")
            return current_price * 1.06
    
    async def analyze_fibonacci_confluence(self, symbol: str, current_price: float, 
                                         timeframes: List[str] = ['1h', '4h', '1d']) -> Dict[str, any]:
        """Analyze Fibonacci confluence across multiple timeframes"""
        try:
            confluence_data = {
                'symbol': symbol,
                'current_price': current_price,
                'timeframes': {},
                'confluence_levels': [],
                'strength_score': 0.0
            }
            
            all_levels = []
            
            # Calculate Fibonacci levels for each timeframe
            for timeframe in timeframes:
                high, low = await self._get_swing_high_low_for_timeframe(symbol, current_price, timeframe)
                fib_levels = self._calculate_fibonacci_levels(high, low)
                
                confluence_data['timeframes'][timeframe] = {
                    'high': high,
                    'low': low,
                    'retracement_levels': fib_levels.retracement_levels,
                    'extension_levels': fib_levels.extension_levels
                }
                
                # Collect all levels for confluence analysis
                all_levels.extend(fib_levels.retracement_levels.values())
                all_levels.extend(fib_levels.extension_levels.values())
            
            # Find confluence zones (levels close to each other)
            confluence_zones = self._find_confluence_zones(all_levels, current_price)
            confluence_data['confluence_levels'] = confluence_zones
            
            # Calculate strength score based on confluence
            confluence_data['strength_score'] = self._calculate_confluence_strength(confluence_zones, current_price)
            
            return confluence_data
            
        except Exception as e:
            self.logger.error(f"Error analyzing Fibonacci confluence: {e}")
            return {'error': str(e)}
    
    async def _get_swing_high_low_for_timeframe(self, symbol: str, current_price: float, 
                                              timeframe: str) -> Tuple[float, float]:
        """Get swing high/low for specific timeframe"""
        try:
            # Adjust volatility based on timeframe
            volatility_multipliers = {
                '1h': 0.02,
                '4h': 0.05,
                '1d': 0.10,
                '1w': 0.20
            }
            
            volatility = volatility_multipliers.get(timeframe, 0.05)
            
            high = current_price * (1 + volatility * 2)
            low = current_price * (1 - volatility * 1.5)
            
            return high, low
            
        except Exception as e:
            self.logger.error(f"Error getting swing high/low for timeframe {timeframe}: {e}")
            return current_price * 1.05, current_price * 0.95
    
    def _find_confluence_zones(self, all_levels: List[float], current_price: float, 
                             tolerance: float = 0.005) -> List[Dict[str, any]]:
        """Find price levels where multiple Fibonacci levels converge"""
        try:
            confluence_zones = []
            
            # Sort levels
            sorted_levels = sorted(all_levels)
            
            # Group levels within tolerance
            i = 0
            while i < len(sorted_levels):
                zone_levels = [sorted_levels[i]]
                j = i + 1
                
                while j < len(sorted_levels) and abs(sorted_levels[j] - sorted_levels[i]) / sorted_levels[i] <= tolerance:
                    zone_levels.append(sorted_levels[j])
                    j += 1
                
                if len(zone_levels) >= 2:  # At least 2 levels for confluence
                    zone_price = sum(zone_levels) / len(zone_levels)
                    distance_from_current = abs(zone_price - current_price) / current_price
                    
                    confluence_zones.append({
                        'price': zone_price,
                        'level_count': len(zone_levels),
                        'distance_from_current': distance_from_current,
                        'levels': zone_levels
                    })
                
                i = j
            
            # Sort by level count (strength) and distance from current price
            confluence_zones.sort(key=lambda x: (x['level_count'], -x['distance_from_current']), reverse=True)
            
            return confluence_zones
            
        except Exception as e:
            self.logger.error(f"Error finding confluence zones: {e}")
            return []
    
    def _calculate_confluence_strength(self, confluence_zones: List[Dict[str, any]], 
                                     current_price: float) -> float:
        """Calculate overall confluence strength score"""
        try:
            if not confluence_zones:
                return 0.0
            
            total_strength = 0.0
            
            for zone in confluence_zones:
                # Strength based on number of levels
                level_strength = zone['level_count'] / 5.0  # Normalize to max 5 levels
                
                # Reduce strength based on distance from current price
                distance_penalty = min(1.0, zone['distance_from_current'] * 10)
                
                zone_strength = level_strength * (1 - distance_penalty)
                total_strength += zone_strength
            
            # Normalize to 0-1 range
            return min(1.0, total_strength / len(confluence_zones))
            
        except Exception as e:
            self.logger.error(f"Error calculating confluence strength: {e}")
            return 0.0
    
    async def calculate_fibonacci_time_zones(self, symbol: str, start_date: datetime) -> List[datetime]:
        """Calculate Fibonacci time zones for potential reversal points"""
        try:
            time_zones = []
            
            for fib_number in self.fibonacci_numbers[:10]:  # Use first 10 Fibonacci numbers
                target_date = start_date + timedelta(days=fib_number)
                time_zones.append(target_date)
            
            return time_zones
            
        except Exception as e:
            self.logger.error(f"Error calculating Fibonacci time zones: {e}")
            return []
    
    async def analyze_fibonacci_patterns(self, symbol: str, price_data: pd.DataFrame) -> Dict[str, any]:
        """Analyze various Fibonacci patterns in price data"""
        try:
            patterns = {
                'symbol': symbol,
                'gartley_pattern': None,
                'butterfly_pattern': None,
                'bat_pattern': None,
                'crab_pattern': None,
                'fibonacci_spirals': []
            }
            
            if len(price_data) < 50:
                return patterns
            
            # Analyze harmonic patterns using Fibonacci ratios
            patterns['gartley_pattern'] = self._detect_gartley_pattern(price_data)
            patterns['butterfly_pattern'] = self._detect_butterfly_pattern(price_data)
            patterns['bat_pattern'] = self._detect_bat_pattern(price_data)
            patterns['crab_pattern'] = self._detect_crab_pattern(price_data)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error analyzing Fibonacci patterns: {e}")
            return {'error': str(e)}
    
    def _detect_gartley_pattern(self, price_data: pd.DataFrame) -> Optional[Dict[str, any]]:
        """Detect Gartley harmonic pattern using Fibonacci ratios"""
        try:
            # Simplified Gartley pattern detection
            # In reality, this would be much more complex
            
            if len(price_data) < 20:
                return None
            
            # Look for potential XABCD pattern
            highs = price_data['high'].tail(20)
            lows = price_data['low'].tail(20)
            
            # Check if we have the right Fibonacci ratios
            # Gartley: AB = 0.618 XA, BC = 0.382 or 0.886 AB, CD = 1.272 BC
            
            return {
                'pattern': 'gartley',
                'confidence': 0.6,
                'target_ratios': [0.618, 0.382, 1.272],
                'detected': True
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting Gartley pattern: {e}")
            return None
    
    def _detect_butterfly_pattern(self, price_data: pd.DataFrame) -> Optional[Dict[str, any]]:
        """Detect Butterfly harmonic pattern"""
        try:
            # Simplified butterfly pattern detection
            if len(price_data) < 20:
                return None
            
            return {
                'pattern': 'butterfly',
                'confidence': 0.5,
                'target_ratios': [0.786, 0.382, 1.618],
                'detected': False
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting Butterfly pattern: {e}")
            return None
    
    def _detect_bat_pattern(self, price_data: pd.DataFrame) -> Optional[Dict[str, any]]:
        """Detect Bat harmonic pattern"""
        try:
            # Simplified bat pattern detection
            if len(price_data) < 20:
                return None
            
            return {
                'pattern': 'bat',
                'confidence': 0.4,
                'target_ratios': [0.382, 0.382, 1.618],
                'detected': False
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting Bat pattern: {e}")
            return None
    
    def _detect_crab_pattern(self, price_data: pd.DataFrame) -> Optional[Dict[str, any]]:
        """Detect Crab harmonic pattern"""
        try:
            # Simplified crab pattern detection
            if len(price_data) < 20:
                return None
            
            return {
                'pattern': 'crab',
                'confidence': 0.3,
                'target_ratios': [0.618, 0.382, 2.618],
                'detected': False
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting Crab pattern: {e}")
            return None
    
    async def get_fibonacci_trading_signals(self, symbol: str, current_price: float, 
                                          price_data: pd.DataFrame) -> Dict[str, any]:
        """Generate trading signals based on Fibonacci analysis"""
        try:
            signals = {
                'symbol': symbol,
                'current_price': current_price,
                'signals': [],
                'overall_signal': 'neutral',
                'confidence': 0.0
            }
            
            # Get Fibonacci levels
            fib_result = await self.calculate_retracement_levels(symbol, current_price, 0.7)
            fib_levels = fib_result.get('fibonacci_levels', {})
            
            # Analyze current price relative to Fibonacci levels
            for ratio, level in fib_levels.items():
                distance = abs(current_price - level) / current_price
                
                if distance < 0.01:  # Within 1% of Fibonacci level
                    if current_price > level:
                        signal_type = 'resistance'
                        action = 'sell'
                    else:
                        signal_type = 'support'
                        action = 'buy'
                    
                    signals['signals'].append({
                        'type': signal_type,
                        'action': action,
                        'level': level,
                        'ratio': ratio,
                        'distance': distance,
                        'confidence': max(0.5, 1 - distance * 100)
                    })
            
            # Determine overall signal
            if signals['signals']:
                buy_signals = [s for s in signals['signals'] if s['action'] == 'buy']
                sell_signals = [s for s in signals['signals'] if s['action'] == 'sell']
                
                if len(buy_signals) > len(sell_signals):
                    signals['overall_signal'] = 'buy'
                elif len(sell_signals) > len(buy_signals):
                    signals['overall_signal'] = 'sell'
                
                # Calculate confidence
                if signals['signals']:
                    signals['confidence'] = sum(s['confidence'] for s in signals['signals']) / len(signals['signals'])
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating Fibonacci trading signals: {e}")
            return {'error': str(e)}
