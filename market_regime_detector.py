
"""
Advanced Market Regime Detection System
Identifies market conditions and phases for better trading decisions
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

class MarketRegime(Enum):
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    CRISIS = "crisis"
    RECOVERY = "recovery"

@dataclass
class RegimeSignal:
    """Signal indicating market regime characteristics"""
    regime: MarketRegime
    confidence: float
    duration: int  # periods
    strength: float
    indicators: Dict[str, float]
    timestamp: datetime

class MarketRegimeDetector:
    """Advanced market regime detection and classification"""
    
    def __init__(self, lookback_period: int = 50):
        self.lookback_period = lookback_period
        self.price_history = {}
        self.volume_history = {}
        self.volatility_history = {}
        self.regime_history = {}
        self.logger = logging.getLogger("MarketRegimeDetector")
        
        # Regime detection parameters
        self.regime_thresholds = {
            'bull_trend_threshold': 0.15,  # 15% gain over lookback
            'bear_trend_threshold': -0.15,  # 15% loss over lookback
            'high_vol_threshold': 0.30,  # 30% annualized volatility
            'low_vol_threshold': 0.10,   # 10% annualized volatility
            'trending_threshold': 0.70,   # R-squared for trend detection
            'sideways_threshold': 0.05    # 5% range for sideways markets
        }
    
    def update_data(self, symbol: str, price: float, volume: int, timestamp: datetime):
        """Update market data for regime detection"""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            self.volume_history[symbol] = []
            self.volatility_history[symbol] = []
            self.regime_history[symbol] = []
        
        self.price_history[symbol].append({'price': price, 'timestamp': timestamp})
        self.volume_history[symbol].append({'volume': volume, 'timestamp': timestamp})
        
        # Keep only recent history
        if len(self.price_history[symbol]) > self.lookback_period * 2:
            self.price_history[symbol] = self.price_history[symbol][-self.lookback_period:]
            self.volume_history[symbol] = self.volume_history[symbol][-self.lookback_period:]
    
    def detect_regime(self, symbol: str) -> Optional[RegimeSignal]:
        """Detect current market regime for a symbol"""
        if symbol not in self.price_history or len(self.price_history[symbol]) < 20:
            return None
        
        try:
            prices = [p['price'] for p in self.price_history[symbol]]
            volumes = [v['volume'] for v in self.volume_history[symbol]]
            
            # Calculate various indicators
            indicators = self._calculate_regime_indicators(prices, volumes)
            
            # Detect multiple regime characteristics
            regime_scores = self._score_regimes(indicators)
            
            # Determine primary regime
            primary_regime = max(regime_scores.items(), key=lambda x: x[1])
            regime_type, confidence = primary_regime
            
            # Calculate regime strength and duration
            strength = self._calculate_regime_strength(indicators, regime_type)
            duration = self._estimate_regime_duration(symbol, regime_type)
            
            signal = RegimeSignal(
                regime=MarketRegime(regime_type),
                confidence=confidence,
                duration=duration,
                strength=strength,
                indicators=indicators,
                timestamp=datetime.now()
            )
            
            # Store in history
            self.regime_history[symbol].append(signal)
            if len(self.regime_history[symbol]) > 100:
                self.regime_history[symbol] = self.regime_history[symbol][-100:]
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error detecting regime for {symbol}: {e}")
            return None
    
    def _calculate_regime_indicators(self, prices: List[float], volumes: List[int]) -> Dict[str, float]:
        """Calculate indicators for regime detection"""
        prices_array = np.array(prices)
        volumes_array = np.array(volumes)
        
        indicators = {}
        
        # Trend indicators
        if len(prices) >= 20:
            # Linear trend over different periods
            short_trend = self._calculate_trend_strength(prices_array[-20:])
            medium_trend = self._calculate_trend_strength(prices_array[-40:] if len(prices) >= 40 else prices_array)
            long_trend = self._calculate_trend_strength(prices_array)
            
            indicators['short_trend'] = short_trend
            indicators['medium_trend'] = medium_trend
            indicators['long_trend'] = long_trend
            indicators['trend_consistency'] = np.mean([short_trend, medium_trend, long_trend])
        
        # Volatility indicators
        returns = np.diff(prices_array) / prices_array[:-1]
        if len(returns) > 0:
            indicators['volatility'] = np.std(returns) * np.sqrt(252)  # Annualized
            indicators['volatility_trend'] = self._calculate_volatility_trend(returns)
            indicators['volatility_clustering'] = self._detect_volatility_clustering(returns)
        
        # Price movement indicators
        if len(prices) > 1:
            total_return = (prices_array[-1] - prices_array[0]) / prices_array[0]
            max_price = np.max(prices_array)
            min_price = np.min(prices_array)
            current_price = prices_array[-1]
            
            indicators['total_return'] = total_return
            indicators['drawdown'] = (max_price - current_price) / max_price
            indicators['recovery'] = (current_price - min_price) / (max_price - min_price) if max_price != min_price else 0
            indicators['price_range'] = (max_price - min_price) / np.mean(prices_array)
        
        # Volume indicators
        if len(volumes) > 1:
            avg_volume = np.mean(volumes_array)
            recent_volume = np.mean(volumes_array[-10:]) if len(volumes) >= 10 else avg_volume
            indicators['volume_trend'] = (recent_volume - avg_volume) / avg_volume
            indicators['volume_volatility'] = np.std(volumes_array) / avg_volume
        
        # Momentum indicators
        if len(prices) >= 10:
            momentum_5 = (prices_array[-1] - prices_array[-6]) / prices_array[-6]
            momentum_10 = (prices_array[-1] - prices_array[-11]) / prices_array[-11] if len(prices) >= 11 else momentum_5
            
            indicators['short_momentum'] = momentum_5
            indicators['medium_momentum'] = momentum_10
            indicators['momentum_divergence'] = abs(momentum_5 - momentum_10)
        
        # Mean reversion indicators
        if len(prices) >= 20:
            mean_price = np.mean(prices_array[-20:])
            current_deviation = (prices_array[-1] - mean_price) / mean_price
            indicators['mean_reversion_signal'] = -current_deviation  # Negative when price is above mean
            
            # Calculate oscillator-like behavior
            price_oscillations = self._detect_oscillations(prices_array[-20:])
            indicators['oscillation_strength'] = price_oscillations
        
        return indicators
    
    def _calculate_trend_strength(self, prices: np.array) -> float:
        """Calculate trend strength using linear regression"""
        if len(prices) < 3:
            return 0.0
        
        x = np.arange(len(prices))
        
        # Linear regression
        slope, intercept = np.polyfit(x, prices, 1)
        fitted_values = slope * x + intercept
        
        # R-squared calculation
        ss_res = np.sum((prices - fitted_values) ** 2)
        ss_tot = np.sum((prices - np.mean(prices)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        r_squared = 1 - (ss_res / ss_tot)
        
        # Return signed trend strength
        trend_direction = 1 if slope > 0 else -1
        return trend_direction * r_squared
    
    def _calculate_volatility_trend(self, returns: np.array) -> float:
        """Calculate if volatility is increasing or decreasing"""
        if len(returns) < 20:
            return 0.0
        
        # Split returns into two halves and compare volatilities
        mid_point = len(returns) // 2
        early_vol = np.std(returns[:mid_point])
        recent_vol = np.std(returns[mid_point:])
        
        if early_vol == 0:
            return 0.0
        
        return (recent_vol - early_vol) / early_vol
    
    def _detect_volatility_clustering(self, returns: np.array) -> float:
        """Detect volatility clustering (GARCH-like behavior)"""
        if len(returns) < 10:
            return 0.0
        
        # Calculate rolling volatility
        window = min(5, len(returns) // 2)
        rolling_vol = []
        
        for i in range(window, len(returns)):
            vol = np.std(returns[i-window:i])
            rolling_vol.append(vol)
        
        if len(rolling_vol) < 2:
            return 0.0
        
        # Autocorrelation of volatility
        vol_array = np.array(rolling_vol)
        if len(vol_array) > 1:
            correlation = np.corrcoef(vol_array[:-1], vol_array[1:])[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        
        return 0.0
    
    def _detect_oscillations(self, prices: np.array) -> float:
        """Detect oscillating/sideways behavior"""
        if len(prices) < 5:
            return 0.0
        
        # Count direction changes
        direction_changes = 0
        for i in range(2, len(prices)):
            prev_dir = prices[i-1] - prices[i-2]
            curr_dir = prices[i] - prices[i-1]
            
            if prev_dir * curr_dir < 0:  # Direction change
                direction_changes += 1
        
        # Normalize by length
        oscillation_rate = direction_changes / (len(prices) - 2)
        return oscillation_rate
    
    def _score_regimes(self, indicators: Dict[str, float]) -> Dict[str, float]:
        """Score different market regimes based on indicators"""
        scores = {}
        
        # Bull Market Score
        bull_score = 0.0
        if 'long_trend' in indicators:
            bull_score += max(0, indicators['long_trend']) * 0.4
        if 'total_return' in indicators:
            bull_score += max(0, min(indicators['total_return'] / self.regime_thresholds['bull_trend_threshold'], 1.0)) * 0.3
        if 'short_momentum' in indicators:
            bull_score += max(0, indicators['short_momentum'] * 5) * 0.3  # Scale momentum
        scores['bull_market'] = min(bull_score, 1.0)
        
        # Bear Market Score
        bear_score = 0.0
        if 'long_trend' in indicators:
            bear_score += max(0, -indicators['long_trend']) * 0.4
        if 'total_return' in indicators:
            bear_score += max(0, min(-indicators['total_return'] / abs(self.regime_thresholds['bear_trend_threshold']), 1.0)) * 0.3
        if 'drawdown' in indicators:
            bear_score += min(indicators['drawdown'] * 2, 1.0) * 0.3
        scores['bear_market'] = min(bear_score, 1.0)
        
        # High Volatility Score
        if 'volatility' in indicators:
            vol_score = min(indicators['volatility'] / self.regime_thresholds['high_vol_threshold'], 1.0)
            scores['high_volatility'] = vol_score
        else:
            scores['high_volatility'] = 0.0
        
        # Low Volatility Score
        if 'volatility' in indicators:
            low_vol_score = max(0, 1 - indicators['volatility'] / self.regime_thresholds['low_vol_threshold'])
            scores['low_volatility'] = low_vol_score
        else:
            scores['low_volatility'] = 0.0
        
        # Trending Score
        if 'trend_consistency' in indicators:
            trending_score = abs(indicators['trend_consistency'])
            scores['trending'] = trending_score
        else:
            scores['trending'] = 0.0
        
        # Mean Reverting Score
        if 'oscillation_strength' in indicators:
            mean_rev_score = indicators['oscillation_strength']
            if 'price_range' in indicators:
                # Higher oscillation in smaller range = more mean reverting
                range_factor = max(0, 1 - indicators['price_range'])
                mean_rev_score *= range_factor
            scores['mean_reverting'] = min(mean_rev_score, 1.0)
        else:
            scores['mean_reverting'] = 0.0
        
        # Sideways Score
        sideways_score = 0.0
        if 'price_range' in indicators:
            sideways_score += max(0, 1 - indicators['price_range'] / self.regime_thresholds['sideways_threshold']) * 0.5
        if 'trend_consistency' in indicators:
            sideways_score += max(0, 1 - abs(indicators['trend_consistency'])) * 0.5
        scores['sideways'] = min(sideways_score, 1.0)
        
        # Crisis Score (high volatility + negative returns + large drawdowns)
        crisis_score = 0.0
        if all(k in indicators for k in ['volatility', 'total_return', 'drawdown']):
            vol_component = min(indicators['volatility'] / 0.5, 1.0) * 0.4  # 50% vol threshold for crisis
            return_component = max(0, -indicators['total_return'] / 0.3) * 0.3  # 30% negative return
            drawdown_component = min(indicators['drawdown'] / 0.2, 1.0) * 0.3  # 20% drawdown
            crisis_score = vol_component + return_component + drawdown_component
        scores['crisis'] = min(crisis_score, 1.0)
        
        # Recovery Score (positive momentum after drawdown)
        recovery_score = 0.0
        if all(k in indicators for k in ['recovery', 'short_momentum', 'drawdown']):
            if indicators['drawdown'] > 0.1:  # Had significant drawdown
                recovery_score = indicators['recovery'] * 0.6 + max(0, indicators['short_momentum'] * 2) * 0.4
        scores['recovery'] = min(recovery_score, 1.0)
        
        return scores
    
    def _calculate_regime_strength(self, indicators: Dict[str, float], regime_type: str) -> float:
        """Calculate strength of the detected regime"""
        if regime_type == 'bull_market':
            return indicators.get('long_trend', 0) * indicators.get('short_momentum', 0) * 2
        elif regime_type == 'bear_market':
            return abs(indicators.get('long_trend', 0)) * abs(indicators.get('short_momentum', 0)) * 2
        elif regime_type == 'high_volatility':
            return min(indicators.get('volatility', 0) / 0.5, 1.0)
        elif regime_type == 'trending':
            return abs(indicators.get('trend_consistency', 0))
        else:
            return 0.5  # Default moderate strength
    
    def _estimate_regime_duration(self, symbol: str, regime_type: str) -> int:
        """Estimate how long the current regime has been active"""
        if symbol not in self.regime_history:
            return 1
        
        duration = 1
        for signal in reversed(self.regime_history[symbol]):
            if signal.regime.value == regime_type:
                duration += 1
            else:
                break
        
        return min(duration, 50)  # Cap at 50 periods
    
    def get_regime_summary(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive regime summary for a symbol"""
        if symbol not in self.regime_history or not self.regime_history[symbol]:
            return {'error': 'No regime data available'}
        
        current_regime = self.regime_history[symbol][-1]
        
        # Calculate regime stability
        recent_regimes = [s.regime.value for s in self.regime_history[symbol][-10:]]
        stability = recent_regimes.count(current_regime.regime.value) / len(recent_regimes)
        
        # Get regime transitions
        regime_changes = 0
        for i in range(1, min(len(self.regime_history[symbol]), 20)):
            if (self.regime_history[symbol][-i].regime != 
                self.regime_history[symbol][-i-1].regime):
                regime_changes += 1
        
        return {
            'current_regime': current_regime.regime.value,
            'confidence': current_regime.confidence,
            'strength': current_regime.strength,
            'duration': current_regime.duration,
            'stability': stability,
            'recent_changes': regime_changes,
            'indicators': current_regime.indicators,
            'regime_history': [s.regime.value for s in self.regime_history[symbol][-20:]]
        }

def main():
    """Test the market regime detector"""
    detector = MarketRegimeDetector()
    
    # Simulate market data for different regimes
    symbols = ['AAPL', 'GOOGL']
    
    print("Testing Market Regime Detection...")
    
    # Simulate 50 days of data
    import random
    base_price = 100.0
    
    for day in range(50):
        timestamp = datetime.now() - timedelta(days=50-day)
        
        for symbol in symbols:
            # Simulate different market conditions
            if day < 20:  # Bull market
                price_change = random.uniform(0.005, 0.02)
            elif day < 35:  # High volatility
                price_change = random.uniform(-0.04, 0.04)
            else:  # Bear market
                price_change = random.uniform(-0.02, -0.005)
            
            base_price *= (1 + price_change)
            volume = random.randint(1000000, 5000000)
            
            detector.update_data(symbol, base_price, volume, timestamp)
            
            # Detect regime every 5 days
            if day % 5 == 0:
                regime = detector.detect_regime(symbol)
                if regime:
                    print(f"Day {day} - {symbol}: {regime.regime.value} "
                          f"(confidence: {regime.confidence:.2f}, "
                          f"strength: {regime.strength:.2f})")
    
    # Display final summaries
    for symbol in symbols:
        print(f"\nFinal Regime Summary for {symbol}:")
        summary = detector.get_regime_summary(symbol)
        print(f"Current Regime: {summary['current_regime']}")
        print(f"Confidence: {summary['confidence']:.2f}")
        print(f"Stability: {summary['stability']:.2f}")
        print(f"Recent Changes: {summary['recent_changes']}")

if __name__ == "__main__":
    main()
