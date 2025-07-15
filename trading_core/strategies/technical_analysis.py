"""Streamlined technical analysis indicators"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional

from ..data_models import MarketData, TechnicalAnalysis

# Optional pandas-ta for enhanced indicators
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False

class TechnicalIndicators:
    """Streamlined technical analysis indicators calculator"""

    def __init__(self):
        self.price_history = {}
        self.volume_history = {}
        self.high_history = {}
        self.low_history = {}
        self.support_resistance_levels = {}

    def update_data(self, market_data: MarketData):
        """Update historical data for technical analysis"""
        symbol = market_data.symbol

        # Initialize histories if not exists
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            self.volume_history[symbol] = []
            self.high_history[symbol] = []
            self.low_history[symbol] = []

        # Add new data
        self.price_history[symbol].append(market_data.price)
        self.volume_history[symbol].append(market_data.volume)
        
        # Use provided high/low or estimate
        high = max(market_data.high_24h, market_data.price)
        low = min(market_data.low_24h, market_data.price) if market_data.low_24h > 0 else market_data.price * 0.98
        
        self.high_history[symbol].append(high)
        self.low_history[symbol].append(low)

        # Update support/resistance levels
        self._update_support_resistance(symbol)

        # Keep only last 200 periods for efficiency
        if len(self.price_history[symbol]) > 200:
            for hist in [self.price_history, self.volume_history, self.high_history, self.low_history]:
                hist[symbol] = hist[symbol][-200:]

    def calculate_indicators(self, symbol: str) -> Optional[TechnicalAnalysis]:
        """Calculate technical indicators for a symbol"""
        if symbol not in self.price_history or len(self.price_history[symbol]) < 10:
            return None

        prices = np.array(self.price_history[symbol])
        volumes = np.array(self.volume_history[symbol])
        highs = np.array(self.high_history[symbol])
        lows = np.array(self.low_history[symbol])

        try:
            # Core indicators
            sma_20 = self._sma(prices, min(20, len(prices)))
            sma_50 = self._sma(prices, min(50, len(prices)))
            ema_12 = self._ema(prices, min(12, len(prices)))
            ema_26 = self._ema(prices, min(26, len(prices)))
            
            rsi = self._calculate_rsi(prices, 14)
            macd, macd_signal, macd_histogram = self._calculate_macd(prices)
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(prices, 20, 2)
            
            volume_sma = self._sma(volumes, 20)
            atr = self._calculate_atr(highs, lows, prices, 14)
            
            # Additional indicators
            stochastic_k, stochastic_d = self._calculate_stochastic(highs, lows, prices, 14, 3)
            williams_r = self._calculate_williams_r(highs, lows, prices, 14)
            volatility = self._calculate_volatility(prices)
            fibonacci_levels = self._calculate_fibonacci_levels(prices)
            
            # Support/resistance
            sr_levels = self.support_resistance_levels.get(symbol, {'support': 0, 'resistance': 0})

            return TechnicalAnalysis(
                symbol=symbol,
                sma_20=sma_20,
                sma_50=sma_50,
                ema_12=ema_12,
                ema_26=ema_26,
                rsi=rsi,
                macd=macd,
                macd_signal=macd_signal,
                macd_histogram=macd_histogram,
                bb_upper=bb_upper,
                bb_middle=bb_middle,
                bb_lower=bb_lower,
                volume_sma=volume_sma,
                atr=atr,
                fibonacci_levels=fibonacci_levels,
                stochastic_k=stochastic_k,
                stochastic_d=stochastic_d,
                williams_r=williams_r,
                timestamp=datetime.now(),
                support_level=sr_levels['support'],
                resistance_level=sr_levels['resistance'],
                volatility=volatility
            )

        except Exception as e:
            print(f"Error calculating indicators for {symbol}: {e}")
            return None

    def _update_support_resistance(self, symbol: str):
        """Update support and resistance levels"""
        if len(self.price_history[symbol]) < 20:
            return

        highs = np.array(self.high_history[symbol][-50:])
        lows = np.array(self.low_history[symbol][-50:])

        resistance = np.percentile(highs, 90)
        support = np.percentile(lows, 10)

        self.support_resistance_levels[symbol] = {
            'support': support,
            'resistance': resistance
        }

    def _sma(self, prices: np.array, period: int) -> float:
        """Simple Moving Average"""
        if len(prices) < period:
            return prices[-1]
        return np.mean(prices[-period:])

    def _ema(self, prices: np.array, period: int) -> float:
        """Exponential Moving Average"""
        if len(prices) < period:
            return prices[-1]

        multiplier = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        return ema

    def _calculate_rsi(self, prices: np.array, period: int = 14) -> float:
        """Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0

        if PANDAS_TA_AVAILABLE:
            try:
                price_series = pd.Series(prices)
                rsi_series = ta.rsi(price_series, length=period)
                return rsi_series.iloc[-1] if not pd.isna(rsi_series.iloc[-1]) else 50.0
            except:
                pass

        # Fallback calculation
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: np.array) -> tuple:
        """MACD calculation"""
        if len(prices) < 26:
            return 0.0, 0.0, 0.0

        ema_12 = self._ema(prices, 12)
        ema_26 = self._ema(prices, 26)
        macd = ema_12 - ema_26
        signal = macd * 0.9  # Simplified signal line
        histogram = macd - signal

        return macd, signal, histogram

    def _calculate_bollinger_bands(self, prices: np.array, period: int = 20, std_dev: int = 2) -> tuple:
        """Bollinger Bands"""
        if len(prices) < period:
            price = prices[-1]
            return price * 1.02, price, price * 0.98

        sma = self._sma(prices, period)
        std = np.std(prices[-period:])

        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)

        return upper, sma, lower

    def _calculate_atr(self, highs: np.array, lows: np.array, closes: np.array, period: int = 14) -> float:
        """Average True Range"""
        if len(highs) < period + 1:
            return abs(highs[-1] - lows[-1]) if len(highs) > 0 else 1.0

        true_ranges = []
        for i in range(1, len(highs)):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            true_ranges.append(max(tr1, tr2, tr3))

        return np.mean(true_ranges[-period:])

    def _calculate_stochastic(self, highs: np.array, lows: np.array, closes: np.array, k_period: int = 14, d_period: int = 3) -> tuple:
        """Stochastic Oscillator"""
        if len(highs) < k_period:
            return 50.0, 50.0

        lowest_low = np.min(lows[-k_period:])
        highest_high = np.max(highs[-k_period:])

        if highest_high == lowest_low:
            return 50.0, 50.0

        k_percent = ((closes[-1] - lowest_low) / (highest_high - lowest_low)) * 100
        return k_percent, k_percent  # Simplified D%

    def _calculate_williams_r(self, highs: np.array, lows: np.array, closes: np.array, period: int = 14) -> float:
        """Williams %R"""
        if len(highs) < period:
            return -50.0

        highest_high = np.max(highs[-period:])
        lowest_low = np.min(lows[-period:])

        if highest_high == lowest_low:
            return -50.0

        return ((highest_high - closes[-1]) / (highest_high - lowest_low)) * -100

    def _calculate_volatility(self, prices: np.array, window: int = 20) -> float:
        """Calculate price volatility"""
        if len(prices) < window + 1:
            return 0.0

        returns = np.diff(prices[-window-1:]) / prices[-window-1:-1]
        return np.std(returns) * np.sqrt(252)  # Annualized

    def _calculate_fibonacci_levels(self, prices: np.array) -> Dict[str, float]:
        """Fibonacci Retracement Levels"""
        if len(prices) < 20:
            price = prices[-1]
            return {'23.6%': price, '38.2%': price, '50%': price, '61.8%': price}

        high = np.max(prices[-50:])
        low = np.min(prices[-50:])
        diff = high - low

        return {
            '23.6%': high - (diff * 0.236),
            '38.2%': high - (diff * 0.382),
            '50%': high - (diff * 0.5),
            '61.8%': high - (diff * 0.618)
        }