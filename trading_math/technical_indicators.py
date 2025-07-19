"""
Technical indicators for trading analysis
"""

# Safe import for numpy/pandas with fallback
try:
    import numpy as np
    import pandas as pd
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Create mock numpy/pandas for basic functionality
    class MockNumpy:
        @staticmethod
        def array(data):
            return data
        @staticmethod
        def mean(data):
            return sum(data) / len(data) if data else 0
        @staticmethod
        def std(data):
            if not data:
                return 0
            mean = sum(data) / len(data)
            return (sum((x - mean) ** 2 for x in data) / len(data)) ** 0.5
    
    class MockPandas:
        @staticmethod
        def DataFrame(data):
            return data
    
    np = MockNumpy()
    pd = MockPandas()
from typing import Dict, List, Optional, Tuple
from datetime import datetime

class TechnicalIndicators:
    """Collection of technical indicators for market analysis"""
    
    def __init__(self):
        self.indicators = {}
    
    def rsi(self, prices: List[float], period: int = 14) -> List[float]:
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            prices: List of closing prices
            period: RSI period (default 14)
            
        Returns:
            List of RSI values
        """
        try:
            if len(prices) < period + 1:
                return []
            
            # Calculate price changes
            changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
            
            # Separate gains and losses
            gains = [change if change > 0 else 0 for change in changes]
            losses = [-change if change < 0 else 0 for change in changes]
            
            # Calculate initial average gain and loss
            avg_gain = sum(gains[:period]) / period
            avg_loss = sum(losses[:period]) / period
            
            rsi_values = []
            
            # Calculate RSI for each period
            for i in range(period, len(gains)):
                if avg_loss == 0:
                    rsi_values.append(100)
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                    rsi_values.append(rsi)
                
                # Update averages using smoothing
                avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
                avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period
            
            return rsi_values
            
        except Exception as e:
            return []
    
    def macd(self, prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            prices: List of closing prices
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period
            
        Returns:
            Dictionary with MACD line, signal line, and histogram
        """
        try:
            if len(prices) < slow:
                return {"macd": [], "signal": [], "histogram": []}
            
            # Calculate EMAs
            ema_fast = self.ema(prices, fast)
            ema_slow = self.ema(prices, slow)
            
            # Calculate MACD line
            macd_line = []
            for i in range(len(ema_slow)):
                if i < len(ema_fast):
                    macd_line.append(ema_fast[i] - ema_slow[i])
            
            # Calculate signal line
            signal_line = self.ema(macd_line, signal)
            
            # Calculate histogram
            histogram = []
            for i in range(len(signal_line)):
                if i < len(macd_line):
                    histogram.append(macd_line[i] - signal_line[i])
            
            return {
                "macd": macd_line,
                "signal": signal_line,
                "histogram": histogram
            }
            
        except Exception as e:
            return {"macd": [], "signal": [], "histogram": []}
    
    def bollinger_bands(self, prices: List[float], period: int = 20, std: float = 2.0) -> Dict:
        """
        Calculate Bollinger Bands
        
        Args:
            prices: List of closing prices
            period: Moving average period
            std: Standard deviation multiplier
            
        Returns:
            Dictionary with upper band, middle band (SMA), and lower band
        """
        try:
            if len(prices) < period:
                return {"upper": [], "middle": [], "lower": []}
            
            # Calculate simple moving average
            sma = self.sma(prices, period)
            
            upper_band = []
            lower_band = []
            
            for i in range(len(sma)):
                # Calculate standard deviation for the period
                start_idx = i
                end_idx = i + period
                
                if end_idx <= len(prices):
                    period_prices = prices[start_idx:end_idx]
                    std_dev = np.std(period_prices)
                    
                    upper_band.append(sma[i] + (std * std_dev))
                    lower_band.append(sma[i] - (std * std_dev))
            
            return {
                "upper": upper_band,
                "middle": sma,
                "lower": lower_band
            }
            
        except Exception as e:
            return {"upper": [], "middle": [], "lower": []}
    
    def stochastic(self, highs: List[float], lows: List[float], closes: List[float], 
                   k_period: int = 14, d_period: int = 3) -> Dict:
        """
        Calculate Stochastic Oscillator
        
        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of closing prices
            k_period: %K period
            d_period: %D period
            
        Returns:
            Dictionary with %K and %D values
        """
        try:
            if len(closes) < k_period:
                return {"k": [], "d": []}
            
            k_values = []
            
            for i in range(k_period - 1, len(closes)):
                # Get highest high and lowest low for the period
                period_highs = highs[i - k_period + 1:i + 1]
                period_lows = lows[i - k_period + 1:i + 1]
                
                highest_high = max(period_highs)
                lowest_low = min(period_lows)
                
                # Calculate %K
                if highest_high == lowest_low:
                    k_values.append(50)
                else:
                    k = ((closes[i] - lowest_low) / (highest_high - lowest_low)) * 100
                    k_values.append(k)
            
            # Calculate %D (SMA of %K)
            d_values = self.sma(k_values, d_period)
            
            return {
                "k": k_values,
                "d": d_values
            }
            
        except Exception as e:
            return {"k": [], "d": []}
    
    def williams_r(self, highs: List[float], lows: List[float], closes: List[float], 
                   period: int = 14) -> List[float]:
        """
        Calculate Williams %R
        
        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of closing prices
            period: Lookback period
            
        Returns:
            List of Williams %R values
        """
        try:
            if len(closes) < period:
                return []
            
            williams_r = []
            
            for i in range(period - 1, len(closes)):
                # Get highest high and lowest low for the period
                period_highs = highs[i - period + 1:i + 1]
                period_lows = lows[i - period + 1:i + 1]
                
                highest_high = max(period_highs)
                lowest_low = min(period_lows)
                
                # Calculate Williams %R
                if highest_high == lowest_low:
                    williams_r.append(-50)
                else:
                    wr = ((highest_high - closes[i]) / (highest_high - lowest_low)) * -100
                    williams_r.append(wr)
            
            return williams_r
            
        except Exception as e:
            return []
    
    def atr(self, highs: List[float], lows: List[float], closes: List[float], 
            period: int = 14) -> List[float]:
        """
        Calculate Average True Range (ATR)
        
        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of closing prices
            period: ATR period
            
        Returns:
            List of ATR values
        """
        try:
            if len(closes) < period + 1:
                return []
            
            true_ranges = []
            
            for i in range(1, len(closes)):
                # Calculate True Range
                tr1 = highs[i] - lows[i]
                tr2 = abs(highs[i] - closes[i-1])
                tr3 = abs(lows[i] - closes[i-1])
                
                true_range = max(tr1, tr2, tr3)
                true_ranges.append(true_range)
            
            # Calculate ATR (smoothed average of true ranges)
            atr_values = []
            
            # Initial ATR (simple average)
            initial_atr = sum(true_ranges[:period]) / period
            atr_values.append(initial_atr)
            
            # Subsequent ATRs (smoothed)
            for i in range(period, len(true_ranges)):
                atr = ((atr_values[-1] * (period - 1)) + true_ranges[i]) / period
                atr_values.append(atr)
            
            return atr_values
            
        except Exception as e:
            return []
    
    def sma(self, prices: List[float], period: int) -> List[float]:
        """
        Calculate Simple Moving Average
        
        Args:
            prices: List of prices
            period: Moving average period
            
        Returns:
            List of SMA values
        """
        try:
            if len(prices) < period:
                return []
            
            sma_values = []
            
            for i in range(period - 1, len(prices)):
                avg = sum(prices[i - period + 1:i + 1]) / period
                sma_values.append(avg)
            
            return sma_values
            
        except Exception as e:
            return []
    
    def ema(self, prices: List[float], period: int) -> List[float]:
        """
        Calculate Exponential Moving Average
        
        Args:
            prices: List of prices
            period: EMA period
            
        Returns:
            List of EMA values
        """
        try:
            if len(prices) < period:
                return []
            
            # Calculate smoothing factor
            multiplier = 2 / (period + 1)
            
            ema_values = []
            
            # Initial EMA (SMA)
            initial_ema = sum(prices[:period]) / period
            ema_values.append(initial_ema)
            
            # Calculate subsequent EMAs
            for i in range(period, len(prices)):
                ema = (prices[i] * multiplier) + (ema_values[-1] * (1 - multiplier))
                ema_values.append(ema)
            
            return ema_values
            
        except Exception as e:
            return []
    
    def momentum(self, prices: List[float], period: int = 10) -> List[float]:
        """
        Calculate Momentum indicator
        
        Args:
            prices: List of prices
            period: Momentum period
            
        Returns:
            List of momentum values
        """
        try:
            if len(prices) < period:
                return []
            
            momentum_values = []
            
            for i in range(period, len(prices)):
                mom = prices[i] - prices[i - period]
                momentum_values.append(mom)
            
            return momentum_values
            
        except Exception as e:
            return []
    
    def roc(self, prices: List[float], period: int = 10) -> List[float]:
        """
        Calculate Rate of Change (ROC)
        
        Args:
            prices: List of prices
            period: ROC period
            
        Returns:
            List of ROC values
        """
        try:
            if len(prices) < period:
                return []
            
            roc_values = []
            
            for i in range(period, len(prices)):
                if prices[i - period] != 0:
                    roc = ((prices[i] - prices[i - period]) / prices[i - period]) * 100
                    roc_values.append(roc)
                else:
                    roc_values.append(0)
            
            return roc_values
            
        except Exception as e:
            return []
    
    def cci(self, highs: List[float], lows: List[float], closes: List[float], 
            period: int = 20) -> List[float]:
        """
        Calculate Commodity Channel Index (CCI)
        
        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of closing prices
            period: CCI period
            
        Returns:
            List of CCI values
        """
        try:
            if len(closes) < period:
                return []
            
            # Calculate typical prices
            typical_prices = []
            for i in range(len(closes)):
                tp = (highs[i] + lows[i] + closes[i]) / 3
                typical_prices.append(tp)
            
            cci_values = []
            
            for i in range(period - 1, len(typical_prices)):
                # Calculate SMA of typical prices
                sma_tp = sum(typical_prices[i - period + 1:i + 1]) / period
                
                # Calculate mean deviation
                deviations = [abs(typical_prices[j] - sma_tp) for j in range(i - period + 1, i + 1)]
                mean_deviation = sum(deviations) / period
                
                # Calculate CCI
                if mean_deviation != 0:
                    cci = (typical_prices[i] - sma_tp) / (0.015 * mean_deviation)
                    cci_values.append(cci)
                else:
                    cci_values.append(0)
            
            return cci_values
            
        except Exception as e:
            return []
    
    def adx(self, highs: List[float], lows: List[float], closes: List[float], 
            period: int = 14) -> Dict:
        """
        Calculate Average Directional Index (ADX)
        
        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of closing prices
            period: ADX period
            
        Returns:
            Dictionary with ADX, +DI, and -DI values
        """
        try:
            if len(closes) < period + 1:
                return {"adx": [], "plus_di": [], "minus_di": []}
            
            # Calculate True Range and Directional Movement
            tr_values = []
            plus_dm = []
            minus_dm = []
            
            for i in range(1, len(closes)):
                # True Range
                tr1 = highs[i] - lows[i]
                tr2 = abs(highs[i] - closes[i-1])
                tr3 = abs(lows[i] - closes[i-1])
                tr = max(tr1, tr2, tr3)
                tr_values.append(tr)
                
                # Directional Movement
                up_move = highs[i] - highs[i-1]
                down_move = lows[i-1] - lows[i]
                
                plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0)
                minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0)
            
            # Calculate smoothed averages
            atr_values = self._smooth_average(tr_values, period)
            plus_di_values = []
            minus_di_values = []
            
            smoothed_plus_dm = self._smooth_average(plus_dm, period)
            smoothed_minus_dm = self._smooth_average(minus_dm, period)
            
            # Calculate DI values
            for i in range(len(atr_values)):
                if atr_values[i] != 0:
                    plus_di_values.append((smoothed_plus_dm[i] / atr_values[i]) * 100)
                    minus_di_values.append((smoothed_minus_dm[i] / atr_values[i]) * 100)
                else:
                    plus_di_values.append(0)
                    minus_di_values.append(0)
            
            # Calculate ADX
            adx_values = []
            dx_values = []
            
            for i in range(len(plus_di_values)):
                di_sum = plus_di_values[i] + minus_di_values[i]
                if di_sum != 0:
                    dx = (abs(plus_di_values[i] - minus_di_values[i]) / di_sum) * 100
                    dx_values.append(dx)
                else:
                    dx_values.append(0)
            
            # Smooth DX to get ADX
            adx_values = self._smooth_average(dx_values, period)
            
            return {
                "adx": adx_values,
                "plus_di": plus_di_values,
                "minus_di": minus_di_values
            }
            
        except Exception as e:
            return {"adx": [], "plus_di": [], "minus_di": []}
    
    def _smooth_average(self, values: List[float], period: int) -> List[float]:
        """Calculate smoothed average (Wilder's smoothing)"""
        try:
            if len(values) < period:
                return []
            
            smoothed = []
            
            # Initial value (simple average)
            initial = sum(values[:period]) / period
            smoothed.append(initial)
            
            # Subsequent values (smoothed)
            for i in range(period, len(values)):
                smooth = ((smoothed[-1] * (period - 1)) + values[i]) / period
                smoothed.append(smooth)
            
            return smoothed
            
        except Exception as e:
            return []
    
    def pivot_points(self, high: float, low: float, close: float) -> Dict:
        """
        Calculate pivot points and support/resistance levels
        
        Args:
            high: Previous period high
            low: Previous period low
            close: Previous period close
            
        Returns:
            Dictionary with pivot point and support/resistance levels
        """
        try:
            # Calculate pivot point
            pivot = (high + low + close) / 3
            
            # Calculate support and resistance levels
            r1 = (2 * pivot) - low
            s1 = (2 * pivot) - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            r3 = high + 2 * (pivot - low)
            s3 = low - 2 * (high - pivot)
            
            return {
                "pivot": pivot,
                "r1": r1,
                "r2": r2,
                "r3": r3,
                "s1": s1,
                "s2": s2,
                "s3": s3
            }
            
        except Exception as e:
            return {}
    
    def ichimoku(self, highs: List[float], lows: List[float], closes: List[float]) -> Dict:
        """
        Calculate Ichimoku Kinko Hyo indicators
        
        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of closing prices
            
        Returns:
            Dictionary with Ichimoku lines
        """
        try:
            if len(closes) < 52:
                return {}
            
            # Calculate lines
            tenkan_sen = []  # (9-period high + 9-period low) / 2
            kijun_sen = []   # (26-period high + 26-period low) / 2
            senkou_span_a = []  # (Tenkan-sen + Kijun-sen) / 2
            senkou_span_b = []  # (52-period high + 52-period low) / 2
            chikou_span = []    # Close shifted back 26 periods
            
            for i in range(52, len(closes)):
                # Tenkan-sen (9 periods)
                if i >= 8:
                    tenkan_high = max(highs[i-8:i+1])
                    tenkan_low = min(lows[i-8:i+1])
                    tenkan = (tenkan_high + tenkan_low) / 2
                    tenkan_sen.append(tenkan)
                
                # Kijun-sen (26 periods)
                if i >= 25:
                    kijun_high = max(highs[i-25:i+1])
                    kijun_low = min(lows[i-25:i+1])
                    kijun = (kijun_high + kijun_low) / 2
                    kijun_sen.append(kijun)
                
                # Senkou Span A
                if len(tenkan_sen) > 0 and len(kijun_sen) > 0:
                    senkou_a = (tenkan_sen[-1] + kijun_sen[-1]) / 2
                    senkou_span_a.append(senkou_a)
                
                # Senkou Span B (52 periods)
                senkou_high = max(highs[i-51:i+1])
                senkou_low = min(lows[i-51:i+1])
                senkou_b = (senkou_high + senkou_low) / 2
                senkou_span_b.append(senkou_b)
                
                # Chikou Span
                if i >= 26:
                    chikou_span.append(closes[i-26])
            
            return {
                "tenkan_sen": tenkan_sen,
                "kijun_sen": kijun_sen,
                "senkou_span_a": senkou_span_a,
                "senkou_span_b": senkou_span_b,
                "chikou_span": chikou_span
            }
            
        except Exception as e:
            return {}
    
    def calculate_all_indicators(self, price_data: List[Dict]) -> Dict:
        """
        Calculate all available indicators for price data
        
        Args:
            price_data: List of price dictionaries with OHLCV data
            
        Returns:
            Dictionary with all calculated indicators
        """
        try:
            if not price_data:
                return {}
            
            # Extract price arrays
            highs = [bar["high"] for bar in price_data]
            lows = [bar["low"] for bar in price_data]
            closes = [bar["close"] for bar in price_data]
            
            # Calculate all indicators
            indicators = {
                "rsi": self.rsi(closes),
                "macd": self.macd(closes),
                "bollinger_bands": self.bollinger_bands(closes),
                "stochastic": self.stochastic(highs, lows, closes),
                "williams_r": self.williams_r(highs, lows, closes),
                "atr": self.atr(highs, lows, closes),
                "sma_20": self.sma(closes, 20),
                "sma_50": self.sma(closes, 50),
                "ema_12": self.ema(closes, 12),
                "ema_26": self.ema(closes, 26),
                "momentum": self.momentum(closes),
                "roc": self.roc(closes),
                "cci": self.cci(highs, lows, closes),
                "adx": self.adx(highs, lows, closes),
                "ichimoku": self.ichimoku(highs, lows, closes)
            }
            
            # Add pivot points for last bar
            if len(price_data) > 0:
                last_bar = price_data[-1]
                indicators["pivot_points"] = self.pivot_points(
                    last_bar["high"], 
                    last_bar["low"], 
                    last_bar["close"]
                )
            
            return indicators
            
        except Exception as e:
            return {"error": f"Error calculating indicators: {str(e)}"}
