"""
Mathematical Models for Trading
Advanced mathematical analysis including Fibonacci, technical indicators, and statistical models
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy import stats
from scipy.signal import find_peaks
import talib
from datetime import datetime, timedelta

@dataclass
class FibonacciLevels:
    """Fibonacci retracement and extension levels"""
    symbol: str
    high: float
    low: float
    direction: str  # 'up' or 'down'
    retracement_levels: Dict[str, float]
    extension_levels: Dict[str, float]
    support_levels: List[float]
    resistance_levels: List[float]
    timestamp: datetime

@dataclass
class TechnicalIndicators:
    """Technical indicator values"""
    symbol: str
    rsi: float
    macd: Dict[str, float]
    bollinger_bands: Dict[str, float]
    stochastic: Dict[str, float]
    williams_r: float
    sma_20: float
    sma_50: float
    sma_200: float
    ema_12: float
    ema_26: float
    volume_sma: float
    timestamp: datetime

@dataclass
class StatisticalAnalysis:
    """Statistical analysis results"""
    symbol: str
    volatility: float
    sharpe_ratio: float
    beta: float
    correlation_spy: float
    mean_reversion_score: float
    momentum_score: float
    trend_strength: float
    support_resistance_zones: List[Tuple[float, float]]
    timestamp: datetime

class MathematicalModels:
    """
    Advanced mathematical models for trading analysis
    """
    
    def __init__(self, config=None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Fibonacci ratios
        self.fib_ratios = {
            'retracement': [0.236, 0.382, 0.5, 0.618, 0.786],
            'extension': [1.272, 1.414, 1.618, 2.618, 4.236]
        }
        
        # Technical indicator parameters
        self.indicator_params = {
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2,
            'stoch_k': 14,
            'stoch_d': 3,
            'williams_period': 14
        }
        
    def calculate_fibonacci_levels(self, data: pd.DataFrame, 
                                 lookback_periods: int = 50) -> FibonacciLevels:
        """
        Calculate Fibonacci retracement and extension levels
        
        Args:
            data: OHLCV data
            lookback_periods: Number of periods to look back for high/low
            
        Returns:
            FibonacciLevels object with all calculated levels
        """
        try:
            if len(data) < lookback_periods:
                lookback_periods = len(data)
                
            # Get recent data for analysis
            recent_data = data.tail(lookback_periods)
            
            # Find significant high and low
            high_price = recent_data['high'].max()
            low_price = recent_data['low'].min()
            
            # Determine trend direction
            current_price = data['close'].iloc[-1]
            direction = 'up' if current_price > (high_price + low_price) / 2 else 'down'
            
            # Calculate retracement levels
            price_range = high_price - low_price
            retracement_levels = {}
            
            if direction == 'up':
                # For uptrend, retracements are below the high
                for ratio in self.fib_ratios['retracement']:
                    level = high_price - (price_range * ratio)
                    retracement_levels[f'{ratio:.3f}'] = level
            else:
                # For downtrend, retracements are above the low
                for ratio in self.fib_ratios['retracement']:
                    level = low_price + (price_range * ratio)
                    retracement_levels[f'{ratio:.3f}'] = level
                    
            # Calculate extension levels
            extension_levels = {}
            
            if direction == 'up':
                # For uptrend, extensions are above the high
                for ratio in self.fib_ratios['extension']:
                    level = high_price + (price_range * (ratio - 1))
                    extension_levels[f'{ratio:.3f}'] = level
            else:
                # For downtrend, extensions are below the low
                for ratio in self.fib_ratios['extension']:
                    level = low_price - (price_range * (ratio - 1))
                    extension_levels[f'{ratio:.3f}'] = level
                    
            # Identify support and resistance levels
            support_levels = []
            resistance_levels = []
            
            all_levels = list(retracement_levels.values()) + list(extension_levels.values())
            
            for level in all_levels:
                if level < current_price:
                    support_levels.append(level)
                else:
                    resistance_levels.append(level)
                    
            support_levels.sort(reverse=True)  # Closest support first
            resistance_levels.sort()  # Closest resistance first
            
            return FibonacciLevels(
                symbol=data.get('symbol', 'UNKNOWN'),
                high=high_price,
                low=low_price,
                direction=direction,
                retracement_levels=retracement_levels,
                extension_levels=extension_levels,
                support_levels=support_levels[:3],  # Top 3 support levels
                resistance_levels=resistance_levels[:3],  # Top 3 resistance levels
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"❌ Fibonacci calculation error: {e}")
            return None
            
    def calculate_technical_indicators(self, data: pd.DataFrame) -> TechnicalIndicators:
        """
        Calculate comprehensive technical indicators
        
        Args:
            data: OHLCV data
            
        Returns:
            TechnicalIndicators object with all calculated indicators
        """
        try:
            if len(data) < 200:
                self.logger.warning("⚠️ Insufficient data for all indicators")
                
            # Extract price arrays
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            volume = data['volume'].values
            
            # RSI
            rsi = talib.RSI(close, timeperiod=self.indicator_params['rsi_period'])[-1]
            
            # MACD
            macd_line, macd_signal, macd_histogram = talib.MACD(
                close,
                fastperiod=self.indicator_params['macd_fast'],
                slowperiod=self.indicator_params['macd_slow'],
                signalperiod=self.indicator_params['macd_signal']
            )
            
            macd_dict = {
                'macd': macd_line[-1],
                'signal': macd_signal[-1],
                'histogram': macd_histogram[-1]
            }
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                close,
                timeperiod=self.indicator_params['bb_period'],
                nbdevup=self.indicator_params['bb_std'],
                nbdevdn=self.indicator_params['bb_std']
            )
            
            bb_dict = {
                'upper': bb_upper[-1],
                'middle': bb_middle[-1],
                'lower': bb_lower[-1],
                'width': bb_upper[-1] - bb_lower[-1],
                'position': (close[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
            }
            
            # Stochastic
            stoch_k, stoch_d = talib.STOCH(
                high, low, close,
                fastk_period=self.indicator_params['stoch_k'],
                slowk_period=self.indicator_params['stoch_d'],
                slowd_period=self.indicator_params['stoch_d']
            )
            
            stoch_dict = {
                'k': stoch_k[-1],
                'd': stoch_d[-1]
            }
            
            # Williams %R
            williams_r = talib.WILLR(
                high, low, close,
                timeperiod=self.indicator_params['williams_period']
            )[-1]
            
            # Moving Averages
            sma_20 = talib.SMA(close, timeperiod=20)[-1]
            sma_50 = talib.SMA(close, timeperiod=50)[-1] if len(close) >= 50 else close[-1]
            sma_200 = talib.SMA(close, timeperiod=200)[-1] if len(close) >= 200 else close[-1]
            
            # Exponential Moving Averages
            ema_12 = talib.EMA(close, timeperiod=12)[-1]
            ema_26 = talib.EMA(close, timeperiod=26)[-1]
            
            # Volume SMA
            volume_sma = talib.SMA(volume, timeperiod=20)[-1]
            
            return TechnicalIndicators(
                symbol=data.get('symbol', 'UNKNOWN'),
                rsi=rsi,
                macd=macd_dict,
                bollinger_bands=bb_dict,
                stochastic=stoch_dict,
                williams_r=williams_r,
                sma_20=sma_20,
                sma_50=sma_50,
                sma_200=sma_200,
                ema_12=ema_12,
                ema_26=ema_26,
                volume_sma=volume_sma,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"❌ Technical indicators calculation error: {e}")
            return None
            
    def calculate_statistical_analysis(self, data: pd.DataFrame, 
                                     benchmark_data: pd.DataFrame = None) -> StatisticalAnalysis:
        """
        Calculate statistical analysis metrics
        
        Args:
            data: OHLCV data for the asset
            benchmark_data: Optional benchmark data (e.g., SPY)
            
        Returns:
            StatisticalAnalysis object with statistical metrics
        """
        try:
            # Calculate returns
            returns = data['close'].pct_change().dropna()
            
            # Volatility (annualized)
            volatility = returns.std() * np.sqrt(252)
            
            # Sharpe ratio (assuming risk-free rate of 2%)
            risk_free_rate = 0.02
            excess_returns = returns.mean() * 252 - risk_free_rate
            sharpe_ratio = excess_returns / volatility if volatility > 0 else 0
            
            # Beta calculation (vs benchmark)
            beta = 1.0
            correlation_spy = 0.0
            
            if benchmark_data is not None:
                benchmark_returns = benchmark_data['close'].pct_change().dropna()
                if len(benchmark_returns) > 0:
                    # Align data
                    aligned_data = pd.DataFrame({
                        'asset': returns,
                        'benchmark': benchmark_returns
                    }).dropna()
                    
                    if len(aligned_data) > 30:
                        covariance = aligned_data['asset'].cov(aligned_data['benchmark'])
                        benchmark_variance = aligned_data['benchmark'].var()
                        beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
                        correlation_spy = aligned_data['asset'].corr(aligned_data['benchmark'])
                        
            # Mean reversion score
            mean_reversion_score = self._calculate_mean_reversion_score(data)
            
            # Momentum score
            momentum_score = self._calculate_momentum_score(data)
            
            # Trend strength
            trend_strength = self._calculate_trend_strength(data)
            
            # Support and resistance zones
            support_resistance_zones = self._identify_support_resistance_zones(data)
            
            return StatisticalAnalysis(
                symbol=data.get('symbol', 'UNKNOWN'),
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                beta=beta,
                correlation_spy=correlation_spy,
                mean_reversion_score=mean_reversion_score,
                momentum_score=momentum_score,
                trend_strength=trend_strength,
                support_resistance_zones=support_resistance_zones,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"❌ Statistical analysis error: {e}")
            return None
            
    def _calculate_mean_reversion_score(self, data: pd.DataFrame) -> float:
        """Calculate mean reversion tendency score"""
        try:
            close_prices = data['close'].values
            
            # Calculate rolling mean and standard deviation
            window = min(20, len(close_prices) // 2)
            rolling_mean = pd.Series(close_prices).rolling(window=window).mean()
            rolling_std = pd.Series(close_prices).rolling(window=window).std()
            
            # Calculate Z-score
            z_score = (close_prices - rolling_mean) / rolling_std
            
            # Mean reversion score based on Z-score reversals
            reversals = 0
            for i in range(1, len(z_score)):
                if not pd.isna(z_score[i-1]) and not pd.isna(z_score[i]):
                    if (z_score[i-1] > 1.5 and z_score[i] < 1.5) or \
                       (z_score[i-1] < -1.5 and z_score[i] > -1.5):
                        reversals += 1
                        
            return reversals / max(1, len(z_score) - 1)
            
        except Exception as e:
            self.logger.error(f"❌ Mean reversion score calculation error: {e}")
            return 0.0
            
    def _calculate_momentum_score(self, data: pd.DataFrame) -> float:
        """Calculate momentum score"""
        try:
            close_prices = data['close'].values
            
            # Calculate multiple timeframe momentum
            momentum_periods = [5, 10, 20, 50]
            momentum_scores = []
            
            for period in momentum_periods:
                if len(close_prices) > period:
                    momentum = (close_prices[-1] - close_prices[-period-1]) / close_prices[-period-1]
                    momentum_scores.append(momentum)
                    
            if momentum_scores:
                return np.mean(momentum_scores)
            return 0.0
            
        except Exception as e:
            self.logger.error(f"❌ Momentum score calculation error: {e}")
            return 0.0
            
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength using ADX-like calculation"""
        try:
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            
            # Calculate ADX
            adx = talib.ADX(high, low, close, timeperiod=14)
            
            # Return latest ADX value normalized to 0-1 scale
            if not pd.isna(adx[-1]):
                return min(adx[-1] / 100, 1.0)
            return 0.0
            
        except Exception as e:
            self.logger.error(f"❌ Trend strength calculation error: {e}")
            return 0.0
            
    def _identify_support_resistance_zones(self, data: pd.DataFrame) -> List[Tuple[float, float]]:
        """Identify support and resistance zones using local extremes"""
        try:
            high_prices = data['high'].values
            low_prices = data['low'].values
            
            # Find local maxima and minima
            high_peaks, _ = find_peaks(high_prices, distance=5)
            low_peaks, _ = find_peaks(-low_prices, distance=5)
            
            # Create zones around peaks
            zones = []
            
            # Resistance zones from highs
            for peak in high_peaks:
                price = high_prices[peak]
                zone_width = price * 0.01  # 1% zone
                zones.append((price - zone_width, price + zone_width))
                
            # Support zones from lows
            for peak in low_peaks:
                price = low_prices[peak]
                zone_width = price * 0.01  # 1% zone
                zones.append((price - zone_width, price + zone_width))
                
            # Remove overlapping zones and sort
            zones = sorted(list(set(zones)))
            
            return zones[-10:]  # Return top 10 zones
            
        except Exception as e:
            self.logger.error(f"❌ Support/resistance zone identification error: {e}")
            return []
            
    def calculate_volatility_bands(self, data: pd.DataFrame, 
                                 method: str = 'bollinger') -> Dict[str, float]:
        """
        Calculate volatility bands using different methods
        
        Args:
            data: OHLCV data
            method: 'bollinger', 'keltner', or 'donchian'
            
        Returns:
            Dictionary with upper, middle, and lower band values
        """
        try:
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            
            if method == 'bollinger':
                upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
                return {
                    'upper': upper[-1],
                    'middle': middle[-1],
                    'lower': lower[-1],
                    'width': upper[-1] - lower[-1]
                }
                
            elif method == 'keltner':
                # Keltner Channels
                ema = talib.EMA(close, timeperiod=20)
                atr = talib.ATR(high, low, close, timeperiod=20)
                
                upper = ema + (2 * atr)
                lower = ema - (2 * atr)
                
                return {
                    'upper': upper[-1],
                    'middle': ema[-1],
                    'lower': lower[-1],
                    'width': upper[-1] - lower[-1]
                }
                
            elif method == 'donchian':
                # Donchian Channels
                period = 20
                upper = talib.MAX(high, timeperiod=period)
                lower = talib.MIN(low, timeperiod=period)
                middle = (upper + lower) / 2
                
                return {
                    'upper': upper[-1],
                    'middle': middle[-1],
                    'lower': lower[-1],
                    'width': upper[-1] - lower[-1]
                }
                
            else:
                raise ValueError(f"Unknown volatility band method: {method}")
                
        except Exception as e:
            self.logger.error(f"❌ Volatility bands calculation error: {e}")
            return {}
            
    def calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics
        
        Args:
            returns: Series of returns
            
        Returns:
            Dictionary with risk metrics
        """
        try:
            # Basic statistics
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Value at Risk (VaR)
            var_95 = returns.quantile(0.05)
            var_99 = returns.quantile(0.01)
            
            # Conditional Value at Risk (CVaR)
            cvar_95 = returns[returns <= var_95].mean()
            cvar_99 = returns[returns <= var_99].mean()
            
            # Maximum Drawdown
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Skewness and Kurtosis
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
            
            # Downside deviation
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0
            
            # Sortino ratio
            sortino_ratio = (mean_return * 252) / (downside_deviation * np.sqrt(252)) if downside_deviation > 0 else 0
            
            return {
                'mean_return': mean_return,
                'volatility': std_return,
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'max_drawdown': max_drawdown,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'downside_deviation': downside_deviation,
                'sortino_ratio': sortino_ratio
            }
            
        except Exception as e:
            self.logger.error(f"❌ Risk metrics calculation error: {e}")
            return {}
            
    def monte_carlo_simulation(self, current_price: float, 
                             volatility: float, 
                             drift: float,
                             days: int = 30, 
                             simulations: int = 1000) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation for price prediction
        
        Args:
            current_price: Current asset price
            volatility: Historical volatility
            drift: Expected return
            days: Number of days to simulate
            simulations: Number of simulation runs
            
        Returns:
            Dictionary with simulation results
        """
        try:
            dt = 1/252  # Daily time step
            
            # Generate random price paths
            price_paths = []
            
            for _ in range(simulations):
                prices = [current_price]
                
                for _ in range(days):
                    random_shock = np.random.normal(0, 1)
                    price_change = drift * dt + volatility * np.sqrt(dt) * random_shock
                    new_price = prices[-1] * np.exp(price_change)
                    prices.append(new_price)
                    
                price_paths.append(prices)
                
            price_paths = np.array(price_paths)
            
            # Calculate statistics
            final_prices = price_paths[:, -1]
            
            results = {
                'current_price': current_price,
                'mean_final_price': np.mean(final_prices),
                'median_final_price': np.median(final_prices),
                'std_final_price': np.std(final_prices),
                'min_final_price': np.min(final_prices),
                'max_final_price': np.max(final_prices),
                'percentile_5': np.percentile(final_prices, 5),
                'percentile_95': np.percentile(final_prices, 95),
                'probability_profit': np.mean(final_prices > current_price),
                'expected_return': (np.mean(final_prices) - current_price) / current_price,
                'price_paths': price_paths
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Monte Carlo simulation error: {e}")
            return {}
            
    def calculate_option_greeks(self, spot_price: float, 
                              strike_price: float,
                              time_to_expiry: float, 
                              risk_free_rate: float,
                              volatility: float, 
                              option_type: str = 'call') -> Dict[str, float]:
        """
        Calculate option Greeks using Black-Scholes model
        
        Args:
            spot_price: Current underlying price
            strike_price: Option strike price
            time_to_expiry: Time to expiry in years
            risk_free_rate: Risk-free rate
            volatility: Implied volatility
            option_type: 'call' or 'put'
            
        Returns:
            Dictionary with option Greeks
        """
        try:
            from scipy.stats import norm
            
            # Calculate d1 and d2
            d1 = (np.log(spot_price / strike_price) + 
                  (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / \
                 (volatility * np.sqrt(time_to_expiry))
            
            d2 = d1 - volatility * np.sqrt(time_to_expiry)
            
            # Calculate Greeks
            if option_type.lower() == 'call':
                delta = norm.cdf(d1)
                theta = (-(spot_price * norm.pdf(d1) * volatility) / (2 * np.sqrt(time_to_expiry)) -
                        risk_free_rate * strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2))
            else:  # put
                delta = norm.cdf(d1) - 1
                theta = (-(spot_price * norm.pdf(d1) * volatility) / (2 * np.sqrt(time_to_expiry)) +
                        risk_free_rate * strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2))
                
            gamma = norm.pdf(d1) / (spot_price * volatility * np.sqrt(time_to_expiry))
            vega = spot_price * norm.pdf(d1) * np.sqrt(time_to_expiry)
            
            if option_type.lower() == 'call':
                rho = strike_price * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
            else:  # put
                rho = -strike_price * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2)
                
            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta / 365,  # Convert to daily theta
                'vega': vega / 100,    # Convert to vega per 1% vol change
                'rho': rho / 100       # Convert to rho per 1% rate change
            }
            
        except Exception as e:
            self.logger.error(f"❌ Option Greeks calculation error: {e}")
            return {}
            
    def get_trading_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive trading signals based on mathematical models
        
        Args:
            data: OHLCV data
            
        Returns:
            Dictionary with trading signals and confidence scores
        """
        try:
            # Calculate all indicators
            fib_levels = self.calculate_fibonacci_levels(data)
            tech_indicators = self.calculate_technical_indicators(data)
            statistical_analysis = self.calculate_statistical_analysis(data)
            
            if not all([fib_levels, tech_indicators, statistical_analysis]):
                return {'signal': 'NEUTRAL', 'confidence': 0.0}
                
            signals = []
            current_price = data['close'].iloc[-1]
            
            # Fibonacci signals
            if fib_levels.support_levels:
                closest_support = fib_levels.support_levels[0]
                if current_price <= closest_support * 1.02:  # Within 2% of support
                    signals.append(('BUY', 0.7, 'Fibonacci support'))
                    
            if fib_levels.resistance_levels:
                closest_resistance = fib_levels.resistance_levels[0]
                if current_price >= closest_resistance * 0.98:  # Within 2% of resistance
                    signals.append(('SELL', 0.7, 'Fibonacci resistance'))
                    
            # RSI signals
            if tech_indicators.rsi < 30:
                signals.append(('BUY', 0.8, 'RSI oversold'))
            elif tech_indicators.rsi > 70:
                signals.append(('SELL', 0.8, 'RSI overbought'))
                
            # MACD signals
            if (tech_indicators.macd['macd'] > tech_indicators.macd['signal'] and 
                tech_indicators.macd['histogram'] > 0):
                signals.append(('BUY', 0.6, 'MACD bullish'))
            elif (tech_indicators.macd['macd'] < tech_indicators.macd['signal'] and 
                  tech_indicators.macd['histogram'] < 0):
                signals.append(('SELL', 0.6, 'MACD bearish'))
                
            # Bollinger Bands signals
            bb_position = tech_indicators.bollinger_bands['position']
            if bb_position < 0.1:  # Near lower band
                signals.append(('BUY', 0.5, 'Bollinger oversold'))
            elif bb_position > 0.9:  # Near upper band
                signals.append(('SELL', 0.5, 'Bollinger overbought'))
                
            # Trend signals
            if (current_price > tech_indicators.sma_20 > tech_indicators.sma_50 and
                statistical_analysis.trend_strength > 0.5):
                signals.append(('BUY', 0.6, 'Strong uptrend'))
            elif (current_price < tech_indicators.sma_20 < tech_indicators.sma_50 and
                  statistical_analysis.trend_strength > 0.5):
                signals.append(('SELL', 0.6, 'Strong downtrend'))
                
            # Aggregate signals
            if not signals:
                return {'signal': 'NEUTRAL', 'confidence': 0.0, 'reasons': []}
                
            buy_signals = [s for s in signals if s[0] == 'BUY']
            sell_signals = [s for s in signals if s[0] == 'SELL']
            
            buy_confidence = sum(s[1] for s in buy_signals) / len(buy_signals) if buy_signals else 0
            sell_confidence = sum(s[1] for s in sell_signals) / len(sell_signals) if sell_signals else 0
            
            if buy_confidence > sell_confidence and buy_confidence > 0.5:
                return {
                    'signal': 'BUY',
                    'confidence': buy_confidence,
                    'reasons': [s[2] for s in buy_signals]
                }
            elif sell_confidence > buy_confidence and sell_confidence > 0.5:
                return {
                    'signal': 'SELL',
                    'confidence': sell_confidence,
                    'reasons': [s[2] for s in sell_signals]
                }
            else:
                return {
                    'signal': 'NEUTRAL',
                    'confidence': 0.0,
                    'reasons': ['Mixed signals']
                }
                
        except Exception as e:
            self.logger.error(f"❌ Trading signals calculation error: {e}")
            return {'signal': 'NEUTRAL', 'confidence': 0.0}
