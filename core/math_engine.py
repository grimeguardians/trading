"""
Mathematical analysis engine for trading calculations
Includes Fibonacci, technical indicators, and statistical models
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize
import ta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)

@dataclass
class FibonacciLevels:
    """Fibonacci retracement and extension levels"""
    swing_high: float
    swing_low: float
    retracement_levels: Dict[float, float]
    extension_levels: Dict[float, float]
    current_price: float
    direction: str  # bullish or bearish
    strength: float

@dataclass
class SupportResistance:
    """Support and resistance levels"""
    support_levels: List[float]
    resistance_levels: List[float]
    strength_scores: Dict[float, float]
    pivot_points: Dict[str, float]

@dataclass
class VolatilityModel:
    """Volatility analysis results"""
    current_volatility: float
    historical_volatility: float
    volatility_percentile: float
    volatility_regime: str  # low, normal, high
    garch_forecast: Optional[float] = None

class MathEngine:
    """Advanced mathematical analysis engine for trading"""
    
    def __init__(self):
        self.fibonacci_ratios = {
            'retracement': [0.236, 0.382, 0.5, 0.618, 0.786],
            'extension': [1.272, 1.414, 1.618, 2.618, 3.618]
        }
        
        # Technical analysis parameters
        self.ta_params = {
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
        
        # Statistical models
        self.scaler = StandardScaler()
        self.ml_models = {}
        
        logger.info("MathEngine initialized")
    
    async def calculate_fibonacci_levels(self, data: pd.DataFrame, config: Dict[str, Any]) -> FibonacciLevels:
        """Calculate Fibonacci retracement and extension levels"""
        try:
            if len(data) < 10:
                logger.warning("Insufficient data for Fibonacci calculation")
                return None
            
            # Find significant swings
            swing_high, swing_low = await self._find_significant_swings(data)
            
            if swing_high is None or swing_low is None:
                return None
            
            # Calculate retracement levels
            retracement_levels = {}
            for ratio in self.fibonacci_ratios['retracement']:
                level = swing_high - (swing_high - swing_low) * ratio
                retracement_levels[ratio] = level
            
            # Calculate extension levels
            extension_levels = {}
            swing_range = swing_high - swing_low
            for ratio in self.fibonacci_ratios['extension']:
                if swing_high > swing_low:  # Uptrend
                    level = swing_high + swing_range * (ratio - 1)
                else:  # Downtrend
                    level = swing_low - swing_range * (ratio - 1)
                extension_levels[ratio] = level
            
            # Determine direction and strength
            current_price = data['close'].iloc[-1]
            direction = "bullish" if current_price > swing_low else "bearish"
            strength = await self._calculate_fibonacci_strength(data, retracement_levels, extension_levels)
            
            return FibonacciLevels(
                swing_high=swing_high,
                swing_low=swing_low,
                retracement_levels=retracement_levels,
                extension_levels=extension_levels,
                current_price=current_price,
                direction=direction,
                strength=strength
            )
            
        except Exception as e:
            logger.error(f"Error calculating Fibonacci levels: {e}")
            return None
    
    async def _find_significant_swings(self, data: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
        """Find significant swing highs and lows"""
        try:
            # Use a simple approach to find recent significant swings
            recent_data = data.tail(50)  # Look at last 50 periods
            
            # Find local maxima and minima
            highs = recent_data['high'].rolling(window=5, center=True).max()
            lows = recent_data['low'].rolling(window=5, center=True).min()
            
            # Find the most significant swing high and low
            swing_high = highs.max()
            swing_low = lows.min()
            
            return swing_high, swing_low
            
        except Exception as e:
            logger.error(f"Error finding significant swings: {e}")
            return None, None
    
    async def _calculate_fibonacci_strength(self, data: pd.DataFrame, retracement_levels: Dict[float, float], extension_levels: Dict[float, float]) -> float:
        """Calculate the strength of Fibonacci levels"""
        try:
            # Simple strength calculation based on how price respects levels
            strength = 0.0
            recent_prices = data['close'].tail(20)
            
            # Check if price has respected retracement levels
            for level_price in retracement_levels.values():
                respect_count = 0
                for price in recent_prices:
                    if abs(price - level_price) / level_price < 0.005:  # Within 0.5%
                        respect_count += 1
                
                if respect_count > 0:
                    strength += respect_count / len(recent_prices)
            
            return min(1.0, strength)
            
        except Exception as e:
            logger.error(f"Error calculating Fibonacci strength: {e}")
            return 0.0
    
    async def find_support_resistance(self, data: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Find support and resistance levels"""
        try:
            if len(data) < 20:
                return [], []
            
            # Use pivot points and level clustering
            pivot_points = await self._calculate_pivot_points(data)
            
            # Find support levels (areas where price bounced up)
            support_levels = []
            resistance_levels = []
            
            # Look for areas where price frequently tested but didn't break
            price_levels = pd.concat([data['high'], data['low'], data['close']])
            
            # Create clusters of similar price levels
            clusters = await self._cluster_price_levels(price_levels)
            
            current_price = data['close'].iloc[-1]
            
            for cluster_center in clusters:
                if cluster_center < current_price:
                    support_levels.append(cluster_center)
                else:
                    resistance_levels.append(cluster_center)
            
            # Sort and return top levels
            support_levels = sorted(support_levels, reverse=True)[:5]
            resistance_levels = sorted(resistance_levels)[:5]
            
            return support_levels, resistance_levels
            
        except Exception as e:
            logger.error(f"Error finding support/resistance: {e}")
            return [], []
    
    async def _calculate_pivot_points(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate pivot points"""
        try:
            last_day = data.tail(1)
            high = last_day['high'].iloc[0]
            low = last_day['low'].iloc[0]
            close = last_day['close'].iloc[0]
            
            pivot = (high + low + close) / 3
            
            return {
                'pivot': pivot,
                'r1': 2 * pivot - low,
                'r2': pivot + (high - low),
                's1': 2 * pivot - high,
                's2': pivot - (high - low)
            }
            
        except Exception as e:
            logger.error(f"Error calculating pivot points: {e}")
            return {}
    
    async def _cluster_price_levels(self, price_levels: pd.Series) -> List[float]:
        """Cluster similar price levels"""
        try:
            # Simple clustering approach
            sorted_prices = price_levels.sort_values()
            clusters = []
            
            tolerance = sorted_prices.std() * 0.1  # 10% of standard deviation
            
            i = 0
            while i < len(sorted_prices):
                cluster_prices = [sorted_prices.iloc[i]]
                j = i + 1
                
                while j < len(sorted_prices) and abs(sorted_prices.iloc[j] - sorted_prices.iloc[i]) <= tolerance:
                    cluster_prices.append(sorted_prices.iloc[j])
                    j += 1
                
                if len(cluster_prices) >= 3:  # Minimum cluster size
                    clusters.append(np.mean(cluster_prices))
                
                i = j
            
            return clusters
            
        except Exception as e:
            logger.error(f"Error clustering price levels: {e}")
            return []
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        try:
            return ta.momentum.RSIIndicator(prices, window=period).rsi().iloc[-1]
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return 50.0
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """Calculate MACD indicator"""
        try:
            macd_indicator = ta.trend.MACD(prices, window_slow=slow, window_fast=fast, window_sign=signal)
            macd_line = macd_indicator.macd().iloc[-1]
            signal_line = macd_indicator.macd_signal().iloc[-1]
            histogram = macd_indicator.macd_diff().iloc[-1]
            
            return macd_line, signal_line, histogram
            
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return 0.0, 0.0, 0.0
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands"""
        try:
            bb_indicator = ta.volatility.BollingerBands(prices, window=period, window_dev=std_dev)
            upper = bb_indicator.bollinger_hband().iloc[-1]
            middle = bb_indicator.bollinger_mavg().iloc[-1]
            lower = bb_indicator.bollinger_lband().iloc[-1]
            
            return upper, middle, lower
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return 0.0, 0.0, 0.0
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[float, float]:
        """Calculate Stochastic Oscillator"""
        try:
            stoch_indicator = ta.momentum.StochasticOscillator(high, low, close, window=k_period, smooth_window=d_period)
            stoch_k = stoch_indicator.stoch().iloc[-1]
            stoch_d = stoch_indicator.stoch_signal().iloc[-1]
            
            return stoch_k, stoch_d
            
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {e}")
            return 50.0, 50.0
    
    def calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """Calculate Williams %R"""
        try:
            return ta.momentum.WilliamsRIndicator(high, low, close, lbp=period).williams_r().iloc[-1]
        except Exception as e:
            logger.error(f"Error calculating Williams %R: {e}")
            return -50.0
    
    async def calculate_volatility_model(self, data: pd.DataFrame) -> VolatilityModel:
        """Calculate comprehensive volatility model"""
        try:
            returns = data['close'].pct_change().dropna()
            
            # Current volatility (annualized)
            current_volatility = returns.std() * np.sqrt(252)
            
            # Historical volatility (rolling 30-day)
            historical_volatility = returns.rolling(window=30).std().mean() * np.sqrt(252)
            
            # Volatility percentile
            volatility_series = returns.rolling(window=30).std() * np.sqrt(252)
            volatility_percentile = stats.percentileofscore(volatility_series.dropna(), current_volatility) / 100
            
            # Volatility regime
            if volatility_percentile < 0.25:
                volatility_regime = "low"
            elif volatility_percentile > 0.75:
                volatility_regime = "high"
            else:
                volatility_regime = "normal"
            
            # GARCH forecast (simplified)
            garch_forecast = await self._simple_garch_forecast(returns)
            
            return VolatilityModel(
                current_volatility=current_volatility,
                historical_volatility=historical_volatility,
                volatility_percentile=volatility_percentile,
                volatility_regime=volatility_regime,
                garch_forecast=garch_forecast
            )
            
        except Exception as e:
            logger.error(f"Error calculating volatility model: {e}")
            return VolatilityModel(
                current_volatility=0.2,
                historical_volatility=0.2,
                volatility_percentile=0.5,
                volatility_regime="normal"
            )
    
    async def _simple_garch_forecast(self, returns: pd.Series) -> Optional[float]:
        """Simple GARCH(1,1) forecast"""
        try:
            # Simplified GARCH implementation
            # In practice, you'd use a proper GARCH library like arch
            
            if len(returns) < 50:
                return None
            
            # Calculate simple volatility forecast
            recent_returns = returns.tail(30)
            forecast = recent_returns.std() * np.sqrt(252)
            
            return forecast
            
        except Exception as e:
            logger.error(f"Error in GARCH forecast: {e}")
            return None
    
    async def calculate_correlation_matrix(self, prices_data: Dict[str, pd.Series]) -> pd.DataFrame:
        """Calculate correlation matrix for multiple assets"""
        try:
            # Create DataFrame from price series
            df = pd.DataFrame(prices_data)
            
            # Calculate returns
            returns = df.pct_change().dropna()
            
            # Calculate correlation matrix
            correlation_matrix = returns.corr()
            
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return pd.DataFrame()
    
    async def calculate_portfolio_metrics(self, returns: pd.Series, benchmark_returns: pd.Series = None) -> Dict[str, float]:
        """Calculate portfolio performance metrics"""
        try:
            metrics = {}
            
            # Basic metrics
            metrics['total_return'] = (1 + returns).prod() - 1
            metrics['annualized_return'] = (1 + returns.mean()) ** 252 - 1
            metrics['volatility'] = returns.std() * np.sqrt(252)
            
            # Sharpe ratio (assuming risk-free rate of 0)
            metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['volatility'] if metrics['volatility'] > 0 else 0
            
            # Maximum drawdown
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            metrics['max_drawdown'] = drawdown.min()
            
            # Calmar ratio
            metrics['calmar_ratio'] = abs(metrics['annualized_return'] / metrics['max_drawdown']) if metrics['max_drawdown'] < 0 else 0
            
            # Win rate
            metrics['win_rate'] = (returns > 0).mean()
            
            # Beta (if benchmark provided)
            if benchmark_returns is not None:
                covariance = np.cov(returns, benchmark_returns)[0][1]
                benchmark_variance = np.var(benchmark_returns)
                metrics['beta'] = covariance / benchmark_variance if benchmark_variance > 0 else 0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return {}
    
    async def calculate_position_sizing(self, account_value: float, risk_per_trade: float, entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on risk management"""
        try:
            # Calculate risk amount
            risk_amount = account_value * risk_per_trade
            
            # Calculate risk per share
            risk_per_share = abs(entry_price - stop_loss)
            
            # Calculate position size
            if risk_per_share > 0:
                position_size = risk_amount / risk_per_share
            else:
                position_size = 0
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position sizing: {e}")
            return 0.0
    
    async def calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate Kelly Criterion for optimal position sizing"""
        try:
            if avg_loss == 0:
                return 0.0
            
            # Kelly formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - win_rate
            
            kelly_fraction = (b * p - q) / b
            
            # Cap at 25% for safety
            return min(0.25, max(0.0, kelly_fraction))
            
        except Exception as e:
            logger.error(f"Error calculating Kelly Criterion: {e}")
            return 0.0
    
    async def monte_carlo_simulation(self, returns: pd.Series, initial_capital: float, num_simulations: int = 1000, time_horizon: int = 252) -> Dict[str, Any]:
        """Run Monte Carlo simulation for portfolio projections"""
        try:
            # Calculate return statistics
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Run simulations
            final_values = []
            
            for _ in range(num_simulations):
                # Generate random returns
                random_returns = np.random.normal(mean_return, std_return, time_horizon)
                
                # Calculate final portfolio value
                final_value = initial_capital * (1 + random_returns).prod()
                final_values.append(final_value)
            
            final_values = np.array(final_values)
            
            # Calculate statistics
            results = {
                'mean_final_value': final_values.mean(),
                'median_final_value': np.median(final_values),
                'std_final_value': final_values.std(),
                'percentile_5': np.percentile(final_values, 5),
                'percentile_95': np.percentile(final_values, 95),
                'probability_of_loss': (final_values < initial_capital).mean(),
                'max_value': final_values.max(),
                'min_value': final_values.min()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation: {e}")
            return {}
    
    async def calculate_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk (VaR)"""
        try:
            return np.percentile(returns, (1 - confidence_level) * 100)
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return 0.0
    
    async def calculate_expected_shortfall(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        try:
            var = await self.calculate_var(returns, confidence_level)
            return returns[returns <= var].mean()
            
        except Exception as e:
            logger.error(f"Error calculating Expected Shortfall: {e}")
            return 0.0
    
    async def optimize_portfolio(self, expected_returns: pd.Series, covariance_matrix: pd.DataFrame, risk_aversion: float = 1.0) -> Dict[str, float]:
        """Optimize portfolio weights using mean-variance optimization"""
        try:
            n_assets = len(expected_returns)
            
            # Objective function (minimize negative utility)
            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
                utility = portfolio_return - 0.5 * risk_aversion * portfolio_variance
                return -utility
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
            ]
            
            # Bounds (no short selling)
            bounds = [(0, 1) for _ in range(n_assets)]
            
            # Initial guess
            initial_guess = np.array([1/n_assets] * n_assets)
            
            # Optimize
            result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                return dict(zip(expected_returns.index, result.x))
            else:
                logger.warning("Portfolio optimization failed")
                return dict(zip(expected_returns.index, initial_guess))
                
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {e}")
            return {}
    
    async def calculate_harmonic_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect harmonic patterns (Gartley, Butterfly, etc.)"""
        try:
            patterns = []
            
            if len(data) < 10:
                return patterns
            
            # Simplified harmonic pattern detection
            # In practice, this would be much more sophisticated
            
            highs = data['high'].values
            lows = data['low'].values
            
            # Look for ABCD pattern (simplified)
            for i in range(len(highs) - 4):
                pattern = await self._detect_abcd_pattern(highs[i:i+5], lows[i:i+5])
                if pattern:
                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error calculating harmonic patterns: {e}")
            return []
    
    async def _detect_abcd_pattern(self, highs: np.ndarray, lows: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect ABCD pattern"""
        try:
            # Simplified ABCD pattern detection
            # This is a basic implementation - real harmonic pattern detection is much more complex
            
            if len(highs) < 4 or len(lows) < 4:
                return None
            
            # Basic pattern recognition logic would go here
            # For now, return None as this requires sophisticated implementation
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting ABCD pattern: {e}")
            return None
    
    async def calculate_market_regime(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Determine current market regime"""
        try:
            if len(data) < 50:
                return {"regime": "unknown", "confidence": 0.0}
            
            # Calculate various indicators
            returns = data['close'].pct_change().dropna()
            
            # Trend indicators
            sma_20 = data['close'].rolling(window=20).mean()
            sma_50 = data['close'].rolling(window=50).mean()
            
            # Volatility indicators
            volatility = returns.rolling(window=20).std()
            
            # Current values
            current_price = data['close'].iloc[-1]
            current_sma_20 = sma_20.iloc[-1]
            current_sma_50 = sma_50.iloc[-1]
            current_volatility = volatility.iloc[-1]
            
            # Determine regime
            if current_price > current_sma_20 > current_sma_50:
                if current_volatility < volatility.quantile(0.33):
                    regime = "bull_low_vol"
                elif current_volatility > volatility.quantile(0.67):
                    regime = "bull_high_vol"
                else:
                    regime = "bull_normal"
            elif current_price < current_sma_20 < current_sma_50:
                if current_volatility < volatility.quantile(0.33):
                    regime = "bear_low_vol"
                elif current_volatility > volatility.quantile(0.67):
                    regime = "bear_high_vol"
                else:
                    regime = "bear_normal"
            else:
                regime = "sideways"
            
            # Calculate confidence based on how clear the signals are
            trend_strength = abs(current_price - current_sma_50) / current_sma_50
            confidence = min(1.0, trend_strength * 10)  # Scale to 0-1
            
            return {
                "regime": regime,
                "confidence": confidence,
                "trend_strength": trend_strength,
                "volatility_percentile": stats.percentileofscore(volatility.dropna(), current_volatility) / 100
            }
            
        except Exception as e:
            logger.error(f"Error calculating market regime: {e}")
            return {"regime": "unknown", "confidence": 0.0}
