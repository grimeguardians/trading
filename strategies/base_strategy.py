"""
Base Strategy class for all trading strategies
Provides common functionality and interface
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

from config import Config


@dataclass
class TradingSignal:
    """Trading signal data structure"""
    symbol: str
    signal_type: str  # BUY, SELL, HOLD
    confidence: float
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timeframe: str = "1h"
    reasoning: str = ""
    metadata: Dict[str, Any] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies
    Provides common functionality and interface
    """
    
    def __init__(self, config: Config, strategy_name: str):
        self.config = config
        self.strategy_name = strategy_name
        self.logger = logging.getLogger(f"Strategy.{strategy_name}")
        
        # Strategy configuration
        self.strategy_config = config.STRATEGIES.get(strategy_name, {})
        self.enabled = self.strategy_config.get("enabled", False)
        self.timeframe = self.strategy_config.get("timeframe", "1h")
        self.max_positions = self.strategy_config.get("max_positions", 5)
        self.min_profit_target = self.strategy_config.get("min_profit_target", 0.05)
        
        # Strategy parameters (can be optimized)
        self.parameters = self._get_default_parameters()
        
        # Strategy state
        self.signals = []
        self.positions = {}
        self.performance_metrics = {
            "total_signals": 0,
            "successful_signals": 0,
            "accuracy": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0
        }
        
        # Data storage
        self.market_data = {}
        self.indicators = {}
        
        # Signal generation settings
        self.signal_threshold = 0.6  # Minimum confidence threshold
        self.max_signals_per_hour = 10
        self.signal_cooldown = 300  # 5 minutes between signals for same symbol
        
        # Last signal timestamps
        self.last_signal_time = {}
        
        self.logger.info(f"📊 Strategy {strategy_name} initialized")
    
    @abstractmethod
    def _get_default_parameters(self) -> Dict[str, Any]:
        """Get default strategy parameters"""
        pass
    
    @abstractmethod
    async def analyze_market(self, symbol: str, market_data: pd.DataFrame) -> Optional[TradingSignal]:
        """Analyze market data and generate signals"""
        pass
    
    @abstractmethod
    async def should_exit_position(self, symbol: str, position: Dict, current_price: float) -> bool:
        """Determine if position should be exited"""
        pass
    
    async def get_signals(self, symbols: List[str] = None) -> List[TradingSignal]:
        """Get trading signals for symbols"""
        try:
            if not self.enabled:
                return []
            
            if symbols is None:
                symbols = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL"]  # Default symbols
            
            signals = []
            
            for symbol in symbols:
                try:
                    # Check signal cooldown
                    if not self._can_generate_signal(symbol):
                        continue
                    
                    # Get market data
                    market_data = await self._get_market_data(symbol)
                    
                    if market_data is None or market_data.empty:
                        continue
                    
                    # Analyze market and generate signal
                    signal = await self.analyze_market(symbol, market_data)
                    
                    if signal and signal.confidence >= self.signal_threshold:
                        signals.append(signal)
                        self.last_signal_time[symbol] = datetime.utcnow()
                        self.performance_metrics["total_signals"] += 1
                        
                        self.logger.info(f"📈 Signal generated: {signal.symbol} {signal.signal_type} (confidence: {signal.confidence:.2%})")
                
                except Exception as e:
                    self.logger.error(f"❌ Signal generation error for {symbol}: {e}")
                    continue
            
            # Store signals
            self.signals.extend(signals)
            
            # Keep signal history manageable
            if len(self.signals) > 100:
                self.signals = self.signals[-50:]
            
            return signals
            
        except Exception as e:
            self.logger.error(f"❌ Get signals error: {e}")
            return []
    
    def _can_generate_signal(self, symbol: str) -> bool:
        """Check if signal can be generated for symbol"""
        try:
            # Check if enough time has passed since last signal
            if symbol in self.last_signal_time:
                time_diff = (datetime.utcnow() - self.last_signal_time[symbol]).total_seconds()
                if time_diff < self.signal_cooldown:
                    return False
            
            # Check hourly signal limit
            current_hour = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
            signals_this_hour = len([
                s for s in self.signals 
                if s.timestamp >= current_hour and s.symbol == symbol
            ])
            
            if signals_this_hour >= self.max_signals_per_hour:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Signal check error: {e}")
            return False
    
    async def _get_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get market data for symbol"""
        try:
            # This would integrate with market data providers
            # For now, generate mock data
            
            dates = pd.date_range(
                start=datetime.utcnow() - timedelta(days=30),
                end=datetime.utcnow(),
                freq='H'
            )
            
            # Generate realistic price data
            np.random.seed(hash(symbol) % 2**32)  # Consistent seed for symbol
            base_price = 100.0
            prices = []
            
            for i, date in enumerate(dates):
                # Random walk with slight upward bias
                change = np.random.normal(0.001, 0.02)
                base_price *= (1 + change)
                prices.append(base_price)
            
            # Create OHLCV data
            data = []
            for i, (date, price) in enumerate(zip(dates, prices)):
                high = price * (1 + abs(np.random.normal(0, 0.01)))
                low = price * (1 - abs(np.random.normal(0, 0.01)))
                open_price = prices[i-1] if i > 0 else price
                close_price = price
                volume = np.random.randint(10000, 100000)
                
                data.append({
                    'timestamp': date,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close_price,
                    'volume': volume
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            
            # Store in cache
            self.market_data[symbol] = df
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Market data error for {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators"""
        try:
            indicators = {}
            
            # Moving averages
            indicators['sma_20'] = df['close'].rolling(window=20).mean()
            indicators['sma_50'] = df['close'].rolling(window=50).mean()
            indicators['ema_12'] = df['close'].ewm(span=12).mean()
            indicators['ema_26'] = df['close'].ewm(span=26).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
            indicators['macd_signal'] = indicators['macd'].ewm(span=9).mean()
            indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            indicators['bb_middle'] = df['close'].rolling(window=bb_period).mean()
            bb_std_dev = df['close'].rolling(window=bb_period).std()
            indicators['bb_upper'] = indicators['bb_middle'] + (bb_std_dev * bb_std)
            indicators['bb_lower'] = indicators['bb_middle'] - (bb_std_dev * bb_std)
            
            # Stochastic
            low_14 = df['low'].rolling(window=14).min()
            high_14 = df['high'].rolling(window=14).max()
            indicators['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
            indicators['stoch_d'] = indicators['stoch_k'].rolling(window=3).mean()
            
            # Volume indicators
            indicators['volume_sma'] = df['volume'].rolling(window=20).mean()
            indicators['volume_ratio'] = df['volume'] / indicators['volume_sma']
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"❌ Technical indicators calculation error: {e}")
            return {}
    
    def calculate_support_resistance(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """Calculate support and resistance levels"""
        try:
            # Find local minima and maxima
            from scipy.signal import argrelextrema
            
            # Get recent data (last 100 periods)
            recent_data = df.tail(100)
            
            # Find support levels (local minima)
            support_indices = argrelextrema(recent_data['low'].values, np.less, order=5)[0]
            support_levels = recent_data.iloc[support_indices]['low'].tolist()
            
            # Find resistance levels (local maxima)  
            resistance_indices = argrelextrema(recent_data['high'].values, np.greater, order=5)[0]
            resistance_levels = recent_data.iloc[resistance_indices]['high'].tolist()
            
            return {
                'support': sorted(support_levels)[-3:],  # Last 3 support levels
                'resistance': sorted(resistance_levels, reverse=True)[:3]  # Last 3 resistance levels
            }
            
        except Exception as e:
            self.logger.error(f"❌ Support/resistance calculation error: {e}")
            return {'support': [], 'resistance': []}
    
    def calculate_volatility(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate volatility metrics"""
        try:
            # Calculate returns
            returns = df['close'].pct_change().dropna()
            
            # Historical volatility (annualized)
            volatility = returns.std() * np.sqrt(252)
            
            # Average True Range (ATR)
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean()
            
            return {
                'volatility': volatility,
                'atr': atr.iloc[-1] if not atr.empty else 0.0,
                'returns_std': returns.std()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Volatility calculation error: {e}")
            return {'volatility': 0.0, 'atr': 0.0, 'returns_std': 0.0}
    
    def calculate_fibonacci_levels(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels"""
        try:
            # Get recent high and low
            recent_data = df.tail(100)
            high = recent_data['high'].max()
            low = recent_data['low'].min()
            
            # Calculate Fibonacci levels
            diff = high - low
            fib_levels = {}
            
            # Retracement levels
            fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
            
            for ratio in fib_ratios:
                fib_levels[f'fib_{ratio}'] = high - (diff * ratio)
            
            # Extension levels
            fib_levels['fib_1.272'] = high + (diff * 0.272)
            fib_levels['fib_1.618'] = high + (diff * 0.618)
            
            return fib_levels
            
        except Exception as e:
            self.logger.error(f"❌ Fibonacci calculation error: {e}")
            return {}
    
    def calculate_risk_metrics(self, signal: TradingSignal, current_price: float) -> Dict[str, float]:
        """Calculate risk metrics for signal"""
        try:
            risk_metrics = {}
            
            # Calculate position size based on risk
            account_balance = 100000  # Mock account balance
            risk_per_trade = account_balance * self.config.RISK_PER_TRADE
            
            # Calculate stop loss distance
            if signal.stop_loss:
                stop_loss_distance = abs(current_price - signal.stop_loss)
                position_size = risk_per_trade / stop_loss_distance
                risk_metrics['position_size'] = position_size
                risk_metrics['risk_amount'] = risk_per_trade
                risk_metrics['stop_loss_distance'] = stop_loss_distance
            
            # Calculate potential profit
            if signal.take_profit:
                profit_distance = abs(signal.take_profit - current_price)
                potential_profit = profit_distance * risk_metrics.get('position_size', 0)
                risk_metrics['potential_profit'] = potential_profit
                risk_metrics['risk_reward_ratio'] = profit_distance / risk_metrics.get('stop_loss_distance', 1)
            
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Risk metrics calculation error: {e}")
            return {}
    
    def validate_signal(self, signal: TradingSignal) -> bool:
        """Validate signal before generation"""
        try:
            # Check confidence threshold
            if signal.confidence < self.signal_threshold:
                return False
            
            # Check if signal has required fields
            if not signal.symbol or not signal.signal_type:
                return False
            
            # Check if signal type is valid
            if signal.signal_type not in ['BUY', 'SELL', 'HOLD']:
                return False
            
            # Check risk/reward ratio
            if signal.stop_loss and signal.take_profit:
                current_price = signal.price_target or 100.0  # Mock current price
                
                if signal.signal_type == 'BUY':
                    risk = current_price - signal.stop_loss
                    reward = signal.take_profit - current_price
                else:  # SELL
                    risk = signal.stop_loss - current_price
                    reward = current_price - signal.take_profit
                
                if risk <= 0 or reward <= 0:
                    return False
                
                risk_reward_ratio = reward / risk
                if risk_reward_ratio < 1.5:  # Minimum 1.5:1 ratio
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Signal validation error: {e}")
            return False
    
    async def update_parameters(self, new_parameters: Dict[str, Any]):
        """Update strategy parameters"""
        try:
            self.parameters.update(new_parameters)
            self.logger.info(f"📊 Strategy parameters updated: {new_parameters}")
            
        except Exception as e:
            self.logger.error(f"❌ Parameter update error: {e}")
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """Get current strategy parameters"""
        return self.parameters.copy()
    
    def get_optimization_parameters(self) -> Dict[str, Tuple[float, float, float]]:
        """Get parameters for optimization (min, max, step)"""
        # Override in subclasses
        return {}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get strategy performance metrics"""
        return self.performance_metrics.copy()
    
    def update_performance_metrics(self, trade_result: Dict[str, Any]):
        """Update performance metrics based on trade result"""
        try:
            if trade_result.get('success', False):
                self.performance_metrics['successful_signals'] += 1
            
            # Calculate accuracy
            if self.performance_metrics['total_signals'] > 0:
                self.performance_metrics['accuracy'] = (
                    self.performance_metrics['successful_signals'] / 
                    self.performance_metrics['total_signals']
                )
            
            # Update other metrics (simplified)
            profit = trade_result.get('profit', 0.0)
            if profit > 0:
                self.performance_metrics['profit_factor'] = max(
                    self.performance_metrics['profit_factor'], 
                    profit / abs(trade_result.get('loss', 1.0))
                )
                
        except Exception as e:
            self.logger.error(f"❌ Performance metrics update error: {e}")
    
    def reset_performance_metrics(self):
        """Reset performance metrics"""
        self.performance_metrics = {
            "total_signals": 0,
            "successful_signals": 0,
            "accuracy": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0
        }
    
    def get_strategy_status(self) -> Dict[str, Any]:
        """Get current strategy status"""
        return {
            "name": self.strategy_name,
            "enabled": self.enabled,
            "timeframe": self.timeframe,
            "max_positions": self.max_positions,
            "signals_generated": len(self.signals),
            "performance": self.performance_metrics,
            "parameters": self.parameters,
            "last_updated": datetime.utcnow().isoformat()
        }
    
    def __str__(self):
        return f"<Strategy {self.strategy_name}>"
    
    def __repr__(self):
        return f"<Strategy {self.strategy_name} enabled={self.enabled}>"
