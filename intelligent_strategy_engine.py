#!/usr/bin/env python3
"""
Intelligent Strategy Engine - Literature-Driven Trading
Implements strategies from uploaded trading books with dynamic adaptation
"""

import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from config import config

try:
    from knowledge_engine import DigitalBrain
    BRAIN_AVAILABLE = True
except ImportError:
    BRAIN_AVAILABLE = False
    print("⚠️ Digital Brain not available - using fallback strategies")

@dataclass 
class TradingSignal:
    """Literature-based trading signal"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float  # 0.0 to 1.0
    strategy: str
    reasoning: List[str]
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None
    literature_source: Optional[str] = None

class IntelligentStrategyEngine:
    """Literature-driven trading strategy engine"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.brain = self._initialize_brain()
        self.active_strategies = self._load_literature_strategies()
        self.position_tracking = {}
        
    def _initialize_brain(self) -> Optional[object]:
        """Initialize Digital Brain with trading literature"""
        if not BRAIN_AVAILABLE:
            return None
            
        try:
            brain = DigitalBrain()
            brain.load_state()
            self.logger.info("✅ Digital Brain initialized with trading literature")
            return brain
        except Exception as e:
            self.logger.warning(f"Digital Brain initialization failed: {e}")
            return None
    
    def _load_literature_strategies(self) -> Dict[str, Dict]:
        """Load trading strategies from literature knowledge"""
        strategies = {
            'brooks_price_action': {
                'description': 'Al Brooks Price Action Trading',
                'focus': 'price_action_patterns',
                'stop_loss_method': 'dynamic_atr_based',
                'entry_signals': ['breakout_pullback', 'trend_channel', 'reversal_bar'],
                'literature_source': 'Al-Brooks-Trading-Price-Action-Trends'
            },
            'fibonacci_retracement': {
                'description': 'Fibonacci Trading Strategies',
                'focus': 'fibonacci_levels',
                'stop_loss_method': 'fibonacci_based',
                'entry_signals': ['61.8_retracement', '38.2_support', 'extension_targets'],
                'literature_source': 'Carolyn_Borden_Fibonacci_Trading'
            },
            'naked_forex': {
                'description': 'Naked Forex Price Action',
                'focus': 'support_resistance',
                'stop_loss_method': 'swing_points',
                'entry_signals': ['support_bounce', 'resistance_break', 'trend_continuation'],
                'literature_source': 'Naked Forex'
            },
            'harmonic_patterns': {
                'description': 'Harmonic Trading Patterns',
                'focus': 'harmonic_geometry',
                'stop_loss_method': 'pattern_invalidation',
                'entry_signals': ['gartley_pattern', 'butterfly_pattern', 'bat_pattern'],
                'literature_source': 'TheHarmonicTrader'
            }
        }
        return strategies
    
    def query_literature_knowledge(self, query: str, symbol: str = None) -> List[str]:
        """Query Digital Brain for trading knowledge"""
        if not self.brain:
            return self._fallback_knowledge(query)
            
        try:
            # Query the brain for relevant trading knowledge
            results = self.brain.query_knowledge(query, context={'symbol': symbol})
            
            # Extract actionable insights
            insights = []
            for result in results[:3]:  # Top 3 most relevant
                if hasattr(result, 'content'):
                    insights.append(result.content)
                    
            return insights if insights else self._fallback_knowledge(query)
            
        except Exception as e:
            self.logger.warning(f"Brain query failed: {e}")
            return self._fallback_knowledge(query)
    
    def _fallback_knowledge(self, query: str) -> List[str]:
        """Fallback trading knowledge when brain unavailable"""
        fallback_rules = {
            'stop_loss': [
                "Use 2% portfolio risk per trade maximum",
                "Place stop loss below recent swing low for long positions",
                "Use ATR-based stops: 2x ATR below entry for trending markets"
            ],
            'moving_stops': [
                "Trail stop loss as position moves in your favor",
                "Use previous swing highs/lows as new stop levels",
                "Move stop to breakeven after 1:1 risk/reward achieved"
            ],
            'entry_signals': [
                "Enter on pullbacks in strong trends",
                "Buy support, sell resistance in ranging markets",
                "Confirm with volume and momentum indicators"
            ],
            'position_sizing': [
                "Risk no more than 2% of portfolio per trade",
                "Adjust size based on volatility (ATR)",
                "Larger positions in higher probability setups"
            ]
        }
        
        for key, rules in fallback_rules.items():
            if key in query.lower():
                return rules
                
        return ["Apply sound risk management and follow trend direction"]
    
    def calculate_dynamic_stop_loss(self, symbol: str, entry_price: float, 
                                  direction: str, market_data: Dict) -> float:
        """Calculate stop loss using literature-based methods"""
        
        # Query brain for stop loss guidance
        stop_knowledge = self.query_literature_knowledge(
            f"stop loss placement {direction} position {symbol}", symbol
        )
        
        # Get ATR for volatility-based stops
        atr = market_data.get('atr', entry_price * 0.02)  # 2% fallback
        
        # Brooks-style ATR stop
        if direction.upper() == 'BUY':
            atr_stop = entry_price - (2 * atr)
            
            # Fibonacci retracement stop (61.8% of recent swing)
            recent_low = market_data.get('recent_swing_low', entry_price * 0.95)
            fib_stop = recent_low * 0.95  # Below swing low
            
            # Use tighter of the two
            calculated_stop = max(atr_stop, fib_stop)
            
        else:  # SELL
            atr_stop = entry_price + (2 * atr)
            recent_high = market_data.get('recent_swing_high', entry_price * 1.05)
            fib_stop = recent_high * 1.05
            calculated_stop = min(atr_stop, fib_stop)
        
        # Apply maximum risk limit
        max_risk = entry_price * config.stop_loss_percentage
        if direction.upper() == 'BUY':
            final_stop = max(calculated_stop, entry_price - max_risk)
        else:
            final_stop = min(calculated_stop, entry_price + max_risk)
            
        self.logger.info(f"📊 Stop loss for {symbol}: {final_stop:.2f} "
                        f"(Literature: {stop_knowledge[0][:50]}...)")
        
        return final_stop
    
    def analyze_market_conditions(self, symbol: str, market_data: Dict) -> TradingSignal:
        """Analyze market using literature-based strategies"""
        
        current_price = market_data.get('price', 0)
        if current_price == 0:
            return TradingSignal(symbol, 'HOLD', 0.0, 'no_data', ['Insufficient market data'])
        
        # Query brain for market analysis
        market_analysis = self.query_literature_knowledge(
            f"market analysis {symbol} price action trend", symbol
        )
        
        # Price action analysis (Brooks method)
        price_action_signal = self._analyze_price_action(symbol, market_data)
        
        # Fibonacci analysis
        fib_signal = self._analyze_fibonacci_levels(symbol, market_data)
        
        # Support/Resistance (Naked Forex)
        sr_signal = self._analyze_support_resistance(symbol, market_data)
        
        # Combine signals with literature knowledge
        combined_signal = self._combine_signals([price_action_signal, fib_signal, sr_signal])
        combined_signal.literature_source = f"Multiple sources: {', '.join(self.active_strategies.keys())}"
        
        # Add literature reasoning
        combined_signal.reasoning.extend(market_analysis[:2])
        
        return combined_signal
    
    def _analyze_price_action(self, symbol: str, data: Dict) -> TradingSignal:
        """Brooks-style price action analysis"""
        price = data.get('price', 0)
        volume = data.get('volume', 0)
        
        # Simple trend analysis
        ma_20 = data.get('ma_20', price)
        ma_50 = data.get('ma_50', price)
        
        reasoning = []
        
        if price > ma_20 > ma_50:
            action = 'BUY'
            confidence = 0.7
            reasoning.append("Price above both moving averages - uptrend confirmed")
            reasoning.append("Brooks: Buy pullbacks in strong uptrends")
        elif price < ma_20 < ma_50:
            action = 'SELL'
            confidence = 0.7
            reasoning.append("Price below both moving averages - downtrend confirmed")  
            reasoning.append("Brooks: Sell rallies in strong downtrends")
        else:
            action = 'HOLD'
            confidence = 0.3
            reasoning.append("Mixed signals - price action unclear")
            
        return TradingSignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            strategy='brooks_price_action',
            reasoning=reasoning,
            literature_source='Al Brooks Price Action Trading'
        )
    
    def _analyze_fibonacci_levels(self, symbol: str, data: Dict) -> TradingSignal:
        """Fibonacci-based analysis"""
        price = data.get('price', 0)
        recent_high = data.get('recent_high', price * 1.1)
        recent_low = data.get('recent_low', price * 0.9)
        
        # Calculate key Fibonacci levels
        fib_range = recent_high - recent_low
        fib_618 = recent_high - (fib_range * 0.618)
        fib_382 = recent_high - (fib_range * 0.382)
        
        reasoning = []
        
        if abs(price - fib_618) / price < 0.005:  # Within 0.5%
            action = 'BUY'
            confidence = 0.8
            reasoning.append("Price at 61.8% Fibonacci retracement - strong support")
            reasoning.append("Fibonacci: Golden ratio level often provides bounce")
        elif abs(price - fib_382) / price < 0.005:
            action = 'BUY'
            confidence = 0.6
            reasoning.append("Price at 38.2% Fibonacci retracement - moderate support")
        else:
            action = 'HOLD'
            confidence = 0.4
            reasoning.append("Price not at key Fibonacci levels")
            
        return TradingSignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            strategy='fibonacci_retracement',
            reasoning=reasoning,
            literature_source='Fibonacci Trading Methods'
        )
    
    def _analyze_support_resistance(self, symbol: str, data: Dict) -> TradingSignal:
        """Support/Resistance analysis (Naked Forex style)"""
        price = data.get('price', 0)
        support_level = data.get('support', price * 0.95)
        resistance_level = data.get('resistance', price * 1.05)
        
        reasoning = []
        
        # Distance from support/resistance
        support_distance = (price - support_level) / price
        resistance_distance = (resistance_level - price) / price
        
        if support_distance < 0.01:  # Within 1% of support
            action = 'BUY'
            confidence = 0.75
            reasoning.append("Price near strong support level - bounce expected")
            reasoning.append("Naked Forex: Buy at support with tight stops")
        elif resistance_distance < 0.01:  # Within 1% of resistance
            action = 'SELL'
            confidence = 0.75
            reasoning.append("Price near strong resistance - rejection likely")
            reasoning.append("Naked Forex: Sell at resistance with tight stops")
        else:
            action = 'HOLD'
            confidence = 0.3
            reasoning.append("Price in no-trade zone between support/resistance")
            
        return TradingSignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            strategy='naked_forex',
            reasoning=reasoning,
            literature_source='Naked Forex'
        )
    
    def _combine_signals(self, signals: List[TradingSignal]) -> TradingSignal:
        """Combine multiple strategy signals intelligently"""
        if not signals:
            return TradingSignal('', 'HOLD', 0.0, 'no_signals', ['No signals generated'])
        
        # Weight signals by confidence
        buy_weight = sum(s.confidence for s in signals if s.action == 'BUY')
        sell_weight = sum(s.confidence for s in signals if s.action == 'SELL')
        hold_weight = sum(s.confidence for s in signals if s.action == 'HOLD')
        
        # Determine final action
        if buy_weight > sell_weight and buy_weight > hold_weight:
            final_action = 'BUY'
            final_confidence = min(buy_weight / len(signals), 1.0)
        elif sell_weight > buy_weight and sell_weight > hold_weight:
            final_action = 'SELL'
            final_confidence = min(sell_weight / len(signals), 1.0)
        else:
            final_action = 'HOLD'
            final_confidence = min(hold_weight / len(signals), 1.0)
        
        # Combine reasoning
        combined_reasoning = []
        for signal in signals:
            if signal.action == final_action:
                combined_reasoning.extend(signal.reasoning)
        
        return TradingSignal(
            symbol=signals[0].symbol,
            action=final_action,
            confidence=final_confidence,
            strategy='literature_combined',
            reasoning=combined_reasoning[:5],  # Top 5 reasons
            literature_source='Combined literature analysis'
        )
    
    def should_move_stop_loss(self, symbol: str, entry_price: float, 
                            current_price: float, current_stop: float, 
                            direction: str) -> Tuple[bool, float]:
        """Determine if stop loss should be moved (trailing stops)"""
        
        # Query brain for trailing stop guidance
        trailing_knowledge = self.query_literature_knowledge(
            f"trailing stop loss moving stops {direction}", symbol
        )
        
        profit_ratio = abs(current_price - entry_price) / entry_price
        
        # Move to breakeven after 1:1 risk/reward (literature rule)
        if profit_ratio >= config.risk_per_trade:
            if direction.upper() == 'BUY' and current_stop < entry_price:
                new_stop = entry_price * 1.001  # Just above breakeven
                return True, new_stop
            elif direction.upper() == 'SELL' and current_stop > entry_price:
                new_stop = entry_price * 0.999  # Just below breakeven
                return True, new_stop
        
        # Trail stop based on ATR or swing points
        if profit_ratio >= 0.05:  # 5% profit
            atr_multiplier = 1.5  # Tighter trailing
            
            if direction.upper() == 'BUY':
                new_stop = current_price - (current_price * 0.02 * atr_multiplier)
                if new_stop > current_stop:
                    return True, new_stop
            else:
                new_stop = current_price + (current_price * 0.02 * atr_multiplier)
                if new_stop < current_stop:
                    return True, new_stop
        
        return False, current_stop
    
    def get_position_size(self, symbol: str, entry_price: float, 
                         stop_loss: float, account_value: float) -> int:
        """Calculate position size based on literature risk management"""
        
        # Query brain for position sizing guidance
        sizing_knowledge = self.query_literature_knowledge(
            f"position sizing risk management {symbol}"
        )
        
        # Risk per trade from config (2% default)
        risk_amount = account_value * config.risk_per_trade
        
        # Risk per share
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share == 0:
            return 1
            
        # Calculate shares
        shares = int(risk_amount / risk_per_share)
        
        # Minimum and maximum limits
        shares = max(1, min(shares, int(config.max_position_size / entry_price)))
        
        self.logger.info(f"📊 Position size for {symbol}: {shares} shares "
                        f"(Risk: ${risk_amount:.2f}, Guidance: {sizing_knowledge[0][:50]}...)")
        
        return shares

# Global strategy engine instance
strategy_engine = IntelligentStrategyEngine()