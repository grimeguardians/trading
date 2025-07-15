"""Market analyst agent with Digital Brain integration"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from .base_agent import BaseAgent
from ..data_models import MarketData, TradingSignal
from ..strategies.technical_analysis import TechnicalIndicators
from ..strategies.sentiment_analyzer import SentimentAnalyzer
from ..strategies.portfolio_optimizer import PortfolioOptimizer

# Digital Brain integration
try:
    from knowledge_engine import DigitalBrain
    DIGITAL_BRAIN_AVAILABLE = True
    print("🧠 Digital Brain components loaded successfully!")
except ImportError:
    print("⚠️  Digital Brain not available - some advanced features disabled.")
    DIGITAL_BRAIN_AVAILABLE = False

# ML components
try:
    from ..strategies.advanced_ml_engine import AdvancedMLEngine
    from ..strategies.ml_performance_monitor import MLPerformanceMonitor
    ML_AVAILABLE = True
    print("🤖 Advanced ML Engine with ensemble methods loaded successfully!")
    print("📊 ML Performance monitoring enabled!")
except ImportError:
    print("⚠️  Advanced ML Engine not available - install scikit-learn for full ML capabilities")
    try:
        from ..strategies.ml_engine import MLPredictionEngine as AdvancedMLEngine
        from ..strategies.ml_performance_monitor import MLPerformanceMonitor
        ML_AVAILABLE = True
        print("🤖 Basic ML Engine loaded as fallback")
    except ImportError:
        ML_AVAILABLE = False

class MarketAnalystAgent(BaseAgent):
    """Streamlined market analyst with preserved Digital Brain functionality"""

    def __init__(self):
        super().__init__("MarketAnalyst")
        self.market_data = {}
        self.signals = []
        self.technical_indicators = TechnicalIndicators()
        self.ml_engine = AdvancedMLEngine() if ML_AVAILABLE else None
        self.ml_monitor = MLPerformanceMonitor() if ML_AVAILABLE else None
        self.sentiment_analyzer = SentimentAnalyzer()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.digital_brain = DigitalBrain() if DIGITAL_BRAIN_AVAILABLE else None
        self.last_price = {}
        
        # Digital Brain tracking
        self.brain_patterns_used = 0
        self.brain_insights_generated = 0
        self.brain_knowledge_queries = 0
        self.brain_learning_events = 0

    def process(self, market_data: MarketData) -> List[TradingSignal]:
        """Enhanced market analysis with Digital Brain integration"""
        try:
            self.market_data[market_data.symbol] = market_data
            self.technical_indicators.update_data(market_data)

            # Digital Brain: Process market event and learn patterns
            if self.digital_brain:
                self._process_digital_brain_event(market_data)

            # Update ML training data
            if self.ml_engine and market_data.symbol in self.last_price:
                self._update_ml_training(market_data)

            # Generate signals from multiple sources
            signals = []
            signals.extend(self._analyze_technical(market_data))
            
            if self.ml_engine:
                signals.extend(self._analyze_ml(market_data))
            
            signals.extend(self._analyze_sentiment(market_data))
            
            if self.digital_brain:
                signals.extend(self._analyze_brain_patterns(market_data))

            # Combine signals intelligently
            final_signals = self._combine_signals(signals, market_data.symbol)
            
            self.signals.extend(final_signals)
            self.last_price[market_data.symbol] = market_data.price

            if final_signals:
                self.logger.info(f"Generated {len(final_signals)} signals for {market_data.symbol}")

            return final_signals

        except Exception as e:
            self.logger.error(f"Error processing market data: {e}")
            return []

    def _process_digital_brain_event(self, market_data: MarketData):
        """Process market events through Digital Brain for learning"""
        try:
            ta = self.technical_indicators.calculate_indicators(market_data.symbol)
            sentiment = self.sentiment_analyzer.analyze_sentiment(market_data.symbol)

            event_data = {
                'symbol': market_data.symbol,
                'price': market_data.price,
                'volume': market_data.volume,
                'timestamp': market_data.timestamp.isoformat(),
                'event_type': 'market_data_update',
                'rsi': ta.rsi if ta else 50,
                'macd': ta.macd if ta else 0,
                'sentiment': sentiment.sentiment_score if sentiment else 0,
                'support': ta.support_level if ta else 0,
                'resistance': ta.resistance_level if ta else 0
            }

            # Add outcome if we have previous price for learning
            if market_data.symbol in self.last_price:
                price_change = (market_data.price - self.last_price[market_data.symbol]) / self.last_price[market_data.symbol]
                event_data['outcome'] = {
                    'price_change': price_change,
                    'successful': abs(price_change) > 0.01,
                    'direction': 'up' if price_change > 0 else 'down',
                    'magnitude': abs(price_change)
                }
                self.brain_learning_events += 1

            brain_result = self.digital_brain.process_market_event(event_data)
            if not brain_result.get('error'):
                self.brain_patterns_used += len(brain_result.get('recognized_patterns', []))

        except Exception as e:
            self.logger.error(f"Error processing Digital Brain event: {e}")

    def _update_ml_training(self, market_data: MarketData):
        """Update ML models with new market data"""
        try:
            price_change = market_data.price - self.last_price[market_data.symbol]
            ta = self.technical_indicators.calculate_indicators(market_data.symbol)
            
            if ta:
                sentiment = self.sentiment_analyzer.analyze_sentiment(market_data.symbol)
                features = self.ml_engine.engineer_features(market_data.symbol, ta, market_data, sentiment)
                self.ml_engine.update_training_data(market_data.symbol, features, price_change, market_data)

                # Update ML performance monitoring
                if self.ml_monitor:
                    self.ml_monitor.update_prediction_outcomes(market_data.symbol, market_data.price)

        except Exception as e:
            self.logger.error(f"Error updating ML training: {e}")

    def _analyze_technical(self, data: MarketData) -> List[TradingSignal]:
        """Technical analysis signals"""
        signals = []
        ta = self.technical_indicators.calculate_indicators(data.symbol)
        if not ta:
            return signals

        # Simple but effective technical signals
        if ta.rsi < 30 and data.price < ta.bb_lower:
            signals.append(TradingSignal(
                symbol=data.symbol,
                action='BUY',
                confidence=0.7,
                reason="Oversold: RSI < 30 and price below BB lower",
                timestamp=datetime.now(),
                stop_loss_price=data.price - (ta.atr * 2),
                take_profit_price=data.price + (ta.atr * 3)
            ))
        elif ta.rsi > 70 and data.price > ta.bb_upper:
            signals.append(TradingSignal(
                symbol=data.symbol,
                action='SELL', 
                confidence=0.7,
                reason="Overbought: RSI > 70 and price above BB upper",
                timestamp=datetime.now(),
                stop_loss_price=data.price + (ta.atr * 2),
                take_profit_price=data.price - (ta.atr * 3)
            ))

        return signals

    def _analyze_ml(self, data: MarketData) -> List[TradingSignal]:
        """ML-based signals"""
        signals = []
        try:
            ta = self.technical_indicators.calculate_indicators(data.symbol)
            if not ta:
                return signals

            sentiment = self.sentiment_analyzer.analyze_sentiment(data.symbol)
            features = self.ml_engine.engineer_features(data.symbol, ta, data, sentiment)
            prediction = self.ml_engine.predict(data.symbol, features)

            if prediction and prediction.prediction in ['BUY', 'SELL'] and prediction.confidence > 0.65:
                stop_distance = ta.atr * 2
                if prediction.prediction == 'BUY':
                    stop_loss_price = data.price - stop_distance
                    take_profit_price = data.price + (stop_distance * 1.5)
                else:
                    stop_loss_price = data.price + stop_distance
                    take_profit_price = data.price - (stop_distance * 1.5)

                signals.append(TradingSignal(
                    symbol=data.symbol,
                    action=prediction.prediction,
                    confidence=prediction.confidence * 0.85,
                    reason=f"ML prediction (accuracy: {prediction.model_accuracy:.2f})",
                    timestamp=datetime.now(),
                    stop_loss_price=stop_loss_price,
                    take_profit_price=take_profit_price
                ))

        except Exception as e:
            self.logger.error(f"Error in ML analysis: {e}")

        return signals

    def _analyze_sentiment(self, data: MarketData) -> List[TradingSignal]:
        """Sentiment-based signals"""
        signals = []
        try:
            sentiment = self.sentiment_analyzer.analyze_sentiment(data.symbol)
            
            if abs(sentiment.sentiment_score) > 0.4:
                ta = self.technical_indicators.calculate_indicators(data.symbol)
                if ta:
                    action = 'BUY' if sentiment.sentiment_score > 0 else 'SELL'
                    stop_distance = ta.atr * 2.5
                    
                    if action == 'BUY':
                        stop_loss_price = data.price - stop_distance
                        take_profit_price = data.price + stop_distance
                    else:
                        stop_loss_price = data.price + stop_distance
                        take_profit_price = data.price - stop_distance

                    signals.append(TradingSignal(
                        symbol=data.symbol,
                        action=action,
                        confidence=min(abs(sentiment.sentiment_score), 0.6),
                        reason=f"Sentiment: {sentiment.sentiment_label} ({sentiment.sentiment_score:.2f})",
                        timestamp=datetime.now(),
                        stop_loss_price=stop_loss_price,
                        take_profit_price=take_profit_price,
                        signal_strength="WEAK"
                    ))

        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {e}")

        return signals

    def _analyze_brain_patterns(self, data: MarketData) -> List[TradingSignal]:
        """Digital Brain pattern recognition signals"""
        signals = []
        if not self.digital_brain:
            return signals

        try:
            ta = self.technical_indicators.calculate_indicators(data.symbol)
            if not ta:
                return signals

            # Prepare context for pattern recognition
            current_context = {
                'symbol': data.symbol,
                'price': data.price,
                'rsi': ta.rsi,
                'macd': ta.macd,
                'volume_ratio': data.volume / ta.volume_sma if ta.volume_sma > 0 else 1,
                'bb_position': (data.price - ta.bb_middle) / (ta.bb_upper - ta.bb_lower) if ta.bb_upper != ta.bb_lower else 0,
                'market_regime': 'normal'  # Simplified regime detection
            }

            # Recognize patterns
            recognized_patterns = self.digital_brain.pattern_engine.recognize_patterns(
                data.symbol, current_context
            )

            # Generate signals from patterns
            for pattern, confidence in recognized_patterns:
                if confidence > 0.6:
                    action = self._determine_action_from_pattern(pattern)
                    
                    if action != 'HOLD':
                        stop_distance = ta.atr * 2
                        if action == 'BUY':
                            stop_loss_price = data.price - stop_distance
                            take_profit_price = data.price + (stop_distance * 2)
                        else:
                            stop_loss_price = data.price + stop_distance
                            take_profit_price = data.price - (stop_distance * 2)

                        signals.append(TradingSignal(
                            symbol=data.symbol,
                            action=action,
                            confidence=confidence * pattern.success_rate,
                            reason=f"Brain pattern: {pattern.pattern_type}",
                            timestamp=datetime.now(),
                            stop_loss_price=stop_loss_price,
                            take_profit_price=take_profit_price
                        ))
                        self.brain_insights_generated += 1

        except Exception as e:
            self.logger.error(f"Error in Brain pattern analysis: {e}")

        return signals

    def _determine_action_from_pattern(self, pattern):
        """Determine trading action from Digital Brain pattern"""
        pattern_lower = pattern.pattern_type.lower()
        
        if any(keyword in pattern_lower for keyword in ['bullish', 'breakout', 'uptrend', 'buy']):
            return 'BUY'
        elif any(keyword in pattern_lower for keyword in ['bearish', 'breakdown', 'downtrend', 'sell']):
            return 'SELL'
        elif pattern.outcomes.get('price_increase', False) and pattern.success_rate > 0.6:
            return 'BUY'
        elif pattern.outcomes.get('price_decrease', False) and pattern.success_rate > 0.6:
            return 'SELL'
        
        return 'HOLD'

    def _combine_signals(self, signals: List[TradingSignal], symbol: str) -> List[TradingSignal]:
        """Combine multiple signals intelligently"""
        if not signals:
            return []

        # Group by action
        buy_signals = [s for s in signals if s.action == 'BUY']
        sell_signals = [s for s in signals if s.action == 'SELL']

        combined = []

        # Combine BUY signals
        if buy_signals:
            avg_confidence = sum(s.confidence for s in buy_signals) / len(buy_signals)
            best_stop = max([s.stop_loss_price for s in buy_signals if s.stop_loss_price], default=None)
            best_profit = min([s.take_profit_price for s in buy_signals if s.take_profit_price], default=None)
            
            combined.append(TradingSignal(
                symbol=symbol,
                action='BUY',
                confidence=min(avg_confidence * 1.1, 0.95),  # Boost for agreement
                reason=f"Combined BUY ({len(buy_signals)} signals)",
                timestamp=datetime.now(),
                stop_loss_price=best_stop,
                take_profit_price=best_profit
            ))

        # Combine SELL signals
        if sell_signals:
            avg_confidence = sum(s.confidence for s in sell_signals) / len(sell_signals)
            best_stop = min([s.stop_loss_price for s in sell_signals if s.stop_loss_price], default=None)
            best_profit = max([s.take_profit_price for s in sell_signals if s.take_profit_price], default=None)
            
            combined.append(TradingSignal(
                symbol=symbol,
                action='SELL',
                confidence=min(avg_confidence * 1.1, 0.95),
                reason=f"Combined SELL ({len(sell_signals)} signals)",
                timestamp=datetime.now(),
                stop_loss_price=best_stop,
                take_profit_price=best_profit
            ))

        # Return strongest signal if conflicting
        if len(combined) > 1:
            return [max(combined, key=lambda x: x.confidence)]
        
        return combined