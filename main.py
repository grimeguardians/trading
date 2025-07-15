import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import random
import numpy as np
import pandas as pd
from enum import Enum
try:
    import pandas_ta as ta
    import yfinance as yf
    from scipy import stats
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import re
    PANDAS_TA_AVAILABLE = True
    ML_AVAILABLE = True
    print("✅ Advanced technical analysis and ML libraries loaded successfully!")
except ImportError:
    print("⚠️  Using simplified indicators. Install pandas-ta and scikit-learn for enhanced analysis.")
    ta = None
    yf = None
    stats = None
    RandomForestClassifier = None
    StandardScaler = None
    train_test_split = None
    re = None
    PANDAS_TA_AVAILABLE = False
    ML_AVAILABLE = False

# Import Digital Brain components
try:
    from knowledge_engine import DigitalBrain, MarketPattern, DocumentEntity
    from document_upload import TradingDocumentUploader
    DIGITAL_BRAIN_AVAILABLE = True
    print("🧠 Digital Brain components loaded successfully!")
    print("📚 Document upload system ready for trading literature!")
except ImportError:
    print("⚠️  Digital Brain not available - some advanced features disabled.")
    DIGITAL_BRAIN_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler()
    ]
)

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LIMIT = "STOP_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT"
    TRAILING_STOP = "TRAILING_STOP"

class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"

@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    price: float
    volume: int
    timestamp: datetime
    bid: float
    ask: float
    high_24h: float = 0.0
    low_24h: float = 0.0

@dataclass
class TradingSignal:
    """Trading signal structure"""
    symbol: str
    action: str  # 'BUY' or 'SELL'
    confidence: float
    reason: str
    timestamp: datetime
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    signal_strength: str = "MEDIUM"  # WEAK, MEDIUM, STRONG

@dataclass
class TradeOrder:
    """Enhanced trade order structure with stop-loss support"""
    order_id: str
    symbol: str
    action: str
    quantity: int
    price: float
    order_type: OrderType
    timestamp: datetime
    status: OrderStatus = OrderStatus.PENDING
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    trailing_stop_distance: Optional[float] = None
    parent_order_id: Optional[str] = None  # For stop-loss orders linked to main orders
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    expiry_time: Optional[datetime] = None

@dataclass
class Position:
    """Enhanced position tracking structure"""
    symbol: str
    quantity: int
    avg_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    trailing_stop_price: Optional[float] = None
    max_price_since_entry: float = 0.0
    entry_timestamp: datetime = field(default_factory=datetime.now)
    risk_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class TechnicalAnalysis:
    """Technical analysis results"""
    symbol: str
    sma_20: float
    sma_50: float
    ema_12: float
    ema_26: float
    rsi: float
    macd: float
    macd_signal: float
    macd_histogram: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    volume_sma: float
    atr: float
    fibonacci_levels: Dict[str, float]
    stochastic_k: float
    stochastic_d: float
    williams_r: float
    timestamp: datetime
    support_level: float = 0.0
    resistance_level: float = 0.0
    volatility: float = 0.0

@dataclass
class MLPrediction:
    """Machine learning prediction results"""
    symbol: str
    prediction: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    feature_importance: Dict[str, float]
    model_accuracy: float
    timestamp: datetime
    price_target: Optional[float] = None
    risk_score: float = 0.0

@dataclass
class SentimentData:
    """Sentiment analysis data"""
    symbol: str
    sentiment_score: float  # -1 to +1
    sentiment_label: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    news_count: int
    social_mentions: int
    timestamp: datetime
    sentiment_trend: str = "STABLE"  # IMPROVING, DECLINING, STABLE

@dataclass
class PortfolioOptimization:
    """Portfolio optimization results"""
    optimal_weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    risk_metrics: Dict[str, float]
    rebalance_suggestions: Dict[str, float]
    timestamp: datetime

@dataclass
class RiskAlert:
    """Risk management alerts"""
    alert_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    message: str
    symbol: Optional[str]
    timestamp: datetime
    action_required: bool = False

class StopLossManager:
    """Advanced stop-loss management system"""

    def __init__(self):
        self.active_stops = {}  # order_id -> stop_loss_config
        self.trailing_stops = {}  # position_symbol -> trailing_config

    def create_stop_loss_order(self, position: Position, stop_loss_price: float, 
                              order_type: OrderType = OrderType.STOP_LOSS) -> TradeOrder:
        """Create a stop-loss order for a position"""
        order_id = f"SL_{position.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Determine action (opposite of position)
        action = 'SELL' if position.quantity > 0 else 'BUY'

        stop_order = TradeOrder(
            order_id=order_id,
            symbol=position.symbol,
            action=action,
            quantity=abs(position.quantity),
            price=stop_loss_price,
            order_type=order_type,
            timestamp=datetime.now(),
            stop_loss_price=stop_loss_price
        )

        self.active_stops[order_id] = {
            'order': stop_order,
            'position_symbol': position.symbol,
            'trigger_price': stop_loss_price
        }

        return stop_order

    def create_trailing_stop(self, position: Position, trail_distance: float) -> None:
        """Create a trailing stop for a position"""
        if position.quantity > 0:  # Long position
            initial_stop = position.current_price - trail_distance
        else:  # Short position
            initial_stop = position.current_price + trail_distance

        self.trailing_stops[position.symbol] = {
            'trail_distance': trail_distance,
            'current_stop_price': initial_stop,
            'highest_price': position.current_price if position.quantity > 0 else None,
            'lowest_price': position.current_price if position.quantity < 0 else None,
            'position_quantity': position.quantity
        }

    def update_trailing_stops(self, market_data: MarketData) -> List[TradeOrder]:
        """Update trailing stops based on new market data"""
        triggered_orders = []

        if market_data.symbol in self.trailing_stops:
            trail_config = self.trailing_stops[market_data.symbol]
            current_price = market_data.price

            if trail_config['position_quantity'] > 0:  # Long position
                # Update highest price
                if current_price > trail_config['highest_price']:
                    trail_config['highest_price'] = current_price
                    # Update stop price
                    new_stop = current_price - trail_config['trail_distance']
                    if new_stop > trail_config['current_stop_price']:
                        trail_config['current_stop_price'] = new_stop

                # Check if stop is triggered
                if current_price <= trail_config['current_stop_price']:
                    # Create market sell order
                    order = self._create_triggered_order(market_data.symbol, trail_config, current_price)
                    triggered_orders.append(order)
                    del self.trailing_stops[market_data.symbol]

            else:  # Short position
                # Update lowest price
                if current_price < trail_config['lowest_price']:
                    trail_config['lowest_price'] = current_price
                    # Update stop price
                    new_stop = current_price + trail_config['trail_distance']
                    if new_stop < trail_config['current_stop_price']:
                        trail_config['current_stop_price'] = new_stop

                # Check if stop is triggered
                if current_price >= trail_config['current_stop_price']:
                    # Create market buy order
                    order = self._create_triggered_order(market_data.symbol, trail_config, current_price)
                    triggered_orders.append(order)
                    del self.trailing_stops[market_data.symbol]

        return triggered_orders

    def check_stop_triggers(self, market_data: MarketData) -> List[TradeOrder]:
        """Check if any stop-loss orders should be triggered"""
        triggered_orders = []
        orders_to_remove = []

        for order_id, stop_config in self.active_stops.items():
            if stop_config['position_symbol'] == market_data.symbol:
                stop_order = stop_config['order']
                trigger_price = stop_config['trigger_price']

                should_trigger = False

                if stop_order.action == 'SELL' and market_data.price <= trigger_price:
                    should_trigger = True
                elif stop_order.action == 'BUY' and market_data.price >= trigger_price:
                    should_trigger = True

                if should_trigger:
                    # Convert to market order
                    market_order = TradeOrder(
                        order_id=f"MKT_{order_id}",
                        symbol=stop_order.symbol,
                        action=stop_order.action,
                        quantity=stop_order.quantity,
                        price=market_data.price,
                        order_type=OrderType.MARKET,
                        timestamp=datetime.now(),
                        parent_order_id=order_id
                    )
                    triggered_orders.append(market_order)
                    orders_to_remove.append(order_id)

        # Remove triggered stops
        for order_id in orders_to_remove:
            del self.active_stops[order_id]

        return triggered_orders

    def _create_triggered_order(self, symbol: str, trail_config: Dict, current_price: float) -> TradeOrder:
        """Create order when trailing stop is triggered"""
        action = 'SELL' if trail_config['position_quantity'] > 0 else 'BUY'

        return TradeOrder(
            order_id=f"TS_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            symbol=symbol,
            action=action,
            quantity=abs(trail_config['position_quantity']),
            price=current_price,
            order_type=OrderType.MARKET,
            timestamp=datetime.now()
        )

class MLPredictionEngine:
    """Enhanced Machine Learning prediction engine"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_history = {}
        self.prediction_history = {}
        self.model_accuracy = {}
        self.min_training_samples = 50
        self.feature_importance_history = {}

    def prepare_features(self, symbol: str, technical_analysis: TechnicalAnalysis, 
                        market_data: MarketData, sentiment: Optional[SentimentData] = None) -> np.array:
        """Enhanced feature preparation with additional technical indicators"""
        features = [
            technical_analysis.rsi / 100.0,  # Normalize RSI
            technical_analysis.macd,
            technical_analysis.macd_signal,
            technical_analysis.macd_histogram,
            (market_data.price - technical_analysis.bb_middle) / (technical_analysis.bb_upper - technical_analysis.bb_lower),  # BB position
            technical_analysis.stochastic_k / 100.0,
            technical_analysis.williams_r / -100.0,
            market_data.volume / technical_analysis.volume_sma,  # Volume ratio
            technical_analysis.atr / market_data.price,  # Normalized ATR
            (technical_analysis.ema_12 - technical_analysis.ema_26) / market_data.price,  # MACD ratio
            technical_analysis.volatility,  # Volatility
            (market_data.price - technical_analysis.support_level) / market_data.price if technical_analysis.support_level > 0 else 0,
            (technical_analysis.resistance_level - market_data.price) / market_data.price if technical_analysis.resistance_level > 0 else 0,
        ]

        # Add sentiment features if available
        if sentiment:
            features.extend([
                sentiment.sentiment_score,
                sentiment.news_count / 10.0,  # Normalize news count
                sentiment.social_mentions / 100.0,  # Normalize social mentions
                1.0 if sentiment.sentiment_trend == 'IMPROVING' else (0.5 if sentiment.sentiment_trend == 'STABLE' else 0.0)
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.5])  # Neutral sentiment

        # Add time-based features
        hour = datetime.now().hour
        features.extend([
            hour / 24.0,  # Time of day
            1.0 if 9 <= hour <= 16 else 0.0,  # Market hours
        ])

        return np.array(features)

    def update_training_data(self, symbol: str, features: np.array, target: float):
        """Enhanced training data updates with feature importance tracking"""
        if symbol not in self.feature_history:
            self.feature_history[symbol] = []

        # Store feature vector with target (price change direction)
        self.feature_history[symbol].append({
            'features': features,
            'target': 1 if target > 0 else 0,  # Binary: 1 for price increase, 0 for decrease
            'target_magnitude': abs(target),  # Store magnitude for regression tasks
            'timestamp': datetime.now()
        })

        # Keep only recent history for efficiency
        if len(self.feature_history[symbol]) > 500:
            self.feature_history[symbol] = self.feature_history[symbol][-500:]

    def train_model(self, symbol: str) -> bool:
        """Enhanced model training with hyperparameter optimization"""
        if symbol not in self.feature_history or len(self.feature_history[symbol]) < self.min_training_samples:
            return False

        try:
            # Prepare training data
            features = np.array([item['features'] for item in self.feature_history[symbol]])
            targets = np.array([item['target'] for item in self.feature_history[symbol]])

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

            # Scale features
            if symbol not in self.scalers:
                self.scalers[symbol] = StandardScaler()

            X_train_scaled = self.scalers[symbol].fit_transform(X_train)
            X_test_scaled = self.scalers[symbol].transform(X_test)

            # Train enhanced Random Forest model
            if symbol not in self.models:
                self.models[symbol] = RandomForestClassifier(
                    n_estimators=150,  # Increased trees
                    max_depth=12,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    class_weight='balanced',
                    bootstrap=True
                )

            self.models[symbol].fit(X_train_scaled, y_train)

            # Calculate accuracy
            accuracy = self.models[symbol].score(X_test_scaled, y_test)
            self.model_accuracy[symbol] = accuracy

            # Store feature importance
            feature_names = ['RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_Position', 
                           'Stoch_K', 'Williams_R', 'Volume_Ratio', 'ATR_Norm', 'MACD_Ratio',
                           'Volatility', 'Support_Distance', 'Resistance_Distance',
                           'Sentiment', 'News_Count', 'Social_Mentions', 'Sentiment_Trend',
                           'Time_Of_Day', 'Market_Hours']

            importance_dict = dict(zip(feature_names, self.models[symbol].feature_importances_))
            self.feature_importance_history[symbol] = importance_dict

            logging.info(f"Enhanced ML model trained for {symbol} with accuracy: {accuracy:.3f}")
            return True

        except Exception as e:
            logging.error(f"Error training ML model for {symbol}: {e}")
            return False

    def predict(self, symbol: str, features: np.array) -> Optional[MLPrediction]:
        """Enhanced prediction with price targets and risk scoring"""
        if symbol not in self.models or symbol not in self.scalers:
            return None

        try:
            # Scale features
            features_scaled = self.scalers[symbol].transform(features.reshape(1, -1))

            # Make prediction
            prediction_proba = self.models[symbol].predict_proba(features_scaled)[0]
            prediction_class = self.models[symbol].predict(features_scaled)[0]

            # Get feature importance
            feature_importance = self.feature_importance_history.get(symbol, {})

            # Determine action and confidence
            confidence = max(prediction_proba)

            # Enhanced decision logic
            if prediction_class == 1 and confidence > 0.6:
                action = 'BUY'
            elif prediction_class == 0 and confidence > 0.6:
                action = 'SELL'
            else:
                action = 'HOLD'

            # Calculate risk score based on feature importance and volatility
            volatility_importance = feature_importance.get('Volatility', 0)
            atr_importance = feature_importance.get('ATR_Norm', 0)
            risk_score = (volatility_importance + atr_importance) * confidence

            return MLPrediction(
                symbol=symbol,
                prediction=action,
                confidence=confidence,
                feature_importance=feature_importance,
                model_accuracy=self.model_accuracy.get(symbol, 0.0),
                timestamp=datetime.now(),
                risk_score=risk_score
            )

        except Exception as e:
            logging.error(f"Error making ML prediction for {symbol}: {e}")
            return None

class SentimentAnalyzer:
    """Enhanced sentiment analysis engine"""

    def __init__(self):
        self.sentiment_history = {}
        self.news_keywords = {
            'positive': ['growth', 'profit', 'increase', 'beat', 'strong', 'bullish', 'upgrade', 'buy', 
                        'earnings', 'revenue', 'expansion', 'innovation', 'breakthrough'],
            'negative': ['loss', 'decline', 'fall', 'miss', 'weak', 'bearish', 'downgrade', 'sell',
                        'lawsuit', 'investigation', 'recession', 'crisis', 'bankruptcy']
        }
        self.sentiment_trends = {}

    def analyze_sentiment(self, symbol: str) -> SentimentData:
        """Enhanced sentiment analysis with trend detection"""
        try:
            # Simulate news and social media data
            news_articles = self._simulate_news_data(symbol)
            social_posts = self._simulate_social_data(symbol)

            # Calculate sentiment score
            sentiment_score = self._calculate_sentiment_score(news_articles + social_posts)

            # Determine sentiment label
            if sentiment_score > 0.1:
                sentiment_label = 'BULLISH'
            elif sentiment_score < -0.1:
                sentiment_label = 'BEARISH'
            else:
                sentiment_label = 'NEUTRAL'

            # Calculate sentiment trend
            sentiment_trend = self._calculate_sentiment_trend(symbol, sentiment_score)

            sentiment_data = SentimentData(
                symbol=symbol,
                sentiment_score=sentiment_score,
                sentiment_label=sentiment_label,
                news_count=len(news_articles),
                social_mentions=len(social_posts),
                timestamp=datetime.now(),
                sentiment_trend=sentiment_trend
            )

            # Store in history
            if symbol not in self.sentiment_history:
                self.sentiment_history[symbol] = []
            self.sentiment_history[symbol].append(sentiment_data)

            # Keep only recent history
            if len(self.sentiment_history[symbol]) > 100:
                self.sentiment_history[symbol] = self.sentiment_history[symbol][-100:]

            return sentiment_data

        except Exception as e:
            logging.error(f"Error analyzing sentiment for {symbol}: {e}")
            return SentimentData(symbol, 0.0, 'NEUTRAL', 0, 0, datetime.now(), 'STABLE')

    def _calculate_sentiment_trend(self, symbol: str, current_sentiment: float) -> str:
        """Calculate sentiment trend over time"""
        if symbol not in self.sentiment_history or len(self.sentiment_history[symbol]) < 3:
            return 'STABLE'

        recent_sentiments = [data.sentiment_score for data in self.sentiment_history[symbol][-3:]]
        recent_sentiments.append(current_sentiment)

        # Calculate trend
        if len(recent_sentiments) >= 2:
            trend_slope = np.polyfit(range(len(recent_sentiments)), recent_sentiments, 1)[0]

            if trend_slope > 0.1:
                return 'IMPROVING'
            elif trend_slope < -0.1:
                return 'DECLINING'
            else:
                return 'STABLE'

        return 'STABLE'

    def _simulate_news_data(self, symbol: str) -> List[str]:
        """Enhanced news simulation with more realistic patterns"""
        news_templates = [
            f"{symbol} reports strong quarterly earnings beating estimates",
            f"{symbol} announces major product launch with innovative features",
            f"Analysts upgrade {symbol} price target citing growth potential",
            f"{symbol} faces regulatory challenges in key markets",
            f"{symbol} stock price shows high volatility amid market uncertainty",
            f"Market experts divided on {symbol} future performance",
            f"{symbol} CEO announces strategic partnership deal",
            f"{symbol} invests heavily in research and development",
            f"Economic conditions impact {symbol} business outlook",
            f"{symbol} receives positive analyst coverage from major firms"
        ]

        # Time-based news frequency (more news during market hours)
        hour = datetime.now().hour
        base_articles = 2 if 9 <= hour <= 16 else 1
        num_articles = random.randint(0, base_articles + 3)

        return random.sample(news_templates, min(num_articles, len(news_templates)))

    def _simulate_social_data(self, symbol: str) -> List[str]:
        """Enhanced social media simulation"""
        social_templates = [
            f"Bullish on {symbol} for the long term! Great fundamentals 📈",
            f"{symbol} to the moon! 🚀 Strong technical setup",
            f"Considering taking profits on my {symbol} position",
            f"{symbol} looks overvalued at current levels, might short",
            f"Great buying opportunity for {symbol} on this dip",
            f"{symbol} earnings were disappointing, selling my shares",
            f"Love the innovation {symbol} is bringing to the market",
            f"{symbol} has solid leadership and vision for the future",
            f"Concerned about {symbol}'s exposure to market risks",
            f"Technical analysis shows strong support for {symbol}"
        ]

        # Social activity varies throughout the day
        hour = datetime.now().hour
        base_posts = 5 if 9 <= hour <= 16 else 3
        num_posts = random.randint(0, base_posts + 5)

        return random.sample(social_templates, min(num_posts, len(social_templates)))

    def _calculate_sentiment_score(self, text_data: List[str]) -> float:
        """Enhanced sentiment scoring with weighted keywords"""
        if not text_data:
            return 0.0

        total_score = 0.0
        total_weight = 0.0

        for text in text_data:
            text_lower = text.lower()

            # Count weighted keywords
            positive_score = 0
            negative_score = 0

            for word in self.news_keywords['positive']:
                if word in text_lower:
                    positive_score += 1 + text_lower.count(word) * 0.5  # Weight repeated mentions

            for word in self.news_keywords['negative']:
                if word in text_lower:
                    negative_score += 1 + text_lower.count(word) * 0.5

            # Calculate sentiment for this text with decay
            total_mentions = positive_score + negative_score
            if total_mentions > 0:
                text_sentiment = (positive_score - negative_score) / total_mentions
                weight = min(total_mentions, 5)  # Cap weight at 5
                total_score += text_sentiment * weight
                total_weight += weight

        # Normalize final score
        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            final_score = 0.0

        return max(-1.0, min(1.0, final_score))

class PortfolioOptimizer:
    """Enhanced portfolio optimization with risk parity and factor models"""

    def __init__(self):
        self.return_history = {}
        self.correlation_matrix = None
        self.optimization_history = []
        self.risk_models = {}

    def optimize_portfolio(self, positions: Dict[str, Position], 
                         technical_data: Dict[str, TechnicalAnalysis]) -> PortfolioOptimization:
        """Enhanced portfolio optimization with multiple strategies"""
        try:
            symbols = list(positions.keys())
            if len(symbols) < 2:
                # Can't optimize with less than 2 assets
                equal_weights = {symbol: 1.0/len(symbols) for symbol in symbols}
                return PortfolioOptimization(
                    optimal_weights=equal_weights,
                    expected_return=0.0,
                    expected_volatility=0.0,
                    sharpe_ratio=0.0,
                    risk_metrics={},
                    rebalance_suggestions={},
                    timestamp=datetime.now()
                )

            # Calculate expected returns and risk metrics
            expected_returns = self._calculate_expected_returns(symbols, technical_data)
            covariance_matrix = self._calculate_covariance_matrix(symbols)

            # Multiple optimization strategies
            strategies = {
                'mean_variance': self._mean_variance_optimization(expected_returns, covariance_matrix),
                'risk_parity': self._risk_parity_optimization(covariance_matrix),
                'maximum_diversification': self._max_diversification_optimization(covariance_matrix),
                'minimum_variance': self._minimum_variance_optimization(covariance_matrix)
            }

            # Select best strategy based on current market conditions
            optimal_weights = self._select_optimal_strategy(strategies, expected_returns, covariance_matrix)

            # Calculate portfolio metrics
            portfolio_return = np.sum(expected_returns * optimal_weights)
            portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(covariance_matrix, optimal_weights)))
            sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0.0

            # Generate rebalancing suggestions
            current_weights = self._calculate_current_weights(positions)
            rebalance_suggestions = {}
            for i, symbol in enumerate(symbols):
                weight_diff = optimal_weights[i] - current_weights.get(symbol, 0)
                if abs(weight_diff) > 0.05:  # 5% threshold
                    rebalance_suggestions[symbol] = weight_diff

            # Enhanced risk metrics
            risk_metrics = {
                'concentration_risk': self._calculate_concentration_risk(optimal_weights),
                'diversification_ratio': self._calculate_diversification_ratio(optimal_weights, covariance_matrix),
                'max_weight': np.max(optimal_weights),
                'min_weight': np.min(optimal_weights),
                'effective_assets': 1 / np.sum(optimal_weights ** 2),  # Effective number of assets
                'var_95': self._calculate_portfolio_var(optimal_weights, expected_returns, covariance_matrix)
            }

            optimization_result = PortfolioOptimization(
                optimal_weights=dict(zip(symbols, optimal_weights)),
                expected_return=portfolio_return,
                expected_volatility=portfolio_volatility,
                sharpe_ratio=sharpe_ratio,
                risk_metrics=risk_metrics,
                rebalance_suggestions=rebalance_suggestions,
                timestamp=datetime.now()
            )

            self.optimization_history.append(optimization_result)
            return optimization_result

        except Exception as e:
            logging.error(f"Error in portfolio optimization: {e}")
            return PortfolioOptimization({}, 0.0, 0.0, 0.0, {}, {}, datetime.now())

    def _mean_variance_optimization(self, expected_returns: np.array, covariance_matrix: np.array) -> np.array:
        """Mean-variance optimization (Markowitz)"""
        n_assets = len(expected_returns)
        risk_aversion = 5.0

        # Simplified mean-variance: w = (1/λ) * Σ^(-1) * μ
        try:
            inv_cov = np.linalg.inv(covariance_matrix + np.eye(n_assets) * 1e-8)  # Add regularization
            weights = np.dot(inv_cov, expected_returns) / risk_aversion
            weights = np.abs(weights)  # Ensure positive weights
            weights = weights / np.sum(weights)  # Normalize
            return weights
        except:
            return np.ones(n_assets) / n_assets  # Equal weights fallback

    def _risk_parity_optimization(self, covariance_matrix: np.array) -> np.array:
        """Risk parity optimization"""
        n_assets = len(covariance_matrix)

        # Simplified risk parity: inverse volatility weights
        volatilities = np.sqrt(np.diag(covariance_matrix))
        weights = 1.0 / volatilities
        weights = weights / np.sum(weights)
        return weights

    def _max_diversification_optimization(self, covariance_matrix: np.array) -> np.array:
        """Maximum diversification optimization"""
        n_assets = len(covariance_matrix)
        volatilities = np.sqrt(np.diag(covariance_matrix))

        # Weights proportional to inverse correlation
        weights = 1.0 / volatilities
        weights = weights / np.sum(weights)
        return weights

    def _minimum_variance_optimization(self, covariance_matrix: np.array) -> np.array:
        """Minimum variance optimization"""
        n_assets = len(covariance_matrix)

        try:
            inv_cov = np.linalg.inv(covariance_matrix + np.eye(n_assets) * 1e-8)
            ones = np.ones(n_assets)
            weights = np.dot(inv_cov, ones) / np.dot(ones, np.dot(inv_cov, ones))
            return np.abs(weights)  # Ensure positive weights
        except:
            return np.ones(n_assets) / n_assets

    def _select_optimal_strategy(self, strategies: Dict[str, np.array], 
                               expected_returns: np.array, covariance_matrix: np.array) -> np.array:
        """Select optimal strategy based on market conditions"""
        # For simplicity, use mean-variance as default
        # In practice, this would consider market regime, volatility, etc.
        return strategies.get('mean_variance', strategies['risk_parity'])

    def _calculate_diversification_ratio(self, weights: np.array, covariance_matrix: np.array) -> float:
        """Calculate diversification ratio"""
        weighted_vol = np.sum(weights * np.sqrt(np.diag(covariance_matrix)))
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
        return weighted_vol / portfolio_vol if portfolio_vol > 0 else 1.0

    def _calculate_portfolio_var(self, weights: np.array, expected_returns: np.array, 
                                covariance_matrix: np.array, confidence: float = 0.05) -> float:
        """Calculate portfolio Value at Risk"""
        portfolio_return = np.sum(weights * expected_returns)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))

        # Assuming normal distribution
        from scipy.stats import norm
        var = portfolio_return - norm.ppf(1 - confidence) * portfolio_vol
        return var

    def _calculate_expected_returns(self, symbols: List[str], 
                                  technical_data: Dict[str, TechnicalAnalysis]) -> np.array:
        """Enhanced expected returns calculation"""
        returns = []
        for symbol in symbols:
            if symbol in technical_data:
                ta = technical_data[symbol]

                # Multi-factor expected return model
                momentum_score = (ta.rsi - 50) / 50  # Momentum factor
                mean_reversion_score = -abs(ta.rsi - 50) / 50  # Mean reversion factor
                volatility_score = -ta.volatility  # Volatility penalty

                # Technical indicators composite score
                macd_score = np.tanh(ta.macd_histogram / 5)  # Bounded MACD score
                bb_score = -abs((ta.bb_upper + ta.bb_lower) / 2 - ta.bb_middle) / ta.bb_middle  # BB position

                expected_return = (0.3 * momentum_score + 
                                 0.2 * mean_reversion_score + 
                                 0.2 * volatility_score +
                                 0.15 * macd_score +
                                 0.15 * bb_score) * 0.1  # Scale to reasonable return
            else:
                expected_return = 0.0
            returns.append(expected_return)

        return np.array(returns)

    def _calculate_covariance_matrix(self, symbols: List[str]) -> np.array:
        """Enhanced covariance matrix calculation"""
        n_assets = len(symbols)

        # Base volatilities (different for each asset class)
        base_volatilities = {
            'AAPL': 0.25, 'GOOGL': 0.28, 'MSFT': 0.22, 
            'TSLA': 0.45, 'NVDA': 0.35
        }

        # Create covariance matrix
        cov_matrix = np.eye(n_assets)

        for i, symbol_i in enumerate(symbols):
            vol_i = base_volatilities.get(symbol_i, 0.25)
            cov_matrix[i, i] = vol_i ** 2

            for j, symbol_j in enumerate(symbols):
                if i != j:
                    vol_j = base_volatilities.get(symbol_j, 0.25)

                    # Sector-based correlations
                    if self._same_sector(symbol_i, symbol_j):
                        correlation = random.uniform(0.4, 0.7)  # Higher correlation within sector
                    else:
                        correlation = random.uniform(0.1, 0.4)  # Lower correlation across sectors

                    cov_matrix[i, j] = correlation * vol_i * vol_j

        return cov_matrix

    def _same_sector(self, symbol1: str, symbol2: str) -> bool:
        """Check if two symbols are in the same sector"""
        tech_stocks = ['AAPL', 'GOOGL', 'MSFT', 'NVDA']
        auto_stocks = ['TSLA']

        return ((symbol1 in tech_stocks and symbol2 in tech_stocks) or
                (symbol1 in auto_stocks and symbol2 in auto_stocks))

    def _calculate_current_weights(self, positions: Dict[str, Position]) -> Dict[str, float]:
        """Calculate current portfolio weights"""
        total_value = sum(pos.quantity * pos.current_price for pos in positions.values() if pos.quantity > 0)
        if total_value == 0:
            return {}

        return {symbol: (pos.quantity * pos.current_price) / total_value 
                for symbol, pos in positions.items() if pos.quantity > 0}

    def _calculate_concentration_risk(self, weights: np.array) -> float:
        """Calculate concentration risk (Herfindahl index)"""
        return np.sum(weights ** 2)

class TechnicalIndicators:
    """Enhanced technical analysis indicators calculator"""

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

        # Use provided high/low or simulate
        high = getattr(market_data, 'high_24h', market_data.price * (1 + random.uniform(0, 0.02)))
        low = getattr(market_data, 'low_24h', market_data.price * (1 - random.uniform(0, 0.02)))

        self.high_history[symbol].append(high)
        self.low_history[symbol].append(low)

        # Update support/resistance levels
        self._update_support_resistance(symbol)

        # Keep only last 200 periods for efficiency
        if len(self.price_history[symbol]) > 200:
            self.price_history[symbol] = self.price_history[symbol][-200:]
            self.volume_history[symbol] = self.volume_history[symbol][-200:]
            self.high_history[symbol] = self.high_history[symbol][-200:]
            self.low_history[symbol] = self.low_history[symbol][-200:]

    def _update_support_resistance(self, symbol: str):
        """Update support and resistance levels"""
        if len(self.price_history[symbol]) < 20:
            return

        prices = np.array(self.price_history[symbol][-50:])  # Last 50 periods
        highs = np.array(self.high_history[symbol][-50:])
        lows = np.array(self.low_history[symbol][-50:])

        # Simple support/resistance calculation
        resistance = np.percentile(highs, 90)
        support = np.percentile(lows, 10)

        self.support_resistance_levels[symbol] = {
            'support': support,
            'resistance': resistance
        }

    def calculate_indicators(self, symbol: str) -> Optional[TechnicalAnalysis]:
        """Enhanced technical indicators calculation"""
        if symbol not in self.price_history or len(self.price_history[symbol]) < 10:
            return None

        prices = np.array(self.price_history[symbol])
        volumes = np.array(self.volume_history[symbol])
        highs = np.array(self.high_history[symbol])
        lows = np.array(self.low_history[symbol])

        try:
            # Moving Averages
            sma_20 = self._sma(prices, min(10, len(prices)))
            sma_50 = self._sma(prices, min(20, len(prices)))
            ema_12 = self._ema(prices, min(5, len(prices)))
            ema_26 = self._ema(prices, min(10, len(prices)))

            # RSI
            rsi = self._calculate_rsi(prices, 14)

            # MACD
            macd, macd_signal, macd_histogram = self._calculate_macd(prices)

            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(prices, 20, 2)

            # Volume Analysis
            volume_sma = self._sma(volumes, 20)

            # ATR (Average True Range)
            atr = self._calculate_atr(highs, lows, prices, 14)

            # Fibonacci Levels
            fibonacci_levels = self._calculate_fibonacci_levels(prices)

            # Stochastic Oscillator
            stochastic_k, stochastic_d = self._calculate_stochastic(highs, lows, prices, 14, 3)

            # Williams %R
            williams_r = self._calculate_williams_r(highs, lows, prices, 14)

            # Volatility
            volatility = self._calculate_volatility(prices)

            # Support/Resistance
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
            logging.error(f"Error calculating indicators for {symbol}: {e}")
            return None

    def _calculate_volatility(self, prices: np.array, window: int = 20) -> float:
        """Calculate price volatility"""
        if len(prices) < window + 1:
            return 0.0

        returns = np.diff(prices[-window-1:]) / prices[-window-1:-1]
        return np.std(returns) * np.sqrt(252)  # Annualized volatility

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
            price_series = pd.Series(prices)
            rsi_series = ta.rsi(price_series, length=period)
            return rsi_series.iloc[-1] if not pd.isna(rsi_series.iloc[-1]) else 50.0

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices: np.array) -> tuple:
        """MACD (Moving Average Convergence Divergence)"""
        if len(prices) < 26:
            return 0.0, 0.0, 0.0

        if PANDAS_TA_AVAILABLE:
            price_series = pd.Series(prices)
            macd_data = ta.macd(price_series, fast=12, slow=26, signal=9)
            if macd_data is not None and len(macd_data.columns) >= 3:
                macd_line = macd_data.iloc[-1, 0]
                signal_line = macd_data.iloc[-1, 1]
                histogram = macd_data.iloc[-1, 2]
                return (
                    macd_line if not pd.isna(macd_line) else 0.0,
                    signal_line if not pd.isna(signal_line) else 0.0,
                    histogram if not pd.isna(histogram) else 0.0
                )

        ema_12 = self._ema(prices, 12)
        ema_26 = self._ema(prices, 26)
        macd = ema_12 - ema_26

        signal = macd * 0.9
        histogram = macd - signal

        return macd, signal, histogram

    def _calculate_bollinger_bands(self, prices: np.array, period: int = 20, std_dev: int = 2) -> tuple:
        """Bollinger Bands"""
        if len(prices) < period:
            price = prices[-1]
            return price * 1.02, price, price * 0.98

        if PANDAS_TA_AVAILABLE:
            price_series = pd.Series(prices)
            bb_data = ta.bbands(price_series, length=period, std=std_dev)
            if bb_data is not None and len(bb_data.columns) >= 3:
                bb_lower = bb_data.iloc[-1, 0]
                bb_middle = bb_data.iloc[-1, 1]
                bb_upper = bb_data.iloc[-1, 2]
                return (
                    bb_upper if not pd.isna(bb_upper) else prices[-1] * 1.02,
                    bb_middle if not pd.isna(bb_middle) else prices[-1],
                    bb_lower if not pd.isna(bb_lower) else prices[-1] * 0.98
                )

        sma = self._sma(prices, period)
        std = np.std(prices[-period:])

        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)

        return upper, sma, lower

    def _calculate_atr(self, highs: np.array, lows: np.array, closes: np.array, period: int = 14) -> float:
        """Average True Range"""
        if len(highs) < period + 1:
            return abs(highs[-1] - lows[-1])

        true_ranges = []
        for i in range(1, len(highs)):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            true_ranges.append(max(tr1, tr2, tr3))

        return np.mean(true_ranges[-period:])

    def _calculate_fibonacci_levels(self, prices: np.array) -> Dict[str, float]:
        """Fibonacci Retracement Levels"""
        if len(prices) < 20:
            price = prices[-1]
            return {'23.6%': price, '38.2%': price, '50%': price, '61.8%': price, '78.6%': price}

        high = np.max(prices[-50:])
        low = np.min(prices[-50:])
        diff = high - low

        return {
            '23.6%': high - (diff * 0.236),
            '38.2%': high - (diff * 0.382),
            '50%': high - (diff * 0.5),
            '61.8%': high - (diff * 0.618),
            '78.6%': high - (diff * 0.786)
        }

    def _calculate_stochastic(self, highs: np.array, lows: np.array, closes: np.array, 
                            k_period: int = 14, d_period: int = 3) -> tuple:
        """Stochastic Oscillator"""
        if len(highs) < k_period:
            return 50.0, 50.0

        lowest_low = np.min(lows[-k_period:])
        highest_high = np.max(highs[-k_period:])

        if highest_high == lowest_low:
            k_percent = 50.0
        else:
            k_percent = ((closes[-1] - lowest_low) / (highest_high - lowest_low)) * 100

        d_percent = k_percent

        return k_percent, d_percent

    def _calculate_williams_r(self, highs: np.array, lows: np.array, closes: np.array, period: int = 14) -> float:
        """Williams %R"""
        if len(highs) < period:
            return -50.0

        highest_high = np.max(highs[-period:])
        lowest_low = np.min(lows[-period:])

        if highest_high == lowest_low:
            return -50.0

        williams_r = ((highest_high - closes[-1]) / (highest_high - lowest_low)) * -100
        return williams_r

class BaseAgent(ABC):
    """Base class for all trading agents"""

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)
        self.is_active = False

    @abstractmethod
    def process(self, data: Any) -> Any:
        """Main processing method for the agent"""
        pass

    def start(self):
        """Start the agent"""
        self.is_active = True
        self.logger.info(f"{self.name} started")

    def stop(self):
        """Stop the agent"""
        self.is_active = False
        self.logger.info(f"{self.name} stopped")

class MarketAnalystAgent(BaseAgent):
    """Enhanced agent responsible for market analysis and signal generation"""

    def __init__(self):
        super().__init__("MarketAnalyst")
        self.market_data = {}
        self.signals = []
        self.technical_indicators = TechnicalIndicators()
        self.ml_engine = MLPredictionEngine() if ML_AVAILABLE else None
        self.sentiment_analyzer = SentimentAnalyzer()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.digital_brain = DigitalBrain() if DIGITAL_BRAIN_AVAILABLE else None
        self.last_price = {}
        self.signal_history = {}

        # Digital Brain integration tracking
        self.brain_patterns_used = 0
        self.brain_insights_generated = 0
        self.brain_knowledge_queries = 0
        self.brain_learning_events = 0

    def process(self, market_data: MarketData) -> List[TradingSignal]:
        """Enhanced market analysis with stop-loss suggestions and Digital Brain integration"""
        try:
            self.market_data[market_data.symbol] = market_data

            # Update technical indicators with new data
            self.technical_indicators.update_data(market_data)

            # Digital Brain: Process market event and learn patterns
            brain_insights = []
            if self.digital_brain:
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
                    'bb_position': ((market_data.price - ta.bb_middle) / (ta.bb_upper - ta.bb_lower)) if ta and ta.bb_upper != ta.bb_lower else 0,
                    'volume_ratio': market_data.volume / ta.volume_sma if ta and ta.volume_sma > 0 else 1,
                    'sentiment': sentiment.sentiment_score if sentiment else 0,
                    'support': ta.support_level if ta else 0,
                    'resistance': ta.resistance_level if ta else 0
                }

                # Add outcome if we have previous price
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

            # Update ML training data if we have previous price
            if market_data.symbol in self.last_price and self.ml_engine:
                price_change = market_data.price - self.last_price[market_data.symbol]
                ta = self.technical_indicators.calculate_indicators(market_data.symbol)
                if ta:
                    sentiment = self.sentiment_analyzer.analyze_sentiment(market_data.symbol)
                    features = self.ml_engine.prepare_features(market_data.symbol, ta, market_data, sentiment)
                    self.ml_engine.update_training_data(market_data.symbol, features, price_change)

                    # Train model periodically
                    if len(self.ml_engine.feature_history.get(market_data.symbol, [])) % 25 == 0:
                        self.ml_engine.train_model(market_data.symbol)

            # Generate enhanced signals with stop-loss levels
            signals = []

            # 1. Traditional technical analysis
            technical_signals = self._analyze_market_technical(market_data)
            signals.extend(technical_signals)

            # 2. Machine learning predictions
            if self.ml_engine:
                ml_signals = self._analyze_market_ml(market_data)
                signals.extend(ml_signals)

            # 3. Sentiment-based signals
            sentiment_signals = self._analyze_market_sentiment(market_data)
            signals.extend(sentiment_signals)

            # 4. Digital Brain pattern recognition
            if self.digital_brain:
                brain_signals = self._analyze_market_brain_patterns(market_data)
                signals.extend(brain_signals)

            # 5. Enhanced ensemble signal combination
            final_signals = self._combine_signals_enhanced(signals, market_data.symbol)

            self.signals.extend(final_signals)
            self.last_price[market_data.symbol] = market_data.price

            if final_signals:
                self.logger.info(f"Generated {len(final_signals)} enhanced signals for {market_data.symbol}")
                for signal in final_signals:
                    sl_info = f" | SL: ${signal.stop_loss_price:.2f}" if signal.stop_loss_price else ""
                    tp_info = f" | TP: ${signal.take_profit_price:.2f}" if signal.take_profit_price else ""
                    self.logger.info(f"  {signal.action} signal: {signal.reason} (Confidence: {signal.confidence:.2f}){sl_info}{tp_info}")

            return final_signals

        except Exception as e:
            self.logger.error(f"Error processing market data: {e}")
            return []

    def _analyze_market_technical(self, data: MarketData) -> List[TradingSignal]:
        """Enhanced technical analysis with stop-loss calculations"""
        signals = []

        ta = self.technical_indicators.calculate_indicators(data.symbol)
        if not ta:
            return signals

        current_price = data.price
        confidence_factors = []
        signal_reasons = []
        signal_type = None

        # Enhanced technical analysis with multiple strategies
        strategies = [
            self._momentum_strategy(data, ta),
            self._mean_reversion_strategy(data, ta),
            self._breakout_strategy(data, ta),
            self._trend_following_strategy(data, ta)
        ]

        # Combine strategy results
        for strategy_signal in strategies:
            if strategy_signal:
                signals.extend(strategy_signal)

        return signals

    def _momentum_strategy(self, data: MarketData, ta: TechnicalAnalysis) -> List[TradingSignal]:
        """Momentum-based strategy with stop-loss"""
        signals = []
        current_price = data.price

        # Momentum indicators
        rsi_momentum = (ta.rsi - 50) / 50
        macd_momentum = 1 if ta.macd > ta.macd_signal else -1
        price_momentum = 1 if current_price > ta.sma_20 else -1

        momentum_score = (rsi_momentum + macd_momentum + price_momentum) / 3

        if abs(momentum_score) > 0.3:  # Significant momentum
            action = 'BUY' if momentum_score > 0 else 'SELL'
            confidence = min(abs(momentum_score), 0.8)

            # Calculate stop-loss based on ATR
            stop_loss_distance = ta.atr * 2
            if action == 'BUY':
                stop_loss_price = current_price - stop_loss_distance
                take_profit_price = current_price + (stop_loss_distance * 2)  # 2:1 risk/reward
            else:
                stop_loss_price = current_price + stop_loss_distance
                take_profit_price = current_price - (stop_loss_distance * 2)

            signal = TradingSignal(
                symbol=data.symbol,
                action=action,
                confidence=confidence,
                reason=f"Momentum strategy: score={momentum_score:.2f}",
                timestamp=datetime.now(),
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                signal_strength="STRONG" if abs(momentum_score) > 0.6 else "MEDIUM"
            )
            signals.append(signal)

        return signals

    def _mean_reversion_strategy(self, data: MarketData, ta: TechnicalAnalysis) -> List[TradingSignal]:
        """Mean reversion strategy with stop-loss"""
        signals = []
        current_price = data.price

        # Mean reversion indicators
        bb_position = (current_price - ta.bb_middle) / (ta.bb_upper - ta.bb_lower)
        rsi_extreme = 1 if ta.rsi < 30 else (-1 if ta.rsi > 70 else 0)

        if abs(bb_position) > 0.8 and rsi_extreme != 0:  # Price at extremes
            action = 'BUY' if bb_position < -0.8 and rsi_extreme == 1 else 'SELL'
            confidence = min(abs(bb_position) + abs(rsi_extreme) * 0.3, 0.9)

            # Tighter stop-loss for mean reversion
            stop_loss_distance = ta.atr * 1.5
            if action == 'BUY':
                stop_loss_price = current_price - stop_loss_distance
                take_profit_price = ta.bb_middle  # Target mean
            else:
                stop_loss_price = current_price + stop_loss_distance
                take_profit_price = ta.bb_middle

            signal = TradingSignal(
                symbol=data.symbol,
                action=action,
                confidence=confidence,
                reason=f"Mean reversion: BB pos={bb_position:.2f}, RSI={ta.rsi:.1f}",
                timestamp=datetime.now(),
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                signal_strength="MEDIUM"
            )
            signals.append(signal)

        return signals

    def _breakout_strategy(self, data: MarketData, ta: TechnicalAnalysis) -> List[TradingSignal]:
        """Breakout strategy with stop-loss"""
        signals = []
        current_price = data.price

        # Volume-confirmed breakouts
        volume_surge = data.volume > ta.volume_sma * 1.5

        if volume_surge:
            if current_price > ta.resistance_level and ta.resistance_level > 0:
                # Upward breakout
                stop_loss_price = ta.resistance_level * 0.98  # Just below resistance
                take_profit_price = current_price + (current_price - stop_loss_price) * 2

                signal = TradingSignal(
                    symbol=data.symbol,
                    action='BUY',
                    confidence=0.7,
                    reason=f"Upward breakout above ${ta.resistance_level:.2f} with volume",
                    timestamp=datetime.now(),
                    stop_loss_price=stop_loss_price,
                    take_profit_price=take_profit_price,
                    signal_strength="STRONG"
                )
                signals.append(signal)

            elif current_price < ta.support_level and ta.support_level > 0:
                # Downward breakout
                stop_loss_price = ta.support_level * 1.02  # Just above support
                take_profit_price = current_price - (stop_loss_price - current_price) * 2

                signal = TradingSignal(
                    symbol=data.symbol,
                    action='SELL',
                    confidence=0.7,
                    reason=f"Downward breakout below ${ta.support_level:.2f} with volume",
                    timestamp=datetime.now(),
                    stop_loss_price=stop_loss_price,
                    take_profit_price=take_profit_price,
                    signal_strength="STRONG"
                )
                signals.append(signal)

        return signals

    def _trend_following_strategy(self, data: MarketData, ta: TechnicalAnalysis) -> List[TradingSignal]:
        """Trend following strategy with stop-loss"""
        signals = []
        current_price = data.price

        # Multiple timeframe trend alignment
        short_trend = 1 if ta.ema_12 > ta.ema_26 else -1
        medium_trend = 1 if ta.sma_20 > ta.sma_50 else -1
        long_trend = 1 if current_price > ta.sma_50 else -1

        trend_strength = (short_trend + medium_trend + long_trend) / 3

        if abs(trend_strength) > 0.6:  # Strong trend alignment
            action = 'BUY' if trend_strength > 0 else 'SELL'
            confidence = min(abs(trend_strength), 0.8)

            # Trend-based stop-loss
            if action == 'BUY':
                stop_loss_price = min(ta.ema_26, ta.sma_20) * 0.98
                take_profit_price = current_price + (current_price - stop_loss_price) * 3
            else:
                stop_loss_price = max(ta.ema_26, ta.sma_20) * 1.02
                take_profit_price = current_price - (stop_loss_price - current_price) * 3

            signal = TradingSignal(
                symbol=data.symbol,
                action=action,
                confidence=confidence,
                reason=f"Trend following: strength={trend_strength:.2f}",
                timestamp=datetime.now(),
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                signal_strength="STRONG" if abs(trend_strength) > 0.8 else "MEDIUM"
            )
            signals.append(signal)

        return signals

    def _analyze_market_ml(self, data: MarketData) -> List[TradingSignal]:
        """Enhanced ML signals with stop-loss calculation"""
        signals = []

        if not self.ml_engine:
            return signals

        try:
            ta = self.technical_indicators.calculate_indicators(data.symbol)
            if not ta:
                return signals

            sentiment = self.sentiment_analyzer.analyze_sentiment(data.symbol)
            features = self.ml_engine.prepare_features(data.symbol, ta, data, sentiment)
            prediction = self.ml_engine.predict(data.symbol, features)

            if prediction and prediction.prediction in ['BUY', 'SELL'] and prediction.confidence > 0.65:
                # ML-based stop-loss using risk score
                risk_multiplier = 1 + prediction.risk_score
                stop_loss_distance = ta.atr * risk_multiplier * 2

                if prediction.prediction == 'BUY':
                    stop_loss_price = data.price - stop_loss_distance
                    take_profit_price = data.price + (stop_loss_distance * 1.5)
                else:
                    stop_loss_price = data.price + stop_loss_distance
                    take_profit_price = data.price - (stop_loss_distance * 1.5)

                signal = TradingSignal(
                    symbol=data.symbol,
                    action=prediction.prediction,
                    confidence=prediction.confidence * 0.85,  # Slight discount for ML
                    reason=f"ML prediction (accuracy: {prediction.model_accuracy:.2f}, risk: {prediction.risk_score:.2f})",
                    timestamp=datetime.now(),
                    stop_loss_price=stop_loss_price,
                    take_profit_price=take_profit_price,
                    signal_strength="STRONG" if prediction.confidence > 0.8 else "MEDIUM"
                )
                signals.append(signal)

        except Exception as e:
            self.logger.error(f"Error in ML analysis for {data.symbol}: {e}")

        return signals

    def _analyze_market_sentiment(self, data: MarketData) -> List[TradingSignal]:
        """Enhanced sentiment signals with stop-loss"""
        signals = []

        try:
            sentiment = self.sentiment_analyzer.analyze_sentiment(data.symbol)

            # Enhanced sentiment signal generation
            if abs(sentiment.sentiment_score) > 0.3:
                # Consider sentiment trend
                trend_multiplier = 1.2 if sentiment.sentiment_trend == 'IMPROVING' else (0.8 if sentiment.sentiment_trend == 'DECLINING' else 1.0)
                adjusted_confidence = min(abs(sentiment.sentiment_score) * trend_multiplier, 0.6)

                if adjusted_confidence > 0.35:  # Threshold for sentiment signals
                    action = 'BUY' if sentiment.sentiment_score > 0 else 'SELL'

                    # Conservative stop-loss for sentiment-based signals
                    ta = self.technical_indicators.calculate_indicators(data.symbol)
                    if ta:
                        stop_loss_distance = ta.atr * 2.5  # Wider stop for sentiment

                        if action == 'BUY':
                            stop_loss_price = data.price - stop_loss_distance
                            take_profit_price = data.price + stop_loss_distance
                        else:
                            stop_loss_price = data.price + stop_loss_distance
                            take_profit_price = data.price - stop_loss_distance

                        signal = TradingSignal(
                            symbol=data.symbol,
                            action=action,
                            confidence=adjusted_confidence,
                            reason=f"Sentiment analysis: {sentiment.sentiment_label} ({sentiment.sentiment_score:.2f}, trend: {sentiment.sentiment_trend})",
                            timestamp=datetime.now(),
                            stop_loss_price=stop_loss_price,
                            take_profit_price=take_profit_price,
                            signal_strength="WEAK"
                        )
                        signals.append(signal)

        except Exception as e:
            self.logger.error(f"Error in sentiment analysis for {data.symbol}: {e}")

        return signals

    def _combine_signals_enhanced(self, signals: List[TradingSignal], symbol: str) -> List[TradingSignal]:
        """Enhanced signal combination with weighted voting"""
        if not signals:
            return []

        try:
            # Group signals by action
            buy_signals = [s for s in signals if s.action == 'BUY']
            sell_signals = [s for s in signals if s.action == 'SELL']

            combined_signals = []

            # Enhanced combination for BUY signals
            if buy_signals:
                # Weighted average based on signal strength
                weights = {'STRONG': 1.5, 'MEDIUM': 1.0, 'WEAK': 0.5}
                total_weight = sum(weights.get(s.signal_strength, 1.0) for s in buy_signals)
                weighted_confidence = sum(s.confidence * weights.get(s.signal_strength, 1.0) for s in buy_signals) / total_weight

                # Best stop-loss from all signals (most conservative)
                stop_losses = [s.stop_loss_price for s in buy_signals if s.stop_loss_price]
                best_stop_loss = max(stop_losses) if stop_losses else None

                # Best take-profit (most conservative)
                take_profits = [s.take_profit_price for s in buy_signals if s.take_profit_price]
                best_take_profit = min(take_profits) if take_profits else None

                # Boost confidence for multiple agreeing signals
                if len(buy_signals) > 1:
                    weighted_confidence = min(weighted_confidence * 1.15, 0.95)

                combined_reason = f"Ensemble BUY ({len(buy_signals)} signals): " + "; ".join([s.reason for s in buy_signals])

                combined_signals.append(TradingSignal(
                    symbol=symbol,
                    action='BUY',
                    confidence=weighted_confidence,
                    reason=combined_reason,
                    timestamp=datetime.now(),
                    stop_loss_price=best_stop_loss,
                    take_profit_price=best_take_profit,
                    signal_strength="STRONG" if weighted_confidence > 0.7 else "MEDIUM"
                ))

            # Enhanced combination for SELL signals
            if sell_signals:
                weights = {'STRONG': 1.5, 'MEDIUM': 1.0, 'WEAK': 0.5}
                total_weight = sum(weights.get(s.signal_strength, 1.0) for s in sell_signals)
                weighted_confidence = sum(s.confidence * weights.get(s.signal_strength, 1.0) for s in sell_signals) / total_weight

                # Best stop-loss for SELL (most conservative)
                stop_losses = [s.stop_loss_price for s in sell_signals if s.stop_loss_price]
                best_stop_loss = min(stop_losses) if stop_losses else None

                # Best take-profit for SELL
                take_profits = [s.take_profit_price for s in sell_signals if s.take_profit_price]
                best_take_profit = max(take_profits) if take_profits else None

                if len(sell_signals) > 1:
                    weighted_confidence = min(weighted_confidence * 1.15, 0.95)

                combined_reason = f"Ensemble SELL ({len(sell_signals)} signals): " + "; ".join([s.reason for s in sell_signals])

                combined_signals.append(TradingSignal(
                    symbol=symbol,
                    action='SELL',
                    confidence=weighted_confidence,
                    reason=combined_reason,
                    timestamp=datetime.now(),
                    stop_loss_price=best_stop_loss,
                    take_profit_price=best_take_profit,
                    signal_strength="STRONG" if weighted_confidence > 0.7 else "MEDIUM"
                ))

            # If conflicting signals, choose the stronger one
            if len(combined_signals) > 1:
                combined_signals = [max(combined_signals, key=lambda x: x.confidence)]

            return combined_signals

        except Exception as e:
            self.logger.error(f"Error combining signals for {symbol}: {e}")
            return signals[:1]

    def _analyze_market_brain_patterns(self, data: MarketData) -> List[TradingSignal]:
        """Enhanced Digital Brain pattern-based signals with comprehensive knowledge integration"""
        signals = []

        if not self.digital_brain:
            return signals

        try:
            # Prepare comprehensive market context for pattern recognition
            ta = self.technical_indicators.calculate_indicators(data.symbol)
            if not ta:
                return signals

            sentiment = self.sentiment_analyzer.analyze_sentiment(data.symbol)

            # Enhanced context with market regime detection
            price_change_5min = 0
            if data.symbol in self.last_price:
                price_change_5min = (data.price - self.last_price[data.symbol]) / self.last_price[data.symbol]

            current_context = {
                'symbol': data.symbol,
                'price': data.price,
                'volume': data.volume,
                'rsi': ta.rsi,
                'macd': ta.macd,
                'macd_signal': ta.macd_signal,
                'macd_histogram': ta.macd_histogram,
                'bb_position': (data.price - ta.bb_middle) / (ta.bb_upper - ta.bb_lower) if ta.bb_upper != ta.bb_lower else 0,
                'bb_width': (ta.bb_upper - ta.bb_lower) / ta.bb_middle if ta.bb_middle > 0 else 0,
                'volume_ratio': data.volume / ta.volume_sma if ta.volume_sma > 0 else 1,
                'previous_price': self.last_price.get(data.symbol, data.price),
                'price_change_5min': price_change_5min,
                'support': ta.support_level,
                'resistance': ta.resistance_level,
                'atr': ta.atr,
                'atr_ratio': ta.atr / data.price if data.price > 0 else 0,
                'volatility': ta.volatility,
                'stochastic_k': ta.stochastic_k,
                'stochastic_d': ta.stochastic_d,
                'williams_r': ta.williams_r,
                'avg_volume': ta.volume_sma,
                'sentiment': sentiment.sentiment_score,
                'sentiment_trend': sentiment.sentiment_trend,
                'time_of_day': datetime.now().hour,
                'market_hours': 9 <= datetime.now().hour <= 16,
                'fibonacci_levels': ta.fibonacci_levels,
                'ema_12': ta.ema_12,
                'ema_26': ta.ema_26,
                'sma_20': ta.sma_20,
                'sma_50': ta.sma_50
            }

            # Market regime detection
            if ta.volatility > 0.3:
                current_context['market_regime'] = 'high_volatility'
            elif ta.volatility < 0.1:
                current_context['market_regime'] = 'low_volatility'
            elif ta.rsi > 70:
                current_context['market_regime'] = 'overbought'
            elif ta.rsi < 30:
                current_context['market_regime'] = 'oversold'
            else:
                current_context['market_regime'] = 'normal'

            # Recognize patterns using the enhanced Digital Brain
            recognized_patterns = self.digital_brain.pattern_engine.recognize_patterns(
                data.symbol, current_context
            )

            # Generate signals from recognized patterns with enhanced logic
            for pattern, confidence in recognized_patterns:
                if confidence > 0.55:  # Lower threshold for more signals

                    # Enhanced action determination based on pattern analysis
                    action = 'HOLD'
                    reasoning = f"Digital Brain pattern: {pattern.pattern_type}"

                    # Pattern type analysis
                    pattern_lower = pattern.pattern_type.lower()
                    if any(keyword in pattern_lower for keyword in ['bullish', 'breakout', 'uptrend', 'buy']):
                        action = 'BUY'
                    elif any(keyword in pattern_lower for keyword in ['bearish', 'breakdown', 'downtrend', 'sell']):
                        action = 'SELL'
                    elif pattern.outcomes.get('price_increase', False) and pattern.success_rate > 0.6:
                        action = 'BUY'
                    elif pattern.outcomes.get('price_decrease', False) and pattern.success_rate > 0.6:
                        action = 'SELL'

                    # Market regime adjustment
                    regime = current_context['market_regime']
                    if regime == 'high_volatility' and confidence < 0.7:
                        action = 'HOLD'  # More conservative in volatile markets
                    elif regime == 'oversold' and action == 'SELL':
                        action = 'HOLD'  # Don't sell in oversold conditions
                    elif regime == 'overbought' and action == 'BUY':
                        action = 'HOLD'  # Don't buy in overbought conditions

                    if action != 'HOLD':
                        # Enhanced stop-loss calculation based on pattern characteristics
                        base_atr_multiplier = 2.0

                        # Adjust based on pattern success rate
                        success_adjustment = 1.5 - pattern.success_rate

                        # Adjust based on market regime
                        regime_multiplier = {
                            'high_volatility': 2.5,
                            'low_volatility': 1.5,
                            'overbought': 2.0,
                            'oversold': 2.0,
                            'normal': 2.0
                        }.get(regime, 2.0)

                        final_multiplier = base_atr_multiplier * (1 + success_adjustment) * regime_multiplier / 2
                        stop_loss_distance = ta.atr * final_multiplier

                        if action == 'BUY':
                            stop_loss_price = data.price - stop_loss_distance
                            # Take profit at resistance or 2:1 ratio
                            if ta.resistance_level > data.price:
                                take_profit_price = min(ta.resistance_level * 0.98, data.price + (stop_loss_distance * 2))
                            else:
                                take_profit_price = data.price + (stop_loss_distance * 2)
                        else:
                            stop_loss_price = data.price + stop_loss_distance
                            # Take profit at support or 2:1 ratio
                            if ta.support_level > 0 and ta.support_level < data.price:
                                take_profit_price = max(ta.support_level * 1.02, data.price - (stop_loss_distance * 2))
                            else:
                                take_profit_price = data.price - (stop_loss_distance * 2)

                        # Multi-factor confidence adjustment
                        sample_factor = min(pattern.sample_size / 20, 1.0)  # Better with more samples
                        regime_factor = 0.9 if regime == 'high_volatility' else 1.0
                        time_factor = 1.1 if current_context['market_hours'] else 0.8

                        adjusted_confidence = (confidence * pattern.success_rate * 
                                             sample_factor * regime_factor * time_factor)

                        # Enhanced reasoning
                        detailed_reason = (f"Brain pattern: {pattern.pattern_type} "
                                         f"(success: {pattern.success_rate:.1%}, "
                                         f"samples: {pattern.sample_size}, "
                                         f"regime: {regime}, "
                                         f"confidence: {confidence:.2f})")

                        signal = TradingSignal(
                            symbol=data.symbol,
                            action=action,
                            confidence=min(adjusted_confidence, 0.95),  # Cap at 95%
                            reason=detailed_reason,
                            timestamp=datetime.now(),
                            stop_loss_price=stop_loss_price,
                            take_profit_price=take_profit_price,
                            signal_strength="STRONG" if adjusted_confidence > 0.75 else ("MEDIUM" if adjusted_confidence > 0.55 else "WEAK")
                        )
                        signals.append(signal)
                        self.brain_insights_generated += 1

            # Enhanced knowledge querying when no strong patterns are found
            if len(recognized_patterns) == 0 or max([conf for _, conf in recognized_patterns], default=0) < 0.7:
                self.brain_knowledge_queries += 1

                # Query for regime-specific insights
                regime_query = f"What {current_context['market_regime']} patterns exist for {data.symbol}?"
                query_result = self.digital_brain.query_brain(regime_query, current_context)

                if query_result.get('confidence', 0) > 0.5 and query_result.get('insights'):
                    # Generate insight-based signal
                    insight_confidence = query_result['confidence'] * 0.6  # Conservative multiplier

                    # Determine action from insights
                    insights_text = ' '.join(query_result['insights']).lower()
                    if any(word in insights_text for word in ['buy', 'bullish', 'upward', 'positive']):
                        insight_action = 'BUY'
                    elif any(word in insights_text for word in ['sell', 'bearish', 'downward', 'negative']):
                        insight_action = 'SELL'
                    else:
                        insight_action = 'HOLD'

                    if insight_action != 'HOLD' and insight_confidence > 0.3:
                        # Conservative stop-loss for insight-based signals
                        stop_distance = ta.atr * 3.0
                        if insight_action == 'BUY':
                            stop_loss_price = data.price - stop_distance
                            take_profit_price = data.price + stop_distance
                        else:
                            stop_loss_price = data.price + stop_distance
                            take_profit_price = data.price - stop_distance

                        insight_signal = TradingSignal(
                            symbol=data.symbol,
                            action=insight_action,
                            confidence=insight_confidence,
                            reason=f"Brain insight ({current_context['market_regime']}): {query_result['insights'][0][:60]}...",
                            timestamp=datetime.now(),
                            stop_loss_price=stop_loss_price,
                            take_profit_price=take_profit_price,
                            signal_strength="WEAK"
                        )
                        signals.append(insight_signal)

        except Exception as e:
            self.logger.error(f"Error in enhanced Digital Brain pattern analysis for {data.symbol}: {e}")

        return signals

    
    def _analyze_market_brain_patterns(self, data: MarketData) -> List[TradingSignal]:
        if not self.digital_brain:
            return []

        return self.digital_brain.analyze_market_patterns(data)

    def _validate_signal_and_risk(self, signal: TradingSignal) -> bool:
        if signal.confidence < 0.8:
            self.logger.warning(f"Drawdown limit exceeded: {self.risk_metrics['max_drawdown']:.2%}")
            return False

        # Volatility-adjusted validation
        volatility_threshold = 0.4  # Increased tolerance
        if self.risk_metrics['volatility'] > volatility_threshold:
            # Require higher confidence for high volatility
            required_confidence = 0.5 + (self.risk_metrics['volatility'] - volatility_threshold) * 2
            if signal.confidence < required_confidence:
                self.logger.warning("High volatility - insufficient signal confidence")
                return False

        # Enhanced position count management
        active_positions = len([p for p in positions.values() if p.quantity > 0])
        if active_positions >= self.position_limits:
            # Allow if we're closing a position or signal is very strong
            if signal.action == 'BUY' and signal.confidence < 0.85:
                self.logger.warning("Maximum position count reached")
                return False

        # Stop-loss validation
        if not signal.stop_loss_price:
            self.logger.warning(f"No stop-loss provided for {signal.symbol} - adding protective stop")
            # This could be handled by setting a default stop-loss

        return True
    def _calculate_enhanced_position_size(self, signal: TradingSignal, positions: Dict[str, Position]) -> int:
        """Enhanced position sizing with stop-loss consideration"""
        estimated_price = 100.0

        # Base position sizing
        base_position_value = self.portfolio_value * self.max_position_size

        # Risk-adjusted sizing based on stop-loss distance
        if signal.stop_loss_price:
            stop_distance = abs(estimated_price - signal.stop_loss_price) / estimated_price
            risk_adjustment = max(0.3, 1 - stop_distance * 5)  # Reduce size for wide stops
        else:
            risk_adjustment = 0.7  # Conservative if no stop-loss

        # Volatility adjustment
        volatility_adjustment = max(0.4, 1 - self.risk_metrics['volatility'])

        # Signal strength adjustment
        if hasattr(signal, 'signal_strength'):
            strength_multiplier = {'STRONG': 1.3, 'MEDIUM': 1.0, 'WEAK': 0.7}.get(signal.signal_strength, 1.0)
        else:
            strength_multiplier = 1.0

        # Confidence adjustment (enhanced)
        confidence_adjustment = 0.5 + (signal.confidence * 0.5)

        # Portfolio heat adjustment (reduce size if many active positions)
        active_positions = len([p for p in positions.values() if p.quantity > 0])
        heat_adjustment = max(0.6, 1 - (active_positions / self.position_limits) * 0.4)

        # Kelly Criterion approximation
        estimated_win_rate = min(signal.confidence + 0.15, 0.9)
        kelly_fraction = estimated_win_rate - (1 - estimated_win_rate)
        kelly_fraction = max(0.1, min(kelly_fraction, 0.4))

        # Calculate final position size
        adjusted_position_value = (base_position_value * 
                                 risk_adjustment * 
                                 volatility_adjustment * 
                                 strength_multiplier *
                                 confidence_adjustment * 
                                 heat_adjustment *
                                 kelly_fraction)

        max_quantity = int(adjusted_position_value / estimated_price)

        self.logger.info(f"Enhanced position sizing for {signal.symbol}: "
                        f"base=${base_position_value:.0f}, "
                        f"risk_adj={risk_adjustment:.2f}, "
                        f"vol_adj={volatility_adjustment:.2f}, "
                        f"strength={strength_multiplier:.2f}, "
                        f"conf_adj={confidence_adjustment:.2f}, "
                        f"kelly={kelly_fraction:.2f}")

        return max(max_quantity, 1)

    def create_stop_loss_orders(self, position: Position) -> List[TradeOrder]:
        """Create stop-loss orders for a position"""
        orders = []

        if position.stop_loss_price:
            stop_order = self.stop_loss_manager.create_stop_loss_order(position, position.stop_loss_price)
            orders.append(stop_order)

        if position.trailing_stop_price:
            trail_distance = abs(position.current_price - position.trailing_stop_price)
            self.stop_loss_manager.create_trailing_stop(position, trail_distance)

        return orders

    def _update_risk_metrics(self, positions: Dict[str, Position]):
        """Enhanced risk metrics calculation"""
        try:
            # Calculate current portfolio value
            current_portfolio_value = self.portfolio_value
            for pos in positions.values():
                if pos.quantity != 0:
                    current_portfolio_value += pos.unrealized_pnl

            # Track portfolio history
            self.portfolio_history.append(current_portfolio_value)
            if len(self.portfolio_history) > 200:
                self.portfolio_history = self.portfolio_history[-200:]

            if len(self.portfolio_history) >= 10:
                returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]

                # Enhanced risk calculations
                self.risk_metrics['volatility'] = np.std(returns) * np.sqrt(252)
                self.risk_metrics['var_95'] = np.percentile(returns, 5) * current_portfolio_value

                # Expected Shortfall (CVaR)
                var_threshold = np.percentile(returns, 5)
                tail_losses = returns[returns <= var_threshold]
                if len(tail_losses) > 0:
                    self.risk_metrics['expected_shortfall'] = np.mean(tail_losses) * current_portfolio_value

                # Enhanced drawdown calculation
                peak = np.maximum.accumulate(self.portfolio_history)
                drawdown = (self.portfolio_history - peak) / peak
                self.risk_metrics['max_drawdown'] = abs(np.min(drawdown))

                # Enhanced Sharpe ratio
                risk_free_rate = 0.02 / 252
                excess_returns = returns - risk_free_rate
                if np.std(excess_returns) > 0:
                    self.risk_metrics['sharpe_ratio'] = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

        except Exception as e:
            self.logger.error(f"Error updating enhanced risk metrics: {e}")

    def get_risk_report(self) -> Dict[str, Any]:
        """Enhanced risk reporting"""
        return {
            'risk_metrics': self.risk_metrics.copy(),
            'portfolio_value': self.portfolio_value,
            'max_position_size': self.max_position_size,
            'max_portfolio_risk': self.max_portfolio_risk,
            'drawdown_limit': self.drawdown_limit,
            'position_limits': self.position_limits,
            'active_stops': len(self.stop_loss_manager.active_stops),
            'trailing_stops': len(self.stop_loss_manager.trailing_stops),
            'portfolio_periods': len(self.portfolio_history),
            'risk_alerts': len(self.risk_alerts)
        }

class TradingExecutorAgent(BaseAgent):
    """Enhanced agent with stop-loss order execution"""

    def __init__(self):
        super().__init__("TradingExecutor")
        self.positions = {}
        self.executed_orders = []
        self.cash_balance = 100000.0
        self.pending_orders = {}
        self.order_history = []

    def process(self, order: TradeOrder) -> bool:
        """Enhanced trade execution with stop-loss handling"""
        try:
            # Simulate market price
            market_price = random.uniform(95, 105)

            execution_successful = self._execute_trade_enhanced(order, market_price)

            if execution_successful:
                order.status = OrderStatus.FILLED
                order.price = market_price
                order.avg_fill_price = market_price
                order.filled_quantity = order.quantity
                self.executed_orders.append(order)
                self.order_history.append(order)

                # Update position with stop-loss information
                if order.symbol in self.positions:
                    position = self.positions[order.symbol]
                    if order.stop_loss_price:
                        position.stop_loss_price = order.stop_loss_price
                    if order.take_profit_price:
                        position.take_profit_price = order.take_profit_price

                self.logger.info(f"Enhanced order executed: {order.order_id} at ${market_price:.2f}")
                if order.stop_loss_price:
                    self.logger.info(f"  Stop-loss active at ${order.stop_loss_price:.2f}")
            else:
                order.status = OrderStatus.REJECTED
                self.logger.warning(f"Enhanced order rejected: {order.order_id}")

            return execution_successful

        except Exception as e:
            self.logger.error(f"Error executing enhanced order: {e}")
            return False

    def _execute_trade_enhanced(self, order: TradeOrder, price: float) -> bool:
        """Enhanced trade execution with position management"""
        trade_value = order.quantity * price

        if order.action == 'BUY':
            if self.cash_balance >= trade_value:
                self.cash_balance -= trade_value
                self._update_position_enhanced(order.symbol, order.quantity, price, order)
                return True
            else:
                self.logger.warning(f"Insufficient cash for order {order.order_id}")
                return False

        elif order.action == 'SELL':
            if order.symbol in self.positions and self.positions[order.symbol].quantity >= order.quantity:
                self.cash_balance += trade_value
                self._update_position_enhanced(order.symbol, -order.quantity, price, order)
                return True
            else:
                self.logger.warning(f"Insufficient position for sell order {order.order_id}")
                return False

        return False

    def _update_position_enhanced(self, symbol: str, quantity: int, price: float, order: TradeOrder):
        """Enhanced position tracking with stop-loss management"""
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=0,
                avg_price=0.0,
                current_price=price,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                entry_timestamp=datetime.now()
            )

        position = self.positions[symbol]

        if quantity > 0:  # Buy
            total_cost = (position.quantity * position.avg_price) + (quantity * price)
            position.quantity += quantity
            position.avg_price = total_cost / position.quantity if position.quantity > 0 else 0

            # Set stop-loss and take-profit from order
            if order.stop_loss_price:
                position.stop_loss_price = order.stop_loss_price
            if order.take_profit_price:
                position.take_profit_price = order.take_profit_price

        else:  # Sell
            # Calculate realized P&L
            realized_pnl = quantity * (price - position.avg_price)
            position.realized_pnl += realized_pnl
            position.quantity += quantity  # quantity is negative for sells

            # Clear stop-loss if position is closed
            if position.quantity == 0:
                position.stop_loss_price = None
                position.take_profit_price = None

        position.current_price = price
        position.unrealized_pnl = (price - position.avg_price) * position.quantity

        # Update max price for trailing stops
        if position.quantity > 0:  # Long position
            position.max_price_since_entry = max(position.max_price_since_entry, price)

    def update_positions(self, market_data: MarketData):
        """Update position current prices and P&L"""
        if market_data.symbol in self.positions:
            position = self.positions[market_data.symbol]
            position.current_price = market_data.price
            position.unrealized_pnl = (market_data.price - position.avg_price) * position.quantity

            # Update max price for trailing stops
            if position.quantity > 0:
                position.max_price_since_entry = max(position.max_price_since_entry, market_data.price)

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Enhanced portfolio summary with stop-loss information"""
        total_portfolio_value = self.cash_balance
        total_unrealized_pnl = 0.0
        total_realized_pnl = 0.0
        active_positions = []
        sector_exposure = {}
        stop_loss_coverage = 0

        for position in self.positions.values():
            if position.quantity != 0:
                position_value = position.quantity * position.current_price
                total_portfolio_value += position_value
                total_unrealized_pnl += position.unrealized_pnl
                total_realized_pnl += position.realized_pnl

                if position.quantity > 0:  # Only count long positions as active
                    active_positions.append(position)
                    if position.stop_loss_price:
                        stop_loss_coverage += 1

                # Sector classification
                sector = self._get_sector(position.symbol)
                sector_exposure[sector] = sector_exposure.get(sector, 0) + abs(position_value)

        # Enhanced performance metrics
        total_return = ((total_portfolio_value - 100000) / 100000) * 100

        # Win rate calculation
        profitable_trades = len([o for o in self.executed_orders if self._is_profitable_trade(o)])
        total_trades = len(self.executed_orders)
        win_rate = (profitable_trades / max(total_trades, 1)) * 100

        # Risk metrics
        avg_trade_size = np.mean([o.quantity * o.avg_fill_price for o in self.executed_orders]) if self.executed_orders else 0
        largest_loss = min([p.unrealized_pnl + p.realized_pnl for p in self.positions.values()] + [0])

        return {
            'cash_balance': self.cash_balance,
            'total_portfolio_value': total_portfolio_value,
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_realized_pnl': total_realized_pnl,
            'total_return_pct': total_return,
            'positions': dict(self.positions),
            'active_positions_count': len(active_positions),
            'executed_orders_count': len(self.executed_orders),
            'win_rate': win_rate,
            'sector_exposure': sector_exposure,
            'cash_allocation_pct': (self.cash_balance / total_portfolio_value) * 100,
            'equity_allocation_pct': ((total_portfolio_value - self.cash_balance) / total_portfolio_value) * 100,
            'stop_loss_coverage': stop_loss_coverage,
            'stop_loss_coverage_pct': (stop_loss_coverage / max(len(active_positions), 1)) * 100,
            'avg_trade_size': avg_trade_size,
            'largest_loss': largest_loss,
            'total_trades': total_trades
        }

    def _is_profitable_trade(self, order: TradeOrder) -> bool:
        """Determine if a trade was profitable"""
        if order.symbol in self.positions:
            position = self.positions[order.symbol]
            return (position.unrealized_pnl + position.realized_pnl) > 0
        return False

    def _get_sector(self, symbol: str) -> str:
        """Enhanced sector classification"""
        sector_map = {
            'AAPL': 'Technology',
            'GOOGL': 'Technology', 
            'MSFT': 'Technology',
            'TSLA': 'Automotive',
            'NVDA': 'Technology'
        }
        return sector_map.get(symbol, 'Other')

class CoordinatorAgent(BaseAgent):
    """Enhanced central coordinator with stop-loss management and document memory bank"""

    def __init__(self):
        super().__init__("Coordinator")
        self.market_analyst = MarketAnalystAgent()
        self.risk_manager = RiskManagerAgent()
        self.trading_executor = TradingExecutorAgent()
        self.document_uploader = TradingDocumentUploader() if DIGITAL_BRAIN_AVAILABLE else None
        self.is_running = False
        self.iteration_count = 0
        self.performance_history = []

    def start_system(self):
        """Start the enhanced trading system"""
        self.logger.info("Starting Enhanced Multi-Agent Trading System with Stop-Loss Management")

        self.market_analyst.start()
        self.risk_manager.start()
        self.trading_executor.start()
        self.start()

        self.is_running = True

    def stop_system(self):
        """Stop the enhanced trading system"""
        self.logger.info("Stopping Enhanced Multi-Agent Trading System")

        self.is_running = False
        self.market_analyst.stop()
        self.risk_manager.stop()
        self.trading_executor.stop()
        self.stop()

    def process(self, market_data: MarketData) -> Dict[str, Any]:
        """Enhanced coordination with stop-loss management"""
        try:
            self.iteration_count += 1

            # Step 1: Update positions with current market data
            self.trading_executor.update_positions(market_data)

            # Step 2: Check stop-loss triggers
            stop_loss_orders = self.risk_manager.process_market_update(market_data)
            stop_loss_executions = 0

            for stop_order in stop_loss_orders:
                if self.trading_executor.process(stop_order):
                    stop_loss_executions += 1

            # Step 3: Market Analysis (Technical + ML + Sentiment)
            signals = self.market_analyst.process(market_data)

            # Step 4: Risk Management and Order Generation
            orders_executed = 0
            risk_alerts = []

            for signal in signals:
                current_positions = self.trading_executor.positions
                order = self.risk_manager.process(signal, current_positions)

                # Step 5: Trade Execution
                if order:
                    execution_success = self.trading_executor.process(order)
                    if execution_success:
                        orders_executed += 1

                        # Create stop-loss orders for new positions
                        if order.action == 'BUY' and order.stop_loss_price:
                            position = self.trading_executor.positions.get(order.symbol)
                            if position:
                                sl_orders = self.risk_manager.create_stop_loss_orders(position)
                                # These would be managed by the stop-loss manager
                else:
                    risk_alerts.append(f"Order blocked for {signal.symbol}: Risk limits")

            # Step 6: Portfolio Optimization (every 20 iterations)
            portfolio_optimization = None
            if self.iteration_count % 20 == 0:
                current_positions = self.trading_executor.positions
                technical_data = {}
                for symbol in current_positions.keys():
                    ta = self.market_analyst.technical_indicators.calculate_indicators(symbol)
                    if ta:
                        technical_data[symbol] = ta

                portfolio_optimization = self.market_analyst.portfolio_optimizer.optimize_portfolio(
                    current_positions, technical_data)

            # Step 7: Enhanced Monitoring & Alerts
            portfolio_summary = self.trading_executor.get_portfolio_summary()
            risk_report = self.risk_manager.get_risk_report()
            alerts = self._generate_enhanced_alerts(portfolio_summary, risk_report)

            # Step 8: Performance Tracking
            performance_metrics = self._calculate_enhanced_performance_metrics(portfolio_summary)
            self.performance_history.append(performance_metrics)

            # Step 9: Advanced Analytics
            advanced_analytics = self._generate_enhanced_advanced_analytics()

            # Return comprehensive system status
            result = {
                'signals_generated': len(signals),
                'orders_executed': orders_executed,
                'stop_loss_executions': stop_loss_executions,
                'portfolio_summary': portfolio_summary,
                'risk_report': risk_report,
                'alerts': alerts + risk_alerts,
                'performance_metrics': performance_metrics,
                'system_health': self._check_enhanced_system_health(),
                'advanced_analytics': advanced_analytics,
                'timestamp': datetime.now()
            }

            if portfolio_optimization:
                result['portfolio_optimization'] = portfolio_optimization
                self.logger.info(f"Portfolio optimization completed: Sharpe ratio = {portfolio_optimization.sharpe_ratio:.3f}")

            return result

        except Exception as e:
            self.logger.error(f"Error in enhanced coordination process: {e}")
            return {'error': str(e)}

    def _generate_enhanced_alerts(self, portfolio: Dict[str, Any], risk_report: Dict[str, Any]) -> List[str]:
        """Generate enhanced alerts including stop-loss information"""
        alerts = []

        # Portfolio performance alerts
        if portfolio['total_return_pct'] < -15:
            alerts.append(f"🚨 High portfolio loss: {portfolio['total_return_pct']:.1f}%")
        elif portfolio['total_return_pct'] > 25:
            alerts.append(f"🎉 Excellent returns: {portfolio['total_return_pct']:.1f}%")

        # Stop-loss coverage alerts
        if portfolio.get('stop_loss_coverage_pct', 0) < 70:
            alerts.append(f"⚠️ Low stop-loss coverage: {portfolio.get('stop_loss_coverage_pct', 0):.0f}%")
        elif portfolio.get('stop_loss_coverage_pct', 0) == 100:
            alerts.append("✅ Full stop-loss protection active")

        # Risk alerts
        risk_metrics = risk_report.get('risk_metrics', {})
        if risk_metrics.get('max_drawdown', 0) > 0.15:
            alerts.append(f"⚠️ High drawdown: {risk_metrics['max_drawdown']:.1%}")

        # Active stop-loss alerts
        active_stops = risk_report.get('active_stops', 0)
        if active_stops > 0:
            alerts.append(f"🛡️ {active_stops} stop-loss orders active")

        # Cash management alerts
        if portfolio['cash_allocation_pct'] < 5:
            alerts.append("⚠️ Very low cash reserves (<5%)")
        elif portfolio['cash_allocation_pct'] > 60:
            alerts.append("💰 High cash allocation (>60%)")

        # Performance alerts
        if portfolio['win_rate'] < 35:
            alerts.append(f"⚠️ Low win rate: {portfolio['win_rate']:.1f}%")
        elif portfolio['win_rate'] > 70:
            alerts.append(f"🎯 High win rate: {portfolio['win_rate']:.1f}%")

        return alerts

    def _calculate_enhanced_performance_metrics(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate enhanced performance metrics"""
        metrics = {
            'total_return_pct': portfolio['total_return_pct'],
            'win_rate': portfolio['win_rate'],
            'profit_factor': self._calculate_profit_factor(portfolio),
            'avg_trade_return': self._calculate_avg_trade_return(portfolio),
            'max_position_size': portfolio.get('avg_trade_size', 0),
            'portfolio_diversification': len(portfolio['sector_exposure']),
            'active_positions': portfolio['active_positions_count'],
            'stop_loss_coverage_pct': portfolio.get('stop_loss_coverage_pct', 0),
            'largest_loss': portfolio.get('largest_loss', 0),
            'total_trades': portfolio.get('total_trades', 0)
        }

        # Risk-adjusted returns
        if len(self.performance_history) > 5:
            returns = [p['total_return_pct'] for p in self.performance_history[-10:]]
            metrics['return_volatility'] = np.std(returns)
            metrics['return_trend'] = 'IMPROVING' if returns[-1] > returns[0] else 'DECLINING'

        return metrics

    def _generate_enhanced_advanced_analytics(self) -> Dict[str, Any]:
        """Generate enhanced advanced analytics"""
        try:
            analytics = {
                'ml_models_active': 0,
                'ml_average_accuracy': 0.0,
                'sentiment_signals_generated': 0,
                'portfolio_optimization_runs': 0,
                'signal_type_distribution': {'technical': 0, 'ml': 0, 'sentiment': 0, 'ensemble': 0},
                'stop_loss_metrics': {
                    'active_stops': len(self.risk_manager.stop_loss_manager.active_stops),
                    'trailing_stops': len(self.risk_manager.stop_loss_manager.trailing_stops),
                    'stops_triggered_today': 0  # Would track in production
                }
            }

            # ML model stats
            if self.market_analyst.ml_engine:
                analytics['ml_models_active'] = len(self.market_analyst.ml_engine.models)
                if self.market_analyst.ml_engine.model_accuracy:
                    analytics['ml_average_accuracy'] = np.mean(list(self.market_analyst.ml_engine.model_accuracy.values()))

            # Sentiment analysis stats
            if hasattr(self.market_analyst.sentiment_analyzer, 'sentiment_history'):
                total_sentiment_data = sum(len(hist) for hist in self.market_analyst.sentiment_analyzer.sentiment_history.values())
                analytics['sentiment_signals_generated'] = total_sentiment_data

            # Portfolio optimization stats
            analytics['portfolio_optimization_runs'] = len(self.market_analyst.portfolio_optimizer.optimization_history)

            # Digital Brain stats
            if self.market_analyst.digital_brain:
                brain_status = self.market_analyst.digital_brain.get_brain_status()
                analytics['digital_brain'] = {
                    'knowledge_nodes': brain_status['knowledge_nodes'],
                    'knowledge_edges': brain_status['knowledge_edges'],
                    'learned_patterns': brain_status['learned_patterns'],
                    'processed_documents': brain_status['processed_documents'],
                    'patterns_used_today': self.market_analyst.brain_patterns_used,
                    'insights_generated': self.market_analyst.brain_insights_generated,
                    'knowledge_queries': self.market_analyst.brain_knowledge_queries,
                    'learning_events': self.market_analyst.brain_learning_events,
                    'brain_health': brain_status['memory_health'],
                    'average_pattern_confidence': brain_status['average_pattern_confidence'],
                    'memory_consolidation': brain_status['last_consolidation']
                }

            # Signal type distribution (last 50 signals)
            recent_signals = self.market_analyst.signals[-50:]
            for signal in recent_signals:
                if 'ML prediction' in signal.reason:
                    analytics['signal_type_distribution']['ml'] += 1
                elif 'sentiment' in signal.reason.lower():
                    analytics['signal_type_distribution']['sentiment'] += 1
                elif 'Brain pattern' in signal.reason:
                    analytics['signal_type_distribution']['brain'] = analytics['signal_type_distribution'].get('brain', 0) + 1
                elif 'Ensemble' in signal.reason:
                    analytics['signal_type_distribution']['ensemble'] += 1
                else:
                    analytics['signal_type_distribution']['technical'] += 1

            return analytics

        except Exception as e:
            self.logger.error(f"Error generating enhanced advanced analytics: {e}")
            return {}

    def _check_enhanced_system_health(self) -> Dict[str, str]:
        """Check enhanced system health including stop-loss management"""
        health = {
            'market_analyst': 'HEALTHY' if self.market_analyst.is_active else 'INACTIVE',
            'risk_manager': 'HEALTHY' if self.risk_manager.is_active else 'INACTIVE',
            'trading_executor': 'HEALTHY' if self.trading_executor.is_active else 'INACTIVE',
            'overall_status': 'OPERATIONAL'
        }

        # Enhanced health checks
        if self.market_analyst.ml_engine:
            ml_models = len(self.market_analyst.ml_engine.models)
            health['ml_engine'] = f'ACTIVE ({ml_models} models)' if ml_models > 0 else 'TRAINING'
        else:
            health['ml_engine'] = 'UNAVAILABLE'

        health['sentiment_analyzer'] = 'ACTIVE'
        health['portfolio_optimizer'] = 'ACTIVE'

        # Stop-loss system health
        active_stops = len(self.risk_manager.stop_loss_manager.active_stops)
        trailing_stops = len(self.risk_manager.stop_loss_manager.trailing_stops)
        health['stop_loss_manager'] = f'ACTIVE ({active_stops} stops, {trailing_stops} trailing)'

        return health

    def _calculate_profit_factor(self, portfolio: Dict[str, Any]) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        gross_profit = sum(max(0, p.unrealized_pnl + p.realized_pnl) for p in portfolio['positions'].values())
        gross_loss = abs(sum(min(0, p.unrealized_pnl + p.realized_pnl) for p in portfolio['positions'].values()))
        return gross_profit / max(gross_loss, 1) if gross_loss > 0 else float('inf')

    def _calculate_avg_trade_return(self, portfolio: Dict[str, Any]) -> float:
        """Calculate average trade return"""
        all_returns = [p.unrealized_pnl + p.realized_pnl for p in portfolio['positions'].values() if p.quantity != 0]
        return np.mean(all_returns) if all_returns else 0.0

class TradingSimulation:
    """Enhanced simulation environment with stop-loss management"""

    def __init__(self):
        self.coordinator = CoordinatorAgent()
        self.symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        self.simulation_running = False

    def start_simulation(self, duration_minutes: int = 3):
        """Start the enhanced trading simulation"""
        print("🚀 Starting Enhanced Multi-Agent Trading System with Stop-Loss Management")
        print("=" * 80)

        self.coordinator.start_system()
        self.simulation_running = True

        start_time = time.time()
        iteration = 0

        try:
            while self.simulation_running and (time.time() - start_time) < (duration_minutes * 60):
                iteration += 1

                # Generate mock market data for each symbol
                for symbol in self.symbols:
                    market_data = self._generate_mock_market_data(symbol)
                    result = self.coordinator.process(market_data)

                    if 'error' not in result:
                        # Real-time processing update
                        if iteration % 2 == 0:
                            print(f"\n[Iteration {iteration}] Processing {symbol}:")
                            print(f"  Signals: {result['signals_generated']} | Orders: {result['orders_executed']} | Stop-Loss: {result.get('stop_loss_executions', 0)}")

                            # Display alerts if any
                            if result.get('alerts'):
                                print(f"  Alerts: {', '.join(result['alerts'][:2])}")

                        # Enhanced dashboard every 10 iterations
                        if iteration % 10 == 0:
                            self._display_enhanced_dashboard(result, iteration)

                time.sleep(2)

        except KeyboardInterrupt:
            print("\n⚠️ Simulation interrupted by user")
        finally:
            self.stop_simulation()

    def _display_enhanced_dashboard(self, result: Dict[str, Any], iteration: int):
        """Display enhanced dashboard with stop-loss information"""
        portfolio = result['portfolio_summary']
        risk_report = result.get('risk_report', {})
        performance = result.get('performance_metrics', {})
        advanced = result.get('advanced_analytics', {})

        print(f"\n{'='*80}")
        print(f"🤖 ENHANCED ML TRADING DASHBOARD - Iteration {iteration}")
        print(f"{'='*80}")

        # Portfolio Overview
        print(f"💰 PORTFOLIO STATUS:")
        print(f"  Total Value: ${portfolio['total_portfolio_value']:,.2f}")
        print(f"  Total Return: {portfolio['total_return_pct']:+.2f}%")
        print(f"  Cash: ${portfolio['cash_balance']:,.2f} ({portfolio['cash_allocation_pct']:.1f}%)")
        print(f"  Active Positions: {portfolio['active_positions_count']}")

        # Enhanced Performance Metrics
        print(f"\n📈 PERFORMANCE METRICS:")
        print(f"  Win Rate: {portfolio['win_rate']:.1f}%")
        print(f"  Profit Factor: {performance.get('profit_factor', 0):.2f}")
        print(f"  Avg Trade Return: ${performance.get('avg_trade_return', 0):.2f}")
        print(f"  Total Trades: {portfolio.get('total_trades', 0)}")

        # Stop-Loss Information
        print(f"\n🛡️ STOP-LOSS PROTECTION:")
        print(f"  Coverage: {portfolio.get('stop_loss_coverage_pct', 0):.0f}%")
        print(f"  Active Stops: {advanced.get('stop_loss_metrics', {}).get('active_stops', 0)}")
        print(f"  Trailing Stops: {advanced.get('stop_loss_metrics', {}).get('trailing_stops', 0)}")
        print(f"  Largest Loss: ${portfolio.get('largest_loss', 0):.2f}")

        # Risk Metrics
        risk_metrics = risk_report.get('risk_metrics', {})
        print(f"\n⚠️ RISK METRICS:")
        print(f"  Max Drawdown: {risk_metrics.get('max_drawdown', 0):.1%}")
        print(f"  Volatility: {risk_metrics.get('volatility', 0):.1%}")
        print(f"  Sharpe Ratio: {risk_metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  VaR (95%): ${risk_metrics.get('var_95', 0):,.2f}")

        # AI & ML Metrics
        print(f"\n🧠 AI & ML METRICS:")
        print(f"  ML Models Active: {advanced.get('ml_models_active', 0)}")
        print(f"  ML Accuracy: {advanced.get('ml_average_accuracy', 0):.1%}")
        print(f"  Sentiment Signals: {advanced.get('sentiment_signals_generated', 0)}")

        # Digital Brain Metrics
        if advanced.get('digital_brain'):
            brain = advanced['digital_brain']
            print(f"\n🧠 DIGITAL BRAIN METRICS:")
            print(f"  Knowledge Graph: {brain.get('knowledge_nodes', 0)} nodes, {brain.get('knowledge_edges', 0)} edges")
            print(f"  Learned Patterns: {brain.get('learned_patterns', 0)} (avg confidence: {brain.get('average_pattern_confidence', 0):.1%})")
            print(f"  Daily Activity: {brain.get('patterns_used_today', 0)} patterns used, {brain.get('insights_generated', 0)} insights")
            print(f"  Learning Events: {brain.get('learning_events', 0)} | Knowledge Queries: {brain.get('knowledge_queries', 0)}")
            print(f"  Brain Health: {brain.get('brain_health', 'unknown').upper()} | Last Consolidation: {brain.get('memory_consolidation', 'N/A')[:10]}")

        # Enhanced System Health
        health = result.get('system_health', {})
        print(f"\n🔧 SYSTEM STATUS: {health.get('overall_status', 'UNKNOWN')}")
        print(f"  ML Engine: {health.get('ml_engine', 'N/A')}")
        print(f"  Stop-Loss Manager: {health.get('stop_loss_manager', 'N/A')}")

        # Active Alerts
        alerts = result.get('alerts', [])
        if alerts:
            print(f"\n🚨 ACTIVE ALERTS:")
            for alert in alerts[:4]:
                print(f"  {alert}")

        print(f"{'='*80}")

    def stop_simulation(self):
        """Stop the enhanced simulation"""
        print("\n🛑 Stopping enhanced simulation...")
        self.simulation_running = False
        self.coordinator.stop_system()

        # Enhanced final portfolio summary
        final_summary = self.coordinator.trading_executor.get_portfolio_summary()
        print("\n" + "=" * 80)
        print("📈 ENHANCED FINAL PORTFOLIO SUMMARY")
        print("=" * 80)
        print(f"Cash Balance: ${final_summary['cash_balance']:,.2f}")
        print(f"Total Portfolio Value: ${final_summary['total_portfolio_value']:,.2f}")
        print(f"Total P&L: ${final_summary['total_unrealized_pnl'] + final_summary['total_realized_pnl']:,.2f}")
        print(f"Total Orders Executed: {final_summary['executed_orders_count']}")
        print(f"Stop-Loss Coverage: {final_summary.get('stop_loss_coverage_pct', 0):.0f}%")

        active_positions = [p for p in final_summary['positions'].values() if p.quantity > 0]
        print(f"Active Positions: {len(active_positions)}")

        for position in active_positions:
            sl_info = f" | SL: ${position.stop_loss_price:.2f}" if position.stop_loss_price else " | No SL"
            tp_info = f" | TP: ${position.take_profit_price:.2f}" if position.take_profit_price else ""
            print(f"  {position.symbol}: {position.quantity} shares @ ${position.avg_price:.2f} (P&L: ${position.unrealized_pnl + position.realized_pnl:.2f}){sl_info}{tp_info}")

    def _generate_mock_market_data(self, symbol: str) -> MarketData:
        """Generate enhanced mock market data"""
        if not hasattr(self, '_price_trends'):
            self._price_trends = {symbol: 100.0 for symbol in self.symbols}

        # Enhanced price movements with volatility clustering
        trend = random.uniform(-1.5, 1.5)
        volatility = random.uniform(-4.0, 4.0)
        self._price_trends[symbol] += trend + volatility

        # Keep price in reasonable range
        self._price_trends[symbol] = max(50, min(150, self._price_trends[symbol]))

        base_price = self._price_trends[symbol]
        spread = 0.05

        return MarketData(
            symbol=symbol,
            price=base_price,
            volume=random.randint(1000, 15000),
            timestamp=datetime.now(),
            bid=base_price - spread,
            ask=base_price + spread,
            high_24h=base_price * (1 + random.uniform(0, 0.03)),
            low_24h=base_price * (1 - random.uniform(0, 0.03))
        )

def main():
    """Enhanced main entry point"""
    print("🚀 Enhanced Multi-Agent Trading System v2.0")
    print("Features: Advanced ML, Sentiment Analysis, Stop-Loss Management, Portfolio Optimization")
    print("Paper trading simulation environment with comprehensive risk management\n")

    simulation = TradingSimulation()

    try:
        simulation.start_simulation(duration_minutes=3)
    except Exception as e:
        print(f"❌ Error in enhanced simulation: {e}")
        logging.error(f"Enhanced simulation error: {e}")

if __name__ == "__main__":
    main()