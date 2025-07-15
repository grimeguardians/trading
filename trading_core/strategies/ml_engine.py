"""Streamlined ML prediction engine"""

import numpy as np
from datetime import datetime
from typing import Dict, Optional

from ..data_models import TechnicalAnalysis, MarketData, SentimentData, MLPrediction

# Optional ML libraries
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    ML_LIBS_AVAILABLE = True
except ImportError:
    ML_LIBS_AVAILABLE = False

class MLPredictionEngine:
    """Streamlined ML prediction engine"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_history = {}
        self.model_accuracy = {}
        self.min_training_samples = 30  # Reduced for faster training

    def prepare_features(self, symbol: str, technical_analysis: TechnicalAnalysis, 
                        market_data: MarketData, sentiment: Optional[SentimentData] = None) -> np.array:
        """Prepare feature vector"""
        features = [
            technical_analysis.rsi / 100.0,
            technical_analysis.macd,
            (market_data.price - technical_analysis.bb_middle) / (technical_analysis.bb_upper - technical_analysis.bb_lower) if technical_analysis.bb_upper != technical_analysis.bb_lower else 0,
            technical_analysis.stochastic_k / 100.0,
            market_data.volume / technical_analysis.volume_sma if technical_analysis.volume_sma > 0 else 1,
            technical_analysis.atr / market_data.price if market_data.price > 0 else 0,
            technical_analysis.volatility,
        ]

        # Add sentiment features
        if sentiment:
            features.extend([
                sentiment.sentiment_score,
                sentiment.news_count / 10.0,
            ])
        else:
            features.extend([0.0, 0.0])

        # Add time features
        hour = datetime.now().hour
        features.extend([
            hour / 24.0,
            1.0 if 9 <= hour <= 16 else 0.0,
        ])

        return np.array(features)

    def update_training_data(self, symbol: str, features: np.array, target: float):
        """Update training data"""
        if symbol not in self.feature_history:
            self.feature_history[symbol] = []

        self.feature_history[symbol].append({
            'features': features,
            'target': 1 if target > 0 else 0,  # Binary classification
            'timestamp': datetime.now()
        })

        # Keep only recent history
        if len(self.feature_history[symbol]) > 100:
            self.feature_history[symbol] = self.feature_history[symbol][-100:]

    def train_model(self, symbol: str) -> bool:
        """Train ML model"""
        if not ML_LIBS_AVAILABLE:
            return False

        if symbol not in self.feature_history or len(self.feature_history[symbol]) < self.min_training_samples:
            return False

        try:
            # Prepare data
            features = np.array([item['features'] for item in self.feature_history[symbol]])
            targets = np.array([item['target'] for item in self.feature_history[symbol]])

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

            # Scale features
            if symbol not in self.scalers:
                self.scalers[symbol] = StandardScaler()

            X_train_scaled = self.scalers[symbol].fit_transform(X_train)
            X_test_scaled = self.scalers[symbol].transform(X_test)

            # Train model
            if symbol not in self.models:
                self.models[symbol] = RandomForestClassifier(
                    n_estimators=50,  # Reduced for speed
                    max_depth=8,
                    random_state=42
                )

            self.models[symbol].fit(X_train_scaled, y_train)

            # Calculate accuracy
            accuracy = self.models[symbol].score(X_test_scaled, y_test)
            self.model_accuracy[symbol] = accuracy

            print(f"ML model trained for {symbol} with accuracy: {accuracy:.3f}")
            return True

        except Exception as e:
            print(f"Error training ML model for {symbol}: {e}")
            return False

    def predict(self, symbol: str, features: np.array) -> Optional[MLPrediction]:
        """Make prediction"""
        if not ML_LIBS_AVAILABLE or symbol not in self.models:
            return None

        try:
            # Scale features
            features_scaled = self.scalers[symbol].transform(features.reshape(1, -1))

            # Make prediction
            prediction_proba = self.models[symbol].predict_proba(features_scaled)[0]
            prediction_class = self.models[symbol].predict(features_scaled)[0]

            confidence = max(prediction_proba)

            # Determine action
            if prediction_class == 1 and confidence > 0.6:
                action = 'BUY'
            elif prediction_class == 0 and confidence > 0.6:
                action = 'SELL'
            else:
                action = 'HOLD'

            return MLPrediction(
                symbol=symbol,
                prediction=action,
                confidence=confidence,
                feature_importance={},  # Simplified
                model_accuracy=self.model_accuracy.get(symbol, 0.0),
                timestamp=datetime.now(),
                risk_score=0.0
            )

        except Exception as e:
            print(f"Error making ML prediction for {symbol}: {e}")
            return None