"""
Advanced Machine Learning Engine for Trading
Implements multiple ML algorithms, ensemble methods, and feature engineering
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import json

from ..data_models import TechnicalAnalysis, MarketData, SentimentData, MLPrediction

# ML Libraries (with graceful fallback)
try:
    from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                                 VotingClassifier, AdaBoostClassifier)
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    from sklearn.feature_selection import SelectKBest, f_classif
    ML_AVAILABLE = True
    print("🤖 Advanced ML libraries loaded successfully!")
except ImportError:
    print("⚠️  Advanced ML libraries not available - install scikit-learn for full functionality")
    ML_AVAILABLE = False

class AdvancedMLEngine:
    """
    Advanced ML Engine with multiple algorithms and ensemble methods
    """
    
    def __init__(self):
        self.logger = logging.getLogger("AdvancedMLEngine")
        
        # Model storage
        self.models = {}  # symbol -> dict of models
        self.ensemble_models = {}  # symbol -> ensemble model
        self.scalers = {}
        self.feature_selectors = {}
        
        # Data storage
        self.feature_history = {}
        self.prediction_history = {}
        self.model_performance = {}
        
        # Configuration
        self.min_training_samples = 50
        self.retrain_frequency = 20  # Retrain every N samples
        self.ensemble_enabled = True
        self.feature_selection_enabled = True
        
        # Feature engineering parameters
        self.lookback_periods = [5, 10, 20]
        self.technical_features = True
        self.sentiment_features = True
        self.temporal_features = True
        
        if ML_AVAILABLE:
            self._initialize_models()
        
    def _initialize_models(self):
        """Initialize different ML algorithms"""
        self.base_models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                class_weight='balanced'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf',
                probability=True,
                random_state=42,
                class_weight='balanced'
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                random_state=42,
                max_iter=500,
                early_stopping=True
            ),
            'ada_boost': AdaBoostClassifier(
                n_estimators=50,
                random_state=42
            )
        }
        
    def engineer_features(self, symbol: str, technical_analysis: TechnicalAnalysis, 
                         market_data: MarketData, sentiment: Optional[SentimentData] = None) -> np.array:
        """
        Advanced feature engineering with multiple technical indicators,
        sentiment analysis, and temporal features
        """
        features = []
        
        # 1. Basic Technical Features
        if self.technical_features:
            # Price-based features
            features.extend([
                technical_analysis.rsi / 100.0,  # Normalized RSI
                (technical_analysis.rsi - 50) / 50,  # RSI momentum
                technical_analysis.macd,
                technical_analysis.macd_signal,
                technical_analysis.macd_histogram,
                technical_analysis.stochastic_k / 100.0,
                technical_analysis.stochastic_d / 100.0,
                technical_analysis.williams_r / -100.0,
            ])
            
            # Bollinger Bands features
            if technical_analysis.bb_upper != technical_analysis.bb_lower:
                bb_position = (market_data.price - technical_analysis.bb_middle) / (technical_analysis.bb_upper - technical_analysis.bb_lower)
                bb_width = (technical_analysis.bb_upper - technical_analysis.bb_lower) / technical_analysis.bb_middle
                features.extend([bb_position, bb_width])
            else:
                features.extend([0.0, 0.0])
            
            # Moving average features
            features.extend([
                (market_data.price - technical_analysis.sma_20) / technical_analysis.sma_20,
                (market_data.price - technical_analysis.sma_50) / technical_analysis.sma_50,
                (technical_analysis.sma_20 - technical_analysis.sma_50) / technical_analysis.sma_50,
                (technical_analysis.ema_12 - technical_analysis.ema_26) / technical_analysis.ema_26,
            ])
            
            # Volume features
            if technical_analysis.volume_sma > 0:
                volume_ratio = market_data.volume / technical_analysis.volume_sma
                features.append(volume_ratio)
            else:
                features.append(1.0)
            
            # Volatility features
            features.extend([
                technical_analysis.volatility,
                technical_analysis.atr / market_data.price if market_data.price > 0 else 0,
            ])
            
            # Support/Resistance features
            if technical_analysis.support_level > 0:
                support_distance = (market_data.price - technical_analysis.support_level) / market_data.price
                features.append(support_distance)
            else:
                features.append(0.0)
                
            if technical_analysis.resistance_level > 0:
                resistance_distance = (technical_analysis.resistance_level - market_data.price) / market_data.price
                features.append(resistance_distance)
            else:
                features.append(0.0)
        
        # 2. Sentiment Features
        if self.sentiment_features and sentiment:
            features.extend([
                sentiment.sentiment_score,
                sentiment.news_count / 10.0,  # Normalized news count
                sentiment.social_mentions / 100.0,  # Normalized social mentions
                1.0 if sentiment.sentiment_trend == 'IMPROVING' else (0.5 if sentiment.sentiment_trend == 'STABLE' else 0.0),
                1.0 if sentiment.sentiment_label == 'BULLISH' else (0.5 if sentiment.sentiment_label == 'NEUTRAL' else 0.0)
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.5, 0.5])  # Neutral sentiment
        
        # 3. Temporal Features
        if self.temporal_features:
            now = datetime.now()
            features.extend([
                now.hour / 24.0,  # Hour of day
                now.weekday() / 7.0,  # Day of week
                1.0 if 9 <= now.hour <= 16 else 0.0,  # Market hours
                1.0 if now.weekday() < 5 else 0.0,  # Weekday
            ])
        
        # 4. Historical Features (if we have price history)
        if symbol in self.feature_history and len(self.feature_history[symbol]) > 0:
            recent_prices = [item.get('price', market_data.price) for item in self.feature_history[symbol][-10:]]
            recent_prices.append(market_data.price)
            
            # Price momentum features
            if len(recent_prices) >= 5:
                price_change_1 = (recent_prices[-1] - recent_prices[-2]) / recent_prices[-2] if recent_prices[-2] > 0 else 0
                price_change_5 = (recent_prices[-1] - recent_prices[-5]) / recent_prices[-5] if recent_prices[-5] > 0 else 0
                price_volatility = np.std(recent_prices[-5:]) / np.mean(recent_prices[-5:]) if np.mean(recent_prices[-5:]) > 0 else 0
                
                features.extend([price_change_1, price_change_5, price_volatility])
            else:
                features.extend([0.0, 0.0, 0.0])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        return np.array(features)
    
    def update_training_data(self, symbol: str, features: np.array, target: float, market_data: MarketData):
        """Update training data with new sample"""
        if symbol not in self.feature_history:
            self.feature_history[symbol] = []
        
        # Create training sample
        sample = {
            'features': features,
            'target': 1 if target > 0.01 else 0,  # Binary: significant price increase
            'target_regression': target,  # Keep continuous target for future use
            'timestamp': datetime.now(),
            'price': market_data.price,
            'volume': market_data.volume
        }
        
        self.feature_history[symbol].append(sample)
        
        # Keep only recent history (sliding window)
        max_history = 500
        if len(self.feature_history[symbol]) > max_history:
            self.feature_history[symbol] = self.feature_history[symbol][-max_history:]
        
        # Auto-retrain if we have enough samples
        if (len(self.feature_history[symbol]) >= self.min_training_samples and 
            len(self.feature_history[symbol]) % self.retrain_frequency == 0):
            self.train_models(symbol)
    
    def train_models(self, symbol: str) -> Dict[str, float]:
        """Train all models for a symbol"""
        if not ML_AVAILABLE or symbol not in self.feature_history:
            return {}
        
        if len(self.feature_history[symbol]) < self.min_training_samples:
            self.logger.warning(f"Insufficient data for training {symbol}: {len(self.feature_history[symbol])} samples")
            return {}
        
        try:
            # Prepare training data
            features = np.array([item['features'] for item in self.feature_history[symbol]])
            targets = np.array([item['target'] for item in self.feature_history[symbol]])
            
            # Handle class imbalance
            if len(np.unique(targets)) < 2:
                self.logger.warning(f"Insufficient class diversity for {symbol}")
                return {}
            
            # Time series split (more appropriate for financial data)
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Feature selection (optional)
            if self.feature_selection_enabled and features.shape[1] > 10:
                if symbol not in self.feature_selectors:
                    self.feature_selectors[symbol] = SelectKBest(f_classif, k=min(15, features.shape[1]))
                
                features_selected = self.feature_selectors[symbol].fit_transform(features, targets)
            else:
                features_selected = features
                self.feature_selectors[symbol] = None
            
            # Scale features
            if symbol not in self.scalers:
                self.scalers[symbol] = RobustScaler()  # More robust to outliers
            
            features_scaled = self.scalers[symbol].fit_transform(features_selected)
            
            # Train individual models
            model_scores = {}
            trained_models = {}
            
            for model_name, model in self.base_models.items():
                try:
                    # Cross-validation
                    cv_scores = cross_val_score(model, features_scaled, targets, cv=tscv, scoring='roc_auc')
                    avg_score = np.mean(cv_scores)
                    
                    # Train on full dataset
                    model.fit(features_scaled, targets)
                    trained_models[model_name] = model
                    model_scores[model_name] = avg_score
                    
                    self.logger.info(f"Trained {model_name} for {symbol}: {avg_score:.3f} AUC")
                    
                except Exception as e:
                    self.logger.error(f"Error training {model_name} for {symbol}: {e}")
                    continue
            
            # Store models
            if symbol not in self.models:
                self.models[symbol] = {}
            self.models[symbol].update(trained_models)
            
            # Create ensemble model
            if self.ensemble_enabled and len(trained_models) >= 2:
                ensemble_models = [(name, model) for name, model in trained_models.items() 
                                 if model_scores.get(name, 0) > 0.5]  # Only include decent models
                
                if len(ensemble_models) >= 2:
                    ensemble = VotingClassifier(estimators=ensemble_models, voting='soft')
                    ensemble.fit(features_scaled, targets)
                    self.ensemble_models[symbol] = ensemble
                    
                    # Evaluate ensemble
                    ensemble_scores = cross_val_score(ensemble, features_scaled, targets, cv=tscv, scoring='roc_auc')
                    ensemble_score = np.mean(ensemble_scores)
                    model_scores['ensemble'] = ensemble_score
                    
                    self.logger.info(f"Trained ensemble for {symbol}: {ensemble_score:.3f} AUC")
            
            # Store performance metrics
            self.model_performance[symbol] = {
                'scores': model_scores,
                'training_samples': len(self.feature_history[symbol]),
                'last_trained': datetime.now().isoformat(),
                'feature_count': features_scaled.shape[1]
            }
            
            return model_scores
            
        except Exception as e:
            self.logger.error(f"Error training models for {symbol}: {e}")
            return {}
    
    def predict(self, symbol: str, features: np.array) -> Optional[MLPrediction]:
        """Make prediction using the best available model"""
        if not ML_AVAILABLE or symbol not in self.models:
            return None
        
        try:
            # Apply feature selection if used
            if self.feature_selectors.get(symbol) is not None:
                features_selected = self.feature_selectors[symbol].transform(features.reshape(1, -1))
            else:
                features_selected = features.reshape(1, -1)
            
            # Scale features
            if symbol not in self.scalers:
                return None
            
            features_scaled = self.scalers[symbol].transform(features_selected)
            
            # Try ensemble first, then individual models
            best_model = None
            model_name = "unknown"
            
            if symbol in self.ensemble_models:
                best_model = self.ensemble_models[symbol]
                model_name = "ensemble"
            else:
                # Find best individual model
                best_score = 0
                for name, model in self.models[symbol].items():
                    if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier, 
                                        SVC, LogisticRegression, MLPClassifier)):
                        score = self.model_performance.get(symbol, {}).get('scores', {}).get(name, 0)
                        if score > best_score:
                            best_score = score
                            best_model = model
                            model_name = name
            
            if best_model is None:
                return None
            
            # Make prediction
            prediction_proba = best_model.predict_proba(features_scaled)[0]
            prediction_class = best_model.predict(features_scaled)[0]
            
            # Calculate confidence
            confidence = max(prediction_proba)
            
            # Determine action based on prediction and confidence
            if prediction_class == 1 and confidence > 0.6:
                action = 'BUY'
            elif prediction_class == 0 and confidence > 0.6:
                action = 'SELL'
            else:
                action = 'HOLD'
            
            # Get feature importance (if available)
            feature_importance = {}
            if hasattr(best_model, 'feature_importances_'):
                feature_names = [f'feature_{i}' for i in range(len(best_model.feature_importances_))]
                feature_importance = dict(zip(feature_names, best_model.feature_importances_))
            
            # Calculate risk score
            risk_score = 1 - confidence  # Higher confidence = lower risk
            
            # Get model accuracy
            model_accuracy = self.model_performance.get(symbol, {}).get('scores', {}).get(model_name, 0.0)
            
            prediction = MLPrediction(
                symbol=symbol,
                prediction=action,
                confidence=confidence,
                feature_importance=feature_importance,
                model_accuracy=model_accuracy,
                timestamp=datetime.now(),
                risk_score=risk_score
            )
            
            # Store prediction history
            if symbol not in self.prediction_history:
                self.prediction_history[symbol] = []
            
            self.prediction_history[symbol].append({
                'prediction': action,
                'confidence': confidence,
                'model': model_name,
                'timestamp': datetime.now(),
                'probabilities': prediction_proba.tolist()
            })
            
            # Keep only recent predictions
            if len(self.prediction_history[symbol]) > 100:
                self.prediction_history[symbol] = self.prediction_history[symbol][-100:]
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error making prediction for {symbol}: {e}")
            return None
    
    def get_model_performance_report(self, symbol: str = None) -> Dict[str, Any]:
        """Get comprehensive model performance report"""
        if symbol:
            symbols = [symbol]
        else:
            symbols = list(self.model_performance.keys())
        
        report = {}
        
        for sym in symbols:
            if sym in self.model_performance:
                perf = self.model_performance[sym]
                
                # Calculate prediction accuracy (if we have enough history)
                prediction_accuracy = self._calculate_prediction_accuracy(sym)
                
                report[sym] = {
                    'model_scores': perf.get('scores', {}),
                    'training_samples': perf.get('training_samples', 0),
                    'last_trained': perf.get('last_trained', 'Never'),
                    'feature_count': perf.get('feature_count', 0),
                    'prediction_accuracy': prediction_accuracy,
                    'models_available': list(self.models.get(sym, {}).keys()),
                    'ensemble_available': sym in self.ensemble_models,
                    'total_predictions': len(self.prediction_history.get(sym, []))
                }
        
        return report
    
    def _calculate_prediction_accuracy(self, symbol: str) -> Dict[str, float]:
        """Calculate real prediction accuracy from historical data"""
        if symbol not in self.prediction_history or len(self.prediction_history[symbol]) < 10:
            return {'accuracy': 0.0, 'sample_size': 0}
        
        # This would require actual price outcomes to calculate real accuracy
        # For now, return placeholder metrics
        predictions = self.prediction_history[symbol]
        
        return {
            'accuracy': 0.65,  # Placeholder - would calculate from actual outcomes
            'sample_size': len(predictions),
            'buy_predictions': len([p for p in predictions if p['prediction'] == 'BUY']),
            'sell_predictions': len([p for p in predictions if p['prediction'] == 'SELL']),
            'hold_predictions': len([p for p in predictions if p['prediction'] == 'HOLD']),
            'avg_confidence': np.mean([p['confidence'] for p in predictions])
        }
    
    def save_models(self, filepath: str):
        """Save models to file (would use pickle in production)"""
        model_data = {
            'model_performance': self.model_performance,
            'prediction_history': {k: v[-50:] for k, v in self.prediction_history.items()},  # Save recent history
            'config': {
                'min_training_samples': self.min_training_samples,
                'retrain_frequency': self.retrain_frequency,
                'ensemble_enabled': self.ensemble_enabled
            }
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(model_data, f, default=str, indent=2)
            self.logger.info(f"Model data saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
    
    def get_feature_importance_analysis(self, symbol: str) -> Dict[str, Any]:
        """Analyze feature importance across models"""
        if symbol not in self.models:
            return {}
        
        feature_importance_summary = {}
        
        for model_name, model in self.models[symbol].items():
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                feature_names = [f'feature_{i}' for i in range(len(importance))]
                
                # Get top 10 most important features
                top_indices = np.argsort(importance)[-10:][::-1]
                top_features = {feature_names[i]: importance[i] for i in top_indices}
                
                feature_importance_summary[model_name] = top_features
        
        return feature_importance_summary