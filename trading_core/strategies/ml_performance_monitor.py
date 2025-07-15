"""
ML Performance Monitoring and Backtesting System
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import logging

class MLPerformanceMonitor:
    """Monitor and analyze ML model performance in real-time"""
    
    def __init__(self):
        self.logger = logging.getLogger("MLPerformanceMonitor")
        
        # Performance tracking
        self.prediction_outcomes = {}  # symbol -> list of predictions with outcomes
        self.model_scores = {}         # symbol -> model -> performance metrics
        self.feature_analysis = {}     # Feature importance tracking
        
        # Backtesting data
        self.backtest_results = {}
        self.rolling_accuracy = {}
        
        # Configuration
        self.prediction_horizon = 30  # Minutes to wait for outcome
        self.min_samples_for_analysis = 20
        
    def record_prediction(self, symbol: str, prediction: Dict[str, Any], current_price: float):
        """Record a prediction for later evaluation"""
        if symbol not in self.prediction_outcomes:
            self.prediction_outcomes[symbol] = []
        
        prediction_record = {
            'timestamp': datetime.now(),
            'prediction': prediction['prediction'],
            'confidence': prediction['confidence'],
            'model_used': prediction.get('model', 'unknown'),
            'entry_price': current_price,
            'outcome_price': None,
            'outcome_determined': False,
            'correct_prediction': None,
            'profit_loss': None
        }
        
        self.prediction_outcomes[symbol].append(prediction_record)
        
        # Keep only recent predictions (last 200)
        if len(self.prediction_outcomes[symbol]) > 200:
            self.prediction_outcomes[symbol] = self.prediction_outcomes[symbol][-200:]
    
    def update_prediction_outcomes(self, symbol: str, current_price: float):
        """Update outcomes for predictions that have reached their horizon"""
        if symbol not in self.prediction_outcomes:
            return
        
        current_time = datetime.now()
        
        for record in self.prediction_outcomes[symbol]:
            if record['outcome_determined']:
                continue
                
            # Check if enough time has passed
            time_elapsed = (current_time - record['timestamp']).total_seconds() / 60
            
            if time_elapsed >= self.prediction_horizon:
                # Determine outcome
                record['outcome_price'] = current_price
                record['outcome_determined'] = True
                
                # Calculate price change
                price_change_pct = (current_price - record['entry_price']) / record['entry_price']
                
                # Determine if prediction was correct
                if record['prediction'] == 'BUY':
                    record['correct_prediction'] = price_change_pct > 0.005  # >0.5% gain
                    record['profit_loss'] = price_change_pct
                elif record['prediction'] == 'SELL':
                    record['correct_prediction'] = price_change_pct < -0.005  # >0.5% loss
                    record['profit_loss'] = -price_change_pct  # Profit from short
                else:  # HOLD
                    record['correct_prediction'] = abs(price_change_pct) < 0.01  # <1% change
                    record['profit_loss'] = 0
        
        # Update model performance scores
        self._update_model_scores(symbol)
    
    def _update_model_scores(self, symbol: str):
        """Update performance scores for each model"""
        if symbol not in self.prediction_outcomes:
            return
        
        # Get completed predictions
        completed = [r for r in self.prediction_outcomes[symbol] if r['outcome_determined']]
        
        if len(completed) < self.min_samples_for_analysis:
            return
        
        # Group by model
        model_predictions = {}
        for record in completed:
            model = record['model_used']
            if model not in model_predictions:
                model_predictions[model] = []
            model_predictions[model].append(record)
        
        # Calculate metrics for each model
        if symbol not in self.model_scores:
            self.model_scores[symbol] = {}
        
        for model, predictions in model_predictions.items():
            if len(predictions) < 5:  # Need minimum samples
                continue
                
            # Calculate metrics
            correct_predictions = sum(1 for p in predictions if p['correct_prediction'])
            accuracy = correct_predictions / len(predictions)
            
            # Calculate average profit/loss
            avg_profit = np.mean([p['profit_loss'] for p in predictions])
            total_profit = sum(p['profit_loss'] for p in predictions)
            
            # Calculate Sharpe-like ratio
            profits = [p['profit_loss'] for p in predictions]
            sharpe_ratio = np.mean(profits) / np.std(profits) if np.std(profits) > 0 else 0
            
            # Calculate by prediction type
            buy_predictions = [p for p in predictions if p['prediction'] == 'BUY']
            sell_predictions = [p for p in predictions if p['prediction'] == 'SELL']
            
            buy_accuracy = (sum(1 for p in buy_predictions if p['correct_prediction']) / 
                           len(buy_predictions)) if buy_predictions else 0
            sell_accuracy = (sum(1 for p in sell_predictions if p['correct_prediction']) / 
                            len(sell_predictions)) if sell_predictions else 0
            
            self.model_scores[symbol][model] = {
                'accuracy': accuracy,
                'total_predictions': len(predictions),
                'avg_profit': avg_profit,
                'total_profit': total_profit,
                'sharpe_ratio': sharpe_ratio,
                'buy_accuracy': buy_accuracy,
                'sell_accuracy': sell_accuracy,
                'buy_count': len(buy_predictions),
                'sell_count': len(sell_predictions),
                'last_updated': datetime.now().isoformat()
            }
    
    def get_performance_report(self, symbol: str = None) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        if symbol:
            symbols = [symbol] if symbol in self.model_scores else []
        else:
            symbols = list(self.model_scores.keys())
        
        report = {
            'summary': {
                'total_symbols': len(symbols),
                'total_models': sum(len(models) for models in self.model_scores.values()),
                'generated_at': datetime.now().isoformat()
            },
            'symbols': {}
        }
        
        for sym in symbols:
            symbol_report = {
                'models': self.model_scores[sym],
                'best_model': self._get_best_model(sym),
                'rolling_performance': self._get_rolling_performance(sym)
            }
            report['symbols'][sym] = symbol_report
        
        return report
    
    def _get_best_model(self, symbol: str) -> Dict[str, Any]:
        """Identify the best performing model for a symbol"""
        if symbol not in self.model_scores:
            return {}
        
        best_model = None
        best_score = -1
        
        for model, metrics in self.model_scores[symbol].items():
            # Composite score: accuracy + profit + sharpe ratio
            score = (metrics['accuracy'] * 0.4 + 
                    min(metrics['avg_profit'] * 10, 0.3) * 0.3 + 
                    min(metrics['sharpe_ratio'], 1.0) * 0.3)
            
            if score > best_score:
                best_score = score
                best_model = model
        
        return {
            'model': best_model,
            'composite_score': best_score,
            'metrics': self.model_scores[symbol].get(best_model, {})
        }
    
    def _get_rolling_performance(self, symbol: str, window: int = 20) -> Dict[str, List]:
        """Calculate rolling performance metrics"""
        if symbol not in self.prediction_outcomes:
            return {}
        
        completed = [r for r in self.prediction_outcomes[symbol] 
                    if r['outcome_determined']]
        
        if len(completed) < window:
            return {}
        
        # Calculate rolling accuracy
        rolling_accuracy = []
        rolling_profit = []
        
        for i in range(window, len(completed) + 1):
            window_data = completed[i-window:i]
            
            accuracy = sum(1 for p in window_data if p['correct_prediction']) / len(window_data)
            avg_profit = np.mean([p['profit_loss'] for p in window_data])
            
            rolling_accuracy.append(accuracy)
            rolling_profit.append(avg_profit)
        
        return {
            'rolling_accuracy': rolling_accuracy,
            'rolling_profit': rolling_profit,
            'window_size': window,
            'data_points': len(rolling_accuracy)
        }
    
    def run_backtest(self, symbol: str, lookback_days: int = 30) -> Dict[str, Any]:
        """Run a simple backtest on historical predictions"""
        if symbol not in self.prediction_outcomes:
            return {}
        
        # Get predictions from the last N days
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        historical_predictions = [
            p for p in self.prediction_outcomes[symbol] 
            if p['timestamp'] >= cutoff_date and p['outcome_determined']
        ]
        
        if len(historical_predictions) < 10:
            return {'error': 'Insufficient historical data for backtesting'}
        
        # Simulate trading based on predictions
        initial_capital = 10000
        current_capital = initial_capital
        positions = []
        trades = []
        
        for prediction in historical_predictions:
            if prediction['prediction'] in ['BUY', 'SELL']:
                # Calculate position size (simple: use 10% of capital per trade)
                position_size = current_capital * 0.1
                
                if prediction['prediction'] == 'BUY':
                    # Buy position
                    shares = position_size / prediction['entry_price']
                    exit_value = shares * prediction['outcome_price']
                    profit = exit_value - position_size
                else:  # SELL (short)
                    # Short position
                    shares = position_size / prediction['entry_price']
                    exit_value = position_size - (shares * (prediction['outcome_price'] - prediction['entry_price']))
                    profit = exit_value - position_size
                
                current_capital += profit
                
                trades.append({
                    'entry_price': prediction['entry_price'],
                    'exit_price': prediction['outcome_price'],
                    'action': prediction['prediction'],
                    'profit': profit,
                    'confidence': prediction['confidence']
                })
        
        # Calculate performance metrics
        total_return = (current_capital - initial_capital) / initial_capital
        winning_trades = len([t for t in trades if t['profit'] > 0])
        win_rate = winning_trades / len(trades) if trades else 0
        
        avg_profit_per_trade = np.mean([t['profit'] for t in trades]) if trades else 0
        max_profit = max([t['profit'] for t in trades]) if trades else 0
        max_loss = min([t['profit'] for t in trades]) if trades else 0
        
        return {
            'initial_capital': initial_capital,
            'final_capital': current_capital,
            'total_return': total_return,
            'total_trades': len(trades),
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'avg_profit_per_trade': avg_profit_per_trade,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'lookback_days': lookback_days,
            'trades': trades[-10:]  # Last 10 trades for analysis
        }
    
    def get_model_comparison(self) -> Dict[str, Any]:
        """Compare performance across all models and symbols"""
        comparison = {
            'model_rankings': {},
            'cross_symbol_performance': {},
            'best_features': {}
        }
        
        # Aggregate performance by model type across all symbols
        model_aggregates = {}
        
        for symbol, models in self.model_scores.items():
            for model, metrics in models.items():
                if model not in model_aggregates:
                    model_aggregates[model] = {
                        'accuracies': [],
                        'profits': [],
                        'sharpe_ratios': [],
                        'total_predictions': 0
                    }
                
                model_aggregates[model]['accuracies'].append(metrics['accuracy'])
                model_aggregates[model]['profits'].append(metrics['avg_profit'])
                model_aggregates[model]['sharpe_ratios'].append(metrics['sharpe_ratio'])
                model_aggregates[model]['total_predictions'] += metrics['total_predictions']
        
        # Calculate average performance
        for model, data in model_aggregates.items():
            comparison['model_rankings'][model] = {
                'avg_accuracy': np.mean(data['accuracies']),
                'avg_profit': np.mean(data['profits']),
                'avg_sharpe': np.mean(data['sharpe_ratios']),
                'total_predictions': data['total_predictions'],
                'symbol_count': len(data['accuracies'])
            }
        
        return comparison