"""Simplified portfolio optimization"""

import numpy as np
from datetime import datetime
from typing import Dict, List

from ..data_models import Position, TechnicalAnalysis, PortfolioOptimization

class PortfolioOptimizer:
    """Simplified portfolio optimization"""

    def __init__(self):
        self.optimization_history = []

    def optimize_portfolio(self, positions: Dict[str, Position], 
                         technical_data: Dict[str, TechnicalAnalysis]) -> PortfolioOptimization:
        """Optimize portfolio allocation"""
        try:
            symbols = list(positions.keys())
            if len(symbols) < 2:
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

            # Simple equal-weight optimization with volatility adjustment
            base_weights = np.ones(len(symbols)) / len(symbols)
            
            # Adjust weights based on volatility
            volatility_adjustments = []
            for symbol in symbols:
                if symbol in technical_data:
                    vol = technical_data[symbol].volatility
                    # Lower volatility gets higher weight
                    vol_adj = 1 / (1 + vol) if vol > 0 else 1.0
                    volatility_adjustments.append(vol_adj)
                else:
                    volatility_adjustments.append(1.0)

            # Normalize volatility-adjusted weights
            vol_adj_array = np.array(volatility_adjustments)
            optimal_weights = vol_adj_array / np.sum(vol_adj_array)

            # Calculate portfolio metrics (simplified)
            expected_return = np.mean(optimal_weights) * 0.1  # Simplified
            expected_volatility = np.std(optimal_weights) * 0.2  # Simplified
            sharpe_ratio = expected_return / expected_volatility if expected_volatility > 0 else 0.0

            # Generate rebalancing suggestions
            current_weights = self._calculate_current_weights(positions)
            rebalance_suggestions = {}
            for i, symbol in enumerate(symbols):
                weight_diff = optimal_weights[i] - current_weights.get(symbol, 0)
                if abs(weight_diff) > 0.05:  # 5% threshold
                    rebalance_suggestions[symbol] = weight_diff

            # Risk metrics
            risk_metrics = {
                'concentration_risk': np.sum(optimal_weights ** 2),
                'max_weight': np.max(optimal_weights),
                'min_weight': np.min(optimal_weights),
                'diversification_score': 1 / np.sum(optimal_weights ** 2)
            }

            optimization_result = PortfolioOptimization(
                optimal_weights=dict(zip(symbols, optimal_weights)),
                expected_return=expected_return,
                expected_volatility=expected_volatility,
                sharpe_ratio=sharpe_ratio,
                risk_metrics=risk_metrics,
                rebalance_suggestions=rebalance_suggestions,
                timestamp=datetime.now()
            )

            self.optimization_history.append(optimization_result)
            return optimization_result

        except Exception as e:
            print(f"Error in portfolio optimization: {e}")
            return PortfolioOptimization({}, 0.0, 0.0, 0.0, {}, {}, datetime.now())

    def _calculate_current_weights(self, positions: Dict[str, Position]) -> Dict[str, float]:
        """Calculate current portfolio weights"""
        total_value = sum(pos.quantity * pos.current_price for pos in positions.values() if pos.quantity > 0)
        if total_value == 0:
            return {}

        return {symbol: (pos.quantity * pos.current_price) / total_value 
                for symbol, pos in positions.items() if pos.quantity > 0}