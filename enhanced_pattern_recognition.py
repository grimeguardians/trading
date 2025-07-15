#!/usr/bin/env python3
"""
Enhanced Pattern Recognition Engine - Phase 5 Step 2
Real-time pattern recognition with knowledge graph integration
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import json

from knowledge_engine import DigitalBrain, MarketPattern

@dataclass
class EnhancedPatternMatch:
    """Enhanced pattern match with confidence scoring"""
    pattern_id: str
    pattern_type: str
    confidence: float
    market_context: Dict[str, Any]
    historical_performance: Dict[str, float]
    risk_assessment: Dict[str, float]
    entry_signals: List[str]
    exit_criteria: List[str]
    timeframe_validity: str
    volume_confirmation: bool
    support_resistance_levels: Dict[str, float]

class EnhancedPatternRecognition:
    """Advanced pattern recognition engine with knowledge graph integration"""

    def __init__(self, digital_brain: DigitalBrain):
        self.digital_brain = digital_brain
        self.logger = logging.getLogger("EnhancedPatternRecognition")
        self.pattern_cache = {}
        self.performance_tracker = defaultdict(list)

        # Pattern confidence thresholds (lowered for better signal generation)
        self.confidence_thresholds = {
            'reversal_patterns': 0.60,  # Lowered from 0.75
            'continuation_patterns': 0.55,  # Lowered from 0.70
            'breakout_patterns': 0.65,  # Lowered from 0.80
            'volume_patterns': 0.50   # Lowered from 0.65
        }

        # Real-time pattern templates
        self.pattern_templates = self._initialize_pattern_templates()

    def _initialize_pattern_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize enhanced pattern recognition templates"""
        return {
            'head_and_shoulders': {
                'category': 'reversal_patterns',
                'required_conditions': ['three_peaks', 'volume_decline', 'neckline_break'],
                'confirmation_signals': ['volume_spike_on_break', 'price_target_calculation'],
                'risk_factors': ['false_breakout', 'market_regime_change'],
                'success_rate_baseline': 0.85,
                'typical_duration': '2-8 weeks'
            },
            'double_bottom': {
                'category': 'reversal_patterns', 
                'required_conditions': ['two_equal_lows', 'volume_confirmation', 'resistance_break'],
                'confirmation_signals': ['higher_high', 'volume_expansion'],
                'risk_factors': ['third_test_failure', 'weak_volume'],
                'success_rate_baseline': 0.78,
                'typical_duration': '3-12 weeks'
            },
            'ascending_triangle': {
                'category': 'continuation_patterns',
                'required_conditions': ['horizontal_resistance', 'rising_lows', 'volume_pattern'],
                'confirmation_signals': ['breakout_with_volume', 'price_target_met'],
                'risk_factors': ['breakdown_below_support', 'volume_divergence'],
                'success_rate_baseline': 0.72,
                'typical_duration': '1-4 weeks'
            },
            'flag_pattern': {
                'category': 'continuation_patterns',
                'required_conditions': ['sharp_move', 'brief_consolidation', 'volume_decline'],
                'confirmation_signals': ['breakout_continuation', 'volume_return'],
                'risk_factors': ['extended_consolidation', 'trend_exhaustion'],
                'success_rate_baseline': 0.68,
                'typical_duration': '1-3 weeks'
            },
            'volume_climax': {
                'category': 'volume_patterns',
                'required_conditions': ['extreme_volume', 'price_exhaustion', 'reversal_signal'],
                'confirmation_signals': ['follow_through', 'momentum_shift'],
                'risk_factors': ['continuation_after_climax', 'false_reversal'],
                'success_rate_baseline': 0.70,
                'typical_duration': '1-5 days'
            }
        }

    def recognize_real_time_patterns(self, symbol: str, market_data: Dict[str, Any]) -> List[EnhancedPatternMatch]:
        """Enhanced real-time pattern recognition with knowledge graph integration"""
        try:
            # Query knowledge graph for relevant patterns
            pattern_query = {
                'symbol': symbol,
                'query_type': 'pattern',
                'min_confidence': 0.6,
                'include_historical': True
            }

            kg_patterns = self.digital_brain.knowledge_graph.query_knowledge(pattern_query)

            # Combine with real-time analysis
            enhanced_matches = []

            # Analyze each pattern template
            for pattern_name, template in self.pattern_templates.items():
                match = self._analyze_pattern_template(
                    pattern_name, template, symbol, market_data, kg_patterns
                )
                if match and match.confidence > self.confidence_thresholds.get(template['category'], 0.7):
                    enhanced_matches.append(match)

            # Sort by confidence and market context relevance
            enhanced_matches.sort(key=lambda x: x.confidence * self._calculate_context_relevance(x), reverse=True)

            return enhanced_matches[:5]  # Return top 5 matches

        except Exception as e:
            self.logger.error(f"Error in enhanced pattern recognition for {symbol}: {e}")
            return []

    def _analyze_pattern_template(self, pattern_name: str, template: Dict[str, Any], 
                                 symbol: str, market_data: Dict[str, Any], 
                                 kg_patterns: List[Dict[str, Any]]) -> Optional[EnhancedPatternMatch]:
        """Analyze a specific pattern template against market data"""
        try:
            # Base confidence from template
            base_confidence = template['success_rate_baseline']

            # Market data analysis
            price = market_data.get('price', 0)
            volume = market_data.get('volume', 0)
            rsi = market_data.get('rsi', 50)
            macd = market_data.get('macd', 0)
            support = market_data.get('support', 0)
            resistance = market_data.get('resistance', 0)

            # Pattern-specific analysis
            pattern_confidence = self._calculate_pattern_confidence(
                pattern_name, template, market_data
            )

            if pattern_confidence < 0.5:
                return None

            # Knowledge graph enhancement
            kg_boost = self._calculate_knowledge_graph_boost(pattern_name, kg_patterns)

            # Market context analysis
            market_context = self._analyze_market_context(symbol, market_data)

            # Historical performance lookup
            historical_perf = self._get_historical_performance(pattern_name, symbol)

            # Risk assessment
            risk_assessment = self._assess_pattern_risk(pattern_name, template, market_data)

            # Entry and exit signal generation
            entry_signals = self._generate_entry_signals(pattern_name, template, market_data)
            exit_criteria = self._generate_exit_criteria(pattern_name, template, market_data)

            # Final confidence calculation
            final_confidence = min(
                (base_confidence * 0.3 + 
                 pattern_confidence * 0.4 + 
                 kg_boost * 0.2 + 
                 market_context.get('regime_confidence', 0.5) * 0.1),
                0.95
            )

            # Volume confirmation
            avg_volume = market_data.get('avg_volume', volume)
            volume_confirmation = volume > (avg_volume * 1.2) if pattern_name in ['breakout', 'head_and_shoulders'] else True

            return EnhancedPatternMatch(
                pattern_id=f"{pattern_name}_{symbol}_{int(datetime.now().timestamp())}",
                pattern_type=pattern_name,
                confidence=final_confidence,
                market_context=market_context,
                historical_performance=historical_perf,
                risk_assessment=risk_assessment,
                entry_signals=entry_signals,
                exit_criteria=exit_criteria,
                timeframe_validity=template['typical_duration'],
                volume_confirmation=volume_confirmation,
                support_resistance_levels={
                    'support': support,
                    'resistance': resistance,
                    'current_price': price
                }
            )

        except Exception as e:
            self.logger.error(f"Error analyzing pattern template {pattern_name}: {e}")
            return None

    def _calculate_pattern_confidence(self, pattern_name: str, template: Dict[str, Any], 
                                    market_data: Dict[str, Any]) -> float:
        """Calculate pattern-specific confidence based on market data"""
        confidence_factors = []

        price = market_data.get('price', 0)
        volume = market_data.get('volume', 0)
        rsi = market_data.get('rsi', 50)
        macd = market_data.get('macd', 0)
        macd_signal = market_data.get('macd_signal', 0)
        support = market_data.get('support', 0)
        resistance = market_data.get('resistance', 0)

        if pattern_name == 'head_and_shoulders':
            # RSI overbought confirmation
            if rsi > 70:
                confidence_factors.append(0.8)
            elif rsi > 60:
                confidence_factors.append(0.6)

            # MACD divergence
            if macd < macd_signal:
                confidence_factors.append(0.7)

            # Price near resistance
            if resistance > 0 and abs(price - resistance) / price < 0.02:
                confidence_factors.append(0.9)

        elif pattern_name == 'double_bottom':
            # RSI oversold confirmation
            if rsi < 30:
                confidence_factors.append(0.8)
            elif rsi < 40:
                confidence_factors.append(0.6)

            # MACD bullish divergence
            if macd > macd_signal:
                confidence_factors.append(0.7)

            # Price near support
            if support > 0 and abs(price - support) / price < 0.02:
                confidence_factors.append(0.9)

        elif pattern_name == 'ascending_triangle':
            # Price compression near resistance
            if resistance > 0:
                distance_to_resistance = (resistance - price) / price
                if 0.01 < distance_to_resistance < 0.05:
                    confidence_factors.append(0.8)

            # Volume decline during formation
            avg_volume = market_data.get('avg_volume', volume)
            if volume < avg_volume * 0.8:
                confidence_factors.append(0.6)

        elif pattern_name == 'flag_pattern':
            # Short-term consolidation after move
            volatility = market_data.get('volatility', 0.2)
            if volatility < 0.15:  # Low volatility consolidation
                confidence_factors.append(0.7)

            # Volume decline during flag
            avg_volume = market_data.get('avg_volume', volume)
            if volume < avg_volume * 0.7:
                confidence_factors.append(0.8)

        elif pattern_name == 'volume_climax':
            # Extreme volume spike
            avg_volume = market_data.get('avg_volume', volume)
            if volume > avg_volume * 3:
                confidence_factors.append(0.9)
            elif volume > avg_volume * 2:
                confidence_factors.append(0.7)

        return np.mean(confidence_factors) if confidence_factors else 0.5

    def _calculate_knowledge_graph_boost(self, pattern_name: str, kg_patterns: List[Dict[str, Any]]) -> float:
        """Calculate confidence boost from knowledge graph patterns"""
        boost = 0.5  # Base boost

        for kg_pattern in kg_patterns:
            if kg_pattern.get('type') == 'node':
                node = kg_pattern.get('node')
                if node and node.node_type == 'chart_pattern':
                    pattern_type = node.attributes.get('pattern_name', '').lower()
                    if pattern_name.replace('_', ' ') in pattern_type:
                        # Found matching pattern in knowledge graph
                        kg_confidence = node.confidence
                        pattern_score = kg_pattern.get('score', 0.5)
                        boost = max(boost, (kg_confidence + pattern_score) / 2)

        return min(boost, 0.9)

    def learn_from_outcome(self, pattern_match: EnhancedPatternMatch, outcome: Dict[str, Any]):
        """Learn from pattern outcome to improve future recognition"""
        try:
            # Update success rate
            if outcome.get('successful', False):
                pattern_match.confidence = min(pattern_match.confidence * 1.1, 1.0)
            else:
                pattern_match.confidence = max(pattern_match.confidence * 0.9, 0.1)

            # Update brain knowledge if available
            if self.digital_brain:
                # Safe regime extraction from market context
                regime_value = 'unknown'
                if hasattr(pattern_match, 'market_context') and pattern_match.market_context:
                    market_conditions = pattern_match.market_context.get('market_conditions', {})
                    regime_value = market_conditions.get('regime', 'unknown')
                    # Ensure regime is a string
                    if regime_value and isinstance(regime_value, str):
                        regime_value = regime_value.lower()
                    else:
                        regime_value = 'unknown'

                learning_data = {
                    'pattern_type': pattern_match.pattern_type,
                    'symbol': pattern_match.symbol,
                    'confidence': pattern_match.confidence,
                    'outcome': outcome,
                    'regime': regime_value,  # Use 'regime' key instead of 'market_regime'
                    'timestamp': datetime.now().isoformat(),
                    'event_type': 'pattern_learning'
                }

                self.digital_brain.process_market_event(learning_data)

            self.logger.info(f"Updated pattern learning for {pattern_match.pattern_type}: "
                           f"confidence = {pattern_match.confidence:.3f}")

        except Exception as e:
            self.logger.error(f"Error in pattern learning: {e}")

    def _analyze_market_context(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current market context for pattern validity"""
        context = {
            'regime': 'normal',
            'regime_confidence': 0.5,
            'volatility_environment': 'normal',
            'trend_strength': 0.5,
            'market_phase': 'neutral'
        }

        try:
            volatility = market_data.get('volatility', 0.2)
            rsi = market_data.get('rsi', 50)
            macd = market_data.get('macd', 0)
            volume_ratio = market_data.get('volume_ratio', 1.0)

            # Volatility environment
            if volatility > 0.4:
                context['volatility_environment'] = 'high'
                context['regime_confidence'] *= 0.8  # Reduce confidence in high vol
            elif volatility < 0.1:
                context['volatility_environment'] = 'low'
                context['regime_confidence'] *= 1.1

            # Market regime detection
            if rsi > 70 and macd > 0:
                context['regime'] = 'overbought'
                context['market_phase'] = 'distribution'
            elif rsi < 30 and macd < 0:
                context['regime'] = 'oversold'
                context['market_phase'] = 'accumulation'
            elif 40 < rsi < 60:
                context['regime'] = 'neutral'
                context['market_phase'] = 'neutral'

            # Trend strength
            if abs(macd) > 2:
                context['trend_strength'] = 0.8
            elif abs(macd) > 1:
                context['trend_strength'] = 0.6

            # Volume context
            if volume_ratio > 1.5:
                context['volume_environment'] = 'high'
            elif volume_ratio < 0.7:
                context['volume_environment'] = 'low'
            else:
                context['volume_environment'] = 'normal'

        except Exception as e:
            self.logger.error(f"Error analyzing market context: {e}")

        return context

    def _get_historical_performance(self, pattern_name: str, symbol: str) -> Dict[str, float]:
        """Get historical performance data for a pattern"""
        try:
            # Query knowledge graph for historical performance
            if self.digital_brain and self.digital_brain.knowledge_graph:
                query = {
                    'pattern_type': pattern_name,
                    'symbol': symbol,
                    'query_type': 'pattern',
                    'min_confidence': 0.5
                }
                
                results = self.digital_brain.knowledge_graph.query_knowledge(query)
                
                if results:
                    # Calculate average performance from knowledge graph
                    total_success = 0
                    total_samples = 0
                    
                    for result in results:
                        if result.get('type') == 'node':
                            node = result.get('node')
                            if node and hasattr(node, 'attributes'):
                                success_rate = node.attributes.get('success_rate', 0.7)
                                sample_size = node.attributes.get('sample_size', 1)
                                total_success += success_rate * sample_size
                                total_samples += sample_size
                    
                    if total_samples > 0:
                        avg_success_rate = total_success / total_samples
                        return {
                            'success_rate': avg_success_rate,
                            'sample_size': total_samples,
                            'confidence': min(avg_success_rate * 1.1, 1.0)
                        }
            
            # Default historical performance based on pattern type
            defaults = {
                'head_and_shoulders': {'success_rate': 0.85, 'sample_size': 100, 'confidence': 0.8},
                'double_bottom': {'success_rate': 0.78, 'sample_size': 80, 'confidence': 0.75},
                'ascending_triangle': {'success_rate': 0.72, 'sample_size': 60, 'confidence': 0.7},
                'flag_pattern': {'success_rate': 0.68, 'sample_size': 50, 'confidence': 0.65},
                'volume_climax': {'success_rate': 0.70, 'sample_size': 40, 'confidence': 0.68}
            }
            
            return defaults.get(pattern_name, {'success_rate': 0.6, 'sample_size': 10, 'confidence': 0.5})
            
        except Exception as e:
            self.logger.error(f"Error getting historical performance for {pattern_name}: {e}")
            return {'success_rate': 0.6, 'sample_size': 10, 'confidence': 0.5}

    def _assess_pattern_risk(self, pattern_name: str, template: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, float]:
        """Assess risk factors for a pattern"""
        try:
            risk_assessment = {
                'overall_risk': 0.3,  # Base risk
                'market_risk': 0.2,
                'pattern_risk': 0.1,
                'timing_risk': 0.1,
                'liquidity_risk': 0.05
            }
            
            # Market-based risk factors
            volatility = market_data.get('volatility', 0.2)
            volume_ratio = market_data.get('volume_ratio', 1.0)
            
            # High volatility increases risk
            if volatility > 0.4:
                risk_assessment['market_risk'] += 0.2
                risk_assessment['overall_risk'] += 0.1
            
            # Low volume increases liquidity risk
            if volume_ratio < 0.8:
                risk_assessment['liquidity_risk'] += 0.1
                risk_assessment['overall_risk'] += 0.05
            
            # Pattern-specific risk factors
            pattern_risks = {
                'head_and_shoulders': 0.25,  # Lower risk, high reliability
                'double_bottom': 0.3,
                'ascending_triangle': 0.35,
                'flag_pattern': 0.4,  # Higher risk, shorter duration
                'volume_climax': 0.45  # Highest risk, timing critical
            }
            
            pattern_risk = pattern_risks.get(pattern_name, 0.4)
            risk_assessment['pattern_risk'] = pattern_risk
            risk_assessment['overall_risk'] = min(
                (risk_assessment['market_risk'] + pattern_risk + risk_assessment['timing_risk'] + risk_assessment['liquidity_risk']) / 4,
                0.8
            )
            
            return risk_assessment
            
        except Exception as e:
            self.logger.error(f"Error assessing pattern risk for {pattern_name}: {e}")
            return {'overall_risk': 0.5, 'market_risk': 0.3, 'pattern_risk': 0.2, 'timing_risk': 0.1, 'liquidity_risk': 0.1}

    def _generate_entry_signals(self, pattern_name: str, template: Dict[str, Any], market_data: Dict[str, Any]) -> List[str]:
        """Generate entry signals for a pattern"""
        try:
            signals = []
            
            price = market_data.get('price', 0)
            rsi = market_data.get('rsi', 50)
            macd = market_data.get('macd', 0)
            support = market_data.get('support', 0)
            resistance = market_data.get('resistance', 0)
            volume_ratio = market_data.get('volume_ratio', 1.0)
            
            # Pattern-specific entry signals
            if pattern_name == 'head_and_shoulders':
                if resistance > 0 and price < resistance * 0.98:
                    signals.append(f"Wait for neckline break below ${resistance * 0.95:.2f}")
                if rsi > 60:
                    signals.append("RSI confirms bearish divergence")
                if volume_ratio > 1.2:
                    signals.append("Volume spike confirms selling pressure")
            
            elif pattern_name == 'double_bottom':
                if support > 0 and price > support * 1.02:
                    signals.append(f"Buy on break above ${support * 1.03:.2f}")
                if rsi < 40:
                    signals.append("RSI oversold supports reversal")
                if macd > 0:
                    signals.append("MACD bullish crossover confirms signal")
            
            elif pattern_name == 'ascending_triangle':
                if resistance > 0:
                    signals.append(f"Buy breakout above ${resistance * 1.01:.2f}")
                if volume_ratio > 1.3:
                    signals.append("Volume expansion confirms breakout")
                
            elif pattern_name == 'flag_pattern':
                signals.append("Buy continuation of main trend")
                if volume_ratio > 1.5:
                    signals.append("Strong volume confirms trend resumption")
                    
            elif pattern_name == 'volume_climax':
                if volume_ratio > 2.0:
                    signals.append("Extreme volume indicates potential reversal")
                signals.append("Wait for follow-through confirmation")
            
            # Generic signals if no specific ones
            if not signals:
                signals.append(f"Monitor {pattern_name} pattern development")
                signals.append("Wait for volume confirmation")
            
            return signals[:3]  # Return top 3 signals
            
        except Exception as e:
            self.logger.error(f"Error generating entry signals for {pattern_name}: {e}")
            return [f"Monitor {pattern_name} pattern", "Wait for confirmation"]

    def _generate_exit_criteria(self, pattern_name: str, template: Dict[str, Any], market_data: Dict[str, Any]) -> List[str]:
        """Generate exit criteria for a pattern"""
        try:
            criteria = []
            
            price = market_data.get('price', 0)
            support = market_data.get('support', 0)
            resistance = market_data.get('resistance', 0)
            
            # Pattern-specific exit criteria
            if pattern_name == 'head_and_shoulders':
                if resistance > 0:
                    target = resistance - (resistance - support) if support > 0 else resistance * 0.9
                    criteria.append(f"Target: ${target:.2f}")
                    criteria.append(f"Stop-loss: ${resistance * 1.02:.2f}")
                    
            elif pattern_name == 'double_bottom':
                if support > 0 and resistance > 0:
                    target = support + (resistance - support) * 1.5
                    criteria.append(f"Target: ${target:.2f}")
                    criteria.append(f"Stop-loss: ${support * 0.98:.2f}")
                    
            elif pattern_name == 'ascending_triangle':
                if resistance > 0:
                    target = resistance * 1.1
                    criteria.append(f"Target: ${target:.2f}")
                    criteria.append(f"Stop-loss: ${resistance * 0.95:.2f}")
                    
            elif pattern_name == 'flag_pattern':
                criteria.append("Exit on trend exhaustion")
                criteria.append("Trail stop-loss with trend")
                
            elif pattern_name == 'volume_climax':
                criteria.append("Exit on volume normalization")
                criteria.append("Quick profit-taking recommended")
            
            # Generic exit criteria
            if not criteria:
                criteria.append("Use 3% stop-loss")
                criteria.append("Take profits at 6% gain")
            
            return criteria[:2]  # Return top 2 criteria
            
        except Exception as e:
            self.logger.error(f"Error generating exit criteria for {pattern_name}: {e}")
            return ["Use standard stop-loss", "Take profits at resistance"]

    def _calculate_context_relevance(self, pattern_match: EnhancedPatternMatch) -> float:
        """Calculate context relevance for pattern match"""
        try:
            relevance = 0.5  # Base relevance
            
            # Market context relevance
            if hasattr(pattern_match, 'market_context'):
                context = pattern_match.market_context
                
                # Regime alignment
                regime = context.get('regime', 'normal')
                if regime in ['trending', 'bullish', 'bearish']:
                    relevance += 0.2
                elif regime == 'volatile':
                    relevance -= 0.1
                
                # Volume confirmation
                if pattern_match.volume_confirmation:
                    relevance += 0.2
                
                # Risk assessment
                risk_level = pattern_match.risk_assessment.get('overall_risk', 0.5)
                relevance += (1.0 - risk_level) * 0.1
            
            return min(relevance, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating context relevance: {e}")
            return 0.5

    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of pattern recognition performance"""
        try:
            return {
                'total_pattern_templates': len(self.pattern_templates),
                'active_pattern_matches': len(self.pattern_cache),
                'average_success_rate': np.mean([
                    template['success_rate_baseline'] 
                    for template in self.pattern_templates.values()
                ]) if self.pattern_templates else 0,
                'performance_tracker_entries': sum(len(entries) for entries in self.performance_tracker.values()),
                'confidence_thresholds': self.confidence_thresholds
            }
        except Exception as e:
            self.logger.error(f"Error getting pattern summary: {e}")
            return {
                'total_pattern_templates': 0,
                'active_pattern_matches': 0,
                'average_success_rate': 0,
                'performance_tracker_entries': 0,
                'confidence_thresholds': {}
            }

    def _create_pattern_context(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create contextual information for pattern analysis"""
        context = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'market_conditions': {},
            'risk_factors': []
        }

        # Market regime context with safe handling
        if hasattr(self, 'market_regime_detector'):
            try:
                regime_signal = self.market_regime_detector.detect_regime(symbol)
                if regime_signal and regime_signal is not None:
                    # Safe regime value extraction
                    regime_value = 'unknown'
                    if hasattr(regime_signal, 'regime') and regime_signal.regime is not None:
                        if hasattr(regime_signal.regime, 'value'):
                            regime_value = regime_signal.regime.value
                        elif hasattr(regime_signal.regime, 'name'):
                            regime_value = regime_signal.regime.name.lower()
                        else:
                            regime_str = str(regime_signal.regime)
                            if regime_str and regime_str.lower() != 'none':
                                regime_value = regime_str.lower().replace('marketregime.', '')

                    context['market_conditions']['regime'] = regime_value
                    context['market_conditions']['regime_confidence'] = getattr(regime_signal, 'confidence', 0.5)
                    context['market_conditions']['regime_strength'] = getattr(regime_signal, 'strength', 0.5)
                else:
                    context['market_conditions']['regime'] = 'unknown'
                    context['market_conditions']['regime_confidence'] = 0.5
                    context['market_conditions']['regime_strength'] = 0.5
            except Exception as e:
                self.logger.warning(f"Error detecting market regime for {symbol}: {e}")
                context['market_conditions']['regime'] = 'unknown'
                context['market_conditions']['regime_confidence'] = 0.5
                context['market_conditions']['regime_strength'] = 0.5

        return context

def main():
    """Test enhanced pattern recognition"""
    print("🔍 Enhanced Pattern Recognition Engine Testing")
    print("=" * 60)

    # Initialize with digital brain
    from knowledge_engine import DigitalBrain
    brain = DigitalBrain()

    # Load existing knowledge graph state
    import os
    if os.path.exists('knowledge_graph_state.json'):
        success = brain.knowledge_graph.load_from_file('knowledge_graph_state.json')
        if success:
            print("✅ Loaded populated knowledge graph")
        else:
            print("⚠️ Failed to load knowledge graph")

    # Initialize enhanced pattern recognition
    enhanced_pr = EnhancedPatternRecognition(brain)

    # Test with sample market data
    test_market_data = {
        'symbol': 'AAPL',
        'price': 150.25,
        'volume': 45000000,
        'avg_volume': 35000000,
        'rsi': 68.5,
        'macd': 1.25,
        'macd_signal': 0.85,
        'support': 148.50,
        'resistance': 152.75,
        'volatility': 0.28,
        'atr': 3.20,
        'volume_ratio': 1.29
    }

    print(f"\n🔍 Testing Pattern Recognition on {test_market_data['symbol']}:")
    print(f"   Price: ${test_market_data['price']}")
    print(f"   RSI: {test_market_data['rsi']}")
    print(f"   Volume: {test_market_data['volume']:,} ({test_market_data['volume_ratio']:.1f}x avg)")

    # Recognize patterns
    patterns = enhanced_pr.recognize_real_time_patterns('AAPL', test_market_data)

    print(f"\n📊 Pattern Recognition Results:")
    print(f"   Patterns Found: {len(patterns)}")

    for i, pattern in enumerate(patterns[:3], 1):
        print(f"\n   {i}. {pattern.pattern_type.upper()}")
        print(f"      Confidence: {pattern.confidence:.2%}")
        print(f"      Market Context: {pattern.market_context.get('regime', 'normal')}")
        print(f"      Success Rate: {pattern.historical_performance.get('success_rate', 0):.1%}")
        print(f"      Volume Confirmed: {'✅' if pattern.volume_confirmation else '❌'}")
        print(f"      Risk Level: {pattern.risk_assessment.get('overall_risk', 0):.1%}")

        if pattern.entry_signals:
            print(f"      Entry Signal: {pattern.entry_signals[0]}")

        if pattern.exit_criteria:
            print(f"      Exit Criteria: {pattern.exit_criteria[0]}")

    # Get summary
    summary = enhanced_pr.get_pattern_summary()
    print(f"\n📈 Pattern Recognition Summary:")
    print(f"   Pattern Templates: {summary.get('total_pattern_templates', 0)}")
    print(f"   Active Matches: {summary.get('active_pattern_matches', 0)}")
    print(f"   Average Success Rate: {summary.get('average_success_rate', 0):.1%}")

if __name__ == "__main__":
    main()