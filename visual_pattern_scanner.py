
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import base64

@dataclass
class VisualPattern:
    """Represents a visually detected pattern"""
    pattern_type: str
    confidence: float
    coordinates: List[Tuple[int, int]]  # Key points
    price_levels: List[float]
    volume_confirmation: bool
    pattern_image: Optional[str] = None  # Base64 encoded image
    metadata: Dict[str, Any] = None

@dataclass
class ChartConfiguration:
    """Configuration for chart analysis"""
    timeframe: str = "1D"
    lookback_periods: int = 50
    min_pattern_length: int = 10
    confidence_threshold: float = 0.7
    enable_volume_analysis: bool = True
    pattern_types: List[str] = None

class VisualPatternScanner:
    """Advanced visual pattern recognition system for trading charts"""
    
    def __init__(self, knowledge_engine=None):
        self.knowledge_engine = knowledge_engine
        self.logger = logging.getLogger("VisualPatternScanner")
        self.pattern_templates = self._initialize_pattern_templates()
        self.scan_history = {}
        
    def _initialize_pattern_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize visual pattern templates"""
        return {
            'head_and_shoulders': {
                'description': 'Three peaks with middle peak highest',
                'key_points': 5,  # Left shoulder, head, right shoulder, neckline points
                'min_duration': 15,
                'volume_pattern': 'decreasing',
                'confidence_factors': ['symmetry', 'volume_confirmation', 'neckline_break']
            },
            'double_top': {
                'description': 'Two peaks at similar levels',
                'key_points': 4,  # Two peaks, valley, support line
                'min_duration': 10,
                'volume_pattern': 'decreasing_on_second_peak',
                'confidence_factors': ['peak_similarity', 'volume_divergence', 'support_break']
            },
            'ascending_triangle': {
                'description': 'Rising lows with horizontal resistance',
                'key_points': 6,  # Multiple touch points
                'min_duration': 12,
                'volume_pattern': 'increasing',
                'confidence_factors': ['trend_line_touches', 'volume_expansion', 'breakout_volume']
            },
            'flag_pattern': {
                'description': 'Brief consolidation after strong move',
                'key_points': 4,  # Flag boundaries
                'min_duration': 5,
                'volume_pattern': 'decreasing_then_expanding',
                'confidence_factors': ['flag_slope', 'volume_pattern', 'breakout_direction']
            },
            'cup_and_handle': {
                'description': 'U-shaped pattern with small consolidation',
                'key_points': 6,  # Cup bottom, handle points
                'min_duration': 20,
                'volume_pattern': 'decreasing_in_handle',
                'confidence_factors': ['cup_depth', 'handle_duration', 'volume_confirmation']
            }
        }
    
    def configure_scanner(self, config: ChartConfiguration):
        """Configure the visual scanner parameters"""
        self.config = config
        if config.pattern_types is None:
            self.config.pattern_types = list(self.pattern_templates.keys())
        
        self.logger.info(f"Scanner configured for {len(self.config.pattern_types)} pattern types")
    
    def scan_chart_for_patterns(self, symbol: str, price_data: pd.DataFrame, 
                               volume_data: pd.DataFrame = None) -> List[VisualPattern]:
        """Scan chart data for visual patterns"""
        detected_patterns = []
        
        try:
            # Prepare data
            if len(price_data) < self.config.min_pattern_length:
                return detected_patterns
            
            # Create price chart visualization
            chart_image = self._create_chart_image(symbol, price_data, volume_data)
            
            # Scan for each pattern type
            for pattern_type in self.config.pattern_types:
                patterns = self._detect_pattern_type(
                    pattern_type, price_data, volume_data, chart_image
                )
                detected_patterns.extend(patterns)
            
            # Store scan results
            self.scan_history[symbol] = {
                'timestamp': datetime.now(),
                'patterns_found': len(detected_patterns),
                'high_confidence_patterns': len([p for p in detected_patterns if p.confidence > 0.8])
            }
            
            return detected_patterns
            
        except Exception as e:
            self.logger.error(f"Error scanning chart for {symbol}: {e}")
            return []
    
    def _create_chart_image(self, symbol: str, price_data: pd.DataFrame, 
                           volume_data: pd.DataFrame = None) -> str:
        """Create chart image for pattern analysis"""
        fig, axes = plt.subplots(2 if volume_data is not None else 1, 1, 
                                figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1] if volume_data is not None else [1]})
        
        if volume_data is not None:
            price_ax, volume_ax = axes
        else:
            price_ax = axes
            volume_ax = None
        
        # Plot price data
        dates = price_data.index
        prices = price_data['close'] if 'close' in price_data.columns else price_data.iloc[:, 0]
        
        price_ax.plot(dates, prices, linewidth=1.5, color='blue', label='Price')
        price_ax.set_title(f'{symbol} - Pattern Analysis Chart')
        price_ax.set_ylabel('Price')
        price_ax.grid(True, alpha=0.3)
        price_ax.legend()
        
        # Plot volume if available
        if volume_data is not None and volume_ax is not None:
            volumes = volume_data['volume'] if 'volume' in volume_data.columns else volume_data.iloc[:, 0]
            volume_ax.bar(dates, volumes, alpha=0.6, color='gray', label='Volume')
            volume_ax.set_ylabel('Volume')
            volume_ax.set_xlabel('Date')
            volume_ax.legend()
        
        # Convert to base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)
        
        return image_base64
    
    def _detect_pattern_type(self, pattern_type: str, price_data: pd.DataFrame, 
                            volume_data: pd.DataFrame, chart_image: str) -> List[VisualPattern]:
        """Detect specific pattern type using multiple methods"""
        patterns = []
        template = self.pattern_templates.get(pattern_type)
        
        if not template:
            return patterns
        
        prices = price_data['close'] if 'close' in price_data.columns else price_data.iloc[:, 0]
        
        # Different detection methods based on pattern type
        if pattern_type == 'head_and_shoulders':
            patterns.extend(self._detect_head_and_shoulders(prices, volume_data, chart_image))
        elif pattern_type == 'double_top':
            patterns.extend(self._detect_double_top(prices, volume_data, chart_image))
        elif pattern_type == 'ascending_triangle':
            patterns.extend(self._detect_ascending_triangle(prices, volume_data, chart_image))
        elif pattern_type == 'flag_pattern':
            patterns.extend(self._detect_flag_pattern(prices, volume_data, chart_image))
        elif pattern_type == 'cup_and_handle':
            patterns.extend(self._detect_cup_and_handle(prices, volume_data, chart_image))
        
        return patterns
    
    def _detect_head_and_shoulders(self, prices: pd.Series, volume_data: pd.DataFrame, 
                                  chart_image: str) -> List[VisualPattern]:
        """Detect head and shoulders pattern"""
        patterns = []
        
        try:
            # Find peaks using simple peak detection
            peaks = self._find_peaks(prices.values, min_distance=5)
            
            if len(peaks) < 3:
                return patterns
            
            # Look for head and shoulders formation
            for i in range(len(peaks) - 2):
                left_shoulder = peaks[i]
                head = peaks[i + 1]
                right_shoulder = peaks[i + 2]
                
                # Check if middle peak is highest (head)
                if (prices.iloc[head] > prices.iloc[left_shoulder] and 
                    prices.iloc[head] > prices.iloc[right_shoulder]):
                    
                    # Check shoulder symmetry
                    left_height = prices.iloc[left_shoulder]
                    right_height = prices.iloc[right_shoulder]
                    height_ratio = min(left_height, right_height) / max(left_height, right_height)
                    
                    if height_ratio > 0.9:  # Shoulders should be similar height
                        # Calculate confidence
                        confidence = self._calculate_hs_confidence(
                            prices, left_shoulder, head, right_shoulder, volume_data
                        )
                        
                        if confidence > self.config.confidence_threshold:
                            pattern = VisualPattern(
                                pattern_type='head_and_shoulders',
                                confidence=confidence,
                                coordinates=[(left_shoulder, int(left_height)), 
                                           (head, int(prices.iloc[head])),
                                           (right_shoulder, int(right_height))],
                                price_levels=[left_height, prices.iloc[head], right_height],
                                volume_confirmation=self._check_volume_confirmation(volume_data, [left_shoulder, head, right_shoulder]),
                                pattern_image=chart_image,
                                metadata={
                                    'neckline_level': (left_height + right_height) / 2,
                                    'pattern_duration': right_shoulder - left_shoulder
                                }
                            )
                            patterns.append(pattern)
            
        except Exception as e:
            self.logger.error(f"Error detecting head and shoulders: {e}")
        
        return patterns
    
    def _detect_double_top(self, prices: pd.Series, volume_data: pd.DataFrame, 
                          chart_image: str) -> List[VisualPattern]:
        """Detect double top pattern"""
        patterns = []
        
        try:
            peaks = self._find_peaks(prices.values, min_distance=5)
            
            if len(peaks) < 2:
                return patterns
            
            # Look for double top formation
            for i in range(len(peaks) - 1):
                first_peak = peaks[i]
                second_peak = peaks[i + 1]
                
                # Check if peaks are at similar levels
                peak1_price = prices.iloc[first_peak]
                peak2_price = prices.iloc[second_peak]
                price_ratio = min(peak1_price, peak2_price) / max(peak1_price, peak2_price)
                
                if price_ratio > 0.98:  # Peaks should be very similar
                    # Find valley between peaks
                    valley_start = first_peak
                    valley_end = second_peak
                    valley_idx = valley_start + np.argmin(prices.iloc[valley_start:valley_end].values)
                    valley_price = prices.iloc[valley_idx]
                    
                    # Calculate pattern strength
                    peak_to_valley_ratio = valley_price / max(peak1_price, peak2_price)
                    
                    if peak_to_valley_ratio < 0.95:  # Significant valley
                        confidence = self._calculate_double_top_confidence(
                            prices, first_peak, second_peak, valley_idx, volume_data
                        )
                        
                        if confidence > self.config.confidence_threshold:
                            pattern = VisualPattern(
                                pattern_type='double_top',
                                confidence=confidence,
                                coordinates=[(first_peak, int(peak1_price)), 
                                           (valley_idx, int(valley_price)),
                                           (second_peak, int(peak2_price))],
                                price_levels=[peak1_price, valley_price, peak2_price],
                                volume_confirmation=self._check_volume_confirmation(volume_data, [first_peak, second_peak]),
                                pattern_image=chart_image,
                                metadata={
                                    'support_level': valley_price,
                                    'resistance_level': max(peak1_price, peak2_price)
                                }
                            )
                            patterns.append(pattern)
        
        except Exception as e:
            self.logger.error(f"Error detecting double top: {e}")
        
        return patterns
    
    def _detect_ascending_triangle(self, prices: pd.Series, volume_data: pd.DataFrame, 
                                  chart_image: str) -> List[VisualPattern]:
        """Detect ascending triangle pattern"""
        patterns = []
        
        try:
            # Find recent highs and lows
            window = min(20, len(prices) // 2)
            highs = []
            lows = []
            
            for i in range(window, len(prices) - window):
                if prices.iloc[i] == prices.iloc[i-window:i+window].max():
                    highs.append((i, prices.iloc[i]))
                if prices.iloc[i] == prices.iloc[i-window:i+window].min():
                    lows.append((i, prices.iloc[i]))
            
            if len(highs) < 2 or len(lows) < 2:
                return patterns
            
            # Check for horizontal resistance (similar highs)
            high_prices = [h[1] for h in highs]
            if len(high_prices) >= 2:
                resistance_level = np.mean(high_prices)
                resistance_variance = np.std(high_prices) / resistance_level
                
                if resistance_variance < 0.02:  # Tight horizontal resistance
                    # Check for ascending support (rising lows)
                    if len(lows) >= 2:
                        low_indices = [l[0] for l in lows]
                        low_prices = [l[1] for l in lows]
                        
                        # Linear regression for trend
                        slope = np.polyfit(low_indices, low_prices, 1)[0]
                        
                        if slope > 0:  # Rising lows
                            confidence = self._calculate_triangle_confidence(
                                highs, lows, volume_data, 'ascending'
                            )
                            
                            if confidence > self.config.confidence_threshold:
                                pattern = VisualPattern(
                                    pattern_type='ascending_triangle',
                                    confidence=confidence,
                                    coordinates=highs + lows,
                                    price_levels=[resistance_level] + low_prices,
                                    volume_confirmation=self._check_volume_trend(volume_data, low_indices),
                                    pattern_image=chart_image,
                                    metadata={
                                        'resistance_level': resistance_level,
                                        'support_slope': slope,
                                        'breakout_target': resistance_level * 1.05
                                    }
                                )
                                patterns.append(pattern)
        
        except Exception as e:
            self.logger.error(f"Error detecting ascending triangle: {e}")
        
        return patterns
    
    def _detect_flag_pattern(self, prices: pd.Series, volume_data: pd.DataFrame, 
                            chart_image: str) -> List[VisualPattern]:
        """Detect flag pattern"""
        patterns = []
        
        try:
            # Look for strong moves followed by consolidation
            window = 10
            
            for i in range(window, len(prices) - window):
                # Check for strong prior move
                prior_move = (prices.iloc[i] - prices.iloc[i-window]) / prices.iloc[i-window]
                
                if abs(prior_move) > 0.05:  # 5% move
                    # Check for consolidation period
                    consolidation_start = i
                    consolidation_end = min(i + window, len(prices) - 1)
                    
                    consolidation_prices = prices.iloc[consolidation_start:consolidation_end]
                    price_range = (consolidation_prices.max() - consolidation_prices.min()) / consolidation_prices.mean()
                    
                    if price_range < 0.03:  # Tight consolidation
                        confidence = self._calculate_flag_confidence(
                            prices, consolidation_start, consolidation_end, prior_move, volume_data
                        )
                        
                        if confidence > self.config.confidence_threshold:
                            pattern = VisualPattern(
                                pattern_type='flag_pattern',
                                confidence=confidence,
                                coordinates=[(i-window, int(prices.iloc[i-window])), 
                                           (consolidation_start, int(prices.iloc[consolidation_start])),
                                           (consolidation_end, int(prices.iloc[consolidation_end]))],
                                price_levels=[prices.iloc[i-window], consolidation_prices.mean()],
                                volume_confirmation=self._check_volume_pattern(volume_data, consolidation_start, consolidation_end),
                                pattern_image=chart_image,
                                metadata={
                                    'prior_move_percent': prior_move * 100,
                                    'consolidation_duration': consolidation_end - consolidation_start,
                                    'flag_direction': 'bullish' if prior_move > 0 else 'bearish'
                                }
                            )
                            patterns.append(pattern)
        
        except Exception as e:
            self.logger.error(f"Error detecting flag pattern: {e}")
        
        return patterns
    
    def _detect_cup_and_handle(self, prices: pd.Series, volume_data: pd.DataFrame, 
                              chart_image: str) -> List[VisualPattern]:
        """Detect cup and handle pattern"""
        patterns = []
        
        try:
            min_length = 30  # Minimum length for cup and handle
            
            if len(prices) < min_length:
                return patterns
            
            # Look for U-shaped formation
            for start in range(0, len(prices) - min_length):
                end = min(start + min_length, len(prices) - 1)
                segment = prices.iloc[start:end]
                
                # Find potential cup bottom
                bottom_idx = start + np.argmin(segment.values)
                left_high = prices.iloc[start]
                right_high = prices.iloc[end-1]
                bottom_price = prices.iloc[bottom_idx]
                
                # Check cup criteria
                cup_depth = (max(left_high, right_high) - bottom_price) / max(left_high, right_high)
                
                if 0.1 < cup_depth < 0.5:  # 10-50% retracement
                    # Look for handle formation after cup
                    handle_start = end
                    handle_end = min(handle_start + 10, len(prices) - 1)
                    
                    if handle_end < len(prices):
                        handle_segment = prices.iloc[handle_start:handle_end]
                        handle_low = handle_segment.min()
                        
                        # Handle should be above cup bottom
                        if handle_low > bottom_price:
                            confidence = self._calculate_cup_handle_confidence(
                                prices, start, bottom_idx, end, handle_start, handle_end, volume_data
                            )
                            
                            if confidence > self.config.confidence_threshold:
                                pattern = VisualPattern(
                                    pattern_type='cup_and_handle',
                                    confidence=confidence,
                                    coordinates=[(start, int(left_high)), 
                                               (bottom_idx, int(bottom_price)),
                                               (end, int(right_high)),
                                               (handle_end, int(handle_segment.iloc[-1]))],
                                    price_levels=[left_high, bottom_price, right_high, handle_low],
                                    volume_confirmation=self._check_cup_handle_volume(volume_data, start, end, handle_start, handle_end),
                                    pattern_image=chart_image,
                                    metadata={
                                        'cup_depth_percent': cup_depth * 100,
                                        'handle_duration': handle_end - handle_start,
                                        'breakout_target': max(left_high, right_high) * 1.1
                                    }
                                )
                                patterns.append(pattern)
        
        except Exception as e:
            self.logger.error(f"Error detecting cup and handle: {e}")
        
        return patterns
    
    def _find_peaks(self, data: np.array, min_distance: int = 5) -> List[int]:
        """Find peaks in price data"""
        peaks = []
        
        for i in range(min_distance, len(data) - min_distance):
            is_peak = True
            
            # Check if current point is higher than surrounding points
            for j in range(i - min_distance, i + min_distance + 1):
                if j != i and data[j] >= data[i]:
                    is_peak = False
                    break
            
            if is_peak:
                peaks.append(i)
        
        return peaks
    
    # Confidence calculation methods for each pattern type
    def _calculate_hs_confidence(self, prices: pd.Series, left: int, head: int, 
                                right: int, volume_data: pd.DataFrame) -> float:
        """Calculate head and shoulders pattern confidence"""
        confidence = 0.0
        
        # Symmetry factor
        left_height = prices.iloc[left]
        right_height = prices.iloc[right]
        symmetry = min(left_height, right_height) / max(left_height, right_height)
        confidence += symmetry * 0.4
        
        # Head prominence
        head_height = prices.iloc[head]
        avg_shoulder = (left_height + right_height) / 2
        head_prominence = (head_height - avg_shoulder) / avg_shoulder
        confidence += min(head_prominence, 0.3) * 0.3
        
        # Volume confirmation
        if volume_data is not None:
            volume_conf = self._check_volume_confirmation(volume_data, [left, head, right])
            confidence += 0.3 if volume_conf else 0.0
        
        return min(confidence, 1.0)
    
    def _calculate_double_top_confidence(self, prices: pd.Series, peak1: int, peak2: int, 
                                        valley: int, volume_data: pd.DataFrame) -> float:
        """Calculate double top pattern confidence"""
        confidence = 0.0
        
        # Peak similarity
        peak1_price = prices.iloc[peak1]
        peak2_price = prices.iloc[peak2]
        similarity = min(peak1_price, peak2_price) / max(peak1_price, peak2_price)
        confidence += similarity * 0.5
        
        # Valley depth
        valley_price = prices.iloc[valley]
        valley_depth = (max(peak1_price, peak2_price) - valley_price) / max(peak1_price, peak2_price)
        confidence += min(valley_depth, 0.2) * 0.3
        
        # Volume divergence (second peak should have lower volume)
        if volume_data is not None:
            try:
                vol1 = volume_data.iloc[peak1]['volume'] if 'volume' in volume_data.columns else volume_data.iloc[peak1, 0]
                vol2 = volume_data.iloc[peak2]['volume'] if 'volume' in volume_data.columns else volume_data.iloc[peak2, 0]
                if vol2 < vol1:
                    confidence += 0.2
            except:
                pass
        
        return min(confidence, 1.0)
    
    def _calculate_triangle_confidence(self, highs: List[Tuple], lows: List[Tuple], 
                                     volume_data: pd.DataFrame, triangle_type: str) -> float:
        """Calculate triangle pattern confidence"""
        confidence = 0.5  # Base confidence
        
        # Touch points (more touches = higher confidence)
        total_touches = len(highs) + len(lows)
        confidence += min(total_touches / 10, 0.3)
        
        # Volume trend (should increase toward breakout)
        if volume_data is not None and self._check_volume_trend(volume_data, [l[0] for l in lows]):
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _calculate_flag_confidence(self, prices: pd.Series, start: int, end: int, 
                                  prior_move: float, volume_data: pd.DataFrame) -> float:
        """Calculate flag pattern confidence"""
        confidence = 0.3  # Base confidence
        
        # Prior move strength
        confidence += min(abs(prior_move), 0.1) * 2  # Up to 0.2
        
        # Consolidation tightness
        consolidation = prices.iloc[start:end]
        tightness = 1 - ((consolidation.max() - consolidation.min()) / consolidation.mean())
        confidence += tightness * 0.3
        
        # Volume pattern
        if volume_data is not None and self._check_volume_pattern(volume_data, start, end):
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _calculate_cup_handle_confidence(self, prices: pd.Series, cup_start: int, bottom: int, 
                                       cup_end: int, handle_start: int, handle_end: int, 
                                       volume_data: pd.DataFrame) -> float:
        """Calculate cup and handle pattern confidence"""
        confidence = 0.3  # Base confidence
        
        # Cup shape (should be rounded)
        cup_segment = prices.iloc[cup_start:cup_end]
        bottom_price = prices.iloc[bottom]
        
        # Check for U-shape vs V-shape
        smoothness = self._calculate_cup_smoothness(cup_segment, bottom)
        confidence += smoothness * 0.3
        
        # Duration appropriateness
        cup_duration = cup_end - cup_start
        if 20 <= cup_duration <= 60:  # Good duration
            confidence += 0.2
        
        # Volume pattern
        if volume_data is not None and self._check_cup_handle_volume(volume_data, cup_start, cup_end, handle_start, handle_end):
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    # Helper methods for volume and pattern analysis
    def _check_volume_confirmation(self, volume_data: pd.DataFrame, indices: List[int]) -> bool:
        """Check if volume confirms the pattern"""
        if volume_data is None or len(indices) < 2:
            return False
        
        try:
            volumes = []
            for idx in indices:
                if idx < len(volume_data):
                    vol = volume_data.iloc[idx]['volume'] if 'volume' in volume_data.columns else volume_data.iloc[idx, 0]
                    volumes.append(vol)
            
            # Volume should generally decrease through pattern formation
            return len(volumes) > 1 and volumes[-1] < volumes[0]
        except:
            return False
    
    def _check_volume_trend(self, volume_data: pd.DataFrame, indices: List[int]) -> bool:
        """Check if volume trend supports pattern"""
        if volume_data is None or len(indices) < 2:
            return False
        
        try:
            volumes = []
            for idx in indices:
                if idx < len(volume_data):
                    vol = volume_data.iloc[idx]['volume'] if 'volume' in volume_data.columns else volume_data.iloc[idx, 0]
                    volumes.append(vol)
            
            # Check for increasing volume trend
            if len(volumes) > 1:
                slope = np.polyfit(range(len(volumes)), volumes, 1)[0]
                return slope > 0
        except:
            pass
        
        return False
    
    def _check_volume_pattern(self, volume_data: pd.DataFrame, start: int, end: int) -> bool:
        """Check volume pattern in consolidation"""
        if volume_data is None:
            return False
        
        try:
            volume_segment = volume_data.iloc[start:end]
            volumes = volume_segment['volume'] if 'volume' in volume_segment.columns else volume_segment.iloc[:, 0]
            
            # Volume should decrease during consolidation
            return volumes.iloc[-1] < volumes.iloc[0]
        except:
            return False
    
    def _check_cup_handle_volume(self, volume_data: pd.DataFrame, cup_start: int, 
                               cup_end: int, handle_start: int, handle_end: int) -> bool:
        """Check volume pattern for cup and handle"""
        if volume_data is None:
            return False
        
        try:
            # Volume should be high at cup start, low in cup, lower in handle
            cup_vol = volume_data.iloc[cup_start:cup_end]
            handle_vol = volume_data.iloc[handle_start:handle_end]
            
            cup_avg = cup_vol.mean().iloc[0] if 'volume' in cup_vol.columns else cup_vol.mean(axis=1).mean()
            handle_avg = handle_vol.mean().iloc[0] if 'volume' in handle_vol.columns else handle_vol.mean(axis=1).mean()
            
            return handle_avg < cup_avg
        except:
            return False
    
    def _calculate_cup_smoothness(self, cup_segment: pd.Series, bottom_idx: int) -> float:
        """Calculate how smooth/rounded the cup is"""
        try:
            # Simple smoothness metric based on gradient changes
            gradients = np.gradient(cup_segment.values)
            gradient_changes = np.abs(np.gradient(gradients))
            smoothness = 1 / (1 + np.mean(gradient_changes))
            return min(smoothness, 1.0)
        except:
            return 0.5
    
    def learn_from_pattern_outcome(self, pattern: VisualPattern, actual_outcome: Dict[str, Any]):
        """Learn from pattern recognition outcomes to improve accuracy"""
        if self.knowledge_engine:
            try:
                # Create learning data
                learning_data = {
                    'pattern_type': pattern.pattern_type,
                    'confidence': pattern.confidence,
                    'coordinates': pattern.coordinates,
                    'price_levels': pattern.price_levels,
                    'volume_confirmation': pattern.volume_confirmation,
                    'metadata': pattern.metadata
                }
                
                # Process through knowledge engine
                self.knowledge_engine.process_market_event({
                    'event_type': 'visual_pattern_outcome',
                    'pattern_data': learning_data,
                    'outcome': actual_outcome,
                    'timestamp': datetime.now().isoformat()
                })
                
                self.logger.info(f"Pattern learning updated for {pattern.pattern_type}")
                
            except Exception as e:
                self.logger.error(f"Error learning from pattern outcome: {e}")
    
    def get_scanner_performance(self) -> Dict[str, Any]:
        """Get scanner performance metrics"""
        total_scans = len(self.scan_history)
        total_patterns = sum(scan['patterns_found'] for scan in self.scan_history.values())
        high_confidence = sum(scan['high_confidence_patterns'] for scan in self.scan_history.values())
        
        return {
            'total_scans': total_scans,
            'total_patterns_detected': total_patterns,
            'high_confidence_patterns': high_confidence,
            'average_patterns_per_scan': total_patterns / max(total_scans, 1),
            'high_confidence_rate': high_confidence / max(total_patterns, 1),
            'scanner_efficiency': min(total_patterns / max(total_scans * 5, 1), 1.0)  # Max 5 patterns per scan
        }

# Integration helper function
def configure_visual_scanning_for_agent(agent, knowledge_engine):
    """Configure visual pattern scanning for a trading agent"""
    
    # Create visual scanner
    scanner = VisualPatternScanner(knowledge_engine)
    
    # Configure scanner
    config = ChartConfiguration(
        timeframe="1D",
        lookback_periods=50,
        min_pattern_length=10,
        confidence_threshold=0.7,
        enable_volume_analysis=True,
        pattern_types=['head_and_shoulders', 'double_top', 'ascending_triangle', 'flag_pattern']
    )
    scanner.configure_scanner(config)
    
    # Add scanner to agent
    agent.visual_scanner = scanner
    
    # Enhance agent's pattern recognition method
    original_analyze_brain_patterns = agent._analyze_market_brain_patterns
    
    def enhanced_analyze_with_visual(market_data):
        """Enhanced analysis with visual pattern scanning"""
        # Get traditional brain patterns
        traditional_signals = original_analyze_brain_patterns(market_data)
        
        # Add visual pattern scanning
        try:
            # Create price data DataFrame (simplified for demo)
            import pandas as pd
            
            # Use historical data from technical indicators
            if hasattr(agent, 'technical_indicators') and market_data.symbol in agent.technical_indicators.price_history:
                price_history = agent.technical_indicators.price_history[market_data.symbol]
                
                if len(price_history) >= 20:
                    # Create DataFrame
                    prices_df = pd.DataFrame([
                        {'close': p} for p in price_history[-50:]  # Last 50 periods
                    ])
                    
                    # Scan for visual patterns
                    visual_patterns = scanner.scan_chart_for_patterns(market_data.symbol, prices_df)
                    
                    # Convert visual patterns to trading signals
                    for pattern in visual_patterns:
                        if pattern.confidence > 0.8:
                            action = 'BUY' if pattern.pattern_type in ['cup_and_handle', 'ascending_triangle'] else 'SELL'
                            
                            # Create enhanced signal
                            from main import TradingSignal
                            visual_signal = TradingSignal(
                                symbol=market_data.symbol,
                                action=action,
                                confidence=pattern.confidence,
                                reason=f"Visual pattern: {pattern.pattern_type} (confidence: {pattern.confidence:.2f})",
                                timestamp=datetime.now(),
                                signal_strength="STRONG" if pattern.confidence > 0.9 else "MEDIUM"
                            )
                            
                            traditional_signals.append(visual_signal)
                            
                            # Learn from pattern for future improvement
                            agent.visual_patterns_detected = getattr(agent, 'visual_patterns_detected', 0) + 1
        
        except Exception as e:
            agent.logger.error(f"Error in visual pattern analysis: {e}")
        
        return traditional_signals
    
    # Replace the method
    agent._analyze_market_brain_patterns = enhanced_analyze_with_visual
    
    agent.logger.info("Visual pattern scanning configured successfully")
    return scanner
