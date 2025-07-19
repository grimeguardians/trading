"""
Fibonacci retracement and extension calculations for trading
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

class FibonacciCalculator:
    """Calculator for Fibonacci retracement and extension levels"""
    
    def __init__(self):
        # Standard Fibonacci levels
        self.retracement_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        self.extension_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.414, 1.618, 2.0, 2.618, 3.14, 4.236]
        
    def calculate_retracement_levels(self, price_data: List[Dict]) -> Dict:
        """
        Calculate Fibonacci retracement levels from price data
        
        Args:
            price_data: List of price dictionaries with 'high', 'low', 'close' keys
            
        Returns:
            Dictionary with Fibonacci levels and key information
        """
        try:
            if not price_data or len(price_data) < 2:
                return {"error": "Insufficient price data"}
            
            # Find swing high and swing low
            swing_high, swing_low = self._find_swing_points(price_data)
            
            if swing_high is None or swing_low is None:
                return {"error": "Could not identify swing points"}
            
            # Calculate Fibonacci levels
            levels = self._calculate_fibonacci_levels(
                swing_high["price"], 
                swing_low["price"], 
                self.retracement_levels
            )
            
            # Determine trend direction
            trend = "uptrend" if swing_low["index"] < swing_high["index"] else "downtrend"
            
            return {
                "swing_high": swing_high,
                "swing_low": swing_low,
                "trend": trend,
                "levels": levels,
                "key_levels": self._identify_key_levels(levels),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Error calculating retracement levels: {str(e)}"}
    
    def calculate_extension_levels(self, price_data: List[Dict], wave_start: int, wave_end: int) -> Dict:
        """
        Calculate Fibonacci extension levels
        
        Args:
            price_data: List of price dictionaries
            wave_start: Index of wave start
            wave_end: Index of wave end
            
        Returns:
            Dictionary with extension levels
        """
        try:
            if not price_data or wave_start >= len(price_data) or wave_end >= len(price_data):
                return {"error": "Invalid wave parameters"}
            
            start_price = price_data[wave_start]["close"]
            end_price = price_data[wave_end]["close"]
            
            # Calculate extension levels
            levels = self._calculate_fibonacci_levels(
                start_price, 
                end_price, 
                self.extension_levels
            )
            
            return {
                "wave_start": {"index": wave_start, "price": start_price},
                "wave_end": {"index": wave_end, "price": end_price},
                "levels": levels,
                "projection_direction": "up" if end_price > start_price else "down",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Error calculating extension levels: {str(e)}"}
    
    def _find_swing_points(self, price_data: List[Dict]) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Find swing high and swing low points"""
        try:
            if len(price_data) < 10:
                return None, None
            
            # Get recent data (last 50 bars or all if less)
            recent_data = price_data[-50:]
            
            # Find swing high
            swing_high = None
            max_high = 0
            
            for i, bar in enumerate(recent_data):
                if bar["high"] > max_high:
                    max_high = bar["high"]
                    swing_high = {
                        "index": i,
                        "price": bar["high"],
                        "timestamp": bar.get("timestamp", datetime.now())
                    }
            
            # Find swing low
            swing_low = None
            min_low = float('inf')
            
            for i, bar in enumerate(recent_data):
                if bar["low"] < min_low:
                    min_low = bar["low"]
                    swing_low = {
                        "index": i,
                        "price": bar["low"],
                        "timestamp": bar.get("timestamp", datetime.now())
                    }
            
            return swing_high, swing_low
            
        except Exception as e:
            return None, None
    
    def _calculate_fibonacci_levels(self, high_price: float, low_price: float, levels: List[float]) -> Dict:
        """Calculate Fibonacci levels between two prices"""
        try:
            price_range = high_price - low_price
            
            fibonacci_levels = {}
            
            for level in levels:
                if high_price > low_price:
                    # Uptrend - retracement from high
                    price = high_price - (price_range * level)
                else:
                    # Downtrend - retracement from low
                    price = low_price + (price_range * level)
                
                fibonacci_levels[f"{level:.3f}"] = {
                    "level": level,
                    "price": round(price, 4),
                    "percentage": level * 100
                }
            
            return fibonacci_levels
            
        except Exception as e:
            return {}
    
    def _identify_key_levels(self, levels: Dict) -> List[str]:
        """Identify key Fibonacci levels"""
        key_levels = ["0.382", "0.500", "0.618"]
        return [level for level in key_levels if level in levels]
    
    def calculate_fibonacci_time_zones(self, price_data: List[Dict], start_index: int = 0) -> Dict:
        """
        Calculate Fibonacci time zones
        
        Args:
            price_data: List of price dictionaries
            start_index: Starting index for time zone calculation
            
        Returns:
            Dictionary with time zone information
        """
        try:
            if not price_data or start_index >= len(price_data):
                return {"error": "Invalid parameters"}
            
            # Fibonacci sequence for time zones
            fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]
            
            time_zones = {}
            
            for i, fib_num in enumerate(fib_sequence):
                zone_index = start_index + fib_num
                
                if zone_index < len(price_data):
                    time_zones[f"zone_{i+1}"] = {
                        "fibonacci_number": fib_num,
                        "index": zone_index,
                        "timestamp": price_data[zone_index].get("timestamp", datetime.now()),
                        "price": price_data[zone_index]["close"]
                    }
            
            return {
                "start_index": start_index,
                "time_zones": time_zones,
                "total_zones": len(time_zones)
            }
            
        except Exception as e:
            return {"error": f"Error calculating time zones: {str(e)}"}
    
    def calculate_fibonacci_arcs(self, price_data: List[Dict], swing_high: Dict, swing_low: Dict) -> Dict:
        """
        Calculate Fibonacci arcs
        
        Args:
            price_data: List of price dictionaries
            swing_high: Swing high point
            swing_low: Swing low point
            
        Returns:
            Dictionary with arc information
        """
        try:
            if not swing_high or not swing_low:
                return {"error": "Invalid swing points"}
            
            # Calculate base distance
            price_distance = abs(swing_high["price"] - swing_low["price"])
            time_distance = abs(swing_high["index"] - swing_low["index"])
            
            # Calculate arc levels
            arc_levels = [0.382, 0.5, 0.618]
            arcs = {}
            
            for level in arc_levels:
                arc_radius = price_distance * level
                
                arcs[f"arc_{level}"] = {
                    "level": level,
                    "radius": arc_radius,
                    "center_price": (swing_high["price"] + swing_low["price"]) / 2,
                    "center_index": (swing_high["index"] + swing_low["index"]) / 2
                }
            
            return {
                "swing_high": swing_high,
                "swing_low": swing_low,
                "arcs": arcs,
                "base_distance": price_distance
            }
            
        except Exception as e:
            return {"error": f"Error calculating arcs: {str(e)}"}
    
    def calculate_fibonacci_fans(self, price_data: List[Dict], swing_high: Dict, swing_low: Dict) -> Dict:
        """
        Calculate Fibonacci fans
        
        Args:
            price_data: List of price dictionaries
            swing_high: Swing high point
            swing_low: Swing low point
            
        Returns:
            Dictionary with fan line information
        """
        try:
            if not swing_high or not swing_low:
                return {"error": "Invalid swing points"}
            
            # Calculate fan levels
            fan_levels = [0.382, 0.5, 0.618]
            fans = {}
            
            price_range = swing_high["price"] - swing_low["price"]
            time_range = swing_high["index"] - swing_low["index"]
            
            for level in fan_levels:
                # Calculate slope for fan line
                slope = (price_range * level) / time_range if time_range != 0 else 0
                
                fans[f"fan_{level}"] = {
                    "level": level,
                    "slope": slope,
                    "start_price": swing_low["price"],
                    "start_index": swing_low["index"]
                }
            
            return {
                "swing_high": swing_high,
                "swing_low": swing_low,
                "fans": fans,
                "trend": "up" if swing_high["price"] > swing_low["price"] else "down"
            }
            
        except Exception as e:
            return {"error": f"Error calculating fans: {str(e)}"}
    
    def find_fibonacci_confluence(self, retracement_levels: Dict, extension_levels: Dict, tolerance: float = 0.01) -> List[Dict]:
        """
        Find confluence zones where multiple Fibonacci levels align
        
        Args:
            retracement_levels: Retracement levels dictionary
            extension_levels: Extension levels dictionary
            tolerance: Price tolerance for confluence (as percentage)
            
        Returns:
            List of confluence zones
        """
        try:
            confluences = []
            
            if "levels" not in retracement_levels or "levels" not in extension_levels:
                return confluences
            
            retracement = retracement_levels["levels"]
            extension = extension_levels["levels"]
            
            # Compare each retracement level with extension levels
            for ret_key, ret_data in retracement.items():
                ret_price = ret_data["price"]
                
                for ext_key, ext_data in extension.items():
                    ext_price = ext_data["price"]
                    
                    # Calculate percentage difference
                    price_diff = abs(ret_price - ext_price) / ((ret_price + ext_price) / 2)
                    
                    if price_diff <= tolerance:
                        confluences.append({
                            "confluence_price": (ret_price + ext_price) / 2,
                            "retracement_level": ret_key,
                            "extension_level": ext_key,
                            "price_difference": price_diff,
                            "strength": "high" if price_diff <= tolerance/2 else "medium"
                        })
            
            # Sort by strength and price difference
            confluences.sort(key=lambda x: x["price_difference"])
            
            return confluences
            
        except Exception as e:
            return []
    
    def calculate_fibonacci_clusters(self, multiple_swings: List[Dict], tolerance: float = 0.005) -> Dict:
        """
        Calculate Fibonacci clusters from multiple swing points
        
        Args:
            multiple_swings: List of swing point dictionaries
            tolerance: Clustering tolerance
            
        Returns:
            Dictionary with cluster information
        """
        try:
            all_levels = []
            
            # Calculate Fibonacci levels for each swing
            for swing in multiple_swings:
                if "high" in swing and "low" in swing:
                    levels = self._calculate_fibonacci_levels(
                        swing["high"], 
                        swing["low"], 
                        self.retracement_levels
                    )
                    
                    for level_data in levels.values():
                        all_levels.append(level_data["price"])
            
            # Find clusters
            clusters = self._find_price_clusters(all_levels, tolerance)
            
            return {
                "clusters": clusters,
                "total_levels": len(all_levels),
                "cluster_count": len(clusters)
            }
            
        except Exception as e:
            return {"error": f"Error calculating clusters: {str(e)}"}
    
    def _find_price_clusters(self, prices: List[float], tolerance: float) -> List[Dict]:
        """Find clusters of similar prices"""
        try:
            if not prices:
                return []
            
            sorted_prices = sorted(prices)
            clusters = []
            current_cluster = [sorted_prices[0]]
            
            for i in range(1, len(sorted_prices)):
                price_diff = abs(sorted_prices[i] - sorted_prices[i-1]) / sorted_prices[i-1]
                
                if price_diff <= tolerance:
                    current_cluster.append(sorted_prices[i])
                else:
                    if len(current_cluster) > 1:
                        clusters.append({
                            "center_price": sum(current_cluster) / len(current_cluster),
                            "price_range": [min(current_cluster), max(current_cluster)],
                            "strength": len(current_cluster),
                            "prices": current_cluster
                        })
                    current_cluster = [sorted_prices[i]]
            
            # Add final cluster
            if len(current_cluster) > 1:
                clusters.append({
                    "center_price": sum(current_cluster) / len(current_cluster),
                    "price_range": [min(current_cluster), max(current_cluster)],
                    "strength": len(current_cluster),
                    "prices": current_cluster
                })
            
            return clusters
            
        except Exception as e:
            return []
    
    def get_fibonacci_support_resistance(self, price_data: List[Dict], current_price: float) -> Dict:
        """
        Get relevant Fibonacci support and resistance levels
        
        Args:
            price_data: List of price dictionaries
            current_price: Current market price
            
        Returns:
            Dictionary with support and resistance levels
        """
        try:
            # Calculate retracement levels
            retracement_data = self.calculate_retracement_levels(price_data)
            
            if "error" in retracement_data:
                return retracement_data
            
            levels = retracement_data["levels"]
            
            # Separate support and resistance
            support_levels = []
            resistance_levels = []
            
            for level_key, level_data in levels.items():
                price = level_data["price"]
                
                if price < current_price:
                    support_levels.append({
                        "level": level_key,
                        "price": price,
                        "distance": current_price - price,
                        "percentage": level_data["percentage"]
                    })
                else:
                    resistance_levels.append({
                        "level": level_key,
                        "price": price,
                        "distance": price - current_price,
                        "percentage": level_data["percentage"]
                    })
            
            # Sort by distance
            support_levels.sort(key=lambda x: x["distance"])
            resistance_levels.sort(key=lambda x: x["distance"])
            
            return {
                "current_price": current_price,
                "nearest_support": support_levels[0] if support_levels else None,
                "nearest_resistance": resistance_levels[0] if resistance_levels else None,
                "all_support": support_levels,
                "all_resistance": resistance_levels,
                "trend": retracement_data["trend"]
            }
            
        except Exception as e:
            return {"error": f"Error getting support/resistance: {str(e)}"}
