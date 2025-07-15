
import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from knowledge_engine import DigitalBrain

@dataclass
class HistoricalPattern:
    """Represents a historical pattern occurrence"""
    pattern_id: str
    symbol: str
    pattern_type: str
    start_date: datetime
    end_date: datetime
    start_price: float
    end_price: float
    target_price: Optional[float]
    success: bool
    confidence_score: float
    volume_profile: str  # low, normal, high
    market_condition: str  # bull, bear, sideways
    sector: str
    pattern_attributes: Dict[str, Any]
    
@dataclass 
class PatternStatistics:
    """Statistical analysis of pattern performance"""
    pattern_type: str
    total_occurrences: int
    success_rate: float
    average_duration_days: float
    average_price_move: float
    best_performing_timeframe: str
    worst_performing_timeframe: str
    sector_performance: Dict[str, float]
    market_condition_performance: Dict[str, float]

class HistoricalPatternDatabase:
    """Comprehensive database for historical pattern analysis"""
    
    def __init__(self, db_path: str = "historical_patterns.db"):
        self.db_path = db_path
        self.logger = logging.getLogger("HistoricalPatternDB")
        self.conn = None
        self._initialize_database()
        self._populate_sample_data()
        
    def _initialize_database(self):
        """Initialize the SQLite database schema"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        
        cursor = self.conn.cursor()
        
        # Create patterns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patterns (
                pattern_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                start_price REAL NOT NULL,
                end_price REAL NOT NULL,
                target_price REAL,
                success INTEGER NOT NULL,
                confidence_score REAL NOT NULL,
                volume_profile TEXT NOT NULL,
                market_condition TEXT NOT NULL,
                sector TEXT NOT NULL,
                pattern_attributes TEXT,
                created_at TEXT NOT NULL
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol ON patterns(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pattern_type ON patterns(pattern_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_start_date ON patterns(start_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_success ON patterns(success)')
        
        # Create pattern statistics cache table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pattern_statistics (
                pattern_type TEXT PRIMARY KEY,
                total_occurrences INTEGER,
                success_rate REAL,
                average_duration_days REAL,
                average_price_move REAL,
                statistics_data TEXT,
                last_updated TEXT
            )
        ''')
        
        self.conn.commit()
        self.logger.info("Historical pattern database initialized")
    
    def _populate_sample_data(self):
        """Populate database with sample historical patterns"""
        # Check if data already exists
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM patterns")
        if cursor.fetchone()[0] > 0:
            return
        
        self.logger.info("Populating sample historical pattern data...")
        
        # Sample patterns for different symbols and types
        sample_patterns = self._generate_sample_patterns()
        
        for pattern in sample_patterns:
            self.add_pattern(pattern)
        
        self.logger.info(f"Added {len(sample_patterns)} sample patterns to database")
    
    def _generate_sample_patterns(self) -> List[HistoricalPattern]:
        """Generate realistic sample pattern data"""
        import random
        
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "SPY", "QQQ", "AMZN", "META", "JPM"]
        pattern_types = [
            "head_and_shoulders", "double_top", "double_bottom", "triangle",
            "flag", "pennant", "cup_and_handle", "breakout", "support_resistance"
        ]
        sectors = ["Technology", "Financial", "Healthcare", "Energy", "Automotive"]
        
        patterns = []
        base_date = datetime.now() - timedelta(days=365 * 3)  # 3 years of data
        
        for i in range(500):  # Generate 500 sample patterns
            symbol = random.choice(symbols)
            pattern_type = random.choice(pattern_types)
            sector = "Technology" if symbol in ["AAPL", "GOOGL", "MSFT", "NVDA", "META"] else random.choice(sectors)
            
            # Generate realistic dates
            start_date = base_date + timedelta(days=random.randint(0, 1000))
            duration = random.randint(5, 45)  # 5-45 day patterns
            end_date = start_date + timedelta(days=duration)
            
            # Generate realistic prices
            start_price = random.uniform(50, 500)
            
            # Pattern success probability varies by type
            success_prob = {
                "head_and_shoulders": 0.75,
                "double_top": 0.70,
                "double_bottom": 0.72,
                "triangle": 0.65,
                "flag": 0.68,
                "pennant": 0.67,
                "cup_and_handle": 0.78,
                "breakout": 0.62,
                "support_resistance": 0.80
            }.get(pattern_type, 0.65)
            
            success = random.random() < success_prob
            
            if success:
                if pattern_type in ["double_bottom", "cup_and_handle"]:
                    price_move = random.uniform(0.05, 0.25)  # 5-25% gain
                    end_price = start_price * (1 + price_move)
                elif pattern_type in ["head_and_shoulders", "double_top"]:
                    price_move = random.uniform(-0.20, -0.05)  # 5-20% loss
                    end_price = start_price * (1 + price_move)
                else:
                    price_move = random.uniform(-0.15, 0.20)
                    end_price = start_price * (1 + price_move)
            else:
                # Failed patterns move opposite or sideways
                price_move = random.uniform(-0.10, 0.10)
                end_price = start_price * (1 + price_move)
            
            target_price = end_price * random.uniform(1.05, 1.20) if success else None
            confidence_score = random.uniform(0.6, 0.95)
            volume_profile = random.choice(["low", "normal", "high"])
            market_condition = random.choice(["bull", "bear", "sideways"])
            
            pattern = HistoricalPattern(
                pattern_id=f"pattern_{i+1:04d}",
                symbol=symbol,
                pattern_type=pattern_type,
                start_date=start_date,
                end_date=end_date,
                start_price=start_price,
                end_price=end_price,
                target_price=target_price,
                success=success,
                confidence_score=confidence_score,
                volume_profile=volume_profile,
                market_condition=market_condition,
                sector=sector,
                pattern_attributes={
                    "duration_days": duration,
                    "price_change_percent": ((end_price - start_price) / start_price) * 100,
                    "volume_spike": volume_profile == "high",
                    "breakout_confirmed": success and pattern_type == "breakout"
                }
            )
            
            patterns.append(pattern)
        
        return patterns
    
    def add_pattern(self, pattern: HistoricalPattern) -> bool:
        """Add a new pattern to the database"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO patterns (
                    pattern_id, symbol, pattern_type, start_date, end_date,
                    start_price, end_price, target_price, success, confidence_score,
                    volume_profile, market_condition, sector, pattern_attributes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                pattern.pattern_id,
                pattern.symbol,
                pattern.pattern_type,
                pattern.start_date.isoformat(),
                pattern.end_date.isoformat(),
                pattern.start_price,
                pattern.end_price,
                pattern.target_price,
                int(pattern.success),
                pattern.confidence_score,
                pattern.volume_profile,
                pattern.market_condition,
                pattern.sector,
                json.dumps(pattern.pattern_attributes),
                datetime.now().isoformat()
            ))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding pattern: {e}")
            return False
    
    def get_pattern_statistics(self, pattern_type: str) -> PatternStatistics:
        """Get comprehensive statistics for a pattern type"""
        cursor = self.conn.cursor()
        
        # Get basic statistics
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) as success_rate,
                AVG(julianday(end_date) - julianday(start_date)) as avg_duration,
                AVG((end_price - start_price) / start_price * 100) as avg_price_move
            FROM patterns 
            WHERE pattern_type = ?
        ''', (pattern_type,))
        
        basic_stats = cursor.fetchone()
        
        # Get sector performance
        cursor.execute('''
            SELECT 
                sector,
                AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) as success_rate
            FROM patterns 
            WHERE pattern_type = ?
            GROUP BY sector
        ''', (pattern_type,))
        
        sector_performance = {row['sector']: row['success_rate'] for row in cursor.fetchall()}
        
        # Get market condition performance
        cursor.execute('''
            SELECT 
                market_condition,
                AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) as success_rate
            FROM patterns 
            WHERE pattern_type = ?
            GROUP BY market_condition
        ''', (pattern_type,))
        
        market_performance = {row['market_condition']: row['success_rate'] for row in cursor.fetchall()}
        
        return PatternStatistics(
            pattern_type=pattern_type,
            total_occurrences=basic_stats['total'],
            success_rate=basic_stats['success_rate'],
            average_duration_days=basic_stats['avg_duration'],
            average_price_move=basic_stats['avg_price_move'],
            best_performing_timeframe="N/A",  # Would need time-based analysis
            worst_performing_timeframe="N/A",
            sector_performance=sector_performance,
            market_condition_performance=market_performance
        )
    
    def find_similar_patterns(self, symbol: str, pattern_type: str, 
                            current_price: float, lookback_days: int = 365) -> List[HistoricalPattern]:
        """Find similar historical patterns for comparison"""
        cursor = self.conn.cursor()
        
        # Price range for similarity (±20%)
        price_min = current_price * 0.8
        price_max = current_price * 1.2
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        
        cursor.execute('''
            SELECT * FROM patterns 
            WHERE pattern_type = ? 
            AND (symbol = ? OR sector = (SELECT sector FROM patterns WHERE symbol = ? LIMIT 1))
            AND start_price BETWEEN ? AND ?
            AND start_date >= ?
            ORDER BY confidence_score DESC, 
                     ABS(start_price - ?) ASC
            LIMIT 10
        ''', (pattern_type, symbol, symbol, price_min, price_max, cutoff_date, current_price))
        
        return [self._row_to_pattern(row) for row in cursor.fetchall()]
    
    def get_pattern_success_by_conditions(self, pattern_type: str, 
                                        market_condition: str = None,
                                        volume_profile: str = None,
                                        sector: str = None) -> float:
        """Get pattern success rate under specific market conditions"""
        cursor = self.conn.cursor()
        
        query = "SELECT AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) FROM patterns WHERE pattern_type = ?"
        params = [pattern_type]
        
        if market_condition:
            query += " AND market_condition = ?"
            params.append(market_condition)
        
        if volume_profile:
            query += " AND volume_profile = ?"
            params.append(volume_profile)
        
        if sector:
            query += " AND sector = ?"
            params.append(sector)
        
        cursor.execute(query, params)
        result = cursor.fetchone()[0]
        return result if result is not None else 0.0
    
    def _row_to_pattern(self, row) -> HistoricalPattern:
        """Convert database row to HistoricalPattern object"""
        return HistoricalPattern(
            pattern_id=row['pattern_id'],
            symbol=row['symbol'],
            pattern_type=row['pattern_type'],
            start_date=datetime.fromisoformat(row['start_date']),
            end_date=datetime.fromisoformat(row['end_date']),
            start_price=row['start_price'],
            end_price=row['end_price'],
            target_price=row['target_price'],
            success=bool(row['success']),
            confidence_score=row['confidence_score'],
            volume_profile=row['volume_profile'],
            market_condition=row['market_condition'],
            sector=row['sector'],
            pattern_attributes=json.loads(row['pattern_attributes']) if row['pattern_attributes'] else {}
        )
    
    def generate_pattern_report(self, pattern_type: str) -> Dict[str, Any]:
        """Generate comprehensive pattern analysis report"""
        stats = self.get_pattern_statistics(pattern_type)
        
        report = {
            'pattern_type': pattern_type,
            'overview': {
                'total_occurrences': stats.total_occurrences,
                'overall_success_rate': f"{stats.success_rate:.1%}",
                'average_duration': f"{stats.average_duration_days:.1f} days",
                'average_price_move': f"{stats.average_price_move:+.2f}%"
            },
            'sector_analysis': {
                sector: f"{success_rate:.1%}" 
                for sector, success_rate in stats.sector_performance.items()
            },
            'market_condition_analysis': {
                condition: f"{success_rate:.1%}"
                for condition, success_rate in stats.market_condition_performance.items()
            },
            'trading_recommendations': self._generate_trading_recommendations(stats),
            'risk_assessment': self._assess_pattern_risk(stats)
        }
        
        return report
    
    def _generate_trading_recommendations(self, stats: PatternStatistics) -> List[str]:
        """Generate trading recommendations based on statistics"""
        recommendations = []
        
        if stats.success_rate > 0.75:
            recommendations.append("High probability pattern - suitable for aggressive position sizing")
        elif stats.success_rate > 0.65:
            recommendations.append("Moderate probability pattern - use standard position sizing")
        else:
            recommendations.append("Lower probability pattern - reduce position size or use as confirmation only")
        
        # Best performing sectors
        if stats.sector_performance:
            best_sector = max(stats.sector_performance.items(), key=lambda x: x[1])
            if best_sector[1] > 0.75:
                recommendations.append(f"Pattern shows highest success in {best_sector[0]} sector ({best_sector[1]:.1%})")
        
        # Market condition recommendations
        if stats.market_condition_performance:
            best_condition = max(stats.market_condition_performance.items(), key=lambda x: x[1])
            recommendations.append(f"Most reliable in {best_condition[0]} market conditions ({best_condition[1]:.1%})")
        
        return recommendations
    
    def _assess_pattern_risk(self, stats: PatternStatistics) -> Dict[str, Any]:
        """Assess risk factors for the pattern"""
        risk_assessment = {
            'risk_level': 'medium',
            'key_risks': [],
            'risk_mitigation': []
        }
        
        if stats.success_rate < 0.6:
            risk_assessment['risk_level'] = 'high'
            risk_assessment['key_risks'].append('Below-average success rate')
            risk_assessment['risk_mitigation'].append('Use tight stop losses and reduced position sizing')
        elif stats.success_rate > 0.8:
            risk_assessment['risk_level'] = 'low'
        
        if stats.average_duration_days > 30:
            risk_assessment['key_risks'].append('Extended holding period increases market risk')
            risk_assessment['risk_mitigation'].append('Monitor for early exit signals')
        
        return risk_assessment
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

def main():
    """Test the historical pattern database"""
    print("🏛️ Historical Pattern Database Testing")
    print("=" * 50)
    
    # Initialize database
    db = HistoricalPatternDatabase()
    
    # Test pattern statistics
    pattern_types = ["head_and_shoulders", "double_bottom", "breakout", "triangle"]
    
    print("\n📊 Pattern Statistics Summary:")
    for pattern_type in pattern_types:
        stats = db.get_pattern_statistics(pattern_type)
        print(f"\n{pattern_type.replace('_', ' ').title()}:")
        print(f"  Total Occurrences: {stats.total_occurrences}")
        print(f"  Success Rate: {stats.success_rate:.1%}")
        print(f"  Average Duration: {stats.average_duration_days:.1f} days")
        print(f"  Average Price Move: {stats.average_price_move:+.2f}%")
    
    # Test similar pattern finding
    print(f"\n🔍 Finding Similar Patterns for AAPL breakout at $150:")
    similar_patterns = db.find_similar_patterns("AAPL", "breakout", 150.0)
    
    for i, pattern in enumerate(similar_patterns[:3]):
        success_text = "✅ Success" if pattern.success else "❌ Failed"
        price_change = ((pattern.end_price - pattern.start_price) / pattern.start_price) * 100
        print(f"  {i+1}. {pattern.symbol} - {success_text} - {price_change:+.1f}% move")
    
    # Test conditional success rates
    print(f"\n📈 Pattern Success by Market Conditions:")
    test_conditions = [
        ("head_and_shoulders", "bull", None, None),
        ("double_bottom", "bear", None, None),
        ("breakout", None, "high", None),
        ("triangle", None, None, "Technology")
    ]
    
    for pattern_type, market, volume, sector in test_conditions:
        success_rate = db.get_pattern_success_by_conditions(pattern_type, market, volume, sector)
        condition_desc = f"{market or ''} {volume or ''} {sector or ''}".strip()
        print(f"  {pattern_type} in {condition_desc}: {success_rate:.1%}")
    
    # Generate sample report
    print(f"\n📋 Detailed Report for Head and Shoulders:")
    report = db.generate_pattern_report("head_and_shoulders")
    
    print(f"  Success Rate: {report['overview']['overall_success_rate']}")
    print(f"  Average Duration: {report['overview']['average_duration']}")
    
    if report['trading_recommendations']:
        print(f"  Key Recommendation: {report['trading_recommendations'][0]}")
    
    risk_level = report['risk_assessment']['risk_level']
    print(f"  Risk Level: {risk_level.upper()}")
    
    # Cleanup
    db.close()
    print(f"\n✅ Historical Pattern Database Testing Complete!")

if __name__ == "__main__":
    main()
