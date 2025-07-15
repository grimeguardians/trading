
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np
from dataclasses import dataclass
from knowledge_engine import KnowledgeNode, KnowledgeEdge, DigitalBrain

# Simulated Neo4j driver for Replit environment
class MockNeo4jDriver:
    """Mock Neo4j driver for development in Replit environment"""
    
    def __init__(self, uri: str, auth: Tuple[str, str]):
        self.uri = uri
        self.auth = auth
        self.logger = logging.getLogger("MockNeo4j")
        self.nodes = {}
        self.relationships = {}
        self.logger.info(f"Connected to mock Neo4j at {uri}")
    
    def session(self):
        return MockNeo4jSession(self)
    
    def close(self):
        self.logger.info("Neo4j connection closed")

class MockNeo4jSession:
    """Mock Neo4j session for development"""
    
    def __init__(self, driver):
        self.driver = driver
    
    def run(self, query: str, parameters: Dict = None):
        # Simulate query execution
        if parameters is None:
            parameters = {}
        
        # Parse basic operations
        if "CREATE" in query.upper():
            return self._handle_create(query, parameters)
        elif "MATCH" in query.upper():
            return self._handle_match(query, parameters)
        elif "MERGE" in query.upper():
            return self._handle_merge(query, parameters)
        
        return MockNeo4jResult([])
    
    def _handle_create(self, query: str, params: Dict):
        # Simple CREATE simulation
        node_id = f"node_{len(self.driver.nodes) + 1}"
        self.driver.nodes[node_id] = params
        return MockNeo4jResult([{"node_id": node_id}])
    
    def _handle_match(self, query: str, params: Dict):
        # Simple MATCH simulation
        results = []
        for node_id, node_data in self.driver.nodes.items():
            if self._matches_criteria(node_data, params):
                results.append({"node": node_data, "id": node_id})
        return MockNeo4jResult(results)
    
    def _handle_merge(self, query: str, params: Dict):
        # Simple MERGE simulation
        return self._handle_create(query, params)
    
    def _matches_criteria(self, node_data: Dict, criteria: Dict) -> bool:
        for key, value in criteria.items():
            if key in node_data and node_data[key] != value:
                return False
        return True
    
    def close(self):
        pass

class MockNeo4jResult:
    """Mock Neo4j result object"""
    
    def __init__(self, records: List[Dict]):
        self.records = records
    
    def __iter__(self):
        return iter(self.records)
    
    def single(self):
        return self.records[0] if self.records else None
    
    def data(self):
        return self.records

@dataclass
class MarketAsset:
    """Represents a market asset in the knowledge graph"""
    symbol: str
    name: str
    sector: str
    market_cap: Optional[float] = None
    exchange: str = "NASDAQ"
    asset_type: str = "stock"  # stock, etf, crypto, commodity
    
@dataclass
class SectorInfo:
    """Represents a market sector"""
    name: str
    description: str
    typical_pe_ratio: Optional[float] = None
    volatility_profile: str = "medium"  # low, medium, high
    economic_sensitivity: str = "medium"  # low, medium, high

class Neo4jKnowledgeGraph:
    """Enhanced knowledge graph with Neo4j backend"""
    
    def __init__(self, neo4j_uri: str = "bolt://localhost:7687", 
                 neo4j_user: str = "neo4j", neo4j_password: str = "password"):
        self.logger = logging.getLogger("Neo4jKnowledgeGraph")
        
        # Use mock driver for Replit development
        self.driver = MockNeo4jDriver(neo4j_uri, (neo4j_user, neo4j_password))
        
        # Initialize market data
        self.assets = self._initialize_market_assets()
        self.sectors = self._initialize_sectors()
        
        # Create the market knowledge graph
        self._create_market_graph()
        
    def _initialize_market_assets(self) -> Dict[str, MarketAsset]:
        """Initialize major market assets"""
        assets = {
            # Major Tech Stocks
            "AAPL": MarketAsset("AAPL", "Apple Inc.", "Technology", 3000000000000),
            "GOOGL": MarketAsset("GOOGL", "Alphabet Inc.", "Technology", 1800000000000),
            "MSFT": MarketAsset("MSFT", "Microsoft Corp.", "Technology", 2800000000000),
            "AMZN": MarketAsset("AMZN", "Amazon.com Inc.", "Technology", 1500000000000),
            "TSLA": MarketAsset("TSLA", "Tesla Inc.", "Automotive", 800000000000),
            "META": MarketAsset("META", "Meta Platforms", "Technology", 900000000000),
            "NVDA": MarketAsset("NVDA", "NVIDIA Corp.", "Technology", 1200000000000),
            
            # Financial Sector
            "JPM": MarketAsset("JPM", "JPMorgan Chase", "Financial", 450000000000),
            "BAC": MarketAsset("BAC", "Bank of America", "Financial", 350000000000),
            "WFC": MarketAsset("WFC", "Wells Fargo", "Financial", 200000000000),
            "GS": MarketAsset("GS", "Goldman Sachs", "Financial", 120000000000),
            
            # Healthcare
            "JNJ": MarketAsset("JNJ", "Johnson & Johnson", "Healthcare", 450000000000),
            "PFE": MarketAsset("PFE", "Pfizer Inc.", "Healthcare", 280000000000),
            "UNH": MarketAsset("UNH", "UnitedHealth Group", "Healthcare", 500000000000),
            
            # Energy
            "XOM": MarketAsset("XOM", "Exxon Mobil", "Energy", 400000000000),
            "CVX": MarketAsset("CVX", "Chevron Corp.", "Energy", 300000000000),
            
            # ETFs
            "SPY": MarketAsset("SPY", "SPDR S&P 500 ETF", "Technology", asset_type="etf"),
            "QQQ": MarketAsset("QQQ", "Invesco QQQ Trust", "Technology", asset_type="etf"),
            "IWM": MarketAsset("IWM", "iShares Russell 2000", "Diversified", asset_type="etf"),
            
            # Crypto
            "BTC": MarketAsset("BTC", "Bitcoin", "Cryptocurrency", asset_type="crypto", exchange="CRYPTO"),
            "ETH": MarketAsset("ETH", "Ethereum", "Cryptocurrency", asset_type="crypto", exchange="CRYPTO"),
        }
        return assets
    
    def _initialize_sectors(self) -> Dict[str, SectorInfo]:
        """Initialize sector information"""
        return {
            "Technology": SectorInfo(
                "Technology", 
                "Software, hardware, and technology services",
                typical_pe_ratio=25.0,
                volatility_profile="high",
                economic_sensitivity="medium"
            ),
            "Financial": SectorInfo(
                "Financial",
                "Banking, insurance, and financial services", 
                typical_pe_ratio=12.0,
                volatility_profile="high",
                economic_sensitivity="high"
            ),
            "Healthcare": SectorInfo(
                "Healthcare",
                "Pharmaceuticals, medical devices, healthcare services",
                typical_pe_ratio=18.0,
                volatility_profile="medium",
                economic_sensitivity="low"
            ),
            "Energy": SectorInfo(
                "Energy",
                "Oil, gas, renewable energy companies",
                typical_pe_ratio=15.0,
                volatility_profile="high",
                economic_sensitivity="high"
            ),
            "Automotive": SectorInfo(
                "Automotive", 
                "Vehicle manufacturers and suppliers",
                typical_pe_ratio=20.0,
                volatility_profile="high",
                economic_sensitivity="high"
            ),
            "Cryptocurrency": SectorInfo(
                "Cryptocurrency",
                "Digital assets and blockchain technology",
                typical_pe_ratio=None,
                volatility_profile="high",
                economic_sensitivity="medium"
            ),
            "Diversified": SectorInfo(
                "Diversified",
                "Broad market exposure across sectors",
                typical_pe_ratio=18.0,
                volatility_profile="medium",
                economic_sensitivity="medium"
            )
        }
    
    def _create_market_graph(self):
        """Create the comprehensive market knowledge graph"""
        with self.driver.session() as session:
            # Create asset nodes
            for symbol, asset in self.assets.items():
                session.run("""
                    MERGE (a:Asset {symbol: $symbol})
                    SET a.name = $name,
                        a.sector = $sector,
                        a.market_cap = $market_cap,
                        a.exchange = $exchange,
                        a.asset_type = $asset_type,
                        a.created_at = $timestamp
                """, {
                    'symbol': asset.symbol,
                    'name': asset.name,
                    'sector': asset.sector,
                    'market_cap': asset.market_cap,
                    'exchange': asset.exchange,
                    'asset_type': asset.asset_type,
                    'timestamp': datetime.now().isoformat()
                })
            
            # Create sector nodes
            for sector_name, sector in self.sectors.items():
                session.run("""
                    MERGE (s:Sector {name: $name})
                    SET s.description = $description,
                        s.typical_pe_ratio = $pe_ratio,
                        s.volatility_profile = $volatility,
                        s.economic_sensitivity = $sensitivity,
                        s.created_at = $timestamp
                """, {
                    'name': sector.name,
                    'description': sector.description,
                    'pe_ratio': sector.typical_pe_ratio,
                    'volatility': sector.volatility_profile,
                    'sensitivity': sector.economic_sensitivity,
                    'timestamp': datetime.now().isoformat()
                })
            
            # Create relationships between assets and sectors
            for symbol, asset in self.assets.items():
                session.run("""
                    MATCH (a:Asset {symbol: $symbol})
                    MATCH (s:Sector {name: $sector})
                    MERGE (a)-[r:BELONGS_TO]->(s)
                    SET r.created_at = $timestamp
                """, {
                    'symbol': asset.symbol,
                    'sector': asset.sector,
                    'timestamp': datetime.now().isoformat()
                })
            
            # Create correlation relationships
            self._create_correlation_relationships(session)
            
        self.logger.info(f"Created market knowledge graph with {len(self.assets)} assets and {len(self.sectors)} sectors")
    
    def _create_correlation_relationships(self, session):
        """Create correlation relationships between assets"""
        # Define some typical correlations
        correlations = [
            ("AAPL", "MSFT", 0.75, "HIGH_CORRELATION"),
            ("GOOGL", "META", 0.80, "HIGH_CORRELATION"), 
            ("JPM", "BAC", 0.85, "HIGH_CORRELATION"),
            ("XOM", "CVX", 0.90, "HIGH_CORRELATION"),
            ("SPY", "QQQ", 0.85, "HIGH_CORRELATION"),
            ("BTC", "ETH", 0.75, "HIGH_CORRELATION"),
            ("TSLA", "NVDA", 0.60, "MEDIUM_CORRELATION"),
            ("JNJ", "PFE", 0.70, "HIGH_CORRELATION"),
            # Cross-sector relationships
            ("AAPL", "TSLA", 0.45, "MEDIUM_CORRELATION"),
            ("JPM", "SPY", 0.75, "HIGH_CORRELATION"),
        ]
        
        for symbol1, symbol2, correlation, rel_type in correlations:
            session.run("""
                MATCH (a1:Asset {symbol: $symbol1})
                MATCH (a2:Asset {symbol: $symbol2})
                MERGE (a1)-[r:CORRELATES_WITH]->(a2)
                SET r.correlation = $correlation,
                    r.relationship_type = $rel_type,
                    r.created_at = $timestamp
                MERGE (a2)-[r2:CORRELATES_WITH]->(a1)
                SET r2.correlation = $correlation,
                    r2.relationship_type = $rel_type,
                    r2.created_at = $timestamp
            """, {
                'symbol1': symbol1,
                'symbol2': symbol2,
                'correlation': correlation,
                'rel_type': rel_type,
                'timestamp': datetime.now().isoformat()
            })
    
    def add_pattern_relationship(self, symbol: str, pattern_type: str, 
                               success_rate: float, sample_size: int):
        """Add pattern recognition relationship to asset"""
        with self.driver.session() as session:
            session.run("""
                MATCH (a:Asset {symbol: $symbol})
                MERGE (p:Pattern {type: $pattern_type})
                MERGE (a)-[r:HAS_PATTERN]->(p)
                SET r.success_rate = $success_rate,
                    r.sample_size = $sample_size,
                    r.confidence = $confidence,
                    r.last_updated = $timestamp
            """, {
                'symbol': symbol,
                'pattern_type': pattern_type,
                'success_rate': success_rate,
                'sample_size': sample_size,
                'confidence': min(success_rate * (sample_size / 100), 1.0),
                'timestamp': datetime.now().isoformat()
            })
    
    def get_asset_correlations(self, symbol: str, min_correlation: float = 0.5) -> List[Dict]:
        """Get correlated assets for a given symbol"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (a1:Asset {symbol: $symbol})-[r:CORRELATES_WITH]->(a2:Asset)
                WHERE r.correlation >= $min_correlation
                RETURN a2.symbol as symbol, a2.name as name, 
                       a2.sector as sector, r.correlation as correlation
                ORDER BY r.correlation DESC
            """, {
                'symbol': symbol,
                'min_correlation': min_correlation
            })
            
            return [dict(record) for record in result]
    
    def get_sector_assets(self, sector_name: str) -> List[Dict]:
        """Get all assets in a specific sector"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:Sector {name: $sector})<-[:BELONGS_TO]-(a:Asset)
                RETURN a.symbol as symbol, a.name as name, 
                       a.market_cap as market_cap, a.asset_type as asset_type
                ORDER BY a.market_cap DESC
            """, {'sector': sector_name})
            
            return [dict(record) for record in result]
    
    def find_pattern_leaders(self, pattern_type: str, min_success_rate: float = 0.7) -> List[Dict]:
        """Find assets that consistently show specific patterns"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (a:Asset)-[r:HAS_PATTERN]->(p:Pattern {type: $pattern_type})
                WHERE r.success_rate >= $min_success_rate AND r.sample_size >= 10
                RETURN a.symbol as symbol, a.name as name, a.sector as sector,
                       r.success_rate as success_rate, r.sample_size as sample_size,
                       r.confidence as confidence
                ORDER BY r.confidence DESC, r.sample_size DESC
            """, {
                'pattern_type': pattern_type,
                'min_success_rate': min_success_rate
            })
            
            return [dict(record) for record in result]
    
    def get_market_insights(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive market insights for a symbol"""
        insights = {
            'asset_info': self.assets.get(symbol),
            'correlations': self.get_asset_correlations(symbol),
            'sector_peers': [],
            'patterns': []
        }
        
        if symbol in self.assets:
            sector = self.assets[symbol].sector
            insights['sector_peers'] = [
                asset for asset in self.get_sector_assets(sector)
                if asset['symbol'] != symbol
            ]
        
        return insights
    
    def close(self):
        """Close Neo4j connection"""
        self.driver.close()

class EnhancedDigitalBrain(DigitalBrain):
    """Enhanced Digital Brain with Neo4j integration"""
    
    def __init__(self):
        super().__init__()
        self.neo4j_graph = Neo4jKnowledgeGraph()
        self.logger.info("Enhanced Digital Brain initialized with Neo4j integration")
    
    def process_market_pattern(self, symbol: str, pattern_type: str, 
                             success_rate: float, sample_size: int) -> Dict[str, Any]:
        """Process and store market pattern in both graphs"""
        # Process in original knowledge graph
        result = super().process_market_event({
            'symbol': symbol,
            'pattern_type': pattern_type,
            'outcome': {'successful': success_rate > 0.5}
        })
        
        # Add to Neo4j graph
        self.neo4j_graph.add_pattern_relationship(
            symbol, pattern_type, success_rate, sample_size
        )
        
        # Get market insights
        insights = self.neo4j_graph.get_market_insights(symbol)
        
        result['market_insights'] = insights
        result['neo4j_updated'] = True
        
        return result
    
    def get_enhanced_insights(self, symbol: str, query_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get enhanced insights combining both knowledge systems"""
        # Get original brain insights
        brain_result = self.query_brain(f"Analysis for {symbol}", query_context or {})
        
        # Get Neo4j market insights
        market_insights = self.neo4j_graph.get_market_insights(symbol)
        
        # Combine insights
        enhanced_result = {
            **brain_result,
            'market_insights': market_insights,
            'recommendations': self._generate_enhanced_recommendations(symbol, market_insights),
            'risk_factors': self._assess_risk_factors(symbol, market_insights)
        }
        
        return enhanced_result
    
    def _generate_enhanced_recommendations(self, symbol: str, insights: Dict) -> List[str]:
        """Generate trading recommendations based on market insights"""
        recommendations = []
        
        # Correlation-based recommendations
        correlations = insights.get('correlations', [])
        if correlations:
            high_corr = [c for c in correlations if c['correlation'] > 0.8]
            if high_corr:
                recommendations.append(
                    f"High correlation with {', '.join([c['symbol'] for c in high_corr[:3]])} "
                    f"- consider diversification or sector rotation"
                )
        
        # Sector-based recommendations
        sector_peers = insights.get('sector_peers', [])
        if len(sector_peers) > 0:
            recommendations.append(
                f"Monitor sector peers: {', '.join([p['symbol'] for p in sector_peers[:3]])} "
                f"for relative strength analysis"
            )
        
        # Asset-specific recommendations
        asset_info = insights.get('asset_info')
        if asset_info:
            if asset_info.asset_type == 'etf':
                recommendations.append(
                    "ETF provides diversified exposure - suitable for broad market plays"
                )
            elif asset_info.market_cap and asset_info.market_cap > 1000000000000:  # $1T+
                recommendations.append(
                    "Large-cap stability with institutional support - lower volatility expected"
                )
        
        return recommendations
    
    def _assess_risk_factors(self, symbol: str, insights: Dict) -> List[str]:
        """Assess risk factors based on market structure"""
        risk_factors = []
        
        # Correlation risks
        correlations = insights.get('correlations', [])
        high_corr_count = len([c for c in correlations if c['correlation'] > 0.8])
        
        if high_corr_count > 3:
            risk_factors.append(
                "High correlation cluster - increased systematic risk during market stress"
            )
        
        # Sector concentration
        asset_info = insights.get('asset_info')
        if asset_info and asset_info.sector in ['Technology', 'Cryptocurrency']:
            risk_factors.append(
                f"{asset_info.sector} sector exposure - sensitive to interest rate changes"
            )
        
        # Market cap considerations
        if asset_info and asset_info.market_cap:
            if asset_info.market_cap < 50000000000:  # <$50B
                risk_factors.append(
                    "Mid/small-cap volatility - higher price sensitivity to market sentiment"
                )
        
        return risk_factors
    
    def close(self):
        """Close all connections"""
        self.neo4j_graph.close()
        self.logger.info("Enhanced Digital Brain connections closed")

def main():
    """Test the enhanced knowledge graph system"""
    print("🧠 Enhanced Digital Brain with Neo4j Integration")
    print("=" * 60)
    
    # Initialize enhanced brain
    brain = EnhancedDigitalBrain()
    
    # Test pattern processing
    print("\n📊 Processing Market Patterns...")
    pattern_results = []
    
    test_patterns = [
        ("AAPL", "head_and_shoulders", 0.75, 25),
        ("GOOGL", "breakout", 0.68, 18),
        ("TSLA", "triangle", 0.72, 22),
        ("SPY", "support_resistance", 0.80, 45),
        ("BTC", "flag_pattern", 0.65, 15)
    ]
    
    for symbol, pattern, success_rate, sample_size in test_patterns:
        result = brain.process_market_pattern(symbol, pattern, success_rate, sample_size)
        pattern_results.append((symbol, pattern, result['market_insights']))
        print(f"✅ Processed {pattern} for {symbol}: Success rate {success_rate}")
    
    # Test enhanced insights
    print("\n🔍 Testing Enhanced Insights...")
    
    test_symbols = ["AAPL", "TSLA", "SPY", "BTC"]
    
    for symbol in test_symbols:
        print(f"\n--- {symbol} Analysis ---")
        insights = brain.get_enhanced_insights(symbol)
        
        print(f"Knowledge Matches: {insights.get('knowledge_matches', 0)}")
        print(f"Confidence: {insights.get('confidence', 0):.2f}")
        
        # Market insights
        market_insights = insights.get('market_insights', {})
        correlations = market_insights.get('correlations', [])
        if correlations:
            print(f"Top Correlations: {', '.join([f\"{c['symbol']}({c['correlation']:.2f})\" for c in correlations[:3]])}")
        
        # Recommendations
        recommendations = insights.get('recommendations', [])
        if recommendations:
            print(f"Key Recommendation: {recommendations[0]}")
        
        # Risk factors
        risk_factors = insights.get('risk_factors', [])
        if risk_factors:
            print(f"Primary Risk: {risk_factors[0]}")
    
    # Test pattern leaders
    print("\n🏆 Pattern Leaders Analysis...")
    pattern_types = ["breakout", "head_and_shoulders", "triangle"]
    
    for pattern_type in pattern_types:
        leaders = brain.neo4j_graph.find_pattern_leaders(pattern_type, 0.65)
        if leaders:
            print(f"{pattern_type.title()}: {leaders[0]['symbol']} "
                  f"(Success: {leaders[0]['success_rate']:.2f}, "
                  f"Samples: {leaders[0]['sample_size']})")
    
    # Cleanup
    brain.close()
    
    print(f"\n✅ Enhanced Knowledge Graph Foundation Complete!")
    print(f"🧠 Market assets tracked: {len(brain.neo4j_graph.assets)}")
    print(f"📊 Sectors analyzed: {len(brain.neo4j_graph.sectors)}")
    print(f"🔗 Relationship types: Asset-Sector, Correlations, Patterns")

if __name__ == "__main__":
    main()
