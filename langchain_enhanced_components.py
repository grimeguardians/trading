
"""
LangChain Enhanced Components for Trading System
Integrates popular LangChain templates for enhanced AI capabilities
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import json
from datetime import datetime
import logging
import asyncio
import re

try:
    # Core LangChain imports
    from langchain.document_loaders import PyPDFLoader, TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    from langchain.output_parsers import PydanticOutputParser, StructuredOutputParser, ResponseSchema
    from langchain.prompts import PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate
    from langchain.chains import LLMChain, SequentialChain
    from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
    from langchain.agents import Tool, AgentExecutor, create_react_agent
    from langchain.callbacks import StdOutCallbackHandler
    from langchain.vectorstores import FAISS
    from langchain.embeddings import OpenAIEmbeddings
    from pydantic import BaseModel, Field
    LANGCHAIN_AVAILABLE = True
    print("📦 Enhanced LangChain components loaded successfully!")
except ImportError:
    print("⚠️ LangChain not available - using fallback implementations")
    LANGCHAIN_AVAILABLE = False
    Document = None
    BaseModel = object

class TradingSignalOutput(BaseModel if LANGCHAIN_AVAILABLE else object):
    """Pydantic model for structured trading signal output"""
    if LANGCHAIN_AVAILABLE:
        symbol: str = Field(description="Stock symbol (e.g., AAPL)")
        action: str = Field(description="Trading action: BUY, SELL, or HOLD")
        confidence: float = Field(description="Confidence score between 0 and 1", ge=0, le=1)
        reasoning: str = Field(description="Detailed reasoning for the trading decision")
        stop_loss: Optional[float] = Field(description="Stop loss price", default=None)
        take_profit: Optional[float] = Field(description="Take profit price", default=None)
        position_size: float = Field(description="Position size as percentage of portfolio", ge=0, le=1)
        risk_level: str = Field(description="Risk level: low, medium, high")
        timeframe: str = Field(description="Trading timeframe: 1m, 5m, 1h, 1d")

class MarketAnalysisOutput(BaseModel if LANGCHAIN_AVAILABLE else object):
    """Pydantic model for market analysis output"""
    if LANGCHAIN_AVAILABLE:
        market_sentiment: str = Field(description="Overall market sentiment: bullish, bearish, neutral")
        key_patterns: List[str] = Field(description="List of identified chart patterns")
        support_resistance: Dict[str, float] = Field(description="Key support and resistance levels")
        risk_factors: List[str] = Field(description="Identified risk factors")
        opportunities: List[str] = Field(description="Trading opportunities")
        confidence_score: float = Field(description="Analysis confidence", ge=0, le=1)

class RAGTradingKnowledge:
    """RAG (Retrieval-Augmented Generation) for trading knowledge"""
    
    def __init__(self, digital_brain=None):
        self.digital_brain = digital_brain
        self.vector_store = None
        self.embeddings = None
        self.logger = logging.getLogger("RAGKnowledge")
        
        if LANGCHAIN_AVAILABLE:
            try:
                self.embeddings = OpenAIEmbeddings()
                self._initialize_vector_store()
            except Exception as e:
                self.logger.warning(f"OpenAI embeddings not available: {e}")
    
    def _initialize_vector_store(self):
        """Initialize FAISS vector store for trading knowledge"""
        if not self.digital_brain:
            return
        
        # Convert Digital Brain knowledge to documents
        documents = []
        memory_bank = self.digital_brain.memory_bank
        
        for doc_id, doc_data in memory_bank.get('stored_documents', {}).items():
            content = doc_data.get('content', '')
            metadata = {
                'doc_id': doc_id,
                'title': doc_data.get('title', 'Unknown'),
                'doc_type': doc_data.get('doc_type', 'unknown'),
                'timestamp': doc_data.get('timestamp', '')
            }
            documents.append(Document(page_content=content, metadata=metadata))
        
        if documents and self.embeddings:
            try:
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
                self.logger.info(f"Initialized RAG vector store with {len(documents)} documents")
            except Exception as e:
                self.logger.error(f"Failed to initialize vector store: {e}")
    
    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """Perform similarity search on trading knowledge"""
        if not self.vector_store:
            return []
        
        try:
            return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            self.logger.error(f"Similarity search failed: {e}")
            return []

class TradingChainTemplates:
    """Popular LangChain chain templates for trading"""
    
    def __init__(self, rag_knowledge: RAGTradingKnowledge = None):
        self.rag_knowledge = rag_knowledge
        self.logger = logging.getLogger("TradingChains")
        self.memory = None
        
        if LANGCHAIN_AVAILABLE:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
    
    def create_signal_generation_chain(self) -> Optional[Any]:
        """Create signal generation chain with RAG enhancement"""
        if not LANGCHAIN_AVAILABLE:
            return None
        
        # Enhanced prompt with RAG context
        signal_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert trading analyst with access to comprehensive market knowledge.
            
            Use the following trading knowledge context to inform your analysis:
            {knowledge_context}
            
            Analyze the market data and provide a structured trading recommendation.
            Consider technical indicators, market sentiment, risk factors, and historical patterns.
            """),
            ("human", """
            Market Data:
            Symbol: {symbol}
            Current Price: ${current_price}
            RSI: {rsi}
            MACD: {macd}
            Volume: {volume}
            Support: ${support}
            Resistance: ${resistance}
            Market Sentiment: {sentiment}
            
            Provide a detailed trading signal with clear reasoning.
            """)
        ])
        
        # Create chain with memory
        if hasattr(self, 'llm'):  # Would be set externally
            return LLMChain(
                llm=self.llm,
                prompt=signal_prompt,
                memory=self.memory,
                verbose=True
            )
        return signal_prompt
    
    def create_market_analysis_chain(self) -> Optional[Any]:
        """Create comprehensive market analysis chain"""
        if not LANGCHAIN_AVAILABLE:
            return None
        
        # Multi-step analysis chain
        trend_analysis_prompt = PromptTemplate(
            input_variables=["market_data", "knowledge_context"],
            template="""
            Based on the market data and historical knowledge:
            {knowledge_context}
            
            Market Data: {market_data}
            
            Analyze the current trend and provide:
            1. Trend direction and strength
            2. Key support/resistance levels
            3. Momentum indicators assessment
            """
        )
        
        risk_analysis_prompt = PromptTemplate(
            input_variables=["trend_analysis", "market_data"],
            template="""
            Given the trend analysis: {trend_analysis}
            And market data: {market_data}
            
            Assess the risk factors:
            1. Market volatility analysis
            2. Potential reversal signals
            3. Risk/reward scenarios
            """
        )
        
        # Sequential chain for comprehensive analysis
        if hasattr(self, 'llm'):
            trend_chain = LLMChain(llm=self.llm, prompt=trend_analysis_prompt, output_key="trend_analysis")
            risk_chain = LLMChain(llm=self.llm, prompt=risk_analysis_prompt, output_key="risk_analysis")
            
            return SequentialChain(
                chains=[trend_chain, risk_chain],
                input_variables=["market_data", "knowledge_context"],
                output_variables=["trend_analysis", "risk_analysis"],
                verbose=True
            )
        return [trend_analysis_prompt, risk_analysis_prompt]
    
    def create_few_shot_pattern_recognition(self) -> Optional[Any]:
        """Create few-shot learning chain for pattern recognition"""
        if not LANGCHAIN_AVAILABLE:
            return None
        
        # Few-shot examples for pattern recognition
        pattern_examples = [
            {
                "input": "Price formed higher highs and higher lows, volume increasing on breakouts, RSI above 50",
                "output": "Bullish trend continuation pattern. Strong upward momentum with volume confirmation. High probability of continued upward movement."
            },
            {
                "input": "Double top formation at resistance, declining volume on second peak, RSI showing bearish divergence",
                "output": "Bearish reversal pattern. Double top with weakening momentum suggests trend exhaustion. High probability of downward reversal."
            },
            {
                "input": "Triangular consolidation with converging trend lines, decreasing volume, RSI neutral",
                "output": "Neutral consolidation pattern. Awaiting breakout direction. Volume and momentum will confirm breakout validity."
            }
        ]
        
        example_formatter_template = """
        Market Conditions: {input}
        Pattern Analysis: {output}
        """
        
        example_prompt = PromptTemplate(
            input_variables=["input", "output"],
            template=example_formatter_template
        )
        
        few_shot_prompt = FewShotPromptTemplate(
            examples=pattern_examples,
            example_prompt=example_prompt,
            prefix="You are an expert at recognizing chart patterns. Analyze the following market conditions:",
            suffix="Market Conditions: {market_conditions}\nPattern Analysis:",
            input_variables=["market_conditions"]
        )
        
        return few_shot_prompt

class TradingAgentTools:
    """Trading-specific tools for LangChain agents"""
    
    def __init__(self, digital_brain=None, trading_system=None):
        self.digital_brain = digital_brain
        self.trading_system = trading_system
        self.logger = logging.getLogger("TradingTools")
    
    def get_tools(self) -> List[Any]:
        """Get list of trading tools for agents"""
        if not LANGCHAIN_AVAILABLE:
            return []
        
        tools = []
        
        # Knowledge Query Tool
        if self.digital_brain:
            tools.append(Tool(
                name="query_trading_knowledge",
                description="Query the trading knowledge base for patterns, strategies, and market insights",
                func=self._query_knowledge_tool
            ))
        
        # Market Data Tool
        tools.append(Tool(
            name="get_market_data",
            description="Get current market data for a specific symbol including price, volume, and technical indicators",
            func=self._get_market_data_tool
        ))
        
        # Risk Assessment Tool
        tools.append(Tool(
            name="assess_risk",
            description="Assess risk for a potential trade including position sizing and stop-loss recommendations",
            func=self._assess_risk_tool
        ))
        
        # Pattern Recognition Tool
        tools.append(Tool(
            name="recognize_patterns",
            description="Identify chart patterns and technical formations in market data",
            func=self._pattern_recognition_tool
        ))
        
        return tools
    
    def _query_knowledge_tool(self, query: str) -> str:
        """Tool function for querying trading knowledge"""
        if not self.digital_brain:
            return "Trading knowledge not available"
        
        try:
            result = self.digital_brain.query_brain(query, {'query_type': 'knowledge'})
            insights = result.get('insights', [])
            return f"Knowledge insights: {'; '.join(insights[:3])}"
        except Exception as e:
            return f"Knowledge query failed: {str(e)}"
    
    def _get_market_data_tool(self, symbol: str) -> str:
        """Tool function for getting market data"""
        # This would integrate with real market data
        return f"Market data for {symbol}: Price=$150.25, Volume=1.2M, RSI=65, MACD=2.1"
    
    def _assess_risk_tool(self, trade_details: str) -> str:
        """Tool function for risk assessment"""
        # This would integrate with risk management system
        return f"Risk assessment for {trade_details}: Medium risk, 2% position size recommended, stop-loss at 5% below entry"
    
    def _pattern_recognition_tool(self, market_data: str) -> str:
        """Tool function for pattern recognition"""
        # This would integrate with pattern recognition system
        return f"Pattern analysis for {market_data}: Ascending triangle detected, bullish breakout probability 70%"

class StructuredOutputParser:
    """Enhanced structured output parsing"""
    
    def __init__(self):
        self.logger = logging.getLogger("OutputParser")
        self.signal_parser = None
        self.analysis_parser = None
        
        if LANGCHAIN_AVAILABLE:
            self.signal_parser = PydanticOutputParser(pydantic_object=TradingSignalOutput)
            self.analysis_parser = PydanticOutputParser(pydantic_object=MarketAnalysisOutput)
    
    def parse_trading_signal(self, ai_response: str) -> Dict[str, Any]:
        """Parse AI response into structured trading signal"""
        if self.signal_parser:
            try:
                return self.signal_parser.parse(ai_response).dict()
            except Exception as e:
                self.logger.error(f"Structured parsing failed: {e}")
        
        # Fallback parsing
        return self._fallback_signal_parsing(ai_response)
    
    def parse_market_analysis(self, ai_response: str) -> Dict[str, Any]:
        """Parse AI response into structured market analysis"""
        if self.analysis_parser:
            try:
                return self.analysis_parser.parse(ai_response).dict()
            except Exception as e:
                self.logger.error(f"Analysis parsing failed: {e}")
        
        # Fallback parsing
        return self._fallback_analysis_parsing(ai_response)
    
    def _fallback_signal_parsing(self, response: str) -> Dict[str, Any]:
        """Fallback signal parsing without LangChain"""
        try:
            if '{' in response and '}' in response:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                json_str = response[json_start:json_end]
                return json.loads(json_str)
        except:
            pass
        
        # Extract key information with regex
        signal = {
            'symbol': self._extract_pattern(response, r'symbol[:\s]+([A-Z]{1,5})', 'UNKNOWN'),
            'action': self._extract_pattern(response, r'action[:\s]+(BUY|SELL|HOLD)', 'HOLD'),
            'confidence': float(self._extract_pattern(response, r'confidence[:\s]+(\d*\.?\d+)', '0.5')),
            'reasoning': response[:200] + '...' if len(response) > 200 else response
        }
        
        return signal
    
    def _fallback_analysis_parsing(self, response: str) -> Dict[str, Any]:
        """Fallback analysis parsing without LangChain"""
        return {
            'market_sentiment': self._extract_pattern(response, r'sentiment[:\s]+(bullish|bearish|neutral)', 'neutral'),
            'key_patterns': re.findall(r'pattern[s]?[:\s]+([^.]+)', response.lower()),
            'confidence_score': float(self._extract_pattern(response, r'confidence[:\s]+(\d*\.?\d+)', '0.5')),
            'analysis_summary': response[:300] + '...' if len(response) > 300 else response
        }
    
    def _extract_pattern(self, text: str, pattern: str, default: str) -> str:
        """Extract pattern from text with default fallback"""
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1) if match else default

class EnhancedTradingProcessor:
    """Main processor combining all LangChain enhancements"""
    
    def __init__(self, digital_brain=None, trading_system=None):
        self.digital_brain = digital_brain
        self.trading_system = trading_system
        self.rag_knowledge = RAGTradingKnowledge(digital_brain)
        self.chain_templates = TradingChainTemplates(self.rag_knowledge)
        self.agent_tools = TradingAgentTools(digital_brain, trading_system)
        self.output_parser = StructuredOutputParser()
        self.logger = logging.getLogger("EnhancedProcessor")
    
    def process_trading_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process trading signal with RAG enhancement"""
        
        # Get relevant knowledge context
        symbol = market_data.get('symbol', 'UNKNOWN')
        query = f"trading strategies for {symbol} technical analysis patterns"
        
        knowledge_docs = self.rag_knowledge.similarity_search(query, k=3)
        knowledge_context = "\n".join([doc.page_content[:500] for doc in knowledge_docs])
        
        # Prepare input for chain
        chain_input = {
            'symbol': symbol,
            'current_price': market_data.get('price', 0),
            'rsi': market_data.get('rsi', 50),
            'macd': market_data.get('macd', 0),
            'volume': market_data.get('volume', 0),
            'support': market_data.get('support', 0),
            'resistance': market_data.get('resistance', 0),
            'sentiment': market_data.get('sentiment', 'neutral'),
            'knowledge_context': knowledge_context
        }
        
        # For now, generate a structured response
        # In production, this would use the actual LLM chain
        mock_response = f"""
        {{
            "symbol": "{symbol}",
            "action": "BUY",
            "confidence": 0.75,
            "reasoning": "Based on technical analysis and knowledge context: {knowledge_context[:100]}...",
            "stop_loss": {market_data.get('price', 100) * 0.95},
            "take_profit": {market_data.get('price', 100) * 1.10},
            "position_size": 0.03,
            "risk_level": "medium",
            "timeframe": "1d"
        }}
        """
        
        return self.output_parser.parse_trading_signal(mock_response)
    
    def process_market_analysis(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process comprehensive market analysis"""
        
        # Get knowledge context
        query = "market analysis technical indicators chart patterns"
        knowledge_docs = self.rag_knowledge.similarity_search(query, k=3)
        knowledge_context = "\n".join([doc.page_content[:500] for doc in knowledge_docs])
        
        # Generate analysis
        mock_analysis = f"""
        {{
            "market_sentiment": "bullish",
            "key_patterns": ["ascending triangle", "volume breakout"],
            "support_resistance": {{"support": {market_data.get('price', 100) * 0.95}, "resistance": {market_data.get('price', 100) * 1.08}}},
            "risk_factors": ["market volatility", "economic uncertainty"],
            "opportunities": ["momentum continuation", "breakout trading"],
            "confidence_score": 0.78
        }}
        """
        
        return self.output_parser.parse_market_analysis(mock_analysis)
    
    def get_enhanced_insights(self, query: str) -> Dict[str, Any]:
        """Get enhanced insights using all available tools"""
        
        insights = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'sources': []
        }
        
        # Query Digital Brain
        if self.digital_brain:
            brain_result = self.digital_brain.query_brain(query, {'query_type': 'insight'})
            insights['brain_insights'] = brain_result.get('insights', [])
            insights['sources'].append('digital_brain')
        
        # RAG similarity search
        rag_docs = self.rag_knowledge.similarity_search(query, k=3)
        insights['rag_context'] = [doc.page_content[:300] for doc in rag_docs]
        insights['sources'].append('rag_knowledge')
        
        # Tool-based insights
        tools = self.agent_tools.get_tools()
        if tools:
            for tool in tools:
                if 'knowledge' in tool.name:
                    try:
                        tool_result = tool.func(query)
                        insights['tool_insights'] = tool_result
                        insights['sources'].append(tool.name)
                        break
                    except Exception as e:
                        self.logger.error(f"Tool {tool.name} failed: {e}")
        
        return insights

def test_enhanced_langchain_integration():
    """Test enhanced LangChain integration"""
    print("🧪 Testing Enhanced LangChain Integration")
    print("=" * 60)
    
    # Mock digital brain for testing
    class MockDigitalBrain:
        def __init__(self):
            self.memory_bank = {
                'stored_documents': {
                    'doc1': {
                        'content': 'Technical analysis patterns include triangles, flags, and head and shoulders formations.',
                        'title': 'Chart Patterns Guide',
                        'doc_type': 'educational'
                    }
                }
            }
        
        def query_brain(self, query, context):
            return {
                'insights': [f"Insight for: {query}"],
                'confidence': 0.8
            }
    
    mock_brain = MockDigitalBrain()
    processor = EnhancedTradingProcessor(digital_brain=mock_brain)
    
    # Test signal processing
    market_data = {
        'symbol': 'AAPL',
        'price': 150.25,
        'rsi': 65,
        'macd': 2.1,
        'volume': 1200000,
        'sentiment': 'bullish'
    }
    
    signal_result = processor.process_trading_signal(market_data)
    print(f"📊 Signal Processing: {'✅ Success' if 'action' in signal_result else '❌ Failed'}")
    print(f"   Action: {signal_result.get('action', 'Unknown')}")
    print(f"   Confidence: {signal_result.get('confidence', 0)}")
    
    # Test market analysis
    analysis_result = processor.process_market_analysis(market_data)
    print(f"📈 Market Analysis: {'✅ Success' if 'market_sentiment' in analysis_result else '❌ Failed'}")
    print(f"   Sentiment: {analysis_result.get('market_sentiment', 'Unknown')}")
    
    # Test enhanced insights
    insights = processor.get_enhanced_insights("What are the best breakout patterns?")
    print(f"🔍 Enhanced Insights: {'✅ Success' if insights['sources'] else '❌ Failed'}")
    print(f"   Sources: {', '.join(insights['sources'])}")
    
    print(f"\n🔧 LangChain Status: {'✅ Available' if LANGCHAIN_AVAILABLE else '⚠️ Fallback Mode'}")

if __name__ == "__main__":
    test_enhanced_langchain_integration()
