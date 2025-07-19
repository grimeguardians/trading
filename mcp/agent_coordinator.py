import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

from mcp.mcp_server import MCPServer
from core.trading_engine import TradingEngine
from core.knowledge_engine import KnowledgeEngine
from agents.base_agent import BaseAgent, AgentMessage
from agents.market_analyst import MarketAnalystAgent
from agents.trader_agent import TraderAgent
from agents.news_agent import NewsAgent
from agents.strategizer import StrategizerAgent

class AgentCoordinator:
    """Coordinates multiple agents using MCP protocol"""
    
    def __init__(self, mcp_server: MCPServer, trading_engine: TradingEngine, 
                 knowledge_engine: KnowledgeEngine):
        self.mcp_server = mcp_server
        self.trading_engine = trading_engine
        self.knowledge_engine = knowledge_engine
        
        # Agent management
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_dependencies: Dict[str, List[str]] = {}
        
        # Coordination state
        self.coordination_rules: Dict[str, Dict[str, Any]] = {}
        self.workflow_templates: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.coordination_metrics = {
            'workflows_executed': 0,
            'messages_routed': 0,
            'coordination_errors': 0,
            'average_response_time': 0.0
        }
        
        # State management
        self.is_running = False
        self.coordination_tasks = []
        
        self.logger = logging.getLogger("AgentCoordinator")
    
    async def initialize(self):
        """Initialize agent coordinator"""
        try:
            self.logger.info("Initializing Agent Coordinator...")
            
            # Initialize agents
            await self._initialize_agents()
            
            # Setup coordination rules
            await self._setup_coordination_rules()
            
            # Setup workflow templates
            await self._setup_workflow_templates()
            
            # Setup message routing
            await self._setup_message_routing()
            
            self.logger.info("Agent Coordinator initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Agent Coordinator: {e}")
            raise
    
    async def start(self):
        """Start agent coordination"""
        try:
            self.logger.info("Starting Agent Coordinator...")
            
            self.is_running = True
            
            # Start all agents
            for agent_id, agent in self.agents.items():
                success = await agent.start()
                if success:
                    self.logger.info(f"Started agent: {agent_id}")
                else:
                    self.logger.error(f"Failed to start agent: {agent_id}")
            
            # Start coordination tasks
            self.coordination_tasks = [
                asyncio.create_task(self._coordinate_trading_workflow()),
                asyncio.create_task(self._coordinate_market_analysis()),
                asyncio.create_task(self._coordinate_news_processing()),
                asyncio.create_task(self._monitor_agent_health()),
                asyncio.create_task(self._process_coordination_messages())
            ]
            
            self.logger.info("Agent Coordinator started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start Agent Coordinator: {e}")
            raise
    
    async def stop(self):
        """Stop agent coordination"""
        try:
            self.logger.info("Stopping Agent Coordinator...")
            
            self.is_running = False
            
            # Cancel coordination tasks
            for task in self.coordination_tasks:
                task.cancel()
            
            # Stop all agents
            for agent_id, agent in self.agents.items():
                await agent.stop()
                self.logger.info(f"Stopped agent: {agent_id}")
            
            self.logger.info("Agent Coordinator stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping Agent Coordinator: {e}")
    
    async def _initialize_agents(self):
        """Initialize all agents"""
        try:
            # Market Analyst Agent
            market_analyst = MarketAnalystAgent(
                agent_id="market_analyst",
                config={
                    'analysis_symbols': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'SPY', 'QQQ'],
                    'analysis_timeframes': ['1h', '4h', '1d'],
                    'analysis_interval': 300
                },
                knowledge_engine=self.knowledge_engine,
                exchange_factory=self.trading_engine.exchange_factory
            )
            
            # Trader Agent
            trader = TraderAgent(
                agent_id="trader",
                config={
                    'max_positions': 10,
                    'max_order_size': 10000,
                    'min_confidence': 0.6,
                    'risk_per_trade': 0.01
                },
                trading_engine=self.trading_engine,
                knowledge_engine=self.knowledge_engine,
                risk_manager=self.trading_engine.risk_manager,
                exchange_factory=self.trading_engine.exchange_factory
            )
            
            # News Agent
            news_agent = NewsAgent(
                agent_id="news_agent",
                config={
                    'news_sources': ['alpha_vantage', 'yahoo_finance'],
                    'tracked_symbols': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'],
                    'update_interval': 300,
                    'sentiment_threshold': 0.1
                },
                knowledge_engine=self.knowledge_engine
            )
            
            # Strategizer Agent
            strategizer = StrategizerAgent(
                agent_id="strategizer",
                config={
                    'optimization_interval': 3600,
                    'min_backtest_days': 30,
                    'min_confidence_threshold': 0.6
                },
                knowledge_engine=self.knowledge_engine,
                trading_engine=self.trading_engine,
                exchange_factory=self.trading_engine.exchange_factory
            )
            
            # Add agents to registry
            self.agents = {
                'market_analyst': market_analyst,
                'trader': trader,
                'news_agent': news_agent,
                'strategizer': strategizer
            }
            
            # Setup agent dependencies
            self.agent_dependencies = {
                'trader': ['market_analyst', 'strategizer'],
                'strategizer': ['market_analyst', 'news_agent'],
                'market_analyst': ['news_agent'],
                'news_agent': []
            }
            
        except Exception as e:
            self.logger.error(f"Error initializing agents: {e}")
            raise
    
    async def _setup_coordination_rules(self):
        """Setup coordination rules between agents"""
        try:
            self.coordination_rules = {
                'trading_signal_processing': {
                    'trigger': 'market_analysis_complete',
                    'workflow': [
                        {'agent': 'market_analyst', 'action': 'analyze_symbol'},
                        {'agent': 'strategizer', 'action': 'recommend_strategy'},
                        {'agent': 'trader', 'action': 'execute_trade'}
                    ],
                    'conditions': {
                        'min_confidence': 0.6,
                        'max_risk': 0.02
                    }
                },
                'news_impact_analysis': {
                    'trigger': 'breaking_news',
                    'workflow': [
                        {'agent': 'news_agent', 'action': 'analyze_impact'},
                        {'agent': 'market_analyst', 'action': 'reassess_analysis'},
                        {'agent': 'trader', 'action': 'adjust_positions'}
                    ],
                    'conditions': {
                        'urgency_threshold': 0.8,
                        'sentiment_threshold': 0.5
                    }
                },
                'strategy_optimization': {
                    'trigger': 'scheduled_optimization',
                    'workflow': [
                        {'agent': 'market_analyst', 'action': 'analyze_performance'},
                        {'agent': 'strategizer', 'action': 'optimize_strategy'},
                        {'agent': 'trader', 'action': 'update_parameters'}
                    ],
                    'conditions': {
                        'optimization_interval': 3600
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error setting up coordination rules: {e}")
            raise
    
    async def _setup_workflow_templates(self):
        """Setup workflow templates for common operations"""
        try:
            self.workflow_templates = {
                'standard_trading_workflow': {
                    'name': 'Standard Trading Workflow',
                    'description': 'Complete trading workflow from analysis to execution',
                    'steps': [
                        {
                            'step': 1,
                            'agent': 'market_analyst',
                            'action': 'analyze_symbol',
                            'inputs': ['symbol', 'timeframe'],
                            'outputs': ['analysis_result']
                        },
                        {
                            'step': 2,
                            'agent': 'strategizer',
                            'action': 'recommend_strategy',
                            'inputs': ['analysis_result', 'risk_tolerance'],
                            'outputs': ['strategy_recommendation']
                        },
                        {
                            'step': 3,
                            'agent': 'trader',
                            'action': 'execute_trade',
                            'inputs': ['strategy_recommendation'],
                            'outputs': ['trade_result']
                        }
                    ]
                },
                'news_driven_trading': {
                    'name': 'News-Driven Trading',
                    'description': 'Trading workflow triggered by news events',
                    'steps': [
                        {
                            'step': 1,
                            'agent': 'news_agent',
                            'action': 'analyze_news_impact',
                            'inputs': ['news_article'],
                            'outputs': ['impact_analysis']
                        },
                        {
                            'step': 2,
                            'agent': 'market_analyst',
                            'action': 'correlate_with_technicals',
                            'inputs': ['impact_analysis'],
                            'outputs': ['combined_analysis']
                        },
                        {
                            'step': 3,
                            'agent': 'trader',
                            'action': 'execute_news_trade',
                            'inputs': ['combined_analysis'],
                            'outputs': ['trade_result']
                        }
                    ]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error setting up workflow templates: {e}")
            raise
    
    async def _setup_message_routing(self):
        """Setup message routing between agents"""
        try:
            # Subscribe agents to relevant message types
            subscriptions = {
                'market_analyst': ['price_update', 'news_event', 'technical_signal'],
                'trader': ['trade_signal', 'risk_alert', 'position_update'],
                'news_agent': ['market_event', 'earnings_announcement'],
                'strategizer': ['performance_update', 'market_regime_change']
            }
            
            # Setup subscriptions
            for agent_id, message_types in subscriptions.items():
                if agent_id in self.agents:
                    for message_type in message_types:
                        # In a real implementation, would setup proper subscriptions
                        pass
                        
        except Exception as e:
            self.logger.error(f"Error setting up message routing: {e}")
            raise
    
    async def _coordinate_trading_workflow(self):
        """Coordinate the main trading workflow"""
        while self.is_running:
            try:
                # Get symbols to analyze
                symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
                
                for symbol in symbols:
                    # Execute trading workflow
                    await self._execute_workflow('standard_trading_workflow', {
                        'symbol': symbol,
                        'timeframe': '1h',
                        'risk_tolerance': 'medium'
                    })
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in trading workflow coordination: {e}")
                await asyncio.sleep(300)
    
    async def _coordinate_market_analysis(self):
        """Coordinate market analysis tasks"""
        while self.is_running:
            try:
                # Request market sentiment analysis
                if 'market_analyst' in self.agents:
                    analyst = self.agents['market_analyst']
                    
                    # Create analysis request
                    analysis_request = AgentMessage(
                        id=f"analysis_request_{datetime.now().timestamp()}",
                        sender="coordinator",
                        receiver="market_analyst",
                        message_type="market_sentiment",
                        content={
                            'symbols': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
                        },
                        timestamp=datetime.now()
                    )
                    
                    # Send message to analyst
                    await analyst.receive_message(analysis_request)
                
                await asyncio.sleep(600)  # Run every 10 minutes
                
            except Exception as e:
                self.logger.error(f"Error in market analysis coordination: {e}")
                await asyncio.sleep(600)
    
    async def _coordinate_news_processing(self):
        """Coordinate news processing tasks"""
        while self.is_running:
            try:
                # Request news summary
                if 'news_agent' in self.agents:
                    news_agent = self.agents['news_agent']
                    
                    # Create news request
                    news_request = AgentMessage(
                        id=f"news_request_{datetime.now().timestamp()}",
                        sender="coordinator",
                        receiver="news_agent",
                        message_type="market_summary",
                        content={
                            'timeframe': '1h'
                        },
                        timestamp=datetime.now()
                    )
                    
                    # Send message to news agent
                    await news_agent.receive_message(news_request)
                
                await asyncio.sleep(900)  # Run every 15 minutes
                
            except Exception as e:
                self.logger.error(f"Error in news processing coordination: {e}")
                await asyncio.sleep(900)
    
    async def _monitor_agent_health(self):
        """Monitor agent health and performance"""
        while self.is_running:
            try:
                unhealthy_agents = []
                
                for agent_id, agent in self.agents.items():
                    # Check agent status
                    if not agent.is_running:
                        unhealthy_agents.append(agent_id)
                        continue
                    
                    # Send health check
                    health_check = AgentMessage(
                        id=f"health_check_{datetime.now().timestamp()}",
                        sender="coordinator",
                        receiver=agent_id,
                        message_type="ping",
                        content={},
                        timestamp=datetime.now()
                    )
                    
                    await agent.receive_message(health_check)
                
                # Handle unhealthy agents
                for agent_id in unhealthy_agents:
                    await self._handle_unhealthy_agent(agent_id)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error monitoring agent health: {e}")
                await asyncio.sleep(60)
    
    async def _handle_unhealthy_agent(self, agent_id: str):
        """Handle unhealthy agent"""
        try:
            self.logger.warning(f"Handling unhealthy agent: {agent_id}")
            
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                
                # Try to restart agent
                await agent.stop()
                success = await agent.start()
                
                if success:
                    self.logger.info(f"Successfully restarted agent: {agent_id}")
                else:
                    self.logger.error(f"Failed to restart agent: {agent_id}")
                    
                    # Disable agent temporarily
                    agent.is_running = False
                    
        except Exception as e:
            self.logger.error(f"Error handling unhealthy agent {agent_id}: {e}")
    
    async def _process_coordination_messages(self):
        """Process coordination messages"""
        while self.is_running:
            try:
                # Check for messages that need coordination
                # This would integrate with the MCP server to process messages
                
                await asyncio.sleep(1)  # Process frequently
                
            except Exception as e:
                self.logger.error(f"Error processing coordination messages: {e}")
                await asyncio.sleep(1)
    
    async def _execute_workflow(self, workflow_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow template"""
        try:
            if workflow_name not in self.workflow_templates:
                raise ValueError(f"Workflow template {workflow_name} not found")
            
            workflow = self.workflow_templates[workflow_name]
            results = {}
            
            self.logger.info(f"Executing workflow: {workflow_name}")
            
            # Execute each step
            for step in workflow['steps']:
                step_num = step['step']
                agent_id = step['agent']
                action = step['action']
                step_inputs = step.get('inputs', [])
                
                # Prepare step inputs
                step_data = {}
                for input_name in step_inputs:
                    if input_name in inputs:
                        step_data[input_name] = inputs[input_name]
                    elif input_name in results:
                        step_data[input_name] = results[input_name]
                
                # Execute step
                step_result = await self._execute_workflow_step(
                    agent_id, action, step_data
                )
                
                if step_result:
                    # Store step outputs
                    for output_name in step.get('outputs', []):
                        if output_name in step_result:
                            results[output_name] = step_result[output_name]
                
                self.logger.info(f"Completed workflow step {step_num}: {action}")
            
            self.coordination_metrics['workflows_executed'] += 1
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error executing workflow {workflow_name}: {e}")
            self.coordination_metrics['coordination_errors'] += 1
            return {}
    
    async def _execute_workflow_step(self, agent_id: str, action: str, 
                                   inputs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute a single workflow step"""
        try:
            if agent_id not in self.agents:
                self.logger.error(f"Agent {agent_id} not found")
                return None
            
            agent = self.agents[agent_id]
            
            # Create message for agent
            message = AgentMessage(
                id=f"workflow_step_{datetime.now().timestamp()}",
                sender="coordinator",
                receiver=agent_id,
                message_type=action,
                content=inputs,
                timestamp=datetime.now()
            )
            
            # Send message to agent
            response = await agent.process_message(message)
            
            if response:
                return response.content
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error executing workflow step: {e}")
            return None
    
    # Public API methods
    async def trigger_workflow(self, workflow_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger a workflow execution"""
        return await self._execute_workflow(workflow_name, inputs)
    
    async def send_message_to_agent(self, agent_id: str, message_type: str, 
                                   content: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send message to specific agent"""
        try:
            if agent_id not in self.agents:
                return None
            
            agent = self.agents[agent_id]
            
            message = AgentMessage(
                id=f"coord_msg_{datetime.now().timestamp()}",
                sender="coordinator",
                receiver=agent_id,
                message_type=message_type,
                content=content,
                timestamp=datetime.now()
            )
            
            response = await agent.process_message(message)
            
            if response:
                return response.content
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error sending message to agent: {e}")
            return None
    
    async def broadcast_message(self, message_type: str, content: Dict[str, Any]):
        """Broadcast message to all agents"""
        try:
            for agent_id, agent in self.agents.items():
                message = AgentMessage(
                    id=f"broadcast_{datetime.now().timestamp()}",
                    sender="coordinator",
                    receiver=agent_id,
                    message_type=message_type,
                    content=content,
                    timestamp=datetime.now()
                )
                
                await agent.receive_message(message)
                
        except Exception as e:
            self.logger.error(f"Error broadcasting message: {e}")
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        return {
            'total_agents': len(self.agents),
            'agents': {
                agent_id: {
                    'name': agent.name,
                    'is_running': agent.is_running,
                    'status': agent.status.value,
                    'last_activity': agent.last_activity.isoformat(),
                    'performance_metrics': agent.performance_metrics
                }
                for agent_id, agent in self.agents.items()
            },
            'coordination_metrics': self.coordination_metrics
        }
    
    def get_coordination_metrics(self) -> Dict[str, Any]:
        """Get coordination metrics"""
        return {
            'coordination_metrics': self.coordination_metrics,
            'agent_dependencies': self.agent_dependencies,
            'workflow_templates': list(self.workflow_templates.keys()),
            'coordination_rules': list(self.coordination_rules.keys())
        }
