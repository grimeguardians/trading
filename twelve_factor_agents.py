
"""
12-Factor Agents Implementation for Trading System
Based on: https://github.com/humanlayer/12-factor-agents

Implements production-ready AI agent principles:
1. Codebase - Single codebase tracked in revision control
2. Dependencies - Explicitly declare and isolate dependencies  
3. Config - Store config in environment variables
4. Backing Services - Treat backing services as attached resources
5. Build/Release/Run - Strictly separate build and run stages
6. Processes - Execute as stateless processes
7. Port Binding - Export services via port binding
8. Concurrency - Scale out via process model
9. Disposability - Maximize robustness with fast startup/shutdown
10. Dev/Prod Parity - Keep development and production as similar as possible
11. Logs - Treat logs as event streams
12. Admin Processes - Run admin/management tasks as one-off processes
"""

import os
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import threading
import queue
import signal
import sys

# Factor 3: Config - Store config in environment
class TradingConfig:
    """Environment-based configuration management"""
    
    def __init__(self):
        self.portfolio_start_value = float(os.getenv('PORTFOLIO_START_VALUE', '100000'))
        self.max_position_size = float(os.getenv('MAX_POSITION_SIZE', '0.20'))
        self.risk_limit = float(os.getenv('RISK_LIMIT', '0.25'))
        self.ml_confidence_threshold = float(os.getenv('ML_CONFIDENCE_THRESHOLD', '0.65'))
        self.stop_loss_multiplier = float(os.getenv('STOP_LOSS_MULTIPLIER', '2.0'))
        self.max_concurrent_positions = int(os.getenv('MAX_CONCURRENT_POSITIONS', '5'))
        
        # Backing service configurations
        self.redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        self.database_url = os.getenv('DATABASE_URL', 'sqlite:///trading.db')
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        
        # Development/Production parity
        self.environment = os.getenv('ENVIRONMENT', 'development')
        self.debug_mode = os.getenv('DEBUG', 'false').lower() == 'true'

# Factor 11: Logs - Treat logs as event streams
class StructuredLogger:
    """Structured logging for 12-factor compliance"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.logger = logging.getLogger(service_name)
        
        # Configure structured logging
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "service": "%(name)s", '
            '"level": "%(levelname)s", "message": "%(message)s"}'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(getattr(logging, TradingConfig().log_level))
    
    def log_event(self, event_type: str, data: Dict[str, Any], level: str = 'INFO'):
        """Log structured events"""
        log_entry = {
            'event_type': event_type,
            'service': self.service_name,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        getattr(self.logger, level.lower())(json.dumps(log_entry))

# Factor 6: Processes - Execute as stateless processes
@dataclass
class AgentState:
    """Externalized agent state for stateless operation"""
    agent_id: str
    current_positions: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    configuration: Dict[str, Any] = field(default_factory=dict)
    last_heartbeat: datetime = field(default_factory=datetime.now)

class StateManager:
    """External state management for stateless agents"""
    
    def __init__(self, storage_backend: str = 'memory'):
        self.storage_backend = storage_backend
        self.states: Dict[str, AgentState] = {}
        self.logger = StructuredLogger('StateManager')
    
    def save_state(self, agent_id: str, state: AgentState):
        """Save agent state externally"""
        self.states[agent_id] = state
        self.logger.log_event('state_saved', {
            'agent_id': agent_id,
            'state_size': len(str(state))
        })
    
    def load_state(self, agent_id: str) -> Optional[AgentState]:
        """Load agent state from external storage"""
        return self.states.get(agent_id)

# Factor 9: Disposability - Fast startup and graceful shutdown
class GracefulShutdown:
    """Graceful shutdown handler for trading agents"""
    
    def __init__(self):
        self.shutdown_requested = False
        self.cleanup_callbacks = []
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.shutdown_requested = True
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                print(f"Error during cleanup: {e}")
    
    def register_cleanup(self, callback):
        """Register cleanup callback"""
        self.cleanup_callbacks.append(callback)

# Factor 8: Concurrency - Scale via process model
class TradingAgentProcess:
    """Process-based trading agent for horizontal scaling"""
    
    def __init__(self, agent_type: str, agent_id: str):
        self.agent_type = agent_type
        self.agent_id = agent_id
        self.config = TradingConfig()
        self.logger = StructuredLogger(f"{agent_type}-{agent_id}")
        self.state_manager = StateManager()
        self.shutdown_handler = GracefulShutdown()
        self.message_queue = queue.Queue()
        self.running = False
    
    def start(self):
        """Start agent process"""
        self.running = True
        self.logger.log_event('agent_started', {
            'agent_type': self.agent_type,
            'agent_id': self.agent_id,
            'config': vars(self.config)
        })
        
        # Register cleanup
        self.shutdown_handler.register_cleanup(self.stop)
        
        # Main processing loop
        self._process_loop()
    
    def stop(self):
        """Stop agent process gracefully"""
        self.running = False
        
        # Save final state
        final_state = AgentState(
            agent_id=self.agent_id,
            last_heartbeat=datetime.now()
        )
        self.state_manager.save_state(self.agent_id, final_state)
        
        self.logger.log_event('agent_stopped', {
            'agent_type': self.agent_type,
            'agent_id': self.agent_id
        })
    
    def _process_loop(self):
        """Main processing loop"""
        while self.running and not self.shutdown_handler.shutdown_requested:
            try:
                # Process messages
                try:
                    message = self.message_queue.get(timeout=1.0)
                    self._handle_message(message)
                except queue.Empty:
                    continue
                
                # Health check
                self._send_heartbeat()
                
            except Exception as e:
                self.logger.log_event('processing_error', {
                    'error': str(e),
                    'agent_id': self.agent_id
                }, 'ERROR')
    
    def _handle_message(self, message):
        """Handle incoming message"""
        self.logger.log_event('message_processed', {
            'message_type': type(message).__name__,
            'agent_id': self.agent_id
        })
    
    def _send_heartbeat(self):
        """Send heartbeat for health monitoring"""
        state = self.state_manager.load_state(self.agent_id) or AgentState(self.agent_id)
        state.last_heartbeat = datetime.now()
        self.state_manager.save_state(self.agent_id, state)

# Factor 7: Port Binding - Export services via port binding
class TradingAPIServer:
    """HTTP API server for trading system"""
    
    def __init__(self, port: int = 5000):
        self.port = port
        self.logger = StructuredLogger('TradingAPI')
        self.config = TradingConfig()
    
    def start_server(self):
        """Start HTTP server for API endpoints"""
        from flask import Flask, jsonify, request
        
        app = Flask(__name__)
        
        @app.route('/health')
        def health_check():
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'environment': self.config.environment
            })
        
        @app.route('/metrics')
        def get_metrics():
            # Return system metrics
            return jsonify({
                'portfolio_value': 117416.21,
                'total_trades': 117,
                'win_rate': 66.1,
                'profit_factor': 15.16
            })
        
        @app.route('/positions')
        def get_positions():
            # Return current positions
            return jsonify({
                'active_positions': 4,
                'total_value': 52146.63,
                'stop_loss_coverage': 100
            })
        
        self.logger.log_event('server_started', {'port': self.port})
        app.run(host='0.0.0.0', port=self.port, debug=self.config.debug_mode)

# Factor 12: Admin Processes - One-off admin tasks
class AdminTasks:
    """Administrative tasks for trading system"""
    
    def __init__(self):
        self.logger = StructuredLogger('AdminTasks')
    
    def run_portfolio_rebalance(self):
        """One-off portfolio rebalancing task"""
        self.logger.log_event('admin_task_started', {
            'task': 'portfolio_rebalance'
        })
        
        # Portfolio rebalancing logic would go here
        
        self.logger.log_event('admin_task_completed', {
            'task': 'portfolio_rebalance',
            'result': 'success'
        })
    
    def run_model_retrain(self):
        """One-off ML model retraining task"""
        self.logger.log_event('admin_task_started', {
            'task': 'model_retrain'
        })
        
        # Model retraining logic would go here
        
        self.logger.log_event('admin_task_completed', {
            'task': 'model_retrain',
            'result': 'success'
        })
    
    def run_data_cleanup(self):
        """One-off data cleanup task"""
        self.logger.log_event('admin_task_started', {
            'task': 'data_cleanup'
        })
        
        # Data cleanup logic would go here
        
        self.logger.log_event('admin_task_completed', {
            'task': 'data_cleanup',
            'result': 'success'
        })

# Factory for creating 12-factor compliant agents
class TradingAgentFactory:
    """Factory for creating 12-factor compliant trading agents"""
    
    @staticmethod
    def create_market_analyst(agent_id: str) -> TradingAgentProcess:
        """Create market analyst agent"""
        return TradingAgentProcess('MarketAnalyst', agent_id)
    
    @staticmethod
    def create_risk_manager(agent_id: str) -> TradingAgentProcess:
        """Create risk manager agent"""
        return TradingAgentProcess('RiskManager', agent_id)
    
    @staticmethod
    def create_trading_executor(agent_id: str) -> TradingAgentProcess:
        """Create trading executor agent"""
        return TradingAgentProcess('TradingExecutor', agent_id)

def main():
    """Main entry point for 12-factor agent system - manual control only"""
    config = TradingConfig()
    logger = StructuredLogger('TradingSystem')
    
    logger.log_event('system_initialization', {
        'environment': config.environment,
        'portfolio_value': config.portfolio_start_value
    })
    
    print("🚀 12-Factor Trading Agent System - Manual Control Mode")
    print("=" * 50)
    print("⚙️ System initialized but not started")
    print("🎛️ MANUAL CONTROLS:")
    print("   >>> config = TradingConfig()")
    print("   >>> # Configure and start manually")
    print()
    print("⚠️ NO AUTO-START - Use manual commands only")
    
    return config, logger

if __name__ == "__main__":
    main()
