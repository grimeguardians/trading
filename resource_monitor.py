
#!/usr/bin/env python3
"""
Resource Monitor for Trading System
Monitors memory and CPU usage to prevent crashes
"""

import psutil
import time
import logging
from typing import Dict, Any
import threading

class ResourceMonitor:
    """Monitor system resources and prevent crashes"""
    
    def __init__(self):
        self.logger = logging.getLogger("ResourceMonitor")
        self.monitoring = False
        self.max_memory_mb = 1000  # 1GB limit
        self.max_cpu_percent = 80
        self.alerts = []
        
    def start_monitoring(self):
        """Start resource monitoring"""
        self.monitoring = True
        monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        monitor_thread.start()
        self.logger.info("Resource monitoring started")
        
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        self.logger.info("Resource monitoring stopped")
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # Get current resource usage
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                cpu_percent = process.cpu_percent()
                
                # Check memory usage
                if memory_mb > self.max_memory_mb:
                    self._handle_high_memory(memory_mb)
                    
                # Check CPU usage
                if cpu_percent > self.max_cpu_percent:
                    self._handle_high_cpu(cpu_percent)
                    
                # Log every 30 seconds
                if int(time.time()) % 30 == 0:
                    self.logger.info(f"Resources: {memory_mb:.1f}MB RAM, {cpu_percent:.1f}% CPU")
                    
            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {e}")
                
            time.sleep(5)  # Check every 5 seconds
            
    def _handle_high_memory(self, memory_mb: float):
        """Handle high memory usage"""
        alert = f"High memory usage: {memory_mb:.1f}MB"
        self.alerts.append(alert)
        self.logger.warning(alert)
        
        # Force garbage collection
        import gc
        gc.collect()
        
    def _handle_high_cpu(self, cpu_percent: float):
        """Handle high CPU usage"""
        alert = f"High CPU usage: {cpu_percent:.1f}%"
        self.alerts.append(alert)
        self.logger.warning(alert)
        
        # Add small delay to reduce CPU pressure
        time.sleep(0.1)
        
    def get_status(self) -> Dict[str, Any]:
        """Get current resource status"""
        try:
            process = psutil.Process()
            return {
                'memory_mb': process.memory_info().rss / 1024 / 1024,
                'cpu_percent': process.cpu_percent(),
                'alerts': self.alerts[-5:],  # Last 5 alerts
                'monitoring': self.monitoring
            }
        except Exception as e:
            return {'error': str(e)}
