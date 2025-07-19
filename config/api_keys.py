"""
API Key Management System
Handles external API keys for various data providers
"""

import os
import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class APIKeyManager:
    """Manages API keys for various data providers"""
    
    def __init__(self):
        self.providers = {
            'alpha_vantage': {
                'env_var': 'ALPHA_VANTAGE_API_KEY',
                'name': 'Alpha Vantage',
                'description': 'Stock market data with free tier'
            },
            'finnhub': {
                'env_var': 'FINNHUB_API_KEY',
                'name': 'Finnhub',
                'description': 'Real-time financial data'
            },
            'polygon': {
                'env_var': 'POLYGON_API_KEY',
                'name': 'Polygon.io',
                'description': 'Professional market data'
            },
            'openai': {
                'env_var': 'OPENAI_API_KEY',
                'name': 'OpenAI',
                'description': 'AI-powered conversations'
            }
        }
    
    def get_status(self) -> Dict[str, bool]:
        """Get status of all API keys"""
        status = {}
        for provider_key, provider_info in self.providers.items():
            env_var = provider_info['env_var']
            status[provider_key] = bool(os.getenv(env_var))
        return status
    
    def get_available_providers(self) -> List[str]:
        """Get list of providers with valid API keys"""
        available = []
        for provider_key, provider_info in self.providers.items():
            env_var = provider_info['env_var']
            if os.getenv(env_var):
                available.append(provider_key)
        return available
    
    def get_provider_info(self, provider_key: str) -> Optional[Dict]:
        """Get information about a specific provider"""
        return self.providers.get(provider_key)
    
    def check_key_validity(self, provider_key: str) -> bool:
        """Check if a specific API key is valid and present"""
        if provider_key not in self.providers:
            return False
        
        env_var = self.providers[provider_key]['env_var']
        key = os.getenv(env_var)
        
        if not key:
            return False
        
        # Basic validation - check if key is not empty and has minimum length
        return len(key.strip()) > 10

# Global instance
api_key_manager = APIKeyManager()