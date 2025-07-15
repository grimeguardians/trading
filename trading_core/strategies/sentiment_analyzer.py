"""Streamlined sentiment analysis"""

import random
import numpy as np
from datetime import datetime
from typing import List, Dict

from ..data_models import SentimentData

class SentimentAnalyzer:
    """Streamlined sentiment analysis engine"""

    def __init__(self):
        self.sentiment_history = {}
        self.news_keywords = {
            'positive': ['growth', 'profit', 'increase', 'beat', 'strong', 'bullish', 'upgrade', 'buy'],
            'negative': ['loss', 'decline', 'fall', 'miss', 'weak', 'bearish', 'downgrade', 'sell']
        }

    def analyze_sentiment(self, symbol: str) -> SentimentData:
        """Analyze sentiment for a symbol"""
        try:
            # Simulate news and social data
            news_articles = self._simulate_news_data(symbol)
            social_posts = self._simulate_social_data(symbol)

            # Calculate sentiment score
            sentiment_score = self._calculate_sentiment_score(news_articles + social_posts)

            # Determine sentiment label
            if sentiment_score > 0.1:
                sentiment_label = 'BULLISH'
            elif sentiment_score < -0.1:
                sentiment_label = 'BEARISH'
            else:
                sentiment_label = 'NEUTRAL'

            # Calculate sentiment trend
            sentiment_trend = self._calculate_sentiment_trend(symbol, sentiment_score)

            sentiment_data = SentimentData(
                symbol=symbol,
                sentiment_score=sentiment_score,
                sentiment_label=sentiment_label,
                news_count=len(news_articles),
                social_mentions=len(social_posts),
                timestamp=datetime.now(),
                sentiment_trend=sentiment_trend
            )

            # Store in history
            if symbol not in self.sentiment_history:
                self.sentiment_history[symbol] = []
            self.sentiment_history[symbol].append(sentiment_data)

            # Keep only recent history
            if len(self.sentiment_history[symbol]) > 100:
                self.sentiment_history[symbol] = self.sentiment_history[symbol][-100:]

            return sentiment_data

        except Exception as e:
            print(f"Error analyzing sentiment for {symbol}: {e}")
            return SentimentData(symbol, 0.0, 'NEUTRAL', 0, 0, datetime.now(), 'STABLE')

    def _calculate_sentiment_trend(self, symbol: str, current_sentiment: float) -> str:
        """Calculate sentiment trend"""
        if symbol not in self.sentiment_history or len(self.sentiment_history[symbol]) < 3:
            return 'STABLE'

        recent_sentiments = [data.sentiment_score for data in self.sentiment_history[symbol][-3:]]
        recent_sentiments.append(current_sentiment)

        if len(recent_sentiments) >= 2:
            trend_slope = np.polyfit(range(len(recent_sentiments)), recent_sentiments, 1)[0]
            
            if trend_slope > 0.1:
                return 'IMPROVING'
            elif trend_slope < -0.1:
                return 'DECLINING'
                
        return 'STABLE'

    def _simulate_news_data(self, symbol: str) -> List[str]:
        """Simulate news data"""
        news_templates = [
            f"{symbol} reports strong quarterly earnings",
            f"{symbol} announces major product launch",
            f"Analysts upgrade {symbol} price target",
            f"{symbol} faces regulatory challenges",
            f"{symbol} shows high volatility",
            f"Market experts divided on {symbol}",
            f"{symbol} CEO announces partnership",
            f"{symbol} invests in R&D"
        ]

        num_articles = random.randint(0, 4)
        return random.sample(news_templates, min(num_articles, len(news_templates)))

    def _simulate_social_data(self, symbol: str) -> List[str]:
        """Simulate social media data"""
        social_templates = [
            f"Bullish on {symbol} long term",
            f"{symbol} to the moon! Strong setup",
            f"Taking profits on {symbol}",
            f"{symbol} looks overvalued",
            f"Great buying opportunity for {symbol}",
            f"{symbol} earnings disappointing",
            f"Love {symbol} innovation",
            f"Strong support for {symbol}"
        ]

        num_posts = random.randint(0, 6)
        return random.sample(social_templates, min(num_posts, len(social_templates)))

    def _calculate_sentiment_score(self, text_data: List[str]) -> float:
        """Calculate sentiment score from text data"""
        if not text_data:
            return 0.0

        total_score = 0.0
        total_weight = 0.0

        for text in text_data:
            text_lower = text.lower()
            positive_score = sum(1 for word in self.news_keywords['positive'] if word in text_lower)
            negative_score = sum(1 for word in self.news_keywords['negative'] if word in text_lower)

            total_mentions = positive_score + negative_score
            if total_mentions > 0:
                text_sentiment = (positive_score - negative_score) / total_mentions
                weight = min(total_mentions, 3)
                total_score += text_sentiment * weight
                total_weight += weight

        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            final_score = 0.0

        return max(-1.0, min(1.0, final_score))