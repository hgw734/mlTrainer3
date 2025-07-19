"""
Sentiment analysis module for social media and news sentiment tracking.
Analyzes sentiment from Reddit, Twitter, news articles, and financial forums.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
import re
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Social sentiment analyzer for institutional-grade sentiment tracking
    from Reddit, news sources, and social media platforms.
    """
    
    def __init__(self):
        """Initialize sentiment analyzer"""
        self.sentiment_sources = ['reddit', 'news', 'twitter', 'fintwit']
        self.sentiment_weights = {
            'reddit': 0.3,
            'news': 0.4,
            'twitter': 0.2,
            'fintwit': 0.1
        }
        
        # Sentiment keywords for basic analysis
        self.positive_keywords = [
            'bullish', 'buy', 'moon', 'rocket', 'pump', 'breakout', 'strong',
            'good', 'great', 'excellent', 'positive', 'up', 'gain', 'profit',
            'bull', 'surge', 'rally', 'momentum', 'upgrade', 'beat'
        ]
        
        self.negative_keywords = [
            'bearish', 'sell', 'crash', 'dump', 'weak', 'bad', 'terrible',
            'negative', 'down', 'loss', 'bear', 'decline', 'drop', 'fall',
            'downgrade', 'miss', 'disappointing', 'concern', 'risk'
        ]
    
    def analyze(self, symbol: str, news_data: Optional[List[Dict]] = None,
                social_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perform comprehensive sentiment analysis
        
        Args:
            symbol: Stock symbol
            news_data: List of news articles
            social_data: Social media sentiment data
            
        Returns:
            Dictionary of sentiment scores and metrics
        """
        try:
            results = {
                'symbol': symbol,
                'sentiment_score': 50.0,
                'news_sentiment': 50.0,
                'social_sentiment': 50.0,
                'sentiment_momentum': 0.0,
                'mention_volume': 0,
                'sentiment_strength': 'NEUTRAL'
            }
            
            # News sentiment analysis
            if news_data:
                news_results = self._analyze_news_sentiment(news_data)
                results.update(news_results)
            
            # Social media sentiment analysis
            if social_data:
                social_results = self._analyze_social_sentiment(social_data)
                results.update(social_results)
            
            # Calculate composite sentiment score
            results['sentiment_score'] = self._calculate_composite_sentiment(results)
            
            # Determine sentiment strength
            results['sentiment_strength'] = self._determine_sentiment_strength(results['sentiment_score'])
            
            return results
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed for {symbol}: {e}")
            return self._empty_result(symbol)
    
    def _analyze_news_sentiment(self, news_data: List[Dict]) -> Dict[str, Any]:
        """Analyze sentiment from news articles"""
        try:
            results = {
                'news_sentiment': 50.0,
                'news_volume': 0,
                'news_sentiment_trend': 0.0
            }
            
            if not news_data:
                return results
            
            sentiments = []
            recent_sentiments = []
            
            for article in news_data:
                # Basic sentiment scoring based on keywords
                sentiment_score = self._score_text_sentiment(
                    article.get('title', '') + ' ' + article.get('description', '')
                )
                sentiments.append(sentiment_score)
                
                # Track recent sentiment (last 24 hours)
                pub_date = article.get('published_utc', '')
                if self._is_recent_article(pub_date):
                    recent_sentiments.append(sentiment_score)
            
            if sentiments:
                results['news_sentiment'] = np.mean(sentiments)
                results['news_volume'] = len(sentiments)
                
                # Calculate sentiment trend (recent vs all)
                if recent_sentiments and len(sentiments) > len(recent_sentiments):
                    recent_avg = np.mean(recent_sentiments)
                    overall_avg = np.mean(sentiments)
                    results['news_sentiment_trend'] = recent_avg - overall_avg
            
            return results
            
        except Exception as e:
            logger.error(f"News sentiment analysis failed: {e}")
            return {'news_sentiment': 50.0}
    
    def _analyze_social_sentiment(self, social_data: Dict) -> Dict[str, Any]:
        """Analyze sentiment from social media"""
        try:
            results = {
                'social_sentiment': 50.0,
                'reddit_sentiment': 50.0,
                'twitter_sentiment': 50.0,
                'mention_volume': 0,
                'social_momentum': 0.0
            }
            
            total_sentiment = 0
            total_weight = 0
            
            # Reddit sentiment
            if 'reddit' in social_data:
                reddit_data = social_data['reddit']
                reddit_sentiment = self._analyze_reddit_sentiment(reddit_data)
                results['reddit_sentiment'] = reddit_sentiment
                total_sentiment += reddit_sentiment * self.sentiment_weights['reddit']
                total_weight += self.sentiment_weights['reddit']
                
                # Count mentions
                if 'posts' in reddit_data:
                    results['mention_volume'] += len(reddit_data['posts'])
            
            # Twitter sentiment
            if 'twitter' in social_data:
                twitter_data = social_data['twitter']
                twitter_sentiment = self._analyze_twitter_sentiment(twitter_data)
                results['twitter_sentiment'] = twitter_sentiment
                total_sentiment += twitter_sentiment * self.sentiment_weights['twitter']
                total_weight += self.sentiment_weights['twitter']
                
                # Count mentions
                if 'tweets' in twitter_data:
                    results['mention_volume'] += len(twitter_data['tweets'])
            
            # Calculate weighted social sentiment
            if total_weight > 0:
                results['social_sentiment'] = total_sentiment / total_weight
            
            return results
            
        except Exception as e:
            logger.error(f"Social sentiment analysis failed: {e}")
            return {'social_sentiment': 50.0}
    
    def _analyze_reddit_sentiment(self, reddit_data: Dict) -> float:
        """Analyze Reddit sentiment from posts and comments"""
        try:
            sentiments = []
            
            if 'posts' in reddit_data:
                for post in reddit_data['posts']:
                    # Analyze post title and content
                    text = post.get('title', '') + ' ' + post.get('selftext', '')
                    sentiment = self._score_text_sentiment(text)
                    
                    # Weight by upvotes/score
                    score = post.get('score', 1)
                    weight = max(1, np.log(score + 1))  # Log scale for score weighting
                    
                    # Add weighted sentiment
                    for _ in range(int(weight)):
                        sentiments.append(sentiment)
            
            return np.mean(sentiments) if sentiments else 50.0
            
        except Exception as e:
            logger.error(f"Reddit sentiment analysis failed: {e}")
            return 50.0
    
    def _analyze_twitter_sentiment(self, twitter_data: Dict) -> float:
        """Analyze Twitter sentiment from tweets"""
        try:
            sentiments = []
            
            if 'tweets' in twitter_data:
                for tweet in twitter_data['tweets']:
                    # Analyze tweet text
                    text = tweet.get('text', '')
                    sentiment = self._score_text_sentiment(text)
                    
                    # Weight by engagement (likes + retweets)
                    likes = tweet.get('public_metrics', {}).get('like_count', 0)
                    retweets = tweet.get('public_metrics', {}).get('retweet_count', 0)
                    engagement = likes + retweets
                    weight = max(1, np.log(engagement + 1))
                    
                    # Add weighted sentiment
                    for _ in range(int(weight)):
                        sentiments.append(sentiment)
            
            return np.mean(sentiments) if sentiments else 50.0
            
        except Exception as e:
            logger.error(f"Twitter sentiment analysis failed: {e}")
            return 50.0
    
    def _score_text_sentiment(self, text: str) -> float:
        """Score text sentiment using keyword analysis"""
        try:
            if not text:
                return 50.0
            
            # Clean and normalize text
            text = text.lower()
            text = re.sub(r'[^\w\s]', ' ', text)
            words = text.split()
            
            # Count positive and negative keywords
            positive_count = sum(1 for word in words if word in self.positive_keywords)
            negative_count = sum(1 for word in words if word in self.negative_keywords)
            
            # Calculate sentiment score (0-100 scale)
            total_sentiment_words = positive_count + negative_count
            
            if total_sentiment_words == 0:
                return 50.0  # Neutral
            
            # Calculate positive ratio and scale to 0-100
            positive_ratio = positive_count / total_sentiment_words
            sentiment_score = 50 + (positive_ratio - 0.5) * 100
            
            # Apply word count weighting (more sentiment words = more confident score)
            confidence = min(1.0, total_sentiment_words / 5.0)  # Max confidence at 5+ sentiment words
            sentiment_score = 50 + (sentiment_score - 50) * confidence
            
            return max(0, min(100, sentiment_score))
            
        except Exception as e:
            logger.error(f"Text sentiment scoring failed: {e}")
            return 50.0
    
    def _is_recent_article(self, pub_date: str, hours: int = 24) -> bool:
        """Check if article is recent (within specified hours)"""
        try:
            if not pub_date:
                return False
            
            # Parse date string (assuming ISO format)
            article_date = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
            cutoff_date = datetime.now() - timedelta(hours=hours)
            
            return article_date >= cutoff_date
            
        except Exception as e:
            logger.error(f"Date parsing failed: {e}")
            return False
    
    def _calculate_composite_sentiment(self, results: Dict[str, Any]) -> float:
        """Calculate weighted composite sentiment score"""
        try:
            news_sentiment = results.get('news_sentiment', 50.0)
            social_sentiment = results.get('social_sentiment', 50.0)
            
            # Weight news more heavily than social media
            composite = (
                news_sentiment * 0.6 +
                social_sentiment * 0.4
            )
            
            # Apply volume boost for high-mention stocks
            mention_volume = results.get('mention_volume', 0)
            if mention_volume > 100:  # High mention volume
                volume_boost = min(5, mention_volume / 100)  # Max 5 point boost
                if composite > 50:
                    composite += volume_boost
                else:
                    composite -= volume_boost
            
            return max(0, min(100, composite))
            
        except Exception as e:
            logger.error(f"Composite sentiment calculation failed: {e}")
            return 50.0
    
    def _determine_sentiment_strength(self, sentiment_score: float) -> str:
        """Determine sentiment strength category"""
        if sentiment_score >= 80:
            return 'VERY_BULLISH'
        elif sentiment_score >= 65:
            return 'BULLISH'
        elif sentiment_score >= 55:
            return 'SLIGHTLY_BULLISH'
        elif sentiment_score >= 45:
            return 'NEUTRAL'
        elif sentiment_score >= 35:
            return 'SLIGHTLY_BEARISH'
        elif sentiment_score >= 20:
            return 'BEARISH'
        else:
            return 'VERY_BEARISH'
    
    def get_sentiment_summary(self, sentiment_score: float) -> str:
        """Get human-readable sentiment summary"""
        strength = self._determine_sentiment_strength(sentiment_score)
        
        summaries = {
            'VERY_BULLISH': 'Extremely positive sentiment with strong buy signals',
            'BULLISH': 'Positive sentiment with bullish indicators',
            'SLIGHTLY_BULLISH': 'Mildly positive sentiment',
            'NEUTRAL': 'Mixed or neutral sentiment',
            'SLIGHTLY_BEARISH': 'Mildly negative sentiment',
            'BEARISH': 'Negative sentiment with bearish indicators',
            'VERY_BEARISH': 'Extremely negative sentiment with strong sell signals'
        }
        
        return summaries.get(strength, 'Neutral sentiment')
    
    def _empty_result(self, symbol: str) -> Dict[str, Any]:
        """Return empty result structure"""
        return {
            'symbol': symbol,
            'sentiment_score': 50.0,
            'news_sentiment': 50.0,
            'social_sentiment': 50.0,
            'reddit_sentiment': 50.0,
            'twitter_sentiment': 50.0,
            'sentiment_momentum': 0.0,
            'mention_volume': 0,
            'news_volume': 0,
            'sentiment_strength': 'NEUTRAL'
        }