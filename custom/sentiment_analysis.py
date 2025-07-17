"""
Sentiment Analysis Models Implementation

Analyzes market sentiment from news, social media, options flow, and volatility.
All models require real data sources - no synthetic sentiment generation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd


@dataclass
class SentimentSignal:
    """Base signal for sentiment analysis."""
    timestamp: datetime
    signal_type: str  # 'bullish', 'bearish', 'neutral'
    strength: float  # 0.0 to 1.0
    sentiment_score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    metrics: Dict[str, float]
    metadata: Dict[str, any]


@dataclass
class NewsSentiment(SentimentSignal):
    """News sentiment specific signals."""
    headline_scores: List[Dict[str, float]]
    entities_mentioned: Dict[str, int]
    event_impact: str  # 'high', 'medium', 'low'
    sentiment_trend: str  # 'improving', 'deteriorating', 'stable'
    key_topics: List[str]


@dataclass
class SocialSentiment(SentimentSignal):
    """Social media sentiment signals."""
    mention_volume: int
    sentiment_distribution: Dict[str, float]  # positive/negative/neutral %
    trending_topics: List[str]
    crowd_consensus: float  # -1.0 to 1.0
    influencer_sentiment: float


@dataclass
class OptionsFlowSignal(SentimentSignal):
    """Options flow sentiment signals."""
    put_call_ratio: float
    unusual_activity: List[Dict[str, any]]
    smart_money_direction: str  # 'bullish', 'bearish', 'neutral'
    gamma_exposure: float
    implied_move: float


@dataclass
class VIXRegimeSignal(SentimentSignal):
    """VIX regime analysis signals."""
    vix_level: float
    term_structure: str  # 'contango', 'backwardation'
    mean_reversion_signal: float
    risk_regime: str  # 'risk-on', 'risk-off', 'transitioning'
    volatility_percentile: float


class BaseSentimentModel(ABC):
    """Base class for sentiment analysis models."""
    
    def __init__(self, min_data_points: int = 10):
        self.min_data_points = min_data_points
    
    @abstractmethod
    def analyze(self, data: any) -> SentimentSignal:
        """Analyze sentiment data."""
        pass
    
    def validate_data(self, data: any) -> bool:
        """Validate input data."""
        if data is None:
            return False
        if isinstance(data, pd.DataFrame):
            return len(data) >= self.min_data_points
        elif isinstance(data, list):
            return len(data) >= self.min_data_points
        elif isinstance(data, dict):
            return len(data) > 0
        return False


class NewsSentimentModel(BaseSentimentModel):
    """
    News sentiment analysis using NLP.
    
    Analyzes:
    - Headline sentiment scoring
    - Named entity extraction
    - Sentiment trend tracking
    - Market-moving event detection
    
    Requires real news feed data with headlines and content.
    """
    
    def __init__(self, sentiment_window: int = 20,
                 entity_threshold: int = 3,
                 event_keywords: Optional[List[str]] = None):
        super().__init__()
        self.sentiment_window = sentiment_window
        self.entity_threshold = entity_threshold
        self.event_keywords = event_keywords or [
            'earnings', 'merger', 'acquisition', 'bankruptcy', 'scandal',
            'investigation', 'recall', 'lawsuit', 'dividend', 'buyback',
            'guidance', 'upgrade', 'downgrade', 'fda', 'sec'
        ]
    
    def analyze(self, news_feed: List[Dict]) -> NewsSentiment:
        """
        Analyze news sentiment from feed.
        
        Expected format: List of dicts with 'headline', 'content', 'timestamp', 'source'
        """
        if not self.validate_data(news_feed):
            return self._default_signal()
        
        try:
            # Score individual headlines
            headline_scores = self._score_headlines(news_feed)
            
            # Extract entities mentioned
            entities_mentioned = self._extract_entities(news_feed)
            
            # Detect market-moving events
            event_impact = self._assess_event_impact(news_feed)
            
            # Calculate sentiment trend
            sentiment_trend = self._calculate_sentiment_trend(headline_scores)
            
            # Extract key topics
            key_topics = self._extract_key_topics(news_feed)
            
            # Calculate overall sentiment
            if headline_scores:
                avg_sentiment = np.mean([score['sentiment'] for score in headline_scores])
                sentiment_score = avg_sentiment
                
                # Weight by recency
                recent_scores = headline_scores[-5:] if len(headline_scores) >= 5 else headline_scores
                recent_sentiment = np.mean([score['sentiment'] for score in recent_scores])
                
                # Determine signal
                if recent_sentiment > 0.3:
                    signal_type = 'bullish'
                    strength = min(recent_sentiment, 1.0)
                elif recent_sentiment < -0.3:
                    signal_type = 'bearish'
                    strength = min(abs(recent_sentiment), 1.0)
                else:
                    signal_type = 'neutral'
                    strength = 0.5
                
                # Adjust for event impact
                if event_impact == 'high':
                    strength = min(strength * 1.3, 1.0)
                
                confidence = self._calculate_confidence(headline_scores, entities_mentioned)
            else:
                sentiment_score = 0.0
                signal_type = 'neutral'
                strength = 0.5
                confidence = 0.0
            
            return NewsSentiment(
                timestamp=datetime.now(),
                signal_type=signal_type,
                strength=strength,
                sentiment_score=sentiment_score,
                confidence=confidence,
                headline_scores=headline_scores,
                entities_mentioned=entities_mentioned,
                event_impact=event_impact,
                sentiment_trend=sentiment_trend,
                key_topics=key_topics,
                metrics={
                    'article_count': len(news_feed),
                    'avg_sentiment': sentiment_score,
                    'sentiment_std': np.std([s['sentiment'] for s in headline_scores]) if headline_scores else 0,
                    'entity_count': len(entities_mentioned)
                },
                metadata={
                    'sources': list(set(article.get('source', 'unknown') for article in news_feed)),
                    'time_range': self._get_time_range(news_feed)
                }
            )
            
        except Exception as e:
            return self._default_signal()
    
    def _score_headlines(self, news_feed: List[Dict]) -> List[Dict[str, float]]:
        """Score sentiment of headlines using keyword analysis."""
        scores = []
        
        # Positive and negative word lists (simplified - real implementation would use NLP)
        positive_words = [
            'surge', 'soar', 'jump', 'rally', 'gain', 'rise', 'boost', 'upgrade',
            'beat', 'exceed', 'record', 'breakthrough', 'innovation', 'growth'
        ]
        negative_words = [
            'plunge', 'crash', 'fall', 'drop', 'decline', 'loss', 'cut', 'downgrade',
            'miss', 'below', 'concern', 'risk', 'threat', 'weak'
        ]
        
        for article in news_feed:
            headline = article.get('headline', '').lower()
            content = article.get('content', '').lower()
            
            # Count positive/negative words
            pos_count = sum(1 for word in positive_words if word in headline)
            neg_count = sum(1 for word in negative_words if word in headline)
            
            # Add content analysis
            if content:
                pos_count += sum(0.5 for word in positive_words if word in content) / 10
                neg_count += sum(0.5 for word in negative_words if word in content) / 10
            
            # Calculate sentiment score
            total_count = pos_count + neg_count
            if total_count > 0:
                sentiment = (pos_count - neg_count) / total_count
            else:
                sentiment = 0.0
            
            scores.append({
                'headline': article.get('headline', ''),
                'sentiment': sentiment,
                'timestamp': article.get('timestamp', datetime.now()),
                'impact': pos_count + neg_count  # Total keyword count as impact proxy
            })
        
        return sorted(scores, key=lambda x: x['timestamp'])
    
    def _extract_entities(self, news_feed: List[Dict]) -> Dict[str, int]:
        """Extract and count named entities (companies, people, etc.)."""
        entities = {}
        
        # Common company indicators (simplified)
        company_indicators = ['Inc.', 'Corp.', 'LLC', 'Ltd.', 'Company', 'Group']
        
        for article in news_feed:
            text = article.get('headline', '') + ' ' + article.get('content', '')
            
            # Simple entity extraction (real implementation would use NER)
            words = text.split()
            for i, word in enumerate(words):
                if word in company_indicators and i > 0:
                    entity = words[i-1] + ' ' + word
                    entities[entity] = entities.get(entity, 0) + 1
                
                # Look for capitalized sequences (potential entities)
                if word[0].isupper() and len(word) > 1:
                    if i > 0 and words[i-1][0].isupper():
                        entity = words[i-1] + ' ' + word
                        entities[entity] = entities.get(entity, 0) + 1
        
        # Filter entities mentioned at least threshold times
        return {k: v for k, v in entities.items() if v >= self.entity_threshold}
    
    def _assess_event_impact(self, news_feed: List[Dict]) -> str:
        """Assess the impact level of news events."""
        high_impact_count = 0
        medium_impact_count = 0
        
        for article in news_feed:
            headline = article.get('headline', '').lower()
            content = article.get('content', '').lower()
            full_text = headline + ' ' + content
            
            # Check for high-impact keywords
            high_impact_found = any(keyword in full_text for keyword in [
                'bankruptcy', 'scandal', 'investigation', 'sec', 'fraud',
                'merger', 'acquisition', 'takeover'
            ])
            
            # Check for medium-impact keywords
            medium_impact_found = any(keyword in full_text for keyword in [
                'earnings', 'guidance', 'upgrade', 'downgrade', 'dividend'
            ])
            
            if high_impact_found:
                high_impact_count += 1
            elif medium_impact_found:
                medium_impact_count += 1
        
        if high_impact_count > 0:
            return 'high'
        elif medium_impact_count >= 2:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_sentiment_trend(self, headline_scores: List[Dict[str, float]]) -> str:
        """Calculate if sentiment is improving or deteriorating."""
        if len(headline_scores) < self.sentiment_window:
            return 'stable'
        
        # Compare recent vs older sentiment
        recent_scores = headline_scores[-self.sentiment_window//2:]
        older_scores = headline_scores[-self.sentiment_window:-self.sentiment_window//2]
        
        recent_avg = np.mean([s['sentiment'] for s in recent_scores])
        older_avg = np.mean([s['sentiment'] for s in older_scores])
        
        if recent_avg > older_avg + 0.1:
            return 'improving'
        elif recent_avg < older_avg - 0.1:
            return 'deteriorating'
        else:
            return 'stable'
    
    def _extract_key_topics(self, news_feed: List[Dict]) -> List[str]:
        """Extract key topics from news."""
        topics = []
        
        # Count keyword occurrences
        keyword_counts = {}
        for keyword in self.event_keywords:
            count = sum(1 for article in news_feed 
                       if keyword in article.get('headline', '').lower() or 
                       keyword in article.get('content', '').lower())
            if count > 0:
                keyword_counts[keyword] = count
        
        # Return top 5 topics
        sorted_topics = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
        return [topic[0] for topic in sorted_topics[:5]]
    
    def _calculate_confidence(self, headline_scores: List[Dict[str, float]], 
                            entities: Dict[str, int]) -> float:
        """Calculate confidence in sentiment signal."""
        if not headline_scores:
            return 0.0
        
        # Base confidence on:
        # 1. Number of articles
        article_confidence = min(len(headline_scores) / 10, 1.0) * 0.3
        
        # 2. Sentiment consistency
        sentiments = [s['sentiment'] for s in headline_scores]
        if len(sentiments) > 1:
            consistency = 1 - np.std(sentiments)
            consistency_confidence = max(0, consistency) * 0.4
        else:
            consistency_confidence = 0.2
        
        # 3. Entity mentions
        entity_confidence = min(len(entities) / 5, 1.0) * 0.3
        
        return article_confidence + consistency_confidence + entity_confidence
    
    def _get_time_range(self, news_feed: List[Dict]) -> Tuple[datetime, datetime]:
        """Get time range of news articles."""
        timestamps = [article.get('timestamp', datetime.now()) for article in news_feed]
        return (min(timestamps), max(timestamps)) if timestamps else (datetime.now(), datetime.now())
    
    def _default_signal(self) -> NewsSentiment:
        """Return default neutral signal when analysis fails."""
        return NewsSentiment(
            timestamp=datetime.now(),
            signal_type='neutral',
            strength=0.5,
            sentiment_score=0.0,
            confidence=0.0,
            headline_scores=[],
            entities_mentioned={},
            event_impact='low',
            sentiment_trend='stable',
            key_topics=[],
            metrics={},
            metadata={'error': 'Insufficient or invalid data'}
        )


class SocialSentimentModel(BaseSentimentModel):
    """
    Social media sentiment analysis.
    
    Analyzes:
    - Mention volume and velocity
    - Sentiment distribution
    - Trending topics
    - Crowd psychology indicators
    
    Requires real social media data (Twitter, Reddit, etc.).
    """
    
    def __init__(self, volume_threshold: int = 100,
                 sentiment_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.volume_threshold = volume_threshold
        self.sentiment_weights = sentiment_weights or {
            'twitter': 0.4,
            'reddit': 0.3,
            'stocktwits': 0.3
        }
    
    def analyze(self, social_data: Dict) -> SocialSentiment:
        """
        Analyze social media sentiment.
        
        Expected format: Dict with platform data containing posts/mentions
        """
        if not self.validate_data(social_data):
            return self._default_signal()
        
        try:
            # Calculate mention volume
            mention_volume = self._calculate_mention_volume(social_data)
            
            # Analyze sentiment distribution
            sentiment_distribution = self._analyze_sentiment_distribution(social_data)
            
            # Identify trending topics
            trending_topics = self._identify_trending_topics(social_data)
            
            # Calculate crowd consensus
            crowd_consensus = self._calculate_crowd_consensus(sentiment_distribution)
            
            # Analyze influencer sentiment
            influencer_sentiment = self._analyze_influencer_sentiment(social_data)
            
            # Calculate weighted sentiment score
            sentiment_score = self._calculate_weighted_sentiment(social_data)
            
            # Determine signal
            if mention_volume > self.volume_threshold:
                if crowd_consensus > 0.3:
                    signal_type = 'bullish'
                    strength = min(0.6 + crowd_consensus * 0.4, 1.0)
                elif crowd_consensus < -0.3:
                    signal_type = 'bearish'
                    strength = min(0.6 + abs(crowd_consensus) * 0.4, 1.0)
                else:
                    signal_type = 'neutral'
                    strength = 0.5
                
                # Adjust for influencer sentiment
                if abs(influencer_sentiment) > 0.5:
                    if (influencer_sentiment > 0 and signal_type == 'bullish') or \
                       (influencer_sentiment < 0 and signal_type == 'bearish'):
                        strength = min(strength * 1.2, 1.0)
            else:
                signal_type = 'neutral'
                strength = 0.4  # Low volume = low confidence
            
            confidence = self._calculate_confidence(mention_volume, sentiment_distribution)
            
            return SocialSentiment(
                timestamp=datetime.now(),
                signal_type=signal_type,
                strength=strength,
                sentiment_score=sentiment_score,
                confidence=confidence,
                mention_volume=mention_volume,
                sentiment_distribution=sentiment_distribution,
                trending_topics=trending_topics,
                crowd_consensus=crowd_consensus,
                influencer_sentiment=influencer_sentiment,
                metrics={
                    'avg_engagement': self._calculate_avg_engagement(social_data),
                    'sentiment_volatility': self._calculate_sentiment_volatility(social_data),
                    'platform_count': len(social_data)
                },
                metadata={
                    'platforms': list(social_data.keys()),
                    'time_window': '24h'  # Assuming 24h data window
                }
            )
            
        except Exception as e:
            return self._default_signal()
    
    def _calculate_mention_volume(self, social_data: Dict) -> int:
        """Calculate total mention volume across platforms."""
        total_volume = 0
        
        for platform, data in social_data.items():
            if isinstance(data, list):
                total_volume += len(data)
            elif isinstance(data, dict) and 'posts' in data:
                total_volume += len(data['posts'])
            elif isinstance(data, dict) and 'count' in data:
                total_volume += data['count']
        
        return total_volume
    
    def _analyze_sentiment_distribution(self, social_data: Dict) -> Dict[str, float]:
        """Analyze distribution of positive/negative/neutral sentiment."""
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for platform, data in social_data.items():
            posts = []
            if isinstance(data, list):
                posts = data
            elif isinstance(data, dict) and 'posts' in data:
                posts = data['posts']
            
            for post in posts:
                # Simple sentiment analysis based on keywords
                text = post.get('text', '').lower() if isinstance(post, dict) else str(post).lower()
                
                # Positive indicators
                positive_words = ['bullish', 'moon', 'buy', 'long', 'calls', 'up', 'green', 'gain']
                negative_words = ['bearish', 'crash', 'sell', 'short', 'puts', 'down', 'red', 'loss']
                
                pos_score = sum(1 for word in positive_words if word in text)
                neg_score = sum(1 for word in negative_words if word in text)
                
                if pos_score > neg_score:
                    positive_count += 1
                elif neg_score > pos_score:
                    negative_count += 1
                else:
                    neutral_count += 1
        
        total = positive_count + negative_count + neutral_count
        if total == 0:
            return {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
        
        return {
            'positive': positive_count / total,
            'negative': negative_count / total,
            'neutral': neutral_count / total
        }
    
    def _identify_trending_topics(self, social_data: Dict) -> List[str]:
        """Identify trending topics/hashtags."""
        topic_counts = {}
        
        for platform, data in social_data.items():
            posts = []
            if isinstance(data, list):
                posts = data
            elif isinstance(data, dict) and 'posts' in data:
                posts = data['posts']
            
            for post in posts:
                # Extract hashtags and common topics
                text = post.get('text', '') if isinstance(post, dict) else str(post)
                
                # Find hashtags
                import re
                hashtags = re.findall(r'#\w+', text)
                for tag in hashtags:
                    topic_counts[tag] = topic_counts.get(tag, 0) + 1
                
                # Find ticker symbols
                tickers = re.findall(r'\$[A-Z]{1,5}', text)
                for ticker in tickers:
                    topic_counts[ticker] = topic_counts.get(ticker, 0) + 1
        
        # Return top 5 trending topics
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        return [topic[0] for topic in sorted_topics[:5]]
    
    def _calculate_crowd_consensus(self, sentiment_distribution: Dict[str, float]) -> float:
        """Calculate overall crowd consensus (-1 to 1)."""
        positive = sentiment_distribution.get('positive', 0)
        negative = sentiment_distribution.get('negative', 0)
        
        # Calculate consensus as difference between positive and negative
        consensus = positive - negative
        
        # Weight by how unanimous the sentiment is
        unanimity = max(positive, negative)
        if unanimity > 0.7:  # Strong consensus
            consensus *= 1.2
        
        return max(-1, min(1, consensus))
    
    def _analyze_influencer_sentiment(self, social_data: Dict) -> float:
        """Analyze sentiment from high-influence accounts."""
        influencer_sentiments = []
        
        for platform, data in social_data.items():
            posts = []
            if isinstance(data, list):
                posts = data
            elif isinstance(data, dict) and 'posts' in data:
                posts = data['posts']
            
            for post in posts:
                if isinstance(post, dict):
                    # Check follower count or engagement metrics
                    followers = post.get('user_followers', 0)
                    engagement = post.get('engagement', post.get('likes', 0) + post.get('retweets', 0))
                    
                    # Consider high-influence if followers > 10k or high engagement
                    if followers > 10000 or engagement > 100:
                        # Simple sentiment scoring
                        text = post.get('text', '').lower()
                        if any(word in text for word in ['bullish', 'buy', 'long', 'calls']):
                            influencer_sentiments.append(1)
                        elif any(word in text for word in ['bearish', 'sell', 'short', 'puts']):
                            influencer_sentiments.append(-1)
        
        if influencer_sentiments:
            return np.mean(influencer_sentiments)
        return 0.0
    
    def _calculate_weighted_sentiment(self, social_data: Dict) -> float:
        """Calculate platform-weighted sentiment score."""
        platform_sentiments = {}
        
        for platform, data in social_data.items():
            posts = []
            if isinstance(data, list):
                posts = data
            elif isinstance(data, dict) and 'posts' in data:
                posts = data['posts']
            
            sentiments = []
            for post in posts:
                text = post.get('text', '').lower() if isinstance(post, dict) else str(post).lower()
                
                # Simple sentiment scoring
                pos_score = sum(1 for word in ['bullish', 'buy', 'long'] if word in text)
                neg_score = sum(1 for word in ['bearish', 'sell', 'short'] if word in text)
                
                if pos_score > neg_score:
                    sentiments.append(1)
                elif neg_score > pos_score:
                    sentiments.append(-1)
                else:
                    sentiments.append(0)
            
            if sentiments:
                platform_sentiments[platform] = np.mean(sentiments)
        
        # Calculate weighted average
        weighted_sum = 0
        weight_sum = 0
        
        for platform, sentiment in platform_sentiments.items():
            weight = self.sentiment_weights.get(platform, 0.2)
            weighted_sum += sentiment * weight
            weight_sum += weight
        
        return weighted_sum / weight_sum if weight_sum > 0 else 0.0
    
    def _calculate_avg_engagement(self, social_data: Dict) -> float:
        """Calculate average engagement across posts."""
        total_engagement = 0
        post_count = 0
        
        for platform, data in social_data.items():
            posts = []
            if isinstance(data, list):
                posts = data
            elif isinstance(data, dict) and 'posts' in data:
                posts = data['posts']
            
            for post in posts:
                if isinstance(post, dict):
                    engagement = post.get('engagement', 
                                        post.get('likes', 0) + 
                                        post.get('retweets', 0) + 
                                        post.get('comments', 0))
                    total_engagement += engagement
                    post_count += 1
        
        return total_engagement / post_count if post_count > 0 else 0
    
    def _calculate_sentiment_volatility(self, social_data: Dict) -> float:
        """Calculate volatility of sentiment over time."""
        # This would ideally track sentiment changes over time
        # For now, return a placeholder
        return 0.2
    
    def _calculate_confidence(self, volume: int, distribution: Dict[str, float]) -> float:
        """Calculate confidence in social sentiment signal."""
        # Volume confidence
        volume_confidence = min(volume / 500, 1.0) * 0.5
        
        # Distribution confidence (how decisive is the sentiment)
        max_sentiment = max(distribution.values()) if distribution else 0.33
        distribution_confidence = (max_sentiment - 0.33) / 0.67 * 0.5
        
        return volume_confidence + distribution_confidence
    
    def _default_signal(self) -> SocialSentiment:
        """Return default neutral signal when analysis fails."""
        return SocialSentiment(
            timestamp=datetime.now(),
            signal_type='neutral',
            strength=0.5,
            sentiment_score=0.0,
            confidence=0.0,
            mention_volume=0,
            sentiment_distribution={'positive': 0.33, 'negative': 0.33, 'neutral': 0.34},
            trending_topics=[],
            crowd_consensus=0.0,
            influencer_sentiment=0.0,
            metrics={},
            metadata={'error': 'Insufficient or invalid data'}
        )


class OptionsFlowModel(BaseSentimentModel):
    """
    Options flow sentiment analysis.
    
    Analyzes:
    - Put/call ratios
    - Unusual options activity
    - Smart money flow
    - Gamma exposure
    
    Requires real options trade data.
    """
    
    def __init__(self, unusual_volume_threshold: float = 3.0,
                 smart_money_threshold: float = 100000):
        super().__init__()
        self.unusual_volume_threshold = unusual_volume_threshold
        self.smart_money_threshold = smart_money_threshold
    
    def analyze(self, options_data: pd.DataFrame) -> OptionsFlowSignal:
        """
        Analyze options flow data.
        
        Expected columns: timestamp, type (call/put), volume, open_interest, 
                         premium, strike, expiry, underlying_price
        """
        required_cols = ['timestamp', 'type', 'volume', 'premium']
        if not isinstance(options_data, pd.DataFrame) or \
           not all(col in options_data.columns for col in required_cols):
            return self._default_signal()
        
        if not self.validate_data(options_data):
            return self._default_signal()
        
        try:
            # Calculate put/call ratio
            put_call_ratio = self._calculate_put_call_ratio(options_data)
            
            # Detect unusual activity
            unusual_activity = self._detect_unusual_activity(options_data)
            
            # Analyze smart money flow
            smart_money_direction = self._analyze_smart_money(options_data)
            
            # Calculate gamma exposure
            gamma_exposure = self._calculate_gamma_exposure(options_data)
            
            # Calculate implied move
            implied_move = self._calculate_implied_move(options_data)
            
            # Determine sentiment
            if put_call_ratio < 0.7 and smart_money_direction == 'bullish':
                signal_type = 'bullish'
                sentiment_score = 0.7
                strength = 0.8
            elif put_call_ratio > 1.3 and smart_money_direction == 'bearish':
                signal_type = 'bearish'
                sentiment_score = -0.7
                strength = 0.8
            else:
                signal_type = 'neutral'
                sentiment_score = (1 - put_call_ratio) / 2  # Convert to -1 to 1 scale
                strength = 0.5
            
            # Adjust for unusual activity
            if len(unusual_activity) > 5:
                strength = min(strength * 1.2, 1.0)
            
            # Adjust for gamma exposure
            if abs(gamma_exposure) > 0.5:
                strength = min(strength * 1.1, 1.0)
            
            confidence = self._calculate_confidence(options_data, unusual_activity)
            
            return OptionsFlowSignal(
                timestamp=options_data['timestamp'].max(),
                signal_type=signal_type,
                strength=strength,
                sentiment_score=sentiment_score,
                confidence=confidence,
                put_call_ratio=put_call_ratio,
                unusual_activity=unusual_activity,
                smart_money_direction=smart_money_direction,
                gamma_exposure=gamma_exposure,
                implied_move=implied_move,
                metrics={
                    'total_volume': options_data['volume'].sum(),
                    'call_volume': options_data[options_data['type'] == 'call']['volume'].sum(),
                    'put_volume': options_data[options_data['type'] == 'put']['volume'].sum(),
                    'avg_premium': options_data['premium'].mean()
                },
                metadata={
                    'data_points': len(options_data),
                    'time_range': (options_data['timestamp'].min(), options_data['timestamp'].max())
                }
            )
            
        except Exception as e:
            return self._default_signal()
    
    def _calculate_put_call_ratio(self, options_data: pd.DataFrame) -> float:
        """Calculate put/call volume ratio."""
        call_volume = options_data[options_data['type'] == 'call']['volume'].sum()
        put_volume = options_data[options_data['type'] == 'put']['volume'].sum()
        
        if call_volume == 0:
            return float('inf') if put_volume > 0 else 1.0
        
        return put_volume / call_volume
    
    def _detect_unusual_activity(self, options_data: pd.DataFrame) -> List[Dict[str, any]]:
        """Detect unusual options activity."""
        unusual_trades = []
        
        if 'open_interest' not in options_data.columns:
            return unusual_trades
        
        # Calculate volume/OI ratio
        options_data['vol_oi_ratio'] = options_data['volume'] / (options_data['open_interest'] + 1)
        
        # Find trades with high vol/OI ratio
        unusual_mask = options_data['vol_oi_ratio'] > self.unusual_volume_threshold
        
        for idx, row in options_data[unusual_mask].iterrows():
            unusual_trades.append({
                'timestamp': row['timestamp'],
                'type': row['type'],
                'strike': row.get('strike', 0),
                'volume': row['volume'],
                'vol_oi_ratio': row['vol_oi_ratio'],
                'premium': row['premium'],
                'total_value': row['volume'] * row['premium'] * 100  # Assuming standard contract size
            })
        
        # Sort by total value
        return sorted(unusual_trades, key=lambda x: x['total_value'], reverse=True)[:10]
    
    def _analyze_smart_money(self, options_data: pd.DataFrame) -> str:
        """Analyze smart money flow direction."""
        # Calculate dollar volume
        options_data['dollar_volume'] = options_data['volume'] * options_data['premium'] * 100
        
        # Filter for large trades (smart money)
        large_trades = options_data[options_data['dollar_volume'] > self.smart_money_threshold]
        
        if len(large_trades) == 0:
            return 'neutral'
        
        # Calculate net positioning
        call_value = large_trades[large_trades['type'] == 'call']['dollar_volume'].sum()
        put_value = large_trades[large_trades['type'] == 'put']['dollar_volume'].sum()
        
        net_positioning = call_value - put_value
        total_value = call_value + put_value
        
        if total_value == 0:
            return 'neutral'
        
        positioning_ratio = net_positioning / total_value
        
        if positioning_ratio > 0.3:
            return 'bullish'
        elif positioning_ratio < -0.3:
            return 'bearish'
        else:
            return 'neutral'
    
    def _calculate_gamma_exposure(self, options_data: pd.DataFrame) -> float:
        """Estimate gamma exposure (simplified)."""
        if 'strike' not in options_data.columns or 'underlying_price' not in options_data.columns:
            return 0.0
        
        # Calculate moneyness
        options_data['moneyness'] = options_data['strike'] / options_data['underlying_price']
        
        # Near-the-money options have highest gamma
        ntm_mask = (options_data['moneyness'] > 0.95) & (options_data['moneyness'] < 1.05)
        ntm_volume = options_data[ntm_mask]['volume'].sum()
        total_volume = options_data['volume'].sum()
        
        # High NTM volume relative to total suggests high gamma exposure
        gamma_exposure = ntm_volume / total_volume if total_volume > 0 else 0.0
        
        # Adjust for put/call imbalance in NTM options
        if len(options_data[ntm_mask]) > 0:
            ntm_put_call = self._calculate_put_call_ratio(options_data[ntm_mask])
            if ntm_put_call > 1.5 or ntm_put_call < 0.67:
                gamma_exposure *= 1.5  # Amplify if imbalanced
        
        return min(gamma_exposure, 1.0)
    
    def _calculate_implied_move(self, options_data: pd.DataFrame) -> float:
        """Calculate implied move from options prices."""
        if 'strike' not in options_data.columns or 'underlying_price' not in options_data.columns:
            return 0.0
        
        # Get ATM options
        current_price = options_data['underlying_price'].iloc[-1]
        options_data['distance'] = abs(options_data['strike'] - current_price)
        
        # Find closest strikes
        atm_options = options_data.nsmallest(4, 'distance')
        
        if len(atm_options) == 0:
            return 0.0
        
        # Calculate average ATM implied volatility proxy (premium / strike)
        avg_iv_proxy = (atm_options['premium'] / atm_options['strike']).mean()
        
        # Rough implied move calculation (simplified)
        # Real calculation would use proper IV and time to expiry
        implied_move = avg_iv_proxy * np.sqrt(30/365) * 100  # Assuming 30-day options
        
        return implied_move
    
    def _calculate_confidence(self, options_data: pd.DataFrame, unusual_activity: List) -> float:
        """Calculate confidence in options flow signal."""
        # Volume confidence
        total_volume = options_data['volume'].sum()
        volume_confidence = min(total_volume / 10000, 1.0) * 0.4
        
        # Unusual activity confidence
        unusual_confidence = min(len(unusual_activity) / 10, 1.0) * 0.3
        
        # Data completeness confidence
        expected_cols = ['strike', 'expiry', 'open_interest', 'underlying_price']
        completeness = sum(1 for col in expected_cols if col in options_data.columns) / len(expected_cols)
        completeness_confidence = completeness * 0.3
        
        return volume_confidence + unusual_confidence + completeness_confidence
    
    def _default_signal(self) -> OptionsFlowSignal:
        """Return default neutral signal when analysis fails."""
        return OptionsFlowSignal(
            timestamp=datetime.now(),
            signal_type='neutral',
            strength=0.5,
            sentiment_score=0.0,
            confidence=0.0,
            put_call_ratio=1.0,
            unusual_activity=[],
            smart_money_direction='neutral',
            gamma_exposure=0.0,
            implied_move=0.0,
            metrics={},
            metadata={'error': 'Insufficient or invalid data'}
        )


class VIXRegimeModel(BaseSentimentModel):
    """
    VIX regime analysis for market sentiment.
    
    Analyzes:
    - VIX term structure (contango/backwardation)
    - Mean reversion signals
    - Risk-on/risk-off regime identification
    - Volatility percentiles
    
    Requires VIX and VIX futures data.
    """
    
    def __init__(self, lookback_period: int = 252,
                 mean_reversion_threshold: float = 1.5,
                 regime_threshold: float = 20.0):
        super().__init__()
        self.lookback_period = lookback_period
        self.mean_reversion_threshold = mean_reversion_threshold
        self.regime_threshold = regime_threshold
    
    def analyze(self, vix_data: pd.DataFrame) -> VIXRegimeSignal:
        """
        Analyze VIX regime.
        
        Expected columns: timestamp, vix_close, vix_1m, vix_2m, vix_3m (futures)
        """
        required_cols = ['timestamp', 'vix_close']
        if not isinstance(vix_data, pd.DataFrame) or \
           not all(col in vix_data.columns for col in required_cols):
            return self._default_signal()
        
        if not self.validate_data(vix_data):
            return self._default_signal()
        
        try:
            # Current VIX level
            vix_level = vix_data['vix_close'].iloc[-1]
            
            # Analyze term structure
            term_structure = self._analyze_term_structure(vix_data)
            
            # Calculate mean reversion signal
            mean_reversion_signal = self._calculate_mean_reversion(vix_data)
            
            # Determine risk regime
            risk_regime = self._determine_risk_regime(vix_level, vix_data)
            
            # Calculate volatility percentile
            volatility_percentile = self._calculate_volatility_percentile(vix_data)
            
            # Determine sentiment based on VIX regime
            if risk_regime == 'risk-on' and term_structure == 'contango':
                signal_type = 'bullish'
                sentiment_score = 0.6
                strength = 0.7
            elif risk_regime == 'risk-off' and term_structure == 'backwardation':
                signal_type = 'bearish'
                sentiment_score = -0.6
                strength = 0.7
            else:
                signal_type = 'neutral'
                sentiment_score = -0.2 if vix_level > self.regime_threshold else 0.2
                strength = 0.5
            
            # Adjust for mean reversion
            if abs(mean_reversion_signal) > 0.5:
                if mean_reversion_signal > 0 and signal_type == 'bearish':
                    # VIX likely to fall, bullish for market
                    signal_type = 'bullish'
                    strength = min(strength * 1.2, 1.0)
                elif mean_reversion_signal < 0 and signal_type == 'bullish':
                    # VIX likely to rise, bearish for market
                    signal_type = 'bearish'
                    strength = min(strength * 1.2, 1.0)
            
            confidence = self._calculate_confidence(vix_data, term_structure)
            
            return VIXRegimeSignal(
                timestamp=vix_data['timestamp'].iloc[-1],
                signal_type=signal_type,
                strength=strength,
                sentiment_score=sentiment_score,
                confidence=confidence,
                vix_level=vix_level,
                term_structure=term_structure,
                mean_reversion_signal=mean_reversion_signal,
                risk_regime=risk_regime,
                volatility_percentile=volatility_percentile,
                metrics={
                    'vix_sma20': vix_data['vix_close'].tail(20).mean(),
                    'vix_sma50': vix_data['vix_close'].tail(50).mean() if len(vix_data) >= 50 else vix_level,
                    'vix_min_30d': vix_data['vix_close'].tail(30).min(),
                    'vix_max_30d': vix_data['vix_close'].tail(30).max()
                },
                metadata={
                    'data_points': len(vix_data),
                    'has_futures': any(col in vix_data.columns for col in ['vix_1m', 'vix_2m'])
                }
            )
            
        except Exception as e:
            return self._default_signal()
    
    def _analyze_term_structure(self, vix_data: pd.DataFrame) -> str:
        """Analyze VIX futures term structure."""
        # Check if we have futures data
        futures_cols = ['vix_1m', 'vix_2m', 'vix_3m']
        available_futures = [col for col in futures_cols if col in vix_data.columns]
        
        if not available_futures:
            # Fallback: use historical VIX to estimate structure
            current_vix = vix_data['vix_close'].iloc[-1]
            avg_vix = vix_data['vix_close'].mean()
            
            if current_vix > avg_vix * 1.2:
                return 'backwardation'  # High VIX tends to mean-revert
            else:
                return 'contango'  # Normal state
        
        # Analyze actual term structure
        current_spot = vix_data['vix_close'].iloc[-1]
        current_row = vix_data.iloc[-1]
        
        # Check if futures are higher (contango) or lower (backwardation) than spot
        contango_count = 0
        backwardation_count = 0
        
        for future_col in available_futures:
            if current_row[future_col] > current_spot:
                contango_count += 1
            else:
                backwardation_count += 1
        
        if contango_count > backwardation_count:
            return 'contango'
        else:
            return 'backwardation'
    
    def _calculate_mean_reversion(self, vix_data: pd.DataFrame) -> float:
        """Calculate mean reversion signal for VIX."""
        if len(vix_data) < self.lookback_period:
            lookback = len(vix_data)
        else:
            lookback = self.lookback_period
        
        # Calculate historical mean and std
        historical_data = vix_data['vix_close'].tail(lookback)
        mean_vix = historical_data.mean()
        std_vix = historical_data.std()
        
        # Current VIX level
        current_vix = vix_data['vix_close'].iloc[-1]
        
        # Calculate z-score
        z_score = (current_vix - mean_vix) / std_vix if std_vix > 0 else 0
        
        # Mean reversion signal (negative z-score suggests VIX will fall)
        if abs(z_score) > self.mean_reversion_threshold:
            return -z_score / abs(z_score)  # Normalize to -1 or 1
        else:
            return -z_score / self.mean_reversion_threshold  # Scale to [-1, 1]
    
    def _determine_risk_regime(self, current_vix: float, vix_data: pd.DataFrame) -> str:
        """Determine current risk regime."""
        # Simple regime based on VIX level
        if current_vix < 15:
            regime = 'risk-on'
        elif current_vix > 25:
            regime = 'risk-off'
        else:
            # Check trend
            if len(vix_data) >= 10:
                vix_sma5 = vix_data['vix_close'].tail(5).mean()
                vix_sma10 = vix_data['vix_close'].tail(10).mean()
                
                if vix_sma5 > vix_sma10 * 1.1:
                    regime = 'transitioning'  # Moving to risk-off
                elif vix_sma5 < vix_sma10 * 0.9:
                    regime = 'risk-on'  # Moving to risk-on
                else:
                    regime = 'risk-on' if current_vix < self.regime_threshold else 'risk-off'
            else:
                regime = 'risk-on' if current_vix < self.regime_threshold else 'risk-off'
        
        return regime
    
    def _calculate_volatility_percentile(self, vix_data: pd.DataFrame) -> float:
        """Calculate current VIX percentile rank."""
        if len(vix_data) < 20:
            return 0.5
        
        lookback = min(len(vix_data), self.lookback_period)
        historical_data = vix_data['vix_close'].tail(lookback)
        current_vix = vix_data['vix_close'].iloc[-1]
        
        # Calculate percentile
        percentile = (historical_data < current_vix).sum() / len(historical_data)
        
        return percentile
    
    def _calculate_confidence(self, vix_data: pd.DataFrame, term_structure: str) -> float:
        """Calculate confidence in VIX regime signal."""
        # Data sufficiency confidence
        data_confidence = min(len(vix_data) / self.lookback_period, 1.0) * 0.4
        
        # Term structure confidence (having futures data)
        futures_cols = ['vix_1m', 'vix_2m', 'vix_3m']
        futures_available = sum(1 for col in futures_cols if col in vix_data.columns)
        structure_confidence = (futures_available / 3) * 0.3
        
        # Regime clarity confidence
        current_vix = vix_data['vix_close'].iloc[-1]
        if current_vix < 12 or current_vix > 30:
            regime_confidence = 0.3  # Clear regime
        elif 18 < current_vix < 22:
            regime_confidence = 0.1  # Unclear regime
        else:
            regime_confidence = 0.2
        
        return data_confidence + structure_confidence + regime_confidence
    
    def _default_signal(self) -> VIXRegimeSignal:
        """Return default neutral signal when analysis fails."""
        return VIXRegimeSignal(
            timestamp=datetime.now(),
            signal_type='neutral',
            strength=0.5,
            sentiment_score=0.0,
            confidence=0.0,
            vix_level=20.0,
            term_structure='contango',
            mean_reversion_signal=0.0,
            risk_regime='risk-on',
            volatility_percentile=0.5,
            metrics={},
            metadata={'error': 'Insufficient or invalid data'}
        )


# Model factory for easy instantiation
def create_sentiment_model(model_type: str, **kwargs) -> BaseSentimentModel:
    """Factory function to create sentiment models."""
    models = {
        'news': NewsSentimentModel,
        'social': SocialSentimentModel,
        'options_flow': OptionsFlowModel,
        'vix_regime': VIXRegimeModel
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return models[model_type](**kwargs)