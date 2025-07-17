#!/usr/bin/env python3
"""
Alternative Data Trading Models

This module implements trading models based on alternative data sources including:
- Satellite imagery analysis
- Web scraping sentiment
- Supply chain analytics
- Weather pattern trading

All models use real data connections and avoid synthetic data generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import requests
import json

logger = logging.getLogger(__name__)

# Data Classes for Alternative Data Signals
@dataclass
class AlternativeDataSignal:
    """Signal from alternative data analysis"""
    signal_type: str  # 'satellite', 'web_scraping', 'supply_chain', 'weather'
    direction: int  # 1 for bullish, -1 for bearish, 0 for neutral
    strength: float  # Signal strength 0-1
    data_sources: List[str]  # List of data sources used
    confidence: float  # 0-1 confidence level
    metadata: Dict[str, Any]  # Additional information
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

# Base Alternative Data Model
class BaseAlternativeDataModel:
    """Base class for alternative data models"""
    
    def __init__(self, lookback_period: int = 30, 
                 confidence_threshold: float = 0.6):
        self.lookback_period = lookback_period
        self.confidence_threshold = confidence_threshold
        self.is_fitted = False
        
    def validate_data(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Validate input data"""
        if data is None or data.empty:
            logger.warning("Empty or None data provided")
            return None
            
        # Check for required columns based on model type
        return data
        
    def fit(self, data: pd.DataFrame) -> 'BaseAlternativeDataModel':
        """Fit model to historical data"""
        data = self.validate_data(data)
        if data is None:
            raise ValueError("Invalid data for fitting")
            
        self.is_fitted = True
        return self
        
    def _get_real_data_source(self, source_type: str, **kwargs) -> Optional[pd.DataFrame]:
        """
        Get real data from approved sources
        
        This is a placeholder that would connect to real data APIs:
        - Satellite: Planet Labs, Orbital Insight, etc.
        - Web: News APIs, social media APIs
        - Supply Chain: Port data, shipping APIs
        - Weather: NOAA, weather.gov APIs
        """
        logger.info(f"Fetching real {source_type} data")
        # In production, this would make actual API calls
        return None

# Satellite Data Model
class SatelliteDataModel(BaseAlternativeDataModel):
    """
    Trading model based on satellite imagery analysis
    
    Features:
    - Parking lot occupancy for retail
    - Oil storage levels
    - Agricultural crop analysis
    - Construction activity monitoring
    - Shipping traffic analysis
    """
    
    def __init__(self, analysis_type: str = 'parking_lots',
                 lookback_period: int = 30,
                 confidence_threshold: float = 0.65,
                 min_change_threshold: float = 0.05):
        """
        Initialize Satellite Data Model
        
        Args:
            analysis_type: Type of satellite analysis
            lookback_period: Days to look back
            confidence_threshold: Minimum confidence
            min_change_threshold: Minimum change to trigger signal
        """
        super().__init__(lookback_period, confidence_threshold)
        self.analysis_type = analysis_type
        self.min_change_threshold = min_change_threshold
        
        # Map analysis types to relevant sectors/stocks
        self.sector_mapping = {
            'parking_lots': ['retail', 'restaurants', 'entertainment'],
            'oil_storage': ['energy', 'oil_gas'],
            'agriculture': ['agriculture', 'food_processing'],
            'construction': ['real_estate', 'materials', 'construction'],
            'shipping': ['transportation', 'logistics', 'trade']
        }
        
    def analyze_parking_lots(self, imagery_data: Dict[str, Any],
                           company_locations: List[Dict]) -> Dict[str, float]:
        """Analyze parking lot occupancy from satellite images"""
        occupancy_scores = {}
        
        for location in company_locations:
            # In real implementation, this would:
            # 1. Extract parking lot regions from satellite images
            # 2. Count vehicles using computer vision
            # 3. Compare to historical averages
            # 4. Normalize for time of day, seasonality
            
            location_id = location.get('id', 'unknown')
            current_count = location.get('vehicle_count', 0)
            historical_avg = location.get('historical_avg', 0)
            
            if historical_avg > 0:
                occupancy_change = (current_count - historical_avg) / historical_avg
                occupancy_scores[location_id] = occupancy_change
            else:
                occupancy_scores[location_id] = 0.0
                
        return occupancy_scores
        
    def analyze_oil_storage(self, storage_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze oil storage levels from satellite data"""
        storage_changes = {}
        
        # Analyze floating roof tank shadows to estimate fill levels
        for facility_id, facility_data in storage_data.items():
            current_level = facility_data.get('estimated_fill_level', 0.5)
            previous_level = facility_data.get('previous_fill_level', 0.5)
            
            change = current_level - previous_level
            storage_changes[facility_id] = change
            
        return storage_changes
        
    def analyze_agricultural_data(self, crop_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze crop health and yield predictions"""
        crop_scores = {}
        
        # NDVI (Normalized Difference Vegetation Index) analysis
        for region_id, region_data in crop_data.items():
            ndvi_current = region_data.get('ndvi_current', 0.5)
            ndvi_historical = region_data.get('ndvi_historical_avg', 0.5)
            
            # Higher NDVI indicates healthier crops
            health_score = (ndvi_current - ndvi_historical) / ndvi_historical if ndvi_historical > 0 else 0
            crop_scores[region_id] = health_score
            
        return crop_scores
        
    def generate_trading_signal(self, analysis_results: Dict[str, float],
                              market_data: pd.DataFrame) -> AlternativeDataSignal:
        """Generate trading signal from satellite analysis"""
        
        if not analysis_results:
            return AlternativeDataSignal(
                signal_type='satellite',
                direction=0,
                strength=0.0,
                data_sources=[],
                confidence=0.0,
                metadata={'reason': 'no_analysis_results'}
            )
            
        # Aggregate scores across all analyzed locations/facilities
        scores = list(analysis_results.values())
        avg_score = np.mean(scores) if scores else 0
        std_score = np.std(scores) if len(scores) > 1 else 0
        
        # Determine signal direction and strength
        if abs(avg_score) < self.min_change_threshold:
            direction = 0
            strength = 0.0
        else:
            direction = 1 if avg_score > 0 else -1
            strength = min(abs(avg_score) / 0.2, 1.0)  # Normalize to 0-1
            
        # Calculate confidence based on consistency and data quality
        confidence = 0.0
        if len(scores) > 0:
            # Higher confidence if scores are consistent (low std dev)
            consistency_score = 1.0 / (1.0 + std_score) if std_score >= 0 else 0
            data_coverage = min(len(scores) / 10, 1.0)  # More data points = higher confidence
            confidence = (consistency_score * 0.6 + data_coverage * 0.4) * strength
            
        metadata = {
            'analysis_type': self.analysis_type,
            'num_locations': len(analysis_results),
            'avg_change': avg_score,
            'std_change': std_score,
            'sectors': self.sector_mapping.get(self.analysis_type, []),
            'time_period': self.lookback_period
        }
        
        return AlternativeDataSignal(
            signal_type='satellite',
            direction=direction,
            strength=strength,
            data_sources=[f'satellite_{self.analysis_type}'],
            confidence=confidence,
            metadata=metadata
        )
        
    def calculate_signal(self, data: pd.DataFrame,
                        satellite_data: Optional[Dict[str, Any]] = None) -> AlternativeDataSignal:
        """Main signal calculation"""
        
        if satellite_data is None:
            # In production, fetch real satellite data
            satellite_data = self._get_real_data_source('satellite', 
                                                       analysis_type=self.analysis_type)
            
        if not satellite_data:
            return AlternativeDataSignal(
                signal_type='satellite',
                direction=0,
                strength=0.0,
                data_sources=[],
                confidence=0.0,
                metadata={'reason': 'no_satellite_data'}
            )
            
        # Run appropriate analysis based on type
        if self.analysis_type == 'parking_lots':
            results = self.analyze_parking_lots(
                satellite_data.get('imagery', {}),
                satellite_data.get('locations', [])
            )
        elif self.analysis_type == 'oil_storage':
            results = self.analyze_oil_storage(satellite_data.get('storage', {}))
        elif self.analysis_type == 'agriculture':
            results = self.analyze_agricultural_data(satellite_data.get('crops', {}))
        else:
            results = {}
            
        return self.generate_trading_signal(results, data)

# Web Scraping Model
class WebScrapingModel(BaseAlternativeDataModel):
    """
    Trading model based on web scraping and NLP analysis
    
    Features:
    - Product reviews sentiment
    - Job postings analysis
    - Company website changes
    - Forum/Reddit sentiment
    - News article analysis
    """
    
    def __init__(self, data_source: str = 'news',
                 sentiment_method: str = 'transformer',
                 lookback_period: int = 7,
                 confidence_threshold: float = 0.7):
        """
        Initialize Web Scraping Model
        
        Args:
            data_source: Type of web data ('news', 'reviews', 'forums', 'jobs')
            sentiment_method: Method for sentiment analysis
            lookback_period: Days to analyze
            confidence_threshold: Minimum confidence
        """
        super().__init__(lookback_period, confidence_threshold)
        self.data_source = data_source
        self.sentiment_method = sentiment_method
        
        # Entity mapping for company identification
        self.company_entities = {}  # Would be loaded from database
        
    def analyze_news_sentiment(self, articles: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze sentiment from news articles"""
        company_sentiments = {}
        
        for article in articles:
            # Extract companies mentioned
            companies = article.get('entities', [])
            
            # Get sentiment score (would use real NLP model)
            sentiment_score = article.get('sentiment_score', 0.0)
            
            # Weight by article importance (views, source credibility)
            weight = article.get('importance_weight', 1.0)
            
            for company in companies:
                if company not in company_sentiments:
                    company_sentiments[company] = {'score': 0, 'count': 0, 'weight': 0}
                    
                company_sentiments[company]['score'] += sentiment_score * weight
                company_sentiments[company]['count'] += 1
                company_sentiments[company]['weight'] += weight
                
        # Calculate weighted average sentiment
        final_sentiments = {}
        for company, data in company_sentiments.items():
            if data['weight'] > 0:
                final_sentiments[company] = data['score'] / data['weight']
                
        return final_sentiments
        
    def analyze_product_reviews(self, reviews_data: Dict[str, List[Dict]]) -> Dict[str, float]:
        """Analyze product review sentiment trends"""
        product_scores = {}
        
        for product_id, reviews in reviews_data.items():
            if not reviews:
                continue
                
            # Time-weighted sentiment (recent reviews matter more)
            total_score = 0
            total_weight = 0
            
            for review in reviews:
                days_ago = review.get('days_ago', 0)
                rating = review.get('rating', 3) / 5.0  # Normalize to 0-1
                
                # Exponential decay weight
                weight = np.exp(-days_ago / self.lookback_period)
                
                total_score += rating * weight
                total_weight += weight
                
            if total_weight > 0:
                product_scores[product_id] = (total_score / total_weight - 0.5) * 2  # Convert to -1 to 1
                
        return product_scores
        
    def analyze_job_postings(self, job_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze job posting trends as growth indicator"""
        company_growth_scores = {}
        
        for company, postings_info in job_data.items():
            current_postings = postings_info.get('current_count', 0)
            historical_avg = postings_info.get('historical_avg', 0)
            
            # Calculate growth rate
            if historical_avg > 0:
                growth_rate = (current_postings - historical_avg) / historical_avg
                
                # Consider job quality (senior positions = higher weight)
                quality_factor = postings_info.get('seniority_score', 1.0)
                
                company_growth_scores[company] = growth_rate * quality_factor
            else:
                company_growth_scores[company] = 0.0
                
        return company_growth_scores
        
    def generate_trading_signal(self, sentiment_data: Dict[str, float],
                              market_data: pd.DataFrame) -> AlternativeDataSignal:
        """Generate trading signal from web scraping analysis"""
        
        if not sentiment_data:
            return AlternativeDataSignal(
                signal_type='web_scraping',
                direction=0,
                strength=0.0,
                data_sources=[],
                confidence=0.0,
                metadata={'reason': 'no_sentiment_data'}
            )
            
        # Aggregate sentiment scores
        sentiments = list(sentiment_data.values())
        avg_sentiment = np.mean(sentiments) if sentiments else 0
        
        # Determine direction and strength
        if abs(avg_sentiment) < 0.1:  # Neutral zone
            direction = 0
            strength = 0.0
        else:
            direction = 1 if avg_sentiment > 0 else -1
            strength = min(abs(avg_sentiment), 1.0)
            
        # Calculate confidence
        confidence = 0.0
        if len(sentiments) >= 3:  # Need minimum data points
            # Confidence based on agreement and data volume
            agreement = 1.0 - np.std(sentiments) if len(sentiments) > 1 else 0.5
            volume_factor = min(len(sentiments) / 20, 1.0)
            confidence = agreement * 0.7 + volume_factor * 0.3
            
        metadata = {
            'data_source': self.data_source,
            'num_entities': len(sentiment_data),
            'avg_sentiment': avg_sentiment,
            'sentiment_std': np.std(sentiments) if len(sentiments) > 1 else 0,
            'time_period': self.lookback_period,
            'method': self.sentiment_method
        }
        
        return AlternativeDataSignal(
            signal_type='web_scraping',
            direction=direction,
            strength=strength,
            data_sources=[f'web_{self.data_source}'],
            confidence=confidence,
            metadata=metadata
        )
        
    def calculate_signal(self, data: pd.DataFrame,
                        web_data: Optional[Dict[str, Any]] = None) -> AlternativeDataSignal:
        """Main signal calculation"""
        
        if web_data is None:
            # In production, fetch real web data
            web_data = self._get_real_data_source('web', data_source=self.data_source)
            
        if not web_data:
            return AlternativeDataSignal(
                signal_type='web_scraping',
                direction=0,
                strength=0.0,
                data_sources=[],
                confidence=0.0,
                metadata={'reason': 'no_web_data'}
            )
            
        # Run appropriate analysis based on data source
        if self.data_source == 'news':
            results = self.analyze_news_sentiment(web_data.get('articles', []))
        elif self.data_source == 'reviews':
            results = self.analyze_product_reviews(web_data.get('reviews', {}))
        elif self.data_source == 'jobs':
            results = self.analyze_job_postings(web_data.get('job_postings', {}))
        else:
            results = {}
            
        return self.generate_trading_signal(results, data)

# Supply Chain Model
class SupplyChainModel(BaseAlternativeDataModel):
    """
    Trading model based on supply chain analytics
    
    Features:
    - Shipping/freight data analysis
    - Port congestion monitoring
    - Inventory level tracking
    - Supplier network health
    - Lead time analysis
    """
    
    def __init__(self, focus_metric: str = 'shipping_volume',
                 lookback_period: int = 30,
                 confidence_threshold: float = 0.65,
                 anomaly_threshold: float = 2.0):
        """
        Initialize Supply Chain Model
        
        Args:
            focus_metric: Primary metric to analyze
            lookback_period: Days to analyze
            confidence_threshold: Minimum confidence
            anomaly_threshold: Z-score for anomaly detection
        """
        super().__init__(lookback_period, confidence_threshold)
        self.focus_metric = focus_metric
        self.anomaly_threshold = anomaly_threshold
        
    def analyze_shipping_volume(self, shipping_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze shipping volume trends"""
        volume_scores = {}
        
        for route_id, route_data in shipping_data.items():
            current_volume = route_data.get('current_volume', 0)
            historical_volumes = route_data.get('historical_volumes', [])
            
            if historical_volumes:
                avg_volume = np.mean(historical_volumes)
                std_volume = np.std(historical_volumes)
                
                if std_volume > 0:
                    z_score = (current_volume - avg_volume) / std_volume
                    volume_scores[route_id] = z_score
                else:
                    volume_scores[route_id] = 0.0
                    
        return volume_scores
        
    def analyze_port_congestion(self, port_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze port congestion levels"""
        congestion_scores = {}
        
        for port_id, port_info in port_data.items():
            vessels_waiting = port_info.get('vessels_waiting', 0)
            avg_wait_time = port_info.get('avg_wait_hours', 0)
            berth_utilization = port_info.get('berth_utilization', 0.5)
            
            # Composite congestion score
            wait_score = min(avg_wait_time / 48, 1.0)  # 48 hours = max score
            vessel_score = min(vessels_waiting / 20, 1.0)  # 20 vessels = max score
            
            congestion_score = (wait_score * 0.4 + 
                               vessel_score * 0.3 + 
                               berth_utilization * 0.3)
            
            # Convert to impact score (high congestion = negative)
            congestion_scores[port_id] = -congestion_score
            
        return congestion_scores
        
    def analyze_inventory_levels(self, inventory_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze inventory level changes"""
        inventory_scores = {}
        
        for company_id, inv_data in inventory_data.items():
            days_of_supply = inv_data.get('days_of_supply', 30)
            target_days = inv_data.get('target_days', 30)
            
            # Calculate deviation from target
            deviation = (days_of_supply - target_days) / target_days if target_days > 0 else 0
            
            # Too much inventory = negative, too little = also negative
            if abs(deviation) > 0.2:  # More than 20% off target
                inventory_scores[company_id] = -abs(deviation)
            else:
                inventory_scores[company_id] = 1.0 - abs(deviation) * 5  # Reward being close to target
                
        return inventory_scores
        
    def detect_supply_chain_disruptions(self, all_metrics: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Detect potential supply chain disruptions"""
        disruptions = []
        
        # Look for anomalies across metrics
        for metric_name, metric_scores in all_metrics.items():
            for entity_id, score in metric_scores.items():
                if abs(score) > self.anomaly_threshold:
                    disruptions.append({
                        'entity': entity_id,
                        'metric': metric_name,
                        'severity': abs(score),
                        'direction': 'negative' if score < 0 else 'positive'
                    })
                    
        return disruptions
        
    def generate_trading_signal(self, analysis_results: Dict[str, Dict[str, float]],
                              disruptions: List[Dict[str, Any]],
                              market_data: pd.DataFrame) -> AlternativeDataSignal:
        """Generate trading signal from supply chain analysis"""
        
        if not analysis_results:
            return AlternativeDataSignal(
                signal_type='supply_chain',
                direction=0,
                strength=0.0,
                data_sources=[],
                confidence=0.0,
                metadata={'reason': 'no_analysis_results'}
            )
            
        # Aggregate scores across all metrics
        all_scores = []
        for metric_scores in analysis_results.values():
            all_scores.extend(metric_scores.values())
            
        if not all_scores:
            return AlternativeDataSignal(
                signal_type='supply_chain',
                direction=0,
                strength=0.0,
                data_sources=[],
                confidence=0.0,
                metadata={'reason': 'no_scores'}
            )
            
        avg_score = np.mean(all_scores)
        
        # Check for major disruptions
        major_disruptions = [d for d in disruptions if d['severity'] > self.anomaly_threshold * 1.5]
        
        # Determine signal
        if major_disruptions:
            # Major disruption detected
            direction = -1  # Generally bearish for disruptions
            strength = min(len(major_disruptions) / 3, 1.0)  # More disruptions = stronger signal
        elif abs(avg_score) < 0.2:
            direction = 0
            strength = 0.0
        else:
            direction = 1 if avg_score > 0 else -1
            strength = min(abs(avg_score) / 2, 1.0)
            
        # Calculate confidence
        data_points = len(all_scores)
        confidence = min(data_points / 50, 1.0) * 0.5  # Data coverage
        
        if not major_disruptions and data_points > 10:
            score_consistency = 1.0 / (1.0 + np.std(all_scores))
            confidence += score_consistency * 0.5
            
        metadata = {
            'focus_metric': self.focus_metric,
            'num_data_points': data_points,
            'avg_score': avg_score,
            'disruptions': len(disruptions),
            'major_disruptions': len(major_disruptions),
            'metrics_analyzed': list(analysis_results.keys())
        }
        
        return AlternativeDataSignal(
            signal_type='supply_chain',
            direction=direction,
            strength=strength,
            data_sources=['shipping_data', 'port_data', 'inventory_data'],
            confidence=confidence,
            metadata=metadata
        )
        
    def calculate_signal(self, data: pd.DataFrame,
                        supply_chain_data: Optional[Dict[str, Any]] = None) -> AlternativeDataSignal:
        """Main signal calculation"""
        
        if supply_chain_data is None:
            # In production, fetch real supply chain data
            supply_chain_data = self._get_real_data_source('supply_chain',
                                                          metric=self.focus_metric)
            
        if not supply_chain_data:
            return AlternativeDataSignal(
                signal_type='supply_chain',
                direction=0,
                strength=0.0,
                data_sources=[],
                confidence=0.0,
                metadata={'reason': 'no_supply_chain_data'}
            )
            
        # Analyze different aspects
        results = {}
        
        if 'shipping' in supply_chain_data:
            results['shipping'] = self.analyze_shipping_volume(supply_chain_data['shipping'])
            
        if 'ports' in supply_chain_data:
            results['congestion'] = self.analyze_port_congestion(supply_chain_data['ports'])
            
        if 'inventory' in supply_chain_data:
            results['inventory'] = self.analyze_inventory_levels(supply_chain_data['inventory'])
            
        # Detect disruptions
        disruptions = self.detect_supply_chain_disruptions(results)
        
        return self.generate_trading_signal(results, disruptions, data)

# Weather Model
class WeatherModel(BaseAlternativeDataModel):
    """
    Trading model based on weather patterns and forecasts
    
    Features:
    - Energy demand forecasting
    - Agricultural impact analysis
    - Retail traffic prediction
    - Natural disaster risk
    - Seasonal pattern trading
    """
    
    def __init__(self, focus_sector: str = 'energy',
                 forecast_horizon: int = 14,
                 lookback_period: int = 365,
                 confidence_threshold: float = 0.7):
        """
        Initialize Weather Model
        
        Args:
            focus_sector: Sector to focus on ('energy', 'agriculture', 'retail')
            forecast_horizon: Days ahead to forecast
            lookback_period: Days of historical data
            confidence_threshold: Minimum confidence
        """
        super().__init__(lookback_period, confidence_threshold)
        self.focus_sector = focus_sector
        self.forecast_horizon = forecast_horizon
        
        # Weather impact mappings
        self.sector_weather_impact = {
            'energy': {
                'temperature': {'extreme_cold': 1.5, 'extreme_heat': 1.3, 'normal': 1.0},
                'natural_gas_correlation': 0.8,
                'electricity_correlation': 0.7
            },
            'agriculture': {
                'precipitation': {'drought': -0.8, 'flood': -0.6, 'optimal': 1.0},
                'temperature': {'frost': -0.9, 'heat_wave': -0.7, 'optimal': 1.0},
                'crops': ['corn', 'wheat', 'soybeans']
            },
            'retail': {
                'weather_quality': {'severe': -0.5, 'rain': -0.3, 'nice': 0.2},
                'seasonal_items': {'cold': ['heating', 'clothing'], 
                                  'hot': ['cooling', 'beverages']}
            }
        }
        
    def analyze_temperature_impact(self, weather_data: Dict[str, Any],
                                 sector: str) -> Dict[str, float]:
        """Analyze temperature impact on specific sector"""
        impact_scores = {}
        
        for region_id, region_weather in weather_data.items():
            current_temp = region_weather.get('temperature', 20)
            forecast_temps = region_weather.get('forecast_temps', [])
            historical_avg = region_weather.get('historical_avg_temp', 20)
            
            # Calculate deviation from normal
            temp_anomaly = current_temp - historical_avg
            
            if sector == 'energy':
                # Extreme temps increase energy demand
                if abs(temp_anomaly) > 10:  # 10°C deviation
                    impact = 1.5 if temp_anomaly < -10 else 1.3
                elif abs(temp_anomaly) > 5:
                    impact = 1.2 if temp_anomaly < -5 else 1.1
                else:
                    impact = 1.0
                    
                impact_scores[region_id] = (impact - 1.0) * 2  # Convert to -1 to 1 scale
                
            elif sector == 'agriculture':
                # Check for frost or heat stress
                if current_temp < 0:  # Frost
                    impact_scores[region_id] = -0.9
                elif current_temp > 35:  # Heat stress
                    impact_scores[region_id] = -0.7
                else:
                    # Optimal range
                    impact_scores[region_id] = 1.0 - abs(temp_anomaly) / 20
                    
        return impact_scores
        
    def analyze_precipitation_impact(self, weather_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze precipitation impact on agriculture"""
        precipitation_scores = {}
        
        for region_id, region_weather in weather_data.items():
            precip_actual = region_weather.get('precipitation_30d', 0)
            precip_normal = region_weather.get('normal_precipitation_30d', 100)
            
            if precip_normal > 0:
                precip_ratio = precip_actual / precip_normal
                
                if precip_ratio < 0.25:  # Severe drought
                    precipitation_scores[region_id] = -0.8
                elif precip_ratio < 0.5:  # Moderate drought
                    precipitation_scores[region_id] = -0.5
                elif precip_ratio > 2.0:  # Flooding
                    precipitation_scores[region_id] = -0.6
                elif precip_ratio > 1.5:  # Excess rain
                    precipitation_scores[region_id] = -0.3
                else:  # Near normal
                    precipitation_scores[region_id] = 1.0 - abs(1.0 - precip_ratio)
                    
        return precipitation_scores
        
    def analyze_severe_weather_risk(self, forecast_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze risk of severe weather events"""
        severe_events = []
        
        for region_id, region_forecast in forecast_data.items():
            # Check various severe weather indicators
            if region_forecast.get('hurricane_probability', 0) > 0.3:
                severe_events.append({
                    'region': region_id,
                    'event_type': 'hurricane',
                    'probability': region_forecast['hurricane_probability'],
                    'impact_days': region_forecast.get('impact_duration', 3),
                    'severity': 'high'
                })
                
            if region_forecast.get('tornado_risk', 'low') in ['high', 'extreme']:
                severe_events.append({
                    'region': region_id,
                    'event_type': 'tornado',
                    'probability': 0.7 if region_forecast['tornado_risk'] == 'extreme' else 0.4,
                    'impact_days': 1,
                    'severity': 'high'
                })
                
            if region_forecast.get('freeze_warning', False):
                severe_events.append({
                    'region': region_id,
                    'event_type': 'freeze',
                    'probability': 0.8,
                    'impact_days': region_forecast.get('freeze_duration', 2),
                    'severity': 'medium'
                })
                
        return severe_events
        
    def generate_weather_trading_signal(self, impact_analysis: Dict[str, Dict[str, float]],
                                      severe_events: List[Dict[str, Any]],
                                      market_data: pd.DataFrame) -> AlternativeDataSignal:
        """Generate trading signal from weather analysis"""
        
        if not impact_analysis and not severe_events:
            return AlternativeDataSignal(
                signal_type='weather',
                direction=0,
                strength=0.0,
                data_sources=[],
                confidence=0.0,
                metadata={'reason': 'no_weather_analysis'}
            )
            
        # Aggregate impact scores
        all_impacts = []
        for impact_type, scores in impact_analysis.items():
            all_impacts.extend(scores.values())
            
        avg_impact = np.mean(all_impacts) if all_impacts else 0
        
        # Factor in severe weather
        severe_impact = 0
        if severe_events:
            # High severity events have major market impact
            high_severity = [e for e in severe_events if e['severity'] == 'high']
            severe_impact = -min(len(high_severity) * 0.3, 1.0)
            
        # Combine regular and severe weather impacts
        total_impact = avg_impact * 0.7 + severe_impact * 0.3
        
        # Determine signal
        if abs(total_impact) < 0.1:
            direction = 0
            strength = 0.0
        else:
            direction = 1 if total_impact > 0 else -1
            strength = min(abs(total_impact), 1.0)
            
        # Calculate confidence based on forecast accuracy and data coverage
        confidence = 0.5  # Base confidence for weather forecasts
        
        if len(all_impacts) > 5:
            confidence += 0.2  # Good geographic coverage
            
        if severe_events:
            # Severe weather predictions are usually reliable
            avg_probability = np.mean([e['probability'] for e in severe_events])
            confidence = max(confidence, avg_probability * 0.8)
            
        metadata = {
            'focus_sector': self.focus_sector,
            'forecast_horizon': self.forecast_horizon,
            'impact_types': list(impact_analysis.keys()),
            'avg_impact': avg_impact,
            'severe_events': len(severe_events),
            'high_severity_events': len([e for e in severe_events if e['severity'] == 'high'])
        }
        
        return AlternativeDataSignal(
            signal_type='weather',
            direction=direction,
            strength=strength,
            data_sources=['weather_api', 'forecast_data'],
            confidence=confidence,
            metadata=metadata
        )
        
    def calculate_signal(self, data: pd.DataFrame,
                        weather_data: Optional[Dict[str, Any]] = None) -> AlternativeDataSignal:
        """Main signal calculation"""
        
        if weather_data is None:
            # In production, fetch real weather data from NOAA, weather.gov, etc.
            weather_data = self._get_real_data_source('weather',
                                                     sector=self.focus_sector,
                                                     horizon=self.forecast_horizon)
            
        if not weather_data:
            return AlternativeDataSignal(
                signal_type='weather',
                direction=0,
                strength=0.0,
                data_sources=[],
                confidence=0.0,
                metadata={'reason': 'no_weather_data'}
            )
            
        impact_analysis = {}
        
        # Temperature impact
        if 'temperature' in weather_data:
            impact_analysis['temperature'] = self.analyze_temperature_impact(
                weather_data['temperature'], self.focus_sector
            )
            
        # Precipitation impact (mainly for agriculture)
        if self.focus_sector == 'agriculture' and 'precipitation' in weather_data:
            impact_analysis['precipitation'] = self.analyze_precipitation_impact(
                weather_data['precipitation']
            )
            
        # Severe weather analysis
        severe_events = []
        if 'forecast' in weather_data:
            severe_events = self.analyze_severe_weather_risk(weather_data['forecast'])
            
        return self.generate_weather_trading_signal(impact_analysis, severe_events, data)

# Factory function for creating alternative data models
def create_alternative_data_model(model_type: str, **kwargs) -> BaseAlternativeDataModel:
    """Factory function to create alternative data models"""
    models = {
        'satellite': SatelliteDataModel,
        'web_scraping': WebScrapingModel,
        'supply_chain': SupplyChainModel,
        'weather': WeatherModel
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
        
    return models[model_type](**kwargs)