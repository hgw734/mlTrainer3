"""
mlTrainer - Data Validator
==========================

Purpose: Comprehensive data validation utilities for ensuring data quality,
completeness, and compliance with mlTrainer standards. Validates data from
all sources (Polygon, FRED, QuiverQuant) before processing.

Features:
- Source verification and authenticity checks
- Data freshness and timestamp validation
- Schema validation for different data types
- Compliance rule enforcement
- Data quality scoring
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import re
import json

logger = logging.getLogger(__name__)

class DataValidator:
    """Validates data quality and compliance for mlTrainer system"""
    
    def __init__(self):
        self.validation_rules = self._load_validation_rules()
        self.schema_definitions = self._load_schema_definitions()
        self.validation_history = []
        
        logger.info("DataValidator initialized")
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules configuration"""
        return {
            "freshness": {
                "max_age_hours": 24,
                "critical_age_hours": 1,
                "require_timestamps": True
            },
            "completeness": {
                "min_data_points": 5,
                "required_fields_ratio": 0.8,
                "allow_null_percentage": 0.1
            },
            "quality": {
                "outlier_threshold": 3.0,  # Standard deviations
                "duplicate_threshold": 0.05,  # 5% duplicates max
                "consistency_threshold": 0.9
            },
            "compliance": {
                "require_source_verification": True,
                "verified_sources": ["polygon", "fred", "quiverquant"],
                "require_data_lineage": True
            }
        }
    
    def _load_schema_definitions(self) -> Dict[str, Dict]:
        """Load data schema definitions for different data types"""
        return {
            "market_data": {
                "required_fields": ["timestamp", "open", "high", "low", "close", "volume"],
                "optional_fields": ["adjusted_close", "split_factor", "dividend"],
                "data_types": {
                    "timestamp": "datetime",
                    "open": "float",
                    "high": "float", 
                    "low": "float",
                    "close": "float",
                    "volume": "int"
                },
                "constraints": {
                    "high >= open": True,
                    "high >= low": True,
                    "high >= close": True,
                    "volume >= 0": True
                }
            },
            "macro_indicators": {
                "required_fields": ["series_id", "date", "value"],
                "optional_fields": ["units", "frequency", "notes"],
                "data_types": {
                    "series_id": "string",
                    "date": "date",
                    "value": "float"
                },
                "constraints": {}
            },
            "sentiment_data": {
                "required_fields": ["source", "timestamp", "sentiment_score"],
                "optional_fields": ["confidence", "mentions", "context"],
                "data_types": {
                    "source": "string",
                    "timestamp": "datetime",
                    "sentiment_score": "float"
                },
                "constraints": {
                    "-1 <= sentiment_score <= 1": True
                }
            },
            "recommendation": {
                "required_fields": ["ticker", "score", "confidence", "timestamp", "source"],
                "optional_fields": ["target_price", "stop_loss", "timeframe", "reasoning"],
                "data_types": {
                    "ticker": "string",
                    "score": "float",
                    "confidence": "float",
                    "timestamp": "datetime",
                    "source": "string"
                },
                "constraints": {
                    "0 <= score <= 100": True,
                    "0 <= confidence <= 100": True
                }
            }
        }
    
    def validate_data_source(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data source authenticity and compliance"""
        validation_result = {
            "valid": True,
            "score": 100.0,
            "errors": [],
            "warnings": [],
            "checks_performed": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # Check source verification
        source = data.get("source")
        if not source:
            validation_result["errors"].append("No data source specified")
            validation_result["valid"] = False
        elif source not in self.validation_rules["compliance"]["verified_sources"]:
            validation_result["errors"].append(f"Unverified data source: {source}")
            validation_result["valid"] = False
        
        validation_result["checks_performed"].append("source_verification")
        
        # Check verification flag
        if not data.get("verified", False):
            validation_result["warnings"].append("Data lacks verification flag")
            validation_result["score"] -= 20
        
        # Check data lineage
        if self.validation_rules["compliance"]["require_data_lineage"]:
            if "lineage" not in data and "metadata" not in data:
                validation_result["warnings"].append("Missing data lineage information")
                validation_result["score"] -= 10
        
        validation_result["checks_performed"].append("compliance_check")
        
        return validation_result
    
    def validate_data_freshness(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data timestamp and freshness"""
        validation_result = {
            "valid": True,
            "score": 100.0,
            "errors": [],
            "warnings": [],
            "age_hours": None,
            "timestamp": datetime.now().isoformat()
        }
        
        timestamp_str = data.get("timestamp")
        if not timestamp_str:
            validation_result["errors"].append("Missing timestamp")
            validation_result["valid"] = False
            return validation_result
        
        try:
            # Parse timestamp
            if isinstance(timestamp_str, str):
                data_timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                data_timestamp = timestamp_str
            
            # Calculate age
            age = datetime.now() - data_timestamp.replace(tzinfo=None)
            age_hours = age.total_seconds() / 3600
            validation_result["age_hours"] = age_hours
            
            # Check freshness thresholds
            max_age = self.validation_rules["freshness"]["max_age_hours"]
            critical_age = self.validation_rules["freshness"]["critical_age_hours"]
            
            if age_hours > max_age:
                validation_result["errors"].append(f"Data too old: {age_hours:.1f}h > {max_age}h limit")
                validation_result["valid"] = False
            elif age_hours > critical_age:
                validation_result["warnings"].append(f"Data aging: {age_hours:.1f}h old")
                validation_result["score"] -= min(30, age_hours - critical_age)
            
        except Exception as e:
            validation_result["errors"].append(f"Invalid timestamp format: {e}")
            validation_result["valid"] = False
        
        return validation_result
    
    def validate_data_schema(self, data: Dict[str, Any], data_type: str) -> Dict[str, Any]:
        """Validate data against predefined schema"""
        validation_result = {
            "valid": True,
            "score": 100.0,
            "errors": [],
            "warnings": [],
            "missing_fields": [],
            "type_errors": []
        }
        
        if data_type not in self.schema_definitions:
            validation_result["errors"].append(f"Unknown data type: {data_type}")
            validation_result["valid"] = False
            return validation_result
        
        schema = self.schema_definitions[data_type]
        
        # Check required fields
        required_fields = schema.get("required_fields", [])
        for field in required_fields:
            if field not in data:
                validation_result["missing_fields"].append(field)
                validation_result["errors"].append(f"Missing required field: {field}")
                validation_result["valid"] = False
        
        # Check data types
        data_types = schema.get("data_types", {})
        for field, expected_type in data_types.items():
            if field in data:
                if not self._validate_field_type(data[field], expected_type):
                    validation_result["type_errors"].append(f"{field}: expected {expected_type}")
                    validation_result["warnings"].append(f"Type mismatch for {field}")
                    validation_result["score"] -= 10
        
        # Check constraints
        constraints = schema.get("constraints", {})
        for constraint, required in constraints.items():
            if required and not self._validate_constraint(data, constraint):
                validation_result["warnings"].append(f"Constraint violation: {constraint}")
                validation_result["score"] -= 15
        
        return validation_result
    
    def _validate_field_type(self, value: Any, expected_type: str) -> bool:
        """Validate individual field type"""
        if value is None:
            return True  # Allow None values
        
        try:
            if expected_type == "string":
                return isinstance(value, str)
            elif expected_type == "float":
                float(value)
                return True
            elif expected_type == "int":
                int(value)
                return True
            elif expected_type == "datetime":
                if isinstance(value, str):
                    datetime.fromisoformat(value.replace('Z', '+00:00'))
                return True
            elif expected_type == "date":
                if isinstance(value, str):
                    datetime.strptime(value, '%Y-%m-%d')
                return True
            else:
                return True  # Unknown type, assume valid
        except:
            return False
    
    def _validate_constraint(self, data: Dict[str, Any], constraint: str) -> bool:
        """Validate data constraint"""
        try:
            # Simple constraint evaluation
            # In production, this would be more sophisticated
            if "high >= low" in constraint:
                return data.get("high", 0) >= data.get("low", 0)
            elif "volume >= 0" in constraint:
                return data.get("volume", 0) >= 0
            elif "0 <= score <= 100" in constraint:
                score = data.get("score", 0)
                return 0 <= score <= 100
            elif "0 <= confidence <= 100" in constraint:
                confidence = data.get("confidence", 0)
                return 0 <= confidence <= 100
            elif "-1 <= sentiment_score <= 1" in constraint:
                sentiment = data.get("sentiment_score", 0)
                return -1 <= sentiment <= 1
            else:
                return True  # Unknown constraint, assume valid
        except:
            return False
    
    def validate_data_quality(self, data: Any, data_type: str = "generic") -> Dict[str, Any]:
        """Validate data quality metrics"""
        validation_result = {
            "valid": True,
            "score": 100.0,
            "errors": [],
            "warnings": [],
            "quality_metrics": {}
        }
        
        try:
            # Convert to DataFrame if possible for analysis
            if isinstance(data, dict) and "data" in data:
                df = pd.DataFrame(data["data"])
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                df = data
            else:
                validation_result["warnings"].append("Cannot analyze data quality for this format")
                return validation_result
            
            if df.empty:
                validation_result["errors"].append("Empty dataset")
                validation_result["valid"] = False
                return validation_result
            
            # Check completeness
            total_cells = df.size
            null_cells = df.isnull().sum().sum()
            completeness = 1 - (null_cells / total_cells) if total_cells > 0 else 0
            validation_result["quality_metrics"]["completeness"] = completeness
            
            null_threshold = self.validation_rules["completeness"]["allow_null_percentage"]
            if (null_cells / total_cells) > null_threshold:
                validation_result["warnings"].append(f"High null percentage: {null_cells/total_cells:.2%}")
                validation_result["score"] -= 20
            
            # Check for duplicates
            if len(df) > 1:
                duplicate_rows = df.duplicated().sum()
                duplicate_ratio = duplicate_rows / len(df)
                validation_result["quality_metrics"]["duplicate_ratio"] = duplicate_ratio
                
                dup_threshold = self.validation_rules["quality"]["duplicate_threshold"]
                if duplicate_ratio > dup_threshold:
                    validation_result["warnings"].append(f"High duplicate ratio: {duplicate_ratio:.2%}")
                    validation_result["score"] -= 15
            
            # Check for outliers in numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            outlier_count = 0
            
            for col in numeric_cols:
                if len(df[col].dropna()) > 3:  # Need at least 3 points for outlier detection
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    outliers = z_scores > self.validation_rules["quality"]["outlier_threshold"]
                    outlier_count += outliers.sum()
            
            if outlier_count > 0:
                outlier_ratio = outlier_count / len(df)
                validation_result["quality_metrics"]["outlier_ratio"] = outlier_ratio
                if outlier_ratio > 0.1:  # More than 10% outliers
                    validation_result["warnings"].append(f"High outlier count: {outlier_count}")
                    validation_result["score"] -= 10
            
            # Check data volume
            min_points = self.validation_rules["completeness"]["min_data_points"]
            if len(df) < min_points:
                validation_result["warnings"].append(f"Low data volume: {len(df)} < {min_points}")
                validation_result["score"] -= 25
            
            validation_result["quality_metrics"]["row_count"] = len(df)
            validation_result["quality_metrics"]["column_count"] = len(df.columns)
            
        except Exception as e:
            validation_result["errors"].append(f"Quality analysis failed: {e}")
            validation_result["valid"] = False
        
        return validation_result
    
    def validate_recommendation(self, recommendation: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive validation for stock recommendations"""
        validation_result = {
            "valid": True,
            "score": 100.0,
            "errors": [],
            "warnings": [],
            "validations": []
        }
        
        # Schema validation
        schema_result = self.validate_data_schema(recommendation, "recommendation")
        validation_result["validations"].append(("schema", schema_result))
        
        if not schema_result["valid"]:
            validation_result["valid"] = False
            validation_result["errors"].extend(schema_result["errors"])
        
        validation_result["score"] = min(validation_result["score"], schema_result["score"])
        
        # Source validation
        source_result = self.validate_data_source(recommendation)
        validation_result["validations"].append(("source", source_result))
        
        if not source_result["valid"]:
            validation_result["valid"] = False
            validation_result["errors"].extend(source_result["errors"])
        
        validation_result["score"] = min(validation_result["score"], source_result["score"])
        
        # Freshness validation
        freshness_result = self.validate_data_freshness(recommendation)
        validation_result["validations"].append(("freshness", freshness_result))
        
        if not freshness_result["valid"]:
            validation_result["valid"] = False
            validation_result["errors"].extend(freshness_result["errors"])
        
        validation_result["score"] = min(validation_result["score"], freshness_result["score"])
        
        # Business logic validation
        ticker = recommendation.get("ticker", "")
        if not re.match(r'^[A-Z]{1,5}$', ticker):
            validation_result["warnings"].append("Invalid ticker format")
            validation_result["score"] -= 10
        
        score = recommendation.get("score", 0)
        confidence = recommendation.get("confidence", 0)
        
        # Check score-confidence consistency
        if abs(score - confidence) > 30:
            validation_result["warnings"].append("Large score-confidence divergence")
            validation_result["score"] -= 5
        
        # Add to validation history
        self.validation_history.append({
            "timestamp": datetime.now().isoformat(),
            "data_type": "recommendation",
            "ticker": ticker,
            "valid": validation_result["valid"],
            "score": validation_result["score"]
        })
        
        return validation_result
    
    def validate_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive validation for market data"""
        validation_result = {
            "valid": True,
            "score": 100.0,
            "errors": [],
            "warnings": [],
            "validations": []
        }
        
        # Source validation
        source_result = self.validate_data_source(market_data)
        validation_result["validations"].append(("source", source_result))
        
        # Data quality validation
        quality_result = self.validate_data_quality(market_data, "market_data")
        validation_result["validations"].append(("quality", quality_result))
        
        # Schema validation for each data point
        data_points = market_data.get("data", [])
        if data_points:
            sample_point = data_points[0] if data_points else {}
            # Map Polygon format to our schema
            schema_data = {
                "timestamp": sample_point.get("t", ""),
                "open": sample_point.get("o", 0),
                "high": sample_point.get("h", 0),
                "low": sample_point.get("l", 0),
                "close": sample_point.get("c", 0),
                "volume": sample_point.get("v", 0)
            }
            
            schema_result = self.validate_data_schema(schema_data, "market_data")
            validation_result["validations"].append(("schema", schema_result))
        
        # Aggregate results
        for validation_type, result in validation_result["validations"]:
            if not result["valid"]:
                validation_result["valid"] = False
                validation_result["errors"].extend(result["errors"])
            
            validation_result["warnings"].extend(result["warnings"])
            validation_result["score"] = min(validation_result["score"], result["score"])
        
        return validation_result
    
    def get_validation_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get validation statistics summary"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_validations = []
        for validation in self.validation_history:
            try:
                validation_time = datetime.fromisoformat(validation["timestamp"])
                if validation_time > cutoff_time:
                    recent_validations.append(validation)
            except:
                continue
        
        if not recent_validations:
            return {
                "period_hours": hours,
                "total_validations": 0,
                "summary": "No validations in period"
            }
        
        total_validations = len(recent_validations)
        valid_count = sum(1 for v in recent_validations if v["valid"])
        avg_score = np.mean([v["score"] for v in recent_validations])
        
        data_types = {}
        for validation in recent_validations:
            data_type = validation.get("data_type", "unknown")
            data_types[data_type] = data_types.get(data_type, 0) + 1
        
        return {
            "period_hours": hours,
            "total_validations": total_validations,
            "valid_percentage": (valid_count / total_validations) * 100,
            "average_score": round(avg_score, 1),
            "by_data_type": data_types,
            "validation_rate": round(total_validations / hours, 1),
            "timestamp": datetime.now().isoformat()
        }
    
    def clear_validation_history(self, older_than_hours: int = 168) -> int:
        """Clear old validation history entries"""
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        
        initial_count = len(self.validation_history)
        
        self.validation_history = [
            v for v in self.validation_history
            if datetime.fromisoformat(v["timestamp"]) > cutoff_time
        ]
        
        removed_count = initial_count - len(self.validation_history)
        logger.info(f"Cleared {removed_count} old validation entries")
        
        return removed_count
