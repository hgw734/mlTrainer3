# Data Source Integration Guide

## Overview

This guide provides the technical steps required to integrate new data sources into the mlTrainer system while maintaining full compliance with data lineage and tagging requirements.

## Integration Checklist

Before integrating any new data source, ensure:

- [ ] Legal approval for data usage and licensing
- [ ] API credentials secured in secrets manager
- [ ] Rate limiting strategy defined
- [ ] Data retention policy established
- [ ] Compliance team sign-off

## Step-by-Step Integration Process

### Step 1: Add to API Configuration

Edit `config/api_config.py` to add the new data source:

```python
# In APISource enum
class APISource(Enum):
    POLYGON = "polygon"
    FRED = "fred"
    REFINITIV = "refinitiv"  # NEW
    BENZINGA = "benzinga"    # NEW

# In APPROVED_ENDPOINTS
APISource.REFINITIV: {
    "tick_data": APIEndpoint(
        name="Refinitiv Tick Data",
        base_url="https://api.refinitiv.com/data/tick/v1",
        requires_auth=True,
        auth_header="Bearer",
        rate_limit_per_minute=300,
        compliance_verified=True,
        timeout_seconds=30,
        retry_attempts=3,
        retry_delay_seconds=1,
    ),
    # Add more endpoints
}
```

### Step 2: Create Data Connector

Create a new connector in the project root:

```python
# refinitiv_connector.py
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
from config.api_config import get_api_auth_config, APISource
from data_lineage_system import DataLineageSystem

logger = logging.getLogger(__name__)

class RefinitivConnector:
    """Connector for Refinitiv data with full compliance"""
    
    def __init__(self):
        self.auth = get_api_auth_config(APISource.REFINITIV)
        self.lineage = DataLineageSystem()
        self.source_tag = "refinitiv"
        
    def get_tick_data(
        self, 
        symbol: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> pd.DataFrame:
        """Get tick data with full lineage tracking"""
        
        # Create lineage entry
        lineage_id = self.lineage.create_lineage_entry(
            data_type="tick_data",
            source=self.source_tag,
            metadata={
                "symbol": symbol,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "api_endpoint": "tick_data"
            }
        )
        
        try:
            # Make API call
            response = self._make_api_call(
                endpoint="tick_data",
                params={
                    "symbol": symbol,
                    "start": start_time,
                    "end": end_time
                }
            )
            
            # Convert to DataFrame
            df = self._parse_tick_response(response)
            
            # Tag every row with source
            df['data_source'] = self.source_tag
            df['lineage_id'] = lineage_id
            df['source_timestamp'] = datetime.now()
            
            # Update lineage with success
            self.lineage.update_lineage_status(
                lineage_id, 
                "success",
                {"rows_retrieved": len(df)}
            )
            
            return df
            
        except Exception as e:
            # Update lineage with failure
            self.lineage.update_lineage_status(
                lineage_id,
                "failed",
                {"error": str(e)}
            )
            raise
```

### Step 3: Update Data Lineage System

Enhance `data_lineage_system.py` to handle new source:

```python
# In DataLineageSystem class
APPROVED_SOURCES = ["polygon", "fred", "refinitiv", "benzinga"]

def validate_source(self, source: str) -> bool:
    """Validate data source is approved"""
    if source not in self.APPROVED_SOURCES:
        raise ValueError(f"Unapproved data source: {source}")
    return True
```

### Step 4: Create Compliance Wrapper

Create a compliance wrapper for the new data source:

```python
# compliance_wrappers/refinitiv_compliance.py
from typing import Any, Dict
import pandas as pd
from core.compliance_mode import ComplianceMode

class RefinitivComplianceWrapper:
    """Ensures all Refinitiv data meets compliance requirements"""
    
    def __init__(self):
        self.compliance = ComplianceMode()
        
    def validate_and_tag_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and permanently tag data"""
        
        # Ensure required tags exist
        required_tags = ['data_source', 'lineage_id', 'source_timestamp']
        for tag in required_tags:
            if tag not in data.columns:
                raise ValueError(f"Missing required tag: {tag}")
        
        # Add compliance verification
        data['compliance_verified'] = True
        data['compliance_timestamp'] = pd.Timestamp.now()
        
        # Log to compliance system
        self.compliance.log_action(
            action="data_validated",
            details={
                "source": "refinitiv",
                "rows": len(data),
                "timestamp": pd.Timestamp.now().isoformat()
            }
        )
        
        return data
```

### Step 5: Update Model to Use New Data

Update the model implementation to use the new data source:

```python
# custom/fractal.py
class FractalModel:
    def get_data(self, symbol: str, start: str, end: str):
        """Get data from appropriate source based on requirements"""
        
        # Check if high-frequency data is available
        if self._has_refinitiv_access():
            from refinitiv_connector import RefinitivConnector
            connector = RefinitivConnector()
            
            # Get tick data
            data = connector.get_tick_data(symbol, start, end)
            
            # Validate compliance
            from compliance_wrappers.refinitiv_compliance import RefinitivComplianceWrapper
            wrapper = RefinitivComplianceWrapper()
            data = wrapper.validate_and_tag_data(data)
            
            return data
        else:
            raise ValueError(
                "FractalModel requires high-frequency data. "
                "Refinitiv access not configured."
            )
```

### Step 6: Remove from Sandbox

Once integration is complete and tested:

1. Update `SANDBOXED_MODELS` in `enforce_data_compliance.py`
2. Remove the model from the sandboxed list
3. Run compliance verification

```python
# In enforce_data_compliance.py
SANDBOXED_MODELS = {
    # Remove integrated models
    # "fractal_model": "...",  # REMOVED - Refinitiv integrated
}
```

### Step 7: Testing Protocol

Create comprehensive tests:

```python
# tests/test_refinitiv_integration.py
import pytest
from refinitiv_connector import RefinitivConnector
from data_lineage_system import DataLineageSystem

def test_data_tagging():
    """Ensure all data is properly tagged"""
    connector = RefinitivConnector()
    data = connector.get_tick_data("AAPL", "2024-01-01", "2024-01-02")
    
    # Verify tags
    assert 'data_source' in data.columns
    assert all(data['data_source'] == 'refinitiv')
    assert 'lineage_id' in data.columns
    assert 'source_timestamp' in data.columns

def test_lineage_tracking():
    """Ensure lineage is tracked"""
    lineage = DataLineageSystem()
    connector = RefinitivConnector()
    
    # Get data
    data = connector.get_tick_data("AAPL", "2024-01-01", "2024-01-02")
    
    # Verify lineage exists
    lineage_id = data['lineage_id'].iloc[0]
    lineage_record = lineage.get_lineage(lineage_id)
    
    assert lineage_record is not None
    assert lineage_record['source'] == 'refinitiv'
```

## Compliance Verification

After integration, run the compliance verification:

```bash
# Run compliance check
python enforce_data_compliance.py

# Verify model is no longer sandboxed
python analyze_alternative_data_models.py

# Run integration tests
pytest tests/test_refinitiv_integration.py -v
```

## Data Source Templates

### High-Frequency Data Template
```python
class HighFrequencyDataConnector:
    """Template for tick/microstructure data sources"""
    
    required_capabilities = [
        "microsecond_timestamps",
        "tick_by_tick_trades",
        "order_book_snapshots",
        "market_depth"
    ]
```

### NLP Data Template
```python
class NLPDataConnector:
    """Template for text/sentiment data sources"""
    
    required_capabilities = [
        "article_full_text",
        "publication_timestamp",
        "sentiment_scores",
        "entity_recognition"
    ]
```

### Alternative Data Template
```python
class AlternativeDataConnector:
    """Template for satellite/weather/shipping data"""
    
    required_capabilities = [
        "geo_coordinates",
        "timestamp",
        "data_type_identifier",
        "measurement_units"
    ]
```

## Security Considerations

1. **API Keys**: Never hardcode, always use secrets manager
2. **Rate Limiting**: Implement exponential backoff
3. **Data Caching**: Cache to reduce API calls but maintain freshness
4. **Audit Trail**: Every API call must be logged
5. **Error Handling**: Graceful degradation, never expose raw errors

## Monitoring and Alerts

Set up monitoring for:
- API usage vs. rate limits
- Data freshness
- Lineage tracking completeness
- Compliance violations
- Cost tracking

## Rollback Procedure

If issues arise:
1. Re-add model to `SANDBOXED_MODELS`
2. Disable API endpoint in `api_config.py`
3. Run `enforce_data_compliance.py`
4. Investigate and fix issues
5. Re-attempt integration

## Summary

Following this guide ensures:
- ✅ Full compliance with data lineage requirements
- ✅ Proper tagging of all data
- ✅ Audit trail for all operations
- ✅ Graceful handling of API limits
- ✅ Security best practices
- ✅ Easy rollback if needed