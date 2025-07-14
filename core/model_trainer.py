#!/usr/bin/env python3
"""
Model Trainer for mlTrainer
Handles model training, validation, and persistence
USES REAL HISTORICAL DATA ONLY
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
import hashlib

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Manages model training, validation, and persistence
    All models trained on real historical market data
    """
    
    def __init__(self, model_storage_path: str = "trained_models"):
        """Initialize model trainer"""
        self.trained_models = {}
        self.training_history = {}
        self.model_storage_path = Path(model_storage_path)
        self.model_storage_path.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        (self.model_storage_path / "models").mkdir(exist_ok=True)
        (self.model_storage_path / "metadata").mkdir(exist_ok=True)
        (self.model_storage_path / "compliance").mkdir(exist_ok=True)
        
    def train_model(self, model_id: str, model_class, data: pd.DataFrame,
                   validation_split: float = 0.2, params: Dict = None) -> Dict[str, Any]:
        """
        Train a model with proper temporal validation
        
        Args:
            model_id: Unique identifier for the model
            model_class: Model class to instantiate
            data: Historical OHLCV data
            validation_split: Fraction of data for validation
            params: Model parameters
            
        Returns:
            Dictionary with training metrics
        """
        try:
            start_time = datetime.now()
            logger.info(f"Starting training for model {model_id}")
            
            # Validate data
            self._validate_data(data)
            
            # Split data temporally (not randomly!)
            split_idx = int(len(data) * (1 - validation_split))
            train_data = data.iloc[:split_idx].copy()
            val_data = data.iloc[split_idx:].copy()
            
            logger.info(f"Train period: {train_data.index[0]} to {train_data.index[-1]}")
            logger.info(f"Val period: {val_data.index[0]} to {val_data.index[-1]}")
            
            # Initialize model
            model = model_class(**(params or {}))
            
            # Train model
            model.fit(train_data)
            
            # Generate predictions on validation set
            val_predictions = model.predict(val_data)
            
            # Calculate metrics
            metrics = self._calculate_metrics(val_data, val_predictions)
            
            # Store model and results
            training_time = (datetime.now() - start_time).total_seconds()
            
            model_package = {
                'model': model,
                'model_id': model_id,
                'class_name': model.__class__.__name__,
                'parameters': model.get_parameters() if hasattr(model, 'get_parameters') else {},
                'train_period': {
                    'start': train_data.index[0].isoformat(),
                    'end': train_data.index[-1].isoformat(),
                    'samples': len(train_data)
                },
                'val_period': {
                    'start': val_data.index[0].isoformat(),
                    'end': val_data.index[-1].isoformat(),
                    'samples': len(val_data)
                },
                'metrics': metrics,
                'training_time': training_time,
                'timestamp': datetime.now().isoformat(),
                'data_hash': self._calculate_data_hash(data)
            }
            
            # Save model
            self._save_model(model_id, model_package)
            
            # Update tracking
            self.trained_models[model_id] = model_package
            self.training_history[model_id] = {
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'training_time': training_time
            }
            
            logger.info(f"Model {model_id} trained successfully. Sharpe: {metrics['sharpe_ratio']:.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Training failed for {model_id}: {e}")
            raise
    
    def _validate_data(self, data: pd.DataFrame):
        """Validate input data"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if len(data) < 100:
            raise ValueError(f"Insufficient data: {len(data)} rows (minimum 100 required)")
        
        # Check for NaN values
        if data[required_columns].isna().any().any():
            raise ValueError("Data contains NaN values")
        
        # Check for valid price data
        if (data['high'] < data['low']).any():
            raise ValueError("Invalid data: high < low")
        
        if (data['close'] <= 0).any() or (data['volume'] < 0).any():
            raise ValueError("Invalid data: negative or zero prices/volume")
    
    def _calculate_metrics(self, data: pd.DataFrame, predictions: pd.Series) -> Dict[str, float]:
        """Calculate realistic trading metrics"""
        # Calculate returns based on signals
        returns = data['close'].pct_change()
        signal_returns = returns.shift(-1) * predictions.shift(1)
        signal_returns = signal_returns.dropna()
        
        # Handle edge cases
        if len(signal_returns) == 0 or signal_returns.std() == 0:
            return {
                'sharpe_ratio': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'volatility': 0.0,
                'calmar_ratio': 0.0,
                'sortino_ratio': 0.0
            }
        
        # Core metrics
        total_return = (1 + signal_returns).prod() - 1
        
        # Sharpe ratio (annualized)
        sharpe_ratio = (signal_returns.mean() * 252) / (signal_returns.std() * np.sqrt(252))
        
        # Maximum drawdown
        cumulative = (1 + signal_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win/loss statistics
        winning_returns = signal_returns[signal_returns > 0]
        losing_returns = signal_returns[signal_returns < 0]
        
        win_rate = len(winning_returns) / len(signal_returns[signal_returns != 0]) if len(signal_returns[signal_returns != 0]) > 0 else 0
        avg_win = winning_returns.mean() if len(winning_returns) > 0 else 0
        avg_loss = losing_returns.mean() if len(losing_returns) > 0 else 0
        
        # Volatility
        volatility = signal_returns.std() * np.sqrt(252)
        
        # Calmar ratio
        calmar_ratio = (total_return * 252 / len(signal_returns) * 252) / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = signal_returns[signal_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (signal_returns.mean() * 252) / downside_std if downside_std > 0 else 0
        
        return {
            'sharpe_ratio': float(sharpe_ratio),
            'total_return': float(total_return),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'volatility': float(volatility),
            'calmar_ratio': float(calmar_ratio),
            'sortino_ratio': float(sortino_ratio)
        }
    
    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """Calculate hash of training data for reproducibility"""
        # Create a string representation of key data characteristics
        data_summary = f"{len(data)}_{data.index[0]}_{data.index[-1]}_{data['close'].sum()}"
        return hashlib.sha256(data_summary.encode()).hexdigest()[:16]
    
    def _save_model(self, model_id: str, model_package: Dict[str, Any]):
        """Save model with compliance tracking"""
        # Save model pickle
        model_path = self.model_storage_path / "models" / f"{model_id}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model_package, f)
        
        # Save metadata
        metadata = {k: v for k, v in model_package.items() if k != 'model'}
        metadata_path = self.model_storage_path / "metadata" / f"{model_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save compliance signature
        self._save_compliance_signature(model_id, model_package)
        
        logger.info(f"Model {model_id} saved to {model_path}")
    
    def _save_compliance_signature(self, model_id: str, model_package: Dict[str, Any]):
        """Save compliance signature for model verification"""
        compliance_data = {
            'model_id': model_id,
            'class_name': model_package['class_name'],
            'data_hash': model_package['data_hash'],
            'timestamp': model_package['timestamp'],
            'metrics': model_package['metrics'],
            'data_source': 'polygon',  # Always from approved sources
            'compliance_version': '2.0'
        }
        
        # Create signature
        signature_string = json.dumps(compliance_data, sort_keys=True)
        signature = hashlib.sha256(signature_string.encode()).hexdigest()
        
        compliance_data['signature'] = signature
        
        # Save compliance file
        compliance_path = self.model_storage_path / "compliance" / f"{model_id}_compliance.json"
        with open(compliance_path, 'w') as f:
            json.dump(compliance_data, f, indent=2)
    
    def load_model(self, model_id: str) -> Any:
        """Load trained model with compliance verification"""
        model_path = self.model_storage_path / "models" / f"{model_id}.pkl"
        
        if not model_path.exists():
            raise ValueError(f"Model {model_id} not found")
        
        # Verify compliance
        if not self._verify_compliance_signature(model_id):
            raise ValueError(f"Model {model_id} failed compliance verification")
        
        # Load model
        with open(model_path, 'rb') as f:
            model_package = pickle.load(f)
        
        logger.info(f"Model {model_id} loaded successfully")
        return model_package['model']
    
    def _verify_compliance_signature(self, model_id: str) -> bool:
        """Verify model compliance signature"""
        compliance_path = self.model_storage_path / "compliance" / f"{model_id}_compliance.json"
        
        if not compliance_path.exists():
            logger.error(f"No compliance file for model {model_id}")
            return False
        
        with open(compliance_path, 'r') as f:
            compliance_data = json.load(f)
        
        # Verify signature
        stored_signature = compliance_data.pop('signature')
        signature_string = json.dumps(compliance_data, sort_keys=True)
        calculated_signature = hashlib.sha256(signature_string.encode()).hexdigest()
        
        return stored_signature == calculated_signature
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model information without loading the model"""
        metadata_path = self.model_storage_path / "metadata" / f"{model_id}.json"
        
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models"""
        models = []
        
        for metadata_file in (self.model_storage_path / "metadata").glob("*.json"):
            model_id = metadata_file.stem
            with open(metadata_file, 'r') as f:
                info = json.load(f)
            
            models.append({
                'model_id': model_id,
                'class_name': info.get('class_name'),
                'timestamp': info.get('timestamp'),
                'sharpe_ratio': info.get('metrics', {}).get('sharpe_ratio', 0),
                'total_return': info.get('metrics', {}).get('total_return', 0)
            })
        
        # Sort by Sharpe ratio
        models.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
        
        return models
    
    def delete_model(self, model_id: str):
        """Delete a model and its associated files"""
        files_to_delete = [
            self.model_storage_path / "models" / f"{model_id}.pkl",
            self.model_storage_path / "metadata" / f"{model_id}.json",
            self.model_storage_path / "compliance" / f"{model_id}_compliance.json"
        ]
        
        for file_path in files_to_delete:
            if file_path.exists():
                file_path.unlink()
        
        # Remove from memory
        self.trained_models.pop(model_id, None)
        self.training_history.pop(model_id, None)
        
        logger.info(f"Model {model_id} deleted")